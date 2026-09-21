"""Source-bound native CPU diagnostic. Preparation is inert; run requires an exact plan.

No production edits, recorder installation, global ETW cancellation or profiler
interval changes. Every mutating WPR command ends with our unique instance name.
The finally path stops only that instance; an explicit same-instance cancellation
is a last resort if saving fails. OS termination/power loss cannot run finally.
"""
import argparse
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / 'artifacts/builds/ocr-bench-fullpages-v2-windows'
WPR = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/wpr.exe'
PERFVIEW = ROOT / 'artifacts/tools/perfview-3.2.6/PerfView.exe'
WORKLOAD = ROOT / 'reference/benchmarks/realistic-fp32-v1-workloads.json'
BASELINE = ROOT / 'artifacts/benchmarks/fullpages-b1-window-v1/fullpages-b1-sequential-a.json'
PIN = {
    'build': 'e8009a665752b3b592c98022218daf5eb52ceba2fa9336c3b02c8f1cffe7ac56',
    'binary': 'e9e625b51b8e6172f35907fdd22181f89ac5d95b6caff4ba388b8849a914a0db',
    'source_archive': 'a085056d700e3db091716f7bb48665a806ee9afee019aa4dc4dc2482fa70da56',
    'pdb': '19d3058d9d409c526839ca6c3387e7fafbd9a4addaf1e870d6b19ed5f58e9879',
    'symbols_receipt': '7e8398aabc236ba83cef03487b85553cffab0aeb63c2d2452da50f131699bc60',
    'perfview': '84b8523f7fb4783fd0baae6b080adb1b5aac388192145ad713e504c69954556d',
    'wpr': '64f6d9f83a8b9f8c720870109a6374f650e3786f0ea3f6e3a6c02463257bbfc2',
    'workload': 'b4080bc6a1b38a26cc5363e2f1e92712ed89ad96a6a04b618c5123fa81735e5c',
    'baseline': '00c0fbeb6910bd20b3cd76276fe93048eee699808d2731cf13e08623477e299e',
}
SOURCE_NAMES = ['capture_windows.py', 'FalconOcrCpu.wprp', 'README.md']
LIMITS = {'model_seconds': 600, 'tool_seconds': 120, 'stop_seconds': 180,
          'collector_limit_bytes': 2048 * 2**20, 'minimum_free_bytes_before_start': 8 * 2**30,
          'minimum_free_bytes_during_recording': 4 * 2**30, 'folder_watchdog_bytes': 4 * 2**30,
          'poll_seconds': 1}


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(file, value):
    with Path(file).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def checked(file, expected, bindings):
    file = Path(file).resolve()
    data = file.read_bytes()
    require(sha(data) == expected, 'Digest mismatch: ' + str(file))
    bindings[str(file)] = expected
    return data


def recheck(bindings):
    for file, expected in bindings.items():
        require(file_sha(file) == expected, 'Changed captured input: ' + file)


def token_privileges():
    """Read the current token only; do not enable privileges or request UAC."""
    require(os.name == 'nt', 'Native Windows only')
    from ctypes import wintypes
    class Luid(ctypes.Structure):
        _fields_ = [('LowPart', wintypes.DWORD), ('HighPart', wintypes.LONG)]
    class Entry(ctypes.Structure):
        _fields_ = [('Luid', Luid), ('Attributes', wintypes.DWORD)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi.LookupPrivilegeNameW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(Luid), wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    handle = wintypes.HANDLE()
    require(advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x8, ctypes.byref(handle)), 'OpenProcessToken failed')
    try:
        needed = wintypes.DWORD()
        advapi.GetTokenInformation(handle, 3, None, 0, ctypes.byref(needed))
        require(needed.value >= 4, 'Token privilege size unavailable')
        data = ctypes.create_string_buffer(needed.value)
        require(advapi.GetTokenInformation(handle, 3, data, needed.value, ctypes.byref(needed)), 'GetTokenInformation failed')
        count = wintypes.DWORD.from_buffer(data).value
        require(4 + count * ctypes.sizeof(Entry) <= len(data), 'Invalid privilege buffer')
        result = {}
        for n in range(count):
            entry = Entry.from_buffer(data, 4 + n * ctypes.sizeof(Entry))
            size = wintypes.DWORD(256)
            name = ctypes.create_unicode_buffer(256)
            require(advapi.LookupPrivilegeNameW(None, ctypes.byref(entry.Luid), name, ctypes.byref(size)), 'LookupPrivilegeName failed')
            result[name.value] = {'enabled': bool(entry.Attributes & 2), 'attributes': entry.Attributes}
        return {'administrator': bool(ctypes.windll.shell32.IsUserAnAdmin()), 'privileges': result}
    finally:
        kernel.CloseHandle(handle)


def command_for(folder, image):
    return [str(BUILD / 'ocr_bench.exe'), '--model', str(ROOT / 'artifacts/model'),
            '--threads', '16', '--backend', 'avx2', '--execution', 'sequential',
            '--cache-layout', 'expanded', '--weight-layout', 'unpacked', '--batches', '1',
            '--warmup', '0', '--repetitions', '3', '--min-dimension', '64', '--max-dimension', '1536',
            '--max-new-tokens', '4096', '--cpu-label', 'AMD Ryzen 9 7950X',
            '--environment-label', 'Native Windows; WPR profiled diagnostic',
            '--output', str(folder / 'benchmark.json'), str(image)]


def prepare(folder):
    require(os.name == 'nt', 'Native Windows only')
    folder = folder.resolve()
    require(not folder.exists(), 'Fresh output directory required')
    bindings = {}
    build = json.loads(checked(BUILD / 'build.json', PIN['build'], bindings))
    require(build['status'] == 'complete' and build['build_exit_code'] == 0 and build['source_unchanged_during_build'], 'Invalid frozen build')
    checked(BUILD / 'ocr_bench.exe', PIN['binary'], bindings)
    checked(BUILD / 'ocr_bench.pdb', PIN['pdb'], bindings)
    archive = checked(BUILD / 'source.zip', PIN['source_archive'], bindings)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        require(len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(build['source_sha256']), 'Build source archive inventory')
        for name, digest in build['source_sha256'].items():
            require(sha(z.read(name)) == digest, 'Build archive member: ' + name)
    symbols = json.loads(checked(ROOT / 'reference/benchmarks/windows-fullpages-symbols-v1.json', PIN['symbols_receipt'], bindings))
    require(symbols['binary_sha256'] == PIN['binary'] and symbols['symbols']['sha256'] == PIN['pdb']
            and symbols['symbols']['matches_executable_codeview_identity'] is True
            and symbols['symbols']['guid'].lower() == '{854fda6b-cea5-4cad-bdd6-d8d27009373b}'
            and symbols['symbols']['age'] == 5, 'Frozen executable/PDB pair')
    checked(PERFVIEW, PIN['perfview'], bindings)
    checked(WPR, PIN['wpr'], bindings)
    workload = json.loads(checked(WORKLOAD, PIN['workload'], bindings))
    checked(BASELINE, PIN['baseline'], bindings)
    item = workload['inputs']['prose']
    image = ROOT / item['canonical_path']
    checked(image, item['canonical_png_sha256'], bindings)
    for name, asset in workload['model']['assets'].items():
        file = ROOT / workload['model']['directory'] / name
        require(file.stat().st_size == asset['bytes'] and file_sha(file) == asset['sha256'], 'Pinned model asset: ' + name)
        bindings[str(file.resolve())] = asset['sha256']
    sources = {name: (HERE / name).read_bytes() for name in SOURCE_NAMES}
    folder.mkdir(parents=True)
    source_folder = folder / 'protocol-source'
    source_folder.mkdir()
    for name, data in sources.items():
        (source_folder / name).write_bytes(data)
        bindings[str((HERE / name).resolve())] = sha(data)
        bindings[str((source_folder / name).resolve())] = sha(data)
    with zipfile.ZipFile(folder / 'protocol-source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sources.items():
            z.writestr(name, data)
    bindings[str(folder / 'protocol-source.zip')] = file_sha(folder / 'protocol-source.zip')
    instance = 'FalconOcrCpu-' + uuid.uuid4().hex
    require(re.fullmatch(r'FalconOcrCpu-[a-f0-9]{32}', instance), 'Invalid owned instance')
    profile = str(source_folder / 'FalconOcrCpu.wprp') + '!FalconOcrCpu.Verbose'
    # WPR requires this option LAST. No unscoped stop/cancel exists anywhere.
    scoped = lambda args: [str(WPR), *args, '-instancename', instance]
    plan = {'schema_version': 1, 'kind': 'falcon-ocr-owned-native-cpu-profile-v1', 'created_utc': utc(),
            'output': str(folder), 'instance': instance, 'pins': PIN, 'limits': LIMITS,
            'input': {key: item[key] for key in ['id', 'canonical_path', 'canonical_png_sha256', 'rgb_sha256', 'width', 'height', 'prepared_dimensions_expected', 'input_tokens_expected']},
            'benchmark_command': command_for(folder, image),
            'wpr_commands': {'status': scoped(['-status', 'collectors', '-details']),
                'start': scoped(['-start', profile, '-filemode', '-recordtempto', str(folder / 'wpr-temp')]),
                'stop': scoped(['-stop', str(folder / 'cpu.etl'), 'Falcon OCR full-page CPU diagnostic', '-skipPdbGen']),
                'cancel_owned_if_stop_fails': scoped(['-cancel'])},
            'profile_inspection_command': [str(WPR), '-profiledetails', profile, '-filemode'],
            'perfview_export_template': [str(PERFVIEW), '/AcceptEULA', '/NoGui', '/LogFile:' + str(folder / 'perfview-csv.log'),
                'UserCommand', 'SaveCPUStacksAsCsv', str(folder / 'cpu.etl'), 'ocr_bench', '10', 'LastProcess'],
            'perfview_stacks_template': [str(PERFVIEW), '/AcceptEULA', '/NoGui', '/LogFile:' + str(folder / 'perfview-stacks.log'),
                'UserCommand', 'SaveCPUStacks', str(folder / 'cpu.etl'), 'ocr_bench'],
            'symbol_environment': {'_NT_SYMBOL_PATH': str(BUILD)},
            'current_token_read_only': token_privileges(), 'files_sha256': bindings,
            'recording_started': False, 'analysis_required': ['Exact ETW PID/image/start/end and unique process-name match',
                'Trace captures full process, all three complete recognitions and zero event/sample/stack losses',
                'Actual project symbols resolved from frozen matching PDB; quantify unresolved project frames',
                'Phase attribution from actual call paths only; no exact phase timestamps are present in frozen harness'],
            'qualification': 'Profiled latency is diagnostic only; no quiet benchmark, numerical or quality promotion.'}
    recheck(bindings)
    write_new(folder / 'plan.json', plan)
    print(json.dumps({'status': 'prepared_no_recording', 'plan': str(folder / 'plan.json'), 'sha256': file_sha(folder / 'plan.json'),
                      'capture_privilege_present': 'SeSystemProfilePrivilege' in plan['current_token_read_only']['privileges']}))


def folder_bytes(folder):
    files = [p for p in folder.rglob('*') if p.is_file()]
    return sum(p.stat().st_size for p in files)


def no_recording(text):
    # English local help/status was inspected. Other locales fail closed.
    return 'WPR is not recording' in text or 'There are no trace profiles running' in text


def loss_counts(text):
    values = [(m.group(1), int(m.group(2))) for m in re.finditer(
        r'(?im)^\s*(Dropped events?|Events Lost|Buffers Lost)\s*:\s*(\d+)\s*$', text)]
    require(values and all(value == 0 for _, value in values), 'Missing/nonzero recorder loss counters')
    return [{'field': name, 'count': value} for name, value in values]


def run_tool(command, stem, folder, receipt, timeout):
    require(os.name == 'nt', 'Native Windows only')
    command = list(map(str, command))
    write_new(folder / (stem + '.command.json'), {'command': command, 'started_utc': utc()})
    with (folder / (stem + '.log')).open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # Only our exact child handle, never a process name or global session.
            if process.poll() is None:
                process.kill()
            process.wait()
            receipt['tool_results'].append({'name': stem, 'pid': process.pid, 'exit_code': process.returncode, 'interrupted': True})
            raise
    receipt['tool_results'].append({'name': stem, 'pid': process.pid, 'exit_code': code, 'interrupted': False})
    return code, (folder / (stem + '.log')).read_text(encoding='utf-8', errors='replace')


def benchmark_check(file):
    actual_bytes = file.read_bytes()
    actual = json.loads(actual_bytes)
    baseline_bytes = BASELINE.read_bytes()
    require(sha(baseline_bytes) == PIN['baseline'], 'Frozen baseline changed before validation')
    old = json.loads(baseline_bytes)
    for key in ['binary_sha256', 'cargo_lock_sha256', 'source_sha256', 'weights_sha256', 'model_revision', 'images',
                'options', 'precision', 'backend', 'cache_layout', 'weight_layout', 'threads']:
        require(actual[key] == old[key], 'Benchmark identity/contract differs: ' + key)
    require(actual['warmup'] == 0 and actual['repetitions'] == 3 and len(actual['cases']) == 1, 'Diagnostic repetition contract')
    case, reference = actual['cases'][0], old['cases'][0]
    require(case['batch_size'] == case['active_batch_size'] == 1 and case['execution'] == 'independent_sequential', 'Single-page execution')
    require(case['image_indices'] == [0] and case['token_ids'] == reference['token_ids'], 'Independent emitted IDs differ')
    require(len(case['samples']) == 3, 'Missing complete recognitions')
    expected = reference['samples'][0]['per_request'][0]
    signature_keys = ['text', 'finish_reason', 'width', 'height', 'input_tokens', 'output_tokens', 'teacher_forced', 'precision']
    for sample in case['samples']:
        require(len(sample['per_request']) == 1 and sample['emitted_tokens'] == 1140, 'Sample inventory')
        result = sample['per_request'][0]
        require({k: result[k] for k in signature_keys} == {k: expected[k] for k in signature_keys}, 'Literal result differs from frozen baseline')
        require(math.isfinite(sample['wall_ms']) and sample['wall_ms'] > 0, 'Invalid diagnostic duration')
        require(all(math.isfinite(v) and v >= 0 for v in result['timings'].values()), 'Invalid stage timing')
    return {'validated_benchmark_sha256': sha(actual_bytes), 'validated_baseline_sha256': sha(baseline_bytes),
            'completed_full_page_outputs': 3, 'literal_output_matches_frozen_baseline': True,
            'output_tokens_each': 1140, 'finish_reason': 'eos', 'teacher_forced': False,
            'prepared_dimensions': [1088, 1536], 'prefix_tokens': 6544}


def run(plan_file, expected):
    require(os.name == 'nt', 'Native Windows only')
    plan_bytes = plan_file.read_bytes()
    require(sha(plan_bytes) == expected, 'Plan digest mismatch')
    plan = json.loads(plan_bytes)
    folder = Path(plan['output']).resolve()
    require(plan_file.resolve() == folder / 'plan.json', 'Plan/output location mismatch')
    require(plan['pins'] == PIN and plan['limits'] == LIMITS, 'Pinned runtime/tool policy changed')
    instance = plan['instance']
    require(re.fullmatch(r'FalconOcrCpu-[a-f0-9]{32}', instance), 'Invalid instance')
    require(not (folder / 'capture-start.json').exists() and not (folder / 'capture.json').exists(), 'No capture retries/overwrites')
    require(set(p.name for p in folder.iterdir()) == {'plan.json', 'protocol-source', 'protocol-source.zip'}, 'Unexpected preexisting capture artifacts')
    bindings = dict(plan['files_sha256'])
    bindings[str(plan_file.resolve())] = expected
    recheck(bindings)
    require(plan['benchmark_command'] == command_for(folder, ROOT / plan['input']['canonical_path']), 'Benchmark command changed')
    for command in plan['wpr_commands'].values():
        require(command[0] == str(WPR) and command[-2:] == ['-instancename', instance], 'Unscoped WPR command forbidden')
    token = token_privileges()
    require(token['administrator'] and 'SeSystemProfilePrivilege' in token['privileges'],
            'WPR needs a permitted elevated token with SeSystemProfilePrivilege; no recording attempted')
    require(shutil.disk_usage(folder).free >= LIMITS['minimum_free_bytes_before_start'], 'Need 8 GiB free on output volume')
    (folder / 'wpr-temp').mkdir()
    receipt = {'schema_version': 1, 'kind': plan['kind'], 'plan_sha256': expected, 'instance': instance,
               'started_utc': utc(), 'status': 'failed', 'analysis_accepted': False, 'tool_results': [],
               'capture_token': token, 'errors': [], 'recording_start_attempted': False,
               'recording_start_succeeded': False, 'cleanup_confirmed_not_recording': False,
               'profiled_latency_is_diagnostic_only': True, 'process': None}
    write_new(folder / 'capture-start.json', receipt)
    owned = False
    process = None
    try:
        code, text = run_tool(plan['profile_inspection_command'], 'profile-details', folder, receipt, 30)
        require(code == 0 and 'SampledProfile' in text, 'Custom profile unavailable')
        code, text = run_tool(plan['wpr_commands']['status'], 'status-before', folder, receipt, 30)
        require(code == 0 and no_recording(text), 'Unique instance unexpectedly already active')
        # A random, observed-unused name is now owned by this attempt, including
        # uncertain start failures. Cleanup never touches any other instance.
        owned = True
        receipt['recording_start_attempted'] = True
        code, _ = run_tool(plan['wpr_commands']['start'], 'start', folder, receipt, LIMITS['tool_seconds'])
        require(code == 0, 'WPR start failed')
        receipt['recording_start_succeeded'] = True
        code, text = run_tool(plan['wpr_commands']['status'], 'status-started', folder, receipt, 30)
        require(code == 0 and ('WPR recording is in progress' in text or 'Collector Name' in text)
                and not no_recording(text), 'Recorder not confirmed active')
        receipt['initial_loss_counters'] = loss_counts(text)
        # Do all bulk hashes before start. Per-process exact argv and UTC bounds
        # are retained; ETW analysis must independently join PID and lifetime.
        with (folder / 'benchmark.log').open('xb') as log:
            t0 = time.perf_counter()
            before = utc()
            process = subprocess.Popen(plan['benchmark_command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            receipt['process'] = {'pid': process.pid, 'launch_before_utc': before, 'launch_after_utc': utc(),
                                  'command': plan['benchmark_command'], 'binary_sha256': PIN['binary'],
                                  'image_path': str(BUILD / 'ocr_bench.exe')}
            write_new(folder / 'benchmark-process-start.json', receipt['process'])
            max_bytes = 0
            while process.poll() is None:
                require(time.perf_counter() - t0 < LIMITS['model_seconds'], '600-second model watchdog reached')
                require(shutil.disk_usage(folder).free >= LIMITS['minimum_free_bytes_during_recording'], 'Free disk watchdog reached')
                size = folder_bytes(folder)
                max_bytes = max(max_bytes, size)
                require(size < LIMITS['folder_watchdog_bytes'], '4-GiB output watchdog reached')
                temporary = [p.stat().st_size for p in (folder / 'wpr-temp').rglob('*') if p.is_file()]
                require(not temporary or max(temporary) < LIMITS['collector_limit_bytes'], 'Collector file cap reached; coverage incomplete')
                time.sleep(LIMITS['poll_seconds'])
            receipt['process'].update({'wait_finished_utc': utc(), 'exit_code': process.returncode,
                                       'elapsed_seconds_diagnostic_only': time.perf_counter() - t0})
            receipt['maximum_observed_output_bytes_during_run'] = max_bytes
            require(process.returncode == 0, 'Benchmark process failed')
        code, text = run_tool(plan['wpr_commands']['status'], 'status-before-stop', folder, receipt, 30)
        require(code == 0 and not no_recording(text), 'Recorder ended before process completion')
        receipt['final_live_loss_counters'] = loss_counts(text)
    except BaseException as exc:
        receipt['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        if process is not None and process.poll() is None:
            try:
                process.kill()  # exact Popen-owned process only
                process.wait(timeout=30)
                receipt['process'].update({'terminated_by_watchdog_or_error': True, 'exit_code': process.returncode,
                                           'wait_finished_utc': utc()})
            except BaseException as exc:
                # Child cleanup failure must never skip the owned WPR cleanup.
                receipt['errors'].append('model child cleanup: ' + repr(exc))
        if owned:
            try:
                code, _ = run_tool(plan['wpr_commands']['stop'], 'stop-owned', folder, receipt, LIMITS['stop_seconds'])
                require(code == 0, 'Owned recording save failed')
            except BaseException as exc:
                receipt['errors'].append('stop: ' + repr(exc))
            try:
                code, text = run_tool(plan['wpr_commands']['status'], 'status-after-stop', folder, receipt, 30)
                if code != 0 or not no_recording(text):
                    # Last resort affects ONLY the unique instance owned above.
                    run_tool(plan['wpr_commands']['cancel_owned_if_stop_fails'], 'cancel-owned', folder, receipt, 30)
                    receipt['errors'].append('Owned save did not clear recording; scoped cancellation attempted')
                    code, text = run_tool(plan['wpr_commands']['status'], 'status-after-cancel', folder, receipt, 30)
                receipt['cleanup_confirmed_not_recording'] = code == 0 and no_recording(text)
                require(receipt['cleanup_confirmed_not_recording'], 'Owned recording cleanup unconfirmed')
            except BaseException as exc:
                receipt['errors'].append('cleanup: ' + repr(exc))
    # Heavy validation and file hashing occur after the recorder is stopped.
    try:
        recheck(bindings)
        require(receipt['cleanup_confirmed_not_recording'], 'Recording cleanup required')
        require((folder / 'cpu.etl').is_file() and (folder / 'cpu.etl').stat().st_size > 0, 'No merged ETL')
        receipt['benchmark_validation'] = benchmark_check(folder / 'benchmark.json')
        require(not receipt['errors'], 'Capture errors preserved')
        receipt['status'] = 'captured_pending_process_stack_loss_and_coverage_analysis'
    except BaseException as exc:
        receipt['errors'].append('validation: ' + repr(exc))
    receipt['finished_utc'] = utc()
    receipt['bound_inputs_sha256'] = bindings
    receipt['output_sha256'] = {}
    try:
        for file in sorted(p for p in folder.rglob('*') if p.is_file()):
            receipt['output_sha256'][str(file.relative_to(folder))] = file_sha(file)
    except BaseException as exc:
        receipt['errors'].append('output hashing: ' + repr(exc))
        receipt['status'] = 'failed'
    try:
        # Bind the exact validated bytes, not a later unvalidated replacement.
        # Recheck the input/source window after collecting the output inventory.
        if 'benchmark_validation' in receipt:
            validated = receipt['benchmark_validation']['validated_benchmark_sha256']
            require(receipt['output_sha256']['benchmark.json'] == validated
                    and file_sha(folder / 'benchmark.json') == validated, 'Validated benchmark changed during final closure')
        recheck(bindings)
    except BaseException as exc:
        receipt['errors'].append('final closure: ' + repr(exc))
        receipt['status'] = 'failed'
    receipt['analysis_requirements'] = plan['analysis_required']
    receipt['limitations'] = [
        'Recorder status loss counters are preliminary; actual ETL sample/event/stack completeness and full process lifetime remain analyzer gates.',
        'No per-phase ETW markers in the frozen binary; aggregate timers cannot be converted into exact phase boundaries.',
        'Finally cleanup is scoped to the project instance. External process termination or power loss cannot execute Python finally; retained plan contains only same-instance recovery commands.',
        'Source/binary/PDB hashes are bound; no hermetic build or quiet unprofiled latency claim.']
    write_new(folder / 'capture.json', receipt)
    print(json.dumps({'status': receipt['status'], 'capture': str(folder / 'capture.json'), 'sha256': file_sha(folder / 'capture.json'),
                      'cleanup_confirmed_not_recording': receipt['cleanup_confirmed_not_recording']}))
    return 0 if not receipt['errors'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('run')
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
        return 0
    return run(args.plan, args.expected_plan_sha256)


if __name__ == '__main__':
    sys.exit(main())
