"""Run one existing offline tool on the completed compact capture; no recording."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
FOLDER = Path('D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1')
BUILD = ROOT / 'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
AUDIT = ROOT / 'artifacts/tools/trace-audit-build-v4/TraceAudit.exe'
PERFVIEW = ROOT / 'artifacts/tools/perfview-3.2.6/PerfView.exe'
DEPENDENCIES = Path('C:/Users/amazi/AppData/Roaming/PerfView/VER.2026-09-20.08.05.04.237')
PINS = {
    FOLDER / 'plan.json': '530cfc0c5808cc525722379c0e0e0a2518d0f3d7067e62e9819030407080332b',
    FOLDER / 'capture.json': '4e80ebf0bee2ed06b3e8a4f7d3400c5ff5b5ef8d3d23299d292dafb37851780b',
    FOLDER / 'cpu.etl': 'dbfb7ae714455ccb01db02a65b1726d9437a53daae7ba2b9a08d58e132bf1111',
    ROOT / 'artifacts/diagnostics/fullpage-compact-profile-launch-v1/launch.json':
        '36ba7131f04bd7c84a73dc4d594af2d557cbf43680177a500967dc7b395a3ec9',
    AUDIT: '930cace564616e131a94f1f9b871c73a2e2536e96692aa16149a0d8aa18a5a1e',
    PERFVIEW: '84b8523f7fb4783fd0baae6b080adb1b5aac388192145ad713e504c69954556d',
    DEPENDENCIES / 'Microsoft.Diagnostics.Tracing.TraceEvent.dll':
        '530946dc20e89754783f0ac76d86b7a4eedf95326f255db34c32fcbf5c3ce0ff',
    BUILD / 'ocr_bench.pdb': '1e7e0f0d0640a94daec5c9b990fd37d4dfdaa6c7e868360fd61f1bb137abcba4',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write('\n')


def check(bound):
    for path, expected in bound.items():
        need(sha(path) == expected, 'Changed bound input/tool/source: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audit', 'export'])
    phase = parser.parse_args().phase
    need(sys.platform == 'win32', 'Native Windows offline tools required')
    bound = {**PINS, Path(__file__).resolve(): sha(Path(__file__).resolve())}
    check(bound)
    capture_raw = (FOLDER / 'capture.json').read_bytes()
    need(hashlib.sha256(capture_raw).hexdigest() == PINS[FOLDER / 'capture.json'], 'Capture read changed')
    capture = json.loads(capture_raw)
    need(capture['cleanup_confirmed_not_recording'] is True and not capture['errors']
         and capture['process']['exit_code'] == 0
         and capture['benchmark_validation']['completed_full_page_outputs'] == 3
         and capture['benchmark_validation']['literal_output_matches_frozen_baseline'] is True,
         'Completed exact-output capture required')
    plan = json.loads((FOLDER / 'plan.json').read_bytes())
    bound.update({Path(p): h for p, h in plan['files_sha256'].items()})
    check(bound)
    env = dict(os.environ)
    if phase == 'audit':
        output = FOLDER / 'trace-audit-v4.json'
        command = [str(AUDIT), '--etl', str(FOLDER / 'cpu.etl'), '--pid', str(capture['process']['pid']),
                   '--output', str(output), '--expected-image', 'ocr_bench.exe',
                   '--expected-command-line', subprocess.list2cmdline(capture['process']['command'])]
        overrides = {'TRACEAUDIT_DEPENDENCIES': str(DEPENDENCIES)}
        candidates = [output, Path(str(output) + '.etlx'), Path(str(output) + '.conversion.log')]
    else:
        command = plan['perfview_stacks_template']
        need(command[0] == str(PERFVIEW) and command[4:] ==
             ['UserCommand', 'SaveCPUStacks', str(FOLDER / 'cpu.etl'), 'ocr_bench'], 'Unexpected exporter command')
        overrides = {'_NT_SYMBOL_PATH': str(BUILD)}
        candidates = [FOLDER / n for n in ('perfview-stacks.log', 'cpu.etlx', 'cpu.perfView.xml.zip')]
    need(all(not path.exists() for path in candidates), 'Preserve previous offline attempt')
    start = FOLDER / (phase + '-execution-start.json')
    result = FOLDER / (phase + '-execution.json')
    stdout, stderr = (FOLDER / (phase + '-tool.' + suffix + '.log') for suffix in ('stdout', 'stderr'))
    need(all(not path.exists() for path in (start, result, stdout, stderr)), 'Fresh execution artifacts required')
    env.update(overrides)
    receipt = {'kind': 'compact-profile-offline-execution-v1', 'phase': phase,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
               'cwd': str(ROOT), 'environment_overrides': overrides, 'timeout_seconds': 600,
               'bound_sha256': {str(p): h for p, h in bound.items()},
               'new_model_or_recording_execution': False, 'tool_exit_is_separate_from_payload_acceptance': True}
    write(start, receipt)
    code = None
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                  creationflags=subprocess.CREATE_NO_WINDOW, timeout=600).returncode
        receipt['exit_code'] = code
        check(bound)
        receipt['inputs_and_sources_unchanged'] = True
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['outputs_sha256'] = {str(p): sha(p) for p in [stdout, stderr, *candidates] if p.is_file()}
        write(result, receipt)
    print(json.dumps({'phase': phase, 'tool_exit_code': code, 'receipt': str(result), 'sha256': sha(result)}))
    return code


if __name__ == '__main__':
    sys.exit(main())
