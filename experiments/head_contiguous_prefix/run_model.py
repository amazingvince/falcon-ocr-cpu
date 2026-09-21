"""Fresh Compact/candidate functional comparison; never a timing measurement."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

import capture

ROOT = capture.ROOT
FUNCTION = 'head_prefix_full_model_qualification'
LAYOUTS = ('compact', 'head_contiguous_prefix')
PINS = {
    'artifacts/model/artifact-manifest.json': 'eea9640e8ab9feccd9358a6b3526e579b14dc4bfcd47ecd561853863952b594d',
    'artifacts/reference/smoke-fp32/metadata.json': 'c84b1fcc4526135677579392d2c2c52b1566c1f361ac950a6409a5145a8fc65e',
    'artifacts/reference/smoke-fp32/canonical-rgb.png': 'd62f8db10dca06c65c99ae4dde59e4a68b9548c56a42c76caa2bf7044ccca0db',
    'artifacts/reference/smoke-fp32/trace.safetensors': '30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4',
    'artifacts/cpu/smoke-trace-sinks-pairwise.safetensors': 'e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309',
}
RUNTIME = {'precision': 'fp32', 'backend': 'avx2', 'threads': 4, 'weight_layout': 'unpacked',
           'min_dimension': 64, 'max_dimension': 256, 'max_new_tokens': 24, 'mixed_batch_size': 4}
ALLOCATIONS = {'decode_starts': 1, 'decode_ends': 1, 'allocation_calls': 0, 'requested_bytes': 0}
need, sha, write = capture.need, capture.sha, capture.write


def canonical_identities(path):
    from safetensors import safe_open
    import hashlib
    import numpy as np
    result = {}
    with safe_open(str(path), framework='np') as tensors:
        need(len(tensors.keys()) == 1904, 'Saved canonical inventory differs')
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            need(value.dtype == np.dtype('<f4') and np.isfinite(value).all(), 'Invalid saved tensor')
            result[name] = {'dtype': 'F32-le', 'shape': list(value.shape), 'elements': int(value.size),
                            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    return result


def validate_result(value, layout, identities, metadata):
    need(value['schema_version'] == 1 and value['status'] == 'completed'
         and value['cache_layout'] == layout and value['runtime'] == RUNTIME
         and value['performance_measurement'] is False, 'Wrong model result contract')
    need(value['model_revision'] == metadata['model_revision']
         and value['weights_sha256'] == metadata['weights_sha256'], 'Wrong checkpoint')
    canonical = value['canonical']
    need(canonical['path_kind'] == 'teacher_forced_same_prefix', 'Wrong teacher path')
    need(canonical['trace']['tensors'] == identities, 'Saved 1904 CPU tensor identities differ')
    ids = metadata['token_ids']
    need(len(ids) == 17 and canonical['result']['token_ids'] == ids
         and canonical['result']['teacher_forced'] is True
         and canonical['result']['finish_reason'] == 'length', 'Wrong teacher output')
    decisions = canonical['trace']['logits_argmax']
    need([name for name, _ in decisions] == ['prefill.logits'] + [f'decode.{i}.logits' for i in range(16)]
         and [row for _, row in decisions] == [[token] for token in ids], 'Actual teacher argmax differs')
    single, mixed = value['independent_free_single'], value['independent_free_mixed']
    need(single == mixed and len(single) == 3 and single[0]['token_ids'] == ids, 'Free outputs differ')
    need([row['output_tokens'] for row in single] == [17, 2, 6]
         and all(row['finish_reason'] == 'eos' and row['teacher_forced'] is False
                 and row['output_tokens'] == len(row['token_ids']) for row in single), 'Free stop/count differs')
    trace = value['mixed_trace']
    need(len(trace['tensors']) == 2144, 'Mixed trace inventory differs')
    logits = trace['logits_argmax']
    need([name for name, _ in logits] == [f'request.{i}.prefill.logits' for i in range(3)]
         + [f'batch.0.decode.{i}.logits' for i in range(16)], 'Mixed logits inventory differs')
    need(len(trace['decode_rows']) == len(trace['active_request_indices']) == 16, 'Missing active-row history')
    derived = [[logits[i][1][0]] for i in range(3)]
    need(all(len(logits[i][1]) == 1 for i in range(3)), 'Prefill decision shape differs')
    for step in range(16):
        active = [i for i, row in enumerate(mixed) if row['output_tokens'] > step + 1]
        need(trace['active_request_indices'][step] == [f'batch.0.decode.{step}.request_indices', active]
             and trace['decode_rows'][step] == len(active)
             and len(logits[step + 3][1]) == len(active), 'Active session mapping differs')
        for index, token in zip(active, logits[step + 3][1]):
            derived[index].append(token)
    need(derived == [row['token_ids'] for row in mixed], 'Mixed argmax/output disagreement')
    for name in ('single', 'mixed'):
        need(value['allocation'][name] == ALLOCATIONS, 'Warmed allocation interval failed: ' + name)


def compare_modes(before, after):
    need(before['cache_layout'] == LAYOUTS[0] and after['cache_layout'] == LAYOUTS[1],
         'Wrong ordered layout comparison')
    normalized = [{k: v for k, v in row.items() if k != 'cache_layout'} for row in (before, after)]
    need(normalized[0] == normalized[1], 'Fresh Compact and candidate differ beyond layout label')


def validate_operator_group(group, selected, names, operator_exe, index):
    need(names and group['filter'] == selected and group['tests'] == names
         and group['passed'] is True and group['exit_code'] == 0
         and group['log'] == f'operator-{index}.log', 'Wrong operator group')
    need(group['command'] == [str(operator_exe), selected, '--show-output', '--test-threads', '1'],
         'Wrong operator command')


def run(args):
    directory, preparation = capture.load(args)
    need(sha(Path(__file__)) == args.runner_sha256, 'Wrong reviewed execution source')
    build = capture.read(directory / 'build.json', args.build_sha256)
    need(build['kind'] == 'head-contiguous-prefix-build-v1' and build['status'] == 'built_not_executed'
         and build['preparation_sha256'] == args.preparation_sha256, 'Wrong build')
    operators = capture.read(directory / 'operators.json', args.operators_sha256)
    need(operators['status'] == 'passed' and operators['source_closure'] is True
         and operators['all_selected_tests_actually_passed'] is True
         and operators['build_sha256'] == args.build_sha256
         and operators['preparation_sha256'] == args.preparation_sha256
         and operators['tests'] == sorted(capture.TEST_NAMES)
         and operators['selected_test_count'] == len(capture.TEST_NAMES), 'Operators unqualified')
    need(operators['binary_sha256'] == build['executables']['operators']['sha256']
         and operators['full_model_or_benchmark_executed'] is False, 'Wrong operator binary/scope')
    bound = {ROOT / name: digest for name, digest in PINS.items()}
    bound.update({Path(__file__).resolve(): args.runner_sha256,
                  directory / 'preparation.json': args.preparation_sha256,
                  directory / 'build.json': args.build_sha256,
                  directory / 'operators.json': args.operators_sha256})
    def stable():
        capture.closure(directory, preparation)
        for path, digest in bound.items():
            need(sha(path) == digest, 'Bound input/source changed: ' + str(path))
    need(len(operators['groups']) == len(capture.TEST_FILTERS), 'Missing operator group')
    operator_exe = directory / build['executables']['operators']['path']
    bound[operator_exe] = build['executables']['operators']['sha256']
    for index, (group, selected) in enumerate(zip(operators['groups'], capture.TEST_FILTERS)):
        names = sorted(name for name in capture.TEST_NAMES if selected in name)
        validate_operator_group(group, selected, names, operator_exe, index)
        logpath = directory / group['log']
        bound[logpath] = group['log_sha256']
        raw = logpath.read_bytes()
        need(capture.digest(raw) == group['log_sha256'], 'Operator log changed')
        log = raw.decode('utf-8')
        need(f'{len(names)} passed; 0 failed' in log
             and all(log.splitlines().count(f'test {name} ... ok') == 1 for name in names), 'Missing test pass')
    item = build['executables']['qualification']
    need(item['path'] == 'qualification.exe', 'Wrong model test executable')
    exe = directory / item['path']
    bound[exe] = item['sha256']
    stable()
    manifest = capture.read(ROOT / 'artifacts/model/artifact-manifest.json',
                            PINS['artifacts/model/artifact-manifest.json'])
    for name, identity in manifest['files'].items():
        path = (ROOT / 'artifacts/model' / name).resolve()
        need(path.is_relative_to(ROOT / 'artifacts/model'), 'Invalid asset path')
        need(path.stat().st_size == identity['bytes'], 'Asset size changed')
        bound[path] = identity['sha256']
    stable()
    identities = canonical_identities(ROOT / 'artifacts/cpu/smoke-trace-sinks-pairwise.safetensors')
    metadata = capture.read(ROOT / 'artifacts/reference/smoke-fp32/metadata.json',
                            PINS['artifacts/reference/smoke-fp32/metadata.json'])
    output = directory / 'model-run-v1'
    need(not output.exists(), 'Preserve previous attempt')
    output.mkdir()
    report = {'kind': 'head-contiguous-prefix-model-comparison-v1', 'status': 'started',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': [],
              'preparation_sha256': args.preparation_sha256, 'build_sha256': args.build_sha256,
              'operators_sha256': args.operators_sha256, 'runtime': RUNTIME,
              'performance_measurement': False, 'source_input_closure': False,
              'limits': 'Fixed smoke and uneven three-request functional checks only; no GPU intermediate tolerance change, corpus qualification, RSS or speed claim.'}
    failure = None
    results = []
    try:
        for layout in LAYOUTS:
            stable()
            target = output / (layout + '.json')
            overrides = {'CUDA_VISIBLE_DEVICES': '-1', 'FOCR_HEAD_PREFIX_MODEL_LAYOUT': layout,
                         'FOCR_HEAD_PREFIX_MODEL_OUTPUT': str(target)}
            command = [str(exe), '--ignored', '--exact', FUNCTION, '--test-threads=1', '--nocapture']
            job = {'layout': layout, 'command': command, 'cwd': str(ROOT),
                   'environment_overrides': overrides, 'binary_sha256': item['sha256'], 'timeout_seconds': 1800}
            write(output / (layout + '-invocation.json'), job)
            report['jobs'].append(job)
            with (output / (layout + '.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **overrides),
                                         stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                job['pid'] = child.pid
                try:
                    write(output / (layout + '-process-start.json'),
                          {'pid': child.pid, 'started_utc': datetime.now(timezone.utc).isoformat(),
                           'binary_sha256': item['sha256'], 'layout': layout})
                    job['exit_code'] = child.wait(timeout=1800)
                finally:
                    if child.poll() is None:
                        child.kill()
                    job['exit_code'] = child.wait()
                    job['owned_process_reaped'] = True
            logpath = output / (layout + '.log')
            raw = logpath.read_bytes()
            bound[logpath] = capture.digest(raw)
            need(job['exit_code'] == 0, 'Model test failed: ' + layout)
            log = raw.decode('utf-8')
            need(f'test {FUNCTION} ... ok' in log and '1 passed; 0 failed; 0 ignored' in log,
                 'Exact model test did not pass')
            raw = target.read_bytes()
            bound[target] = capture.digest(raw)
            value = json.loads(raw)
            validate_result(value, layout, identities, metadata)
            results.append(value)
        compare_modes(*results)
        report.update(status='passed', canonical_cpu_tensors_exact=1904,
                      mixed_tensors_exact=2144, actual_teacher_argmax_exact=17,
                      independent_eos_counts=[17, 2, 6], fresh_processes=2,
                      warmed_zero_allocation_intervals=4)
    except BaseException as error:
        failure = error
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        try:
            for path in output.iterdir():
                if path.is_file() and path not in bound:
                    bound[path] = sha(path)
            stable()
            report['source_input_closure'] = True
        except BaseException as error:
            failure = failure or error
            report.update(status='failed', closure_error=repr(error))
        report['bound_sha256'] = {str(path): digest for path, digest in bound.items()}
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(output / 'comparison.json', report)
    print(json.dumps({'status': report['status'], 'report_sha256': sha(output / 'comparison.json')}))
    if failure:
        raise failure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    for name in ('preparation', 'build', 'operators', 'runner'):
        parser.add_argument('--' + name + '-sha256', required=True)
    run(parser.parse_args())
