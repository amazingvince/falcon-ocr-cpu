"""One copied-CLI staged-attention smoke after exact operators; no benchmark.

This post-build helper has a separate launch/end source closure. The complete
saved CPU trace is the numerical control; it is not a fresh GPU qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from capture import ROOT, read, sha, write, need, load, closure
from patch import INPUT_PINS, TEST_FILTER, TEST_NAMES

REFERENCE = ROOT / 'artifacts/diagnostics/stage-timing-windows-v1/trace.safetensors'
REFERENCE_SHA = 'e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309'
REFERENCE_RESULT_SHA = '694c726f3d259df9237939f216a1022ed581d082c4f7669aff867d44b4d8f4a9'
FIXTURE = ROOT / 'artifacts/reference/smoke-fp32/trace.safetensors'
FIXTURE_SHA = '30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4'
RUNTIME_ASSET_PINS = {
    'artifacts/model/config.json': 'ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf',
    'artifacts/model/model.safetensors': '3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16',
    'artifacts/model/tokenizer.json': '4a9892af2b1ef021a421f140c7e3c064f5b255f7d75ba18c883996d86e1cf15a',
    'artifacts/model/tokenizer_config.json': '074e03d3fd56d190dac763ec5dfe75a728e15783fc5d919d4b5cbe72bcd24d26',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--preparation-sha256', required=True)
    p.add_argument('--build-sha256', required=True)
    p.add_argument('--operators-sha256', required=True)
    return p


def helper_sources(*extra):
    # These post-build helpers are intentionally not retroactively inserted
    # into the earlier preparation source inventory. Each launch binds its
    # actual helper closure separately, before reading experiment inputs.
    here = Path(__file__).resolve().parent
    return {p.resolve(): sha(p) for p in [here / 'run_smoke.py', here / 'capture.py',
            here / 'patch.py', *extra]}


def check_bound(bound):
    for path, expected in bound.items():
        need(sha(path) == expected, 'Changed bound file: ' + str(path))


def validated_inputs(args, helpers):
    directory, prep = load(args)
    build = read(directory / 'build.json', args.build_sha256)
    need(build['kind'] == 'attention64-staged-build-v1' and build['status'] == 'built_not_executed'
         and build['preparation_sha256'] == args.preparation_sha256, 'Wrong candidate build')
    ops = read(directory / 'operators.json', args.operators_sha256)
    names = sorted(TEST_NAMES)
    need(len(names) == 17 and len(names) == len(set(names))
         and all(name.startswith(TEST_FILTER) for name in names)
         and ops['kind'] == 'attention64-staged-operators-v1'
         and ops['tests'] == names and ops['status'] == 'passed'
         and ops['exit_code'] == 0 and ops['build_sha256'] == args.build_sha256
         and ops['preparation_sha256'] == args.preparation_sha256
         and ops['source_closure'] is True and ops['operator_fixture_sha256'] == INPUT_PINS
         and ops['all_selected_tests_actually_passed'] is True
         and ops['selected_test_count'] == len(names)
         and ops['full_model_or_benchmark_executed'] is False, 'Mandatory staged-operator gate failed')
    operator_binary = build['executables']['operators']
    need(ops['binary_sha256'] == operator_binary['sha256'], 'Operator executable identity differs')
    bound = {**helpers, directory / 'preparation.json': args.preparation_sha256,
             directory / 'build.json': args.build_sha256,
             directory / 'operators.json': args.operators_sha256,
             directory / 'operator.log': ops['log_sha256'],
             directory / operator_binary['path']: operator_binary['sha256'],
             **{ROOT / name: value for name, value in INPUT_PINS.items()},
             **{ROOT / name: value for name, value in RUNTIME_ASSET_PINS.items()}}
    log_bytes = (directory / 'operator.log').read_bytes()
    need(digest(log_bytes) == ops['log_sha256'], 'Operator log changed')
    log = log_bytes.decode('utf-8')
    need(f'{len(names)} passed; 0 failed' in log
         and all(log.splitlines().count('test ' + n + ' ... ok') == 1 for n in names),
         'Operator tests did not actually execute')
    check_bound(bound)
    return directory, prep, build, bound


def close_inputs(directory, prep, bound):
    closure(directory, prep)
    check_bound(bound)


def output_closure(output, expected):
    need({p.name for p in output.iterdir() if p.is_file()} == set(expected),
         'Output artifact inventory changed')
    for name, value in expected.items():
        need(sha(output / name) == value, 'Output artifact changed: ' + name)


def main():
    args = parser(__doc__).parse_args()
    helpers = helper_sources()
    directory, prep, build, bound = validated_inputs(args, helpers)
    need(sys.platform == 'win32', 'Native Windows comparison required for this saved trace')
    item = build['executables']['cli']
    binary = directory / item['path']
    bound.update({binary: item['sha256'], REFERENCE: REFERENCE_SHA,
                  REFERENCE.with_suffix('.json'): REFERENCE_RESULT_SHA, FIXTURE: FIXTURE_SHA})
    check_bound(bound)
    output = directory / 'smoke'
    output.mkdir(exist_ok=False)
    trace = output / 'trace.safetensors'
    command = [str(binary), '--model', str(ROOT / 'artifacts/model'), '--threads', '4',
               '--backend', 'avx2', '--precision', 'fp32', '--batch-size', '1',
               '--cache-layout', 'compact', '--weight-layout', 'unpacked',
               'trace', '--fixture', str(FIXTURE), '--output', str(trace), '--max-new-tokens', '17']
    report = {'kind': 'attention64-staged-smoke-v1', 'status': 'failed',
              'preparation_sha256': args.preparation_sha256, 'build_sha256': args.build_sha256,
              'operators_sha256': args.operators_sha256, 'binary_sha256': item['sha256'],
              'reference_sha256': REFERENCE_SHA, 'fixture_sha256': FIXTURE_SHA,
              'helper_source_sha256': {str(p): h for p, h in helpers.items()},
              'input_sha256': {str(p): h for p, h in bound.items()},
              'source_and_input_closure': False, 'performance_claim': False, 'gpu_qualification_claim': False,
              'control_scope': 'Immutable native Windows CPU trace, 1904 complete tensors and 17 teacher IDs; no fresh control inference.',
              'historical_control_cache_layout': 'expanded', 'candidate_cache_layout': 'compact',
              'helper_provenance': 'Separate model-launch source closure; not retroactively included in build preparation.'}
    try:
        write(output / 'invocation.json', {**report, 'command': command, 'cwd': str(ROOT),
              'timeout_seconds': 600, 'platform': platform.platform(), 'python': sys.version,
              'environment_overrides': {'CUDA_VISIBLE_DEVICES': '-1'}})
        with (output / 'stdout.log').open('x') as out, (output / 'stderr.log').open('x') as err:
            code = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'),
                                  stdout=out, stderr=err, timeout=600).returncode
        report['exit_code'] = code
        need(code == 0, 'CLI failed; preserve this attempt')
        raw = trace.read_bytes()
        trace_sha = digest(raw)
        need(len(raw) >= 8, 'Invalid trace header')
        header_len = int.from_bytes(raw[:8], 'little')
        need(8 + header_len <= len(raw), 'Truncated trace header')
        header = json.loads(raw[8:8 + header_len])
        count = len([name for name in header if name != '__metadata__'])
        result_bytes = trace.with_suffix('.json').read_bytes()
        result = json.loads(result_bytes)
        control = read(REFERENCE.with_suffix('.json'), REFERENCE_RESULT_SHA)
        fields = ['token_ids', 'text', 'finish_reason', 'output_tokens', 'input_tokens', 'precision',
                  'teacher_forced', 'width', 'height', 'backend', 'weight_layout', 'packed_weight_bytes']
        need(result['cache_layout'] == 'compact' and control['cache_layout'] == 'expanded', 'Wrong cache layout')
        need(all(result.get(f) == control[f] for f in fields), 'Result differs from saved CPU control')
        need(trace_sha == REFERENCE_SHA and count == 1904, 'Trace not byte-exact')
        need(len(result['token_ids']) == result['output_tokens'] == 17
             and result['teacher_forced'] is True and result['finish_reason'] == 'length', 'Wrong teacher trace')
        need(not any(k in result for k in ('postprocessing_replay', 'derived_text_replay')),
             'Unexpected replay metadata')
        artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        need(artifacts['trace.safetensors'] == trace_sha and artifacts['trace.json'] == digest(result_bytes),
             'Validated outputs changed before closure')
        close_inputs(directory, prep, bound)
        output_closure(output, artifacts)
        report.update(status='bit_exact', trace_sha256=trace_sha, tensor_count=count,
                      teacher_token_count=17, compared_result_fields=fields,
                      source_and_input_closure=True, output_sha256=artifacts)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if 'output_sha256' not in report:
            report['output_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report_sha256': sha(output / 'report.json')}))


if __name__ == '__main__':
    main()
