"""One unchanged eight-interval allocation test after staged operators and smoke."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from run_smoke import (ROOT, read, sha, write, need, parser, helper_sources, validated_inputs,
                       check_bound, close_inputs, output_closure, digest, REFERENCE_SHA)

TEST = 'warm_fp32_decode_has_no_heap_allocations'
TEST_SHA = 'c2dac69087350a3c3669294e80f88d4a4c291950d8159285b6853b9eacff3bde'


def main():
    p = parser(__doc__)
    p.add_argument('--smoke-sha256', required=True)
    args = p.parse_args()
    helpers = helper_sources(Path(__file__).resolve())
    directory, prep, build, bound = validated_inputs(args, helpers)
    need(sys.platform == 'win32', 'This allocation receipt is native Windows only')
    smoke_path = directory / 'smoke/report.json'
    smoke = read(smoke_path, args.smoke_sha256)
    need(smoke['kind'] == 'attention64-staged-smoke-v1' and smoke['status'] == 'bit_exact'
         and smoke['build_sha256'] == args.build_sha256
         and smoke['preparation_sha256'] == args.preparation_sha256
         and smoke['operators_sha256'] == args.operators_sha256
         and smoke['source_and_input_closure'] is True
         and smoke['trace_sha256'] == REFERENCE_SHA and smoke['tensor_count'] == 1904
         and smoke['teacher_token_count'] == 17, 'Exact candidate smoke gate failed')
    bound[smoke_path] = args.smoke_sha256
    for name, value in smoke['input_sha256'].items():
        path = Path(name)
        need(path not in bound or bound[path] == value, 'Conflicting smoke input identity')
        bound[path] = value
    for name, value in smoke['output_sha256'].items():
        path = directory / 'smoke' / name
        need(path.resolve().parent == (directory / 'smoke').resolve(), 'Invalid smoke output path')
        bound[path] = value
    item = build['executables']['allocations']
    binary = directory / item['path']
    test_source = directory / 'project/tests/decode_allocations.rs'
    need(prep['project_source_sha256']['tests/decode_allocations.rs'] == TEST_SHA, 'Allocation test source changed')
    image = ROOT / 'artifacts/reference/smoke-fp32/canonical-rgb.png'
    # No historical canonical-PNG digest is asserted here: record the actual
    # launch bytes and require them unchanged after all eight intervals.
    image_sha = sha(image)
    bound.update({binary: item['sha256'], test_source: TEST_SHA, image: image_sha})
    check_bound(bound)
    output = directory / 'allocations'
    output.mkdir(exist_ok=False)
    command = [str(binary), TEST, '--exact', '--ignored', '--show-output', '--test-threads', '1']
    report = {'kind': 'attention64-staged-allocations-v1', 'status': 'failed',
              'preparation_sha256': args.preparation_sha256, 'build_sha256': args.build_sha256,
              'operators_sha256': args.operators_sha256, 'smoke_sha256': args.smoke_sha256,
              'binary_sha256': item['sha256'], 'test_source_sha256': TEST_SHA,
              'allocation_image': str(image), 'allocation_image_sha256': image_sha,
              'allocation_image_identity_source': 'actual launch bytes, not historical startup attestation',
              'helper_source_sha256': {str(p): h for p, h in helpers.items()},
              'input_sha256': {str(p): h for p, h in bound.items()},
              'source_and_input_closure': False, 'performance_claim': False,
              'helper_provenance': 'Separate model-launch source closure; not retroactively included in build preparation.',
              'gpu_qualification_claim': False}
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
        listing = subprocess.run([str(binary), '--list'], env=env, cwd=ROOT,
                                 capture_output=True, timeout=60)
        with (output / 'test-list.stdout').open('xb') as f:
            f.write(listing.stdout)
        with (output / 'test-list.stderr').open('xb') as f:
            f.write(listing.stderr)
        need(listing.returncode == 0 and listing.stdout.decode('utf-8').splitlines().count(TEST + ': test') == 1,
             'Wrong allocation test binary inventory')
        write(output / 'invocation.json', {**report, 'command': command, 'cwd': str(ROOT),
              'timeout_seconds': 600, 'platform': platform.platform(), 'python': sys.version,
              'listing_command': [str(binary), '--list'], 'environment_overrides': {'CUDA_VISIBLE_DEVICES': '-1'}})
        with (output / 'run.log').open('x') as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=600).returncode
        report['exit_code'] = code
        raw_log = (output / 'run.log').read_bytes()
        text = raw_log.decode('utf-8')
        need(code == 0 and '1 passed; 0 failed' in text and 'test ' + TEST + ' ... ok' in text,
             'Allocation test did not pass; preserve this attempt')
        artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        need(artifacts['run.log'] == digest(raw_log), 'Validated log changed before closure')
        need(artifacts['test-list.stdout'] == digest(listing.stdout)
             and artifacts['test-list.stderr'] == digest(listing.stderr), 'Validated listing changed before closure')
        close_inputs(directory, prep, bound)
        output_closure(output, artifacts)
        report.update(status='passed', command=command, run_log_sha256=digest(raw_log),
                      asserted_warmed_decode_intervals=8, asserted_allocations_per_interval=0,
                      intervals='expanded/compact × unpacked/phase-packed × single/batch4, unchanged test',
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
