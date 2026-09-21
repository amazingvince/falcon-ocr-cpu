"""Launch exactly the reviewed telescope once and retain native process logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WSL_ROOT = '/mnt/c/Users/amazi/Documents/ChatGPT/falcon-ocr'
ROLE = Path('C:/Users/amazi/.codex/skills/running-wsl-dual-gpu/scripts/gpu-role.sh')
PLAN = ROOT / 'artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/plan.json'
CAPTURE = ROOT / 'artifacts/diagnostics/fp32-prefix-suffix-telescope-gpu-v1'
LAUNCH = ROOT / 'artifacts/diagnostics/fp32-prefix-suffix-telescope-launch-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if sys.platform != 'win32' or digest(PLAN) != args.plan_sha256:
        raise ValueError('Native Windows launcher and exact reviewed plan required')
    if CAPTURE.exists() or LAUNCH.exists():
        raise ValueError('Fresh capture and launch directories required; no retry')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    bound = {str(PLAN): args.plan_sha256,
             str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
             str(ROLE): digest(ROLE)}
    for name, expected in plan['source_sha256'].items():
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError('Source differs: ' + name)
        bound[str(path)] = expected
    clear = ['OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES', 'MKL_DYNAMIC',
             'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'TORCHINDUCTOR_COMPILE_THREADS', 'PYTHONHASHSEED']
    command = ['wsl.exe', '-d', 'Ubuntu-24.04-CUDA', '--cd', WSL_ROOT, '--exec', 'env']
    for key in clear:
        command += ['-u', key]
    command += ['OMP_NUM_THREADS=8', 'MKL_NUM_THREADS=8', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                'bash', '/mnt/c/Users/amazi/.codex/skills/running-wsl-dual-gpu/scripts/gpu-role.sh',
                'isolate', 'teacher', '/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '600',
                '/home/amazi/falcon-ocr-rust-reference/.venv/bin/python',
                'experiments/fp32_prefix_suffix_telescope/export_gpu.py',
                '--plan', 'artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/plan.json',
                '--plan-sha256', args.plan_sha256,
                '--output', 'artifacts/diagnostics/fp32-prefix-suffix-telescope-gpu-v1',
                '--execute-reviewed-plan']
    LAUNCH.mkdir()
    report = {'kind': 'fp32-prefix-suffix-telescope-launch-v1', 'started_utc': now(),
              'command': command, 'cwd': str(ROOT), 'source_sha256_before': bound,
              'purpose': 'One fixed numerical diagnostic; no performance measurement',
              'linux_child_timeout_seconds': 600}
    with (LAUNCH / 'stdout.log').open('xb') as stdout, (LAUNCH / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        report['windows_dispatch_pid'] = child.pid
        write(LAUNCH / 'started.json', report)
        report['exit_code'] = child.wait()
    report['finished_utc'] = now()
    report['source_sha256_after'] = {name: digest(Path(name)) for name in bound}
    report['sources_unchanged'] = report['source_sha256_after'] == bound
    report['output_sha256'] = {p.name: digest(p) for p in (LAUNCH / 'stdout.log', LAUNCH / 'stderr.log')}
    if (CAPTURE / 'report.json').exists():
        report['gpu_report_sha256'] = digest(CAPTURE / 'report.json')
    write(LAUNCH / 'finished.json', report)
    print(json.dumps({'exit_code': report['exit_code'], 'sources_unchanged': report['sources_unchanged'],
                      'launch_receipt_sha256': digest(LAUNCH / 'finished.json')}))
    return report['exit_code'] if report['sources_unchanged'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
