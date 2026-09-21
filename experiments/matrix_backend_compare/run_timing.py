"""Run the preserved operator probe once; do not build or modify its frozen sources."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'artifacts/diagnostics/matrix-backends-rust194-v1'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    build_path = FOLDER/'build.json'
    accuracy_path = FOLDER/'accuracy-v1/report.json'
    assert sha(build_path) == '9f3b8f9660ca2173e1feba87f132a46fc3cb180913e84eb84c75e78e52a932bb'
    assert sha(accuracy_path) == '504547b2bec9e4f48abc1354a72808eb73861d713879c54a7496ba9c4d82a73a'
    build = json.loads(build_path.read_text())
    exe = FOLDER/build['executable']
    assert sha(exe) == build['binary_sha256'] == build['copied_binary_sha256']
    output = FOLDER/'timing-v1'
    assert not output.exists()
    command = [str(exe), '--root', str(ROOT), '--output', str(output), '--measure', '--prefill']
    before = {str(path): sha(path) for path in [build_path, accuracy_path, exe, Path(__file__).resolve()]}
    write(FOLDER/'timing-invocation-v1.json', {
        'command': command, 'cwd': str(ROOT), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': before, 'cpu': 'AMD Ryzen 9 7950X',
        'quiet_window': 'All owned builds/models/profilers are terminal; immediate Get-Process scan showed no cargo, rustc, falcon, ocr_bench, python, PerfView or WPR process. Collaborator restricted to small source-only reads. No claim of OS-wide isolation.',
        'prospective_decision': 'Operator screening only. Consider isolated model integration if repeatable operator gains justify it; require unchanged full-model quality gates and the existing >=5% page target before promotion.',
        'scope': 'One process, 16-thread pool; 2 warm suites and 7 samples of 5 suites per arm, control/candidate/control. RTen chooses AVX512 while single-row control uses AVX2. No full-page inference.'})
    with (FOLDER/'timing-v1.log').open('xb') as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    closure = before == {path: sha(Path(path)) for path in before}
    write(FOLDER/'timing-execution-v1.json', {'exit_code': process.returncode,
          'finished_utc': datetime.now(timezone.utc).isoformat(), 'controller_inputs_unchanged': closure,
          'log_sha256': sha(FOLDER/'timing-v1.log')})
    assert process.returncode == 0 and closure
    report_path = output/'report.json'
    report = json.loads(report_path.read_text())
    print('report_sha256', sha(report_path))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
