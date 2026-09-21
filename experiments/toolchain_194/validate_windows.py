"""Run a fresh 1.94 trace and compare exact tensor bytes with the saved 1.92 CPU trace."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/diagnostics/toolchain-1.94-windows-trace-v2'
EXE = Path('D:/falcon-ocr-rust-builds/toolchain-1.94-v1/release/falcon-ocr.exe')

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def save(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')

def tensors(path):
    data = path.read_bytes()
    size = int.from_bytes(data[:8], 'little')
    header = json.loads(data[8:8+size])
    payload = memoryview(data)[8+size:]
    return {name: (item['dtype'], item['shape'], payload[item['data_offsets'][0]:item['data_offsets'][1]])
            for name, item in header.items() if name != '__metadata__'}

def tests(path):
    raw = path.read_bytes()
    text = raw.decode('utf-16') if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else raw.decode('utf-8-sig')
    rows = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    assert rows and 'test result: FAILED' not in text
    totals = [sum(int(row[i]) for row in rows) for i in range(3)]
    return dict(zip(('passed', 'failed', 'ignored'), totals)) | {'harnesses': len(rows), 'sha256': sha(path)}

def main():
    OUT.mkdir(exist_ok=False)
    build_path = ROOT / 'artifacts/builds/toolchain-1.94-windows-v1/build.json'
    build = json.loads(build_path.read_text())
    fixture = ROOT / 'artifacts/reference/smoke-fp32/trace.safetensors'
    reference = ROOT / 'artifacts/cpu/smoke-trace-sinks-pairwise.safetensors'
    assert sha(reference) == 'e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309'
    assert sha(fixture) == '30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4'
    source_now = {name: sha(ROOT/name) for name in build['source_sha256']}
    assert source_now == build['source_sha256']
    tracked = [EXE, build_path, fixture, reference, Path(__file__).resolve()]
    before = {str(path): sha(path) for path in tracked}
    trace = OUT / 'trace.safetensors'
    command = [str(EXE), '--model', str(ROOT/'artifacts/model'), '--threads', '4',
               '--backend', 'avx2', '--precision', 'fp32', '--cache-layout', 'expanded',
               'trace', '--fixture', str(fixture), '--output', str(trace), '--max-new-tokens', '17']
    save(OUT/'invocation.json', {'command': command, 'cwd': str(ROOT),
         'started_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': before,
         'source_matches_captured_benchmark_build': True,
         'binary_origin': 'CLI emitted by the completed --lib --bins --tests release test build; separate from captured ocr_bench executable'})
    with (OUT/'execution.log').open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.returncode
    expected, actual = tensors(reference), tensors(trace)
    assert expected.keys() == actual.keys()
    mismatches = [name for name in expected if expected[name] != actual[name]]
    old_result = json.loads(reference.with_suffix('.json').read_text())
    new_result = json.loads(trace.with_suffix('.json').read_text())
    result_fields = ('text', 'token_ids', 'finish_reason', 'input_tokens', 'output_tokens', 'teacher_forced')
    result_exact = all(old_result[key] == new_result[key] for key in result_fields)
    after = {str(path): sha(path) for path in tracked}
    unchanged = before == after and source_now == {name: sha(ROOT/name) for name in source_now}
    report = {'kind': 'rust-1.94-upgrade-validation-v1', 'trace_exit_code': result.returncode,
              'tensor_count': len(expected), 'mismatched_tensors': mismatches,
              'result_fields_exact': result_exact, 'result_fields': result_fields,
              'input_binary_source_unchanged': unchanged,
              'trace_sha256': sha(trace), 'invocation_sha256': sha(OUT/'invocation.json'),
              'windows_regression': tests(ROOT/'artifacts/diagnostics/toolchain-1.94-windows-tests-v1.log'),
              'wsl_regression': tests(ROOT/'artifacts/diagnostics/toolchain-1.94-linux-tests-v1.log'),
              'windows_gpu_reference_smoke': tests(ROOT/'artifacts/diagnostics/toolchain-1.94-windows-smoke-v2.log'),
              'wsl_gpu_reference_smoke': {'passed': 1, 'failed': 0, 'evidence': 'Observed terminal tool chunk 24d99f, exit 0; no separate saved log. CPU-only test uses existing GPU reference.'},
              'rustc_version': build['rustc_version'], 'cargo_version': build['cargo_version'],
              'limits': ['No new full corpus or performance qualification.', 'WSL is not bare-metal Linux.',
                         'Exact CPU trace agreement does not resolve the ten existing GPU numerical mismatches.',
                         'CLI build provenance is the saved test log and current source closure, not a separately captured Cargo JSON artifact.']}
    save(OUT/'report.json', report)
    print(json.dumps(report, indent=2))
    assert not mismatches and result_exact and unchanged
    assert len(expected) == 1904

if __name__ == '__main__':
    main()
