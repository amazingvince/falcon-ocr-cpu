"""Prepare and build one isolated faer/control operator probe; never run it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTROL = ROOT / 'artifacts/diagnostics/attention64-compact-v1/benchmark-build'
CONTROL_BUILD = '68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0'
CONTROL_ARCHIVE = 'f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3'
KERNELS = 'ad812805e03b536de0408d6f0dd21d3e944d6d752bc510331c59da44fe80fc5c'
NAME = 'falcon-matrix-backend-probe'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(path, expected):
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected, 'Changed input: ' + str(path))
    return json.loads(raw)


def sources():
    return {p: sha(p) for p in [HERE / n for n in
        ['main.rs', 'candidate.rs', 'Cargo.toml', 'Cargo.lock', 'capture.py']]
        + [ROOT / 'rust-toolchain.toml']}


def closure(folder, prep):
    need({str(p): h for p, h in sources().items()} == prep['source_sha256'], 'Probe sources changed')
    need(sha(CONTROL / 'build.json') == CONTROL_BUILD and
         sha(CONTROL / 'source.zip') == CONTROL_ARCHIVE, 'Frozen control changed')
    need(sha(folder / 'source.zip') == prep['source_archive_sha256'], 'Probe archive changed')
    for name, digest in prep['project_sha256'].items():
        need(sha(folder / 'project' / name) == digest, 'Copied source changed: ' + name)


def prepare(args):
    folder = args.output.resolve()
    need(folder.is_relative_to(ROOT) and folder != ROOT and not folder.exists(), 'Require fresh workspace output')
    original = read(CONTROL / 'build.json', CONTROL_BUILD)
    need(sha(CONTROL / 'source.zip') == CONTROL_ARCHIVE, 'Wrong baseline archive')
    bound = sources()
    with zipfile.ZipFile(CONTROL / 'source.zip') as archive:
        kernel = archive.read('src/kernels.rs')
    need(hashlib.sha256(kernel).hexdigest() == KERNELS == original['source_sha256']['src/kernels.rs'],
         'Wrong baseline kernel bytes')
    project = {n: (HERE / n).read_bytes() for n in ['main.rs', 'candidate.rs', 'Cargo.toml', 'Cargo.lock']}
    project['baseline_kernels.rs'] = kernel
    project['rust-toolchain.toml'] = (ROOT / 'rust-toolchain.toml').read_bytes()
    need(b'channel = "1.94.0"' in project['rust-toolchain.toml'], 'Expected upgraded pinned toolchain')
    folder.mkdir(parents=True)
    (folder / 'project').mkdir()
    with zipfile.ZipFile(folder / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in project.items():
            (folder / 'project' / name).write_bytes(raw)
            archive.writestr(name, raw)
    prep = {'kind': 'matrix-backend-probe-preparation-v1', 'status': 'prepared_not_built',
            'source_sha256': {str(p): h for p, h in bound.items()},
            'project_sha256': {n: hashlib.sha256(raw).hexdigest() for n, raw in project.items()},
            'source_archive_sha256': sha(folder / 'source.zip'),
            'baseline_build_sha256': CONTROL_BUILD, 'baseline_archive_sha256': CONTROL_ARCHIVE,
            'baseline_kernels_sha256': KERNELS,
            'scope': 'Same-compiler operator comparison only; no production dependency or performance qualification'}
    closure(folder, prep)
    write(folder / 'preparation.json', prep)
    print(json.dumps({'preparation_sha256': sha(folder / 'preparation.json')}))


def build(args):
    folder = args.prepared.resolve()
    need(folder.is_relative_to(ROOT), 'Prepared folder outside workspace')
    prep = read(folder / 'preparation.json', args.preparation_sha256)
    closure(folder, prep)
    target = args.target.resolve()
    need(sys.platform == 'win32' and target.drive.upper() == 'D:' and not target.exists(),
         'Require fresh native Windows D: target')
    need(not (folder / 'build-start.json').exists(), 'Build already attempted')
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), RUSTUP_TOOLCHAIN='1.94.0')
    need(not any(env.get(k) for k in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET']),
         'Unexpected compiler overrides')
    project = folder / 'project'
    rustc = subprocess.check_output(['rustc', '-Vv'], cwd=project, env=env, text=True)
    cargo = subprocess.check_output(['cargo', '-V'], cwd=project, env=env, text=True)
    need('release: 1.94.0\n' in rustc, 'Wrong compiler')
    command = ['cargo', 'build', '--locked', '--offline', '--release', '--bin', NAME,
               '--jobs', '2', '--message-format=json-render-diagnostics']
    record = {'kind': 'matrix-backend-probe-build-v1', 'status': 'failed',
              'preparation_sha256': args.preparation_sha256, 'command': command,
              'cwd': str(project), 'target': str(target), 'rustc_version': rustc, 'cargo_version': cargo,
              'environment_overrides': {k: env[k] for k in ['CARGO_TARGET_DIR', 'RUSTUP_TOOLCHAIN']}}
    write(folder / 'build-start.json', record)
    with (folder / 'build.log').open('x', encoding='utf-8') as log:
        code = subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    record.update(exit_code=code, log_sha256=sha(folder / 'build.log'))
    try:
        closure(folder, prep)
        need(code == 0, 'Build failed; preserved build.log contains diagnostics')
        emitted = []
        for line in (folder / 'build.log').read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('reason') == 'compiler-artifact' and item.get('target', {}).get('name') == NAME and item.get('executable'):
                emitted.append(item)
        need(len(emitted) == 1 and emitted[0]['fresh'] is False, 'Expected one freshly compiled probe')
        item = emitted[0]
        need(Path(item['manifest_path']).resolve() == project / 'Cargo.toml'
             and Path(item['target']['src_path']).resolve() == project / 'main.rs', 'Wrong Cargo target source')
        binary = Path(item['executable']).resolve()
        need(binary.is_relative_to(target) and binary.is_file(), 'Wrong emitted executable')
        shutil.copyfile(binary, folder / (NAME + '.exe'))
        metadata = subprocess.check_output(['cargo', 'metadata', '--locked', '--offline', '--format-version', '1'],
                                           cwd=project, env=env)
        with (folder / 'cargo-metadata.json').open('xb') as stream:
            stream.write(metadata)
        closure(folder, prep)
        record.update(status='built_not_executed', executable=NAME + '.exe', binary_sha256=sha(binary),
                      copied_binary_sha256=sha(folder / (NAME + '.exe')), cargo_emitted_artifact=item,
                      metadata_sha256=sha(folder / 'cargo-metadata.json'), source_closure=True)
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        write(folder / 'build.json', record)
    print(json.dumps({'status': record['status'], 'build_sha256': sha(folder / 'build.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('build')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--preparation-sha256', required=True)
    p.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else build)(args)
