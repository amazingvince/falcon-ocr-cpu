#!/usr/bin/env python3
"""Build and preserve a Rust example with a checked source archive and tool identity."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(example):
    paths = list((ROOT / "src").glob("*.rs"))
    paths += list((ROOT / "examples/support").rglob("*.rs"))
    paths += [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "rust-toolchain.toml",
              ROOT / "examples" / f"{example}.rs", pathlib.Path(__file__),
              ROOT / "scripts/build_windows.ps1", ROOT / "scripts/build_linux.sh"]
    config = ROOT / ".cargo/config.toml"
    if config.exists():
        paths.append(config)
    if example == "quant_probe":
        paths += list((ROOT / "experiments/quantization").glob("*.rs"))
        paths += list((ROOT / "experiments/quantization").glob("*.py"))
    return sorted(paths)


def capture_sources(paths):
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}


def write_archive(path, source):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in source.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)


def emitted_executable(log, example):
    found = set()
    for line in log.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(item, dict) and item.get("reason") == "compiler-artifact"
                and item.get("target", {}).get("name") == example and item.get("executable")):
            found.add(item["executable"])
    if len(found) != 1:
        raise ValueError(f"Expected one Cargo-emitted executable for {example}; found {sorted(found)}")
    return pathlib.Path(found.pop())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["corpus_eval", "ocr_bench", "redecode_corpus", "aocl_probe", "quant_probe"], required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="New directory; existing directories are never overwritten")
    parser.add_argument("--cargo-target-dir", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    windows = sys.platform == "win32"
    target = (args.cargo_target_dir or pathlib.Path(os.environ.get(
        "CARGO_TARGET_DIR", str(ROOT / ("target" if windows else "target/linux"))))).resolve()
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    cargo_args = ["build", "--locked", "--release", "--message-format=json-render-diagnostics",
                  "--example", args.example, "--jobs", str(args.jobs)]
    command = (["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(ROOT / "scripts/build_windows.ps1")] if windows else
               ["bash", str(ROOT / "scripts/build_linux.sh")]) + cargo_args
    source = capture_sources(source_paths(args.example))
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in source.items()}
    archive = output / "source.zip"
    write_archive(archive, source)
    whitelist = ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS",
                 "CARGO_TARGET_DIR", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ASM_NASM",
                 "CMAKE_GENERATOR", "FOCR_TOOL_DIR"]
    record = {"schema_version": 1, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "example": args.example, "command": command, "cwd": str(ROOT), "platform": sys.platform,
              "environment_overrides": {key: env[key] for key in whitelist if key in env},
              "source_sha256": hashes, "source_archive": archive.name, "source_archive_sha256": sha(archive),
              "rustc_version": subprocess.check_output(["rustc", "-Vv"], text=True, cwd=ROOT, env=env),
              "cargo_version": subprocess.check_output(["cargo", "-V"], text=True, cwd=ROOT, env=env),
              "status": "building",
              "limitations": ["Records explicitly selected build flags and installed tool versions; not a hermetic toolchain image.",
                              "Cargo may reuse cached dependency objects; native dependency build provenance also depends on Cargo's cache.",
                              "The executable hash is the exact resume identity even if rebuilding archived sources yields different bytes."]}
    record_path = output / "build.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (output / "build.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    after = capture_sources(source_paths(args.example))
    unchanged = source == after
    record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  build_exit_code=completed.returncode, source_unchanged_during_build=unchanged)
    if completed.returncode or not unchanged:
        record["status"] = "failed" if completed.returncode else "source_changed_during_build"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Build not accepted; inspect {record_path} and build.log")
    binary = emitted_executable((output / "build.log").read_text(encoding="utf-8"), args.example)
    preserved = output / binary.name
    shutil.copy2(binary, preserved)
    record.update(status="complete", cargo_emitted_executable=str(binary), binary=preserved.name, binary_sha256=sha(preserved),
                  binary_bytes=preserved.stat().st_size)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"build_manifest": str(record_path), "binary": str(preserved),
                      "binary_sha256": record["binary_sha256"], "source_archive_sha256": record["source_archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
