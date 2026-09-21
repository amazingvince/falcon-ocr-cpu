#!/usr/bin/env python3
"""Capture the existing CLI for functional batch checks; never run inference."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import capture_rust_build as capture

ROOT = capture.ROOT
PROTOCOL_FILES = ["scripts/capture_functional_cli.py", "scripts/functional_batch_regression.py",
                  "scripts/corpus_comparison.py", "scripts/validate_text_replay.py", "scripts/fetch_reference.py"]


def source_paths():
    # Reuse the established complete Rust/source inventory. The benchmark
    # example itself is not compiled by this CLI build and is omitted.
    paths = [p for p in capture.source_paths("ocr_bench") if p != ROOT / "examples/ocr_bench.rs"]
    return sorted(set(paths + list((ROOT / "src").rglob("*.rs")) + [ROOT / p for p in PROTOCOL_FILES]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cargo-target-dir", type=pathlib.Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    windows = sys.platform == "win32"
    target = (args.cargo_target_dir or pathlib.Path(env.get("CARGO_TARGET_DIR", str(ROOT / ("target" if windows else "target/linux"))))).resolve()
    env["CARGO_TARGET_DIR"] = str(target)
    command = (["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/build_windows.ps1")]
               if windows else ["bash", str(ROOT / "scripts/build_linux.sh")])
    command += ["build", "--locked", "--release", "--message-format=json-render-diagnostics", "--bin", "falcon-ocr", "--jobs", str(args.jobs)]
    source = capture.capture_sources(source_paths())
    capture.write_archive(output / "source.zip", source)
    record = {"schema_version": 1, "target_kind": "bin", "target_name": "falcon-ocr", "status": "building",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command,
              "cwd": str(ROOT), "platform": sys.platform, "resolved_cargo_target_dir": str(target),
              "source_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in source.items()},
              "source_archive": "source.zip", "source_archive_sha256": capture.sha(output / "source.zip"),
              "rustc_version": subprocess.check_output(["rustc", "-Vv"], cwd=ROOT, env=env, text=True),
              "cargo_version": subprocess.check_output(["cargo", "-V"], cwd=ROOT, env=env, text=True),
              "environment_overrides": {k: env[k] for k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS", "CARGO_BUILD_JOBS", "CARGO_TARGET_DIR", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ASM_NASM", "CMAKE_GENERATOR", "FOCR_TOOL_DIR"] if k in env},
              "qualification": "Fresh source-bound CLI build for functional output checks. No inference or timing qualification.",
              "limitations": ["Not a hermetic compiler/native-dependency image; Cargo may reuse dependency objects."]}
    manifest = output / "build.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (output / "build.log").open("x", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    unchanged = source == capture.capture_sources(source_paths())
    record.update(build_exit_code=process.returncode, source_unchanged_during_build=unchanged,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if process.returncode or not unchanged:
        record["status"] = "failed_or_source_changed"
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("CLI build rejected; inspect build.json and build.log")
    binary = capture.emitted_executable((output / "build.log").read_text(encoding="utf-8"), "falcon-ocr")
    preserved = output / binary.name
    shutil.copy2(binary, preserved)
    record.update(status="complete", cargo_emitted_executable=str(binary), binary=preserved.name,
                  binary_sha256=capture.sha(preserved), binary_bytes=preserved.stat().st_size)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"build_manifest": str(manifest), "binary": str(preserved), "binary_sha256": record["binary_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
