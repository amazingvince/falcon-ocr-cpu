"""Preserve a quant_probe build and optionally run its bounded scalar diagnostic."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("capture_rust_build", ROOT / "research/benchmarks/scripts/capture_rust_build.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run", action="store_true", help="Execute scalar diagnostic only, without timing")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(set(capture.source_paths("quant_probe") + list((ROOT / "experiments/quantization").glob("*.rs"))
                       + list((ROOT / "experiments/quantization").glob("*.py"))))
    source = capture.capture_sources(paths)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in source.items()}
    capture.write_archive(output / "source.zip", source)
    windows = sys.platform == "win32"
    command = (["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools/build_windows.ps1")]
               if windows else ["bash", str(ROOT / "tools/build_linux.sh")])
    command += ["build", "--locked", "--release", "--message-format=json-render-diagnostics", "--example", "quant_probe", "--jobs", "2"]
    env = dict(os.environ)
    record = {"schema_version": 1, "example": "quant_probe", "scope": "scalar arithmetic diagnostic, no timings", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": command, "source_sha256": hashes, "source_archive_sha256": capture.sha(output / "source.zip"),
              "rustc": subprocess.check_output(["rustc", "-Vv"], text=True), "cargo": subprocess.check_output(["cargo", "-V"], text=True),
              "environment_overrides": {key: env[key] for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR", "CC", "CXX", "CFLAGS", "CXXFLAGS") if key in env},
              "status": "building", "limitations": ["Source/binary snapshot; Cargo may reuse native dependency objects. Not a hermetic toolchain image."]}
    manifest = output / "build.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (output / "build.log").open("x", encoding="utf-8") as stream:
        done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    after_paths = sorted(set(capture.source_paths("quant_probe") + list((ROOT / "experiments/quantization").glob("*.rs"))
                             + list((ROOT / "experiments/quantization").glob("*.py"))))
    unchanged = source == capture.capture_sources(after_paths)
    record.update(build_exit_code=done.returncode, source_unchanged_during_build=unchanged,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if done.returncode or not unchanged:
        record["status"] = "failed_or_source_changed"
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("Build not accepted; inspect build log/manifest")
    binary = capture.emitted_executable((output / "build.log").read_text(encoding="utf-8"), "quant_probe")
    preserved = output / binary.name
    shutil.copy2(binary, preserved)
    record.update(status="complete", cargo_emitted_executable=str(binary), binary=preserved.name,
                  binary_sha256=capture.sha(preserved), binary_bytes=preserved.stat().st_size)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.run:
        result = output / "operators.json"
        with (output / "probe.log").open("x", encoding="utf-8") as log:
            subprocess.run([str(preserved), "--output", str(result)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (output / "checker.log").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(ROOT / "research/quantization-feasibility/experiments/quantization/check_probe.py"), str(result), "--output", str(output / "independent-check.json")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(manifest)


if __name__ == "__main__":
    main()
