#!/usr/bin/env python3
"""Build/test and preserve only the fixed observed-W2 Rust operator diagnostic."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/linear/observed_partition_probe"
KERNELS_SHA = "19f1f18e164fffcab63dd1d747aae76f3a4a75dfcc15edebacd46b5dc32f7f16"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def locked_registry(path):
    result = {}
    for block in path.read_text().split("[[package]]")[1:]:
        fields = {k: re.search(r'^' + k + r' = "([^"]+)"', block, re.M) for k in ("name", "version", "source", "checksum")}
        if fields["source"]:
            result[(fields["name"][1], fields["version"][1])] = (fields["source"][1], fields["checksum"][1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    require(1 <= args.jobs <= 4, "Build jobs must be1..4")
    names = ["Cargo.lock", "rust-toolchain.toml", "src/kernels.rs",
             "experiments/linear/observed_partition_probe/Cargo.toml",
             "experiments/linear/observed_partition_probe/Cargo.lock",
             "experiments/linear/observed_partition_probe/src/main.rs",
             "experiments/linear/capture_observed_partition_probe.py"]
    source = {name: (ROOT / name).read_bytes() for name in names}
    require(sha(source["src/kernels.rs"]) == KERNELS_SHA, "Production comparison source changed")
    baseline, isolated = locked_registry(ROOT / "Cargo.lock"), locked_registry(PACKAGE / "Cargo.lock")
    require(all(baseline.get(k) == v for k, v in isolated.items()), "Isolated dependency differs from existing production lock")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "source.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in source.items():
            archive.writestr(name, data)
    target = (args.target_dir or output / "target").resolve()
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    common = ["--manifest-path", str(PACKAGE / "Cargo.toml"), "--locked", "--offline", "--release", "--jobs", str(args.jobs)]
    build = ["cargo", "build", *common, "--message-format=json-render-diagnostics"]
    tests = ["cargo", "test", *common, "--", "--test-threads=1"]
    whitelist = ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS", "CARGO_TARGET_DIR",
                 "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ASM_NASM", "CMAKE_GENERATOR", "FOCR_TOOL_DIR"]
    record = {"schema_version": 1, "status": "building", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.executable, "cwd": str(ROOT), "commands": [build, tests],
              "environment_overrides": {k: env[k] for k in whitelist if k in env},
              "rustc_version": subprocess.check_output(["rustc", "-Vv"], cwd=ROOT, env=env, text=True),
              "cargo_version": subprocess.check_output(["cargo", "-V"], cwd=ROOT, env=env, text=True),
              "source_sha256": {name: sha(data) for name, data in source.items()}, "source_archive_sha256": sha(archive_path.read_bytes()),
              "registry_dependencies_match_production_lock": len(isolated),
              "limits": "One functional operator; no timings. Source/lock/tool flags are preserved, not a hermetic toolchain image. Cargo may reuse cached registry dependencies."}
    manifest = output / "build.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    codes = []
    for command, name in [(build, "build.log"), (tests, "tests.log")]:
        with (output / name).open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        codes.append(result.returncode)
        if result.returncode:
            break
    unchanged = all((ROOT / name).read_bytes() == data for name, data in source.items())
    record.update(exit_codes=codes, source_unchanged=unchanged, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    record["status"] = "failed"
    if codes == [0, 0] and unchanged:
        emitted = set()
        for line in (output / "build.log").read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("reason") == "compiler-artifact" and item.get("target", {}).get("name") == "observed-w2-partitions" and item.get("executable"):
                emitted.add(item["executable"])
        require(len(emitted) == 1, "Expected exactly one Cargo-emitted executable")
        binary = Path(emitted.pop())
        preserved = output / binary.name
        shutil.copy2(binary, preserved)
        record.update(status="complete", cargo_emitted_executable=str(binary), binary=preserved.name,
                      binary_sha256=sha(preserved.read_bytes()), binary_bytes=preserved.stat().st_size)
    record["logs"] = {p.name: sha(p.read_bytes()) for p in output.glob("*.log")}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "build_manifest": str(manifest), "binary": record.get("binary")}, indent=2))
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
