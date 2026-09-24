"""Isolated width768 RMS intervention using an exhaustively checked GPU rsqrt table.

This is a diagnostic only. Live production sources, defaults and tolerances are
unchanged. Width64 Q/K normalization stays on the production implementation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from capture_width768_intervention import ROOT, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-sha256", required=True)
    args = parser.parse_args()
    require(sys.platform == "win32", "uses the native Windows build wrapper")
    table, validation = args.table.resolve(), args.validation.resolve()
    require(table.stat().st_size == (1 << 24) * 4, "canonical table must contain 2^24 F32 values")
    require(sha(table) == args.table_sha256, "rsqrt table hash changed")
    require(sha(validation) == args.validation_sha256, "rsqrt validation hash changed")
    proof = json.loads(validation.read_text(encoding="utf-8"))
    require(proof["status"] == "passed_exact", "rsqrt table did not pass exhaustive validation")
    require(proof["total_checked"] == 151 * (1 << 23), "incomplete rsqrt argument coverage")
    require(proof["total_mismatches"] == proof["total_bit_mapping_mismatches"] == 0,
            "rsqrt output or exponent mapping differs")
    require(proof["table"]["input_begin_bits"] == 0x3f800000
            and proof["table"]["input_end_bits_exclusive"] == 0x40800000
            and proof["table"]["entries"] == 1 << 24, "wrong canonical table domain")
    require(proof["artifacts"]["rsqrt-table.f32le"]["sha256"] == args.table_sha256,
            "validation describes another table")
    require(proof["source_after_sha256"] == proof["source_sha256"], "export sources changed")
    records = proof["records"]
    require(len(records) == 151 and [row["biased_exponent"] for row in records] == list(range(104, 255)),
            "exhaustive exponent inventory differs")
    require(all(row["checked"] == 1 << 23 and row["mismatches"] == row["bit_mapping_mismatches"] == 0
                for row in records), "an exponent has incomplete or failed validation")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    project.mkdir()
    paths = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    paths += sorted((ROOT / "src").glob("*.rs"))
    paths += sorted((ROOT / "examples/support").glob("*.rs"))
    paths += sorted((ROOT / "tests/fixtures").glob("*.json"))
    lookup = ROOT / "research/gpu-reference/experiments/rms_norm/rsqrt_lookup.rs"
    require(sha(lookup) == proof["source_sha256"]["research/gpu-reference/experiments/rms_norm/rsqrt_lookup.rs"],
            "lookup source differs from the exhaustively validated capture")
    controls = [ROOT / "tools/build_windows.ps1", ROOT / "research/gpu-reference/scripts/compare_traces.py",
                Path(__file__).resolve(), ROOT / "research/gpu-reference/experiments/rms_norm/capture_width768_intervention.py", lookup]
    original = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths + controls}
    inputs = {
        ROOT / "artifacts/reference/smoke-fp32/trace.safetensors": "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4",
        ROOT / "reference/tolerances-smoke-fp32-v1.json": "8f6382a4386b15f76e6984679208a44d0fa644b9d55a21ce579bbb26e0bfbe1c",
        ROOT / "artifacts/model/model.safetensors": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16",
        table: args.table_sha256,
        validation: args.validation_sha256,
    }
    for path, digest in inputs.items():
        require(sha(path) == digest, "fixed input changed: " + str(path))
    for path in paths:
        destination = project / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(lookup, project / "src/rsqrt_lookup.rs")

    patches = {
        "src/numerical_diagnostics.rs": [
            ("if variant == 0 || !matches!(width, 64 | 768) {", "if variant == 0 || width != 768 {"),
            ("(1.0 / (variance as f64).sqrt()) as f32", "rsqrt_lookup::rsqrt_from_table(variance, &RMS_RSQRT_TABLE)"),
        ],
        "src/model_diagnostics.rs": [
            ('trace_numerical_interventions(&["production", "cuda_rms", "cuda_rms_rounded_rsqrt"])',
             'trace_numerical_interventions(&["production", "cuda_rms_gpu_rsqrt", "production_after"])'),
            ('"cuda_rms_rounded_rsqrt" => 2,', '"cuda_rms_gpu_rsqrt" => 2,'),
        ],
    }
    loader = r'''

#[path = "rsqrt_lookup.rs"]
mod rsqrt_lookup;
static RMS_RSQRT_TABLE: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| {
    use sha2::{Digest, Sha256};
    let path = std::env::var("FOCR_RSQRT_TABLE").expect("diagnostic table path");
    let expected = std::env::var("FOCR_RSQRT_TABLE_SHA256").expect("diagnostic table SHA256");
    let bytes = std::fs::read(path).expect("read diagnostic table");
    assert_eq!(bytes.len(), (1usize << 24) * 4);
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
    bytes.chunks_exact(4).map(|part| f32::from_le_bytes(part.try_into().unwrap())).collect()
});
'''
    for name, replacements in patches.items():
        path = project / name
        data = path.read_text(encoding="utf-8")
        for old, new in replacements:
            require(data.count(old) == 1, "patch anchor must be unique: " + old)
            data = data.replace(old, new)
        if name == "src/numerical_diagnostics.rs":
            data += loader
        path.write_text(data, encoding="utf-8", newline="\n")
    copied = {path.relative_to(project).as_posix(): sha(path) for path in sorted(project.rglob("*")) if path.is_file()}
    require({name for name, digest in copied.items() if name not in original or digest != original[name]}
            == set(patches) | {"src/rsqrt_lookup.rs"}, "unexpected isolated source change")
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in copied:
            archive.write(project / name, name)
        for control in controls:
            archive.write(control, "capture/" + control.relative_to(ROOT).as_posix())
    shutil.copyfile(validation, output / "table-validation.json")

    environment = dict(os.environ)
    environment["CARGO_TARGET_DIR"] = str(ROOT / "target")
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools/build_windows.ps1"),
               "test", "--locked", "--release", "--no-run", "--lib", "--jobs", "2", "--message-format=json-render-diagnostics",
               "--manifest-path", str(project / "Cargo.toml")]
    build = {"scope": __doc__, "original_source_sha256": original, "isolated_source_sha256": copied,
             "patches": patches, "loader_source": loader, "source_archive_sha256": sha(output / "source.zip"),
             "input_sha256": {str(path): digest for path, digest in inputs.items()}, "command": command,
             "rustc_version": subprocess.check_output(["rustc", "-Vv"], env=environment, text=True),
             "cargo_version": subprocess.check_output(["cargo", "-V"], env=environment, text=True),
             "environment": {key: environment[key] for key in ("CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "ASM_NASM") if key in environment},
             "limitations": "Nonhermetic diagnostic build under concurrent load; no performance claim or production table design."}
    write(output / "build-start.json", build)
    with (output / "build.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    require(code == 0, "build failed; preserve build.log")
    executables = []
    for line in (output / "build.log").read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("reason") == "compiler-artifact" and item.get("executable") and item.get("profile", {}).get("test"):
            executables.append(Path(item["executable"]))
    require(len(executables) == 1, "ambiguous test executable")
    binary = output / "rms-gpu-rsqrt-tests.exe"
    shutil.copyfile(executables[0], binary)
    require(all(sha(ROOT / name) == digest for name, digest in original.items()), "live source changed during build")
    require(all(sha(project / name) == digest for name, digest in copied.items()), "isolated source changed during build")
    build.update(binary=str(binary), binary_sha256=sha(binary), build_exit_code=code, live_source_unchanged=True)
    write(output / "build.json", build)
    traces = output / "traces"
    traces.mkdir()
    environment.update(FOCR_DIAGNOSTIC_OUTPUT_DIR=str(traces), FOCR_RSQRT_TABLE=str(table), FOCR_RSQRT_TABLE_SHA256=args.table_sha256)
    listing = subprocess.check_output([str(binary), "--list"], cwd=ROOT, env=environment, text=True)
    selected = [line[:-6] for line in listing.splitlines() if line.endswith("trace_cuda_rms_interventions: test")]
    require(len(selected) == 1, "diagnostic test was not uniquely discovered")
    run_command = [str(binary), selected[0], "--exact", "--ignored", "--nocapture"]
    with (output / "run.log").open("x", encoding="utf-8") as log:
        run_code = subprocess.run(run_command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    require(run_code == 0, "execution failed; preserve run.log")
    receipt = json.loads((traces / "interventions.json").read_text(encoding="utf-8"))
    require(receipt["test_binary_sha256"] == sha(binary), "wrong diagnostic executable")
    require(len(receipt["interventions"]) == 3, "missing intervention or control")
    trace_hashes = {name: sha(traces / (name + ".safetensors")) for name in ("production", "cuda_rms_gpu_rsqrt", "production_after")}
    require(trace_hashes["production"] == trace_hashes["production_after"] == "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309", "production controls changed")
    require(all(sha(ROOT / name) == digest for name, digest in original.items()), "live source changed during execution")
    require(all(sha(project / name) == digest for name, digest in copied.items()), "isolated source changed during execution")
    require(all(sha(path) == digest for path, digest in inputs.items()), "fixed input changed during execution")
    write(output / "execution.json", {"status": "diagnostic_complete", "command": run_command, "exit_code": run_code,
          "environment": {key: environment[key] for key in ("FOCR_DIAGNOSTIC_OUTPUT_DIR", "FOCR_RSQRT_TABLE", "FOCR_RSQRT_TABLE_SHA256")},
          "build_manifest_sha256": sha(output / "build.json"), "trace_sha256": trace_hashes,
          "interventions_sha256": sha(traces / "interventions.json"), "run_log_sha256": sha(output / "run.log"),
          "source_and_inputs_unchanged": True, "frozen_policy_comparison": "pending; controls alone do not qualify candidate"})
    print(json.dumps({"status": "diagnostic_complete", "trace_sha256": trace_hashes}))


if __name__ == "__main__":
    main()
