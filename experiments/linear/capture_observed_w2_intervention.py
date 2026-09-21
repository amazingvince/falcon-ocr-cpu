#!/usr/bin/env python3
"""One isolated W2 full-graph diagnostic; never edits the live production tree."""
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
PROOF = ROOT / "reference/w2-observed-partitions-rust-v1.json"
PROOF_SHA = "dd16ea379c9f01b3065e92a5f3d779ad30790e18af2209a11761e8c71c2dc229"
PROBE = ROOT / "experiments/linear/observed_partition_probe/src/main.rs"
PROBE_SHA = "8207d535f073a188c4aaa65b3913154fe5b317bcd12aa0279f1976fcaf628905"
CONTROL_SHA = "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validate_proof(proof):
    require(proof["status"] == "observed_operator_outputs_exact_both_platforms", "Operator prerequisite failed")
    require(proof["artifact_closure_unchanged"] and proof["independent_raw_partial_and_fold_comparison"]
            and proof["all_three_output_files_cross_platform_byte_exact"], "Operator prerequisite incomplete")
    require(proof["fixed_k_ranges"] == [[165*s, min(165*(s+1), 2304)] for s in range(14)], "Observed ranges differ")
    require(set(proof["platforms"]) == {"windows", "linux"}, "Missing platform proof")
    for platform in proof["platforms"].values():
        require(platform["tests_passed"] == 3 and platform["runtime_threads"] == 4, "Operator execution differs")
        for key, count in [("all_partials_vs_gpu", 1548288), ("ascending_fold_vs_gpu_output", 110592)]:
            row = platform[key]
            require(row["elements"] == count and row["bit_mismatches"] == row["nonfinite_pairs"] == 0
                    and row["max_abs_error"] == row["rms_error"] == 0, "Operator outputs not fully exact")


def module_source(probe_source):
    # Extract unchanged function text from the exact successful isolated probe.
    begin, end = probe_source.index("fn gather("), probe_source.index("fn compare(")
    arithmetic = probe_source[begin:end]
    require(arithmetic.count("fn gather(") == arithmetic.count("fn linear(") == 1, "Ambiguous arithmetic extraction")
    prefix = '''// Test-only copied arithmetic; fixed observed W2 partition diagnostic.
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
thread_local! { static ENABLED: Cell<bool> = const { Cell::new(false) }; }
static CALLS: AtomicUsize = AtomicUsize::new(0);
pub(crate) fn set_enabled(enabled: bool) { ENABLED.set(enabled); }
pub(crate) fn reset_calls() { CALLS.store(0, Ordering::SeqCst); }
pub(crate) fn calls() -> usize { CALLS.load(Ordering::SeqCst) }

'''
    suffix = '''
pub(crate) fn apply(input: &[f32], rows: usize, width: usize, weight: &[f32], channels: usize, output: &mut [f32]) -> bool {
    if !ENABLED.get() || rows != 144 || width != 2304 || channels != 768 { return false; }
    assert_eq!(input.len(), 144 * 2304);
    assert_eq!(weight.len(), 768 * 2304);
    assert_eq!(output.len(), 144 * 768);
    assert_eq!(rayon::current_num_threads(), 4);
    let mut partials = vec![0.0_f32; 14 * 144 * 768];
    for slot in 0..14 {
        let start = slot * 165;
        let end = ((slot + 1) * 165).min(2304);
        let x = gather(input, 144, 2304, start, end);
        let w = gather(weight, 768, 2304, start, end);
        linear(&x, 144, end-start, &w, 768, &mut partials[slot*144*768..(slot+1)*144*768]);
    }
    output.fill(0.0_f32);
    for part in partials.chunks_exact(144 * 768) {
        for (dst, &value) in output.iter_mut().zip(part) { *dst += value; }
    }
    CALLS.fetch_add(1, Ordering::SeqCst);
    true
}
'''
    return prefix + arithmetic + suffix, hashlib.sha256(arithmetic.encode()).hexdigest()


def patch_copy(project, probe_source):
    module, arithmetic_sha = module_source(probe_source)
    patches = {
        "src/numerical_diagnostics.rs": [
            ("    // The pinned architecture has only W2 at [N=768,K=2304]. Restrict to",
             "    if observed_w2::apply(input, rows, width, weight, channels, output) { return true; }\n    // The pinned architecture has only W2 at [N=768,K=2304]. Restrict to"),
        ],
        "src/model_diagnostics.rs": [
            ("fn trace_cuda_rms_interventions() -> Result<()> {", "fn trace_observed_w2_intervention() -> Result<()> {"),
            ('trace_numerical_interventions(&["production", "cuda_rms", "cuda_rms_rounded_rsqrt"])',
             'trace_numerical_interventions(&["production", "observed_split_w2", "production_after"])'),
            ('ensure!(!teachers.is_empty(), "teacher tokens are required");',
             'ensure!(tokens.len() == 144 && teachers.len() == 17, "fixed144/17 diagnostic scope");'),
            ("        let pool = rayon::ThreadPoolBuilder::new()\n            .num_threads(4)\n            .start_handler(move |_| {",
             '        let observed_w2_enabled = intervention == "observed_split_w2";\n        crate::numerical_diagnostics::reset_observed_w2_calls();\n        let pool = rayon::ThreadPoolBuilder::new()\n            .num_threads(4)\n            .start_handler(move |_| {\n                crate::numerical_diagnostics::set_thread_observed_w2(observed_w2_enabled);'),
            ('            let path = output.join(format!("{intervention}.safetensors"));',
             '            let observed_w2_calls = crate::numerical_diagnostics::observed_w2_calls();\n            ensure!(observed_w2_calls == if intervention == "observed_split_w2" { 22 } else { 0 }, "unexpected W2 intervention call count");\n            let path = output.join(format!("{intervention}.safetensors"));'),
            ('"argmax_matches_teacher":chosen == teachers,"teacher_forced":true}),',
             '"argmax_matches_teacher":chosen == teachers,"teacher_forced":true,"observed_w2_calls":observed_w2_calls}),'),
        ],
    }
    for name, changes in patches.items():
        path = project / name
        text = path.read_text(encoding="utf-8")
        for old, new in changes:
            require(text.count(old) == 1, "Patch anchor is not unique: " + old)
            text = text.replace(old, new)
        if name == "src/numerical_diagnostics.rs":
            text += '''
#[path = "observed_w2.rs"]
mod observed_w2;
pub(crate) fn set_thread_observed_w2(enabled: bool) { observed_w2::set_enabled(enabled); }
pub(crate) fn reset_observed_w2_calls() { observed_w2::reset_calls(); }
pub(crate) fn observed_w2_calls() -> usize { observed_w2::calls() }
'''
        path.write_text(text, encoding="utf-8", newline="\n")
    (project / "src/observed_w2.rs").write_text(module, encoding="utf-8", newline="\n")
    return patches, arithmetic_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(sys.platform == "win32", "This bounded capture uses the existing native Windows build wrapper")
    require(sha(PROOF) == PROOF_SHA and sha(PROBE) == PROBE_SHA, "Fixed prerequisite/source changed")
    proof = json.loads(PROOF.read_bytes())
    validate_proof(proof)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    project.mkdir()
    paths = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    paths += sorted((ROOT / "src").glob("*.rs")) + sorted((ROOT / "examples/support").glob("*.rs")) + sorted((ROOT / "tests/fixtures").glob("*.json"))
    controls = [Path(__file__).resolve(), PROBE, ROOT / "experiments/linear/test_observed_w2_intervention.py",
                ROOT / "scripts/build_windows.ps1", ROOT / "scripts/compare_traces.py"]
    original = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths + controls}
    require(original["src/kernels.rs"] == proof["production_kernels_sha256"], "Production kernel source changed")
    inputs = {ROOT / name.replace("\\", "/"): digest for name, digest in proof["evidence_sha256"].items()}
    inputs.update({PROOF: PROOF_SHA,
        ROOT / "artifacts/reference/smoke-fp32/trace.safetensors": "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4",
        ROOT / "reference/tolerances-smoke-fp32-v1.json": "8f6382a4386b15f76e6984679208a44d0fa644b9d55a21ce579bbb26e0bfbe1c",
        ROOT / "artifacts/model/model.safetensors": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"})
    for path, digest in inputs.items():
        require(sha(path) == digest, "Prerequisite artifact/input changed: " + str(path))
    for path in paths:
        destination = project / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    patches, arithmetic_sha = patch_copy(project, PROBE.read_text(encoding="utf-8"))
    copied = {p.relative_to(project).as_posix(): sha(p) for p in sorted(project.rglob("*")) if p.is_file()}
    require({n for n, h in copied.items() if n not in original or h != original[n]} == set(patches) | {"src/observed_w2.rs"}, "Unexpected isolated change")
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in copied:
            archive.write(project / name, name)
        for path in controls:
            archive.write(path, "capture/" + path.relative_to(ROOT).as_posix())
    build = {"status": "prepared", "scope": __doc__, "original_source_sha256": original,
             "isolated_source_sha256": copied, "patches": patches, "copied_gather_gemm_text_sha256": arithmetic_sha,
             "successful_probe_source_sha256": PROBE_SHA, "operator_receipt_sha256": PROOF_SHA,
             "input_sha256": {str(p): h for p, h in inputs.items()}, "source_archive_sha256": sha(output / "source.zip"),
             "limits": "One fixed W2 prefill intervention. No RMS combination, tolerance changes, shape generalization, algorithm/order sweep, timing or production change."}
    write(output / "preparation.json", build)
    if args.prepare_only:
        print(json.dumps({"status": "prepared_only", "project": str(project)}))
        return
    environment = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target"))
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/build_windows.ps1"),
               "test", "--locked", "--release", "--no-run", "--lib", "--jobs", "2", "--message-format=json-render-diagnostics",
               "--manifest-path", str(project / "Cargo.toml")]
    whitelist = ["CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS",
                 "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ASM_NASM", "CMAKE_GENERATOR", "FOCR_TOOL_DIR"]
    build.update(command=command, rustc_version=subprocess.check_output(["rustc", "-Vv"], text=True, env=environment),
                 cargo_version=subprocess.check_output(["cargo", "-V"], text=True, env=environment),
                 environment={key: environment[key] for key in whitelist if key in environment})
    write(output / "build-start.json", build)
    with (output / "build.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    require(code == 0, "Build failed; preserved build.log")
    executables = set()
    for line in (output / "build.log").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("reason") == "compiler-artifact" and row.get("executable") and row.get("profile", {}).get("test"):
            executables.add(row["executable"])
    require(len(executables) == 1, "Ambiguous test executable")
    binary = output / "observed-w2-tests.exe"
    shutil.copyfile(executables.pop(), binary)

    def unchanged():
        require(all(sha(ROOT / n) == h for n, h in original.items()), "Live source changed")
        require(all(sha(project / n) == h for n, h in copied.items()), "Copied source changed")
        require(all(sha(p) == h for p, h in inputs.items()), "Pinned prerequisite/input changed")
        require(sha(output / "source.zip") == build["source_archive_sha256"], "Source archive changed")

    unchanged()
    build.update(status="built", binary=str(binary), binary_sha256=sha(binary), build_log_sha256=sha(output / "build.log"))
    write(output / "build.json", build)
    traces = output / "traces"
    traces.mkdir()
    environment["FOCR_DIAGNOSTIC_OUTPUT_DIR"] = str(traces)
    listing = subprocess.check_output([str(binary), "--list"], cwd=ROOT, env=environment, text=True)
    selected = [line[:-6] for line in listing.splitlines() if line.endswith("trace_observed_w2_intervention: test")]
    require(len(selected) == 1, "Missing/ambiguous isolated test")
    run_command = [str(binary), selected[0], "--exact", "--ignored", "--nocapture"]
    with (output / "run.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(run_command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    require(code == 0, "Trace execution failed; preserved run.log")
    receipt = json.loads((traces / "interventions.json").read_bytes())
    require(receipt["test_binary_sha256"] == build["binary_sha256"] == sha(binary), "Wrong or changed executable")
    rows = receipt["interventions"]
    names = ["production", "observed_split_w2", "production_after"]
    require([r["intervention"] for r in rows] == names, "Missing/interchanged intervention")
    require([r["observed_w2_calls"] for r in rows] == [0,22,0], "Wrong hook activation count")
    require(all(r["tensors"] == 1904 and len(r["same_prefix_argmax"]) == 17 and r["teacher_forced"] for r in rows), "Trace protocol differs")
    trace_hashes = {name: sha(traces / (name + ".safetensors")) for name in names}
    require(trace_hashes["production"] == trace_hashes["production_after"] == CONTROL_SHA, "Production controls changed")
    unchanged()
    write(output / "execution.json", {"status":"diagnostic_complete", "command":run_command, "exit_code":code,
          "build_manifest_sha256":sha(output / "build.json"), "trace_sha256":trace_hashes,
          "interventions_sha256":sha(traces / "interventions.json"), "run_log_sha256":sha(output / "run.log"),
          "source_and_inputs_unchanged":True,"hook_counts":[0,22,0],
          "frozen_policy_comparison":"pending original1904-tensor policy; controls and operator evidence do not qualify the graph"})
    print(json.dumps({"status":"diagnostic_complete","trace_sha256":trace_hashes}))


if __name__ == "__main__":
    main()
