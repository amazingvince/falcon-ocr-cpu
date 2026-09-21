"""Build/run one general width-specific RMS diagnostic in an isolated source copy.

Changes two diagnostic-only lines in that copy, never the live production tree.
Keeps Q/K width64 normalization unchanged; tests CUDA-shaped + rounded rsqrt only
for width768. Preserves production controls before/after and frozen tolerances.
"""
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


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(sys.platform == "win32", "this capture invokes the native Windows build wrapper")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    project = out / "project"; project.mkdir()
    paths = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    paths += sorted((ROOT / "src").glob("*.rs"))
    paths += sorted((ROOT / "examples/support").glob("*.rs"))
    paths += sorted((ROOT / "tests/fixtures").glob("*.json"))
    controls = [ROOT / "scripts/build_windows.ps1", ROOT / "scripts/compare_traces.py", Path(__file__).resolve()]
    inputs = {
        ROOT / "artifacts/reference/smoke-fp32/trace.safetensors": "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4",
        ROOT / "reference/tolerances-smoke-fp32-v1.json": "8f6382a4386b15f76e6984679208a44d0fa644b9d55a21ce579bbb26e0bfbe1c",
        ROOT / "artifacts/model/model.safetensors": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16",
    }
    original = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths + controls}
    for pth, digest in inputs.items():
        require(sha(pth) == digest, "fixed diagnostic input changed: " + str(pth))
    for path in paths:
        dest = project / path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    replacements = {
        "src/numerical_diagnostics.rs": ("if variant == 0 || !matches!(width, 64 | 768) {", "if variant == 0 || width != 768 {"),
        "src/model_diagnostics.rs": ('trace_numerical_interventions(&["production", "cuda_rms", "cuda_rms_rounded_rsqrt"])',
                                     'trace_numerical_interventions(&["production", "cuda_rms_rounded_rsqrt", "production_after"])'),
    }
    for name, (old, new) in replacements.items():
        path = project / name; data = path.read_text(encoding="utf-8")
        require(data.count(old) == 1, "diagnostic patch anchor must be unique")
        path.write_text(data.replace(old, new), encoding="utf-8", newline="\n")
    copied = {p.relative_to(project).as_posix(): sha(p) for p in sorted(project.rglob("*")) if p.is_file()}
    require({name for name, digest in copied.items() if digest != original[name]} == set(replacements), "unexpected source-copy change")
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in copied:
            archive.write(project / name, name)
        for control in controls:
            archive.write(control, "capture/" + control.relative_to(ROOT).as_posix())
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(ROOT / "target")
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/build_windows.ps1"),
               "test", "--locked", "--release", "--no-run", "--lib", "--jobs", "2", "--message-format=json-render-diagnostics",
               "--manifest-path", str(project / "Cargo.toml")]
    build = {"scope": __doc__, "original_source_sha256": original, "isolated_source_sha256": copied,
             "patches": replacements, "source_archive_sha256": sha(out / "source.zip"),
             "input_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
             "command": command, "rustc_version": subprocess.check_output(["rustc", "-Vv"], env=env, text=True),
             "cargo_version": subprocess.check_output(["cargo", "-V"], env=env, text=True),
             "environment": {k: env[k] for k in ("CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "ASM_NASM") if k in env},
             "limitations": "Captured source/build, not hermetic compiler/dependency objects; concurrent diagnostic, no timings."}
    write(out / "build-start.json", build)
    with (out / "build.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    require(code == 0, "diagnostic build failed; preserve build.log")
    executables = []
    for line in (out / "build.log").read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("reason") == "compiler-artifact" and item.get("executable") and item.get("profile", {}).get("test"):
            executables.append(Path(item["executable"]))
    require(len(executables) == 1, "ambiguous test executable")
    binary = out / "rms-width768-tests.exe"; shutil.copyfile(executables[0], binary)
    require(all(sha(ROOT / n) == v for n, v in original.items()), "live source changed during build")
    require(all(sha(project / n) == v for n, v in copied.items()), "isolated source changed during build")
    build.update(binary=str(binary), binary_sha256=sha(binary), build_exit_code=code, live_source_unchanged=True)
    write(out / "build.json", build)
    traces = out / "traces"; traces.mkdir()
    env["FOCR_DIAGNOSTIC_OUTPUT_DIR"] = str(traces)
    run_command = [str(binary), "model::diagnostics::trace_cuda_rms_interventions", "--exact", "--ignored", "--nocapture"]
    # Discover the actual module-qualified name rather than silently running zero tests.
    listing = subprocess.check_output([str(binary), "--list"], env=env, cwd=ROOT, text=True)
    selected = [line[:-6] for line in listing.splitlines() if line.endswith("trace_cuda_rms_interventions: test")]
    require(len(selected) == 1, "diagnostic test was not uniquely discovered")
    run_command[1] = selected[0]
    with (out / "run.log").open("x", encoding="utf-8") as log:
        run_code = subprocess.run(run_command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    require(run_code == 0, "diagnostic execution failed; preserve run.log")
    receipt = json.loads((traces / "interventions.json").read_text(encoding="utf-8"))
    require(receipt["test_binary_sha256"] == sha(binary), "wrong diagnostic executable")
    require(len(receipt["interventions"]) == 3, "missing intervention or control")
    trace_hashes = {name: sha(traces / (name + ".safetensors")) for name in ("production", "cuda_rms_rounded_rsqrt", "production_after")}
    require(trace_hashes["production"] == trace_hashes["production_after"] == "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309", "production controls changed")
    require(all(sha(ROOT / n) == v for n, v in original.items()), "live source changed during execution")
    require(all(sha(project / n) == v for n, v in copied.items()), "isolated source changed during execution")
    require(all(sha(p) == digest for p, digest in inputs.items()), "fixed input changed during execution")
    write(out / "execution.json", {"status": "diagnostic_complete", "command": run_command, "exit_code": run_code,
          "build_manifest_sha256": sha(out / "build.json"), "trace_sha256": trace_hashes,
          "interventions_sha256": sha(traces / "interventions.json"), "run_log_sha256": sha(out / "run.log"),
          "source_and_inputs_unchanged": True, "frozen_policy_comparison": "pending; controls alone do not qualify candidate"})
    print(json.dumps({"status": "diagnostic_complete", "trace_sha256": trace_hashes}))


if __name__ == "__main__":
    main()
