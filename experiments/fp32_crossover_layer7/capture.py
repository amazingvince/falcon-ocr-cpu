#!/usr/bin/env python3
"""Prospective copy/build/run capture. Each phase requires explicit quiet release.

No phase is automatically followed by another. Never modifies the live project.
The plan is shared with the separately scheduled GPU arm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile

from adapter import make_adapter
from contract import (ROOT, KIND, PINS, METADATA, RUNTIME, STAGES, CONTROL_STAGES,
                      require, sha, read_bound, resolve, write_new, validate_plan)

TEST = "model::fp32_crossover_layer7::capture_cpu_state_crossover"
EXPERIMENT_SOURCES = ["contract.py", "adapter.py", "rust_arm.rs", "capture.py",
                      "compare.py", "test_source.py", "export_gpu.py", "README.md", "SOURCE_REVIEW_NOTES.md"]
GPU_SOURCES = ["scripts/export_reference.py", "scripts/reference_preflight.py",
               "scripts/fetch_reference.py", "requirements/reference-lock.txt", "reference/manifest.json",
               "artifacts/model/artifact-manifest.json", "artifacts/model/modeling_falcon_ocr.py",
               "artifacts/model/configuration_falcon_ocr.py", "artifacts/model/attention.py",
               "artifacts/model/rope.py", "artifacts/model/processing_falcon_ocr.py",
               "artifacts/model/config.json", "artifacts/model/tokenizer.json",
               "artifacts/model/tokenizer_config.json"]
ENV_KEYS = ["CARGO_TARGET_DIR", "CARGO_HOME", "RUSTUP_HOME", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS",
            "LDFLAGS", "ASM_NASM", "CMAKE_GENERATOR", "FOCR_TOOL_DIR", "CUDA_VISIBLE_DEVICES"]
MODULE = '\n#[cfg(test)]\n#[path = "fp32_crossover_layer7_arm.rs"]\nmod fp32_crossover_layer7;\n'


def source_paths():
    # This bounded inventory is evaluated only during authorized preparation.
    copied = [ROOT / n for n in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    copied += sorted((ROOT / "src").glob("*.rs"))
    copied += sorted((ROOT / "examples/support").glob("*.rs"))
    copied += sorted((ROOT / "tests/fixtures").glob("*.json"))
    extras = [ROOT / "scripts/build_windows.ps1"]
    extras += [ROOT / "experiments/fp32_crossover_layer7" / n for n in EXPERIMENT_SOURCES]
    extras += [ROOT / n for n in GPU_SOURCES]
    return copied, sorted(set(copied + extras))


def inside_workspace(path):
    path = path.resolve()
    require(path.is_relative_to(ROOT) and path != ROOT, "Expected a new path inside the project")
    return path


def inventory(project):
    return {p.relative_to(project).as_posix(): sha(p)
            for p in sorted(project.rglob("*")) if p.is_file()}


def unchanged(directory, prep, plan):
    validate_plan(plan)
    require(sha(directory / "plan.json") == prep["plan_sha256"], "Plan changed")
    require(inventory(directory / "project") == prep["isolated_source_sha256"], "Copied source changed")
    require(sha(directory / "source.zip") == prep["source_archive_sha256"], "Source archive changed")
    # New/deleted live source in the compilation closure must also invalidate it.
    _, current = source_paths()
    require({p.relative_to(ROOT).as_posix() for p in current} == set(plan["source_sha256"]),
            "Source inventory changed")


def prepare(args):
    directory = inside_workspace(args.output)
    require(not directory.exists(), "Preparation output already exists")
    copied, sources = source_paths()
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    inputs = {role: {"path": path, "sha256": digest} for role, (path, digest) in PINS.items()}
    inputs.update({role: {"path": path, "sha256": sha(resolve(path))} for role, path in METADATA.items()})
    plan = {"kind": KIND, "inputs": inputs, "source_sha256": source_hashes,
            "model_directory": "artifacts/model", "runtime": RUNTIME, "stages": STAGES,
            "control_stages": CONTROL_STAGES,
            "original_state_key": "layer.6.hidden", "original_state_shape": [144, 768],
            "fixed_endpoint": {"stage": "layer.7.hidden", "coordinate": [112, 249]},
            "branches": {"gpu": ["gpu_state", "cpu_state"], "rust": ["cpu_state"]},
            "historical_startup_attestation_complete": False,
            "scope": "One native layer7 crossover from saved layer6 states; five controls per original-state arm precede interpretation.",
            "limits": ["Current captures do not retroactively attest original source or build startup.",
                       "No full model execution, numerical policy change, inference qualification or performance claim.",
                       "GPU mask/cache capacity256 and original Rust capacity161 are retained separately."]}
    validate_plan(plan)
    directory.mkdir(parents=True)
    project = directory / "project"
    project.mkdir()
    for source in copied:
        target = project / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    original_model = (project / "src/model.rs").read_text(encoding="utf-8")
    adapter, original_forward = make_adapter(original_model)
    require("mod fp32_crossover_layer7;" not in original_model, "Diagnostic already installed")
    (project / "src/model.rs").write_text(original_model + MODULE, encoding="utf-8", newline="\n")
    shutil.copyfile(ROOT / "experiments/fp32_crossover_layer7/rust_arm.rs", project / "src/fp32_crossover_layer7_arm.rs")
    (project / "src/fp32_crossover_layer7_forward.rs").write_text(adapter, encoding="utf-8", newline="\n")
    isolated = inventory(project)
    changed = {n for n, h in isolated.items() if source_hashes.get(n) != h}
    require(changed == {"src/model.rs", "src/fp32_crossover_layer7_arm.rs", "src/fp32_crossover_layer7_forward.rs"},
            "Unexpected copied source change")
    with zipfile.ZipFile(directory / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in isolated:
            archive.write(project / name, "project/" + name)
        for source in sources:
            archive.write(source, "original/" + source.relative_to(ROOT).as_posix())
    write_new(directory / "plan.json", plan)
    prep = {"kind": "fp32-crossover-layer7-preparation-v1", "status": "prepared_not_executed",
            "plan_sha256": sha(directory / "plan.json"), "isolated_source_sha256": isolated,
            "source_archive_sha256": sha(directory / "source.zip"), "changed_files": sorted(changed),
            "original_forward_text_sha256": hashlib.sha256(original_forward.encode()).hexdigest(),
            "selected_test": TEST, "original_model_patch": MODULE,
            "execution_requires_separate_review_release": True}
    unchanged(directory, prep, plan)
    write_new(directory / "preparation.json", prep)
    print(json.dumps({"status": prep["status"], "preparation_sha256": sha(directory / "preparation.json"),
                      "plan_sha256": prep["plan_sha256"]}))


def load_prepared(args):
    directory = inside_workspace(args.prepared)
    prep = read_bound(directory / "preparation.json", args.preparation_sha256)
    require(prep["kind"] == "fp32-crossover-layer7-preparation-v1" and prep["selected_test"] == TEST,
            "Wrong preparation")
    plan = read_bound(directory / "plan.json", prep["plan_sha256"])
    unchanged(directory, prep, plan)
    return directory, prep, plan


def tool_environment(target):
    environment = dict(os.environ, CARGO_TARGET_DIR=str(target), CUDA_VISIBLE_DEVICES="-1")
    # Do not let the generic wrapper download or select unrecorded fallback tools.
    nasm = environment.get("ASM_NASM") or shutil.which("nasm")
    if not nasm:
        nasm = str(ROOT / "artifacts/tools/nasm-2.16.03/nasm.exe")
    cmake = shutil.which("cmake")
    if not cmake:
        cmake = str(ROOT / "artifacts/tools/cmake-3.31.10-windows-x86_64/bin/cmake.exe")
        environment["PATH"] = str(Path(cmake).parent) + os.pathsep + environment["PATH"]
    require(Path(nasm).is_file() and Path(cmake).is_file(), "Existing NASM/CMake tools required; no download")
    environment["ASM_NASM"] = nasm
    tools = {name: Path(path).resolve() for name, path in {"nasm": nasm, "cmake": cmake,
             "rustc": shutil.which("rustc"), "cargo": shutil.which("cargo")}.items() if path}
    require(set(tools) == {"nasm", "cmake", "rustc", "cargo"}, "Rust build tools missing")
    return environment, {name: {"path": str(path), "sha256": sha(path)} for name, path in tools.items()}


def selected_artifact(log, project, target):
    artifacts = []
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("reason") == "compiler-artifact" and row.get("executable") and row.get("profile", {}).get("test"):
            artifacts.append(row)
    require(len(artifacts) == 1, "Expected exactly one Cargo test executable")
    artifact = artifacts[0]
    require(artifact.get("fresh") is False, "Fresh isolated target must compile the diagnostic")
    require(Path(artifact["manifest_path"]).resolve() == project / "Cargo.toml"
            and Path(artifact["target"]["src_path"]).resolve() == project / "src/lib.rs"
            and artifact["target"]["kind"] == ["lib"], "Wrong Cargo source/artifact")
    executable = Path(artifact["executable"]).resolve()
    require(executable.is_relative_to(target), "Executable escaped fresh target directory")
    return artifact, executable


def build(args):
    require(sys.platform == "win32", "Original saved CPU control is native Windows")
    directory, prep, plan = load_prepared(args)
    require(not (directory / "build-start.json").exists(), "Build already attempted; preserve it")
    target = args.target_dir.resolve()
    require(not target.exists() and not target.is_relative_to(directory / "project")
            and target != ROOT and target != ROOT / "target", "Require a fresh dedicated Cargo target")
    target.parent.mkdir(parents=True, exist_ok=True)
    environment, tools = tool_environment(target)
    project = directory / "project"
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(ROOT / "scripts/build_windows.ps1"), "test", "--locked", "--offline", "--release",
               "--no-run", "--lib", "--jobs", "2", "--message-format=json-render-diagnostics",
               "--manifest-path", str(project / "Cargo.toml")]
    start = {"command": command, "target_directory": str(target), "preparation_sha256": args.preparation_sha256,
             "plan_sha256": prep["plan_sha256"], "platform": platform.platform(), "python": sys.version,
             "tools": tools, "environment": {k: environment[k] for k in ENV_KEYS if k in environment},
             "rustc_version": subprocess.check_output(["rustc", "-Vv"], env=environment, text=True),
             "cargo_version": subprocess.check_output(["cargo", "-V"], env=environment, text=True)}
    write_new(directory / "build-start.json", start)
    with (directory / "build.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    require(code == 0, "Build failed; preserved log, no model execution")
    artifact, emitted = selected_artifact((directory / "build.log").read_text(encoding="utf-8"), project, target)
    binary = directory / "crossover-tests.exe"
    shutil.copyfile(emitted, binary)
    require(sha(emitted) == sha(binary), "Copied binary differs")
    listing = subprocess.check_output([str(binary), "--list"], cwd=ROOT, env=environment, text=True)
    require(listing.splitlines().count(TEST + ": test") == 1, "Missing selected ignored diagnostic")
    (directory / "test-list.txt").write_text(listing, encoding="utf-8", newline="\n")
    unchanged(directory, prep, plan)
    require(all(sha(Path(v["path"])) == v["sha256"] for v in tools.values()), "Build tool changed")
    result = {**start, "kind": "fp32-crossover-layer7-build-v1", "status": "built_not_executed",
              "cargo_artifact": artifact, "binary": binary.relative_to(ROOT).as_posix(),
              "binary_sha256": sha(binary), "build_log_sha256": sha(directory / "build.log"),
              "test_list_sha256": sha(directory / "test-list.txt"),
              "build_start_sha256": sha(directory / "build-start.json"), "selected_test": TEST}
    write_new(directory / "build.json", result)
    print(json.dumps({"status": result["status"], "build_sha256": sha(directory / "build.json")}))


def run(args):
    require(sys.platform == "win32", "Original CPU control is native Windows")
    directory, prep, plan = load_prepared(args)
    build_record = read_bound(directory / "build.json", args.build_sha256)
    require(build_record["kind"] == "fp32-crossover-layer7-build-v1"
            and build_record["preparation_sha256"] == args.preparation_sha256
            and build_record["plan_sha256"] == prep["plan_sha256"] and build_record["selected_test"] == TEST,
            "Build identity differs")
    bound = {directory / "build.json": args.build_sha256,
             directory / "preparation.json": args.preparation_sha256,
             resolve(build_record["binary"]): build_record["binary_sha256"]}
    for filename, field in [("build.log", "build_log_sha256"), ("test-list.txt", "test_list_sha256"),
                            ("build-start.json", "build_start_sha256")]:
        bound[directory / filename] = build_record[field]
    require(all(sha(p) == h for p, h in bound.items()), "Build artifact changed")
    require(not (directory / "run-start.json").exists(), "Execution already attempted; no overwrite/retry")
    binary = resolve(build_record["binary"])
    output = directory / "rust"
    require(not output.exists(), "Rust output already exists")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", FOCR_CROSSOVER_ROOT=str(ROOT),
                       FOCR_CROSSOVER_OUTPUT=str(output), FOCR_CROSSOVER_PLAN=str(directory / "plan.json"),
                       FOCR_CROSSOVER_PLAN_SHA256=prep["plan_sha256"])
    command = [str(binary), TEST, "--exact", "--ignored", "--nocapture", "--test-threads", "1"]
    start = {"command": command, "plan_sha256": prep["plan_sha256"], "build_sha256": args.build_sha256,
             "platform": platform.platform(), "python": sys.version, "python_executable": sys.executable,
             "binary_sha256": sha(binary), "CUDA_VISIBLE_DEVICES": "-1"}
    write_new(directory / "run-start.json", start)
    bound[directory / "run-start.json"] = sha(directory / "run-start.json")
    with (directory / "run.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    log_text = (directory / "run.log").read_text(encoding="utf-8")
    bound[directory / "run.log"] = sha(directory / "run.log")
    require(code == 0 and "1 passed; 0 failed" in log_text and f"test {TEST} ... ok" in log_text,
            "Selected Rust control failed; preserve raw artifacts")
    report_path = output / "report.json"
    report_sha = sha(report_path)
    report = read_bound(report_path, report_sha)
    require(report["status"] == "control_exact" and report["plan_sha256"] == prep["plan_sha256"]
            and report["test_binary_sha256"] == build_record["binary_sha256"], "Rust control/build mismatch")
    require([x["stage"] for x in report["controls"]] == CONTROL_STAGES
            and all(x["passed"] and x["bit_mismatches"] == 0 for x in report["controls"]), "Original controls failed")
    require(set(report["tensors"]) == {"cpu_state." + x for x in STAGES}, "Missing stage outputs")
    bound[report_path] = report_sha
    bound[output / "tensors.safetensors"] = report["tensor_file_sha256"]
    unchanged(directory, prep, plan)
    require(all(sha(p) == h for p, h in bound.items()), "Changed execution/input artifacts")
    write_new(directory / "execution.json", {"kind": "fp32-crossover-layer7-execution-v1", "status": "cpu_control_exact",
              "plan_sha256": prep["plan_sha256"], "build_sha256": args.build_sha256,
              "artifact_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in bound.items()},
              "source_and_input_closure": True, "qualification": "Isolated CPU control only; GPU crossover and analysis pending."})
    print(json.dumps({"status": "cpu_control_exact", "execution_sha256": sha(directory / "execution.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet-window-released", action="store_true", required=True,
                        help="Use only after root explicitly ends the quiet window and approves this phase")
    subs = parser.add_subparsers(dest="phase", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    for name in ("build", "run"):
        p = subs.add_parser(name)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--preparation-sha256", required=True)
        if name == "build":
            p.add_argument("--target-dir", type=Path, required=True)
        else:
            p.add_argument("--build-sha256", required=True)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "build": build, "run": run}[args.phase](args)
    except Exception as error:
        directory = inside_workspace(args.output if args.phase == "prepare" else args.prepared)
        if directory.exists():
            path = directory / (args.phase + "-failed.json")
            if not path.exists():
                write_new(path, {"status": "failed", "phase": args.phase, "error": str(error),
                                 "traceback": traceback.format_exc(), "accepted": False})
        raise


if __name__ == "__main__":
    main()
