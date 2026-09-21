#!/usr/bin/env python3
"""Prepare/build/run unchanged temporal model qualification sources on Linux, in separate stages."""
import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import subprocess
import sys
import zipfile

import capture_temporal_model_v2 as native
import compare_temporal_model_saved_v1 as corrected

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TARGET_ROOT = PurePosixPath("/home/amazi/falcon-ocr-rust-reference/temporal-model-linux-v1-targets")
NATIVE_DIR = ROOT / "artifacts/diagnostics/prefix-temporal-model-review-v2"
PINS = {
    "reference/prefix-temporal-model-windows-v1.json": "3a0566d6d31ad72ef2f7d6dd5cd5c1dace9c2428c32d90c48aadd485dfbe50ea",
    "reference/prefix-temporal-model-saved-comparator-independent-review-v1.json": "ef78b3becfa3dc68354d9cfcca48a99e737f9a05a89a0b3c5dd69be536d30d46",
    "scripts/build_linux.sh": "25203d4275f25a786bc91531f36d1ebe25ae77736d0dd59b7af1832b0f759e76",
}
CAPTURE_FILES = ["experiments/cache_layout/" + name for name in [
    "capture_temporal_model_linux_v1.py", "test_temporal_model_linux_v1.py",
    "capture_temporal_model_v2.py", "compare_temporal_model_saved_v1.py",
    "test_temporal_model_capture_v2.py", "test_temporal_model_saved_v1.py"]] + ["scripts/build_linux.sh"]
NATIVE_CAPTURE_FILES = ["experiments/cache_layout/capture_temporal_model_v2.py",
    "experiments/cache_layout/test_temporal_model_capture_v2.py", "scripts/build_windows.ps1",
    "tests/cache_layout.rs", "tests/decode_allocations.rs", "tests/weight_layout.rs"]
require, sha, digest, write = native.require, native.sha, native.digest, native.write


def recorded_path(value):
    if sys.platform != "win32" and len(value) >= 3 and value[1] == ":":
        parsed = PureWindowsPath(value)
        return Path("/mnt") / parsed.drive[0].lower() / Path(*parsed.parts[1:])
    return Path(value)


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def checked_json(path, expected):
    return json.loads(native.checked_bytes(path, expected))


def source_members(raw, plan):
    expected = {label + "/" + name: h for label, inventory in plan["projects_sha256"].items() for name, h in inventory.items()}
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate source ZIP member")
        require(set(names) == set(expected) | {"capture/" + n for n in plan.get("capture_files", NATIVE_CAPTURE_FILES)},
                "Complete source/archive inventory differs")
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in name, "Unsafe source ZIP member")
            if name.startswith("capture/"):
                h = plan["inputs_sha256"].get(name[len("capture/"):])
                require(h is not None and digest(archive.read(name)) == h, "Changed native capture member")
                continue
            require(name in expected and digest(archive.read(name)) == expected[name], "Changed/unlisted source member")
            result[name] = archive.read(name)
    require(set(result) == set(expected), "Missing source member")
    return result


def check_inputs(inputs):
    require(all(sha(ROOT / n) == h for n, h in inputs.items()), "Source/evidence/input identity changed")


def prepare(output):
    require(not output.exists(), "Fresh Linux preparation directory required")
    inputs = dict(PINS)
    check_inputs(inputs)
    receipt = checked_json(ROOT / "reference/prefix-temporal-model-windows-v1.json", PINS["reference/prefix-temporal-model-windows-v1.json"])
    require(receipt["status"] == "saved_output_revalidation_passed" and receipt["source_and_artifact_closure_unchanged"], "Native qualification did not pass")
    for value, h in receipt["checked_files_sha256"].items():
        name = relative(recorded_path(value))
        require(name not in inputs or inputs[name] == h, "Conflicting native identity")
        inputs[name] = h
    for name in CAPTURE_FILES:
        actual = sha(ROOT / name)
        require(name not in inputs or inputs[name] == actual, "Frozen comparator/helper changed")
        inputs[name] = actual
    check_inputs(inputs)
    native_plan = checked_json(NATIVE_DIR / "plan.json", inputs[relative(NATIVE_DIR / "plan.json")])
    raw = native.checked_bytes(NATIVE_DIR / "source.zip", native_plan["source_archive_sha256"])
    members = source_members(raw, native_plan)
    require(members["control/tests/temporal_model_qualification.rs"] == members["candidate/tests/temporal_model_qualification.rs"], "Integration sources differ")
    output.mkdir(parents=True)
    for name, contents in members.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
        for name in CAPTURE_FILES:
            archive.writestr("capture/" + name, native.checked_bytes(ROOT / name, inputs[name]))
    plan = {"schema_version": 1, "kind": "temporal-model-linux-qualification-v1", "output_directory_relative": relative(output),
            "inputs_sha256": inputs, "projects_sha256": native_plan["projects_sha256"],
            "capture_files": CAPTURE_FILES,
            "source_archive_sha256": sha(output / "source.zip"), "runtime": native.RUNTIME,
            "build_target_directories": {label: str(TARGET_ROOT / label) for label in ["control", "candidate"]},
            "jobs": native.JOBS, "integration_test": native.TEST_NAME, "test_function": native.FUNCTION,
            "native_receipt_sha256": PINS["reference/prefix-temporal-model-windows-v1.json"],
            "preparation_inference_executed": False,
            "qualification": "Within-Linux CPU layout equivalence first; cross-platform equality is separately observed, not a prerequisite or assumed. No timing/performance/GPU hidden-stage/corpus/default claim."}
    write(output / "plan.json", plan)
    validate(output / "plan.json")
    return {"status": "prepared_without_build_or_inference", "plan_sha256": sha(output / "plan.json")}


def validate(path):
    plan, plan_sha = native.read_json(path)
    require(plan["kind"] == "temporal-model-linux-qualification-v1" and plan["runtime"] == native.RUNTIME, "Plan scope differs")
    require(plan["capture_files"] == CAPTURE_FILES, "Capture source inventory differs")
    require(plan["jobs"] == [list(x) for x in native.JOBS] and plan["integration_test"] == native.TEST_NAME and
            plan["test_function"] == native.FUNCTION, "Invocation/test inventory differs")
    expected_targets = {label: str(TARGET_ROOT / label) for label in ["control", "candidate"]}
    require(plan["build_target_directories"] == expected_targets, "Targets differ")
    if sys.platform == "linux":
        paths = [Path(value).resolve() for value in expected_targets.values()]
        require(paths[0] != paths[1] and not paths[0].is_relative_to(paths[1]) and not paths[1].is_relative_to(paths[0]), "Targets overlap")
    output = ROOT / plan["output_directory_relative"]
    require(output.resolve().is_relative_to(ROOT.resolve()) and Path(path).resolve() == (output / "plan.json").resolve(), "Plan location differs")
    check_inputs(plan["inputs_sha256"])
    members = source_members(native.checked_bytes(output / "source.zip", plan["source_archive_sha256"]), plan)
    for label, inventory in plan["projects_sha256"].items():
        project = output / label
        require({p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file()} == set(inventory), "Source inventory differs")
        require(all(sha(project / n) == h and digest(members[label + "/" + n]) == h for n, h in inventory.items()), "Source bytes differ")
    return plan, plan_sha, output


def linux_env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", CARGO_BUILD_JOBS="2")
    env["PATH"] = "/home/amazi/.cargo/bin:" + env.get("PATH", "")
    env.pop("CARGO_TARGET_DIR", None)
    return env


def build(path):
    require(sys.platform == "linux", "Build requires Linux/WSL")
    plan, plan_sha, output = validate(path)
    require(not (output / "build-start.json").exists(), "No build retry in existing output")
    targets = {k: Path(v) for k, v in plan["build_target_directories"].items()}
    require(all(not p.exists() for p in targets.values()), "Separate fresh Linux targets required")
    env = linux_env()
    rustc = subprocess.check_output(["rustc", "-Vv"], env=env, text=True)
    require("release: 1.92.0\n" in rustc, "Rust version differs from native")
    report = {"schema_version": 1, "status": "building", "plan_sha256": plan_sha, "platform": platform.platform(),
              "python": sys.version, "rustc_version": rustc, "cargo_version": subprocess.check_output(["cargo", "-V"], env=env, text=True),
              "projects": {}, "inference_executed": False,
              "environment": {k: v for k, v in env.items() if k in ["CUDA_VISIBLE_DEVICES", "CARGO_BUILD_JOBS", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ASM_NASM", "FOCR_TOOL_DIR"]}}
    write(output / "build-start.json", report)
    try:
        for label, target in targets.items():
            validate(path)
            env["CARGO_TARGET_DIR"] = str(target)
            command = ["bash", str(ROOT / "scripts/build_linux.sh"), "test", "--offline", "--locked", "--release", "--no-run",
                       "--test", native.TEST_NAME, "--jobs", "2", "--message-format=json-render-diagnostics", "--manifest-path", str(output / label / "Cargo.toml")]
            log = output / (label + "-build.log")
            with log.open("x", encoding="utf-8") as f:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            require(code == 0, "Linux no-run build failed: " + label)
            emitted = native.checked_emitted_executable(log.read_text(encoding="utf-8"), output / label, target)
            binary = output / (label + "-qualification")
            shutil.copyfile(emitted, binary)
            binary.chmod(0o755)
            binary_sha = sha(binary)
            listing = subprocess.check_output([str(binary), "--list"], cwd=ROOT, env=env, text=True)
            require([line for line in listing.splitlines() if line.endswith(": test")] == [native.FUNCTION + ": test"], "Test inventory differs")
            require(sha(binary) == binary_sha, "Listed executable changed")
            report["projects"][label] = {"command": command, "process_exit_code": code, "target_directory": str(target),
                "emitted_executable": str(emitted), "binary": str(binary), "binary_sha256": binary_sha, "log_sha256": sha(log), "test_listing": listing}
        require(report["projects"]["control"]["binary_sha256"] != report["projects"]["candidate"]["binary_sha256"], "Identical control/candidate executable")
        require(validate(path)[1] == plan_sha, "Plan changed during build")
        report["status"] = "built_without_inference"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write(output / "build.json", report)
    return report


def cross_platform_observation(reports, windows):
    fields = ["inputs", "canonical", "independent_free_single", "independent_free_mixed", "mixed_trace", "allocation"]
    return {"scope": "Additional saved Windows/Linux equality observation, separate from within-Linux acceptance.",
            "per_invocation": {label: {field: reports[label][field] == windows[label][field] for field in fields} for label, _, _ in native.JOBS}}


def run(path):
    require(sys.platform == "linux", "Run requires Linux/WSL")
    plan, plan_sha, output = validate(path)
    require(not (output / "execution-start.json").exists(), "No inference retry/resume")
    built, build_sha = native.read_json(output / "build.json")
    require(built["status"] == "built_without_inference" and built["plan_sha256"] == plan_sha and
            set(built["projects"]) == {"control", "candidate"}, "Missing matching Linux build")
    require(built["projects"]["control"]["binary_sha256"] != built["projects"]["candidate"]["binary_sha256"], "Identical executables")
    bound = {output / "build.json": build_sha, Path(path): plan_sha}
    for label, item in built["projects"].items():
        require(item["target_directory"] == plan["build_target_directories"][label], "Build target differs")
        bound[Path(item["binary"])] = item["binary_sha256"]
        log = output / (label + "-build.log")
        bound[log] = item["log_sha256"]
        native.checked_emitted_executable(native.checked_bytes(log, bound[log]).decode("utf-8"), output / label, Path(item["target_directory"]))
    def stable():
        require(validate(path)[1] == plan_sha, "Plan changed")
        require(all(sha(p) == h for p, h in bound.items()), "Build/result evidence changed")
    stable()
    env = linux_env()
    execution = {"schema_version": 1, "status": "running", "plan_sha256": plan_sha, "build_sha256": build_sha,
                 "platform": platform.platform(), "jobs": [], "performance_measurement": False}
    write(output / "execution-start.json", execution)
    reports = {}
    try:
        for label, project, layout in native.JOBS:
            stable()
            result_path = output / (label + ".json")
            require(not result_path.exists(), "Existing result")
            env.update(FOCR_TEMPORAL_MODEL_OUTPUT=str(result_path), FOCR_TEMPORAL_MODEL_LAYOUT=layout)
            command = [built["projects"][project]["binary"], "--ignored", "--exact", native.FUNCTION, "--test-threads=1", "--nocapture"]
            invocation = output / (label + "-invocation.json")
            write(invocation, {"command": command, "cwd": str(ROOT), "layout": layout, "build_kind": project,
                  "binary_sha256": built["projects"][project]["binary_sha256"], "plan_sha256": plan_sha,
                  "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "FOCR_TEMPORAL_MODEL_OUTPUT", "FOCR_TEMPORAL_MODEL_LAYOUT"]}})
            bound[invocation] = sha(invocation)
            log = output / (label + ".log")
            with log.open("x", encoding="utf-8") as f:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            bound[log] = sha(log)
            job = {"name": label, "process_exit_code": code, "log_sha256": bound[log]}
            execution["jobs"].append(job)
            require(code == 0 and "1 passed; 0 failed; 0 ignored" in log.read_text(encoding="utf-8"), "Linux model test failed or did not execute")
            reports[label], bound[result_path] = native.read_json(result_path)
            job["result_sha256"] = bound[result_path]
            require(len(reports[label]["canonical"]["trace"]["tensors"]) == 1904 and len(reports[label]["mixed_trace"]["tensors"]) == 2144, "Tensor inventory differs")
            stable()
        metadata = checked_json(ROOT / "artifacts/reference/smoke-fp32/metadata.json", plan["inputs_sha256"]["artifacts/reference/smoke-fp32/metadata.json"])
        comparison = corrected.compare_reports(reports, metadata["token_ids"])
        windows = {label: checked_json(NATIVE_DIR / (label + ".json"), plan["inputs_sha256"][relative(NATIVE_DIR / (label + ".json"))]) for label, _, _ in native.JOBS}
        comparison["cross_platform_observation"] = cross_platform_observation(reports, windows)
        write(output / "comparison.json", comparison)
        bound[output / "comparison.json"] = sha(output / "comparison.json")
        stable()
        execution.update(status="passed", source_and_artifact_closure_unchanged=True)
    except Exception as error:
        execution.update(status="failed", error=repr(error))
        raise
    finally:
        execution["artifacts_sha256"] = {str(p): h for p, h in bound.items()}
        write(output / "execution.json", execution)
    return execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("--output", type=Path, required=True)
    for action in ["validate", "build", "run"]:
        sub.add_parser(action).add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare": result = prepare(args.output.resolve())
    elif args.action == "validate": result = {"status": "validated_without_build_or_inference", "plan_sha256": validate(args.plan.resolve())[1]}
    elif args.action == "build": result = build(args.plan.resolve())
    else: result = run(args.plan.resolve())
    print(json.dumps({k: result[k] for k in ["status", "plan_sha256", "inference_executed"] if k in result}))


if __name__ == "__main__":
    main()
