#!/usr/bin/env python3
"""Frozen-copy temporal-cache model qualification. Preparation never runs inference."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT / "artifacts/diagnostics/prefix-temporal-candidate-windows-v1"
TEST_NAME = "temporal_model_qualification"
FUNCTION = "temporal_full_model_qualification"
TARGET_ROOT = Path("D:/falcon-ocr-rust-builds/temporal-model-v2")
JOBS = [("expanded-before", "control", "expanded"), ("compact-before", "control", "compact"),
        ("candidate", "candidate", "temporal_candidate"), ("compact-after", "control", "compact"),
        ("expanded-after", "control", "expanded")]
PINS = {
    "reference/prefix-temporal-candidate-linux-v1.json": "356aa2de4cf0bdd285c1b25cb18798601a355bec4e15c9e18d5b52ab9a23cdc8",
    "reference/prefix-temporal-candidate-windows-v1.json": "d5cda80c9eee15fab3347484e38fdf7c391f6c39279763d586b33004f20b0e93",
    "artifacts/diagnostics/prefix-temporal-candidate-windows-v1/build.json": "a1d075c11cc520d50543e33499c2461fa4a98af54ec51fa5b37440cb879a8cb6",
    "artifacts/diagnostics/prefix-temporal-candidate-windows-v1/source.zip": "3c56404aa7a996b0e29b93916a71d50b75cfa033dc33ca8bc12480e51a574176",
    "reference/manifest.json": "245d8f97674066030fdc6a80a6f2eef6641a190cf52baa0ba78755b74db74cc2",
    "artifacts/reference/smoke-fp32/trace.safetensors": "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4",
    "artifacts/reference/smoke-fp32/canonical-rgb.png": "d62f8db10dca06c65c99ae4dde59e4a68b9548c56a42c76caa2bf7044ccca0db",
    "artifacts/reference/smoke-fp32/metadata.json": "c84b1fcc4526135677579392d2c2c52b1566c1f361ac950a6409a5145a8fc65e",
    "tests/cache_layout.rs": "7060aaf05bee2660a7433ce172ab433be9cea261f404a1e4e0af1d6d8092544b",
    "tests/decode_allocations.rs": "c2dac69087350a3c3669294e80f88d4a4c291950d8159285b6853b9eacff3bde",
    "tests/weight_layout.rs": "cca7c3b1cf0da115c801c299efb43b873888a94c421f514576d78c2069a88ffb",
}
RUNTIME = {"precision": "fp32", "backend": "avx2", "threads": 4, "weight_layout": "unpacked",
           "min_dimension": 64, "max_dimension": 256, "max_new_tokens": 24, "mixed_batch_size": 4}
MODEL_REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
WEIGHTS_SHA = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), digest(raw)


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(digest(raw) == expected, "Changed bytes: " + str(path))
    return raw


def check_files(mapping):
    for name, expected in mapping.items():
        require(sha(ROOT / name) == expected, "Identity changed: " + name)


def native_members(raw, build):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate native ZIP entry")
        result = {}
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in name, "Unsafe native member")
            expected = (build["original_source_sha256"].get(name[8:]) if name.startswith("capture/")
                        else build["isolated_source_sha256"].get(name))
            require(expected is not None, "Unlisted native source member")
            data = archive.read(name)
            require(digest(data) == expected, "Changed native source member")
            if not name.startswith("capture/"):
                result[name] = data
        require(set(result) == set(build["isolated_source_sha256"]), "Missing native candidate source")
        return result


def prepare(output):
    require(not output.exists(), "Use a new preparation directory")
    inputs = dict(PINS)
    check_files(inputs)
    linux, _ = read_json(ROOT / "reference/prefix-temporal-candidate-linux-v1.json")
    require(linux["validation"]["rust_tests_passed"] == 8 and linux["validation"]["same_tests_actually_executed"],
            "Same-source Linux focused tests must have completed first")
    inputs.update(linux["evidence_sha256"])
    check_files(inputs)
    native, _ = read_json(NATIVE / "build.json")
    require(linux["identical_candidate_source_inventory"] == native["isolated_source_sha256"], "Linux/native source differs")
    manifest, _ = read_json(ROOT / "reference/manifest.json")
    require(manifest["model_revision"] == MODEL_REVISION and manifest["weights_sha256"] == WEIGHTS_SHA, "Model pin differs")
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        inputs["artifacts/model/" + name] = manifest["files"][name]["sha256"]
    inputs["artifacts/model/model.safetensors"] = WEIGHTS_SHA
    for path in [Path(__file__), HERE / "temporal_model_qualification.rs", HERE / "test_temporal_model_capture_v2.py"]:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    inputs["scripts/build_windows.ps1"] = native["original_source_sha256"]["scripts/build_windows.ps1"]
    candidate = native_members(checked_bytes(NATIVE / "source.zip", inputs[(NATIVE / "source.zip").relative_to(ROOT).as_posix()]), native)
    control = {}
    for name in candidate:
        if name in ["src/temporal_candidate.rs", "src/temporal_model_tests.rs"]:
            continue
        expected = native["original_source_sha256"][name]
        control[name] = checked_bytes(ROOT / name, expected)
        inputs[name] = expected
    differences = {n for n, raw in candidate.items() if control.get(n) != raw}
    require(differences == {"src/config.rs", "src/lib.rs", "src/model.rs", "src/kernels.rs",
                            "src/temporal_candidate.rs", "src/temporal_model_tests.rs"}, "Candidate patch scope differs")
    require(candidate["src/tokenizer.rs"] == control["src/tokenizer.rs"], "Tokenizer control differs")
    harness = checked_bytes(HERE / "temporal_model_qualification.rs", inputs[(HERE / "temporal_model_qualification.rs").relative_to(ROOT).as_posix()])
    for project in [control, candidate]:
        project["tests/" + TEST_NAME + ".rs"] = harness
    check_files(inputs)
    output.mkdir(parents=True)
    inventories = {}
    for label, contents in [("control", control), ("candidate", candidate)]:
        inventories[label] = {n: digest(raw) for n, raw in contents.items()}
        for name, raw in contents.items():
            target = output / label / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, contents in [("control", control), ("candidate", candidate)]:
            for name, raw in contents.items():
                archive.writestr(label + "/" + name, raw)
        for name in [Path(__file__).relative_to(ROOT).as_posix(),
                     (HERE / "test_temporal_model_capture_v2.py").relative_to(ROOT).as_posix(),
                     "scripts/build_windows.ps1", "tests/cache_layout.rs", "tests/decode_allocations.rs", "tests/weight_layout.rs"]:
            archive.writestr("capture/" + name, checked_bytes(ROOT / name, inputs[name]))
    plan = {"schema_version": 2, "kind": "temporal-prefix-model-qualification-v2", "inference_executed": False,
            "output_directory": str(output), "inputs_sha256": inputs, "projects_sha256": inventories,
            "build_target_directories": {label: str((TARGET_ROOT / label).resolve()) for label in ["control", "candidate"]},
            "source_archive_sha256": sha(output / "source.zip"), "runtime": RUNTIME,
            "jobs": JOBS, "integration_test": TEST_NAME, "test_function": FUNCTION,
            "control_provenance": "Unmodified original source, verified against frozen native original-source inventory, with only the same new integration test added.",
            "candidate_provenance": "Exact thirty candidate source files from the already passed Windows/Linux focused-test archive; only the same integration test added.",
            "trace_policy": "Teacher-forced canonical: all1904 F32 tensor name/dtype/shape/raw-byte SHA256 identities plus17 actual logit argmaxes. Mixed trace is free-running; exact emitted IDs must agree first before common-prefix equivalence is assessed.",
            "allocation_policy": "Same warmed Runner; disabled Trace callbacks, exactly one decode interval for single and three-input batch. Zero allocation calls/requested bytes required; excludes prefill and postprocessing.",
            "performance_claim": False, "qualification": "Bounded CPU layout equivalence only; no GPU hidden-stage, corpus-quality, peak-RSS or performance promotion."}
    write(output / "plan.json", plan)
    validate_plan(output / "plan.json")
    return plan


def validate_plan(path):
    plan, plan_sha = read_json(path)
    require(plan["kind"] == "temporal-prefix-model-qualification-v2" and plan["runtime"] == RUNTIME, "Changed plan options")
    validate_targets(plan["build_target_directories"])
    require(plan["jobs"] == [list(job) for job in JOBS], "Changed invocation order")
    require(plan["integration_test"] == TEST_NAME and plan["test_function"] == FUNCTION, "Changed test selection")
    output = Path(plan["output_directory"])
    require(Path(path).resolve() == (output / "plan.json").resolve(), "Plan location differs")
    check_files(plan["inputs_sha256"])
    require(sha(output / "source.zip") == plan["source_archive_sha256"], "Source ZIP changed")
    with zipfile.ZipFile(output / "source.zip") as archive:
        expected = {label + "/" + n: h for label, inventory in plan["projects_sha256"].items() for n, h in inventory.items()}
        expected.update({"capture/" + n: plan["inputs_sha256"][n] for n in [Path(__file__).relative_to(ROOT).as_posix(),
            (HERE / "test_temporal_model_capture_v2.py").relative_to(ROOT).as_posix(), "scripts/build_windows.ps1",
            "tests/cache_layout.rs", "tests/decode_allocations.rs", "tests/weight_layout.rs"]})
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected), "Archive inventory differs")
        require(all(digest(archive.read(n)) == h for n, h in expected.items()), "Archive member differs")
    for label, inventory in plan["projects_sha256"].items():
        project = output / label
        require({p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file()} == set(inventory), "Project inventory differs")
        require(all(sha(project / n) == h for n, h in inventory.items()), "Project source differs")
    return plan, plan_sha


def validate_targets(targets):
    require(set(targets) == {"control", "candidate"}, "Target directory inventory differs")
    resolved = {label: Path(name).resolve() for label, name in targets.items()}
    require(resolved == {label: (TARGET_ROOT / label).resolve() for label in targets}, "Target directory differs")
    require(resolved["control"] != resolved["candidate"] and
            not resolved["control"].is_relative_to(resolved["candidate"]) and
            not resolved["candidate"].is_relative_to(resolved["control"]), "Build targets overlap")
    return resolved


def checked_emitted_executable(log_text, project, target):
    """Bind Cargo's selected artifact to this copy and its fresh isolated target."""
    manifest = (project / "Cargo.toml").resolve()
    emitted = set()
    library_seen = False
    for line in log_text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("reason") != "compiler-artifact":
            continue
        name = item.get("target", {}).get("name")
        if name not in ["falcon_ocr", "falcon-ocr", TEST_NAME]:
            continue
        require(Path(item["manifest_path"]).resolve() == manifest, "Cargo emitted another package manifest")
        require(item.get("fresh") is False, "Project artifact was reused instead of compiled in an empty target")
        expected_source = project / ("tests/" + TEST_NAME + ".rs" if name == TEST_NAME else
                                     "src/lib.rs" if name == "falcon_ocr" else "src/main.rs")
        require(Path(item["target"]["src_path"]).resolve() == expected_source.resolve(), "Cargo emitted another source copy")
        require(all(Path(p).resolve().is_relative_to(target.resolve()) for p in item["filenames"]), "Cargo artifact outside isolated target")
        if name == "falcon_ocr":
            library_seen = True
        if name == TEST_NAME:
            require(item["target"]["kind"] == ["test"] and item["profile"]["test"] and item.get("executable"), "Unexpected test artifact")
            executable = Path(item["executable"]).resolve()
            require(executable.is_relative_to(target.resolve()), "Executable outside isolated target")
            emitted.add(executable)
    require(library_seen and len(emitted) == 1, "Missing library or ambiguous qualification executable")
    return emitted.pop()


def validate_build_identity(built, plan):
    targets = validate_targets(plan["build_target_directories"])
    require(set(built["projects"]) == set(targets), "Build inventory differs")
    for label, target in targets.items():
        require(Path(built["projects"][label]["target_directory"]).resolve() == target, "Build target identity differs")
    require(built["projects"]["control"]["binary_sha256"] != built["projects"]["candidate"]["binary_sha256"],
            "Control and candidate binaries are identical; candidate build is not established")


def build(path):
    require(sys.platform == "win32", "Windows-first qualification")
    plan, plan_sha = validate_plan(path)
    output = Path(plan["output_directory"])
    targets = validate_targets(plan["build_target_directories"])
    require(all(not target.exists() for target in targets.values()), "Targets must be new, empty build locations")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", CARGO_BUILD_JOBS="2")
    env.pop("CARGO_TARGET_DIR", None)
    require(not (output / "build.json").exists(), "Existing build must be preserved")
    report = {"schema_version": 1, "plan_sha256": plan_sha, "status": "building", "platform": platform.platform(),
              "python": sys.version, "rustc_version": subprocess.check_output(["rustc", "-Vv"], text=True),
              "cargo_version": subprocess.check_output(["cargo", "-V"], text=True), "projects": {}, "inference_executed": False,
              "environment": {k: v for k, v in env.items() if k in ["CARGO_TARGET_DIR", "CUDA_VISIBLE_DEVICES", "CARGO_BUILD_JOBS", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ASM_NASM", "CMAKE_GENERATOR", "FOCR_TOOL_DIR"]}}
    write(output / "build-start.json", report)
    try:
        for label in ["control", "candidate"]:
            validate_plan(path)
            env["CARGO_TARGET_DIR"] = str(targets[label])
            command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/build_windows.ps1"),
                       "test", "--offline", "--locked", "--release", "--no-run", "--test", TEST_NAME, "--jobs", "2",
                       "--message-format=json-render-diagnostics", "--manifest-path", str(output / label / "Cargo.toml")]
            log = output / (label + "-build.log")
            with log.open("x", encoding="utf-8") as f:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            require(result.returncode == 0, "Build failed: " + label)
            emitted = checked_emitted_executable(log.read_text(encoding="utf-8"), output / label, targets[label])
            binary = output / (label + "-qualification.exe")
            shutil.copyfile(emitted, binary)
            binary_sha = sha(binary)
            listing = subprocess.check_output([str(binary), "--list"], cwd=ROOT, env=env, text=True)
            require([s for s in listing.splitlines() if s.endswith(": test")] == [FUNCTION + ": test"], "Unexpected integration test inventory")
            require(sha(binary) == binary_sha, "Executable changed during listing")
            report["projects"][label] = {"command": command, "binary": str(binary), "binary_sha256": binary_sha,
                                          "target_directory": str(targets[label]), "emitted_executable": str(emitted),
                                          "log_sha256": sha(log), "test_listing": listing}
        validate_build_identity(report, plan)
        require(validate_plan(path)[1] == plan_sha, "Plan changed during build")
        report["status"] = "built_without_inference"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write(output / "build.json", report)
    return report


def validate_trace(trace, canonical=False):
    tensors = trace["tensors"]
    require(isinstance(tensors, dict) and tensors, "Missing tensor inventory")
    for name, item in tensors.items():
        require(isinstance(name, str) and name and item["dtype"] == "F32-le", "Tensor name/dtype")
        require(all(type(x) is int and x > 0 for x in item["shape"]) and math.prod(item["shape"]) == item["elements"], "Tensor shape/count")
        require(isinstance(item["sha256"], str) and len(item["sha256"]) == 64 and all(c in "0123456789abcdef" for c in item["sha256"]), "Tensor hash")
    require(trace["duplicate_head_tensors_checked"] > 0, "Duplicate-head checks missing")
    names = [v[0] for v in trace["logits_argmax"]]
    require(len(names) == len(set(names)) and set(names) == {n for n in tensors if n.endswith(".logits")}, "Logit inventory")
    for name, chosen in trace["logits_argmax"]:
        shape = tensors[name]["shape"]
        require(shape[-1] == 65536 and len(chosen) == math.prod(shape[:-1])
                and all(type(v) is int and 0 <= v < shape[-1] for v in chosen), "Logit decision shape/range")
    if canonical:
        require(len(tensors) == 1904 and names == ["prefill.logits"] + [f"decode.{n}.logits" for n in range(16)], "Canonical inventory")
        require(all(len(v[1]) == 1 for v in trace["logits_argmax"]), "Canonical decision rows")
    else:
        require(len(trace["decode_rows"]) == 16 and set(trace["decode_rows"]) == {1, 2, 3}, "Mixed row compaction")
        require(len(trace["active_request_indices"]) == 16, "Mixed request schedule")


def validate_record(record, teacher):
    require(record["teacher_forced"] is teacher and record["precision"] == "fp32" and record["backend"] == "rust-gemm/avx2", "Inference path")
    require(record["weight_layout"] == "unpacked" and record["packed_weight_bytes"] == 0, "Weight layout")
    ids = record["token_ids"]
    require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < 65536 for x in ids), "Token IDs")
    require(record["output_tokens"] == len(ids) and record["finish_reason"] == "eos", "Count/stop")
    require(ids[-1] in [11, 263] and not any(v in [11, 263] for v in ids[:-1]), "EOS location")
    require(isinstance(record["text"], str) and type(record["input_tokens"]) is int and record["input_tokens"] > 0, "Text/prefix")
    require((record["width"], record["height"]) == (0, 0) if teacher else all(type(record[x]) is int and record[x] > 0 and record[x] % 16 == 0 for x in ["width", "height"]), "Dimensions")


def compare_reports(reports, smoke_ids):
    require(list(reports) == [job[0] for job in JOBS], "Incomplete/changed invocation inventory")
    baseline = reports["expanded-before"]
    for label, _, layout in JOBS:
        report = reports[label]
        require(report["status"] == "completed" and report["cache_layout"] == layout and report["runtime"] == RUNTIME, "Result/runtime identity")
        require(report["model_revision"] == MODEL_REVISION and report["weights_sha256"] == WEIGHTS_SHA and report["performance_measurement"] is False, "Model/result scope")
        require(report["canonical"]["path_kind"] == "teacher_forced_same_prefix", "Canonical path label")
        validate_record(report["canonical"]["result"], True)
        validate_trace(report["canonical"]["trace"], True)
        require(report["canonical"]["result"]["token_ids"] == smoke_ids
                and [v[1][0] for v in report["canonical"]["trace"]["logits_argmax"]] == smoke_ids,
                "Actual same-prefix decisions differ from pinned smoke IDs")
        for group in ["independent_free_single", "independent_free_mixed"]:
            require(len(report[group]) == 3, "Missing/extra free result")
            for record in report[group]:
                validate_record(record, False)
            require([r["output_tokens"] for r in report[group]] == [17, 2, 6], "Uneven fixture work missing")
        require(report["independent_free_single"] == report["independent_free_mixed"], "Single/joint outputs differ")
        require(report["independent_free_single"][0]["token_ids"] == smoke_ids, "Independent smoke differs from pinned IDs")
        # Establish identical independent prefixes before comparing free traces.
        require(report["independent_free_mixed"] == baseline["independent_free_mixed"], "Free emitted IDs/text/stops/dimensions differ")
        validate_trace(report["mixed_trace"])
        trace = report["mixed_trace"]
        logits = trace["logits_argmax"]
        expected_names = [f"request.{i}.prefill.logits" for i in range(3)] + [f"batch.0.decode.{s}.logits" for s in range(16)]
        require([entry[0] for entry in logits] == expected_names, "Mixed logit order/inventory")
        derived = [[logits[i][1][0]] for i in range(3)]
        for step in range(16):
            active = [i for i, value in enumerate(report["independent_free_mixed"]) if value["output_tokens"] > step + 1]
            require(trace["active_request_indices"][step] == [f"batch.0.decode.{step}.request_indices", active]
                    and trace["decode_rows"][step] == len(active), "Actual mixed request schedule differs")
            choices = logits[3 + step][1]
            require(len(choices) == len(active), "Mixed decision row count")
            for index, token in zip(active, choices):
                derived[index].append(token)
        require(derived == [r["token_ids"] for r in report["independent_free_mixed"]], "Mixed logits do not reconstruct emitted IDs")
        for key in ["inputs", "canonical", "mixed_trace"]:
            require(report[key] == baseline[key], "Trace/input identity differs: " + label + "/" + key)
        for phase in ["single", "mixed"]:
            require(report["allocation"][phase] == {"decode_starts": 1, "decode_ends": 1, "allocation_calls": 0, "requested_bytes": 0}, "Warmed decode allocation failure")
    return {"status": "bounded_cpu_layout_qualification_passed", "invocations": 5,
            "canonical_tensor_count": len(baseline["canonical"]["trace"]["tensors"]),
            "canonical_actual_argmax_decisions": len(baseline["canonical"]["trace"]["logits_argmax"]),
            "mixed_tensor_count": len(baseline["mixed_trace"]["tensors"]), "mixed_decode_rows": baseline["mixed_trace"]["decode_rows"],
            "independent_free_single_results": 15, "independent_free_mixed_results": 15,
            "allocation_intervals": 10, "all_warmed_decode_allocation_calls_zero": True,
            "tensor_comparison": "Complete name/dtype/shape/element-count and SHA256 of raw F32 little-endian bytes, including signed zero.",
            "same_prefix_scope": "Canonical teacher-forced prefix; mixed common prefixes established from exact independently emitted IDs before trace comparison.",
            "performance_claim": False, "gpu_hidden_stage_qualification": False, "corpus_quality_qualification": False}


def run(path):
    require(sys.platform == "win32", "Windows-first qualification")
    plan, plan_sha = validate_plan(path)
    output = Path(plan["output_directory"])
    require(not (output / "execution-start.json").exists(), "No inference retry/resume in existing output")
    built, build_sha = read_json(output / "build.json")
    require(built["status"] == "built_without_inference" and built["plan_sha256"] == plan_sha, "Missing matching preserved build")
    validate_build_identity(built, plan)
    bound = {output / "build.json": build_sha, Path(path): plan_sha}
    for label, item in built["projects"].items():
        bound[Path(item["binary"])] = item["binary_sha256"]
        bound[output / (label + "-build.log")] = item["log_sha256"]
    def stable():
        require(validate_plan(path)[1] == plan_sha, "Plan changed")
        require(all(sha(p) == h for p, h in bound.items()), "Build/execution artifact changed")
    stable()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    execution = {"schema_version": 1, "status": "running", "plan_sha256": plan_sha, "build_sha256": build_sha,
                 "platform": platform.platform(), "jobs": [], "performance_measurement": False}
    write(output / "execution-start.json", execution)
    reports = {}
    try:
        for label, project, layout in JOBS:
            stable()
            result_path = output / (label + ".json")
            require(not result_path.exists(), "Existing result must be preserved")
            env.update(FOCR_TEMPORAL_MODEL_OUTPUT=str(result_path), FOCR_TEMPORAL_MODEL_LAYOUT=layout)
            binary = built["projects"][project]["binary"]
            command = [binary, "--ignored", "--exact", FUNCTION, "--test-threads=1", "--nocapture"]
            invocation = output / (label + "-invocation.json")
            write(invocation, {"command": command, "cwd": str(ROOT), "layout": layout, "build_kind": project,
                               "binary_sha256": built["projects"][project]["binary_sha256"], "plan_sha256": plan_sha,
                               "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "FOCR_TEMPORAL_MODEL_OUTPUT", "FOCR_TEMPORAL_MODEL_LAYOUT"]}})
            bound[invocation] = sha(invocation)
            log = output / (label + ".log")
            with log.open("x", encoding="utf-8") as f:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            bound[log] = sha(log)
            job = {"name": label, "process_exit_code": result.returncode, "log_sha256": bound[log]}
            execution["jobs"].append(job)
            require(result.returncode == 0, "Model qualification process failed: " + label)
            require("1 passed; 0 failed; 0 ignored" in log.read_text(encoding="utf-8"), "Test did not execute")
            reports[label], bound[result_path] = read_json(result_path)
            job["result_sha256"] = bound[result_path]
            stable()
        metadata, _ = read_json(ROOT / "artifacts/reference/smoke-fp32/metadata.json")
        comparison = compare_reports(reports, metadata["token_ids"])
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
    for name in ["validate", "build", "run"]:
        sub.add_parser(name).add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output.resolve())
    elif args.action == "validate":
        _, plan_sha = validate_plan(args.plan.resolve())
        result = {"status": "validated_without_inference", "plan_sha256": plan_sha}
    elif args.action == "build":
        result = build(args.plan.resolve())
    else:
        result = run(args.plan.resolve())
    print(json.dumps({k: result[k] for k in ["status", "kind", "inference_executed"] if k in result}))


if __name__ == "__main__":
    main()
