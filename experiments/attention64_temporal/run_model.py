#!/usr/bin/env python3
"""One explicit temporal-candidate model test, after the frozen operator gates.

Windows only. This is a functional comparison with preserved historical Windows
controls, not a new before/after performance bracket. It never builds a binary.
The original teacher trace intentionally finishes with Length at seventeen
steps; the independent free runs must still finish with EOS.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HISTORICAL = ROOT / "artifacts/diagnostics/prefix-temporal-model-review-v2"
ACCEPTED = ROOT / "reference/prefix-temporal-model-windows-v1.json"
FUNCTION = "temporal_full_model_qualification"
HARNESS_SHA = "26759aeb8c2789a4522639a9a22e5a3dce0900fd7adac5fbf431dc42e18c5b74"
PINS = {
    "reference/prefix-temporal-model-windows-v1.json": "3a0566d6d31ad72ef2f7d6dd5cd5c1dace9c2428c32d90c48aadd485dfbe50ea",
    "artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json": "37136e77186727f7381a329a793e9a81753de4a76b7d2725fddeea85fa189d47",
    "artifacts/diagnostics/prefix-temporal-model-review-v2/build.json": "1c7fd97ecb68bd35db82752d423829d7dbb98182dd80cdc8f7a0d3671a2a820b",
    "artifacts/diagnostics/prefix-temporal-model-review-v2/execution.json": "54462f3bfd0c2a1d3ae032777907687ba419e3af44ac5b34efe0c5d420d5d375",
    "experiments/cache_layout/capture_temporal_model_v2.py": "629be9702c587cb68dd66c22c7762ed96e2b1ff62bf475f78ab3795ce2082414",
    "experiments/cache_layout/compare_temporal_model_saved_v1.py": "c23d207f48513a02777dad145754ed31183e19b33019663abae24c5c64e8dee2",
}
RUNTIME = {"precision": "fp32", "backend": "avx2", "threads": 4,
           "weight_layout": "unpacked", "min_dimension": 64,
           "max_dimension": 256, "max_new_tokens": 24, "mixed_batch_size": 4}
CHANGED = {"src/kernels.rs", "src/config.rs", "src/lib.rs", "src/model.rs"}
ALLOCATIONS = {"decode_starts": 1, "decode_ends": 1,
               "allocation_calls": 0, "requested_bytes": 0}
LIMITS = [
    "One new temporal_candidate invocation is compared with immutable historical Windows controls; no fresh control process or timing bracket.",
    "Complete tensor names, shapes and raw-F32 SHA256 values are compared; this harness preserves tensor digests, not tensor payload files.",
    "The historical execution receipt remains failed because of its old Python teacher-EOS assertion. Its pinned corrected revalidation accepted Length/17 without changing any inference output.",
    "Same compiler strings, unchanged surrounding Rust sources, harness and input bytes establish compatibility; builds are not hermetic and dependency-object provenance remains limited by their original build receipts.",
    "Fixed smoke/single/mixed CPU functional equivalence only: no new performance, GPU hidden-state, natural-corpus, RSS, or production-promotion claim.",
]


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def child(directory, name):
    path = (directory / name).resolve()
    need(path.is_relative_to(directory) and path != directory, "Escaping artifact path")
    return path


class Window:
    """Bind parsed bytes and recheck every consumed source/input/output at closure."""
    def __init__(self):
        self.files = {}

    def register(self, path, expected):
        path = Path(path).resolve()
        need(isinstance(expected, str) and len(expected) == 64
             and all(c in "0123456789abcdef" for c in expected), "Invalid SHA256")
        need(path not in self.files or self.files[path] == expected, "Conflicting identity")
        self.files[path] = expected
        return path

    def data(self, path, expected):
        path = self.register(path, expected)
        raw = path.read_bytes()
        need(digest(raw) == expected, "Changed bytes: " + str(path))
        return raw

    def json(self, path, expected):
        return json.loads(self.data(path, expected))

    def bind(self, path, expected):
        path = self.register(path, expected)
        need(sha(path) == expected, "Changed file: " + str(path))

    def produced(self, path):
        raw = Path(path).read_bytes()
        self.register(path, digest(raw))
        return raw

    def stable(self):
        for path, expected in self.files.items():
            need(sha(path) == expected, "Closure changed: " + str(path))

    def inventory(self):
        return {str(p): h for p, h in sorted(self.files.items(), key=lambda x: str(x[0]))}


def module_from_bytes(name, path, raw):
    """Execute precisely the source bytes already checked, without .pyc reuse."""
    need(name not in sys.modules, "Conflicting preloaded helper: " + name)
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def load_helpers(window, preparation):
    mapping = preparation["experiment_source_sha256"]
    # This independently reviewed launcher is frozen after the build preparation.
    # Its caller-supplied hash and startup archive bind it separately; do not
    # retroactively claim it was included in the preparation/build source set.
    for name, expected in mapping.items():
        window.bind(child(ROOT, name), expected)
    for name in ["patch", "capture"]:
        path = HERE / (name + ".py")
        module = module_from_bytes(name, path, window.data(path, mapping[path.relative_to(ROOT).as_posix()]))
        if name == "capture":
            current = module
    oldpath = ROOT / "experiments/cache_layout/capture_temporal_model_v2.py"
    old = module_from_bytes("capture_temporal_model_v2", oldpath,
                            window.data(oldpath, PINS[oldpath.relative_to(ROOT).as_posix()]))
    correctedpath = oldpath.with_name("compare_temporal_model_saved_v1.py")
    corrected = module_from_bytes("compare_temporal_model_saved_v1", correctedpath,
                                  window.data(correctedpath, PINS[correctedpath.relative_to(ROOT).as_posix()]))
    need(old.RUNTIME == RUNTIME and old.FUNCTION == FUNCTION, "Historical helper contract changed")
    return current, old, corrected


def validate_operators(directory, build, operators, args, capture, window):
    need(operators["status"] == "passed" and operators["source_closure"] is True
         and operators["all_selected_tests_actually_passed"] is True
         and operators["full_model_or_benchmark_executed"] is False, "Operators not accepted")
    need(operators["preparation_sha256"] == args.preparation_sha256
         and operators["build_sha256"] == args.build_sha256, "Wrong operator ancestry")
    expected_names = sorted(capture.TEST_NAMES)
    need(operators["tests"] == expected_names
         and operators["selected_test_count"] == len(expected_names) == 13, "Wrong operator inventory")
    binary = build["executables"]["operators"]
    exe = child(directory, binary["path"])
    window.bind(exe, binary["sha256"])
    need(operators["binary_sha256"] == binary["sha256"], "Wrong operator executable")
    need(len(operators["groups"]) == len(capture.TEST_FILTERS) == 3, "Incomplete groups")
    all_names = []
    for index, (group, selected_filter) in enumerate(zip(operators["groups"], capture.TEST_FILTERS)):
        names = sorted(n for n in expected_names if selected_filter in n)
        need(names and group["filter"] == selected_filter and group["tests"] == names,
             "Wrong group selection")
        need(group["command"] == [str(exe), selected_filter, "--show-output", "--test-threads", "1"]
             and group["exit_code"] == 0 and group["passed"] is True, "Wrong operator command/outcome")
        need(group["log"] == f"operator-{index}.log", "Wrong operator log")
        log = window.data(directory / group["log"], group["log_sha256"]).decode("utf-8")
        need(f"{len(names)} passed; 0 failed" in log
             and all(f"test {name} ... ok" in log for name in names), "Missing actual test passes")
        all_names.extend(names)
    need(sorted(all_names) == expected_names and len(set(all_names)) == 13, "Duplicate/missing operators")


def historical_controls(window, preparation, build, directory, old, corrected):
    def pinned(name):
        return window.json(ROOT / name, PINS[name])
    accepted = pinned("reference/prefix-temporal-model-windows-v1.json")
    need(accepted["status"] == "saved_output_revalidation_passed"
         and accepted["source_and_artifact_closure_unchanged"] is True
         and accepted["new_inference_executed"] is False, "Historical correction not accepted")
    recorded = {Path(p).resolve(): h for p, h in accepted["checked_files_sha256"].items()}
    plan = pinned("artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json")
    built = pinned("artifacts/diagnostics/prefix-temporal-model-review-v2/build.json")
    executed = pinned("artifacts/diagnostics/prefix-temporal-model-review-v2/execution.json")
    need(plan["runtime"] == RUNTIME and built["status"] == "built_without_inference"
         and built["plan_sha256"] == PINS["artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json"],
         "Historical plan/build mismatch")
    need(executed["status"] == "failed" and executed["error"] == "ValueError('Count/stop')"
         and executed["plan_sha256"] == built["plan_sha256"]
         and executed["build_sha256"] == PINS["artifacts/diagnostics/prefix-temporal-model-review-v2/build.json"],
         "Historical reporting failure differs")
    window.bind(HISTORICAL / "source.zip", plan["source_archive_sha256"])
    old_inventory = plan["projects_sha256"]["control"]
    new_inventory = preparation["project_source_sha256"]
    need(set(old_inventory) <= set(new_inventory), "Historical surrounding source is absent")
    differences = {p for p in old_inventory if old_inventory[p] != new_inventory[p]}
    need(differences == CHANGED, "Unexpected source difference versus historical control")
    need(old_inventory["tests/temporal_model_qualification.rs"] == HARNESS_SHA
         == new_inventory["tests/temporal_model_qualification.rs"], "Different harness")
    for name, expected in old_inventory.items():
        window.bind(HISTORICAL / "control" / name, expected)
    benchmark = window.json(directory / "benchmark-build/build.json", build["benchmark_build_sha256"])
    for key in ["rustc_version", "cargo_version"]:
        need(benchmark[key] == built[key], "Historical compiler identity differs: " + key)
    need(benchmark["platform"] == "win32" and benchmark["status"] == "complete", "Non-Windows candidate build")
    # Only immutable historical files and actual runtime inputs are rebound here.
    # Old live production paths are not assumed to still describe an old build.
    for path, expected in executed["artifacts_sha256"].items():
        path = Path(path).resolve()
        need(path.is_relative_to(HISTORICAL) and recorded.get(path) == expected, "Unaccepted historical artifact")
        window.bind(path, expected)
    for name, expected in plan["inputs_sha256"].items():
        if name == "reference/manifest.json" or name.startswith("artifacts/model/") or name.startswith("artifacts/reference/smoke-fp32/"):
            window.bind(ROOT / name, expected)
    metadata_path = ROOT / "artifacts/reference/smoke-fp32/metadata.json"
    metadata = window.json(metadata_path, plan["inputs_sha256"][metadata_path.relative_to(ROOT).as_posix()])
    need([j["name"] for j in executed["jobs"]] == [j[0] for j in old.JOBS], "Historical job inventory")
    reports = {}
    for job, (label, project, layout) in zip(executed["jobs"], old.JOBS):
        need(job["process_exit_code"] == 0, "Historical process failed")
        log = window.data(HISTORICAL / (label + ".log"), job["log_sha256"]).decode("utf-8")
        need("1 passed; 0 failed; 0 ignored" in log, "Historical ignored test did not run")
        invpath = HISTORICAL / (label + "-invocation.json")
        invocation = window.json(invpath, recorded[invpath])
        item = built["projects"][project]
        need(invocation["command"] == [item["binary"], "--ignored", "--exact", FUNCTION,
                                       "--test-threads=1", "--nocapture"]
             and invocation["binary_sha256"] == item["binary_sha256"]
             and invocation["layout"] == layout and invocation["build_kind"] == project
             and Path(invocation["cwd"]).resolve() == ROOT
             and invocation["plan_sha256"] == built["plan_sha256"], "Historical invocation differs")
        reports[label] = window.json(HISTORICAL / (label + ".json"), job["result_sha256"])
    need(corrected.compare_reports(reports, metadata["token_ids"]) == accepted["comparison"],
         "Accepted historical comparison changed")
    return reports["expanded-before"], metadata["token_ids"], {
        "accepted_receipt_sha256": PINS[ACCEPTED.relative_to(ROOT).as_posix()],
        "historical_invocations_revalidated": len(reports),
        "baseline_record_sha256": recorded[HISTORICAL / "expanded-before.json"],
        "unchanged_harness_sha256": HARNESS_SHA,
        "unchanged_surrounding_source_sha256": {p: h for p, h in old_inventory.items() if p not in CHANGED},
        "intended_changed_source_files": sorted(differences),
        "compiler_versions_equal": True,
        "new_historical_control_execution": False,
    }


def compare_candidate(report, baseline, smoke_ids, old, corrected):
    need(report["schema_version"] == 1 and report["status"] == "completed"
         and report["cache_layout"] == "temporal_candidate" and report["runtime"] == RUNTIME,
         "Candidate mode/runtime mismatch")
    need(report["model_revision"] == old.MODEL_REVISION and report["weights_sha256"] == old.WEIGHTS_SHA
         and report["performance_measurement"] is False, "Candidate model/scope mismatch")
    canonical = report["canonical"]
    need(canonical["path_kind"] == "teacher_forced_same_prefix", "Teacher path label")
    # The corrected validator requires Length/17 here, and EOS for free runs.
    corrected.validate_record(canonical["result"], True)
    old.validate_trace(canonical["trace"], True)
    need(len(canonical["trace"]["tensors"]) == 1904
         and canonical["result"]["token_ids"] == smoke_ids
         and [x[1][0] for x in canonical["trace"]["logits_argmax"]] == smoke_ids,
         "Canonical inventory or actual seventeen decisions differ")
    for group in ["independent_free_single", "independent_free_mixed"]:
        need(len(report[group]) == 3, "Missing/extra free result")
        for record in report[group]:
            corrected.validate_record(record, False)
        need([r["output_tokens"] for r in report[group]] == [17, 2, 6], "Wrong uneven workloads")
        need(report[group] == baseline[group], "Independent IDs/text/stops/prefixes differ: " + group)
    need(report["independent_free_single"] == report["independent_free_mixed"], "Single/batch free mismatch")
    trace = report["mixed_trace"]
    old.validate_trace(trace)
    need(len(trace["tensors"]) == 2144, "Mixed tensor inventory differs")
    logits = trace["logits_argmax"]
    need([x[0] for x in logits] == [f"request.{i}.prefill.logits" for i in range(3)]
         + [f"batch.0.decode.{s}.logits" for s in range(16)], "Mixed logit inventory/order")
    derived = [[logits[i][1][0]] for i in range(3)]
    for step in range(16):
        active = [i for i, r in enumerate(report["independent_free_mixed"]) if r["output_tokens"] > step + 1]
        need(trace["active_request_indices"][step] == [f"batch.0.decode.{step}.request_indices", active]
             and trace["decode_rows"][step] == len(active), "Wrong shrinking active rows")
        chosen = logits[step + 3][1]
        need(len(chosen) == len(active), "Wrong actual batch decision count")
        for index, token in zip(active, chosen):
            derived[index].append(token)
    need(derived == [r["token_ids"] for r in report["independent_free_mixed"]], "Mixed logits/output mismatch")
    for key in ["inputs", "canonical", "mixed_trace"]:
        need(report[key] == baseline[key], "Complete input/trace identity differs: " + key)
    for phase in ["single", "mixed"]:
        need(report["allocation"][phase] == ALLOCATIONS, "Warmed allocation interval failed: " + phase)
    return {"status": "historical_windows_model_equivalence_passed",
            "canonical_tensor_count": 1904, "mixed_tensor_count": 2144,
            "teacher_actual_argmax_count": 17, "teacher_finish_reason": "length",
            "independent_free_counts_each_mode": [17, 2, 6], "independent_free_finish_reason": "eos",
            "mixed_decode_rows": trace["decode_rows"], "warmed_zero_allocation_intervals": 2,
            "full_trace_raw_sha256_equal": True, "new_candidate_invocations": 1,
            "new_control_invocations": 0, "performance_claim": False}


def run(args):
    need(sys.platform == "win32", "Native Windows-only historical comparison")
    directory = args.prepared.resolve()
    need(directory.is_relative_to(ROOT) and directory != ROOT, "Preparation outside workspace")
    output = directory / "model-run-v1"
    need(not output.exists(), "Preserve existing model attempt; no implicit retry")
    window = Window()
    window.bind(Path(__file__), args.runner_sha256)
    output.mkdir()
    report = {"kind": "attention64-temporal-model-execution-v1", "status": "preflight",
              "started_utc": utc(), "platform": platform.platform(), "python": sys.version,
              "preparation_sha256": args.preparation_sha256, "build_sha256": args.build_sha256,
              "operators_sha256": args.operators_sha256, "runner_sha256": args.runner_sha256,
              "cache_layout": "temporal_candidate", "process_started": False,
              "process_exit_code": None, "source_input_artifact_closure": False,
              "performance_measurement": False, "limitations": LIMITS}
    proc = None
    capture = preparation = None
    failure = None
    try:
        preparation = window.json(directory / "preparation.json", args.preparation_sha256)
        capture, old, corrected = load_helpers(window, preparation)
        capture.load(args)
        for name, expected in preparation["project_source_sha256"].items():
            window.bind(child(directory / "project", name), expected)
        window.bind(directory / "source.zip", preparation["source_archive_sha256"])
        window.bind(capture.CONTROL / "build.json", capture.CONTROL_BUILD)
        window.bind(capture.CONTROL / "source.zip", capture.CONTROL_ARCHIVE)
        build = window.json(directory / "build.json", args.build_sha256)
        need(build["kind"] == "attention64-temporal-build-v1" and build["status"] == "built_not_executed"
             and build["preparation_sha256"] == args.preparation_sha256, "Wrong candidate build")
        operators = window.json(directory / "operators.json", args.operators_sha256)
        validate_operators(directory, build, operators, args, capture, window)
        item = build["executables"]["qualification"]
        exe = child(directory, item["path"])
        need(item["path"] == "qualification.exe", "Unexpected qualification executable")
        window.bind(exe, item["sha256"])
        inv = window.json(directory / "qualification-invocation.json", item["invocation_sha256"])
        buildlog = window.data(directory / "qualification-build.log", item["log_sha256"]).decode("utf-8")
        artifacts = []
        for line in buildlog.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (row.get("reason") == "compiler-artifact" and row.get("executable")
                    and row["target"]["kind"] == ["test"] and row["target"]["name"] == "temporal_model_qualification"):
                artifacts.append(row)
        need(artifacts == [item["cargo_artifact"]], "Qualification Cargo artifact ambiguous/different")
        artifact = artifacts[0]
        project = directory / "project"
        need(artifact.get("fresh") is False and Path(artifact["manifest_path"]).resolve() == project / "Cargo.toml"
             and Path(artifact["target"]["src_path"]).resolve() == project / "tests/temporal_model_qualification.rs",
             "Wrong qualification project/source")
        emitted = Path(artifact["executable"]).resolve()
        need(emitted.is_relative_to(Path(build["target_directory"]).resolve()), "Wrong qualification target")
        window.bind(emitted, item["sha256"])
        need(inv == {"command": ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                  str(project / "scripts/build_windows.ps1"), "test", "--locked", "--release",
                                  "--no-run", "--test", "temporal_model_qualification", "--jobs", "2",
                                  "--message-format=json-render-diagnostics"], "cwd": str(project)},
             "Qualification build command changed")
        window.bind(project / "tests/temporal_model_qualification.rs", HARNESS_SHA)
        baseline, smoke_ids, compatibility = historical_controls(window, preparation, build, directory, old, corrected)
        report["historical_compatibility"] = compatibility
        report["qualification_binary_sha256"] = item["sha256"]
        # Preserve the exact inspected Python source bytes, including fixed old helpers.
        with zipfile.ZipFile(output / "startup-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, expected in list(window.files.items()):
                if path.suffix == ".py" and path.is_relative_to(ROOT):
                    archive.writestr(path.relative_to(ROOT).as_posix(), window.data(path, expected))
        window.produced(output / "startup-sources.zip")
        command = [str(exe), "--ignored", "--exact", FUNCTION, "--test-threads=1", "--nocapture"]
        overrides = {"CUDA_VISIBLE_DEVICES": "-1", "FOCR_TEMPORAL_MODEL_LAYOUT": "temporal_candidate",
                     "FOCR_TEMPORAL_MODEL_OUTPUT": str(output / "candidate.json")}
        invocation = {"command": command, "cwd": str(ROOT), "environment_overrides": overrides,
                      "preparation_sha256": args.preparation_sha256, "build_sha256": args.build_sha256,
                      "operators_sha256": args.operators_sha256, "binary_sha256": item["sha256"],
                      "harness_sha256": HARNESS_SHA, "runtime": RUNTIME, "timeout_seconds": 1800,
                      "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"]},
                      "checked_inputs_sha256": window.inventory()}
        write(output / "invocation.json", invocation)
        window.produced(output / "invocation.json")
        window.stable()
        capture.closure(directory, preparation)
        need(not (output / "candidate.json").exists(), "Candidate output already exists")
        with (output / "model.log").open("xb") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **overrides),
                                    stdout=log, stderr=subprocess.STDOUT)
            report.update(process_started=True, process_pid=proc.pid, process_started_utc=utc())
            write(output / "process-start.json", {"pid": proc.pid, "started_utc": report["process_started_utc"],
                                                   "invocation_sha256": window.files[output / "invocation.json"]})
            window.produced(output / "process-start.json")
            report["process_exit_code"] = proc.wait(timeout=1800)
        need(report["process_exit_code"] == 0, "Qualification process failed; retain output")
        log = window.produced(output / "model.log").decode("utf-8")
        need(f"test {FUNCTION} ... ok" in log and "1 passed; 0 failed; 0 ignored" in log,
             "The exact ignored integration test did not pass")
        candidate = json.loads(window.produced(output / "candidate.json"))
        report["comparison"] = compare_candidate(candidate, baseline, smoke_ids, old, corrected)
        report["status"] = "passed"
    except BaseException as error:
        failure = error
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        # The receipt-write and wait are both inside the cleanup scope.
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                report["process_exit_code"] = proc.wait()
                report["owned_process_reaped"] = True
            except BaseException as error:
                report.update(status="failed", cleanup_error=repr(error), owned_process_reaped=False)
                failure = failure or error
        try:
            for name in ["model.log", "candidate.json", "process-start.json"]:
                path = output / name
                if path.exists() and path not in window.files:
                    window.produced(path)
            window.stable()
            if capture is not None:
                capture.closure(directory, preparation)
            report["source_input_artifact_closure"] = True
        except BaseException as error:
            report.update(status="failed", closure_error=repr(error))
            failure = failure or error
        report["finished_utc"] = utc()
        report["checked_files_sha256"] = window.inventory()
        write(output / "execution.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / "execution.json"),
                      "sha256": sha(output / "execution.json")}))
    if failure is not None:
        raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-authorized", action="store_true", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    for name in ["preparation", "build", "operators", "runner"]:
        parser.add_argument("--" + name + "-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
