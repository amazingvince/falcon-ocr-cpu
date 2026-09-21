#!/usr/bin/env python3
"""Read-only correction of teacher-forced stop reporting; no inference or rebuild."""
import argparse
import json
from pathlib import Path

import capture_temporal_model_v2 as original
from capture_temporal_model_v2 import (require, validate_trace, JOBS, RUNTIME, MODEL_REVISION, WEIGHTS_SHA)

ROOT = original.ROOT
RUN = ROOT / "artifacts/diagnostics/prefix-temporal-model-review-v2"
PINS = {
    "experiments/cache_layout/capture_temporal_model_v2.py": "629be9702c587cb68dd66c22c7762ed96e2b1ff62bf475f78ab3795ce2082414",
    "artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json": "37136e77186727f7381a329a793e9a81753de4a76b7d2725fddeea85fa189d47",
    "artifacts/diagnostics/prefix-temporal-model-review-v2/build.json": "1c7fd97ecb68bd35db82752d423829d7dbb98182dd80cdc8f7a0d3671a2a820b",
    "artifacts/diagnostics/prefix-temporal-model-review-v2/execution.json": "54462f3bfd0c2a1d3ae032777907687ba419e3af44ac5b34efe0c5d420d5d375",
    "reference/prefix-temporal-model-source-review-v2.json": "30e9c46a53663ae72d9827b432415695daff479d785885462f8b1735ec22ff1f",
}

def validate_record(record, teacher):
    require(record["teacher_forced"] is teacher and record["precision"] == "fp32" and record["backend"] == "rust-gemm/avx2", "Inference path")
    require(record["weight_layout"] == "unpacked" and record["packed_weight_bytes"] == 0, "Weight layout")
    ids = record["token_ids"]
    require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < 65536 for x in ids), "Token IDs")
    require(record["output_tokens"] == len(ids) and record["finish_reason"] == ("length" if teacher else "eos"), "Count/stop")
    require(not teacher or len(ids) == 17, "Teacher trace length must be exactly seventeen")
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


def revalidate_saved():
    bound = {ROOT / name: expected for name, expected in PINS.items()}
    for path in [Path(__file__), Path(__file__).with_name("test_temporal_model_saved_v1.py")]:
        bound[path] = original.sha(path)

    def load(path, expected=None):
        path = Path(path)
        if expected is not None:
            require(path not in bound or bound[path] == expected, "Conflicting recorded identity")
            bound[path] = expected
        require(path in bound, "Unbound saved input")
        return json.loads(original.checked_bytes(path, bound[path]))

    def stable():
        for path, expected in bound.items():
            require(original.sha(path) == expected, "Saved source/artifact changed: " + str(path))

    stable()
    plan = load(RUN / "plan.json")
    original.validate_plan(RUN / "plan.json")
    bound.update({ROOT / name: h for name, h in plan["inputs_sha256"].items()})
    bound[RUN / "source.zip"] = plan["source_archive_sha256"]
    for label, inventory in plan["projects_sha256"].items():
        bound.update({RUN / label / name: h for name, h in inventory.items()})
    build = load(RUN / "build.json")
    require(build["status"] == "built_without_inference" and build["plan_sha256"] == bound[RUN / "plan.json"], "Saved build identity")
    original.validate_build_identity(build, plan)
    execution = load(RUN / "execution.json")
    require(execution["status"] == "failed" and execution["error"] == "ValueError('Count/stop')", "Not the preserved reporting-only failure")
    require(execution["plan_sha256"] == bound[RUN / "plan.json"] and
            execution["build_sha256"] == bound[RUN / "build.json"], "Execution ancestry differs")
    require([job["name"] for job in execution["jobs"]] == [job[0] for job in JOBS], "Saved invocation inventory")
    for name, expected in execution["artifacts_sha256"].items():
        path = Path(name)
        require(path not in bound or bound[path] == expected, "Conflicting execution identity")
        bound[path] = expected
    runner_hashes = {}
    for label, item in build["projects"].items():
        log = RUN / (label + "-build.log")
        require(bound[log] == item["log_sha256"] and bound[Path(item["binary"])] == item["binary_sha256"], "Saved build artifact binding")
        original.checked_emitted_executable(original.checked_bytes(log, bound[log]).decode("utf-8"),
                                            RUN / label, Path(item["target_directory"]))
        runner = RUN / label / "src/runner.rs"
        raw = original.checked_bytes(runner, bound[runner])
        require(b"if stops.contains(&token) && teacher_tokens.is_empty()" in raw and
                b"let mut reason = FinishReason::Length;" in raw, "Compiled teacher-stop source differs")
        runner_hashes[label] = bound[runner]
    require(len(set(runner_hashes.values())) == 1, "Control/candidate teacher-stop API differs")
    reports = {}
    for job, (label, project, layout) in zip(execution["jobs"], JOBS):
        require(job["process_exit_code"] == 0, "Model process failed")
        log = RUN / (label + ".log")
        require(bound[log] == job["log_sha256"] and
                b"1 passed; 0 failed; 0 ignored" in original.checked_bytes(log, bound[log]), "Test did not execute")
        invocation = load(RUN / (label + "-invocation.json"))
        require(invocation["command"] == [build["projects"][project]["binary"], "--ignored", "--exact", original.FUNCTION,
                                           "--test-threads=1", "--nocapture"] and invocation["layout"] == layout and
                invocation["build_kind"] == project and invocation["plan_sha256"] == bound[RUN / "plan.json"] and
                invocation["binary_sha256"] == build["projects"][project]["binary_sha256"], "Saved command identity differs")
        reports[label] = load(RUN / (label + ".json"), job["result_sha256"])
        require(len(reports[label]["canonical"]["trace"]["tensors"]) == 1904 and
                len(reports[label]["mixed_trace"]["tensors"]) == 2144, "Observed full tensor inventories differ")
    metadata = load(ROOT / "artifacts/reference/smoke-fp32/metadata.json")
    comparison = compare_reports(reports, metadata["token_ids"])
    stable()
    original.validate_plan(RUN / "plan.json")
    stable()
    return {"schema_version": 1, "status": "saved_output_revalidation_passed", "comparison": comparison,
            "new_inference_executed": False, "rebuild_performed": False, "preserved_original_execution_status": "failed",
            "correction": "The compiled teacher-token path bypasses EOS and retains Length at its exact seventeen-step cap. Only the Python teacher-stop assertion changes to length/17; free runs still require EOS. No tensor, decision or allocation tolerance changed.",
            "compiled_runner_sha256": runner_hashes,
            "process_exit_codes": {job["name"]: job["process_exit_code"] for job in execution["jobs"]},
            "source_and_artifact_closure_unchanged": True,
            "checked_files_sha256": {str(path): expected for path, expected in sorted(bound.items(), key=lambda x: str(x[0]))},
            "qualification": "Windows CPU cache-layout equivalence for fixed smoke/single/mixed cases under concurrent load only; no GPU hidden-stage, natural-corpus, capacity/RSS, performance or production-promotion claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing saved-output receipts")
    result = revalidate_saved()
    original.write(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": original.sha(args.output)}))


if __name__ == "__main__":
    main()
