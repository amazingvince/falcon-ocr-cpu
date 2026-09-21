"""Independent saved-result review. Run only after the quiet bracket has ended.

This never imports a benchmark driver or executes a model, build, or test. It
checks the fixed two-request bracket from saved reports and archived sources.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "artifacts/benchmarks/attention64-compact-mixed-b2-window-v1"
OUTPUT = ROOT / "reference/benchmarks/windows-attention64-compact-mixed-b2-review-v1.json"
PLAN_SHA = "904841b075f91c3f2109b00db07f4a697fadbb35bfef0a5345f8956e081407cc"
KEYS = ["sparse-room", "table"]
GPU_RECORDS = {
    "sparse-room": ("artifacts/reference/functional-originals-fp32-4096/sparse-room.json",
                    "a9a77300c5f25f03d420603b2172eeeb4f35c09b215bf0d793aadca44148dd0c"),
    "table": ("artifacts/reference/corpus-v3-fp32-4096/2c243b31d36eb729.json",
              "5434fc7bd87510672eb81c7227412cb43f370977c99d8e4d3575612304e361d3"),
}
COUNTS = [6, 2280]
PREFIXES = [3088, 7120]
PREPARED = [[1024, 768], [1184, 1536]]
STAGES = ("image_decode_ms", "preprocessing_ms", "image_projection_ms",
          "transformer_prefill_ms", "prefill_ms", "decode_ms", "total_ms",
          "time_to_first_token_ms")
REPORTED_STAGES = ("preprocessing_ms", "prefill_ms", "decode_ms", "total_ms")
JOBS = [{"name": "control-before", "build": "control"},
        {"name": "candidate", "build": "candidate"},
        {"name": "control-after", "build": "control"}]
bound = {}
checks = 0


def check(ok, label):
    global checks
    checks += 1
    if not ok:
        raise ValueError(label)


def payload(path, expected=None):
    path = Path(path).resolve()
    check(path.is_relative_to(ROOT), "Only project evidence is read")
    check(path.suffix.lower() not in (".etl", ".etlx", ".safetensors")
          and path.stat().st_size < 64 * 1024 * 1024,
          "Bounded metadata/source/executable review")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        check(digest == expected, "Hash mismatch: " + str(path))
    if path in bound:
        check(bound[path] == digest, "Changed during review: " + str(path))
    bound[path] = digest
    return raw


def read(path, expected=None):
    return json.loads(payload(path, expected))


def finite(value, label, positive=False):
    check(type(value) in (int, float) and math.isfinite(value)
          and (value > 0 if positive else value >= 0), label)


def close(actual, expected, label):
    check(type(actual) in (int, float) and math.isfinite(actual)
          and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)


def golden_signatures(plan, workload):
    check(plan["gpu_records"] == {k: str(ROOT / p) for k, (p, _) in GPU_RECORDS.items()},
          "Exact two accepted GPU record paths")
    profiles = [p for p in workload["profiles"] if p["name"] == "mixed-sparse-first"]
    check(len(profiles) == 1 and profiles[0]["order"][:2] == KEYS,
          "Frozen first two input order")
    signatures = []
    for index, key in enumerate(KEYS):
        path, digest = GPU_RECORDS[key]
        gpu = read(ROOT / path, digest)
        item = workload["inputs"][key]
        cfg = gpu["configuration"]
        check(gpu["id"] == item["id"] and gpu["canonical_rgb_sha256"] == item["rgb_sha256"],
              "GPU page and pixels: " + key)
        for scope in (gpu, cfg):
            check("error" not in scope and "postprocessing_replay" not in scope
                  and ("teacher_forced" not in scope or scope["teacher_forced"] is False)
                  and ("inference_reexecuted" not in scope or scope["inference_reexecuted"] is True),
                  "Original free inference: " + key)
        for field in ("max_new_tokens", "min_dimension", "max_dimension", "precision"):
            check(cfg[field] == workload["runtime"][field], "GPU request option: " + field)
        check(cfg["model_revision"] == workload["model"]["revision"]
              and cfg["weights_sha256"] == workload["model"]["assets"]["model.safetensors"]["sha256"]
              and cfg["tf32"] is False and cfg["flex_float32_precision"] == "ieee"
              and cfg["compiled_blocks"] is False, "GPU model and math contract")
        ids = gpu["token_ids"]
        check(len(ids) == COUNTS[index] and all(type(i) is int and 0 <= i < 65536 for i in ids)
              and ids[-1] in (11, 263) and not any(i in (11, 263) for i in ids[:-1])
              and gpu["finish_reason"] == "eos", "Golden full ID/stop contract")
        check(gpu["prefix_length"] == item["input_tokens_expected"] == PREFIXES[index]
              and item["prepared_dimensions_expected"] == PREPARED[index]
              and isinstance(gpu["text"], str), "Golden prefix/text and expected CPU dimensions")
        signatures.append({"token_ids": ids, "text": gpu["text"], "finish_reason": "eos",
                           "output_tokens": len(ids), "input_tokens": gpu["prefix_length"],
                           "width": PREPARED[index][0], "height": PREPARED[index][1]})
    check(plan["expected_signatures"] == signatures, "Plan exactly matches accepted GPU signatures")
    return signatures


def main():
    check(not OUTPUT.exists(), "Fresh review receipt required")
    check(not (FOLDER / "execution-failed.json").exists(), "Failed brackets cannot be accepted as complete")
    payload(__file__)
    plan = read(FOLDER / "plan.json", PLAN_SHA)
    comparison = read(FOLDER / "comparison.json")
    check(comparison["status"] == "complete" and comparison["plan_sha256"] == PLAN_SHA,
          "Completed comparison binding")
    check(plan["jobs"] == JOBS and len(comparison["reports"]) == 3, "Three-process bracket order")
    check(plan["kind"] == "mixed-b2-compact-candidate-bracket-v1"
          and plan["output_directory"] == str(FOLDER), "Exact B2 scope")
    check(plan["repetitions"] == 3 and plan["max_process_seconds"] == 1800
          and plan["default_promotion"] is False and comparison["default_promotion"] is False,
          "Frozen bracket limits")
    for name in ("control_drift_limit_percent", "target_latency_reduction_percent",
                 "maximum_latency_regression_percent"):
        check(plan[name] == 5, "Prospective five-percent criteria: " + name)
    check(set(plan["builds"]) == {"control", "candidate"}, "Build inventory")
    for path, digest in plan["files_sha256"].items():
        payload(path, digest)
    workload = read(plan["workload"])
    runtime = workload["runtime"]
    check({k: runtime[k] for k in ("warmup", "threads", "backend", "precision", "min_dimension", "max_dimension", "max_new_tokens")} ==
          {"warmup": 2, "threads": 16, "backend": "avx2", "precision": "fp32",
           "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}, "Frozen runtime")
    check(workload["model"]["revision"] == "fe757d59ecd79d4d68760162306a70a015761ad9"
          and workload["model"]["assets"]["model.safetensors"]["sha256"] ==
          "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16", "Pinned model labels")
    expected = golden_signatures(plan, workload)
    protocol_names = {"experiments/profiling/benchmark_mixed_compact_candidate.py", "scripts/realistic_benchmark.py"}
    with zipfile.ZipFile(io.BytesIO(payload(FOLDER / "protocol-source.zip"))) as archive:
        check(len(archive.namelist()) == 2 and set(archive.namelist()) == protocol_names,
              "Exact protocol archive inventory")
        for name in protocol_names:
            check(hashlib.sha256(archive.read(name)).hexdigest() == plan["files_sha256"][str(ROOT / name)],
                  "Protocol archive matches frozen script")
    required = {str(ROOT / n) for n in protocol_names}
    required.update({str(FOLDER / "protocol-source.zip"), plan["workload"],
                     *plan["builds"].values(), *plan["gpu_records"].values()})
    builds = {}
    for label, path in plan["builds"].items():
        build = read(path)
        check(build["status"] == "complete" and build["build_exit_code"] == 0
              and build["source_unchanged_during_build"] is True, "Completed preserved build")
        parent = Path(path).parent
        required.update({str(parent / build["binary"]), str(parent / build["source_archive"])})
        check(len(payload(parent / build["binary"], build["binary_sha256"])) == build["binary_bytes"],
              "Binary identity and length")
        with zipfile.ZipFile(io.BytesIO(payload(parent / build["source_archive"], build["source_archive_sha256"]))) as archive:
            check(len(archive.namelist()) == len(build["source_sha256"])
                  and set(archive.namelist()) == set(build["source_sha256"]), "Exact build archive inventory")
            for name, digest in build["source_sha256"].items():
                check(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Archived build source: " + name)
            harness = archive.read("examples/ocr_bench.rs").decode()
            check('previous == &ids, "nondeterministic output during benchmark"' in harness
                  and 'previous == &outputs' in harness and 'for _ in 0..args.repetitions' in harness,
                  "Compiled harness enforces every measured token vector/text/stop")
            runner = archive.read("src/runner.rs").decode()
            check("if chunk.len() == 1 {" in runner and "self.model.decode_batch(" in runner,
                  "Archived runner distinguishes singleton and joint decode")
        builds[label] = build
    check(set(plan["files_sha256"]) == required, "Plan bound file inventory")
    control, candidate = builds["control"], builds["candidate"]
    check(control["binary_sha256"] != candidate["binary_sha256"], "Distinct executables")
    check(Path(control["environment_overrides"]["CARGO_TARGET_DIR"]).resolve() !=
          Path(candidate["environment_overrides"]["CARGO_TARGET_DIR"]).resolve(), "Distinct build targets")
    check(control["source_sha256"].keys() == candidate["source_sha256"].keys(), "Identical source inventory")
    changed = [n for n in control["source_sha256"] if control["source_sha256"][n] != candidate["source_sha256"][n]]
    check(changed == plan["source_changes"] == ["src/kernels.rs"], "Only kernel source differs")
    for key in ("rustc_version", "cargo_version"):
        check(control[key] == candidate[key], "Build tool: " + key)
    check(control["command"][control["command"].index("build"):] ==
          candidate["command"][candidate["command"].index("build"):], "Cargo arguments")
    check({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
          {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"}, "Build environment")
    start = read(FOLDER / "execution-start.json")
    check(start["plan_sha256"] == PLAN_SHA and bool(start["quiet_attestation"].strip()), "Quiet attestation and plan")
    previous_end = dt.datetime.fromisoformat(start["started_utc"])
    rows = []
    measured = 0
    embedded = {"model", "kernels", "runner", "preprocess", "packed_kernels", "config", "tokenizer", "trace", "lib", "harness"}
    for job, saved in zip(JOBS, comparison["reports"]):
        name = job["name"]
        check(saved["name"] == name, "Comparison order")
        report = read(FOLDER / (name + ".json"), saved["report_sha256"])
        build = builds[job["build"]]
        execution = read(FOLDER / (name + ".execution.json"))
        began = read(FOLDER / (name + ".start.json"))
        check(execution["exit_code"] == 0 and not execution.get("timed_out", False), "Process exit without timeout")
        check(all(execution[k] == v for k, v in began.items()), "Process start/end binding")
        began_time, ended_time = (dt.datetime.fromisoformat(execution[k]) for k in ("started_utc", "finished_utc"))
        check(previous_end <= began_time < ended_time, "Sequential nonoverlapping processes")
        previous_end = ended_time
        command = [str(Path(plan["builds"][job["build"]]).parent / build["binary"]), "--model", str(ROOT / workload["model"]["directory"]),
                   "--threads", "16", "--backend", "avx2", "--execution", "joint", "--cache-layout", "compact", "--weight-layout", "unpacked",
                   "--batches", "2", "--warmup", "2", "--repetitions", "3", "--min-dimension", "64", "--max-dimension", "1536", "--max-new-tokens", "4096",
                   "--cpu-label", plan["cpu_label"], "--environment-label", plan["environment_label"], "--output", str(FOLDER / (name + ".json"))]
        command += [str(ROOT / workload["inputs"][key]["canonical_path"]) for key in KEYS]
        check(execution["command"] == command and execution["cwd"] == str(ROOT), "Exact B2 command and order")
        payload(FOLDER / (name + ".log"))
        check(report["schema_version"] == 2 and report["binary_sha256"] == build["binary_sha256"], "Saved report/build")
        check(report["os"] == "windows" and report["arch"] == "x86_64", "Actual native Windows/x86_64 report platform")
        for key, value in {"precision":"fp32", "threads":16, "backend":"avx2", "warmup":2, "repetitions":3,
                           "cache_layout":"compact", "weight_layout":"unpacked", "cpu_label":plan["cpu_label"],
                           "environment_label":plan["environment_label"]}.items():
            check(report[key] == value, "Report runtime: " + key)
        check(report["options"] == {k:runtime[k] for k in ("min_dimension","max_dimension","max_new_tokens")}, "Report request options")
        check(report["model_revision"] == workload["model"]["revision"]
              and report["weights_sha256"] == workload["model"]["assets"]["model.safetensors"]["sha256"], "Report model labels")
        check(report["cargo_lock_sha256"] == build["source_sha256"]["Cargo.lock"]
              and set(report["source_sha256"]) == embedded, "Embedded source inventory and dependency lock")
        for source, digest in report["source_sha256"].items():
            check(build["source_sha256"]["examples/ocr_bench.rs" if source == "harness" else "src/"+source+".rs"] == digest,
                  "Embedded source binding")
        for key in ("read_decode_ms", "verified_model_load_ms", "weight_packing_ms", "packed_weight_bytes"):
            finite(report[key], "Finite setup metric: " + key)
        check(report["packed_weight_bytes"] == 0, "Unpacked weights")
        check(len(report["images"]) == 2, "Two input images")
        for image, key in zip(report["images"], KEYS):
            want = workload["inputs"][key]
            for dest, source in (("sha256","canonical_png_sha256"),("rgb_sha256","rgb_sha256"),("width","width"),("height","height")):
                check(image[dest] == want[source], "Input metadata: " + key)
            check(Path(image["path"]).resolve() == (ROOT / want["canonical_path"]).resolve(), "Input path/order")
        check(len(report["cases"]) == 1, "One case per process")
        case = report["cases"][0]
        check(case["batch_size"] == case["active_batch_size"] == 2
              and case["execution"] == "independent_prefill_joint_decode" and case["image_indices"] == [0, 1], "B2 dispatch/order")
        check(case["token_ids"] == [e["token_ids"] for e in expected], "Both complete stored token vectors")
        check(len(case["samples"]) == 3, "Three measured pair repetitions")
        for sample in case["samples"]:
            check(len(sample["per_request"]) == 2 and sample["emitted_tokens"] == sum(COUNTS), "Measured request/token counts")
            finite(sample["wall_ms"], "Positive measured wall time", positive=True)
            close(sample["pages_per_second"], 2000 / sample["wall_ms"], "Two-page throughput")
            for request, want in zip(sample["per_request"], expected):
                for key in ("text", "finish_reason", "output_tokens", "input_tokens", "width", "height"):
                    check(request[key] == want[key], "Literal measured output: " + key)
                check(request["teacher_forced"] is False and request["precision"] == "fp32", "Free FP32 generation")
                for stage in STAGES:
                    finite(request["timings"][stage], "Finite per-request stage interval")
                check(request["timings"]["image_decode_ms"] == 0, "File decode excluded")
                check(request["timings"]["image_projection_ms"] + request["timings"]["transformer_prefill_ms"] <=
                      request["timings"]["prefill_ms"] + 1e-6, "Prefill subintervals")
                measured += 1
        med = statistics.median(s["wall_ms"] for s in case["samples"])
        close(case["median_ms"], med, "Case median")
        close(case["median_pages_per_second"], 2000 / med, "Case median throughput")
        close(saved["median_ms"], med, "Comparison median")
        close(saved["median_pages_per_second"], 2000 / med, "Comparison throughput")
        check(saved["samples_ms"] == [s["wall_ms"] for s in case["samples"]], "Comparison sample times")
        check(len(saved["per_request_stage_medians_ms"]) == 2, "Separate request-stage summaries")
        request_stages = []
        for index, key in enumerate(KEYS):
            stages = {stage:statistics.median(s["per_request"][index]["timings"][stage] for s in case["samples"])
                      for stage in STAGES}
            check(set(saved["per_request_stage_medians_ms"][index]) == set(REPORTED_STAGES), "Frozen summary stage inventory")
            for stage in REPORTED_STAGES:
                close(saved["per_request_stage_medians_ms"][index][stage], stages[stage], "Per-request stage median")
            request_stages.append({"input_key":key, "stage_medians_ms":stages})
        check(saved["binary_sha256"] == build["binary_sha256"] and saved["all_measured_outputs_exact"] is True,
              "Comparison executable/parity claims")
        rows.append({"name":name, "median_pair_ms":med, "samples_ms":saved["samples_ms"],
                     "median_pages_per_second":2000/med, "per_request":request_stages, "pid":execution["pid"]})
    check(measured == 18, "Eighteen literal measured request outputs")
    check(previous_end <= dt.datetime.fromisoformat(comparison["finished_utc"]), "Final comparison follows all processes")
    before, candidate_ms, after = (r["median_pair_ms"] for r in rows)
    drift = 100 * (after / before - 1)
    gains = {"control_before":100*(1-candidate_ms/before), "control_after":100*(1-candidate_ms/after)}
    close(comparison["control_drift_percent"], drift, "Control drift arithmetic")
    for key, value in gains.items():
        close(comparison["candidate_latency_reduction_percent"][key], value, "Matched gain arithmetic")
    stable = abs(drift) <= 5
    target = stable and min(gains.values()) >= 5
    no_regression = stable and min(gains.values()) >= -5
    check(comparison["control_stability_pass"] == stable
          and comparison["target_met_in_this_pair"] == target
          and comparison["no_regression_in_this_pair"] == no_regression, "Prospective acceptance predicates")
    phase_drift = []
    for index, key in enumerate(KEYS):
        a, b = (rows[i]["per_request"][index]["stage_medians_ms"] for i in (0, 2))
        phase_drift.append({"input_key":key, **{stage:None if a[stage] == 0 else 100*(b[stage]/a[stage]-1)
                                                for stage in ("prefill_ms", "decode_ms", "total_ms")}})
    for path, digest in list(bound.items()):
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "End-of-review closure")
    receipt = {"schema_version":1, "kind":"windows-attention64-compact-mixed-b2-independent-review-v1",
        "status":"passed_saved_result_review", "plan_sha256":PLAN_SHA, "checks":checks,
        "bound_file_count":len(bound), "inputs_sha256":{str(p):h for p,h in bound.items()}, "all_bound_files_unchanged":True,
        "input_order":KEYS, "measured_pair_repetitions":9, "measured_request_outputs_checked":measured,
        "output_token_counts_per_request":COUNTS, "exact_literal_text_stops_counts_dimensions":True,
        "exact_token_vectors_per_case":True,
        "token_ids_storage":"Two stored vectors per case; the bound compiled harness asserts each measured repetition equals both vectors. All 18 literal request results are separately stored and checked.",
        "gpu_prepared_dimensions_recorded":False, "cpu_dimensions_match_frozen_workload":True,
        "source_changes":changed, "compiler_and_selected_build_environment_match_except_target":True,
        "protocol_archive_verified":True, "three_processes_exit_zero_without_overlap":True,
        "results":rows, "control_drift_percent":drift, "candidate_latency_reduction_percent":gains,
        "control_stability_pass":stable, "target_met_in_this_pair":target, "no_regression_in_this_pair":no_regression,
        "control_stage_drift_percent":phase_drift, "stage_intervals_may_overlap_between_requests":True,
        "default_promotion":False,
        "limitations":[
            "One fixed sparse/table pair and one control/candidate/control bracket, with three measured pair repetitions per process; not a broad workload or platform qualification.",
            "The sparse request emits 6 tokens and the table 2280. This has a short shared decode interval and a long singleton tail; it does not measure sustained B2/B4/B8 throughput.",
            "Per-request decode/total intervals include queue or batch waiting and can overlap. They are not added as service cost or used to infer kernel time; stage medians need not sum to total medians.",
            "GPU records do not record observed prepared dimensions. CPU dimensions match the frozen workload; GPU input pixels, prefix and request options match without claiming observed dimension equality.",
            "The historical table GPU record has incomplete startup provenance. This saved-output join preserves that limitation.",
            "This compact-versus-compact candidate result cannot be added to the separate historical expanded/compact gain.",
            "Two warmups per process are unmeasured and not independently exported. Model load and file decode precede measured warm RGB recognition.",
            "Selected compiler versions/flags and archived sources match except the kernel candidate and target path; builds are not claimed hermetic.",
            "Model/image bytes are not rehashed here; frozen metadata and the completed driver's input validation are retained.",
            "No inference, tests, builds, recorder or ETL/tensor scan; only saved small files, source archives and explicit executable identities are checked."]}
    with OUTPUT.open("x", encoding="utf8", newline="\n") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output":str(OUTPUT), "sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      "checks":checks, "measured_outputs":measured, "gains":gains,
                      "no_regression_in_this_pair":no_regression}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-after-benchmark", action="store_true",
                        help="Affirm the parent released saved-file review after all timed processes exited.")
    args = parser.parse_args()
    if not args.released_after_benchmark:
        parser.error("Wait for benchmark completion/release; this script hashes archives and executables.")
    main()
