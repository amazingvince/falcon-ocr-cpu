"""Paired-GEMV B1 saved-result review, released only after the timed bracket.

Derived from the preserved compact-B1 reviewer; the control is the accepted
combined-attention candidate and the new candidate comes from the exact plan.
No model, benchmark, test, or build is executed by this script.
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
CONTROL = ROOT / "artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json"
CONTROL_SHA = "68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0"
CONTROL_EXE_SHA = "8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac"
CONTROL_ARCHIVE_SHA = "f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3"
BASELINE = ROOT / "artifacts/benchmarks/attention64-compact-fullpage-window-v1/candidate.json"
BASELINE_SHA = "de250d2ad37db564503762952fa8557c05dde24c48d2f4d659af136a7752fa91"
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
    check(path.suffix.lower() not in (".etl", ".etlx", ".safetensors") and path.stat().st_size < 64 * 1024 * 1024, "Bounded small-file/executable review")
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


def close(actual, expected, label):
    check(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)


def main(args):
    # Argument parsing must require release before reaching any file reads.
    check(args.released_after_benchmark is True, "Parent benchmark release required")
    FOLDER = args.plan.resolve().parent
    OUTPUT = args.output.resolve()
    PLAN_SHA = args.plan_sha256
    check(args.plan.resolve().name == "plan.json" and FOLDER.is_relative_to(ROOT)
          and OUTPUT.is_relative_to(ROOT), "Project plan and fresh result paths")
    check(not OUTPUT.exists(), "Fresh review receipt required")
    check(not (FOLDER / "execution-failed.json").exists(), "Failed brackets cannot qualify as complete")
    payload(__file__)
    plan = read(FOLDER / "plan.json", PLAN_SHA)
    comparison = read(FOLDER / "comparison.json")
    check(comparison["status"] == "complete" and comparison["plan_sha256"] == PLAN_SHA, "Completed comparison binding")
    check(plan["jobs"] == [{"name": "control-before", "build": "control"}, {"name": "candidate", "build": "candidate"}, {"name": "control-after", "build": "control"}], "Bracket order")
    check(len(comparison["reports"]) == 3, "Three comparison rows")
    check(plan["kind"] == "single-page-compact-candidate-bracket-v1" and plan["output_directory"] == str(FOLDER), "Exact compact plan scope")
    check(plan["repetitions"] == 3 and plan["max_process_seconds"] == 900
          and plan["default_promotion"] is False and comparison["default_promotion"] is False, "Scope")
    check(set(plan["builds"]) == {"control", "candidate"}
          and Path(plan["builds"]["control"]).resolve() == CONTROL
          and Path(plan["historical_baseline"]).resolve() == BASELINE, "Pinned combined-attention control and baseline paths")
    check(plan["files_sha256"][str(CONTROL)] == CONTROL_SHA
          and plan["files_sha256"][str(BASELINE)] == BASELINE_SHA, "Prior independently reviewed control identities")
    for path, digest in plan["files_sha256"].items():
        payload(path, digest)
    workload = read(plan["workload"])
    runtime = workload["runtime"]
    check(runtime["precision"] == "fp32"
          and workload["model"]["revision"] == "fe757d59ecd79d4d68760162306a70a015761ad9"
          and workload["model"]["assets"]["model.safetensors"]["sha256"] ==
          "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16", "Pinned FP32 model labels")
    check({k: runtime[k] for k in ("warmup", "threads", "backend", "min_dimension", "max_dimension", "max_new_tokens")} ==
          {"warmup": 2, "threads": 16, "backend": "avx2", "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}, "Frozen workload runtime")
    protocol_path = FOLDER / "protocol-source.zip"
    with zipfile.ZipFile(io.BytesIO(payload(protocol_path))) as archive:
        check(len(archive.namelist()) == 2 and set(archive.namelist()) == {"experiments/profiling/benchmark_compact_candidate.py", "scripts/realistic_benchmark.py"}, "Protocol archive inventory")
        for name in archive.namelist():
            check(hashlib.sha256(archive.read(name)).hexdigest() == plan["files_sha256"][str(ROOT / name)], "Protocol archived/current identity")
    required = {str(ROOT / "experiments/profiling/benchmark_compact_candidate.py"),
                str(ROOT / "scripts/realistic_benchmark.py"), str(protocol_path),
                plan["workload"], str(BASELINE), *plan["builds"].values()}
    builds = {}
    for label, path in plan["builds"].items():
        build = read(path)
        check(build["status"] == "complete" and build["build_exit_code"] == 0 and build["source_unchanged_during_build"] is True, "Build completion")
        parent = Path(path).parent
        required.update({str(parent / build["binary"]), str(parent / build["source_archive"])})
        binary = payload(parent / build["binary"], build["binary_sha256"])
        check(len(binary) == build["binary_bytes"], "Binary length")
        with zipfile.ZipFile(io.BytesIO(payload(parent / build["source_archive"], build["source_archive_sha256"]))) as archive:
            check(len(archive.namelist()) == len(build["source_sha256"]) and set(archive.namelist()) == set(build["source_sha256"]), "Build source inventory")
            for name, digest in build["source_sha256"].items():
                check(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Archived source " + name)
            harness = archive.read("examples/ocr_bench.rs").decode()
            check('previous == &ids, "nondeterministic output during benchmark"' in harness and
                  'previous == &outputs' in harness and 'for _ in 0..args.repetitions' in harness, "Frozen measured repetition guards")
            runner = archive.read("src/runner.rs").decode()
            check("if chunk.len() == 1 {" in runner and "self.recognize_with_trace(&chunk[0], options, &mut scoped)?" in runner, "Archived B1 singleton fallback")
        builds[label] = build
    check(set(plan["files_sha256"]) == required, "Exact plan bound source/build/input inventory")
    control, candidate = builds["control"], builds["candidate"]
    check(control["binary_sha256"] == CONTROL_EXE_SHA and control["source_archive_sha256"] == CONTROL_ARCHIVE_SHA,
          "Exact combined-attention binary/archive control")
    check(control["binary_sha256"] != candidate["binary_sha256"], "Distinct control/candidate executable")
    check(Path(control["environment_overrides"]["CARGO_TARGET_DIR"]).resolve() != Path(candidate["environment_overrides"]["CARGO_TARGET_DIR"]).resolve(), "Distinct build targets")
    check(control["source_sha256"].keys() == candidate["source_sha256"].keys(), "Source keys identical")
    changed = [name for name in control["source_sha256"] if control["source_sha256"][name] != candidate["source_sha256"][name]]
    check(changed == plan["source_changes"] == ["src/kernels.rs"], "Only kernel source differs")
    for key in ("rustc_version", "cargo_version"):
        check(control[key] == candidate[key], "Build tool " + key)
    check(control["command"][control["command"].index("build"):] == candidate["command"][candidate["command"].index("build"):], "Cargo argument equality")
    check({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
          {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"}, "Compiler environment equality")
    baseline = read(plan["historical_baseline"], BASELINE_SHA)
    expected = plan["expected_signatures"][0]
    check(len(plan["expected_signatures"]) == 1 and len(expected["token_ids"]) == expected["output_tokens"] == 1140 and expected["finish_reason"] == "eos", "Expected page output")
    check(all(type(t) is int and 0 <= t < 65536 for t in expected["token_ids"]), "Expected token vocabulary range")
    check(expected["input_tokens"] == workload["inputs"]["prose"]["input_tokens_expected"] and [expected["width"], expected["height"]] == workload["inputs"]["prose"]["prepared_dimensions_expected"], "Expected prepared dimensions and prefix")
    check(expected["token_ids"][-1] in (11, 263) and not any(t in (11, 263) for t in expected["token_ids"][:-1]), "Expected EOS")
    start = read(FOLDER / "execution-start.json")
    check(start["plan_sha256"] == PLAN_SHA and bool(start["quiet_attestation"].strip()), "Quiet attestation/plan binding")
    previous_end = dt.datetime.fromisoformat(start["started_utc"])
    rows = []
    exact_measured = 0
    all_reports = [("historical-baseline", baseline, control)]
    for job, saved in zip(plan["jobs"], comparison["reports"]):
        name = job["name"]
        check(saved["name"] == name, "Saved report order")
        report = read(FOLDER / (name + ".json"), saved["report_sha256"])
        build = builds[job["build"]]
        execution = read(FOLDER / (name + ".execution.json"))
        began = read(FOLDER / (name + ".start.json"))
        check(execution["exit_code"] == 0 and not execution.get("timed_out", False), "Process success")
        check(all(execution[k] == v for k, v in began.items()), "Start/finish identity")
        began_time, ended_time = (dt.datetime.fromisoformat(execution[k]) for k in ("started_utc", "finished_utc"))
        check(previous_end <= began_time < ended_time, "Processes execute without overlap")
        previous_end = ended_time
        command = [str(Path(plan["builds"][job["build"]]).parent / build["binary"]), "--model", str(ROOT / workload["model"]["directory"]),
            "--threads", "16", "--backend", "avx2", "--execution", "joint", "--cache-layout", "compact", "--weight-layout", "unpacked",
            "--batches", "1", "--warmup", "2", "--repetitions", "3", "--min-dimension", "64", "--max-dimension", "1536", "--max-new-tokens", "4096",
            "--cpu-label", plan["cpu_label"], "--environment-label", plan["environment_label"], "--output", str(FOLDER / (name + ".json")),
            str(ROOT / workload["inputs"]["prose"]["canonical_path"])]
        check(execution["command"] == command and execution["cwd"] == str(ROOT), "Exact process command")
        payload(FOLDER / (name + ".log"))
        all_reports.append((name, report, build))
        case = report["cases"][0]
        med = statistics.median(s["wall_ms"] for s in case["samples"])
        stages = {k: statistics.median(s["per_request"][0]["timings"][k] for s in case["samples"]) for k in saved["stage_medians_ms"]}
        close(saved["median_ms"], med, "Comparison median")
        check(saved["samples_ms"] == [s["wall_ms"] for s in case["samples"]], "Comparison sample times")
        for k, v in stages.items(): close(saved["stage_medians_ms"][k], v, "Stage median " + k)
        check(saved["binary_sha256"] == build["binary_sha256"] and saved["all_measured_outputs_exact"] is True, "Comparison binary/parity claims")
        rows.append({"name": name, "median_ms": med, "samples_ms": saved["samples_ms"], "stage_medians_ms": stages, "pid": execution["pid"]})
    for name, report, build in all_reports:
        check(report["schema_version"] == 2 and report["precision"] == "fp32" and report["binary_sha256"] == build["binary_sha256"], "Report precision/build")
        check(report["os"] == "windows" and report["arch"] == "x86_64", "Actual native Windows/x86_64 report platform")
        for key, value in {"threads":16,"backend":"avx2","warmup":2,"repetitions":3,"cache_layout":"compact","weight_layout":"unpacked","cpu_label":plan["cpu_label"],"environment_label":plan["environment_label"]}.items(): check(report[key] == value, "Report runtime " + key)
        check(report["options"] == {k:runtime[k] for k in ("min_dimension","max_dimension","max_new_tokens")}, "Report options")
        check(report["model_revision"] == workload["model"]["revision"] and report["weights_sha256"] == workload["model"]["assets"]["model.safetensors"]["sha256"], "Model metadata pins")
        check(report["cargo_lock_sha256"] == build["source_sha256"]["Cargo.lock"], "Cargo lock binding")
        check(set(report["source_sha256"]) == {"model", "kernels", "runner", "preprocess", "packed_kernels",
                                               "config", "tokenizer", "trace", "lib", "harness"}, "Embedded source inventory")
        for source, digest in report["source_sha256"].items(): check(build["source_sha256"]["examples/ocr_bench.rs" if source == "harness" else "src/"+source+".rs"] == digest, "Embedded source binding")
        check(len(report["images"]) == 1, "Single input")
        image, want = report["images"][0], workload["inputs"]["prose"]
        for dest, source in (("sha256","canonical_png_sha256"),("rgb_sha256","rgb_sha256"),("width","width"),("height","height")): check(image[dest] == want[source], "Input metadata")
        check(Path(image["path"]).resolve() == (ROOT / want["canonical_path"]).resolve(), "Input path")
        check(len(report["cases"]) == 1, "Single report case")
        case = report["cases"][0]
        check(case["batch_size"] == case["active_batch_size"] == 1 and case["execution"] == "independent_prefill_joint_decode" and case["image_indices"] == [0], "Case dispatch")
        check(case["token_ids"] == [expected["token_ids"]], "Exact per-case 1140-ID vector")
        check(len(case["samples"]) == 3, "Three measured outputs")
        for sample in case["samples"]:
            check(len(sample["per_request"]) == 1 and sample["emitted_tokens"] == 1140, "Measured request count")
            request = sample["per_request"][0]
            for key in ("text","finish_reason","output_tokens","input_tokens","width","height"): check(request[key] == expected[key], "Literal measured output " + key)
            check(request["teacher_forced"] is False and request["precision"] == "fp32", "Free measured FP32")
            check(sample["wall_ms"] > 0 and math.isfinite(sample["wall_ms"]), "Positive wall time")
            close(sample["pages_per_second"], 1000 / sample["wall_ms"], "Sample throughput")
            for value in request["timings"].values(): check(type(value) in (int,float) and value >= 0 and math.isfinite(value), "Finite stage interval")
            check(request["timings"]["image_decode_ms"] == 0, "File decode excluded")
            check(request["timings"]["image_projection_ms"] + request["timings"]["transformer_prefill_ms"] <= request["timings"]["prefill_ms"] + 1e-6, "Prefill stage subintervals")
            if name != "historical-baseline": exact_measured += 1
        med = statistics.median(s["wall_ms"] for s in case["samples"])
        close(case["median_ms"], med, "Independent report median")
        close(case["median_pages_per_second"], 1000 / med, "Report median throughput")
    check(exact_measured == 9, "Nine checked measured outputs")
    check(previous_end <= dt.datetime.fromisoformat(comparison["finished_utc"]), "Comparison follows process completion")
    before, candidate_ms, after = (r["median_ms"] for r in rows)
    drift = 100 * (after / before - 1)
    gains = {"control_before":100*(1-candidate_ms/before),"control_after":100*(1-candidate_ms/after)}
    close(comparison["control_drift_percent"], drift, "Drift arithmetic")
    for k,v in gains.items(): close(comparison["candidate_latency_reduction_percent"][k],v,"Gain arithmetic")
    check(plan["control_drift_limit_percent"] == plan["target_latency_reduction_percent"] == 5, "Prospective limits")
    check(comparison["control_stability_pass"] == (abs(drift)<=5) and comparison["target_met_in_this_page"] == (abs(drift)<=5 and min(gains.values())>=5), "Target predicates")
    check(comparison["historical_median_ms"] == plan["historical_median_ms"] == baseline["cases"][0]["median_ms"], "Historical timing binding")
    phase_drift = {k:100*(rows[2]["stage_medians_ms"][k]/rows[0]["stage_medians_ms"][k]-1) for k in ("prefill_ms","decode_ms")}
    for path, digest in list(bound.items()): check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,"End-of-review file closure")
    receipt={"schema_version":1,"kind":"windows-gemv-pair-compact-fullpage-independent-review-v1","status":"passed_saved_result_review","plan_sha256":PLAN_SHA,
        "checks":checks,"bound_file_count":len(bound),"inputs_sha256":{str(p):h for p,h in bound.items()},"all_bound_files_unchanged":True,
        "measured_outputs_checked":exact_measured,"output_tokens_each":1140,"exact_literal_text_stops_counts_dimensions":True,
        "exact_token_vector_per_case":True,"token_ids_storage":"One stored vector per case; the bound compiled harness asserts every measured repetition equals that vector. Literal text/stop/count/dimensions are independently stored and checked for all9 repetitions.",
        "source_changes":changed,"compiler_and_selected_build_environment_match_except_target":True,"protocol_archive_verified":True,"three_processes_exit_zero_without_overlap":True,
        "combined_attention_control_build_sha256":CONTROL_SHA,"combined_attention_control_binary_sha256":CONTROL_EXE_SHA,
        "historical_combined_attention_output_sha256":BASELINE_SHA,
        "results":rows,"control_drift_percent":drift,"candidate_latency_reduction_percent":gains,"whole_page_stability_pass":abs(drift)<=5,"target_met_in_this_page":comparison["target_met_in_this_page"],
        "control_stage_drift_percent":phase_drift,"stage_timings_are_nonisolated_wall_intervals":True,"default_promotion":False,
        "limitations":["One page, one control/candidate/control bracket and three measured repetitions per process; this does not establish broad workload, batch or platform promotion.","All three processes use joint/compact/unpacked B1; the archived Runner uses its single-page fallback, so this is not a joint multirow performance claim.","This tests the paired-GEMV candidate incrementally over combined fixed64 attention with compact cache. Gains are not added to historical cache/attention percentages.","The historical candidate report fixes outputs and options; its timing is not a fresh control in this bracket.","Stage timings are descriptive wall intervals with control variation, not isolated kernel measurements or causal phase attribution.","Selected compiler versions, flags and captured source match except the kernel candidate and isolated target; neither build is claimed hermetic.","Stage medians need not sum to the median of total time because different repetitions can supply each median.","Two warmups per process are unmeasured and not independently exported.","Model/image bytes were not rehashed in this bounded review; their frozen metadata joins and the completed capture's existing input validation are retained.","Operator/smoke/allocation and compiled-code evidence are separate prerequisites; this receipt does not substitute for them or close GPU numerical failures.","No inference, ETL scan, tests or builds were run. Only small metadata/source/archive and explicit executable identities were read and hashed."]}
    with OUTPUT.open("x",encoding="utf-8",newline="\n") as f: json.dump(receipt,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"output":str(OUTPUT),"sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),"checks":checks,"files":len(bound),"measured_outputs":exact_measured,"gains":gains,"phase_drift":phase_drift}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reference/benchmarks/windows-gemv-pair-compact-fullpage-review-v1.json")
    parser.add_argument("--released-after-benchmark", action="store_true")
    args = parser.parse_args()
    if not args.released_after_benchmark:
        parser.error("Wait for parent release after all timed processes exit; this script reads and hashes artifacts.")
    main(args)
