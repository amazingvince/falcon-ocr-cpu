"""Independent saved-result arithmetic/identity review; no model or build execution."""
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[4]
FOLDER = ROOT / "artifacts/benchmarks/attention64-fullpage-window-v1"
OUTPUT = ROOT / "reference/benchmarks/windows-attention64-fullpage-review-v1.json"
PLAN_SHA = "38d1fa812e0cf2d78a05355593f6d236fafdc018d5398754e8c9bb0674e63c28"
bound = {}
checks = 0


def check(ok, label):
    global checks
    checks += 1
    if not ok:
        raise ValueError(label)


def payload(path, expected=None):
    path = Path(path).resolve()
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


def main():
    check(not OUTPUT.exists(), "Fresh review receipt required")
    payload(__file__)
    plan = read(FOLDER / "plan.json", PLAN_SHA)
    comparison = read(FOLDER / "comparison.json")
    check(comparison["status"] == "complete" and comparison["plan_sha256"] == PLAN_SHA, "Completed comparison binding")
    check(plan["jobs"] == [{"name": "control-before", "build": "control"}, {"name": "candidate", "build": "candidate"}, {"name": "control-after", "build": "control"}], "Bracket order")
    check(plan["repetitions"] == 3 and plan["default_promotion"] is False and comparison["default_promotion"] is False, "Scope")
    for path, digest in plan["files_sha256"].items():
        payload(path, digest)
    workload = read(plan["workload"])
    runtime = workload["runtime"]
    check({k: runtime[k] for k in ("warmup", "threads", "backend", "min_dimension", "max_dimension", "max_new_tokens")} ==
          {"warmup": 2, "threads": 16, "backend": "avx2", "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}, "Frozen workload runtime")
    protocol_path = FOLDER / "protocol-source.zip"
    with zipfile.ZipFile(io.BytesIO(payload(protocol_path))) as archive:
        check(set(archive.namelist()) == {"research/benchmarks/experiments/profiling/benchmark_candidate.py", "research/benchmarks/scripts/realistic_benchmark.py"}, "Protocol archive inventory")
        for name in archive.namelist():
            check(hashlib.sha256(archive.read(name)).hexdigest() == plan["files_sha256"][str(ROOT / name)], "Protocol archived/current identity")
    builds = {}
    for label, path in plan["builds"].items():
        build = read(path)
        check(build["status"] == "complete" and build["build_exit_code"] == 0 and build["source_unchanged_during_build"] is True, "Build completion")
        parent = Path(path).parent
        binary = payload(parent / build["binary"], build["binary_sha256"])
        check(len(binary) == build["binary_bytes"], "Binary length")
        with zipfile.ZipFile(io.BytesIO(payload(parent / build["source_archive"], build["source_archive_sha256"]))) as archive:
            check(len(archive.namelist()) == len(build["source_sha256"]) and set(archive.namelist()) == set(build["source_sha256"]), "Build source inventory")
            for name, digest in build["source_sha256"].items():
                check(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Archived source " + name)
            harness = archive.read("examples/ocr_bench.rs").decode()
            check('previous == &ids, "nondeterministic output during benchmark"' in harness and
                  'previous == &outputs' in harness and 'for _ in 0..args.repetitions' in harness, "Frozen measured repetition guards")
        builds[label] = build
    control, candidate = builds["control"], builds["candidate"]
    check(control["source_sha256"].keys() == candidate["source_sha256"].keys(), "Source keys identical")
    changed = [name for name in control["source_sha256"] if control["source_sha256"][name] != candidate["source_sha256"][name]]
    check(changed == plan["source_changes"] == ["src/kernels.rs"], "Only kernel source differs")
    for key in ("rustc_version", "cargo_version"):
        check(control[key] == candidate[key], "Build tool " + key)
    check(control["command"][control["command"].index("build"):] == candidate["command"][candidate["command"].index("build"):], "Cargo argument equality")
    check({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
          {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"}, "Compiler environment equality")
    baseline = read(plan["historical_baseline"])
    expected = plan["expected_signatures"][0]
    check(len(plan["expected_signatures"]) == 1 and len(expected["token_ids"]) == expected["output_tokens"] == 1140 and expected["finish_reason"] == "eos", "Expected page output")
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
            "--threads", "16", "--backend", "avx2", "--execution", "sequential", "--cache-layout", "expanded", "--weight-layout", "unpacked",
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
        for key, value in {"threads":16,"backend":"avx2","warmup":2,"repetitions":3,"cache_layout":"expanded","weight_layout":"unpacked","cpu_label":plan["cpu_label"],"environment_label":plan["environment_label"]}.items(): check(report[key] == value, "Report runtime " + key)
        check(report["options"] == {k:runtime[k] for k in ("min_dimension","max_dimension","max_new_tokens")}, "Report options")
        check(report["model_revision"] == workload["model"]["revision"] and report["weights_sha256"] == workload["model"]["assets"]["model.safetensors"]["sha256"], "Model metadata pins")
        check(report["cargo_lock_sha256"] == build["source_sha256"]["Cargo.lock"], "Cargo lock binding")
        check(set(report["source_sha256"]) == set(baseline["source_sha256"]), "Embedded source inventory")
        for source, digest in report["source_sha256"].items(): check(build["source_sha256"]["examples/ocr_bench.rs" if source == "harness" else "src/"+source+".rs"] == digest, "Embedded source binding")
        check(len(report["images"]) == 1, "Single input")
        image, want = report["images"][0], workload["inputs"]["prose"]
        for dest, source in (("sha256","canonical_png_sha256"),("rgb_sha256","rgb_sha256"),("width","width"),("height","height")): check(image[dest] == want[source], "Input metadata")
        check(Path(image["path"]).resolve() == (ROOT / want["canonical_path"]).resolve(), "Input path")
        check(len(report["cases"]) == 1, "Single report case")
        case = report["cases"][0]
        check(case["batch_size"] == case["active_batch_size"] == 1 and case["execution"] == "independent_sequential" and case["image_indices"] == [0], "Case dispatch")
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
            if name != "historical-baseline": exact_measured += 1
        med = statistics.median(s["wall_ms"] for s in case["samples"])
        close(case["median_ms"], med, "Independent report median")
        close(case["median_pages_per_second"], 1000 / med, "Report median throughput")
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
    receipt={"schema_version":1,"kind":"windows-attention64-fullpage-independent-review-v1","status":"passed_saved_result_review","plan_sha256":PLAN_SHA,
        "checks":checks,"bound_file_count":len(bound),"inputs_sha256":{str(p):h for p,h in bound.items()},"all_bound_files_unchanged":True,
        "measured_outputs_checked":exact_measured,"output_tokens_each":1140,"exact_literal_text_stops_counts_dimensions":True,
        "exact_token_vector_per_case":True,"token_ids_storage":"One stored vector per case; the bound compiled harness asserts every measured repetition equals that vector. Literal text/stop/count/dimensions are independently stored and checked for all9 repetitions.",
        "source_changes":changed,"compiler_and_selected_build_environment_match_except_target":True,"protocol_archive_verified":True,"three_processes_exit_zero_without_overlap":True,
        "results":rows,"control_drift_percent":drift,"candidate_latency_reduction_percent":gains,"whole_page_stability_pass":abs(drift)<=5,"target_met_in_this_page":comparison["target_met_in_this_page"],
        "control_stage_drift_percent":phase_drift,"phase_instability_retained":True,"default_promotion":False,
        "limitations":["One page, one control/candidate/control bracket and three measured repetitions per process; this does not establish broad workload, batch or platform promotion.","Control prefill rises from10.6976s to12.5642s while decode falls from64.7545s to60.2765s. Whole-page drift passes but phase timing is unstable; candidate prefill12.7073s/decode55.6175s cannot support clean phase-isolated causal attribution.","Stage medians need not sum to the median of total time because different repetitions can supply each median.","Two warmups per process are unmeasured and not independently exported.","Model/image bytes were not rehashed in this bounded review; their frozen metadata joins and the completed capture's existing input validation are retained.","No inference, ETL scan, tests or builds were run. Only small metadata/source/archive and explicit executable identities were read and hashed."]}
    with OUTPUT.open("x",encoding="utf-8",newline="\n") as f: json.dump(receipt,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"output":str(OUTPUT),"sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),"checks":checks,"files":len(bound),"measured_outputs":exact_measured,"gains":gains,"phase_drift":phase_drift}))


if __name__ == "__main__":
    main()
