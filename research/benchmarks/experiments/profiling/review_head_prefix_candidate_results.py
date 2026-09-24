"""Explicit head-contiguous-prefix B1 saved-result review after parent timing release.

The compact control is the accepted combined-attention build. The candidate
uses the distinct head_contiguous_prefix report / head-contiguous-prefix CLI spelling.
No model, benchmark, test or build is executed; no outputs are read before the
release argument. Model/image bulk validation remains the completed wrapper's
responsibility, as in the preserved bounded B1 reviewers.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import importlib.util
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[4]
CONTROL = ROOT / "artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json"
CONTROL_SHA = "68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0"
CONTROL_EXE_SHA = "8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac"
CONTROL_ARCHIVE_SHA = "f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3"
BASELINE = ROOT / "artifacts/benchmarks/attention64-compact-fullpage-window-v1/candidate.json"
BASELINE_SHA = "de250d2ad37db564503762952fa8557c05dde24c48d2f4d659af136a7752fa91"
PROTOCOL_NAME = "research/benchmarks/experiments/profiling/benchmark_head_prefix_candidate.py"
PROTOCOL_SHA = "9ac50902dab24f36c6da5591117c7eea003ba5e210f1847b92a78594c582d8b2"
COMPARISON_SHA = "b9fd89428424e9a093996378b0623d6d4755aad89d1865c70fd9f7878f4a11db"
CHANGED = ("src/config.rs", "src/lib.rs", "src/model.rs", "src/kernels.rs")
ADDED = ("src/head_contiguous_prefix.rs", "src/head_contiguous_prefix_model_tests.rs")
LAYOUTS = {"control": {"cli": "compact", "report": "compact"},
           "candidate": {"cli": "head-contiguous-prefix", "report": "head_contiguous_prefix"}}
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


def load_protocol(plan):
    """Bind the actual shared validator bytes before loading its pure helpers."""
    path = ROOT / PROTOCOL_NAME
    payload(path, PROTOCOL_SHA)
    helper = ROOT / "research/benchmarks/scripts/realistic_benchmark.py"
    payload(helper, plan["files_sha256"][str(helper)])
    spec = importlib.util.spec_from_file_location("head_prefix_review_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check(module.CONTROL_SHA == CONTROL_SHA and module.CONTROL_EXE_SHA == CONTROL_EXE_SHA
          and module.BASELINE_SHA == BASELINE_SHA and module.LAYOUTS == LAYOUTS
          and module.SOURCE_CHANGES == CHANGED and module.SOURCE_ADDITIONS == ADDED,
          "Reviewed protocol constants")
    check(Path(module.rb.__file__).resolve() == helper, "Loaded exact shared validator path")
    payload(path, PROTOCOL_SHA)
    payload(helper, plan["files_sha256"][str(helper)])
    return module


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
    comparison = read(FOLDER / "comparison.json", COMPARISON_SHA)
    check(comparison["status"] == "complete" and comparison["plan_sha256"] == PLAN_SHA, "Completed comparison binding")
    check(plan["jobs"] == [{"name": "control-before", "build": "control"}, {"name": "candidate", "build": "candidate"}, {"name": "control-after", "build": "control"}], "Bracket order")
    check(len(comparison["reports"]) == 3, "Three comparison rows")
    check(plan["kind"] == "single-page-head-contiguous-prefix-bracket-v1" and plan["output_directory"] == str(FOLDER), "Exact head-prefix plan scope")
    check(plan["repetitions"] == 3 and plan["max_process_seconds"] == 900
          and plan["default_promotion"] is False and comparison["default_promotion"] is False, "Scope")
    check(set(plan["builds"]) == {"control", "candidate"}
          and Path(plan["builds"]["control"]).resolve() == CONTROL
          and Path(plan["historical_baseline"]).resolve() == BASELINE, "Pinned combined-attention control and baseline paths")
    check(plan["files_sha256"][str(CONTROL)] == CONTROL_SHA
          and plan["files_sha256"][str(BASELINE)] == BASELINE_SHA, "Prior independently reviewed control identities")
    inherited_bulk = {}
    for path, digest in plan["files_sha256"].items():
        resolved = Path(path).resolve()
        if resolved.is_relative_to(ROOT / "artifacts/model") or resolved.suffix in (".safetensors", ".png"):
            inherited_bulk[path] = digest
        else:
            payload(path, digest)
    check(plan["cache_layouts"] == LAYOUTS and plan["source_changes"] == sorted(CHANGED)
          and plan["source_additions"] == sorted(ADDED), "Explicit layout and source-delta plan")
    check(plan["host_platform"] == platform.platform(), "Current pinned host identity")
    protocol = load_protocol(plan)
    qualification, qualification_bound = protocol.qualification_receipts(
        plan["builds"]["candidate"], plan["qualification"]["path"], plan["qualification"]["sha256"])
    check(qualification == plan["qualification"], "Accepted functional qualification joins")
    check(all(plan["files_sha256"].get(p) == h for p, h in qualification_bound.items()),
          "Qualification artifact identity joins")
    workload = read(plan["workload"], protocol.WORKLOAD_SHA)
    check(Path(plan["workload"]).resolve() == protocol.rb.DEFAULT_WORKLOAD.resolve(), "Frozen workload path")
    protocol.runtime_check(workload)
    runtime = workload["runtime"]
    check(runtime["precision"] == "fp32"
          and workload["model"]["revision"] == "fe757d59ecd79d4d68760162306a70a015761ad9"
          and workload["model"]["assets"]["model.safetensors"]["sha256"] ==
          "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16", "Pinned FP32 model labels")
    check({k: runtime[k] for k in ("warmup", "threads", "backend", "min_dimension", "max_dimension", "max_new_tokens")} ==
          {"warmup": 2, "threads": 16, "backend": "avx2", "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}, "Frozen workload runtime")
    protocol_path = FOLDER / "protocol-source.zip"
    with zipfile.ZipFile(io.BytesIO(payload(protocol_path))) as archive:
        check(len(archive.namelist()) == 2 and set(archive.namelist()) == {PROTOCOL_NAME, "research/benchmarks/scripts/realistic_benchmark.py"}, "Protocol archive inventory")
        for name in archive.namelist():
            check(hashlib.sha256(archive.read(name)).hexdigest() == plan["files_sha256"][str(ROOT / name)], "Protocol archived/current identity")
    required = {str(ROOT / PROTOCOL_NAME),
                str(ROOT / "research/benchmarks/scripts/realistic_benchmark.py"), str(protocol_path),
                plan["workload"], str(BASELINE), *plan["builds"].values()}
    required.update(qualification_bound)
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
    check(not (control["source_sha256"].keys() - candidate["source_sha256"].keys()), "No deleted control source")
    changed = sorted(name for name in control["source_sha256"] if control["source_sha256"][name] != candidate["source_sha256"][name])
    added = sorted(candidate["source_sha256"].keys() - control["source_sha256"].keys())
    check(changed == plan["source_changes"] == sorted(CHANGED)
          and added == plan["source_additions"] == sorted(ADDED), "Exactly four changed/two added runtime sources")
    check(protocol.compare_builds(control, candidate) == (changed, added), "Shared exact build/source-delta validation")
    for key in ("rustc_version", "cargo_version"):
        check(control[key] == candidate[key], "Build tool " + key)
    check(control["command"][control["command"].index("build"):] == candidate["command"][candidate["command"].index("build"):], "Cargo argument equality")
    check({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
          {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"}, "Compiler environment equality")
    baseline = read(plan["historical_baseline"], BASELINE_SHA)
    check(protocol.baseline_signatures(baseline, workload, control) == plan["expected_signatures"], "Shared frozen baseline signature validation")
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
        check(protocol.report_signatures(report, name, plan, workload, build) == plan["expected_signatures"], "Shared report validator and exact signature")
        check(saved["cache_layout"] == LAYOUTS[job["build"]]["report"], "Saved role layout")
        execution = read(FOLDER / (name + ".execution.json"))
        began = read(FOLDER / (name + ".start.json"))
        check(execution["exit_code"] == 0 and not execution.get("timed_out", False), "Process success")
        check(all(execution[k] == v for k, v in began.items()), "Start/finish identity")
        began_time, ended_time = (dt.datetime.fromisoformat(execution[k]) for k in ("started_utc", "finished_utc"))
        check(previous_end <= began_time < ended_time, "Processes execute without overlap")
        previous_end = ended_time
        command = [str(Path(plan["builds"][job["build"]]).parent / build["binary"]), "--model", str(ROOT / workload["model"]["directory"]),
            "--threads", "16", "--backend", "avx2", "--execution", "joint", "--cache-layout", LAYOUTS[job["build"]]["cli"], "--weight-layout", "unpacked",
            "--batches", "1", "--warmup", "2", "--repetitions", "3", "--min-dimension", "64", "--max-dimension", "1536", "--max-new-tokens", "4096",
            "--cpu-label", plan["cpu_label"], "--environment-label", plan["environment_label"], "--output", str(FOLDER / (name + ".json")),
            str(ROOT / workload["inputs"]["prose"]["canonical_path"])]
        check(execution["command"] == command and execution["cwd"] == str(ROOT), "Exact process command")
        check(protocol.command_for(dict(plan, binary=str(Path(plan["builds"][job["build"]]).parent / build["binary"])), workload, name, FOLDER / (name + ".json")) == command, "Shared CLI/report layout mapping")
        payload(FOLDER / (name + ".log"))
        all_reports.append((name, report, build))
        case = report["cases"][0]
        med = statistics.median(s["wall_ms"] for s in case["samples"])
        check(set(saved["stage_medians_ms"]) == {"preprocessing_ms", "prefill_ms", "decode_ms", "total_ms"}, "Exact reported stage inventory")
        stages = {k: statistics.median(s["per_request"][0]["timings"][k] for s in case["samples"]) for k in saved["stage_medians_ms"]}
        close(saved["median_ms"], med, "Comparison median")
        check(saved["samples_ms"] == [s["wall_ms"] for s in case["samples"]], "Comparison sample times")
        for k, v in stages.items(): close(saved["stage_medians_ms"][k], v, "Stage median " + k)
        check(saved["binary_sha256"] == build["binary_sha256"] and saved["all_measured_outputs_exact"] is True, "Comparison binary/parity claims")
        all_stage_keys = set(case["samples"][0]["per_request"][0]["timings"])
        check(all(set(s["per_request"][0]["timings"]) == all_stage_keys for s in case["samples"]),
              "Complete per-request stage inventory stable")
        all_stages = {k: statistics.median(s["per_request"][0]["timings"][k] for s in case["samples"])
                      for k in sorted(all_stage_keys)}
        memory = {"loaded": report["loaded_memory"], "after": case["memory_after"]}
        for snapshot in memory.values():
            check(set(snapshot) == {"resident_bytes", "peak_resident_bytes", "private_commit_bytes", "peak_private_commit_bytes"},
                  "Windows memory inventory")
            check(all(type(v) is int and v >= 0 for v in snapshot.values()), "Nonnegative memory bytes")
            check(snapshot["peak_resident_bytes"] >= snapshot["resident_bytes"]
                  and snapshot["peak_private_commit_bytes"] >= snapshot["private_commit_bytes"], "Memory peak bounds")
        rows.append({"name": name, "cache_layout": LAYOUTS[job["build"]]["report"], "median_ms": med,
                     "samples_ms": saved["samples_ms"], "stage_medians_ms": stages,
                     "all_request_stage_medians_ms": all_stages, "memory": memory, "pid": execution["pid"]})
    for name, report, build in all_reports:
        check(report["schema_version"] == 2 and report["precision"] == "fp32" and report["binary_sha256"] == build["binary_sha256"], "Report precision/build")
        check(report["os"] == "windows" and report["arch"] == "x86_64", "Actual native Windows/x86_64 report platform")
        role = "candidate" if name == "candidate" else "control"
        for key, value in {"threads":16,"backend":"avx2","warmup":2,"repetitions":3,"cache_layout":LAYOUTS[role]["report"],"weight_layout":"unpacked","cpu_label":plan["cpu_label"],"environment_label":plan["environment_label"]}.items(): check(report[key] == value, "Report runtime " + key)
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
    output_paths = {str(FOLDER / (job["name"] + "." + extension))
                    for job in plan["jobs"] for extension in ("json", "log", "start.json", "execution.json")}
    check(comparison["validated_outputs_unchanged"] is True
          and set(comparison["output_artifact_sha256"]) == output_paths, "Exact completed output artifact closure")
    for path, digest in comparison["output_artifact_sha256"].items():
        payload(path, digest)
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
    memory_deltas = {name: {k: rows[1]["memory"]["after"][k] - rows[index]["memory"]["after"][k]
                           for k in rows[1]["memory"]["after"]}
                     for name, index in (("candidate_minus_control_before_bytes", 0),
                                         ("candidate_minus_control_after_bytes", 2))}
    for path, digest in list(bound.items()): check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,"End-of-review file closure")
    receipt={"schema_version":1,"kind":"windows-head-prefix-fullpage-independent-review-v1","status":"passed_saved_result_review","plan_sha256":PLAN_SHA,
        "comparison_sha256":COMPARISON_SHA,"qualification":qualification,
        "inherited_bulk_artifact_sha256_not_rehashed":inherited_bulk,
        "checks":checks,"bound_file_count":len(bound),"inputs_sha256":{str(p):h for p,h in bound.items()},"all_bound_files_unchanged":True,
        "measured_outputs_checked":exact_measured,"output_tokens_each":1140,"exact_literal_text_stops_counts_dimensions":True,
        "exact_token_vector_per_case":True,"token_ids_storage":"One stored vector per case; the bound compiled harness asserts every measured repetition equals that vector. Literal text/stop/count/dimensions are independently stored and checked for all9 repetitions.",
        "source_changes":changed,"source_additions":added,"cache_layouts":LAYOUTS,"shared_report_validator_passed":True,"completed_output_artifact_hashes_verified":True,"compiler_and_selected_build_environment_match_except_target":True,"protocol_archive_verified":True,"three_processes_exit_zero_without_overlap":True,
        "combined_attention_control_build_sha256":CONTROL_SHA,"combined_attention_control_binary_sha256":CONTROL_EXE_SHA,
        "historical_combined_attention_output_sha256":BASELINE_SHA,
        "results":rows,"control_drift_percent":drift,"candidate_latency_reduction_percent":gains,"whole_page_stability_pass":abs(drift)<=5,"target_met_in_this_page":comparison["target_met_in_this_page"],
        "memory_deltas":memory_deltas,
        "memory_qualification":"Windows whole-process snapshots/high-water marks across load, warmups and measurements; not KV capacities, per-repetition peaks or causal attribution to scratch allocation.",
        "control_stage_drift_percent":phase_drift,"stage_timings_are_nonisolated_wall_intervals":True,"default_promotion":False,
        "limitations":["One page, one control/candidate/control bracket and three measured repetitions per process; this does not establish broad workload, batch or platform promotion.","All processes use joint/unpacked B1: controls use Compact and candidate uses the explicit HeadContiguousPrefix cache. The archived Runner uses its single-page fallback, so this is not a joint multirow performance claim.","This tests the head-contiguous-prefix candidate incrementally over combined fixed64 attention with compact cache. Gains are not added to historical cache/attention percentages.","The historical candidate report fixes outputs and options; its timing is not a fresh control in this bracket.","Stage timings are descriptive wall intervals with control variation, not isolated kernel measurements or causal phase attribution.","Selected compiler versions and flags match; captured sources differ only by the four declared integration files and two added head-prefix modules, and targets are isolated. Neither build is claimed hermetic.","Stage medians need not sum to the median of total time because different repetitions can supply each median.","Two warmups per process are unmeasured and not independently exported.","Model/image bytes were not rehashed in this bounded review; their frozen metadata joins and the completed capture's existing input validation are retained.","Operator/smoke/allocation and compiled-code evidence are separate prerequisites; this receipt does not substitute for them or close GPU numerical failures.","No inference, ETL scan, tests or builds were run. Only small metadata/source/archive and explicit executable identities were read and hashed."]}
    with OUTPUT.open("x",encoding="utf-8",newline="\n") as f: json.dump(receipt,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"output":str(OUTPUT),"sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),"checks":checks,"files":len(bound),"measured_outputs":exact_measured,"gains":gains,"phase_drift":phase_drift}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reference/benchmarks/windows-head-prefix-fullpage-review-v1.json")
    parser.add_argument("--released-after-benchmark", action="store_true")
    args = parser.parse_args()
    if not args.released_after_benchmark:
        parser.error("Wait for parent release after all timed processes exit; this script reads and hashes artifacts.")
    main(args)
