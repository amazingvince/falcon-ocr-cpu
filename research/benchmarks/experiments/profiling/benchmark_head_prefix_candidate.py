#!/usr/bin/env python3
"""One explicit head-contiguous prefix B1 candidate between pinned compact CPU controls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "research/benchmarks/scripts"))  # realistic_benchmark moved here in the 2026-09-24 archive

import realistic_benchmark as rb

JOBS = [{"name": "control-before", "build": "control"},
        {"name": "candidate", "build": "candidate"},
        {"name": "control-after", "build": "control"}]
KIND = "single-page-head-contiguous-prefix-bracket-v1"
CONTROL = ROOT / "artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json"
CONTROL_SHA = "68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0"
CONTROL_EXE_SHA = "8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac"
BASELINE = ROOT / "artifacts/benchmarks/attention64-compact-fullpage-window-v1/candidate.json"
BASELINE_SHA = "de250d2ad37db564503762952fa8557c05dde24c48d2f4d659af136a7752fa91"
WORKLOAD_SHA = "b4080bc6a1b38a26cc5363e2f1e92712ed89ad96a6a04b618c5123fa81735e5c"
LAYOUTS = {"control": {"cli": "compact", "report": "compact"},
           "candidate": {"cli": "head-contiguous-prefix", "report": "head_contiguous_prefix"}}
# Exact allowlist confirmed by the integration author. Existing Compact remains
# independent; the benchmark harness, runner, tokenizer and other files match.
SOURCE_CHANGES = ("src/config.rs", "src/lib.rs", "src/model.rs", "src/kernels.rs")
SOURCE_ADDITIONS = ("src/head_contiguous_prefix.rs", "src/head_contiguous_prefix_model_tests.rs")
QUALIFICATION_RUNTIME = {"precision": "fp32", "backend": "avx2", "threads": 4,
                         "weight_layout": "unpacked", "min_dimension": 64,
                         "max_dimension": 256, "max_new_tokens": 24, "mixed_batch_size": 4}
OPERATOR_FILTERS = ("head_contiguous_prefix::tests::", "model::head_contiguous_prefix_tests::")
OPERATOR_TESTS = tuple(OPERATOR_FILTERS[0] + name for name in (
    "storage_full_head_order_and_special_bits", "duplicate_bit_failures_leave_every_buffer_unchanged",
    "invalid_sizes_offsets_and_overflow_reject_before_mutation",
    "reserved_capacity_and_addresses_stay_fixed_through_appends",
    "reusable_prefill_scratch_has_compact_order_and_transactional_errors",
    "cache_attention_rejects_wrong_intervals_and_retained_scratch",
    "operator_short_prefix_zero_full_and_tile_crossings_all_supported_backends",
    "operator_fullpage_crossing_tile_and_16384_context_all_supported_backends",
    "unchanged_prefill_and_storage_decode_dispatch_match_compact",
    "operator_invalid_shapes_preserve_output_before_dispatch",
)) + tuple(OPERATOR_FILTERS[1] + name for name in (
    "model_head_prefix_prefill_then_decode_matches_unchanged_compact",
    "model_head_prefix_enum_is_explicit_and_invalid_sessions_reject",
))


def job(name):
    rb.require(name in {j["name"] for j in JOBS} | {"historical-baseline"}, "Unknown fixed job")
    role = "candidate" if name == "candidate" else "control"
    return {"id": name, "profile": "fullpages", "batch": 1,
            "mode": {"execution": "joint", "cache_layout": LAYOUTS[role]["report"], "weight_layout": "unpacked"},
            "image_keys": ["prose"], "request_keys": ["prose"]}


def command_for(plan, workload, name, output):
    role = "candidate" if name == "candidate" else "control"
    command = rb.command_for(plan, workload, job(name), output)
    index = command.index("--cache-layout") + 1
    rb.require(command[index] == LAYOUTS[role]["report"], "Unexpected report layout mapping")
    command[index] = LAYOUTS[role]["cli"]
    return command


def checked_json(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rb.require(expected is None or expected == digest, "File identity changed: " + str(path))
    return json.loads(raw), digest


def qualification_receipts(candidate_path, qualification_path, qualification_sha):
    """Join already accepted functional evidence; never execute its helpers."""
    candidate_path, qualification_path = Path(candidate_path).resolve(), Path(qualification_path).resolve()
    directory = candidate_path.parent.parent
    rb.require(candidate_path == directory / "benchmark-build/build.json"
               and qualification_path == directory / "model-run-v1/comparison.json",
               "Qualification must belong to the same candidate experiment")
    comparison, actual_sha = checked_json(qualification_path, qualification_sha)
    rb.require(comparison.get("kind") == "head-contiguous-prefix-model-comparison-v1"
               and comparison.get("status") == "passed"
               and comparison.get("source_input_closure") is True
               and comparison.get("performance_measurement") is False
               and comparison.get("runtime") == QUALIFICATION_RUNTIME
               and not any(k in comparison for k in ("error", "closure_error")), "Model qualification not accepted")
    expected = {"canonical_cpu_tensors_exact": 1904, "mixed_tensors_exact": 2144,
                "actual_teacher_argmax_exact": 17, "independent_eos_counts": [17, 2, 6],
                "fresh_processes": 2, "warmed_zero_allocation_intervals": 4}
    rb.require(all(comparison.get(k) == v for k, v in expected.items()), "Incomplete model qualification")
    bound = {str(qualification_path): actual_sha}
    receipts = {}
    for role in ("preparation", "build", "operators"):
        path, digest = directory / (role + ".json"), comparison[role + "_sha256"]
        receipts[role], _ = checked_json(path, digest)
        bound[str(path)] = digest
    prep, build, ops = (receipts[k] for k in ("preparation", "build", "operators"))
    rb.require(prep.get("kind") == "head-contiguous-prefix-preparation-v1"
               and prep.get("status") == "prepared_not_built"
               and prep.get("control_build_sha256") == CONTROL_SHA
               and sorted(prep.get("changed_control_sources", [])) == sorted(SOURCE_CHANGES)
               and sorted(prep.get("added_sources", [])) == sorted(SOURCE_ADDITIONS)
               and prep.get("selected_test_count") == len(OPERATOR_TESTS)
               and sorted(prep.get("selected_tests", [])) == sorted(OPERATOR_TESTS)
               and prep.get("test_filters") == list(OPERATOR_FILTERS), "Wrong qualification preparation")
    rb.require(build.get("kind") == "head-contiguous-prefix-build-v1"
               and build.get("status") == "built_not_executed"
               and build.get("preparation_sha256") == comparison["preparation_sha256"], "Wrong qualification build")
    benchmark, benchmark_sha = checked_json(candidate_path, build["benchmark_build_sha256"])
    rb.require(all(prep["project_source_sha256"].get(n) == h for n, h in benchmark["source_sha256"].items()),
               "Qualified project and benchmark source maps differ")
    bound[str(candidate_path)] = benchmark_sha
    rb.require(ops.get("status") == "passed" and ops.get("source_closure") is True
               and ops.get("all_selected_tests_actually_passed") is True
               and ops.get("full_model_or_benchmark_executed") is False
               and ops.get("build_sha256") == comparison["build_sha256"]
               and ops.get("preparation_sha256") == comparison["preparation_sha256"]
               and ops.get("tests") == sorted(OPERATOR_TESTS)
               and ops.get("selected_test_count") == len(OPERATOR_TESTS)
               and ops.get("binary_sha256") == build["executables"]["operators"]["sha256"],
               "Operators are not accepted for this build")
    groups = ops.get("groups", [])
    rb.require(len(groups) == 2, "Missing operator groups")
    for index, (group, selected) in enumerate(zip(groups, OPERATOR_FILTERS)):
        names = sorted(n for n in OPERATOR_TESTS if n.startswith(selected))
        rb.require(group.get("filter") == selected and group.get("tests") == names
                   and group.get("passed") is True and group.get("exit_code") == 0
                   and group.get("log") == f"operator-{index}.log", "Incomplete operator group")
    jobs = comparison.get("jobs", [])
    rb.require(len(jobs) == 2 and [j.get("layout") for j in jobs] == ["compact", "head_contiguous_prefix"],
               "Wrong fresh functional controls")
    for j in jobs:
        rb.require(j.get("exit_code") == 0 and j.get("owned_process_reaped") is True
                   and j.get("binary_sha256") == build["executables"]["qualification"]["sha256"],
                   "Functional process failed or uses another build")
    declared = comparison.get("bound_sha256", {})
    rb.require(declared and all(declared.get(p) == h for p, h in bound.items()
                               if p != str(qualification_path) and p != str(candidate_path)),
               "Qualification source receipt closure missing")
    for name, digest in declared.items():
        path = str(Path(name).resolve())
        rb.require(path not in bound or bound[path] == digest, "Conflicting qualification artifact identity")
        bound[path] = digest
    for label in ("operators", "qualification"):
        exe = str((directory / build["executables"][label]["path"]).resolve())
        rb.require(bound.get(exe) == build["executables"][label]["sha256"], "Qualification executable not bound")
    for group in groups:
        rb.require(bound.get(str(directory / group["log"])) == group["log_sha256"], "Operator log not bound")
    for layout in ("compact", "head_contiguous_prefix"):
        rb.require(str(directory / "model-run-v1" / (layout + ".json")) in bound,
                   "Raw functional result not bound")
    return {"path": str(qualification_path), "sha256": actual_sha,
            "preparation_sha256": comparison["preparation_sha256"], "build_sha256": comparison["build_sha256"],
            "operators_sha256": comparison["operators_sha256"], "benchmark_build_sha256": benchmark_sha,
            "scope": "Accepted operator, smoke/mixed and allocation evidence; no speed qualification"}, bound


def compare_builds(control, candidate):
    rb.require(SOURCE_CHANGES and SOURCE_ADDITIONS, "Integration source allowlist not confirmed")
    rb.require(control["binary_sha256"] == CONTROL_EXE_SHA, "Pinned compact executable changed")
    rb.require(candidate["binary_sha256"] != control["binary_sha256"], "Candidate must be a distinct executable")
    old, new = control["source_sha256"], candidate["source_sha256"]
    rb.require(not (old.keys() - new.keys()), "Candidate deleted baseline sources")
    additions = sorted(new.keys() - old.keys())
    changed = sorted(name for name in old if new[name] != old[name])
    rb.require(additions == sorted(SOURCE_ADDITIONS) and changed == sorted(SOURCE_CHANGES),
               f"Unexpected integration source delta: changed={changed}, added={additions}")
    for name in ("rustc_version", "cargo_version"):
        rb.require(control[name] == candidate[name], "Build tool differs: " + name)
    rb.require(control["command"][control["command"].index("build"):] ==
               candidate["command"][candidate["command"].index("build"):], "Cargo build arguments differ")
    rb.require({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
               {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"},
               "Build environment differs beyond target")
    rb.require(Path(control["environment_overrides"]["CARGO_TARGET_DIR"]).resolve() !=
               Path(candidate["environment_overrides"]["CARGO_TARGET_DIR"]).resolve(), "Build targets must be isolated")
    return changed, additions


def runtime_check(workload):
    expected = {"warmup": 2, "threads": 16, "backend": "avx2", "precision": "fp32",
                "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}
    rb.require({k: workload["runtime"][k] for k in expected} == expected, "Fixed runtime changed")
    profile = [p for p in workload["profiles"] if p["name"] == "fullpages"]
    rb.require(len(profile) == 1 and profile[0]["order"][0] == "prose", "Fixed full-page selection changed")


def report_signatures(report, name, plan, workload, build):
    rb.require(report.get("os") == "windows" and report.get("arch") == "x86_64", "Actual report platform differs")
    expected_sources = {"model", "kernels", "runner", "preprocess", "packed_kernels", "config", "tokenizer", "trace", "lib", "harness"}
    rb.require(set(report["source_sha256"]) == expected_sources, "Embedded harness inventory changed")
    return rb.validate_report(report, job(name), plan, workload, build)


def validate(plan_path, expected_sha):
    plan_bytes = plan_path.read_bytes()
    rb.require(hashlib.sha256(plan_bytes).hexdigest() == expected_sha, "Plan hash mismatch")
    plan = json.loads(plan_bytes)
    rb.require(plan["kind"] == KIND, "Wrong plan kind")
    rb.require(plan["output_directory"] == str(plan_path.parent), "Plan moved")
    rb.require(plan["host_platform"] == platform.platform(), "Host changed")
    rb.require(plan["jobs"] == JOBS and plan["repetitions"] == 3, "Fixed bracket changed")
    rb.require(set(plan["builds"]) == {"control", "candidate"}, "Build labels changed")
    rb.require(plan["cache_layouts"] == LAYOUTS, "Explicit control/candidate layout mapping changed")
    rb.require(Path(plan["builds"]["control"]).resolve() == CONTROL and Path(plan["historical_baseline"]).resolve() == BASELINE,
               "Pinned control/baseline paths changed")
    rb.require(plan["source_changes"] == sorted(SOURCE_CHANGES) and plan["source_additions"] == sorted(SOURCE_ADDITIONS),
               "Integration source allowlist changed")
    qualification, qualification_bound = qualification_receipts(plan["builds"]["candidate"],
        plan["qualification"]["path"], plan["qualification"]["sha256"])
    rb.require(plan["qualification"] == qualification, "Qualification receipt joins changed")
    rb.require(plan["max_process_seconds"] == 900 and plan["control_drift_limit_percent"] == 5
               and plan["target_latency_reduction_percent"] == 5 and plan["default_promotion"] is False,
               "Benchmark limits changed")
    required = {str(Path(__file__).resolve()), str(ROOT / "research/benchmarks/scripts/realistic_benchmark.py"),
                str(plan_path.parent / "protocol-source.zip"), plan["workload"], plan["historical_baseline"],
                *plan["builds"].values()}
    required.update(qualification_bound)
    rb.require(all(plan["files_sha256"].get(p) == h for p, h in qualification_bound.items()),
               "Qualification artifact hashes missing from plan")
    for name, digest in plan["files_sha256"].items():
        rb.verify(name, digest)
    rb.require(Path(plan["workload"]).resolve() == rb.DEFAULT_WORKLOAD.resolve(), "Frozen workload path changed")
    workload, _ = checked_json(plan["workload"], WORKLOAD_SHA)
    runtime_check(workload)
    checked_json(CONTROL, CONTROL_SHA)
    baseline, _ = checked_json(BASELINE, BASELINE_SHA)
    rb.validate_inputs(workload)
    builds = {}
    for label, path in plan["builds"].items():
        builds[label] = rb.validate_build(path, check_current=False)
        build, binary = builds[label]
        required.update((str(binary), str(Path(path).parent / build["source_archive"])))
    rb.require(set(plan["files_sha256"]) == required, "Bound file inventory changed")
    changed, additions = compare_builds(builds["control"][0], builds["candidate"][0])
    rb.require(changed == plan["source_changes"] and additions == plan["source_additions"], "Actual source delta differs")
    rb.require(baseline_signatures(baseline, workload, builds["control"][0]) == plan["expected_signatures"], "Expected baseline outputs changed")
    return plan, workload, builds


def baseline_signatures(baseline, workload, control):
    baseline_plan = {"repetitions": 3, "cpu_label": "AMD Ryzen 9 7950X", "environment_label": "Native Windows"}
    return report_signatures(baseline, "historical-baseline", baseline_plan, workload, control)


def prepare(args):
    workload_path = rb.DEFAULT_WORKLOAD
    rb.require(args.control.resolve() == CONTROL and args.baseline.resolve() == BASELINE, "Use pinned combined-attention control and prior output")
    qualification, qualification_bound = qualification_receipts(args.candidate, args.qualification, args.qualification_sha256)
    checked_json(CONTROL, CONTROL_SHA)
    workload, _ = checked_json(workload_path, WORKLOAD_SHA)
    runtime_check(workload)
    rb.validate_inputs(workload)
    builds = {label: rb.validate_build(path, check_current=False)
              for label, path in (("control", args.control), ("candidate", args.candidate))}
    control, candidate = (builds[k][0] for k in ("control", "candidate"))
    changed, additions = compare_builds(control, candidate)
    baseline, _ = checked_json(args.baseline, BASELINE_SHA)
    rb.require(baseline["binary_sha256"] == control["binary_sha256"], "Baseline/control binary differs")
    expected = baseline["cases"][0]
    signatures = baseline_signatures(baseline, workload, control)
    rb.require(len(signatures) == 1 and signatures[0]["output_tokens"] == 1140
               and signatures[0]["finish_reason"] == "eos", "Wrong frozen page output")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), ROOT / "research/benchmarks/scripts/realistic_benchmark.py"]
    protocol = folder / "protocol-source.zip"
    with zipfile.ZipFile(protocol, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    bound = source_paths + [protocol, workload_path, args.baseline.resolve(), args.control.resolve(), args.candidate.resolve()]
    for path, build in ((args.control, control), (args.candidate, candidate)):
        bound += [path.parent / build["binary"], path.parent / build["source_archive"]]
    for name, digest in qualification_bound.items():
        rb.verify(name, digest)
    files_sha256 = {str(p.resolve()): rb.sha(p) for p in bound}
    rb.require(all(p not in files_sha256 or files_sha256[p] == h for p, h in qualification_bound.items()),
               "Qualification and benchmark artifacts conflict")
    files_sha256.update(qualification_bound)
    plan = {"kind": KIND, "created_utc": rb.utc(),
            "output_directory": str(folder), "host_platform": platform.platform(),
            "workload": str(workload_path), "cpu_label": "AMD Ryzen 9 7950X",
            "environment_label": "Native Windows", "repetitions": 3,
            "builds": {"control": str(args.control.resolve()), "candidate": str(args.candidate.resolve())},
            "source_changes": changed, "source_additions": additions, "cache_layouts": LAYOUTS, "expected_signatures": signatures,
            "qualification": qualification,
            "historical_baseline": str(args.baseline.resolve()),
            "historical_median_ms": expected["median_ms"],
            "files_sha256": files_sha256,
            "max_process_seconds": 900, "control_drift_limit_percent": 5,
            "target_latency_reduction_percent": 5, "default_promotion": False,
            "jobs": JOBS,
            "limitations": ["One full page: compact control versus explicit head-contiguous prefix candidate; B1 uses the single-page path, not joint multirow throughput.",
                           "Baseline contains fixed64 attention. Gains are incremental over that compact control and are not added to historical cache/attention percentages.",
                           "Whole-process memory high-water marks are not KV capacity measurements or per-phase peak memory.",
                           "Stage timings are wall intervals, not hardware counters.",
                           "Model load and file decode precede warmed measured recognitions."]}
    rb.write_new(folder / "plan.json", plan)
    print(json.dumps({"status": "planned", "plan": str(folder / "plan.json"), "sha256": rb.sha(folder / "plan.json")}))


def run(args):
    plan_path = args.plan.resolve()
    plan, workload, builds = validate(plan_path, args.plan_sha256)
    baseline, _ = checked_json(plan["historical_baseline"], BASELINE_SHA)
    rb.require(baseline_signatures(baseline, workload, builds["control"][0]) ==
               plan["expected_signatures"], "Frozen expected signatures differ")
    folder = plan_path.parent
    rb.require(args.quiet_attestation and args.quiet_attestation.strip(), "Quiet attestation required")
    rb.require(not any((folder / f"{item['name']}.{extension}").exists()
               for item in plan["jobs"] for extension in ("json", "log", "execution.json", "start.json"))
               and not any((folder / n).exists() for n in ("execution-start.json", "execution-failed.json", "comparison.json")),
               "Partial/completed bracket exists; create a fresh whole bracket")
    rb.write_new(folder / "execution-start.json", {"started_utc": rb.utc(), "plan_sha256": rb.sha(plan_path),
                 "quiet_attestation": args.quiet_attestation})
    reports = []
    output_bindings = {}
    try:
        for item in plan["jobs"]:
            build, binary = builds[item["build"]]
            name = item["name"]
            output = folder / f"{name}.json"
            command = command_for(dict(plan, binary=str(binary)), workload, name, output)
            execution = {"command": command, "cwd": str(ROOT), "started_utc": rb.utc()}
            print(f"Running {name}", flush=True)
            with (folder / f"{name}.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    execution["pid"] = child.pid
                    rb.write_new(folder / f"{name}.start.json", execution)
                    code = child.wait(timeout=plan["max_process_seconds"])
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
                    execution["timed_out"] = True
                except BaseException:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
                    raise
                execution.update(exit_code=code, finished_utc=rb.utc())
                rb.write_new(folder / f"{name}.execution.json", execution)
            rb.require(code == 0 and not execution.get("timed_out"), f"{name} failed")
            report, report_sha = checked_json(output)
            signatures = report_signatures(report, name, plan, workload, build)
            rb.require(signatures == plan["expected_signatures"], f"{name}: output differs from frozen baseline")
            case = report["cases"][0]
            output_bindings[str(output)] = report_sha
            for extension in ("log", "start.json", "execution.json"):
                path = folder / f"{name}.{extension}"
                output_bindings[str(path)] = rb.sha(path)
            reports.append({"name": name, "report_sha256": report_sha, "binary_sha256": build["binary_sha256"],
                            "cache_layout": LAYOUTS[item["build"]]["report"],
                            "all_measured_outputs_exact": True, "median_ms": case["median_ms"],
                            "samples_ms": [sample["wall_ms"] for sample in case["samples"]],
                            "stage_medians_ms": {key: statistics.median(s["per_request"][0]["timings"][key]
                                                  for s in case["samples"]) for key in
                                                 ("preprocessing_ms", "prefill_ms", "decode_ms", "total_ms")}})
        validate(plan_path, args.plan_sha256)  # Bulk checks only outside the measured process bracket.
        for name, digest in output_bindings.items():
            rb.verify(name, digest)
        before, candidate, after = [r["median_ms"] for r in reports]
        drift = 100 * (after / before - 1)
        gains = {"control_before": 100 * (1 - candidate / before), "control_after": 100 * (1 - candidate / after)}
        result = {"status": "complete", "finished_utc": rb.utc(), "plan_sha256": rb.sha(plan_path),
                  "reports": reports, "control_drift_percent": drift,
                  "control_stability_pass": abs(drift) <= plan["control_drift_limit_percent"],
                  "candidate_latency_reduction_percent": gains,
                  "target_met_in_this_page": abs(drift) <= plan["control_drift_limit_percent"]
                       and min(gains.values()) >= plan["target_latency_reduction_percent"],
                  "default_promotion": False, "historical_median_ms": plan["historical_median_ms"],
                  "output_artifact_sha256": output_bindings, "validated_outputs_unchanged": True}
        rb.write_new(folder / "comparison.json", result)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        rb.write_new(folder / "execution-failed.json", {"finished_utc": rb.utc(), "error": str(error), "completed_reports": reports})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--qualification-sha256")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet-attestation")
    args = parser.parse_args()
    if args.run:
        rb.require(args.plan and args.plan_sha256, "--run requires --plan and --plan-sha256")
        run(args)
    elif args.plan:
        rb.require(args.plan_sha256, "Validation requires --plan-sha256")
        validate(args.plan.resolve(), args.plan_sha256)
        print("Plan validated; no inference")
    else:
        rb.require(all((args.control, args.candidate, args.baseline, args.output,
                        args.qualification, args.qualification_sha256)),
                   "Planning needs both builds, baseline, output and accepted qualification/hash")
        prepare(args)


if __name__ == "__main__":
    main()
