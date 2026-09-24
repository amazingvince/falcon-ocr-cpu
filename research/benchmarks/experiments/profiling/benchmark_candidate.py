#!/usr/bin/env python3
"""One isolated expanded-cache candidate between two full-page CPU controls."""
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


def job(name):
    return {"id": name, "profile": "fullpages", "batch": 1,
            "mode": {"execution": "sequential", "cache_layout": "expanded", "weight_layout": "unpacked"},
            "image_keys": ["prose"], "request_keys": ["prose"]}


def validate(plan_path, expected_sha):
    plan_bytes = plan_path.read_bytes()
    rb.require(hashlib.sha256(plan_bytes).hexdigest() == expected_sha, "Plan hash mismatch")
    plan = json.loads(plan_bytes)
    rb.require(plan["kind"] == "single-page-isolated-candidate-bracket-v1", "Wrong plan kind")
    rb.require(plan["output_directory"] == str(plan_path.parent), "Plan moved")
    rb.require(plan["host_platform"] == platform.platform(), "Host changed")
    rb.require(plan["jobs"] == JOBS and plan["repetitions"] == 3, "Fixed bracket changed")
    rb.require(set(plan["builds"]) == {"control", "candidate"}, "Build labels changed")
    rb.require(plan["max_process_seconds"] == 900 and plan["control_drift_limit_percent"] == 5
               and plan["target_latency_reduction_percent"] == 5 and plan["default_promotion"] is False,
               "Benchmark limits changed")
    required = {str(Path(__file__).resolve()), str(ROOT / "research/benchmarks/scripts/realistic_benchmark.py"),
                str(plan_path.parent / "protocol-source.zip"), plan["workload"], plan["historical_baseline"],
                *plan["builds"].values()}
    for name, digest in plan["files_sha256"].items():
        rb.verify(name, digest)
    workload = rb.read(plan["workload"])
    rb.require(workload["runtime"]["warmup"] == 2 and workload["runtime"]["threads"] == 16
               and workload["runtime"]["backend"] == "avx2", "Runtime changed")
    rb.validate_inputs(workload)
    builds = {}
    for label, path in plan["builds"].items():
        builds[label] = rb.validate_build(path, check_current=False)
        build, binary = builds[label]
        required.update((str(binary), str(Path(path).parent / build["source_archive"])))
    rb.require(set(plan["files_sha256"]) == required, "Bound file inventory changed")
    return plan, workload, builds


def baseline_signatures(baseline, workload, control):
    baseline_plan = {"repetitions": 3, "cpu_label": "AMD Ryzen 9 7950X", "environment_label": "Native Windows"}
    return rb.validate_report(baseline, job("historical-baseline"), baseline_plan, workload, control)


def prepare(args):
    workload_path = rb.DEFAULT_WORKLOAD
    workload = rb.read(workload_path)
    rb.validate_inputs(workload)
    builds = {label: rb.validate_build(path, check_current=False)
              for label, path in (("control", args.control), ("candidate", args.candidate))}
    control, candidate = (builds[k][0] for k in ("control", "candidate"))
    for name in ("rustc_version", "cargo_version"):
        rb.require(control[name] == candidate[name], f"Build tool differs: {name}")
    # The copied project and fresh Cargo target necessarily have different paths.
    for build in (control, candidate):
        rb.require("build" in build["command"], "Missing Cargo build command")
    rb.require(control["command"][control["command"].index("build"):] ==
               candidate["command"][candidate["command"].index("build"):], "Cargo build arguments differ")
    rb.require({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
               {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"},
               "Build environment differs beyond isolated target directory")
    rb.require(control["source_sha256"].keys() == candidate["source_sha256"].keys(), "Source inventory changed")
    changed = [name for name in control["source_sha256"]
               if control["source_sha256"][name] != candidate["source_sha256"][name]]
    rb.require(changed == ["src/kernels.rs"], f"Expected isolated kernel source difference, got {changed}")
    rb.require(workload["runtime"]["warmup"] == 2, "Expected baseline two warmups")
    rb.require(workload["runtime"]["threads"] == 16 and workload["runtime"]["backend"] == "avx2", "Runtime changed")
    baseline = rb.read(args.baseline)
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
    plan = {"kind": "single-page-isolated-candidate-bracket-v1", "created_utc": rb.utc(),
            "output_directory": str(folder), "host_platform": platform.platform(),
            "workload": str(workload_path), "cpu_label": "AMD Ryzen 9 7950X",
            "environment_label": "Native Windows", "repetitions": 3,
            "builds": {"control": str(args.control.resolve()), "candidate": str(args.candidate.resolve())},
            "source_changes": changed, "expected_signatures": signatures,
            "historical_baseline": str(args.baseline.resolve()),
            "historical_median_ms": expected["median_ms"],
            "files_sha256": {str(p.resolve()): rb.sha(p) for p in bound},
            "max_process_seconds": 900, "control_drift_limit_percent": 5,
            "target_latency_reduction_percent": 5, "default_promotion": False,
            "jobs": JOBS,
            "limitations": ["One full page, one candidate, expanded cache only; no batch/platform promotion.",
                           "Stage timings are wall intervals, not hardware counters.",
                           "Model load and file decode precede warmed measured recognitions."]}
    rb.write_new(folder / "plan.json", plan)
    print(json.dumps({"status": "planned", "plan": str(folder / "plan.json"), "sha256": rb.sha(folder / "plan.json")}))


def run(args):
    plan_path = args.plan.resolve()
    plan, workload, builds = validate(plan_path, args.plan_sha256)
    rb.require(baseline_signatures(rb.read(plan["historical_baseline"]), workload, builds["control"][0]) ==
               plan["expected_signatures"], "Frozen expected signatures differ")
    folder = plan_path.parent
    rb.require(args.quiet_attestation and args.quiet_attestation.strip(), "Quiet attestation required")
    rb.require(not any((folder / f"{item['name']}.{extension}").exists()
               for item in plan["jobs"] for extension in ("json", "log", "execution.json")), "Partial bracket exists")
    rb.write_new(folder / "execution-start.json", {"started_utc": rb.utc(), "plan_sha256": rb.sha(plan_path),
                 "quiet_attestation": args.quiet_attestation})
    reports = []
    try:
        for item in plan["jobs"]:
            build, binary = builds[item["build"]]
            name = item["name"]
            output = folder / f"{name}.json"
            command = rb.command_for(dict(plan, binary=str(binary)), workload, job(name), output)
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
            report = rb.read(output)
            signatures = rb.validate_report(report, job(name), plan, workload, build)
            rb.require(signatures == plan["expected_signatures"], f"{name}: output differs from frozen baseline")
            case = report["cases"][0]
            reports.append({"name": name, "report_sha256": rb.sha(output), "binary_sha256": build["binary_sha256"],
                            "all_measured_outputs_exact": True, "median_ms": case["median_ms"],
                            "samples_ms": [sample["wall_ms"] for sample in case["samples"]],
                            "stage_medians_ms": {key: statistics.median(s["per_request"][0]["timings"][key]
                                                  for s in case["samples"]) for key in
                                                 ("preprocessing_ms", "prefill_ms", "decode_ms", "total_ms")}})
        validate(plan_path, args.plan_sha256)  # Bulk checks only outside the measured process bracket.
        before, candidate, after = [r["median_ms"] for r in reports]
        drift = 100 * (after / before - 1)
        gains = {"control_before": 100 * (1 - candidate / before), "control_after": 100 * (1 - candidate / after)}
        result = {"status": "complete", "finished_utc": rb.utc(), "plan_sha256": rb.sha(plan_path),
                  "reports": reports, "control_drift_percent": drift,
                  "control_stability_pass": abs(drift) <= plan["control_drift_limit_percent"],
                  "candidate_latency_reduction_percent": gains,
                  "target_met_in_this_page": abs(drift) <= plan["control_drift_limit_percent"]
                       and min(gains.values()) >= plan["target_latency_reduction_percent"],
                  "default_promotion": False, "historical_median_ms": plan["historical_median_ms"]}
        rb.write_new(folder / "comparison.json", result)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        rb.write_new(folder / "execution-failed.json", {"finished_utc": rb.utc(), "error": str(error), "completed_reports": reports})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
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
        rb.require(all((args.control, args.candidate, args.baseline, args.output)), "Planning needs both builds, baseline, output")
        prepare(args)


if __name__ == "__main__":
    main()
