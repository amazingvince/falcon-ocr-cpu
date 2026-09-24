#!/usr/bin/env python3
"""Validate frozen full-page benchmark evidence and compare bracketed controls."""
import argparse
import pathlib
import statistics
from realistic_benchmark import read, write_new, require, sha, validate_plan, validate_report


def compare(plan_path):
    plan_path = pathlib.Path(plan_path).resolve()
    plan, workload, build = validate_plan(plan_path, current=False, inputs=False)
    folder = plan_path.parent
    start = read(folder / "execution-start.json")
    end = read(folder / "execution-complete.json")
    require(start["plan_sha256"] == end["plan_sha256"] == sha(plan_path), "Execution is not bound to this plan")
    require(start.get("quiet_attestation"), "Missing quiet-host attestation")
    receipts = {p["job"]: p["sha256"] for p in end["reports"]}
    require(len(receipts) == len(end["reports"]) == len(plan["jobs"]), "Incomplete/duplicate report receipts")
    grouped = {}
    host_identity = None
    for job in plan["jobs"]:
        path = folder / f"{job['id']}.json"
        require(sha(path) == receipts.get(job["id"]), f"Report changed since execution: {path}")
        report = read(path)
        signature = validate_report(report, job, plan, workload, build)
        host = {k: report[k] for k in ("os", "arch", "logical_cpus", "cpu_label", "environment_label")}
        if host_identity is None:
            host_identity = host
        require(host == host_identity, "Host identity differs across modes")
        group = grouped.setdefault((job["profile"], job["batch"]), {})
        group[job["mode"]["name"]] = (report, signature)
    results = []
    policy = workload["comparison"]
    for (profile, batch), modes in grouped.items():
        baseline = modes["sequential-a"][1]
        require(all(signature == baseline for _, signature in modes.values()), f"{profile}/batch{batch}: tokens/text/stops/request order differ across modes")
        median = {k: v[0]["cases"][0]["median_ms"] for k, v in modes.items()}
        drift = {kind: 100 * (median[f"{kind}-b"] / median[f"{kind}-a"] - 1) for kind in ("sequential", "expanded")}
        stable = all(abs(x) <= policy["maximum_absolute_control_drift_percent"] for x in drift.values())
        candidates = {}
        for candidate, controls in {"expanded-a": ["sequential-a", "sequential-b"],
                                    "expanded-b": ["sequential-a", "sequential-b"],
                                    "compact": ["expanded-a", "expanded-b"],
                                    "phase-packed": ["expanded-a", "expanded-b"]}.items():
            reductions = {c: 100 * (1 - median[candidate] / median[c]) for c in controls}
            candidates[candidate] = {"latency_reduction_percent_vs_controls": reductions,
                "throughput_multiplier_vs_controls": {c: median[c] / median[candidate] for c in controls},
                "meets_target_in_this_group": stable and min(reductions.values()) >= policy["minimum_target_latency_reduction_percent"],
                "no_primary_regression_in_this_group": stable and min(reductions.values()) >= -policy["maximum_primary_latency_regression_percent"]}
        stage_keys = ("preprocessing_ms", "image_projection_ms", "transformer_prefill_ms", "prefill_ms", "decode_ms", "time_to_first_token_ms", "total_ms")
        mode_reports = {}
        for name, (report, _) in modes.items():
            case = report["cases"][0]
            mode_reports[name] = {"median_ms": median[name], "median_pages_per_second": case["median_pages_per_second"],
                "sample_wall_ms": [s["wall_ms"] for s in case["samples"]],
                "per_request_stage_medians_ms": [{key: statistics.median(s["per_request"][i]["timings"][key] for s in case["samples"]) for key in stage_keys} for i in range(batch)],
                "loaded_memory": report["loaded_memory"], "process_memory_after": case["memory_after"],
                "read_decode_ms": report["read_decode_ms"], "verified_model_load_ms": report["verified_model_load_ms"],
                "packed_weight_bytes": report["packed_weight_bytes"], "weight_packing_ms": report["weight_packing_ms"]}
        results.append({"profile": profile, "batch": batch, "request_keys": next(j["request_keys"] for j in plan["jobs"] if j["profile"] == profile and j["batch"] == batch),
            "outputs_exact_across_all_modes_and_repetitions": True,
            "output_lengths": [len(s["token_ids"]) for s in baseline], "finish_reasons": [s["finish_reason"] for s in baseline],
            "input_tokens": [s["input_tokens"] for s in baseline], "control_drift_percent": drift,
            "control_stability_pass": stable, "candidates": candidates, "modes": mode_reports})
    return {"schema_version": 1, "kind": "realistic-fp32-benchmark-comparison", "plan_sha256": sha(plan_path),
            "workload_sha256": plan["workload_sha256"], "build_manifest_sha256": plan["build_manifest_sha256"],
            "binary_sha256": plan["binary_sha256"], "source_archive_sha256": plan["source_archive_sha256"],
            "coverage": plan["coverage"], "host": host_identity, "quiet_attestation": start["quiet_attestation"],
            "default_promotion": False, "groups": results,
            "limitations": ["Control stability and group-specific speed thresholds do not establish full-model numerical parity or OCR accuracy.",
                "Joint decode_ms is elapsed shared decode time until that request finishes, not exclusive CPU time; never sum it across requests.",
                "Joint TTFT/total include preceding prefills/shared work and use chunk start, while sequential request timings have different origins; batch wall time is the comparable primary metric.",
                "Peak RSS/commit is a process-lifetime high-water mark including loading/warmup, even with one batch per process. Resident snapshots are not isolated KV allocation measurements.",
                "Image projection and transformer prefill are subintervals of prefill_ms, not additive extra work. Their medians need not sum to the median enclosing time. No per-layer, memory-bandwidth, hardware-counter or device-energy measurement is available.",
                "RGB read/decode, verified model load and one-time packing are separately reported; warm sample wall time includes preprocessing, prefill, decode and text decoding.",
                "Native Windows, WSL and bare-metal Linux reports are separate environments; these results do not transfer between them."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite comparison")
    report = compare(args.plan)
    write_new(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
