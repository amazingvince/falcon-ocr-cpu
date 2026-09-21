#!/usr/bin/env python3
"""Compare matched same-binary CPU runs without treating one fixture as a release gate."""
import argparse
import hashlib
import json
import pathlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("artifacts/benchmarks"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference/benchmarks/windows-mode-comparison.json"))
    args = parser.parse_args()
    names = ["sequential-a", "joint-expanded", "joint-compact", "sequential-b"]
    paths = {name: args.directory / ("quiet-modes-" + name + ".json") for name in names}
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    controls = [records["sequential-a"], records["sequential-b"]]
    same = ["binary_sha256", "source_sha256", "cargo_lock_sha256", "model_revision", "weights_sha256",
            "images", "options", "threads", "backend", "precision", "warmup", "repetitions", "os", "arch"]
    for key in same:
        if any(record[key] != controls[0][key] for record in records.values()):
            raise ValueError(f"Comparison configuration differs: {key}")
    cases = {name: {case["batch_size"]: case for case in record["cases"]} for name, record in records.items()}
    if any(set(case) != set(cases["sequential-a"]) for case in cases.values()):
        raise ValueError("Compared batch groups differ")
    comparisons = []
    for batch in sorted(cases["sequential-a"]):
        first, last = cases["sequential-a"][batch], cases["sequential-b"][batch]
        for by_batch in cases.values():
            if by_batch[batch]["token_ids"] != first["token_ids"]:
                raise ValueError(f"Output tokens differ in batch {batch}")
        drift = 100 * (last["median_ms"] / first["median_ms"] - 1)
        row = {"batch_size": batch, "control_a_median_ms": first["median_ms"],
               "control_b_median_ms": last["median_ms"], "control_drift_percent": drift,
               "controls_within_five_percent": abs(drift) <= 5, "candidates": {}}
        for name in ["joint-expanded", "joint-compact"]:
            candidate = cases[name][batch]
            times = [sample["wall_ms"] for sample in candidate["samples"]]
            row["candidates"][name] = {"median_ms": candidate["median_ms"],
                "minimum_sample_ms": min(times), "maximum_sample_ms": max(times),
                "pages_per_second": candidate["median_pages_per_second"],
                "latency_reduction_vs_controls_percent": [100 * (1 - candidate["median_ms"] / control["median_ms"]) for control in [first, last]],
                "throughput_ratio_vs_controls": [control["median_ms"] / candidate["median_ms"] for control in [first, last]],
                "process_lifetime_memory_after": candidate["memory_after"]}
        row["compact_latency_reduction_vs_expanded_percent"] = 100 * (1 - cases["joint-compact"][batch]["median_ms"] / cases["joint-expanded"][batch]["median_ms"])
        comparisons.append(row)
    report = {"schema_version": 1, "source_reports": {name: {"path": str(paths[name]), "sha256": sha(paths[name])} for name in names},
        "execution_order": names, "configuration": {key: controls[0][key] for key in same},
        "cpu_label": controls[0]["cpu_label"], "environment_label": controls[0]["environment_label"],
        "sample_count_per_case": controls[0]["repetitions"], "all_output_token_ids_match": True,
        "all_controls_within_five_percent": all(row["controls_within_five_percent"] for row in comparisons),
        "cases": comparisons, "limitations": [
            "A small repeated-image fixture measures this workload only; real full-page/mixed-length benchmarks remain required.",
            "Project CPU/GPU jobs were paused. Other interactive applications were not controlled, so this is not a dedicated host measurement.",
            "Candidate modes run between repeated controls; retained samples and control drift expose some temporal variation but do not establish a statistical confidence interval.",
            "Memory counters are process-lifetime high-water marks, not isolated per-case peaks.",
            "Compact is still opt-in; this report alone does not promote a backend or establish the full performance acceptance gate."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_tokens_match": True,
        "stable_controls": report["all_controls_within_five_percent"], "cases": comparisons}, indent=2))


if __name__ == "__main__":
    main()
