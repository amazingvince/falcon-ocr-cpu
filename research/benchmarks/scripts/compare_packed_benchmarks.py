#!/usr/bin/env python3
"""Compare repeated packed/unpacked runs from the same native benchmark binary."""
import argparse
import hashlib
import json
import pathlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("artifacts/benchmarks"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference/benchmarks/windows-packed-comparison.json"))
    args = parser.parse_args()
    names = ["unpacked-a", "packed-a", "packed-b", "unpacked-b"]
    paths = {name: args.directory / ("quiet-packed-" + name + ".json") for name in names}
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    same = ["binary_sha256", "source_sha256", "cargo_lock_sha256", "model_revision", "weights_sha256",
            "images", "options", "threads", "backend", "precision", "warmup", "repetitions", "os", "arch", "cache_layout"]
    control = records["unpacked-a"]
    for key in same:
        if any(record[key] != control[key] for record in records.values()):
            raise ValueError(f"Comparison configuration differs: {key}")
    cases = {name: {case["batch_size"]: case for case in record["cases"]} for name, record in records.items()}
    if any(set(case) != set(cases["unpacked-a"]) for case in cases.values()):
        raise ValueError("Compared batch groups differ")
    for name, record in records.items():
        expected = "unpacked" if name.startswith("unpacked") else "phase_packed"
        if record["weight_layout"] != expected:
            raise ValueError(f"Unexpected weight layout: {name}")
    rows = []
    for batch in sorted(cases["unpacked-a"]):
        first, last = cases["unpacked-a"][batch], cases["unpacked-b"][batch]
        for by_batch in cases.values():
            case = by_batch[batch]
            for key in ["execution", "image_indices", "token_ids"]:
                if case[key] != first[key]:
                    raise ValueError(f"Batch {batch} differs: {key}")
        drift = 100 * (last["median_ms"] / first["median_ms"] - 1)
        candidates = {}
        for name in ["packed-a", "packed-b"]:
            case = cases[name][batch]
            candidates[name] = {"median_ms": case["median_ms"],
                "pages_per_second": case["median_pages_per_second"],
                "latency_reduction_vs_controls_percent": [100 * (1 - case["median_ms"] / ref["median_ms"]) for ref in [first, last]],
                "throughput_ratio_vs_controls": [ref["median_ms"] / case["median_ms"] for ref in [first, last]],
                "minimum_sample_ms": min(sample["wall_ms"] for sample in case["samples"]),
                "maximum_sample_ms": max(sample["wall_ms"] for sample in case["samples"])}
        rows.append({"batch_size": batch, "control_medians_ms": [first["median_ms"], last["median_ms"]],
                     "control_drift_percent": drift, "candidates": candidates})
    report = {"schema_version": 1, "execution_order": names,
        "configuration": {key: control[key] for key in same},
        "source_reports": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "all_output_token_ids_match": True,
        "controls_within_five_percent": all(abs(row["control_drift_percent"]) <= 5 for row in rows),
        "cases": rows,
        "mode_costs": {name: {"packed_weight_bytes": record["packed_weight_bytes"],
            "weight_packing_ms": record["weight_packing_ms"],
            "process_peak_resident_bytes": max(case["memory_after"]["peak_resident_bytes"] for case in record["cases"])}
            for name, record in records.items()},
        "limitations": ["Repeated small image only; representative full-page and mixed-length benchmarks remain required.",
            "Project CPU/GPU jobs paused; interactive applications remain open.",
            "Controls bracket two candidate runs; samples retained, no statistical confidence interval claimed.",
            "Peak memory is a whole-process high-water mark, not isolated per-case memory.",
            "Packed is opt-in and adds a shared weight copy. This report alone does not promote it."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stable_controls": report["controls_within_five_percent"], "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
