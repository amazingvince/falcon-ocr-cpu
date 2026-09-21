"""Summarize the two native Windows operator-only measurement rounds."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def group_cases(cases):
    groups = []
    for rows in sorted({case["shape"]["rows"] for case in cases}):
        selected = [case for case in cases if case["shape"]["rows"] == rows]
        rounds = []
        for iteration in range(2):
            before = sum(case["rounds"][iteration]["baseline_median_ms"] for case in selected)
            after = sum(case["rounds"][iteration]["candidate_median_ms"] for case in selected)
            rounds.append({"baseline_sum_medians_ms": before,
                           "candidate_sum_medians_ms": after, "speedup": before / after})
        ratios = [case["mean_median_speedup"] for case in selected]
        groups.append({"rows": rows, "operator_cases": len(selected), "rounds": rounds,
                       "geometric_mean_speedup": math.exp(mean(map(math.log, ratios))),
                       "median_speedup": median(ratios),
                       "min_speedup": min(ratios), "max_speedup": max(ratios),
                       "cases_faster_in_both_rounds": sum(
                           all(item["speedup"] > 1.0 for item in case["rounds"])
                           for case in selected)})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("reference/operator-windows-summary.json"))
    args = parser.parse_args()
    records, sources = {}, []
    for name in ["fp32-baseline", "fp32-packed", "bf16-avx512"]:
        for iteration in "ab":
            key = f"{name}-{iteration}"
            record, source = load(args.directory / f"operator-windows-{key}.json")
            assert record["target_os"] == "windows" and record["threads"] == 16
            assert record["debug_assertions"] is False
            records[key] = record
            sources.append(source)
    fp32, bf16 = [], []
    base_maps = [{case["operator"]: case for case in records[f"fp32-baseline-{r}"]["operators"]}
                 for r in "ab"]
    packed_maps = [{case["operator"]: case for case in records[f"fp32-packed-{r}"]["operators"]}
                   for r in "ab"]
    bf16_maps = [{case["operator"]: case for case in records[f"bf16-avx512-{r}"]["operators"]}
                 for r in "ab"]
    packing_maps = [{item["weight_key"]: item for item in records[f"fp32-packed-{r}"]["packing"]}
                    for r in "ab"]
    for name, item in packed_maps[0].items():
        comparisons = []
        for index in range(2):
            baseline, candidate = base_maps[index][name], packed_maps[index][name]
            assert baseline["shape"] == candidate["shape"]
            assert candidate["phase_packed_vs_production_avx2"]["different_bits"] == 0
            a, b = baseline["timing"], candidate["timing"]
            comparisons.append({"baseline_median_ms": a["median_ms"],
                                "candidate_median_ms": b["median_ms"],
                                "speedup": a["median_ms"] / b["median_ms"],
                                "baseline_p95_sample_mean_ms": percentile(a["samples_ms"], .95),
                                "candidate_p95_sample_mean_ms": percentile(b["samples_ms"], .95)})
        before = mean(item["baseline_median_ms"] for item in comparisons)
        after = mean(item["candidate_median_ms"] for item in comparisons)
        packing_ms = mean(mapping[item["weight_key"]]["packing_ms"] for mapping in packing_maps)
        fp32.append({"operator": name, "shape": item["shape"], "rounds": comparisons,
                     "mean_median_speedup": before / after,
                     "mean_packing_ms": packing_ms,
                     "estimated_break_even_operator_calls": math.ceil(packing_ms / (before - after))
                     if before > after else None})
    for name, item in bf16_maps[0].items():
        comparisons = []
        for mapping in bf16_maps:
            a, b = mapping[name]["timing"]["direct"], mapping[name]["timing"]["packed"]
            comparisons.append({"baseline_median_ms": a["median_ms"],
                                "candidate_median_ms": b["median_ms"],
                                "speedup": a["median_ms"] / b["median_ms"],
                                "baseline_p95_sample_mean_ms": percentile(a["samples_ms"], .95),
                                "candidate_p95_sample_mean_ms": percentile(b["samples_ms"], .95)})
        bf16.append({"operator": name, "shape": item["shape"], "rounds": comparisons,
                     "mean_median_speedup": mean(v["baseline_median_ms"] for v in comparisons)
                     / mean(v["candidate_median_ms"] for v in comparisons)})
    packing = {}
    for precision in ["fp32-packed", "bf16-avx512"]:
        packing[precision] = [{"round": r,
                               "unique_weights": len(records[f"{precision}-{r}"]["packing"]),
                               "packed_tensor_bytes": sum(item["packed_tensor_bytes"] for item in records[f"{precision}-{r}"]["packing"]),
                               "packing_ms": sum(item["packing_ms"] for item in records[f"{precision}-{r}"]["packing"])}
                              for r in "ab"]
    result = {"schema_version": 1, "sources": sources,
              "platform": "native Windows x86_64, 16 Rayon threads",
              "round_order": ["fp32-baseline-a", "fp32-packed-a", "bf16-avx512-a",
                              "bf16-avx512-b", "fp32-packed-b", "fp32-baseline-b"],
              "configuration": {key: value["timing_configuration"] for key, value in records.items()},
              "interpretation": "Hot isolated operators only; matrices and operands already resident. Extra packed memory and one-time packing are separate. P95 values describe per-sample means, not request tail latency. Break-even counts are operator-only estimates. No model backend is promoted.",
              "scope": {"fp32_rows_1": "QKV, WO, W13, W2; actual GPU fixture operands",
                        "fp32_rows_2_4_8": "QKV and W13 only; existing fixture does not expose short WO/W2 rows",
                        "bf16_rows_1_2_4_8_32": "QKV, WO, W13, W2; BF16 operands with FP32 output",
                        "vocabulary_projection": "not included"},
              "packing": packing,
              "fp32_phase_packed_vs_current_avx2": {"groups": group_cases(fp32), "operators": fp32},
              "bf16_output_packed_vs_direct_avx512bf16": {"groups": group_cases(bf16), "operators": bf16}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packing": packing, "fp32": group_cases(fp32), "bf16": group_cases(bf16)}, indent=2))


if __name__ == "__main__":
    main()
