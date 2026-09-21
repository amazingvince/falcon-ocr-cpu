#!/usr/bin/env python3
"""Reject signal-scale BF16 calibration without modifying its immutable evidence."""
import json
import pathlib
from fetch_reference import sha256

root = pathlib.Path(__file__).resolve().parents[1]
policy_path = root / "reference/tolerances-smoke-bf16-v1.json"
policy = json.loads(policy_path.read_text())
stages = []
for name, value in policy["stages"].items():
    peak = value["reference_max_abs"]
    stages.append({"stage": name, **value,
                   "observed_error_to_peak": value["baseline_max_abs"] / peak if peak else 0,
                   "tolerance_to_peak": value["absolute_tolerance"] / peak if peak else 0})
stages.sort(key=lambda x: x["observed_error_to_peak"], reverse=True)
report = {
    "schema_version": 1, "policy": str(policy_path.relative_to(root)),
    "policy_sha256": sha256(policy_path), "qualification_accepted": False,
    "disposition": "Immutable diagnostic evidence only; must not qualify CPU BF16 model parity",
    "reason": "Dense global softmax BF16 rounding and Flex block-local unnormalized exponential BF16 rounding are different numerical algorithms. Their accumulated full-graph difference approaches signal scale and produces uninformative acceptance bounds.",
    "stages": len(stages), "stages_with_tolerance_at_least_peak": sum(x["tolerance_to_peak"] >= 1 for x in stages),
    "worst_stages": stages[:20],
    "required_next_step": "Read actual compiled tile metadata; independently emulate block-local running max, rescale, exp2, BF16 probability cast, FP32 accumulation and denominator, final BF16 cast, then sink scaling. Validate equal-input isolated operators before considering a new policy.",
    "scope": "No CPU BF16 candidate was used to set the rejected policy or this review. FP32 frozen policy is unchanged.",
}
target = root / "reference/bf16-policy-review.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "worst_stages"}, indent=2))
