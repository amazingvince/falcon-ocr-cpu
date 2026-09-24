#!/usr/bin/env python3
"""Record where independently rounded BF16 GPU trajectories first diverge."""
import json
import pathlib
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

root = pathlib.Path(__file__).resolve().parents[3]
paths = [root / f"artifacts/reference/{name}/trace.safetensors" for name in ["smoke-bf16", "smoke-blockwise-bf16"]]
reference, candidate = [load_file(str(p)) for p in paths]
report_path = root / "reference/gpu-blockwise-smoke-bf16.json"
report = json.loads(report_path.read_text())
layers = []
for i in range(22):
    record = {"layer": i}
    for name in ["q", "k", "v", "attention", "hidden"]:
        key = f"layer.{i}.{name}"
        a, b = reference[key].float(), candidate[key].float()
        delta = (a-b).abs().reshape(a.shape[0], -1)
        row_error = delta.amax(1)
        top = row_error.topk(5)
        record[name] = {"elements": a.numel(), "different_elements": int((a != b).sum()),
                        "max_abs": float(delta.max()), "rms_abs": float(delta.square().mean().sqrt()),
                        "peak": float(a.abs().max()),
                        "worst_rows": [{"row": int(row), "token_id": int(reference["tokens"][row]), "max_abs": float(value)}
                                       for row, value in zip(top.indices, top.values)]}
    layers.append(record)
ratios = [{"stage": name, **stage, "error_to_peak": stage["baseline_max_abs"] / stage["reference_max_abs"] if stage["reference_max_abs"] else 0}
          for name, stage in report["stage_aggregates"].items()]
ratios.sort(key=lambda x: x["error_to_peak"], reverse=True)
out = {"schema_version": 1, "reference_sha256": sha256(paths[0]), "blockwise_sha256": sha256(paths[1]),
       "comparison_sha256": sha256(report_path), "new_full_model_policy_created": False,
       "all_17_argmax_match": len(report["logit_decisions"]) == 17 and all(x["argmax_matches"] for x in report["logit_decisions"].values()),
       "qualification": "Full-graph calibration rejected: sparse local BF16 rounding differences still amplify to signal-scale hidden differences. Use independent local operator checks and separate output/corpus quality evaluation; this report does not qualify a BF16 CPU model.",
       "worst_stage_ratios": ratios[:20], "prefill_layers": layers}
target = root / "reference/bf16-blockwise-trajectory-review.json"
target.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"output": str(target), "worst": ratios[0], "first_layer": layers[0], "last_layer": layers[-1]}, indent=2))
