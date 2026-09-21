#!/usr/bin/env python3
"""Record concrete immutable-bound failures for kernel investigation."""
import json
import pathlib
import torch
from safetensors.torch import load_file
from fetch_reference import sha256

root = pathlib.Path(__file__).resolve().parents[1]
contract_path = root / "reference/bf16-local-contract-v1.json"
policy = json.loads(contract_path.read_text())
bounds = load_file(str(root / "artifacts/reference/bf16-local-contract-v1.safetensors"))
linear = load_file(str(root / "artifacts/reference/bf16-operators.safetensors"))
attention = load_file(str(root / "artifacts/reference/attention-operators-bf16.safetensors"))
oracle_attention = load_file(str(root / "artifacts/reference/attention-blockwise-bf16.safetensors"))
records = []
for backend in ["avx512", "scalar"]:
    for kind in ["linear", "attention"]:
        artifact = root / f"artifacts/cpu/bf16-native-{kind}-{backend}.safetensors"
        candidate = load_file(str(artifact))
        assessment = json.loads((root / f"reference/bf16-native-{kind}-assessment-{backend}.json").read_text())
        for key in assessment["failures"]:
            if key.startswith("provenance"):
                continue
            metric = assessment["checks"][key]
            name, variant = key.rsplit(".", 1)
            if kind == "linear":
                entry = next(x for x in policy["linear_cases"] if x["name"] == name)
                expected = bounds[entry["native_expected_key"]]
                bound = bounds[entry["native_bound_key"]]
            else:
                entry = next(x for x in policy["attention_cases"] if x["name"] == name)
                expected = attention[entry[variant]["expected_key"]]
                bound = bounds[entry[variant]["bound_key"]]
            for flat in metric["first_violation_indices"]:
                index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), expected.shape))
                value, actual, limit = float(expected[index]), float(candidate[key][index]), float(bound[index])
                record = {"backend": backend, "kind": kind, "case": key, "index": list(index), "flat_index": flat,
                          "gpu": value, "cpu": actual, "bound": limit, "error": abs(value-actual),
                          "error_to_bound": abs(value-actual)/limit}
                if kind == "linear":
                    row, out = index
                    x, w = linear[name + ".input"][row].double(), linear[entry["weight_key"]][out].double()
                    exact = torch.dot(x, w)
                    record.update({"f64_dot": float(exact), "f64_rounded_bf16": float(exact.bfloat16()),
                                   "gpu_fp32_accumulation": float(linear[name + ".expected"][index])})
                else:
                    record["independent_blockwise_oracle"] = float(oracle_attention[name + "." + variant][index])
                records.append(record)
result = {"contract_sha256": sha256(contract_path), "bound_changes": False, "violating_elements": records,
          "qualification": "Exact locations and values only; numerical failures remain failures"}
(root / "reference/bf16-local-failure-details.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
