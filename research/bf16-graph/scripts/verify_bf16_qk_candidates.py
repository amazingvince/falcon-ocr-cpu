#!/usr/bin/env python3
"""Cross-check the Rust diagnostic with the unchanged GPU-team comparator."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from compare_bf16_local_tensors import compare
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

root = Path(__file__).resolve().parents[3]
torch.set_num_threads(2)
policy_path = root / "reference/bf16-local-contract-v1.json"
policy = json.loads(policy_path.read_text())
for path, expected in policy["sources"].items():
    assert sha256(root / path) == expected, path
input_path = root / "artifacts/reference/attention-operators-bf16.safetensors"
source = load_file(str(input_path))
bound = load_file(str(root / "artifacts/reference/bf16-local-contract-v1.safetensors"))
tensor_path = root / "artifacts/cpu/bf16-qk-reduction-candidates.safetensors"
candidates = load_file(str(tensor_path))
report_path = root / "reference/bf16-qk-reduction-candidates.json"
rust_report = json.loads(report_path.read_text())
assert rust_report["contract_sha256"] == sha256(policy_path)
assert rust_report["fixture_sha256"] == sha256(input_path)
records = []
for candidate in rust_report["candidates"]:
    checks = {}
    for item in policy["attention_cases"]:
        for stage in ["raw", "scaled", "lse"]:
            key = item["name"] + "." + stage
            actual = candidates[candidate["name"] + "." + key]
            if stage == "lse":
                expected = source[key]
                result = compare(expected, actual, torch.full_like(expected, item["lse_max_abs_bound"]), require_bf16=False)
            else:
                gate = item[stage]
                result = compare(source[gate["expected_key"]], actual, bound[gate["bound_key"]], gate["rms_bound"])
            rust = candidate["checks"][key]
            for metric in ["passed", "elements", "different_elements", "bound_violations", "max_abs"]:
                assert rust[metric] == result[metric], (candidate["name"], key, metric)
            assert abs(rust["rms_abs"] - result["rms_abs"]) < 1e-13
            checks[key] = result
    records.append({"name": candidate["name"], "passed_gates": sum(c["passed"] for c in checks.values()),
                    "element_bound_violations": sum(c["bound_violations"] for c in checks.values()), "checks": checks})
report = {"rust_and_python_assessments_match": True, "bounds_changed": False,
          "contract_sha256": sha256(policy_path), "tensor_export_sha256": sha256(tensor_path),
          "rust_report_sha256": sha256(report_path), "verification_script_sha256": sha256(Path(__file__)),
          "assessment_script_sha256": sha256(root / "research/bf16-graph/scripts/compare_bf16_local_tensors.py"), "candidates": records}
(root / "reference/bf16-qk-reduction-independent-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"rust_and_python_assessments_match": True, "candidates": [{k: v for k, v in c.items() if k != "checks"} for c in records]}, indent=2))
