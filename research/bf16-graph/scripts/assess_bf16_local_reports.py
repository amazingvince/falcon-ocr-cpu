#!/usr/bin/env python3
"""Assess recorded Rust operators against frozen GPU-only bounds, without edits."""
import json
import pathlib
from safetensors.torch import load_file
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

root = pathlib.Path(__file__).resolve().parents[3]
policy_path = root / "reference/bf16-local-contract-v1.json"
policy = json.loads(policy_path.read_text())
for name, digest in policy["sources"].items():
    assert sha256(root/name) == digest, name
bounds = load_file(str(root / "artifacts/reference/bf16-local-contract-v1.safetensors"))
linear_policy = {x["name"]: x for x in policy["linear_cases"]}
matrix = []
for basename in ["bf16-isolated-probe-avx512.json", "bf16-isolated-probe-scalar.json"]:
    path = root / "reference" / basename
    report = json.loads(path.read_text())
    assert report["fixture_sha256"] == policy["sources"]["artifacts/reference/bf16-operators.safetensors"]
    assert report["kernel_source_sha256"] == sha256(root / "src/bf16_kernels.rs")
    assert set(x["operator"] for x in report["operators"]) == set(linear_policy)
    cases = []
    for op in report["operators"]:
        bound = linear_policy[op["operator"]]["fp32_accumulation_max_abs_bound"]
        for backend in ["direct", "packed"]:
            observed = op[backend + "_vs_gpu_full"]["max_abs"]
            cases.append({"operator": op["operator"], "path": backend, "max_abs": observed,
                          "bound": bound, "error_to_bound": observed/bound, "passed": observed <= bound})
    matrix.append({"report": str(path.relative_to(root)), "report_sha256": sha256(path),
                   "backend": report["backend_resolved"], "passed": all(x["passed"] for x in cases),
                   "checks": len(cases), "worst_error_to_bound": max(x["error_to_bound"] for x in cases), "cases": cases})
path = root / "reference/bf16-ops-windows.json"
ops = json.loads(path.read_text())
assert ops["source_sha256"] == sha256(root / "src/bf16_ops.rs")
assert ops["harness_sha256"] == sha256(root / "examples/bf16_ops_probe.rs")
assert ops["fixture_sha256"] == policy["sources"]["artifacts/reference/bf16-operators.safetensors"]
assert ops["affine_fixture_sha256"] == policy["sources"]["artifacts/reference/attention-operators-bf16.safetensors"]
norms = {x["name"]: x for x in policy["rms_cases"]}
gates = {x["name"] for x in policy["exact_cases"]}
operator_checks = []
for record in ops["operators"]:
    name, cmp = record["operator"], record["comparison"]
    if name in gates:
        passed = cmp["different_bits"] == 0 and cmp["nonfinite_pairs"] == 0
        evidence = "Exact GPU output bit equality"
    else:
        assert name in norms
        # Exact equality proves every bound. A single differing element can be
        # checked at its recorded index without requiring a tensor re-export.
        if cmp["different_bits"] == 0:
            passed = cmp["nonfinite_pairs"] == 0
            evidence = "Exact GPU output bit equality"
        elif cmp["different_bits"] == 1:
            index = cmp["worst_ulp_index"]
            bound = float(bounds[norms[name]["bound_key"]].flatten()[index])
            passed = cmp["maximum_absolute_error"] <= bound and cmp["nonfinite_pairs"] == 0
            evidence = {"sole_differing_index": index, "actual_max_error": cmp["maximum_absolute_error"], "frozen_element_bound": bound}
        else:
            passed = None
            evidence = "Full per-element output needed to apply nonuniform frozen bounds"
    operator_checks.append({"operator": name, "passed": passed, "evidence": evidence, "comparison": cmp})
result = {"schema_version": 1, "contract_sha256": sha256(policy_path),
          "bound_changes": False, "matrix_fp32_accumulation": matrix,
          "rms_gate_report_sha256": sha256(path), "rms_gate_cases": operator_checks,
          "completed_operator_checks_passed": all(x["passed"] for x in matrix) and all(x["passed"] for x in operator_checks),
          "local_contract_complete": False,
          "pending": ["Per-element native BF16 linear candidate outputs (posthoc count summaries alone are insufficient)",
                      "Same-input Rust BF16 attention raw/LSE/scaled operator outputs", "RoPE/residual/image projector graph coverage beyond current named local fixtures",
                      "Same-prefix BF16 output logits and exact free generation", "Corrected evaluation corpus and BF16 long-boundary output parity"],
          "qualification": "Only completed named local checks pass. No BF16 runner/full-trajectory/corpus qualification is claimed."}
native, outputs = [], []
for backend in ["avx512", "scalar"]:
    for kind in ["linear", "attention"]:
        assessment_path = root / f"reference/bf16-native-{kind}-assessment-{backend}.json"
        if assessment_path.exists():
            value = json.loads(assessment_path.read_text())
            assert value["contract_sha256"] == sha256(policy_path)
            native.append({"backend": backend, "kind": kind, "report": str(assessment_path.relative_to(root)),
                           "sha256": sha256(assessment_path), "passed": value["passed"], "checks": len(value["checks"]),
                           "failures": value["failures"]})
    assessment_path = root / f"reference/bf16-same-prefix-output-assessment-{backend}.json"
    if assessment_path.exists():
        value = json.loads(assessment_path.read_text())
        assert value["contract_sha256"] == sha256(policy_path)
        outputs.append({"backend": backend, "report": str(assessment_path.relative_to(root)), "sha256": sha256(assessment_path),
                        "passed": value["same_prefix_output_passed"], "failures": value["failures"],
                        "all_argmax_match": all(x["actual_argmax"] == x["bounds"]["required_argmax"] for x in value["logits"].values())})
if native:
    result["native_operator_assessments"] = native
    result["same_prefix_output_assessments"] = outputs
    result["completed_operator_checks_passed"] &= all(x["passed"] for x in native)
    result["qualification"] = "FP32 accumulation and RMS/gate subchecks pass; native attention/packed-linear and same-prefix logit failures remain. BF16 path is experimental and unqualified. Bounds unchanged."
    result["pending"] = ["Resolve recorded native BF16 local operator failures without widening bounds",
                         "Resolve same-prefix logit RMS failures without widening bounds", "Corrected corpus and long-boundary BF16 output parity",
                         "RoPE/residual/image projector coverage beyond current named fixtures"]
target = root / "reference/bf16-local-contract-assessment.json"
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"output": str(target), "completed_checks_passed": result["completed_operator_checks_passed"],
                  "matrix": [{k:v for k,v in m.items() if k != "cases"} for m in matrix], "rms_gate_cases": operator_checks}, indent=2))
