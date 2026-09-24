#!/usr/bin/env python3
"""Apply frozen per-element BF16 local bounds to actual Rust operator outputs."""
import argparse
import json
import pathlib
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256


def compare(expected, actual, bounds, rms_bound=None, require_bf16=True):
    if actual.shape != expected.shape:
        return {"passed": False, "shape_mismatch": [list(expected.shape), list(actual.shape)]}
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return {"passed": False, "nonfinite": True}
    bf16_exact = torch.equal(actual.float(), actual.bfloat16().float())
    error = (actual.double()-expected.double()).abs()
    violations = (error > bounds.double()).flatten().nonzero().flatten()
    rms = float(error.square().mean().sqrt())
    return {"passed": not len(violations) and (rms_bound is None or rms <= rms_bound) and (not require_bf16 or bf16_exact),
            "elements": expected.numel(), "different_elements": int((actual.float()!=expected.float()).sum()),
            "max_abs": float(error.max()), "rms_abs": rms, "rms_bound": rms_bound,
            "bound_violations": len(violations), "first_violation_indices": violations[:20].tolist(),
            "represents_bf16_values": bf16_exact if require_bf16 else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=pathlib.Path)
    p.add_argument("--kind", choices=["linear", "attention"], required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    policy_path = root / "reference/bf16-local-contract-v1.json"
    policy = json.loads(policy_path.read_text())
    for name, digest in policy["sources"].items():
        assert sha256(root/name) == digest, name
    contract = load_file(str(root / "artifacts/reference/bf16-local-contract-v1.safetensors"))
    candidate = load_file(str(args.candidate))
    meta_path = args.candidate.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    failures, records = [], {}
    source_name = "bf16-operators" if args.kind == "linear" else "attention-operators-bf16"
    source_path = root / f"artifacts/reference/{source_name}.safetensors"
    if meta.get("fixture_sha256") != sha256(source_path):
        failures.append("provenance: fixture hash mismatch")
    if meta.get("contract_sha256") != sha256(policy_path):
        failures.append("provenance: contract hash mismatch")
    if args.kind == "linear":
        for item in policy["linear_cases"]:
            for backend in ["direct", "packed"]:
                key = item["name"] + "." + backend
                if key not in candidate:
                    failures.append(key + ":missing")
                    continue
                records[key] = compare(contract[item["native_expected_key"]], candidate[key], contract[item["native_bound_key"]])
    else:
        source = load_file(str(source_path))
        for item in policy["attention_cases"]:
            for kind in ["raw", "scaled", "lse"]:
                key = item["name"] + "." + kind
                if key not in candidate:
                    failures.append(key + ":missing")
                    continue
                if kind == "lse":
                    expected = source[key]
                    bound = torch.full_like(expected, item["lse_max_abs_bound"])
                    records[key] = compare(expected, candidate[key], bound, require_bf16=False)
                else:
                    gate = item[kind]
                    records[key] = compare(source[gate["expected_key"]], candidate[key], contract[gate["bound_key"]], gate["rms_bound"])
    failures.extend(key for key, record in records.items() if not record["passed"])
    result = {"schema_version": 1, "kind": args.kind, "passed": not failures,
              "contract_sha256": sha256(policy_path), "candidate_sha256": sha256(args.candidate),
              "candidate_metadata_sha256": sha256(meta_path), "failures": failures, "checks": records,
              "qualification": "Named equal-input BF16 operators only; does not qualify full-model output or corpus parity"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": not failures, "checks": len(records), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
