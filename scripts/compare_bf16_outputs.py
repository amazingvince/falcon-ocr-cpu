#!/usr/bin/env python3
"""Apply frozen same-prefix BF16 output gates without accepting hidden drift."""
import argparse
import json
import pathlib
import torch
from safetensors.torch import load_file
from compare_traces import normalized, stats
from fetch_reference import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=pathlib.Path)
    p.add_argument("--contract", type=pathlib.Path, default=pathlib.Path("reference/bf16-local-contract-v1.json"))
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    contract = json.loads(args.contract.read_text())
    for source, expected in contract["sources"].items():
        if sha256(root / source) != expected:
            raise ValueError(f"Frozen contract source hash mismatch: {source}")
    reference_path = root / "artifacts/reference/smoke-bf16/trace.safetensors"
    frozen_graph = json.loads((root / "reference/gpu-blockwise-smoke-bf16.json").read_text())
    if sha256(reference_path) != frozen_graph["reference_sha256"]:
        raise ValueError("BF16 reference trace differs from the frozen output calibration")
    ref_meta = json.loads(reference_path.with_name("metadata.json").read_text())
    reference = normalized(load_file(str(reference_path)))
    candidate = normalized(load_file(str(args.candidate)))
    candidate_meta_path = args.candidate.with_suffix(".json")
    if not candidate_meta_path.exists():
        candidate_meta_path = args.candidate.with_name("metadata.json")
    meta = json.loads(candidate_meta_path.read_text())
    failures = []
    if meta.get("precision") != "bf16" or not meta.get("teacher_forced"):
        failures.append("provenance: requires BF16 teacher-forced candidate")
    for key in ["model_revision", "weights_sha256"]:
        if meta.get(key) != ref_meta[key]:
            failures.append("provenance:" + key)
    fixture_matches = meta.get("fixture_sha256") == sha256(reference_path)
    copied_inputs_match = all(key in candidate
                              and torch.equal(torch.isnan(candidate[key].float()), torch.isnan(reference[key].float()))
                              and torch.equal(torch.nan_to_num(candidate[key].float()), torch.nan_to_num(reference[key].float()))
                              for key in ["tokens", "patches", "pos_t", "pos_hw", "teacher_tokens"])
    # GPU oracle export has all copied inputs plus its trace hash in metadata.
    if not (fixture_matches or copied_inputs_match):
        failures.append("provenance: canonical fixture hash or copied input tensors required")
    if fixture_matches and meta.get("teacher_tokens") != reference["teacher_tokens"].tolist():
        failures.append("provenance: teacher token sequence mismatch")
    records = {}
    for name, bounds in contract["same_prefix_logits"].items():
        if name not in candidate:
            failures.append(name + ":missing")
            continue
        expected, actual = reference[name], candidate[name]
        metric = stats(expected, actual)
        if actual.shape != expected.shape or not torch.isfinite(actual).all():
            failures.append(name + ":shape/nonfinite")
            continue
        exact_bf16_values = torch.equal(actual.float(), actual.bfloat16().float())
        winner = int(actual.argmax())
        passed = (metric["max_abs"] <= bounds["max_abs_bound"] and metric["rms_abs"] <= bounds["rms_bound"]
                  and winner == bounds["required_argmax"] and exact_bf16_values)
        if not passed:
            failures.append(name)
        records[name] = {**metric, "bounds": bounds, "actual_argmax": winner,
                         "represents_bf16_values": exact_bf16_values, "passed": passed}
    report = {"schema_version": 1, "same_prefix_output_passed": not failures,
              "qualification": "Named same-prefix output gates only; local operator and free-generation/corpus gates are separate. No full hidden-trajectory parity claim.",
              "contract_sha256": sha256(args.contract), "reference_sha256": sha256(reference_path),
              "candidate_sha256": sha256(args.candidate), "candidate_metadata_sha256": sha256(candidate_meta_path),
              "failures": failures, "logits": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": not failures, "logits": len(records), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
