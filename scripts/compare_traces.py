#!/usr/bin/env python3
"""Calibrate before Rust comparison, then enforce a frozen numerical policy."""
import argparse
import datetime
import json
import math
import pathlib
import re

import torch
from safetensors.torch import load_file

from fetch_reference import sha256


def normalized(trace):
    result = {}
    for key, value in trace.items():
        name = key.removeprefix("prefill.")
        if name in result:
            raise ValueError(f"Ambiguous trace names normalize to {name}")
        result[name] = value
    return result


def validate_policy(policy):
    if not isinstance(policy.get("stages"), dict) or not policy["stages"]:
        raise ValueError("Numerical policy has no stage bounds")
    for name, stage_policy in policy["stages"].items():
        bound = stage_policy.get("absolute_tolerance")
        if (isinstance(bound, bool) or not isinstance(bound, (int, float))
                or not math.isfinite(bound) or bound < 0):
            raise ValueError(f"Invalid absolute tolerance for {name}: {bound}")


def logit_decision(reference, candidate, max_error):
    a, b = reference.float().flatten(), candidate.float().flatten()
    if a.numel() < 2 or a.shape != b.shape:
        raise ValueError("Logit decisions require equal vectors with at least two values")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Nonfinite model logits cannot qualify a greedy decision")
    values = a.topk(2).values
    margin = float(values[0] - values[1])
    reference_argmax, candidate_argmax = int(a.argmax()), int(b.argmax())
    must_match = margin > 2 * max_error
    return {"reference_argmax": reference_argmax, "candidate_argmax": candidate_argmax,
            "winner_margin": margin, "must_match": must_match,
            "argmax_matches": reference_argmax == candidate_argmax, "near_tie": not must_match}


def stage(key):
    return re.sub(r"^decode\.\d+\.", "decode.*.", key)


def stats(a, b):
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    a, b = a.double(), b.double()
    valid = torch.isfinite(a) & torch.isfinite(b)
    if (not torch.equal(torch.isnan(a), torch.isnan(b)) or not torch.equal(torch.isinf(a), torch.isinf(b))
            or not torch.equal(a[torch.isinf(a)], b[torch.isinf(b)])):
        return {"nonfinite_mismatch": True}
    delta = (a[valid] - b[valid]).abs()
    return {"max_abs": float(delta.max()) if delta.numel() else 0,
            "rms_abs": float(delta.square().mean().sqrt()) if delta.numel() else 0,
            "reference_max_abs": float(a[valid].abs().max()) if valid.any() else 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=pathlib.Path)
    p.add_argument("candidate", type=pathlib.Path)
    p.add_argument("--calibrate", type=pathlib.Path, help="Create NEW tolerances from GPU vs operator, never from Rust")
    p.add_argument("--tolerances", type=pathlib.Path)
    p.add_argument("--diagnostic", action="store_true", help="Measure two GPU sources without creating or accepting a policy")
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.calibrate and args.calibrate.exists():
        raise ValueError("Refusing to overwrite frozen tolerances; use a new versioned filename")
    if sum(map(bool, [args.calibrate, args.tolerances, args.diagnostic])) != 1:
        raise ValueError("Specify exactly one of --calibrate, --tolerances, or --diagnostic")
    reference = normalized(load_file(str(args.reference)))
    candidate = normalized(load_file(str(args.candidate)))
    # Input tensors are verified by preprocessing tests; this gate requires every captured model tensor.
    expected = [k for k in reference if k.startswith(("embedding", "layer.", "final_norm.", "logits", "decode."))]
    if not expected:
        raise ValueError("Reference contains no model tensors to compare")
    missing = sorted(set(expected) - set(candidate))
    policy = json.loads(args.tolerances.read_text()) if args.tolerances else None
    if policy is not None:
        validate_policy(policy)
    if policy and policy.get("precision") == "bf16":
        review_path = pathlib.Path(__file__).resolve().parents[1] / "reference/bf16-policy-review.json"
        if review_path.exists():
            review = json.loads(review_path.read_text())
            if review.get("policy_sha256") == sha256(args.tolerances) and not review.get("qualification_accepted"):
                raise ValueError(f"BF16 policy rejected for qualification by {review_path}; preserve it as diagnostic evidence")
    if policy and policy["reference_sha256"] != sha256(args.reference):
        raise ValueError("Reference trace does not match the immutable tolerance policy source")
    measurements, aggregates, decisions = {}, {}, {}
    failures = list(missing)
    for key in sorted(set(expected) & set(candidate)):
        metric = stats(reference[key], candidate[key])
        measurements[key] = metric
        if "max_abs" not in metric:
            failures.append(key)
            continue
        group = stage(key)
        aggregate = aggregates.setdefault(group, {"baseline_max_abs": 0.0, "reference_max_abs": 0.0})
        aggregate["baseline_max_abs"] = max(aggregate["baseline_max_abs"], metric["max_abs"])
        aggregate["reference_max_abs"] = max(aggregate["reference_max_abs"], metric["reference_max_abs"])
        if policy:
            tolerance = policy["stages"].get(group)
            if tolerance is None or metric["max_abs"] > tolerance["absolute_tolerance"]:
                failures.append(key)
        if key.endswith("logits"):
            decisions[key] = logit_decision(reference[key], candidate[key], metric["max_abs"])
            if not decisions[key]["argmax_matches"] and decisions[key]["must_match"]:
                failures.append(key + ":argmax")
    if args.calibrate or args.diagnostic:
        # These two source identities are checked before a tolerance policy can be created.
        ref_meta = json.loads(args.reference.with_name("metadata.json").read_text())
        cand_meta = json.loads(args.candidate.with_name("metadata.json").read_text())
        assert ref_meta["attention"] == "flex" and cand_meta["attention"] in ["dense", "blockwise"]
        assert ref_meta["precision"] == cand_meta["precision"]
        assert ref_meta["precision"] in ["fp32", "bf16"]
        for identity in ["model_revision", "weights_sha256", "image_sha256", "prompt", "prefix_length",
                         "min_dimension", "max_dimension", "cast_contract", "rms_norm_implicit_epsilon"]:
            assert ref_meta.get(identity) == cand_meta.get(identity), f"Calibration source mismatch: {identity}"
        assert ref_meta["environment"]["packages"] == cand_meta["environment"]["packages"]
        assert ref_meta["token_ids"] == cand_meta["token_ids"]
        assert ref_meta["trace_sha256"] == sha256(args.reference)
        assert cand_meta["trace_sha256"] == sha256(args.candidate)
        assert not missing and not failures, "Incomplete or invalid reference comparison"
    if args.calibrate:
        assert ref_meta["precision"] == "fp32", "BF16 policies require an explicit calibration-quality review; use --diagnostic"
        for value in aggregates.values():
            value["absolute_tolerance"] = max(4 * value["baseline_max_abs"],
                                              32 * torch.finfo(torch.float32).eps * max(1, value["reference_max_abs"]))
        policy = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "source": f"strict {ref_meta['precision']} GPU FlexAttention versus dense GPU attention under identical prefixes",
                  "precision": ref_meta["precision"],
                  "cast_contract": ref_meta.get("cast_contract"),
                  "limitation": "Calibrated only on the named small fixture; not a corpus or long-context qualification",
                  "rule": "atol = max(4 * observed max error, 32 * FP32 epsilon * max(1, reference peak)); rtol = 0",
                  "reference_sha256": sha256(args.reference), "operator_sha256": sha256(args.candidate), "stages": aggregates}
        args.calibrate.parent.mkdir(parents=True, exist_ok=True)
        args.calibrate.write_text(json.dumps(policy, indent=2) + "\n")
    report = {"passed": None if args.diagnostic else not failures, "diagnostic_only": args.diagnostic,
              "reference": str(args.reference), "candidate": str(args.candidate),
              "reference_sha256": sha256(args.reference), "candidate_sha256": sha256(args.candidate),
              "compared_tensors": len(measurements), "missing": missing, "failures": sorted(set(failures)),
              "logit_decisions": decisions, "measurements": measurements,
              "stage_aggregates": aggregates,
              "tolerances": str(args.tolerances or args.calibrate) if not args.diagnostic else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "compared": report["compared_tensors"],
                      "failures": report["failures"], "logit_decisions": decisions}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
