#!/usr/bin/env python3
"""Offline fixed-boundary telescope; no model invocation or policy changes."""
import argparse
import hashlib
from pathlib import Path

from contract import (ROOT, KIND, RUNTIME, STAGES, BRANCHES, CONTROL_BRANCHES, ENTRY_STATE,
                      ENDPOINT, require, read_bound, plan_window, stable, write_new, sha)
from evidence import load_evidence, exact, raw_sha, assert_f32, bind_extra
from source_guard import verify_sources


def signed_terms(a, endpoints, d):
    """Same explicit arithmetic for scalar host tests and FP64 array reporting."""
    require(len(endpoints) == 10, "Exactly E0 through E9 required")
    return {"E0_minus_A_embedding_propagated": endpoints[0] - a,
            **{f"E{b+1}_minus_E{b}_block{b}_conditional": endpoints[b+1] - endpoints[b]
               for b in range(9)},
            "D_minus_E9_layer9_pre_qkv_engine": d - endpoints[9]}


def fixed_endpoint(a, endpoints, d):
    total = d - a
    terms = signed_terms(a, endpoints, d)
    return {**ENDPOINT, "A": a, "D": d,
            "endpoints": {f"E{b}": value for b, value in enumerate(endpoints)},
            "D_minus_A_original": total, "signed_terms": terms,
            "signed_fractions_of_original_total": {
                k: None if total == 0 else v / total for k, v in terms.items()},
            "sum_abs_terms_over_abs_total": None if total == 0 else
                sum(abs(v) for v in terms.values()) / abs(total),
            "fp64_telescoping_residual": total - sum(terms.values()),
            "terms_exceed_original_bound_descriptive_only": {
                k: abs(v) > ENDPOINT["original_absolute_bound"] for k, v in terms.items()}}


def moments(value, np):
    return {"max_abs": float(np.max(np.abs(value))), "rms": float(np.sqrt(np.mean(value * value))),
            "signed_mean": float(np.mean(value))}


def decompose(a, endpoints, d, np):
    a, d = a.astype(np.float64), d.astype(np.float64)
    endpoints = [v.astype(np.float64) for v in endpoints]
    terms = signed_terms(a, endpoints, d)
    total = d - a
    require(all(np.isfinite(v).all() for v in [total, *terms.values()]), "Nonfinite telescope")
    residual = total - sum(terms.values())
    values = {"D_minus_A_original": total, **terms}
    index = tuple(int(i) for i in np.unravel_index(int(np.argmax(np.abs(total))), total.shape))
    total_rms = moments(total, np)["rms"]
    return {"shape": list(total.shape), "elements": int(total.size),
            "global": {k: moments(v, np) for k, v in values.items()},
            "rows": [{"row": row, **{k: moments(v[row], np) for k, v in values.items()},
                      "fp64_telescoping_residual_max_abs": float(np.max(np.abs(residual[row])))}
                     for row in range(144)],
            "endpoints_relative_to_A": {f"E{b}": moments(v - a, np) for b, v in enumerate(endpoints)},
            "worst_original_coordinate": list(index),
            "at_worst_original_coordinate": {"A": float(a[index]), "D": float(d[index]),
                "endpoints": {f"E{b}": float(v[index]) for b, v in enumerate(endpoints)},
                **{k: float(v[index]) for k, v in values.items()}},
            "fp64_telescoping_residual_max_abs": float(np.max(np.abs(residual))),
            "term_rms_sum_over_total_rms": None if total_rms == 0 else
                sum(moments(v, np)["rms"] for v in terms.values()) / total_rms}


def validate_capture_header(capture, plan_sha256):
    require(capture.get("kind") == KIND + "-gpu"
            and capture.get("status") == "three_controls_exact_eight_endpoints_captured"
            and capture.get("plan_sha256") == plan_sha256 and capture.get("runtime") == RUNTIME
            and capture.get("controls_passed") == CONTROL_BRANCHES
            and capture.get("completed_branches") == BRANCHES
            and capture.get("source_and_input_closure") is True
            and capture.get("effective_compiled_blocks") is False
            and capture.get("native_block_calls") == 54
            and capture.get("native_layer9_qkv_calls") == 11, "Capture not accepted")
    controls = capture.get("controls", [])
    require([c["branch"] for c in controls] == CONTROL_BRANCHES, "Wrong control inventory")
    for record in controls:
        require([r["stage"] for r in record["stages"]] == list(STAGES[record["branch"]])
                and all(r["bit_exact"] is True and r["mismatched_elements"] == 0
                        for r in record["stages"]), "Wrong/failed complete control stages")


def compare(args):
    require(args.execute_reviewed_plan, "Offline tensor work requires coordinator release")
    import numpy as np
    from safetensors.numpy import load

    output, plan_path, report_path = args.output.resolve(), args.plan.resolve(), args.gpu_report.resolve()
    require(all(p.is_relative_to(ROOT) for p in (output, plan_path, report_path)) and not output.exists(),
            "Fresh output and project input paths required")
    plan, bound = plan_window(plan_path, args.plan_sha256)
    require(verify_sources() == plan["source_guard"], "Source guard changed")
    capture = read_bound(report_path, args.gpu_report_sha256)
    bound[report_path] = args.gpu_report_sha256
    validate_capture_header(capture, args.plan_sha256)
    require(capture["native_source_guard"] == plan["source_guard"], "Capture/source guard differs")
    for name, digest in capture["artifact_sha256"].items():
        bind_extra(bound, name, digest)
    evidence = load_evidence(bound)
    require(capture["historical_evidence"] == plan["historical_evidence"] == evidence["proof"]
            and plan["state_raw_sha256"] == evidence["proof"]["state_raw_sha256"], "Historical state proof differs")
    for field in ("effective_torch_math", "effective_threads"):
        require(capture[field] == evidence["reports"]["old_gpu_report"][field], "Effective runtime changed")
    require({k: v for k, v in capture["environment"].items() if k != "free_memory_mib"}
            == {k: v for k, v in evidence["reports"]["old_gpu_report"]["environment"].items() if k != "free_memory_mib"},
            "Actual GPU/driver/package/interpreter environment differs")
    path = report_path.parent / "tensors.safetensors"
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == capture["tensor_file_sha256"], "New tensor archive changed")
    bind_extra(bound, path.relative_to(ROOT).as_posix(), capture["tensor_file_sha256"])
    arrays = load(raw)
    expected_names = {b + "." + s for b in BRANCHES for s in STAGES[b]}
    require(set(arrays) == set(capture["tensors"]) == expected_names, "New array inventory differs")
    control_checks = []
    for branch in BRANCHES:
        require(exact(arrays[branch + ".input"], evidence["entries"][ENTRY_STATE[branch]], np),
                "Branch is not complete saved entry: " + branch)
        for stage, shape in STAGES[branch].items():
            name = branch + "." + stage
            value, metadata = arrays[name], capture["tensors"][name]
            assert_f32(value, shape, np)
            require(metadata == {"shape": shape, "dtype": "F32", "raw_sha256": raw_sha(value)},
                    "New raw tensor identity differs")
            if branch in CONTROL_BRANCHES:
                require(exact(value, evidence["controls"][branch][stage], np), "Actual control fails: " + name)
                control_checks.append({"branch": branch, "stage": stage, "bit_exact": True})
    stage, coordinate = ENDPOINT["stage"], tuple(ENDPOINT["coordinate"])
    a, d = evidence["endpoints"]["A"], evidence["endpoints"]["D"]
    endpoints = [arrays[f"E{b}." + stage] for b in range(10)]
    require(exact(arrays["control_gpu." + stage], a, np), "Original GPU endpoint differs")
    require(exact(endpoints[7], evidence["endpoints"]["E7"], np)
            and exact(endpoints[8], evidence["endpoints"]["E8"], np), "Accepted downstream endpoints differ")
    require(tuple(float(v[coordinate]) for v in (a, endpoints[7], endpoints[8], d))
            == (-19.037837982177734, -19.031208038330078, -19.031253814697266, -19.03121566772461),
            "Original failing coordinate changed")
    require(evidence["reports"]["policy"]["stages"][stage]["absolute_tolerance"] == ENDPOINT["original_absolute_bound"],
            "Frozen original bound changed")
    decomposition = decompose(a, endpoints, d, np)
    fixed = fixed_endpoint(float(a[coordinate]), [float(v[coordinate]) for v in endpoints], float(d[coordinate]))
    # This is an offline algebra check on F32 endpoints promoted to F64, not a model tolerance.
    require(decomposition["fp64_telescoping_residual_max_abs"] == 0.0
            and fixed["fp64_telescoping_residual"] == 0.0, "FP64 telescope did not close exactly")
    stable(bound)
    report = {"kind": KIND + "-analysis", "status": "one_valid_fixed_boundary_decomposition_only",
              "plan_sha256": args.plan_sha256, "gpu_report_sha256": args.gpu_report_sha256,
              "runtime": RUNTIME, "historical_evidence": evidence["proof"],
              "fresh_control_checks": control_checks, "fresh_control_arrays_exact": len(control_checks),
              "new_entry_states_exact": len(BRANCHES), "new_payloads_checked": len(arrays),
              "endpoint_raw_sha256": {"A": raw_sha(a), "D": raw_sha(d),
                                      **{f"E{b}": raw_sha(v) for b, v in enumerate(endpoints)}},
              "fixed_endpoint": fixed, "endpoint_decomposition": decomposition,
              "source_and_input_closure": True,
              "artifact_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in bound.items()},
              "limits": ["Each adjacent difference is a fixed-order conditional boundary substitution through a nonlinear suffix, not an independent kernel cause.",
                         "D-E9 includes the layer9 RMS/QKV path; no finer operator attribution is inferred.",
                         "Signed fractions can be negative or exceed one; cancellation and nonadditive max/RMS summaries are retained.",
                         "E0-A includes the embedding/projector state difference propagated through the suffix.",
                         "Original startup source gaps and changed observation/allocation history remain; complete finite controls are not universal numerical proofs.",
                         "The frozen bound is descriptive only; this does not relax policy, fix production or qualify full hidden trajectories.",
                         "Stop after the eleven specified branches; no follow-up sweep or model generation is implied."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, report)
    print(sha(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--gpu-report", type=Path, required=True)
    p.add_argument("--gpu-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute-reviewed-plan", action="store_true")
    compare(p.parse_args())


if __name__ == "__main__":
    main()
