#!/usr/bin/env python3
"""Offline four-endpoint telescoping only; never invokes a model or changes policy."""
import argparse
import hashlib
from pathlib import Path

from contract import (ROOT, KIND, RUNTIME, STAGES, CONTROLS, ENDPOINT, require,
                      read_bound, plan_window, stable, write_new, sha)
from evidence import load_evidence, exact, raw_sha, assert_f32, bind_extra
from source_guard import verify_sources


def moments(value, np):
    return {"max_abs": float(np.max(np.abs(value))), "rms": float(np.sqrt(np.mean(value * value))),
            "signed_mean": float(np.mean(value))}


def decompose(a, b, c, d, np):
    a, b, c, d = (v.astype(np.float64) for v in (a, b, c, d))
    terms = {"D_minus_A_original": d - a,
             "D_minus_C_downstream_accumulated_engine": d - c,
             "C_minus_B_layer7_engine_propagated": c - b,
             "B_minus_A_earlier_state_propagated": b - a}
    require(all(np.isfinite(v).all() for v in terms.values()), "Nonfinite telescoping terms")
    residual = terms["D_minus_A_original"] - (terms["D_minus_C_downstream_accumulated_engine"]
                 + terms["C_minus_B_layer7_engine_propagated"] + terms["B_minus_A_earlier_state_propagated"])
    total = terms["D_minus_A_original"]
    index = tuple(int(i) for i in np.unravel_index(int(np.argmax(np.abs(total))), total.shape))
    rms = moments(total, np)["rms"]
    return {"shape": list(total.shape), "elements": int(total.size),
            "global": {k: moments(v, np) for k, v in terms.items()},
            "rows": [{"row": i, **{k: moments(v[i], np) for k, v in terms.items()}} for i in range(144)],
            "worst_original_coordinate": list(index),
            "at_worst_original_coordinate": {"A": float(a[index]), "B": float(b[index]),
                "C": float(c[index]), "D": float(d[index]), **{k: float(v[index]) for k, v in terms.items()}},
            "fp64_telescoping_residual_max_abs": float(np.max(np.abs(residual))),
            "term_rms_sum_over_total_rms": None if rms == 0 else
                sum(moments(v, np)["rms"] for k, v in terms.items() if k != "D_minus_A_original") / rms}


def fixed_endpoint(a, b, c, d):
    total, downstream, layer7, earlier = d - a, d - c, c - b, b - a
    return {**ENDPOINT, "A": a, "B": b, "C": c, "D": d,
            "D_minus_A_original": total, "D_minus_C_downstream_accumulated_engine": downstream,
            "C_minus_B_layer7_engine_propagated": layer7, "B_minus_A_earlier_state_propagated": earlier,
            "signed_fractions_of_original_total": {name: None if total == 0 else term / total
                for name, term in (("downstream_engine", downstream), ("layer7_engine_propagated", layer7),
                                   ("earlier_state_propagated", earlier))},
            "fp64_telescoping_residual": total - (downstream + layer7 + earlier),
            "B_minus_A_exceeds_original_bound_descriptive_only": abs(earlier) > ENDPOINT["original_absolute_bound"]}


def compare(args):
    require(args.execute_reviewed_plan, "Offline tensor work is held during the quiet benchmark")
    import numpy as np
    from safetensors.numpy import load
    output, plan_path, report_path = args.output.resolve(), args.plan.resolve(), args.gpu_report.resolve()
    require(all(p.is_relative_to(ROOT) for p in (output, plan_path, report_path)) and not output.exists(),
            "Fresh output and project input paths required")
    plan, bound = plan_window(plan_path, args.plan_sha256)
    require(verify_sources() == plan["source_guard"], "Native source guard changed")
    capture = read_bound(report_path, args.gpu_report_sha256)
    bound[report_path] = args.gpu_report_sha256
    require(capture["kind"] == KIND + "-gpu" and capture["status"] == "control_exact_substitution_captured"
            and capture["plan_sha256"] == args.plan_sha256 and capture["runtime"] == RUNTIME
            and capture["control_passed"] is True and capture["substitution_executed"] is True
            and capture["source_and_input_closure"] is True
            and capture["native_layer8_calls"] == capture["native_layer9_qkv_calls"] == 2,
            "New capture not accepted")
    require([r["stage"] for r in capture["controls"]] == CONTROLS
            and all(r["bit_exact"] is True and r["mismatched_elements"] == 0 for r in capture["controls"]),
            "Wrong or failed 17-stage control")
    for name, digest in capture["artifact_sha256"].items():
        bind_extra(bound, name, digest)
    evidence = load_evidence(bound)
    require(capture["historical_evidence"] == evidence["proof"], "Historical evidence proof differs")
    require(capture["effective_torch_math"] == evidence["reports"]["old_gpu_report"]["effective_torch_math"]
            and capture["effective_threads"] == evidence["reports"]["old_gpu_report"]["effective_threads"],
            "Actual runtime differs from the old control")
    require({k: v for k, v in capture["environment"].items() if k != "free_memory_mib"}
            == {k: v for k, v in evidence["reports"]["old_gpu_report"]["environment"].items() if k != "free_memory_mib"},
            "Actual GPU/driver/package/interpreter environment differs")
    path = report_path.parent / "tensors.safetensors"
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == capture["tensor_file_sha256"], "New tensor archive changed")
    bound[path] = capture["tensor_file_sha256"]
    arrays = load(raw)
    expected_names = {b + "." + s for b in ("control_c", "substitution_s") for s in STAGES}
    require(set(arrays) == set(capture["tensors"]) == expected_names, "New tensor inventory differs")
    control_checks = []
    for stage, shape in STAGES.items():
        for branch in ("control_c", "substitution_s"):
            name = branch + "." + stage
            value, metadata = arrays[name], capture["tensors"][name]
            assert_f32(value, shape, np)
            require(metadata == {"shape": shape, "dtype": "F32", "raw_sha256": raw_sha(value)},
                    "New raw tensor identity differs")
        require(exact(arrays["control_c." + stage], evidence["stages"]["C"][stage], np),
                "Actual 17-stage control fails: " + stage)
        control_checks.append({"stage": stage, "bit_exact": True})
    require(exact(arrays["substitution_s.input"], evidence["states"]["S"], np), "New B entry is not full saved S")
    require(exact(arrays["control_c.input"], evidence["states"]["C7"], np), "New C entry is not full saved C7")
    values = evidence["stages"]
    stages = {s: decompose(values["A"][s], arrays["substitution_s." + s], values["C"][s], values["D"][s], np)
              for s in STAGES}
    stage, coordinate = ENDPOINT["stage"], tuple(ENDPOINT["coordinate"])
    a, b, c, d = (float(v[coordinate]) for v in (values["A"][stage], arrays["substitution_s." + stage],
                                               values["C"][stage], values["D"][stage]))
    require((a, c, d) == (-19.037837982177734, -19.031253814697266, -19.03121566772461),
            "Original failing endpoint changed")
    require(evidence["reports"]["policy"]["stages"][stage]["absolute_tolerance"] == ENDPOINT["original_absolute_bound"],
            "Frozen original bound changed")
    prior = evidence["reports"]["old_analysis"]["stages"][stage]
    require(prior["worst_original_error_coordinate"] == ENDPOINT["coordinate"]
            and prior["at_worst_original_error"]["rust_cpu"] == d
            and prior["at_worst_original_error"]["gpu_cpu"] == c
            and prior["at_worst_original_error"]["gpu_gpu"] == a, "Prior decomposition join differs")
    stable(bound)
    report = {"kind": KIND + "-analysis", "status": "one_valid_downstream_decomposition_only",
              "plan_sha256": args.plan_sha256, "gpu_report_sha256": args.gpu_report_sha256,
              "historical_evidence": evidence["proof"], "fresh_control_checks": control_checks,
              "new_entry_states_exact": 2, "new_payloads_checked": len(arrays),
              "fixed_endpoint": fixed_endpoint(a, b, c, d), "stages": stages,
              "source_and_input_closure": True,
              "artifact_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in bound.items()},
              "limits": ["D-C is an accumulated downstream path difference, not an isolated kernel error.",
                         "C-B and B-A are conditional substitutions through a nonlinear suffix; the order is fixed, not a unique allocation of interactions.",
                         "Signed fractions can be negative or exceed one; maxima/RMS are not additive attribution.",
                         "Original startup gaps and altered allocation/observation history remain.",
                         "The original bound is reused descriptively; no policy relaxation, production fix or full-model qualification.",
                         "Stop after this decomposition; no further layer, sweep or inference is implied."]}
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
