#!/usr/bin/env python3
"""One offline three-branch decomposition; no model invocation or policy changes."""
import argparse
import hashlib
import json
from pathlib import Path

from contract import (ROOT, RUNTIME, STAGES, CONTROL_STAGES, PINS, FAILURE_CONTEXT, validate_cpu_control,
                      require, sha, resolve, read_bound, write_new, validate_plan)


def saved_name(keys, name):
    matches = [key for key in (name, "prefill." + name) if key in keys]
    require(len(matches) == 1, "Missing/ambiguous original stage " + name)
    return matches[0]


def array_record(value):
    return {"shape": list(value.shape), "dtype": "F32",
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def exact(a, b, np):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def moments(values, np):
    return {"max_abs": float(np.max(np.abs(values))),
            "rms": float(np.sqrt(np.mean(values * values))), "signed_mean": float(np.mean(values))}


def decompose(rust, gpu_cpu, gpu_gpu, np):
    # FP64 accounting only. This does not recompute any model-stage output.
    r, c, g = (x.astype(np.float64) for x in (rust, gpu_cpu, gpu_gpu))
    total, engine, incoming = r - g, r - c, c - g
    require(all(np.isfinite(x).all() for x in (total, engine, incoming)), "Invalid FP64 accounting")
    names = {"original_rust_minus_gpu": total,
             "engine_difference_on_cpu_entry_state": engine,
             "incoming_state_difference_through_gpu": incoming}
    closure = total - (engine + incoming)
    index = np.unravel_index(int(np.argmax(np.abs(total))), total.shape)
    index = tuple(int(i) for i in index)
    denominator = moments(total, np)["rms"]
    return {"shape": list(total.shape), "elements": int(total.size),
            "global": {name: moments(value, np) for name, value in names.items()},
            "rows": [{"row": row, **{name: moments(value[row], np) for name, value in names.items()}}
                     for row in range(144)],
            "worst_original_error_coordinate": list(index),
            "at_worst_original_error": {"rust_cpu": float(r[index]), "gpu_cpu": float(c[index]),
                                         "gpu_gpu": float(g[index]),
                                         **{name: float(value[index]) for name, value in names.items()}},
            "fp64_decomposition_residual_max_abs": float(np.max(np.abs(closure))),
            "term_rms_sum_over_total_rms": None if denominator == 0 else
                (moments(engine, np)["rms"] + moments(incoming, np)["rms"]) / denominator}


def compare(args):
    # Imports/tensor reads occur only after explicit release and within this call.
    require(args.execute_reviewed_plan, "Offline tensor work is held until explicit release")
    import numpy as np
    from safetensors.numpy import load

    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh project report required")
    plan_path = args.plan.resolve()
    require(plan_path.is_relative_to(ROOT), "Project plan required")
    plan = read_bound(plan_path, args.plan_sha256)
    validate_plan(plan)
    bound = {plan_path: args.plan_sha256}

    def read_report(path, digest):
        path = path.resolve()
        require(path.is_relative_to(ROOT), "Report must be inside project")
        result = read_bound(path, digest)
        bound[path] = digest
        return result

    execution = read_report(args.cpu_execution, args.cpu_execution_sha256)
    require(execution["kind"] == "fp32-crossover-block0-execution-v1"
            and execution["status"] == "cpu_control_exact" and execution["source_and_input_closure"] is True
            and execution["plan_sha256"] == args.plan_sha256, "CPU capture not accepted")
    for name, digest in execution["artifact_sha256"].items():
        bound[resolve(name)] = digest
    require(all(sha(p) == h for p, h in bound.items()), "CPU execution closure changed")
    cpu_path = args.cpu_execution.resolve().parent / "rust/report.json"
    require(cpu_path in bound, "CPU report missing from execution identity")
    cpu = read_report(cpu_path, bound[cpu_path])
    validate_cpu_control(execution, cpu, args.plan_sha256)
    gpu_path = args.gpu_report.resolve()
    gpu = read_report(gpu_path, args.gpu_report_sha256)
    require(cpu["status"] == "control_exact" and cpu["plan_sha256"] == args.plan_sha256
            and cpu["branch"] == "cpu_state" and cpu["arithmetic_changed"] is False, "CPU arm differs")
    require(gpu["status"] == "gpu_arm_control_passed_crossover_captured"
            and gpu["gpu_control_passed"] is True and gpu["crossover_executed"] is True
            and gpu["source_and_input_closure"] is True and gpu["plan_sha256"] == args.plan_sha256
            and gpu["native_block0_calls"] == 2
            and gpu["runtime"] == RUNTIME, "GPU arm differs or failed control")
    require(gpu["cpu_prelaunch_gate"] == {
                "execution_path": args.cpu_execution.resolve().relative_to(ROOT).as_posix(),
                "execution_sha256": args.cpu_execution_sha256, "report_sha256": bound[cpu_path],
                "five_controls_exact": True, "input_bit_exact": True, "native_block0_calls": 1,
                "checked_before_cuda_import": True}, "GPU launch did not bind the accepted CPU arm")
    require([x["stage"] for x in cpu["controls"]] == CONTROL_STAGES
            and all(x["passed"] is True and x["bit_mismatches"] == 0 for x in cpu["controls"]), "CPU reported controls")
    require([x["stage"] for x in gpu["controls"]] == CONTROL_STAGES
            and all(x["bit_exact"] is True and x["mismatched_elements"] == 0 for x in gpu["controls"]), "GPU reported controls")
    # Recheck GPU final recorded inputs, including complete model assets.
    for name, digest in gpu["inputs_sha256"].items():
        path = resolve(name.replace("\\", "/"))
        require(path not in bound or bound[path] == digest, "Conflicting GPU identity")
        bound[path] = digest
    require(all(sha(p) == h for p, h in bound.items()), "Saved arm/input identity changed")

    def tensor_file(path, digest):
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "Tensor archive changed")
        bound[path] = digest
        return load(raw)

    cpu_arrays = tensor_file(cpu_path.parent / "tensors.safetensors", cpu["tensor_file_sha256"])
    gpu_arrays = tensor_file(gpu_path.parent / "tensors.safetensors", gpu["tensor_file_sha256"])
    for arrays, report, branches, digest_key in [(cpu_arrays, cpu, ["cpu_state"], "raw_sha256"),
                                                 (gpu_arrays, gpu, ["gpu_state", "cpu_state"], "sha256")]:
        expected = {branch + "." + name for branch in branches for name in STAGES}
        require(set(arrays) == set(report["tensors"]) == expected, "Stage inventory differs")
        for branch in branches:
            for name, shape in STAGES.items():
                key = branch + "." + name
                value, metadata = arrays[key], report["tensors"][key]
                require(value.dtype == np.dtype("float32") and list(value.shape) == shape
                        and np.isfinite(value).all(), "Invalid stage " + key)
                actual = array_record(value)
                require(metadata["dtype"] == "F32" and metadata["shape"] == shape
                        and metadata[digest_key] == actual["sha256"], "Stage raw-bit identity differs")
    originals = {role: tensor_file(resolve(path), digest) for role, (path, digest) in PINS.items()
                 if role.endswith("trace")}
    controls = []
    for arrays, branch, role in [(cpu_arrays, "cpu_state", "cpu_trace"),
                                 (gpu_arrays, "gpu_state", "gpu_trace")]:
        original = originals[role]
        for stage in CONTROL_STAGES:
            want = original[saved_name(original, stage)]
            got = arrays[branch + "." + stage]
            require(want.dtype == np.dtype("float32") and exact(got, want, np),
                    "Actual original control mismatch: " + role + "/" + stage)
            controls.append({"role": role, "stage": stage, "bit_exact": True,
                             "original_raw_sha256": array_record(want)["sha256"]})
    for arrays, branch, role in [(cpu_arrays, "cpu_state", "cpu_trace"),
                                 (gpu_arrays, "cpu_state", "cpu_trace"),
                                 (gpu_arrays, "gpu_state", "gpu_trace")]:
        original = originals[role]
        want = original[saved_name(original, "embedding")]
        require(want.dtype == np.dtype("float32") and exact(arrays[branch + ".input"], want, np),
                "Branch entry state is not original full saved tensor")
    stages = {name: decompose(cpu_arrays["cpu_state." + name], gpu_arrays["cpu_state." + name],
                              gpu_arrays["gpu_state." + name], np) for name in STAGES}
    # No block0 channel is selected from the downstream coordinate: attention
    # couples all rows/channels. All-row accounting is the prospective endpoint.
    first_difference = next((name for name in STAGES if name != "input" and
        not exact(cpu_arrays["cpu_state." + name], gpu_arrays["cpu_state." + name], np)), None)
    validate_plan(plan)
    require(all(sha(p) == h for p, h in bound.items()), "Input/source/artifact changed during comparison")
    report = {"kind": "fp32-frozen-state-crossover-block0-analysis-v1", "status": "validated_decomposition_only",
              "plan_sha256": args.plan_sha256, "controls": controls, "stages": stages,
              "segment_endpoint": "layer.0.hidden", "original_failure_context": FAILURE_CONTEXT,
              "first_captured_stage_with_same_entry_bit_difference": first_difference,
              "telescope_context": {role: {"path": path, "sha256": digest}
                                    for role, (path, digest) in PINS.items() if role.startswith("telescope_")},
              "payload_count": len(cpu_arrays) + len(gpu_arrays), "entry_states_checked": 3,
              "row112_in_every_stage": True, "source_and_input_closure": True,
              "artifact_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in bound.items()},
              "limits": ["Signed terms are descriptive engine/state crossover differences, not isolated-kernel error bounds.",
                         "At later stages the engine term includes different intermediate inputs within this segment.",
                         "Capture order is not a total causal order: Q/K/V are sibling branches, and pass-through hooks synchronize execution.",
                         "The first captured bit difference and largest local error do not identify the cause of the original V9 endpoint; no downstream suffix ran.",
                         "FP64 decomposition residual is recorded, not used to invent a model tolerance.",
                         "No original policy change, hidden-stage qualification, model rerun, performance or startup-attestation claim.",
                         "Stop after this one decomposition; the two decode attention failures are outside scope."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, report)
    print(json.dumps({"status": report["status"], "report_sha256": sha(output)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--cpu-execution", type=Path, required=True)
    p.add_argument("--cpu-execution-sha256", required=True)
    p.add_argument("--gpu-report", type=Path, required=True)
    p.add_argument("--gpu-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute-reviewed-plan", action="store_true")
    compare(p.parse_args())


if __name__ == "__main__":
    main()
