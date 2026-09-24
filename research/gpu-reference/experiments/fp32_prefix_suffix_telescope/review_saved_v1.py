"""Independent post-capture recomputation; imports no experiment implementation."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load

ROOT = Path(__file__).resolve().parents[4]
PLAN = "artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/plan.json"
CAPTURE = "artifacts/diagnostics/fp32-prefix-suffix-telescope-gpu-v1/report.json"
TENSORS = "artifacts/diagnostics/fp32-prefix-suffix-telescope-gpu-v1/tensors.safetensors"
ANALYSIS = "reference/fp32-prefix-suffix-telescope-decomposition-v1.json"
LAUNCH = "artifacts/diagnostics/fp32-prefix-suffix-telescope-launch-v1/finished.json"
PINS = {PLAN: "17afd8974541318b3380f4c75b20dfbb8434500db68e125d0d3bc490211bf561",
        CAPTURE: "1c9882505c5c381129435d1ae97ece3886a7f17dcd91e2007274971da9799c5a",
        TENSORS: "0f0a8d9c4de1af9ca10bc710768751d9da053c98e18851f312eeac9f247dacd8",
        ANALYSIS: "8bbe942e152340e6a0361e7e40aa8384dbd07b19177791ad2dcfe48e0f2a4f71",
        LAUNCH: "855d3e6d0c4665377f985398e866964b54c8efb831c4882c1159c5438e949db2"}
checked = {}
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / name


def bind(name, expected=None):
    path = resolve(name).resolve()
    actual = digest(path)
    check(expected is None or actual == expected, "Changed file: " + str(path))
    check(path not in checked or checked[path] == actual, "Conflicting file identity")
    checked[path] = actual
    return path


def read_json(name, expected=None):
    path = resolve(name).resolve()
    raw = path.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    check(expected is None or h == expected, "Changed JSON: " + str(path))
    check(path not in checked or checked[path] == h, "Conflicting JSON identity")
    checked[path] = h
    return json.loads(raw)


def raw_hash(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def identical(a, b, label):
    check(a.dtype == b.dtype == np.dtype("float32") and a.shape == b.shape
          and np.array_equal(a.view(np.uint32), b.view(np.uint32)), label)


def tensors(name, expected):
    path = resolve(name).resolve()
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == expected, "Changed tensor archive")
    check(path not in checked or checked[path] == expected, "Conflicting tensor archive identity")
    checked[path] = expected
    return load(raw)


def original_arrays(item, names):
    path = bind(item["path"], item["sha256"])
    result = {}
    with safe_open(str(path), framework="numpy") as handle:
        keys = set(handle.keys())
        for name in names:
            found = [key for key in (name, "prefill." + name) if key in keys]
            check(len(found) == 1, "Missing or ambiguous original tensor")
            result[name] = handle.get_tensor(found[0])
    return result


def exact_json(got, expected, label):
    if isinstance(expected, dict):
        check(isinstance(got, dict) and set(got) == set(expected), label + " keys")
        for key in expected:
            exact_json(got[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        check(isinstance(got, list) and len(got) == len(expected), label + " count")
        for i, value in enumerate(expected):
            exact_json(got[i], value, label + f"[{i}]")
    else:
        check(got == expected, label + f": {got!r} != {expected!r}")


def stats(value):
    return {"max_abs": float(np.max(np.abs(value))),
            "rms": float(np.sqrt(np.mean(np.square(value)))),
            "signed_mean": float(np.mean(value))}


def main():
    output = ROOT / "reference/fp32-prefix-suffix-telescope-independent-review-v1.json"
    check(not output.exists(), "Review output must be fresh")
    bind(__file__)
    plan, capture, analysis, launch = [read_json(name, PINS[name]) for name in (PLAN, CAPTURE, ANALYSIS, LAUNCH)]
    prep = read_json("artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/preparation.json")
    check(prep["plan_sha256"] == PINS[PLAN], "Preparation identity")
    branches = ["control_gpu", "E8", "E7", "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E9"]
    check(plan["branches"] == capture["completed_branches"] == branches, "Branch order")
    check(capture["controls_passed"] == branches[:3], "Control order")
    check(capture["native_block_calls"] == 54 and capture["native_layer9_qkv_calls"] == 11, "Call budget")
    check(capture["status"] == "three_controls_exact_eight_endpoints_captured"
          and capture["source_and_input_closure"] is True and capture["effective_compiled_blocks"] is False, "Capture acceptance")
    check(analysis["status"] == "one_valid_fixed_boundary_decomposition_only"
          and analysis["source_and_input_closure"] is True, "Analysis acceptance")
    check(capture["plan_sha256"] == analysis["plan_sha256"] == PINS[PLAN]
          and analysis["gpu_report_sha256"] == launch["gpu_report_sha256"] == PINS[CAPTURE], "Report joins")
    check(plan["runtime"] == capture["runtime"] == analysis["runtime"], "Runtime joins")
    check(launch["exit_code"] == 0 and launch["sources_unchanged"] is True
          and launch["source_sha256_before"] == launch["source_sha256_after"], "Launch source closure")
    for name, expected in launch["source_sha256_before"].items():
        bind(name, expected)
    for name, expected in launch["output_sha256"].items():
        bind(str(Path(LAUNCH).parent / name), expected)
    started = read_json(str(Path(LAUNCH).parent / "started.json"))
    check(all(launch[k] == v for k, v in started.items()), "Startup launch differs")

    # Recheck all small bound files and every tensor archive actually used below.
    # Large historical binaries/archives and weights retain the exporter's closure claim.
    deferred = {}
    for mapping in (capture["artifact_sha256"], analysis["artifact_sha256"]):
        for name, expected in mapping.items():
            path = resolve(name)
            check(name not in deferred or deferred[name] == expected, "Conflicting inherited identity")
            if path.stat().st_size <= 8 << 20:
                bind(name, expected)
            else:
                deferred[name] = expected
    for role, item in plan["inputs"].items():
        check(capture["artifact_sha256"][item["path"]] == item["sha256"], "Capture input not bound: " + role)
    for name, expected in plan["source_sha256"].items():
        check(capture["artifact_sha256"][name] == expected, "Capture source not bound")
        bind(name, expected)
    archive = bind("artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/source.zip", plan["source_archive_sha256"])
    with zipfile.ZipFile(archive) as source:
        check(set(source.namelist()) == set(plan["source_sha256"]), "Startup archive inventory")
        for name, expected in plan["source_sha256"].items():
            check(hashlib.sha256(source.read(name)).hexdigest() == expected, "Startup archive source differs")

    payload = tensors(TENSORS, PINS[TENSORS])
    check(capture["tensor_file_sha256"] == PINS[TENSORS], "Capture tensor join")
    check(set(payload) == set(capture["tensors"]) == {b + "." + s for b in branches for s in plan["stages"][b]}, "110-array inventory")
    check(len(payload) == 110, "Wrong array count")
    for name, value in payload.items():
        branch, stage = name.split(".", 1)
        shape = plan["stages"][branch][stage]
        check(value.dtype == np.dtype("float32") and list(value.shape) == shape and np.isfinite(value).all(), "Payload shape/dtype/finite")
        exact_json(capture["tensors"][name], {"shape": shape, "dtype": "F32", "raw_sha256": raw_hash(value)}, "Payload metadata")
    inputs = plan["inputs"]
    original_gpu_names = ["embedding"] + [f"layer.{i}.{s}" for i in range(9) for s in ("q", "k", "v", "attention", "hidden")] + ["layer.9.v"]
    gpu = original_arrays(inputs["gpu_trace"], original_gpu_names)
    cpu = original_arrays(inputs["cpu_trace"], ["embedding"] + [f"layer.{i}.hidden" for i in range(9)] + ["layer.9.v"])
    old_gpu = tensors(inputs["old_gpu_tensors"]["path"], inputs["old_gpu_tensors"]["sha256"])
    old_layer7 = tensors(inputs["layer7_gpu_tensors"]["path"], inputs["layer7_gpu_tensors"]["sha256"])
    old_downstream = tensors(inputs["downstream_gpu_tensors"]["path"], inputs["downstream_gpu_tensors"]["sha256"])
    known_controls = {}
    for stage in plan["stages"]["control_gpu"]:
        known_controls["control_gpu." + stage] = gpu["embedding" if stage == "input" else stage]
    for stage in plan["stages"]["E8"]:
        known_controls["E8." + stage] = old_gpu["cpu_state." + stage]
        identical(old_gpu["cpu_state." + stage], old_downstream["control_c." + stage], "Earlier E8 bridge")
    for stage in plan["stages"]["E7"]:
        known_controls["E7." + stage] = (old_layer7["cpu_state." + stage]
            if stage == "input" or stage.startswith("layer.7.") else old_downstream["substitution_s." + stage])
    identical(old_layer7["cpu_state.layer.7.hidden"], old_downstream["substitution_s.input"], "Earlier E7 composition")
    controls = []
    for name, expected in known_controls.items():
        identical(payload[name], expected, "Fresh control " + name)
        branch, stage = name.split(".", 1)
        controls.append({"branch": branch, "stage": stage, "bit_exact": True})
    check(len(controls) == 94, "Control count")
    exact_json(analysis["fresh_control_checks"], controls, "Control reporting")
    for row in capture["controls"]:
        exact_json(row["stages"], [{"stage": s, "bit_exact": True, "mismatched_elements": 0}
            for s in plan["stages"][row["branch"]]], "Capture control reporting")
    entry_hashes = {}
    for branch in branches:
        expected = (gpu["embedding"] if branch == "control_gpu" else
                    cpu["embedding" if branch == "E0" else f"layer.{int(branch[1:]) - 1}.hidden"])
        identical(payload[branch + ".input"], expected, "Complete saved entry " + branch)
        entry_hashes["G0" if branch == "control_gpu" else "C" + branch[1:]] = raw_hash(expected)
    exact_json(plan["state_raw_sha256"], entry_hashes, "Prepared entry identities")
    check(capture["historical_evidence"] == plan["historical_evidence"] == analysis["historical_evidence"], "Historical proof reporting")
    exact_json(plan["historical_evidence"]["state_raw_sha256"], entry_hashes, "Evidence entry identities")
    exact_json(plan["historical_evidence"]["control_raw_sha256"], {
        branch: {s: raw_hash(known_controls[branch + "." + s]) for s in plan["stages"][branch]}
        for branch in branches[:3]}, "Evidence complete control identities")
    check((analysis["fresh_control_arrays_exact"], analysis["new_entry_states_exact"], analysis["new_payloads_checked"]) == (94, 11, 110), "Reported counts")
    old_report = read_json(inputs["old_gpu_report"]["path"], inputs["old_gpu_report"]["sha256"])
    check(capture["effective_torch_math"] == old_report["effective_torch_math"]
          and capture["effective_threads"] == old_report["effective_threads"], "Effective math/thread runtime")
    check({k:v for k,v in capture["environment"].items() if k != "free_memory_mib"}
          == {k:v for k,v in old_report["environment"].items() if k != "free_memory_mib"}, "Effective GPU/package environment")
    policy = read_json(inputs["policy"]["path"], inputs["policy"]["sha256"])
    check(policy["stages"]["layer.9.v"]["absolute_tolerance"] == 0.005096435546875, "Original bound")

    # Independent endpoint stack and adjacent finite differences, no imported comparison code.
    endpoints_f32 = {"A": gpu["layer.9.v"], "D": cpu["layer.9.v"],
                     **{f"E{i}": payload[f"E{i}.layer.9.v"] for i in range(10)}}
    exact_json(analysis["endpoint_raw_sha256"], {k: raw_hash(v) for k,v in endpoints_f32.items()}, "Endpoint identities")
    chain_names = ["A"] + [f"E{i}" for i in range(10)] + ["D"]
    chain = np.stack([endpoints_f32[k].astype(np.float64) for k in chain_names])
    contributions = np.diff(chain, axis=0)
    labels = ["E0_minus_A_embedding_propagated"] + [f"E{i+1}_minus_E{i}_block{i}_conditional" for i in range(9)] + ["D_minus_E9_layer9_pre_qkv_engine"]
    total = chain[-1] - chain[0]
    # Ordered FP64 sum checks the same claimed arithmetic identity independently.
    addition = np.zeros_like(total)
    for part in contributions:
        addition += part
    residual = total - addition
    check(np.count_nonzero(residual) == 0, "Full-array telescope residual")
    all_differences = {"D_minus_A_original": total, **dict(zip(labels, contributions))}
    worst = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(total)), total.shape))
    expected_global = {k: stats(v) for k,v in all_differences.items()}
    expected_rows = [{"row": r, **{k: stats(v[r]) for k,v in all_differences.items()},
                     "fp64_telescoping_residual_max_abs": float(np.max(np.abs(residual[r])))} for r in range(144)]
    expected_summary = {"shape": [144, 16, 64], "elements": 144 * 16 * 64,
        "global": expected_global, "rows": expected_rows,
        "endpoints_relative_to_A": {f"E{i}": stats(chain[i+1]-chain[0]) for i in range(10)},
        "worst_original_coordinate": list(worst),
        "at_worst_original_coordinate": {"A": float(chain[0][worst]), "D": float(chain[-1][worst]),
            "endpoints": {f"E{i}": float(chain[i+1][worst]) for i in range(10)},
            **{k: float(v[worst]) for k,v in all_differences.items()}},
        "fp64_telescoping_residual_max_abs": 0.0,
        "term_rms_sum_over_total_rms": sum(stats(v)["rms"] for v in contributions) / stats(total)["rms"]}
    exact_json(analysis["endpoint_decomposition"], expected_summary, "All-array/per-row reporting")
    point = (112,14,2)
    values = {name: float(endpoints_f32[name][point]) for name in chain_names}
    signed = {name: float(value[point]) for name,value in zip(labels, contributions)}
    total_point = float(total[point])
    fixed = {"stage": "layer.9.v", "coordinate": list(point), "original_absolute_bound": 0.005096435546875,
        "A": values["A"], "D": values["D"], "endpoints": {f"E{i}": values[f"E{i}"] for i in range(10)},
        "D_minus_A_original": total_point, "signed_terms": signed,
        "signed_fractions_of_original_total": {k:v/total_point for k,v in signed.items()},
        "sum_abs_terms_over_abs_total": sum(abs(v) for v in signed.values()) / abs(total_point),
        "fp64_telescoping_residual": total_point - sum(signed.values()),
        "terms_exceed_original_bound_descriptive_only": {k:abs(v)>0.005096435546875 for k,v in signed.items()}}
    exact_json(analysis["fixed_endpoint"], fixed, "Fixed endpoint reporting")
    for path, expected in checked.items():
        check(digest(path) == expected, "Review input changed before completion: " + str(path))
    inherited = {k:v for k,v in deferred.items() if resolve(k).resolve() not in checked}
    receipt = {"kind": "fp32-prefix-suffix-telescope-independent-review-v1", "status": "saved_arrays_and_telescope_verified",
        "plan_sha256": PINS[PLAN], "gpu_report_sha256": PINS[CAPTURE], "tensor_sha256": PINS[TENSORS],
        "analysis_sha256": PINS[ANALYSIS], "launch_sha256": PINS[LAUNCH],
        "method": "Independent array joins, uint32 bit equality, np.diff on FP64 stacked endpoints and explicit signed accumulation; no imported experiment functions or comparison-script invocation.",
        "checks": checks, "fresh_control_arrays_bit_exact": 94, "full_entry_states_bit_exact": 11,
        "fresh_payloads_verified": 110, "control_shapes": {b:len(plan["stages"][b]) for b in branches[:3]},
        "rows_verified": 144, "row_term_statistics_verified": 144 * 12,
        "endpoint_elements": int(total.size), "full_array_residual_nonzero_elements": 0,
        "all_reported_numeric_summaries_exact": True, "fixed_endpoint": fixed,
        "global_statistics": expected_global,
        "files_directly_rehashed_before_and_end": len(checked),
        "direct_artifact_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):h for p,h in checked.items()},
        "inherited_large_artifact_closure": inherited,
        "source_input_closure_scope": "Direct before/end hashes cover every source/archive/report used here and all six saved tensor archives used for array comparisons. Remaining large historical artifacts and model weights inherit accepted exporter before/end closure; this review does not rehash the weights.",
        "runtime": {"python_scope": "Offline NumPy/safetensors only", "numpy_version": np.__version__, "gpu_or_model_execution": False},
        "limits": ["This independently recomputes results, but the review author also authored the experiment; the source review was by a separate agent.",
                   "Conditional fixed-order terms are not unique independent kernel causes; attention mixes all rows.",
                   "Historical startup source gaps and changed allocation/observer history remain explicit.",
                   "Original absolute bound and all ten intermediate failures remain unchanged; no production fix or qualification is claimed."]}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "checks": checks, "files": len(checked), "inherited_large": len(inherited), "receipt_sha256": digest(output)}))


if __name__ == "__main__":
    main()
