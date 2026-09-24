"""Extend the preserved historical loader; payload work occurs only on call."""
import hashlib

from contract import (PINS, STAGES, BRANCHES, CONTROL_BRANCHES, ENTRY_KEYS,
                      DOWNSTREAM_STAGES, require, resolve, read_bound)
from legacy_evidence import (load_evidence as load_prior, raw_sha, exact, saved_name,
                             assert_f32, bind_extra, validate_reported_controls)


def load_archive(role):
    from safetensors.numpy import load
    path, digest = PINS[role]
    raw = resolve(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "Changed archive: " + role)
    return load(raw)


def load_evidence(bound):
    import numpy as np
    from safetensors import safe_open

    prior = load_prior(bound)
    reports = prior["reports"]
    reports.update({role: read_bound(resolve(path), h) for role, (path, h) in PINS.items()
                    if role.startswith("downstream_") and path.endswith(".json")})
    plan, capture, analysis, review = (reports["downstream_" + r]
        for r in ("plan", "gpu_report", "analysis", "review"))
    require(plan["kind"] == "fp32-crossover-downstream-v1" and plan["stages"] == DOWNSTREAM_STAGES
            and plan["branches"] == ["control_c", "substitution_s"], "Wrong historical downstream scope")
    require(capture["kind"] == "fp32-crossover-downstream-v1-gpu"
            and capture["status"] == "control_exact_substitution_captured"
            and capture["source_and_input_closure"] is True
            and capture["control_passed"] is True and capture["substitution_executed"] is True
            and capture["native_layer8_calls"] == capture["native_layer9_qkv_calls"] == 2
            and capture["plan_sha256"] == PINS["downstream_plan"][1], "Downstream capture unaccepted")
    validate_reported_controls(capture, list(DOWNSTREAM_STAGES), True)
    require(analysis["status"] == "one_valid_downstream_decomposition_only"
            and analysis["source_and_input_closure"] is True
            and analysis["gpu_report_sha256"] == PINS["downstream_gpu_report"][1]
            and analysis["plan_sha256"] == PINS["downstream_plan"][1], "Downstream analysis unaccepted")
    require(review["status"] == "saved_controls_and_telescoping_verified"
            and review["analysis_sha256"] == PINS["downstream_analysis"][1]
            and review["gpu_report_sha256"] == PINS["downstream_gpu_report"][1]
            and review["plan_sha256"] == PINS["downstream_plan"][1], "Downstream review join differs")
    for role in ("gpu_trace", "cpu_trace", "policy", "model_manifest"):
        require(plan["inputs"][role] == {"path": PINS[role][0], "sha256": PINS[role][1]},
                "Downstream checkpoint/trace/policy join differs")
    for name, h in capture["artifact_sha256"].items():
        bind_extra(bound, name, h)
    require(capture["historical_evidence"] == prior["proof"], "Prior proof changed")
    require(capture["runtime"] == plan["runtime"], "Historical runtime/plan mismatch")
    for field in ("effective_torch_math", "effective_threads"):
        require(capture[field] == reports["old_gpu_report"][field], "Historical runtime changed: " + field)
    require({k: v for k, v in capture["environment"].items() if k != "free_memory_mib"}
            == {k: v for k, v in reports["old_gpu_report"]["environment"].items() if k != "free_memory_mib"},
            "Historical environment changed")
    downstream = load_archive("downstream_gpu_tensors")
    require(capture["tensor_file_sha256"] == PINS["downstream_gpu_tensors"][1], "Downstream archive/report mismatch")
    expected_names = {b + "." + s for b in ("control_c", "substitution_s") for s in DOWNSTREAM_STAGES}
    require(set(downstream) == set(capture["tensors"]) == expected_names, "Incomplete downstream arrays")
    for name, value in downstream.items():
        stage = name.split(".", 1)[1]
        assert_f32(value, DOWNSTREAM_STAGES[stage], np)
        require(capture["tensors"][name] == {"shape": DOWNSTREAM_STAGES[stage], "dtype": "F32",
                "raw_sha256": raw_sha(value)}, "Downstream payload identity differs")
    for stage in DOWNSTREAM_STAGES:
        require(exact(downstream["control_c." + stage], prior["stages"]["C"][stage], np),
                "Downstream C stage join differs: " + stage)
    require(exact(downstream["substitution_s.input"], prior["states"]["S"], np), "Downstream S join differs")

    entries, gpu_control = {}, {}
    with safe_open(str(resolve(PINS["gpu_trace"][0])), framework="np") as handle:
        for stage, shape in STAGES["control_gpu"].items():
            original = "embedding" if stage == "input" else stage
            value = handle.get_tensor(saved_name(handle.keys(), original))
            assert_f32(value, shape, np)
            gpu_control[stage] = value
        entries["G0"] = gpu_control["input"]
    with safe_open(str(resolve(PINS["cpu_trace"][0])), framework="np") as handle:
        for b in range(10):
            key = "embedding" if b == 0 else f"layer.{b - 1}.hidden"
            entries["C" + str(b)] = handle.get_tensor(saved_name(handle.keys(), key))
            assert_f32(entries["C" + str(b)], [144, 768], np)
        d = handle.get_tensor(saved_name(handle.keys(), "layer.9.v"))
        assert_f32(d, [144, 16, 64], np)
    require(set(entries) == set(ENTRY_KEYS), "Missing boundary state")
    require(exact(d, prior["stages"]["D"]["layer.9.v"], np)
            and exact(gpu_control["layer.9.v"], prior["stages"]["A"]["layer.9.v"], np),
            "Original endpoint join differs")
    require(exact(entries["C8"], prior["states"]["C7"], np), "CPU7 entry join differs")
    # The archive has already been fully validated by the preserved loader.
    # Rechecking its immutable bytes supplies the existing fourteen E7 stages.
    layer7 = load_archive("layer7_gpu_tensors")
    require(exact(entries["C7"], layer7["cpu_state.input"], np), "CPU6 entry join differs")
    e7 = {"input": entries["C7"]}
    e7.update({s: layer7["cpu_state." + s] for s in reports["layer7_plan"]["stages"] if s != "input"})
    e7.update({s: downstream["substitution_s." + s] for s in DOWNSTREAM_STAGES if s != "input"})
    require(exact(e7["layer.7.hidden"], downstream["substitution_s.input"], np), "E7 composed boundary changed")
    controls = {"control_gpu": gpu_control, "E8": prior["stages"]["C"], "E7": e7}
    require(list(controls) == CONTROL_BRANCHES, "Wrong control order")
    for branch, arrays in controls.items():
        require(set(arrays) == set(STAGES[branch]), "Missing complete control")
        for stage, value in arrays.items():
            assert_f32(value, STAGES[branch][stage], np)
    endpoints = {"A": gpu_control["layer.9.v"], "D": d,
                 "E7": controls["E7"]["layer.9.v"], "E8": controls["E8"]["layer.9.v"]}
    coordinate = (112, 14, 2)
    require(tuple(float(endpoints[k][coordinate]) for k in ("A", "D", "E7", "E8"))
            == (-19.037837982177734, -19.03121566772461, -19.031208038330078, -19.031253814697266),
            "Known failing endpoint values differ")
    proof = {"prior": prior["proof"], "downstream_payloads_checked": 34,
             "downstream_control_stage_joins": 17, "downstream_S_join_exact": True,
             "original_gpu_control_tensors": 47, "original_cpu_entries_and_endpoint": 11,
             "control_stage_counts": {b: len(STAGES[b]) for b in CONTROL_BRANCHES},
             "state_raw_sha256": {k: raw_sha(v) for k, v in entries.items()},
             "control_raw_sha256": {b: {k: raw_sha(v) for k, v in arrays.items()} for b, arrays in controls.items()},
             "endpoint_raw_sha256": {k: raw_sha(v) for k, v in endpoints.items()},
             "limits": "Original startup gaps and changed observation/allocation history persist; controls are complete saved finite inputs, not universal kernel proofs."}
    return {"entries": entries, "controls": controls, "endpoints": endpoints,
            "positions": prior["positions"], "reports": reports, "proof": proof}
