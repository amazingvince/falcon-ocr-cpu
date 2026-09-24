"""Read and validate saved evidence only when explicitly called after release."""
import hashlib

from legacy_contract import (PINS, STAGES, ORIGINAL_CONTROLS, LAYER7_CONTROLS, STATE_HASHES,
                      require, resolve, read_bound)


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def exact(a, b, np):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def saved_name(keys, name):
    matches = [key for key in (name, "prefill." + name) if key in keys]
    require(len(matches) == 1, "Missing/ambiguous original tensor: " + name)
    return matches[0]


def assert_f32(array, shape, np):
    require(array.dtype == np.dtype("float32") and list(array.shape) == shape
            and np.isfinite(array).all(), "Invalid saved finite FP32 shape/dtype")


def bind_extra(bound, name, digest):
    path = resolve(name.replace("\\", "/"))
    require(path not in bound or bound[path] == digest, "Conflicting historical evidence identity")
    bound[path] = digest
    return path


def validate_reported_controls(report, stages, gpu):
    require([r["stage"] for r in report["controls"]] == stages, "Wrong historical control inventory")
    for row in report["controls"]:
        require(row["bit_exact" if gpu else "passed"] is True
                and row["mismatched_elements" if gpu else "bit_mismatches"] == 0,
                "Historical control failed")


def load_evidence(bound):
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import load

    reports = {role: read_bound(resolve(path), digest) for role, (path, digest) in PINS.items()
               if path.endswith(".json")}
    old, layer7 = reports["old_plan"], reports["layer7_plan"]
    require(old["stages"] == STAGES and old["control_stages"] == ORIGINAL_CONTROLS,
            "Old segment stage contract differs")
    require(old["original_state_key"] == "layer.7.hidden" and layer7["original_state_key"] == "layer.6.hidden",
            "Wrong original state entry points")
    require(layer7["control_stages"] == LAYER7_CONTROLS and len(layer7["stages"]) == 14,
            "Wrong layer7 scope")
    for prefix, plan, controls in (("old", old, ORIGINAL_CONTROLS), ("layer7", layer7, LAYER7_CONTROLS)):
        g, c, execution = (reports[prefix + s] for s in ("_gpu_report", "_cpu_report", "_cpu_execution"))
        plan_sha = PINS[prefix + "_plan"][1]
        require(g["status"] == "gpu_arm_control_passed_crossover_captured"
                and g["gpu_control_passed"] is True and g["crossover_executed"] is True
                and g["source_and_input_closure"] is True and g["plan_sha256"] == plan_sha,
                "Historical GPU arm unaccepted")
        require(c["status"] == "control_exact" and c["arithmetic_changed"] is False
                and c["branch"] == "cpu_state" and c["plan_sha256"] == plan_sha,
                "Historical CPU arm unaccepted")
        require(execution["status"] == "cpu_control_exact" and execution["source_and_input_closure"] is True
                and execution["plan_sha256"] == plan_sha, "Historical CPU capture not closed")
        validate_reported_controls(g, controls, True)
        validate_reported_controls(c, controls, False)
        require(g["runtime"] == plan["runtime"] and g["runtime"]["precision"] == "fp32"
                and g["runtime"]["gpu_cache_capacity"] == 256 and g["runtime"]["compiled_blocks"] is False
                and g["effective_torch_math"]["cuda_matmul_allow_tf32"] is False,
                "Historical GPU runtime differs")
        for role in ("gpu_trace", "cpu_trace", "policy", "model_manifest"):
            require(plan["inputs"][role] == {"path": PINS[role][0], "sha256": PINS[role][1]},
                    "Historical checkpoint/fixture/policy join differs")
        for name, digest in execution["artifact_sha256"].items():
            bind_extra(bound, name, digest)
        cpu_report_path = resolve(PINS[prefix + "_cpu_report"][0])
        require(bound.get(cpu_report_path) == PINS[prefix + "_cpu_report"][1], "CPU report not bound by execution")
        directory = cpu_report_path.parent.parent
        prep_path = directory / "preparation.json"
        build_path = directory / "build.json"
        require(prep_path in bound and build_path in bound, "Missing historical build/preparation closure")
        prep = read_bound(prep_path, bound[prep_path])
        build = read_bound(build_path, bound[build_path])
        require(prep["plan_sha256"] == build["plan_sha256"] == plan_sha
                and c["test_binary_sha256"] == build["binary_sha256"], "Historical build/control identity mismatch")
        bound[directory / "source.zip"] = prep["source_archive_sha256"]
    require(reports["old_analysis"]["status"] == "validated_decomposition_only"
            and reports["old_analysis"]["source_and_input_closure"] is True,
            "Previous decomposition not accepted")
    require(reports["old_review"]["status"] == reports["layer7_review"]["status"]
            == "saved_controls_and_signed_decomposition_verified", "Independent reviews not accepted")
    require(reports["layer7_completion"]["status"] == "one_valid_decomposition_completed",
            "Layer7 capture incomplete")

    archives = {}
    for role, shape_map, branches, report_role, digest_key in (
        ("old_gpu_tensors", STAGES, ["gpu_state", "cpu_state"], "old_gpu_report", "sha256"),
        ("old_cpu_tensors", STAGES, ["cpu_state"], "old_cpu_report", "raw_sha256"),
        ("layer7_gpu_tensors", layer7["stages"], ["gpu_state", "cpu_state"], "layer7_gpu_report", "sha256"),
        ("layer7_cpu_tensors", layer7["stages"], ["cpu_state"], "layer7_cpu_report", "raw_sha256"),
    ):
        path, digest = PINS[role]
        raw = resolve(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest == reports[report_role]["tensor_file_sha256"],
                "Historical archive hash differs")
        arrays = load(raw)
        metadata = reports[report_role]["tensors"]
        expected = {b + "." + s for b in branches for s in shape_map}
        require(set(arrays) == set(metadata) == expected, "Historical stage inventory differs")
        for branch in branches:
            for stage, shape in shape_map.items():
                name = branch + "." + stage
                assert_f32(arrays[name], shape, np)
                require(metadata[name]["shape"] == shape and metadata[name]["dtype"] == "F32"
                        and metadata[name][digest_key] == raw_sha(arrays[name]), "Historical raw stage hash differs")
        archives[role] = arrays

    originals = {}
    for role in ("gpu_trace", "cpu_trace"):
        with safe_open(str(resolve(PINS[role][0])), framework="np") as handle:
            selected = set(ORIGINAL_CONTROLS + LAYER7_CONTROLS + ["layer.6.hidden", "layer.7.hidden"])
            originals[role] = {stage: handle.get_tensor(saved_name(handle.keys(), stage)) for stage in selected}
            for stage, value in originals[role].items():
                assert_f32(value, ([144, 768] if stage.endswith("hidden") else
                                  [144, 1024] if stage.endswith("attention") else [144, 16, 64]), np)
            if role == "gpu_trace":
                positions = {name: handle.get_tensor(name) for name in ("tokens", "pos_t", "pos_hw")}
    require(positions["tokens"].dtype == positions["pos_t"].dtype == np.dtype("int64")
            and positions["tokens"].shape == positions["pos_t"].shape == (144,)
            and positions["pos_hw"].dtype == np.dtype("float32") and positions["pos_hw"].shape == (144, 2),
            "Original mask/position input shape or dtype differs")
    checks = []
    for prefix, controls in (("old", ORIGINAL_CONTROLS), ("layer7", LAYER7_CONTROLS)):
        for platform, branch in (("gpu", "gpu_state"), ("cpu", "cpu_state")):
            arrays = archives[prefix + "_" + platform + "_tensors"]
            for stage in controls:
                require(exact(arrays[branch + "." + stage], originals[platform + "_trace"][stage], np),
                        "Original historical control mismatch: " + prefix + "/" + platform + "/" + stage)
                checks.append({"experiment": prefix, "platform": platform, "stage": stage, "bit_exact": True})
    og, oc, lg, lc = (archives[k] for k in ("old_gpu_tensors", "old_cpu_tensors", "layer7_gpu_tensors", "layer7_cpu_tensors"))
    joins = [(og["gpu_state.input"], originals["gpu_trace"]["layer.7.hidden"]),
             (og["cpu_state.input"], originals["cpu_trace"]["layer.7.hidden"]),
             (oc["cpu_state.input"], og["cpu_state.input"]),
             (lg["gpu_state.input"], originals["gpu_trace"]["layer.6.hidden"]),
             (lg["cpu_state.input"], originals["cpu_trace"]["layer.6.hidden"]),
             (lc["cpu_state.input"], lg["cpu_state.input"]),
             (lg["gpu_state.layer.7.hidden"], og["gpu_state.input"]),
             (lc["cpu_state.layer.7.hidden"], og["cpu_state.input"])]
    require(all(exact(a, b, np) for a, b in joins), "Full entry-state join differs")
    states = {"G7": og["gpu_state.input"], "C7": og["cpu_state.input"], "S": lg["cpu_state.layer.7.hidden"]}
    require({name: raw_sha(value) for name, value in states.items()} == STATE_HASHES, "Selected full states changed")
    stages = {"A": {s: og["gpu_state." + s] for s in STAGES},
              "C": {s: og["cpu_state." + s] for s in STAGES},
              "D": {s: oc["cpu_state." + s] for s in STAGES}}
    proof = {"historical_original_controls_exact": checks, "complete_state_joins_exact": len(joins),
             "historical_stage_payloads_checked": sum(len(a) for a in archives.values()),
             "state_raw_sha256": STATE_HASHES,
             "position_raw_sha256": {name: raw_sha(v) for name, v in positions.items()},
             "historical_sources": "Preserved build archives and accepted source receipts; current live Rust sources are not retroactive startup evidence."}
    return {"states": states, "positions": positions, "stages": stages, "reports": reports, "proof": proof}
