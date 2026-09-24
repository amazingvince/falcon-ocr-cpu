"""Fixed eleven-branch diagnostic; imports do no I/O or numerical work."""
from pathlib import Path

from legacy_contract import (ROOT, UUID, PINS as OLD_PINS, RUNTIME as OLD_RUNTIME,
                             STAGES as DOWNSTREAM_STAGES, ENDPOINT,
                             require, sha, resolve, read_bound, write_new, stable)

KIND = "fp32-prefix-suffix-telescope-v1"
HERE = "experiments/fp32_prefix_suffix_telescope/"
PRIOR = "experiments/fp32_crossover_downstream/"
PRESERVED_SOURCE_PINS = {
    PRIOR + "contract.py": "fd10303f9947d1c18d0b2b1d8553e8a3510bbb7c8e4fc29909c450e910e5a56f",
    PRIOR + "evidence.py": "76defc9db95d06566247d2f396227caf8e4ae4177928e8f08a46adf99d6c24ff",
    PRIOR + "native_segment.py": "f93a4d961c87848c4ad7f89be512ab46c204e77cece3a898747d7ae1e7690ba9",
    PRIOR + "export_gpu.py": "0923acc8164565de4d2d3fd1922a55b466741a8fcbe04cc366d42b4da7e25eeb",
}
PINS = {**OLD_PINS,
    "downstream_plan": ("artifacts/diagnostics/fp32-crossover-downstream-plan-v1/plan.json", "89d79b4717157f9d66c43092c9c83acd80880e0f823550e2a34111dafc6e7c36"),
    "downstream_gpu_report": ("artifacts/diagnostics/fp32-crossover-downstream-gpu-v1/report.json", "9d14800ac47f217afeb0f390e9cf96271fbb8e8d4765cdfb424d75eea43fbd78"),
    "downstream_gpu_tensors": ("artifacts/diagnostics/fp32-crossover-downstream-gpu-v1/tensors.safetensors", "37ef2f78a224032d7193de5504ae4087259827fd23d4dfabcd60ade8649467a3"),
    "downstream_analysis": ("reference/fp32-crossover-downstream-decomposition-v1.json", "32d15683c8d226f4d0ea9163147cb6cee1d19b823a8d3c89fc5ec1861e792806"),
    "downstream_review": ("reference/fp32-crossover-downstream-independent-review-v1.json", "90b753b3e754c1e287f81cb589efafe2b98513a47335be83da0ed8d08728743d"),
}
RUNTIME = {k: v for k, v in OLD_RUNTIME.items() if k not in ("layer", "next_layer")}
RUNTIME.update(first_layer=0, last_full_layer=8, endpoint_layer=9,
               native_block_calls=54, native_layer9_qkv_calls=11)
BRANCHES = ["control_gpu", "E8", "E7"] + ["E" + str(b) for b in range(7)] + ["E9"]
CONTROL_BRANCHES = BRANCHES[:3]
START_LAYER = {"control_gpu": 0, **{"E" + str(b): b for b in range(10)}}
ENTRY_STATE = {"control_gpu": "G0", **{"E" + str(b): "C" + str(b) for b in range(10)}}
ENTRY_KEYS = ["G0"] + ["C" + str(b) for b in range(10)]


def block_stages(layer):
    return {name.replace("layer.8.", f"layer.{layer}."): shape
            for name, shape in DOWNSTREAM_STAGES.items() if name.startswith("layer.8.")}


STAGES = {
    "control_gpu": {"input": [144, 768], **{f"layer.{i}.{s}": block_stages(i)[f"layer.{i}.{s}"]
        for i in range(9) for s in ("q", "k", "v", "attention", "hidden")}, "layer.9.v": [144, 16, 64]},
    "E8": dict(DOWNSTREAM_STAGES),
    "E7": {"input": [144, 768], **block_stages(7), **block_stages(8),
           **{s: v for s, v in DOWNSTREAM_STAGES.items() if s.startswith("layer.9.")}},
}
STAGES.update({b: {"input": [144, 768], "layer.9.v": [144, 16, 64]}
               for b in BRANCHES if b not in CONTROL_BRANCHES})
SOURCE_FILES = [HERE + n for n in ("contract.py", "legacy_contract.py", "legacy_evidence.py",
    "evidence.py", "native_segment.py", "source_guard.py", "prepare.py", "export_gpu.py",
    "compare.py", "test_source.py", "README.md")]
SOURCE_FILES += list(PRESERVED_SOURCE_PINS) + ["scripts/export_reference.py",
    "scripts/reference_preflight.py", "scripts/fetch_reference.py", "requirements/reference-lock.txt",
    "reference/manifest.json", "reference/fp32-prefix-suffix-telescope-assessment-v1.md"]


def validate_shape_contract(plan):
    require(plan.get("kind") == KIND and plan.get("runtime") == RUNTIME, "Wrong fixed runtime")
    require(plan.get("branches") == BRANCHES and plan.get("control_branches") == CONTROL_BRANCHES,
            "Wrong fixed branch/control order")
    require(plan.get("stages") == STAGES and plan.get("start_layer") == START_LAYER
            and plan.get("entry_state") == ENTRY_STATE and plan.get("endpoint") == ENDPOINT,
            "Wrong stage/entry/endpoint contract")
    require(plan.get("inputs") == {k: {"path": p, "sha256": h} for k, (p, h) in PINS.items()},
            "Changed historical evidence pins")
    states = plan.get("state_raw_sha256", {})
    require(set(states) == set(ENTRY_KEYS) and all(isinstance(h, str) and len(h) == 64
            and set(h) <= set("0123456789abcdef") for h in states.values()), "Missing complete entry identity")
    sources = plan.get("source_sha256", {})
    require(set(SOURCE_FILES) <= set(sources), "Missing source closure")
    require(all(sources[p] == h for p, h in PRESERVED_SOURCE_PINS.items()), "Preserved source changed")
    require(plan.get("model_directory") == "artifacts/model", "Wrong checkpoint path")


def plan_window(plan_path, expected):
    plan = read_bound(plan_path, expected)
    validate_shape_contract(plan)
    bound = {Path(plan_path).resolve(): expected}
    for item in plan["inputs"].values():
        bound[resolve(item["path"])] = item["sha256"]
    for name, value in plan["source_sha256"].items():
        p = resolve(name)
        require(p not in bound or bound[p] == value, "Conflicting source identity")
        bound[p] = value
    manifest = read_bound(resolve(PINS["model_manifest"][0]), PINS["model_manifest"][1])
    require(plan.get("model_files") == manifest["files"], "Model asset set changed")
    for name, item in manifest["files"].items():
        p = resolve("artifacts/model/" + name)
        require(p.stat().st_size == item["bytes"], "Model asset size differs")
        require(p not in bound or bound[p] == item["sha256"], "Conflicting asset identity")
        bound[p] = item["sha256"]
    bound[plan_path.parent / "source.zip"] = plan["source_archive_sha256"]
    stable(bound)
    return plan, bound


def require_next_branch(branch, completed, controls_passed):
    require(len(completed) < len(BRANCHES) and completed == BRANCHES[:len(completed)]
            and branch == BRANCHES[len(completed)], "Branch repeated/reordered")
    required = CONTROL_BRANCHES[:min(len(completed), len(CONTROL_BRANCHES))]
    require(controls_passed == required, "Cannot continue after a failed or missing exact control")
