"""Fixed crossover contract. Importing this module performs no I/O or execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
KIND = "fp32-frozen-state-crossover-block0-v1"
PINS = {
    "gpu_trace": ("artifacts/reference/smoke-fp32/trace.safetensors", "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4"),
    "cpu_trace": ("artifacts/cpu/smoke-trace-sinks-pairwise.safetensors", "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309"),
    "policy": ("reference/tolerances-smoke-fp32-v1.json", "8f6382a4386b15f76e6984679208a44d0fa644b9d55a21ce579bbb26e0bfbe1c"),
    "telescope_plan": ("artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/plan.json", "17afd8974541318b3380f4c75b20dfbb8434500db68e125d0d3bc490211bf561"),
    "telescope_capture": ("artifacts/diagnostics/fp32-prefix-suffix-telescope-gpu-v1/report.json", "1c9882505c5c381129435d1ae97ece3886a7f17dcd91e2007274971da9799c5a"),
    "telescope_analysis": ("reference/fp32-prefix-suffix-telescope-decomposition-v1.json", "8bbe942e152340e6a0361e7e40aa8384dbd07b19177791ad2dcfe48e0f2a4f71"),
    "telescope_review": ("reference/fp32-prefix-suffix-telescope-independent-review-v1.json", "c11085dd7e6b85afd654a843247157a90230f9329d44ed830711b626a538d329"),
    "reused_source_plan": ("artifacts/diagnostics/fp32-crossover-layer7-v1/plan.json", "832d98ec8ba2807935017d0eb7ec2b5dca9ecfec0680211a5b41313f8554f4d2"),
}
METADATA = {
    "gpu_metadata": "artifacts/reference/smoke-fp32/metadata.json",
    "cpu_metadata": "artifacts/cpu/smoke-trace-sinks-pairwise.json",
    "model_manifest": "artifacts/model/artifact-manifest.json",
    "baseline_comparison": "reference/windows-rust-smoke-fp32-sinks-pairwise.json",
}
MODEL_REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
WEIGHTS_SHA = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"
PRODUCTION_MODEL_SHA = "a048d6d618fc800c8331cb1d8eae51ced65d813d93784391b65c77c02cfd2fe6"
PRODUCTION_KERNELS_SHA = "19f1f18e164fffcab63dd1d747aae76f3a4a75dfcc15edebacd46b5dc32f7f16"
FROZEN_TOOLCHAIN = "artifacts/diagnostics/fp32-crossover-layer7-v1/project/rust-toolchain.toml"
RUNTIME = {
    "precision": "fp32", "rows": 144, "dim": 768, "heads": 16,
    "kv_heads": 8, "head_dim": 64, "ffn_dim": 2304,
    "layer": 0, "gpu_cache_capacity": 256,
    "rust_cache_capacity": 161, "rust_threads": 4, "rust_backend": "avx2", "rust_toolchain": "1.92.0",
    "tf32": False, "flex_float32_precision": "ieee", "compiled_blocks": False,
}
STAGES = {
    "input": [144, 768],
    "layer.0.attention_norm": [144, 768], "layer.0.qkv": [144, 2048],
    "layer.0.q": [144, 16, 64], "layer.0.k": [144, 16, 64],
    "layer.0.v": [144, 16, 64], "layer.0.attention": [144, 1024],
    "layer.0.wo": [144, 768], "layer.0.attention_residual": [144, 768],
    "layer.0.ffn_norm": [144, 768], "layer.0.w13": [144, 4608],
    "layer.0.gate": [144, 2304], "layer.0.w2": [144, 768],
    "layer.0.hidden": [144, 768],
}
CONTROL_STAGES = ["layer.0.q", "layer.0.k", "layer.0.v", "layer.0.attention",
                  "layer.0.hidden"]
BUDGET = {"rust_native_blocks": 1, "gpu_native_blocks": 2,
          "suffix_blocks": 0, "generation_steps": 0}
FAILURE_CONTEXT = {"stage": "layer.9.v", "coordinate": [112, 14, 2],
                   "original_absolute_bound": 0.005096435546875, "executed_in_this_diagnostic": False}
REQUIRED_SOURCES = {
    "Cargo.toml", "Cargo.lock", FROZEN_TOOLCHAIN, "src/lib.rs", "src/model.rs",
    "src/kernels.rs", "src/config.rs", "src/trace.rs", "src/numerical_diagnostics.rs",
    "src/model_diagnostics.rs", "tools/build_windows.ps1", "scripts/export_reference.py",
    "scripts/reference_preflight.py", "scripts/fetch_reference.py", "requirements/reference-lock.txt",
    "reference/manifest.json",
    *{"experiments/fp32_crossover_block0/" + name for name in (
        "contract.py", "adapter.py", "rust_arm.rs", "capture.py", "compare.py", "export_gpu.py",
        "test_source.py", "source_guard.py", "README.md", "SOURCE_REVIEW_NOTES.md")},
    "research/gpu-reference/experiments/fp32_crossover_layer7/adapter.py", "research/gpu-reference/experiments/fp32_crossover_layer7/export_gpu.py",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def read_bound(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "Changed JSON: " + str(path))
    return json.loads(raw)


def resolve(relative):
    path = (ROOT / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(ROOT), "Expected project-relative path")
    return path


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_cpu_control(execution, cpu, plan_sha256):
    """Validate the source-bound receipt's claims; callers bind and recheck bytes."""
    require(execution.get("kind") == "fp32-crossover-block0-execution-v1"
            and execution.get("status") == "cpu_control_exact"
            and execution.get("plan_sha256") == plan_sha256
            and execution.get("source_and_input_closure") is True, "CPU execution not accepted")
    require(cpu.get("kind") == "fp32-crossover-block0-rust-arm-v1"
            and cpu.get("status") == "control_exact" and cpu.get("plan_sha256") == plan_sha256
            and cpu.get("branch") == "cpu_state" and cpu.get("arithmetic_changed") is False
            and cpu.get("native_block0_calls") == 1 and cpu.get("input_bit_exact") is True,
            "CPU block0 control receipt differs")
    require(cpu.get("runtime") == {"threads": 4, "backend": "avx2", "precision": "fp32", "cache_capacity": 161},
            "CPU runtime differs")
    require([x["stage"] for x in cpu["controls"]] == CONTROL_STAGES
            and all(x["passed"] is True and x["bit_mismatches"] == 0
                    and x["actual_raw_sha256"] == x["expected_raw_sha256"] for x in cpu["controls"]),
            "CPU original control failed")
    require(set(cpu["tensors"]) == {"cpu_state." + name for name in STAGES}, "CPU stage inventory differs")
    for name, shape in STAGES.items():
        metadata = cpu["tensors"]["cpu_state." + name]
        require(metadata["shape"] == shape and metadata["dtype"] == "F32", "CPU stage metadata differs")
    require(cpu["input_raw_sha256"] == cpu["tensors"]["cpu_state.input"]["raw_sha256"], "CPU entry identity differs")


def validate_plan(plan, *, check_files=True):
    require(plan.get("kind") == KIND and plan.get("runtime") == RUNTIME, "Wrong crossover scope/runtime")
    require(plan.get("stages") == STAGES and plan.get("control_stages") == CONTROL_STAGES, "Stage/control scope changed")
    require(plan.get("model_directory") == "artifacts/model", "Wrong checkpoint directory")
    require(plan.get("original_state_key") == "embedding"
            and plan.get("original_state_shape") == [144, 768]
            and plan.get("segment_endpoint") == "layer.0.hidden"
            and plan.get("original_failure_context") == FAILURE_CONTEXT
            and plan.get("evaluation_budget") == BUDGET
            and plan.get("branches") == {"gpu": ["gpu_state", "cpu_state"], "rust": ["cpu_state"]},
            "Wrong entry-state or branch scope")
    require(set(plan.get("inputs", {})) == set(PINS) | set(METADATA), "Input role inventory differs")
    for role, (path, digest) in PINS.items():
        require(plan["inputs"][role] == {"path": path, "sha256": digest}, "Wrong fixed input: " + role)
    for role, path in METADATA.items():
        require(plan["inputs"][role]["path"] == path, "Wrong metadata path: " + role)
    require(REQUIRED_SOURCES <= set(plan["source_sha256"]), "Incomplete source closure")
    require(plan["source_sha256"].get("src/model.rs") == PRODUCTION_MODEL_SHA, "Production model changed")
    require(plan["source_sha256"].get("src/kernels.rs") == PRODUCTION_KERNELS_SHA, "Production kernels changed")
    if not check_files:
        return
    for entry in plan["inputs"].values():
        require(sha(resolve(entry["path"])) == entry["sha256"], "Changed input: " + entry["path"])
    for name, digest in plan["source_sha256"].items():
        require(sha(resolve(name)) == digest, "Changed source: " + name)
    prior = read_bound(resolve(PINS["reused_source_plan"][0]), PINS["reused_source_plan"][1])
    require(plan["source_sha256"][FROZEN_TOOLCHAIN] == prior["source_sha256"]["rust-toolchain.toml"],
            "Frozen original toolchain differs")
    for name, digest in plan["source_sha256"].items():
        if name in ("Cargo.toml", "Cargo.lock") or name.startswith(("src/", "examples/support/", "tests/fixtures/")):
            require(prior["source_sha256"].get(name) == digest,
                    "Compilation input differs from preserved original control: " + name)
    for name in ("adapter.py", "export_gpu.py"):
        path = "experiments/fp32_crossover_layer7/" + name
        require(plan["source_sha256"][path] == prior["source_sha256"][path], "Reused original source changed")
    metadata = {role: read_bound(resolve(entry["path"]), entry["sha256"])
                for role, entry in plan["inputs"].items() if role in METADATA}
    gpu, cpu, baseline, manifest = (metadata[k] for k in ("gpu_metadata", "cpu_metadata", "baseline_comparison", "model_manifest"))
    require(gpu["trace_sha256"] == PINS["gpu_trace"][1] and gpu["precision"] == "fp32"
            and gpu["tf32"] is False and gpu["flex_float32_precision"] == "ieee"
            and gpu["prefix_length"] == 144 and gpu["cache_capacity"] == 256
            and gpu["compiled_blocks"] is False, "Historical GPU fixture contract differs")
    require(cpu["precision"] == "fp32" and cpu["teacher_forced"] is True
            and cpu["input_tokens"] == 144 and cpu["output_tokens"] == 17
            and cpu["backend"] == "rust-gemm/avx2", "Historical CPU fixture contract differs")
    require(baseline["reference_sha256"] == PINS["gpu_trace"][1]
            and baseline["candidate_sha256"] == PINS["cpu_trace"][1]
            and baseline["compared_tensors"] == 1904 and len(baseline["failures"]) == 10,
            "Wrong frozen original comparison")
    require(manifest["model_revision"] == gpu["model_revision"] == MODEL_REVISION
            and manifest["weights_sha256"] == gpu["weights_sha256"] == WEIGHTS_SHA,
            "Wrong pinned model")
    for name, item in manifest["files"].items():
        path = resolve("artifacts/model/" + name)
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Changed model asset: " + name)
