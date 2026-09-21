#!/usr/bin/env python3
"""Prospective native GPU arm. Execution requires a separately reviewed, pinned plan.

Preparing/importing this file performs no tensor loading or CUDA work. Do not run
it during a quiet benchmark or before root releases this diagnostic explicitly.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from contract import (CONTROL_STAGES, KIND, METADATA, PINS, RUNTIME,
                      STAGES as SHAPES, validate_plan)

ROOT = Path(__file__).resolve().parents[2]
UUID = "GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58"
GPU_TRACE_SHA = PINS["gpu_trace"][1]
CPU_TRACE_SHA = PINS["cpu_trace"][1]
ROLES = set(PINS) | set(METADATA)
REQUIRED_SOURCES = {
    "experiments/fp32_crossover_layer7/export_gpu.py", "experiments/fp32_crossover_layer7/contract.py",
    "scripts/export_reference.py",
    "scripts/reference_preflight.py", "scripts/fetch_reference.py",
    "requirements/reference-lock.txt", "reference/manifest.json",
    "artifacts/model/artifact-manifest.json", "artifacts/model/modeling_falcon_ocr.py",
    "artifacts/model/configuration_falcon_ocr.py", "artifacts/model/attention.py",
    "artifacts/model/rope.py", "artifacts/model/processing_falcon_ocr.py",
    "artifacts/model/config.json", "artifacts/model/tokenizer.json",
    "artifacts/model/tokenizer_config.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(value):
    p = Path(value)
    require(not p.is_absolute() and "\\" not in value and ".." not in p.parts,
            "Plan paths must be workspace-relative POSIX paths")
    p = (ROOT / p).resolve()
    require(p.is_relative_to(ROOT), "Plan path escaped workspace")
    return p


def checked_json(path, expected):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "JSON identity differs: " + str(path))
    return json.loads(data)


def execute(args):
    # Environment is checked before importing Torch or the existing import helper.
    require(args.execute_reviewed_plan, "Execution is held until explicit reviewed-plan release")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == UUID, "Isolate the reviewed RTX4090 UUID")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Pin cuBLAS workspace before CUDA")
    require(os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "8", "Pin reference thread environment")
    plan_path = args.plan.resolve()
    require(plan_path.is_relative_to(ROOT), "Reviewed plan must be inside the workspace")
    plan = checked_json(plan_path, args.plan_sha256)
    validate_plan(plan, check_files=False)
    require(plan["kind"] == KIND, "Wrong plan kind")
    require(plan["runtime"] == RUNTIME, "Changed bounded runtime contract")
    require(set(plan["inputs"]) == ROLES, "Changed input role inventory")
    require(plan["model_directory"] == "artifacts/model", "Wrong checkpoint directory")
    require(REQUIRED_SOURCES <= set(plan["source_sha256"]), "Missing startup source identity")
    roles = {name: local(item["path"]) for name, item in plan["inputs"].items()}
    require(plan["inputs"]["gpu_trace"]["sha256"] == GPU_TRACE_SHA and
            plan["inputs"]["cpu_trace"]["sha256"] == CPU_TRACE_SHA, "Changed original trace pins")
    bound = {plan_path: args.plan_sha256}
    for name, item in plan["inputs"].items():
        bound[roles[name]] = item["sha256"]
    for name, h in plan["source_sha256"].items():
        p = local(name)
        require(p not in bound or bound[p] == h, "Conflicting source/input hash")
        bound[p] = h

    def stable():
        require(all(sha(p) == h for p, h in bound.items()), "Frozen source/input changed")

    stable()
    gpu_meta = checked_json(roles["gpu_metadata"], plan["inputs"]["gpu_metadata"]["sha256"])
    cpu_meta = checked_json(roles["cpu_metadata"], plan["inputs"]["cpu_metadata"]["sha256"])
    baseline = checked_json(roles["baseline_comparison"], plan["inputs"]["baseline_comparison"]["sha256"])
    require(gpu_meta["trace_sha256"] == GPU_TRACE_SHA and gpu_meta["precision"] == "fp32" and
            gpu_meta["tf32"] is False and gpu_meta["compiled_blocks"] is False and
            gpu_meta["attention"] == "flex" and gpu_meta["prefix_length"] == 144 and
            gpu_meta["cache_capacity"] == 256, "Original GPU metadata differs")
    require(cpu_meta["precision"] == "fp32" and cpu_meta["teacher_forced"] is True,
            "Original CPU trace metadata differs")
    require(baseline["reference_sha256"] == GPU_TRACE_SHA and baseline["candidate_sha256"] == CPU_TRACE_SHA,
            "Baseline report does not bind the original payload pair")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh workspace output directory required")
    output.mkdir(parents=True)
    report = {"schema_version": 1, "status": "started", "plan_sha256": args.plan_sha256,
              "runtime": RUNTIME, "native_layer7_calls": 0,
              "gpu_control_passed": False, "crossover_executed": False,
              "source_and_input_closure": False,
              "controls": [], "tensors": {},
              "qualification": "New GPU diagnostic only. Rust(cpu_state) must independently reproduce its saved bits before any three-arm attribution.",
              "provenance_limits": [
                  "Original GPU metadata does not fully attest historical exporter/build startup source. Current source hashes do not repair that gap.",
                  "Original CPU sidecar has incomplete build/fixture provenance; the pinned baseline comparison binds its tensor payload. Current Rust replay needs its own bit-exact control.",
                  "This re-enters native layer7 from full saved layer6 states. Earlier-layer execution and allocator history are absent.",
                  "Pass-through hooks copy activations to CPU, adding allocations/synchronization. Native arithmetic/functions, tensor shape/strides, masks/positions/dtype are retained; exact controls are mandatory.",
                  "No numerical policy or tolerance changes; no performance or model-qualification claim."]}
    tensors = {}
    torch = None
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from reference_preflight import preflight
        report["environment"] = preflight(ROOT)
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from tokenizers import Tokenizer
        from export_reference import import_model
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        report["effective_torch_math"] = {
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "cuda_runtime": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        require(report["effective_torch_math"]["float32_matmul_precision"] == "highest" and
                all(report["effective_torch_math"][key] is False for key in (
                    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "allow_fp16_reduced_precision_reduction",
                    "allow_bf16_reduced_precision_reduction")), "Effective strict precision flags differ")
        report["effective_threads"] = {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
                "MKL_NUM_THREADS", "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS", "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG")},
        }
        torch.manual_seed(42)
        model_dir = local(plan["model_directory"])
        manifest = checked_json(roles["model_manifest"], plan["inputs"]["model_manifest"]["sha256"])
        for name, item in manifest["files"].items():
            p = model_dir / name
            require(p not in bound or bound[p] == item["sha256"], "Asset/source hash conflict")
            bound[p] = item["sha256"]
        stable()

        def saved(handle, name):
            candidates = [k for k in (name, "prefill." + name) if k in handle.keys()]
            require(len(candidates) == 1, "Missing/ambiguous saved prefill tensor: " + name)
            value = handle.get_tensor(candidates[0])
            require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "Invalid saved FP32 tensor")
            return value.contiguous()

        with safe_open(str(roles["gpu_trace"]), framework="pt", device="cpu") as g:
            gpu_state = saved(g, "layer.6.hidden")
            expected = {key: saved(g, key) for key in CONTROL_STAGES}
            tokens, pos_t, pos_hw = (g.get_tensor(k) for k in ("tokens", "pos_t", "pos_hw"))
        with safe_open(str(roles["cpu_trace"]), framework="pt", device="cpu") as c:
            cpu_state = saved(c, "layer.6.hidden")
        require(list(gpu_state.shape) == list(cpu_state.shape) == [144, 768], "Full original state shape required")
        require(list(tokens.shape) == list(pos_t.shape) == [144] and list(pos_hw.shape) == [144, 2], "Original position shapes")
        require(tokens.dtype == pos_t.dtype == torch.int64 and pos_hw.dtype == torch.float32, "Original position dtypes")
        module = import_model(model_dir)
        for name in ("compiled_flex_attn_prefill", "compiled_flex_attn_decode"):
            setattr(module, name, functools.partial(getattr(module, name), kernel_options={"FLOAT32_PRECISION": "'ieee'"}))
        config = module.FalconOCRConfig.from_json_file(str(model_dir / "config.json"))
        model = module.FalconOCRForCausalLM(config)
        model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True, assign=True)
        model = model.to(device="cuda:0", dtype=torch.float32).eval()
        model._ensure_device_buffers()
        report["effective_compiled_blocks"] = model._is_compiled
        require(model._is_compiled is False, "Native transformer blocks must remain uncompiled")
        pad = Tokenizer.from_file(str(model_dir / "tokenizer.json")).token_to_id("<|pad|>")
        require(type(pad) is int, "Missing original pad token")
        model._pad_token_id = pad
        padded = torch.full((1, 256), pad, dtype=torch.long, device="cuda:0")
        padded[:, :144] = tokens.to("cuda:0")
        mask = model.get_attention_mask(padded, max_len=256)
        mask.seq_lengths = (144, 144)
        temporal_positions = pos_t.to("cuda:0")[None]
        spatial_positions = pos_hw.to("cuda:0")[None]
        temporal = model.freqs_cis[temporal_positions]
        spatial = module.apply_golden_freqs_cis_to_visual_pos(model.freqs_cis_golden, spatial_positions)
        state = {"branch": None}

        def capture(key, value):
            name = state["branch"] + "." + key
            require(name not in tensors, "Duplicate stage capture")
            value = value.detach().squeeze(0).to("cpu").contiguous().clone()
            require(value.dtype == torch.float32 and list(value.shape) == SHAPES[key] and bool(torch.isfinite(value).all()), "Unexpected native stage shape/dtype/value")
            tensors[name] = value

        handles = []
        layer7 = model.layers["7"]
        for submodule, before, after in [
            (layer7.attention.wqkv, "layer.7.attention_norm", "layer.7.qkv"),
            (layer7.attention.wo, "layer.7.attention", "layer.7.wo"),
            (layer7.feed_forward.w13, "layer.7.ffn_norm", "layer.7.w13"),
            (layer7.feed_forward.w2, "layer.7.gate", "layer.7.w2")]:
            handles.append(submodule.register_forward_pre_hook(lambda m, args, key=before: capture(key, args[0])))
            handles.append(submodule.register_forward_hook(lambda m, args, out, key=after: capture(key, out)))
        handles.append(layer7.feed_forward.register_forward_pre_hook(lambda m, args: capture("layer.7.attention_residual", args[0])))
        original_qkv = layer7.attention._pre_attention_qkv
        original_rope = module.apply_3d_rotary_emb

        def qkv(*a, **kw):
            q, k, v = original_qkv(*a, **kw)
            capture("layer.7.v", v)
            return q, k, v

        def rope(*a, **kw):
            q, k = original_rope(*a, **kw)
            capture("layer.7.q", q)
            capture("layer.7.k", k)
            return q, k

        layer7.attention._pre_attention_qkv = qkv
        module.apply_3d_rotary_emb = rope
        try:
            with torch.inference_mode():
                for branch, input_state in (("gpu_state", gpu_state), ("cpu_state", cpu_state)):
                    stable()
                    state["branch"] = branch
                    x = input_state.to("cuda:0")[None]
                    capture("input", x)
                    cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)
                    cache.set_pos_t(temporal_positions[:, -1:])
                    report["native_layer7_calls"] += 1
                    hidden = layer7(x, freqs_cis=temporal, freqs_cis_2d=spatial,
                                    pos_hw=spatial_positions, attention_masks=mask, kv_cache=cache)
                    capture("layer.7.hidden", hidden)
                    require({k[len(branch) + 1:] for k in tensors if k.startswith(branch + ".")} == set(SHAPES), "Incomplete stage capture")
                    if branch == "gpu_state":
                        for key, want in expected.items():
                            got = tensors[branch + "." + key]
                            equal = got.shape == want.shape and torch.equal(got.view(torch.int32), want.view(torch.int32))
                            report["controls"].append({"stage": key, "bit_exact": equal,
                                "mismatched_elements": int((got.view(torch.int32) != want.view(torch.int32)).sum()) if got.shape == want.shape else None})
                        report["gpu_control_passed"] = all(c["bit_exact"] for c in report["controls"])
                        require(report["gpu_control_passed"], "Native GPU control mismatch; no crossover attribution or retry")
                    else:
                        report["crossover_executed"] = True
        finally:
            for handle in handles:
                handle.remove()
            layer7.attention._pre_attention_qkv = original_qkv
            module.apply_3d_rotary_emb = original_rope
        stable()
        report.update(status="gpu_arm_control_passed_crossover_captured", source_and_input_closure=True)
    except Exception as error:
        report.update(status="rejected", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # Preserve the bounded attempted capture even on control failure. No
        # stage payload is accepted for attribution unless the report gates pass.
        try:
            if tensors:
                from safetensors.torch import save_file
                tensor_path = output / "tensors.safetensors"
                save_file(tensors, str(tensor_path))
                report["tensor_file_sha256"] = sha(tensor_path)
                report["tensors"] = {name: {"shape": list(value.shape), "dtype": "F32",
                    "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()} for name, value in tensors.items()}
        except Exception as artifact_error:
            report.update(status="rejected", artifact_error=repr(artifact_error))
        try:
            stable()
            report["source_and_input_closure"] = True
        except Exception as closure_error:
            report.update(status="rejected", source_and_input_closure=False, closure_error=repr(closure_error))
        report["inputs_sha256"] = {str(p.relative_to(ROOT)): h for p, h in bound.items()}
        with (output / "report.json").open("x", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
    require(report["status"] == "gpu_arm_control_passed_crossover_captured", "GPU arm rejected")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args()
    report = execute(args)
    print(json.dumps({k: report[k] for k in ("status", "native_layer7_calls", "gpu_control_passed", "crossover_executed")}))


if __name__ == "__main__":
    main()
