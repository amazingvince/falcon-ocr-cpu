#!/usr/bin/env python3
"""Fixed eleven-branch GPU diagnostic, only after reviewed-plan release."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import traceback

from contract import (ROOT, KIND, UUID, RUNTIME, STAGES, BRANCHES, CONTROL_BRANCHES, require, sha,
                      plan_window, stable, write_new)
from evidence import load_evidence
from source_guard import verify_sources
from native_segment import run


def check_environment():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == UUID, "Use only the isolated RTX4090 UUID")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA order differs")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "cuBLAS workspace must be set before CUDA")
    require(os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "8",
            "Pinned reference host thread environment required")


def execute(args):
    require(args.execute_reviewed_plan, "No execution before root's reviewed-plan release")
    check_environment()  # Before any import that can initialize CUDA.
    plan_path = args.plan.resolve()
    require(plan_path.is_relative_to(ROOT), "Project plan required")
    plan, bound = plan_window(plan_path, args.plan_sha256)
    guard = verify_sources()
    require(plan["source_guard"] == guard, "Native fragment guard differs from preparation")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh output directory required")
    output.mkdir(parents=True)
    report = {"kind": KIND + "-gpu", "status": "started", "plan_sha256": args.plan_sha256,
              "runtime": RUNTIME, "native_block_calls": 0, "native_layer9_qkv_calls": 0,
              "controls_passed": [], "completed_branches": [], "controls": [],
              "source_and_input_closure": False, "native_source_guard": guard,
              "limits": ["Fixed 11 branches only; no Rust, generation, full-model run, retry or arithmetic variant.",
                         "Added observations/allocation history differ; all 47/17/30 control stages must match before unseen endpoints.",
                         "Historical startup attestation gaps remain. No numerical tolerance or qualification change."]}
    tensors = {}
    try:
        evidence = load_evidence(bound)
        stable(bound)
        report["historical_evidence"] = evidence["proof"]
        require(evidence["proof"] == plan["historical_evidence"]
                and evidence["proof"]["state_raw_sha256"] == plan["state_raw_sha256"],
                "Prepared full-state evidence changed")
        sys.path.insert(0, str(ROOT / "scripts"))
        from reference_preflight import preflight
        report["environment"] = preflight(ROOT)
        old_environment = evidence["reports"]["old_gpu_report"]["environment"]
        require({k: v for k, v in report["environment"].items() if k != "free_memory_mib"}
                == {k: v for k, v in old_environment.items() if k != "free_memory_mib"},
                "Pinned GPU/driver/package/interpreter environment differs from the prior capture")
        import torch
        from safetensors.torch import load_file
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
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        old_math = evidence["reports"]["old_gpu_report"]["effective_torch_math"]
        require(report["effective_torch_math"] == old_math, "Effective native precision/runtime differs from old control")
        report["effective_threads"] = {
            "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
            "environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "MKL_NUM_THREADS",
                "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG")}}
        require(report["effective_threads"] == evidence["reports"]["old_gpu_report"]["effective_threads"],
                "Effective host threading environment differs from saved control")
        torch.manual_seed(42)
        def tensor(value):
            return torch.from_numpy(value.copy()).contiguous()
        positions = {k: tensor(v) for k, v in evidence["positions"].items()}
        expected = {b: {s: tensor(v) for s, v in values.items()}
                    for b, values in evidence["controls"].items()}
        entries = {s: tensor(v) for s, v in evidence["entries"].items()}
        stable(bound)
        run(torch, import_model, load_file, Tokenizer, ROOT / "artifacts/model",
            positions["tokens"], positions["pos_t"], positions["pos_hw"], entries,
            expected, report, tensors, lambda: stable(bound))
        require(report["controls_passed"] == CONTROL_BRANCHES and report["completed_branches"] == BRANCHES
                and report["native_block_calls"] == 54 and report["native_layer9_qkv_calls"] == 11,
                "Bounded native calls did not complete")
        require([c["branch"] for c in report["controls"]] == CONTROL_BRANCHES, "Wrong control inventory")
        require(set(tensors) == {b + "." + s for b in BRANCHES for s in STAGES[b]}, "Incomplete final inventory")
        report["status"] = "three_controls_exact_eight_endpoints_captured"
    except BaseException as error:
        report.update(status="rejected", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if tensors:
                from safetensors.torch import save_file
                path = output / "tensors.safetensors"
                save_file(tensors, str(path))
                report["tensor_file_sha256"] = sha(path)
                report["tensors"] = {name: {"shape": list(value.shape), "dtype": "F32",
                    "raw_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()} for name, value in tensors.items()}
            stable(bound)
            report["source_and_input_closure"] = True
        except BaseException as error:
            report.update(status="rejected", source_and_input_closure=False, closure_or_artifact_error=repr(error))
        report["artifact_sha256"] = {p.relative_to(ROOT).as_posix(): h for p, h in bound.items()}
        write_new(output / "report.json", report)
    require(report["status"] == "three_controls_exact_eight_endpoints_captured", "GPU capture rejected")
    print(report["status"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute-reviewed-plan", action="store_true")
    execute(p.parse_args())


if __name__ == "__main__":
    main()
