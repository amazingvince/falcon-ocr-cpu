#!/usr/bin/env python3
"""Freeze only GPU/F64/blockwise evidence; never inspect a CPU candidate."""
import datetime
import json
import pathlib
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from fetch_reference import sha256
from reference_preflight import preflight


def ulp(value):
    return torch.pow(2., torch.floor(torch.log2(value.float().abs().clamp_min(2.**-126))) - 7)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    target = root / "reference/bf16-local-contract-v1.json"
    if target.exists():
        raise ValueError("Refusing to overwrite frozen BF16 local contract")
    environment = preflight(root)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    op_path = root / "artifacts/reference/bf16-operators.safetensors"
    ops = load_file(str(op_path))
    metadata = json.loads(op_path.with_suffix(".json").read_text())
    attn_path = root / "artifacts/reference/attention-operators-bf16.safetensors"
    attn = load_file(str(attn_path))
    alt_path = root / "artifacts/reference/attention-blockwise-bf16.safetensors"
    alt = load_file(str(alt_path))
    attn_meta = json.loads(attn_path.with_suffix(".json").read_text())
    eps = torch.finfo(torch.float32).eps
    tensors, linear, norm, attention = {}, [], [], []
    with torch.inference_mode():
        for fixture in metadata["fixtures"]:
            name, weight_key = fixture["name"], fixture["weight_key"]
            x, w = ops[name + ".input"], ops[weight_key]
            gpu = ops[name + ".expected"]
            oracle = x.double() @ w.double().T
            err = (gpu.double() - oracle).abs()
            peak = float(gpu.abs().max())
            bound = max(4 * float(err.max()), 32 * eps * max(1., peak))
            native = F.linear(x.cuda(), w.cuda()).cpu()
            delta = (native.float() - oracle.bfloat16().float()).abs()
            native_bound = torch.maximum(4 * delta, ulp(native))
            tensors[name + ".native_expected"] = native.contiguous()
            tensors[name + ".native_bound"] = native_bound.contiguous()
            linear.append({"name": name, "weight_key": weight_key, "fp32_accumulation_max_abs_bound": bound,
                           "gpu_fp32_vs_f64_max_abs": float(err.max()),
                           "native_bf16_vs_rounded_f64_different": int((native != oracle.bfloat16()).sum()),
                           "native_bf16_equals_round_gpu_fp32": torch.equal(native, gpu.bfloat16()),
                           "native_expected_key": name + ".native_expected", "native_bound_key": name + ".native_bound"})
        for fixture in metadata["rms_probes"]:
            name = fixture["name"]
            x = ops[name + ".input"].double()
            oracle = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).bfloat16()
            expected = ops[name + ".implicit"]
            bound = torch.maximum(4 * (expected.float()-oracle.float()).abs(), ulp(expected))
            tensors[name + ".bound"] = bound.contiguous()
            norm.append({"name": name, "expected_source": "bf16-operators", "expected_key": name + ".implicit", "bound_key": name + ".bound"})
        for fixture in attn_meta["final_norm_probes"]:
            name = fixture["name"]
            x, w = attn[name + ".input"].double(), attn["final_norm.weight"].double()
            oracle = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + fixture["eps"]) * w).bfloat16()
            expected = attn[name + ".expected"]
            tensors[name + ".bound"] = torch.maximum(4 * (expected.float()-oracle.float()).abs(), ulp(expected)).contiguous()
            norm.append({"name": name, "expected_source": "attention-operators-bf16", "expected_key": name + ".expected", "bound_key": name + ".bound"})
        for name in attn_meta["cases"]:
            record = {"name": name}
            for kind, suffix in [("raw", "sink_free_expected"), ("scaled", "expected")]:
                expected, oracle = attn[name + "." + suffix], alt[name + "." + kind]
                err = (expected.float()-oracle.float()).abs()
                tensors[name + "." + kind + ".bound"] = torch.maximum(4 * err, ulp(expected)).contiguous()
                record[kind] = {"expected_key": name + "." + suffix, "bound_key": name + "." + kind + ".bound",
                                "rms_bound": max(4 * float(err.square().mean().sqrt()), 8 * eps * max(1., float(expected.float().abs().max())))}
            expected, oracle = attn[name + ".lse"], alt[name + ".lse"]
            record["lse_max_abs_bound"] = max(4*float((expected-oracle).abs().max()), 32*eps*max(1.,float(expected.abs().max())))
            attention.append(record)
    graph_path = root / "reference/gpu-blockwise-smoke-bf16.json"
    graph = json.loads(graph_path.read_text())
    logits = {}
    for name, decision in graph["logit_decisions"].items():
        measure = graph["measurements"][name]
        logits[name] = {"max_abs_bound": max(4*measure["max_abs"],32*eps*max(1.,measure["reference_max_abs"])),
                        "rms_bound": max(4*measure["rms_abs"],8*eps*max(1.,measure["reference_max_abs"])),
                        "required_argmax": decision["reference_argmax"], "gpu_winner_margin": decision["winner_margin"]}
    tensor_path = root / "artifacts/reference/bf16-local-contract-v1.safetensors"
    save_file(tensors, str(tensor_path))
    contract = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "qualification_kind": "Local equal-input operators plus same-prefix output checks; NOT full hidden-trajectory parity",
      "calibration_inputs": "Only pinned GPU operators, CPU FP64 mathematical oracle on exact BF16 operands, and independent GPU blockwise attention. No Rust candidate measurements were read.",
      "environment": environment,
      "sources": {str(p.relative_to(root)): sha256(p) for p in [op_path, attn_path, alt_path, graph_path, tensor_path, root / "reference/bf16-flex-kernel-metadata.json"]},
      "bound_rule": "BF16 element bounds=max(4*GPU-vs-independent-oracle absolute error, one BF16 ULP at reference magnitude); attention also requires a tight calibrated RMS bound. FP32 operator/logit bounds=max(4*observed error,32*FP32eps*max(1,peak)); RMS floor=8*FP32eps*max(1,peak).",
      "cast_contract": {
        "model_weights": "All BF16, including image projector, embedding, linears, sinks, learned final norm and golden spatial frequencies",
        "image_input": "Upstream RGB/resize/normalization FP32; cast valid normalized patches to BF16 before projection; projector output BF16 replaces embedding rows",
        "linears": "BF16 inputs/weights, FP32 accumulation, BF16 native output; FP32 accumulation output tested separately",
        "rms": "Hidden-width768 and Q/K-width64 normalize in FP32 with implicit epsilon=FP32epsilon; single BF16 final output round",
        "final_norm": "epsilon1e-5; multiply BF16-promoted learned weights in FP32 before single final BF16 output round",
        "rope": "Temporal complex64 table is regenerated after model conversion. Spatial golden frequencies are BF16 then promoted FP32; positions FP32; theta FP32, polar complex64. Promote Q/K BF16 to FP32 for rotation and round each rotated half back to BF16; nonvisual spatial half unchanged.",
        "attention": "Observed prefill128x64 tiles, partial sparse blocks then full; decode16x64 tiles with16KV splits and reverse full-block assignment. Local unnormalized exp2 weights round BF16 before FP32 P*V; denominator staysFP32; raw outputBF16, LSEFP32; sigmoid(LSE-promotedBF16sink)FP32; final scaled outputBF16",
        "residual": "Each attention and FFN residual addition rounds BF16 before the next operation",
        "gate": "Packed gate/up interleaved0::2/1::2; ReLU square rounds BF16 before BF16 multiply, whose output rounds BF16",
        "kv_and_logits": "Post-RoPE K and raw V stored BF16; final logits BF16"},
      "linear_cases": linear, "rms_cases": norm, "attention_cases": attention,
      "exact_cases": [{"name": x["name"], "source": "bf16-operators", "expected_key": x["name"]+".expected"} for x in metadata["gate_probes"]],
      "same_prefix_logits": logits,
      "same_prefix_rules": ["All tensors finite and shapes/inputs identical", "Every named logit tensor must satisfy max-absolute AND RMS bounds", "All17 fixture argmax IDs must match exactly; near-tie labels do not waive this fixture gate"],
      "corpus_gates": {
        "primary": "Corrected, frozen evaluation corpus; separate calibration split. Same pins, canonical pixels, prompt, image limits and token budgets on CPU and BF16 GPU.",
        "strict_output_parity": "Every evaluation page must have exactly matching generated token IDs, decoded text and stopping reason against BF16 GPU. Report missing pages and any divergent page as failed strict parity, even if quality is similar.",
        "ground_truth": "Compute diagnostic normalized CER and per-stratum aggregates for CPU BF16 and GPU BF16; require equality for a strict-parity claim. Structured/layout scores and BF16-versus-FP32 quality deltas must be reported separately; this contract does not authorize a general OCR-quality claim from diagnostic CER.",
        "long_context": "Also require the explicit8192-output boundary fixture to match BF16 GPU tokens/text/stop with context<=16384; a capped run is not natural-EOS evidence.",
        "failure_handling": "Do not widen this policy after observing CPU failures. Diagnose local operators or label the path experimental and qualification incomplete."},
      "limitations": ["Existing dense BF16 trajectory policy is rejected and preserved unchanged", "Blockwise oracle also showed signal-scale accumulated hidden drift; no full-hidden-trajectory acceptance threshold is defined", "Local bounds apply only to named fixtures and must be extended before claiming additional shapes/context coverage"]}
    target.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"output": str(target), "sha256": sha256(target), "linear_cases": len(linear), "attention_cases": len(attention), "norm_cases": len(norm)}, indent=2))


if __name__ == "__main__":
    main()
