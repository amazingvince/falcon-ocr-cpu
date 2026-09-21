#!/usr/bin/env python3
"""Measure BF16 operator semantics; this does not implement a BF16 OCR runner."""
import json
import pathlib

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from export_reference import import_model
from fetch_reference import sha256
from reference_preflight import preflight


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    module = import_model(root / "artifacts/model")
    source = root / "artifacts/reference/linear-operators.safetensors"
    output, fixtures, norms, gates = {}, [], [], []
    with torch.inference_mode(), safe_open(str(source), framework="pt") as linear:
        for layer in [0, 12]:
            for kind, checkpoint in [("qkv", "attention.wqkv"), ("wo", "attention.wo"),
                                     ("w13", "feed_forward.w13"), ("w2", "feed_forward.w2")]:
                base = f"prefill.layer.{layer}.{kind}"
                weight_key = f"weights.layers.{layer}.{checkpoint}.weight"
                weight = linear.get_tensor(weight_key).to("cuda:0", dtype=torch.bfloat16)
                output[weight_key] = weight.cpu().contiguous()
                raw = linear.get_tensor(base + ".input")
                for count in [1, 2, 4, 8, 32]:
                    name = base + f".rows{count}"
                    inputs = raw[:count].to("cuda:0", dtype=torch.bfloat16)
                    expected = torch.mm(inputs, weight.T, out_dtype=torch.float32)
                    output[name + ".input"] = inputs.cpu().contiguous()
                    output[name + ".expected"] = expected.cpu().contiguous()
                    fixtures.append({"name": name, "weight_key": weight_key,
                                     "input_dtype": "bfloat16", "weight_dtype": "bfloat16", "output_dtype": "float32"})
                if kind == "w13":
                    packed = linear.get_tensor(base + ".expected")[:32].to("cuda:0", dtype=torch.bfloat16)
                    actual = module.squared_relu_gate(packed, packed.shape[-1] // 2)
                    gate, value = packed[..., 0::2], packed[..., 1::2]
                    rounded_once = (F.relu(gate.float()).square() * value.float()).bfloat16()
                    rounded_stages = F.relu(gate).square() * value
                    name = f"gate.layer.{layer}"
                    output[name + ".input"] = packed.cpu().contiguous()
                    output[name + ".expected"] = actual.cpu().contiguous()
                    output[name + ".fp32_then_bf16"] = rounded_once.cpu().contiguous()
                    output[name + ".staged_bf16"] = rounded_stages.cpu().contiguous()
                    gates.append({"name": name, "upstream_output_dtype": str(actual.dtype),
                                  "equals_single_final_round": torch.equal(actual, rounded_once),
                                  "equals_staged_bf16_rounds": torch.equal(actual, rounded_stages),
                                  "max_error_vs_single_round": (actual.float() - rounded_once.float()).abs().max().item(),
                                  "max_error_vs_staged_rounds": (actual.float() - rounded_stages.float()).abs().max().item()})
    rms_source = root / "artifacts/reference/rms-operators.safetensors"
    with torch.inference_mode(), safe_open(str(rms_source), framework="pt") as rms:
        cases = {"rms.width768.actual": rms.get_tensor("layer.0.attention_norm.input")[:32],
                 "rms.width64.actual": rms.get_tensor("layer.12.q_norm.input").reshape(-1, 64)[:32],
                 "rms.width768.small": torch.linspace(-1e-4, 1e-4, 768).reshape(1, -1),
                 "rms.width64.small": torch.linspace(-1e-4, 1e-4, 64).reshape(1, -1)}
        for name, raw in cases.items():
            value = raw.to("cuda:0", dtype=torch.bfloat16)
            implicit = F.rms_norm(value, (value.shape[-1],))
            eps_f32 = F.rms_norm(value, (value.shape[-1],), eps=torch.finfo(torch.float32).eps)
            eps_bf16 = F.rms_norm(value, (value.shape[-1],), eps=torch.finfo(torch.bfloat16).eps)
            f32_then_bf16 = F.rms_norm(value.float(), (value.shape[-1],), eps=torch.finfo(torch.float32).eps).bfloat16()
            for suffix, tensor in [("input", value), ("implicit", implicit), ("eps_f32", eps_f32),
                                   ("eps_bf16", eps_bf16), ("fp32_then_bf16", f32_then_bf16)]:
                output[name + "." + suffix] = tensor.cpu().contiguous()
            norms.append({"name": name, "implicit_equals_eps_f32": torch.equal(implicit, eps_f32),
                          "implicit_equals_eps_bf16": torch.equal(implicit, eps_bf16),
                          "implicit_equals_fp32_then_bf16": torch.equal(implicit, f32_then_bf16),
                          "output_dtype": str(implicit.dtype)})
    target = root / "artifacts/reference/bf16-operators.safetensors"
    save_file(output, str(target))
    metadata = {"schema_version": 1, "environment": environment, "source_linear_sha256": sha256(source),
                "source_rms_sha256": sha256(rms_source), "output_sha256": sha256(target),
                "linear_operation": "torch.mm(BF16 input, BF16 weight.T, out_dtype=torch.float32)",
                "allow_bf16_reduced_precision_reduction": False, "tf32": False,
                "eps_f32": torch.finfo(torch.float32).eps, "eps_bf16": torch.finfo(torch.bfloat16).eps,
                "fixtures": fixtures, "rms_probes": norms, "gate_probes": gates,
                "qualification": "Isolated BF16 operands and GPU operators only; no BF16 full-model parity claim."}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(target), "bytes": target.stat().st_size, "linear_cases": len(fixtures), "rms_probes": norms, "gate_probes": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
