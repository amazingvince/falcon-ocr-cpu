#!/usr/bin/env python3
"""Strict GPU linear projections on real, independently fixed layer inputs."""
import json
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from export_reference import import_model
from fetch_reference import sha256
from reference_preflight import preflight


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = root / "artifacts/reference/smoke-fp32/trace.safetensors"
    trace = load_file(str(source))
    weights = load_file(str(root / "artifacts/model/model.safetensors"))
    module = import_model(root / "artifacts/model")
    result, fixtures = {}, []
    def linear(name, x, weight_key):
        weight = weights[weight_key].to("cuda:0")
        out = F.linear(x, weight)
        result[name + ".input"] = x.detach().cpu().contiguous().clone()
        result[name + ".expected"] = out.detach().cpu().contiguous().clone()
        shared = "weights." + weight_key
        result[shared] = weights[weight_key].contiguous()
        fixtures.append({"name": name, "weight_key": shared})
        return out
    with torch.inference_mode():
        for prefix, layers in [("", [0, 6, 9, 12, 17, 21]), ("decode.2.", [9, 12, 19, 21]), ("decode.6.", [9, 12, 19, 21])]:
            for layer in layers:
                label = prefix or "prefill."
                name = label + f"layer.{layer}"
                hkey = "embedding" if layer == 0 else f"layer.{layer - 1}.hidden"
                h = trace[prefix + hkey].to("cuda:0")
                normalized = F.rms_norm(h, (h.shape[-1],))
                linear(name + ".qkv", normalized, f"layers.{layer}.attention.wqkv.weight")
                attn = trace[prefix + f"layer.{layer}.attention"].to("cuda:0")
                post_attn = h + linear(name + ".wo", attn, f"layers.{layer}.attention.wo.weight")
                ffn_input = F.rms_norm(post_attn, (post_attn.shape[-1],))
                packed = linear(name + ".w13", ffn_input, f"layers.{layer}.feed_forward.w13.weight")
                gated = module.squared_relu_gate(packed, 2304)
                linear(name + ".w2", gated, f"layers.{layer}.feed_forward.w2.weight")
                if not prefix:
                    for rows in [1, 2, 4, 8]:
                        linear(name + f".qkv.rows{rows}", normalized[:rows].contiguous(), f"layers.{layer}.attention.wqkv.weight")
                        linear(name + f".w13.rows{rows}", ffn_input[:rows].contiguous(), f"layers.{layer}.feed_forward.w13.weight")
    target = root / "artifacts/reference/linear-operators.safetensors"
    save_file(result, str(target))
    metadata = {"environment": environment, "source_trace_sha256": sha256(source), "output_sha256": sha256(target),
                "dtype": "float32", "tf32": False, "fixtures": fixtures,
                "schema": "Each fixture uses <name>.input[S,K], <name>.expected[S,N], shared weight_key[N,K].",
                "note": "Inputs reconstructed with strict GPU RMSNorm/linear and the actual upstream Triton squared_relu_gate on captured previous-layer/attention outputs. Small row slices recompute GPU linear outputs at that row count."}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"path": str(target), "fixtures": len(fixtures), "bytes": target.stat().st_size, "sha256": metadata["output_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
