#!/usr/bin/env python3
"""Strict CUDA RMSNorm fixtures on captured real-model activations."""
import json
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

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
    output = {}
    def emit(name, value):
        value = value.contiguous().to("cuda:0")
        expected = F.rms_norm(value, (value.shape[-1],))
        output[name + ".input"] = value.cpu().contiguous()
        output[name + ".expected"] = expected.cpu().contiguous()
        return expected
    with torch.inference_mode():
        for prefix in ["", "decode.0.", "decode.6.", "decode.15."]:
            for layer in range(22):
                input_key = "embedding" if layer == 0 else f"layer.{layer - 1}.hidden"
                x = trace[prefix + input_key].to("cuda:0")
                normed = emit(prefix + f"layer.{layer}.attention_norm", x)
                post_attn = x + F.linear(trace[prefix + f"layer.{layer}.attention"].to("cuda:0"),
                                         weights[f"layers.{layer}.attention.wo.weight"].to("cuda:0"))
                emit(prefix + f"layer.{layer}.ffn_norm", post_attn)
                if layer in [0, 6, 12, 21]:
                    qkv = F.linear(normed, weights[f"layers.{layer}.attention.wqkv.weight"].to("cuda:0"))
                    q, k, _ = qkv.split([1024, 512, 512], dim=-1)
                    emit(prefix + f"layer.{layer}.q_norm", q.reshape(-1, 16, 64))
                    emit(prefix + f"layer.{layer}.k_norm", k.reshape(-1, 8, 64))
    target = root / "artifacts/reference/rms-operators.safetensors"
    save_file(output, str(target))
    metadata = {"schema_version": 1, "environment": environment, "source_trace_sha256": sha256(source),
                "operation": "torch.nn.functional.rms_norm(input, (input.shape[-1],))",
                "dtype": "float32", "implicit_epsilon": torch.finfo(torch.float32).eps,
                "tf32": False, "output_sha256": sha256(target), "tensor_count": len(output)}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
