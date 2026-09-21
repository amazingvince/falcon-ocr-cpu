#!/usr/bin/env python3
"""Isolate upstream temporal/spatial RoPE factors and rotations."""
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
    config = module.FalconOCRConfig.from_json_file(str(root / "artifacts/model/config.json"))
    temporal_all = module.precompute_freqs_cis(config.head_dim // 2, config.max_seq_len, config.rope_theta).to("cuda:0")
    pos_t = trace["pos_t"].unsqueeze(0).to("cuda:0")
    pos_hw = trace["pos_hw"].unsqueeze(0).to("cuda:0")
    golden = weights["freqs_cis_golden"].to("cuda:0")
    temporal = temporal_all[pos_t]
    spatial = module.apply_golden_freqs_cis_to_visual_pos(golden, pos_hw)
    valid_hw = pos_hw[~pos_hw.isnan().any(-1)]
    theta = torch.einsum("tp,hfp->thf", valid_hw.float(), golden.float())
    def cpu(value):
        return value.detach().cpu().contiguous()
    result = {"freqs_cis_all": cpu(torch.view_as_real(temporal_all)), "golden_freqs": cpu(golden),
              "pos_t": cpu(pos_t[0]), "pos_hw": cpu(pos_hw[0]), "temporal_cis": cpu(torch.view_as_real(temporal[0])),
              "spatial_cis": cpu(torch.view_as_real(spatial)), "spatial_theta": cpu(theta)}
    with torch.inference_mode():
        for layer in [0, 6, 12, 21]:
            key = "embedding" if layer == 0 else f"layer.{layer - 1}.hidden"
            x = trace[key].unsqueeze(0).to("cuda:0")
            qkv = F.linear(F.rms_norm(x, (x.shape[-1],)), weights[f"layers.{layer}.attention.wqkv.weight"].to("cuda:0"))
            q, k, _ = qkv.split([1024, 512, 512], -1)
            q = F.rms_norm(q.reshape(1, -1, 16, 64), (64,))
            k = module.repeat_kv(F.rms_norm(k.reshape(1, -1, 8, 64), (64,)), 2)
            qr, kr = module.apply_3d_rotary_emb(q, k, temporal, spatial, pos_hw)
            for name, value in [("q.input", q), ("k.input", k), ("q.expected", qr), ("k.expected", kr)]:
                result[f"layer.{layer}.{name}"] = cpu(value[0])
    target = root / "artifacts/reference/rope-operators.safetensors"
    save_file(result, str(target))
    meta = {"environment": environment, "source_trace_sha256": sha256(source), "output_sha256": sha256(target),
            "dtype": "float32", "tf32": False, "rope_theta": config.rope_theta,
            "note": "Temporal frequencies computed by upstream on CPU then copied to GPU, as in HF. Spatial einsum/polar and rotations execute on GPU."}
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
