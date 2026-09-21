#!/usr/bin/env python3
"""Validate equal-input blockwise oracle before any model-policy calibration."""
import json
import pathlib

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from bf16_blockwise_attention import blockwise, _blocks
from export_reference import import_model
from fetch_reference import sha256
from reference_preflight import preflight


def metrics(expected, actual):
    a, b = expected.float(), actual.float()
    delta = (a - b).abs()
    peak = a.abs().max().item()
    result = {"elements": a.numel(), "equal": int((a == b).sum()), "max_abs": delta.max().item(),
              "rms_abs": delta.square().mean().sqrt().item(), "peak": peak,
              "error_to_peak": delta.max().item() / peak if peak else 0}
    if expected.dtype == torch.bfloat16:
        # Spacing at each reference magnitude, with the BF16 subnormal floor.
        spacing = torch.pow(2.0, torch.floor(torch.log2(a.abs().clamp_min(2.0 ** -126))) - 7)
        ulp = delta / spacing
        result.update({"max_reference_ulp": ulp.max().item(), "over_one_reference_ulp": int((ulp > 1).sum()),
                       "over_two_reference_ulp": int((ulp > 2).sum())})
    return result


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source = root / "artifacts/reference/attention-operators-bf16.safetensors"
    fixtures = load_file(str(source))
    metadata = json.loads(source.with_suffix(".json").read_text())
    trace = load_file(str(root / "artifacts/reference/smoke-bf16/trace.safetensors"))
    module = import_model(root / "artifacts/model")
    attention = __import__("pinned_falcon_ocr.attention", fromlist=["create_batch_attention_mask"])
    config = module.FalconOCRConfig.from_json_file(str(root / "artifacts/model/config.json"))
    pad = Tokenizer.from_file(str(root / "artifacts/model/tokenizer.json")).token_to_id("<|pad|>")
    padded = torch.full((1, 256), pad, dtype=torch.long, device="cuda:0")
    tokens = trace["tokens"].to("cuda:0")
    padded[:, :len(tokens)] = tokens
    original_mask = attention.create_batch_attention_mask(padded, pad_token_id=pad, eos_token_id=config.eos_id,
                    soi_token_id=config.image_cls_token_id, eoi_token_id=config.img_end_id, max_len=256)
    records, exported = [], {}
    with torch.inference_mode():
        for name in metadata["cases"]:
            q, k, v = [fixtures[name + "." + key].to("cuda:0").transpose(0, 1)[None] for key in ["q", "k", "v"]]
            offset = int(fixtures[name + ".params"][0])
            if offset:
                mask = original_mask[:, :, offset // original_mask.BLOCK_SIZE[0]]
                mask.seq_lengths = (1, k.shape[2])
                mask.mask_mod = attention.offset_mask_mod(original_mask.mask_mod, offset)
            else:
                mask = original_mask
                mask.seq_lengths = (q.shape[2], k.shape[2])
            raw, aux = blockwise(q, k, v, block_mask=mask)
            sinks = fixtures[name + ".sinks"].to("cuda:0")
            scaled = (raw * torch.sigmoid(aux.lse - sinks[None, :, None])[..., None]).to(raw.dtype)
            raw, lse, scaled = raw[0].transpose(0, 1).cpu(), aux.lse[0].cpu(), scaled[0].transpose(0, 1).cpu()
            schedules = [{"query_block": i, "partial": _blocks(mask, i, False, k.shape[2]), "full": _blocks(mask, i, True, k.shape[2])}
                         for i in range((q.shape[2] + 127) // 128)]
            record = {"name": name, "schedule": schedules,
                      "raw": metrics(fixtures[name + ".sink_free_expected"], raw),
                      "lse": metrics(fixtures[name + ".lse"], lse), "scaled": metrics(fixtures[name + ".expected"], scaled)}
            exported.update({name + ".raw": raw.contiguous(), name + ".lse": lse.contiguous(), name + ".scaled": scaled.contiguous()})
            records.append(record)
            print(json.dumps(record), flush=True)
    target = root / "artifacts/reference/attention-blockwise-bf16.safetensors"
    save_file(exported, str(target))
    report = {"environment": environment, "source_sha256": sha256(source), "oracle_sha256": sha256(pathlib.Path(__file__).with_name("bf16_blockwise_attention.py")),
              "result_sha256": sha256(target), "cases": records,
              "qualification": "Independent operator diagnostic; not a full-model BF16 acceptance policy"}
    (root / "reference/bf16-blockwise-attention-probe.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
