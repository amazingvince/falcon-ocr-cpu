#!/usr/bin/env python3
"""Isolate attention from inherited layer errors using real captured Q/K/V."""
import argparse
import json
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.nn.attention.flex_attention import AuxRequest

from export_reference import import_model
from fetch_reference import sha256
from reference_preflight import preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--source", type=pathlib.Path, default=pathlib.Path("artifacts/reference/smoke-fp32/trace.safetensors"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/attention-operators.safetensors"))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source = args.source
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    trace = load_file(str(source))
    weights = load_file(str(root / "artifacts/model/model.safetensors"))
    module = import_model(root / "artifacts/model")
    attention = __import__("pinned_falcon_ocr.attention", fromlist=["create_batch_attention_mask"])
    config = module.FalconOCRConfig.from_json_file(str(root / "artifacts/model/config.json"))
    tokens = trace["tokens"].unsqueeze(0).to("cuda:0")
    image_start = int(torch.nonzero(tokens[0] == config.image_cls_token_id)[0])
    image_end = int(torch.nonzero(tokens[0] == config.img_end_id)[0])
    pad_id = 2  # resolved below from the pinned tokenizer JSON
    from tokenizers import Tokenizer
    pad_id = Tokenizer.from_file(str(root / "artifacts/model/tokenizer.json")).token_to_id("<|pad|>")
    padded = torch.full((1, 256), pad_id, device="cuda:0", dtype=torch.long)
    padded[:, :tokens.shape[1]] = tokens
    mask = attention.create_batch_attention_mask(padded, pad_token_id=pad_id, eos_token_id=config.eos_id,
                                                 soi_token_id=config.image_cls_token_id, eoi_token_id=config.img_end_id, max_len=256)
    result = {}
    norm_probes = []
    cases = [(None, 0), (None, 17), (None, 19), (None, 21), (2, 21), (6, 19)]
    with torch.inference_mode():
        for step, layer in cases:
            name = f"prefill.layer.{layer}" if step is None else f"decode.{step}.layer.{layer}"
            prefix = "" if step is None else f"decode.{step}."
            q = trace[prefix + f"layer.{layer}.q"].to("cuda:0")
            k = trace[f"layer.{layer}.k"].to("cuda:0")
            v = trace[f"layer.{layer}.v"].to("cuda:0")
            offset = 0
            if step is not None:
                k = torch.cat([k] + [trace[f"decode.{i}.layer.{layer}.k"].to("cuda:0") for i in range(step + 1)])
                v = torch.cat([v] + [trace[f"decode.{i}.layer.{layer}.v"].to("cuda:0") for i in range(step + 1)])
                offset = tokens.shape[1] + step
                this_mask = mask[:, :, offset // mask.BLOCK_SIZE[0]]
                this_mask.seq_lengths = (1, k.shape[0])
                this_mask.mask_mod = attention.offset_mask_mod(mask.mask_mod, offset)
                flex = module.compiled_flex_attn_decode
            else:
                this_mask = mask
                this_mask.seq_lengths = (q.shape[0], k.shape[0])
                flex = module.compiled_flex_attn_prefill
            assert q.dtype == k.dtype == v.dtype == dtype
            sinks = weights[f"layers.{layer}.attention.sinks"].to("cuda:0", dtype=dtype)
            qgpu, kgpu, vgpu = [x.transpose(0, 1).unsqueeze(0) for x in [q, k, v]]
            out, aux = flex(qgpu, kgpu, vgpu, block_mask=this_mask, return_aux=AuxRequest(lse=True), kernel_options={"FLOAT32_PRECISION": "'ieee'"})
            sink_scale = torch.sigmoid(aux.lse - sinks[None, :, None]).unsqueeze(-1)
            expected = (out * sink_scale).to(out.dtype)
            qi = torch.arange(q.shape[0], device="cuda:0")[:, None]
            ki = torch.arange(k.shape[0], device="cuda:0")[None, :]
            zero = torch.tensor(0, device="cuda:0", dtype=torch.long)
            allowed = this_mask.mask_mod(zero, zero, qi, ki)
            scores = (qgpu.float() @ kgpu.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
            scores = scores.masked_fill(~allowed[None, None], -torch.inf)
            lse = scores.logsumexp(-1)
            probabilities = torch.softmax(scores, -1)
            dense_sink_free = probabilities.to(dtype) @ vgpu
            dense_expected = (dense_sink_free * torch.sigmoid(lse - sinks[None, :, None]).unsqueeze(-1)).to(dtype)
            def cpu(x):
                return x.detach().cpu().contiguous()
            result.update({name + ".q": cpu(q), name + ".k": cpu(k), name + ".v": cpu(v), name + ".sinks": cpu(sinks),
                           name + ".expected": cpu(expected[0].transpose(0, 1)),
                           name + ".dense_expected": cpu(dense_expected[0].transpose(0, 1)),
                           name + ".sink_free_expected": cpu(out[0].transpose(0, 1)), name + ".lse": cpu(aux.lse[0]),
                           name + ".dense_lse": cpu(lse[0]), name + ".allowed": cpu(allowed.to(torch.uint8)),
                           name + ".params": torch.tensor([offset, image_start, image_end], dtype=torch.long)})
            if args.precision == "bf16":
                fp32_pv = (probabilities @ vgpu.float()).to(dtype)
                result.update({name + ".dense_probabilities_f32": cpu(probabilities[0]),
                               name + ".dense_probabilities_bf16": cpu(probabilities[0].to(dtype)),
                               name + ".dense_fp32_pv_sink_free": cpu(fp32_pv[0].transpose(0, 1)),
                               name + ".dense_bf16_pv_sink_free": cpu(dense_sink_free[0].transpose(0, 1)),
                               name + ".sink_scale_f32": cpu(sink_scale[0]),
                               name + ".post_sink_before_output_cast_f32": cpu((out * sink_scale)[0].transpose(0, 1))})
        if args.precision == "bf16":
            norm_weight = trace["weights.final_norm"].to("cuda:0")
            result["final_norm.weight"] = norm_weight.cpu().contiguous()
            for prefix in ["", "decode.2.", "decode.6."]:
                value = trace[prefix + "final_norm.input"].to("cuda:0")
                expected_norm = F.rms_norm(value, (config.dim,), weight=norm_weight, eps=config.norm_eps)
                normalized_f32 = F.rms_norm(value.float(), (config.dim,), eps=config.norm_eps)
                single_round = (normalized_f32 * norm_weight.float()).to(dtype)
                staged_round = normalized_f32.to(dtype) * norm_weight
                name = (prefix or "prefill.") + "final_norm"
                result.update({name + ".input": value.cpu().contiguous(), name + ".expected": expected_norm.cpu().contiguous(),
                               name + ".fp32_then_bf16": single_round.cpu().contiguous(), name + ".staged_bf16": staged_round.cpu().contiguous()})
                norm_probes.append({"name": name, "eps": config.norm_eps, "weight_dtype": str(norm_weight.dtype),
                                    "matches_full_trace": torch.equal(expected_norm.cpu(), trace[prefix + "final_norm.output"]),
                                    "equals_single_final_round": torch.equal(expected_norm, single_round),
                                    "equals_staged_bf16_rounds": torch.equal(expected_norm, staged_round)})
    target = args.output
    save_file(result, str(target))
    meta = {"environment": environment, "source_trace_sha256": sha256(source), "output_sha256": sha256(target),
            "cases": [f"{'prefill' if s is None else 'decode.' + str(s)}.layer.{l}" for s, l in cases],
            "dtype": str(dtype), "tf32": False, "image_mask_interval": "[image_start,image_end) excludes end-of-image",
            "cast_contract": {"qkv_dtype": str(dtype), "sinks_dtype": str(dtype), "raw_output_dtype": str(out.dtype),
                              "lse_dtype": str(aux.lse.dtype), "sink_scale_dtype": str(sink_scale.dtype), "scaled_output_dtype": str(expected.dtype)},
            "note": "Identical Q/K/V for FlexAttention and independent dense attention; sink scaling applied separately."}
    if args.precision == "bf16":
        template = pathlib.Path(torch.__file__).parent / "_inductor/kernel/flex/templates/common.py.jinja"
        template_copy = target.with_suffix(".flex-common.py.jinja")
        template_copy.write_bytes(template.read_bytes())
        meta["final_norm_probes"] = norm_probes
        meta["flex_bf16_probability_contract"] = {
            "template_sha256": sha256(template), "template_copy": str(template_copy),
            "observed_source": "p=exp2(scores-running_max); l_i accumulates unrounded FP32 p; dot receives p.to(MATMUL_PRECISION) where MATMUL_PRECISION=Q.dtype; accumulator is FP32 and is normalized by l_i later.",
            "qualification": "Flex casts local unnormalized exponential weights to BF16 before P*V. This is not numerically identical to rounding globally normalized softmax probabilities; dense variants are independent calibration alternatives."}
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
