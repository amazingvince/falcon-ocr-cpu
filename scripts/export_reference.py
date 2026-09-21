#!/usr/bin/env python3
"""Capture unmodified pinned HF model forwards and explicit same-prefix decode traces.

Model computation remains upstream; hooks only copy activations. FlexAttention IEEE
precision is explicit. The decode loop enforces the requested limit rather than the
upstream rounded-cache loop, and stores every chosen token for Rust teacher forcing.
"""
import argparse
import functools
import importlib
import importlib.util
import json
import pathlib
import sys
import time
import types

import einops as E
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from reference_preflight import preflight


def import_model(model_path):
    package = types.ModuleType("pinned_falcon_ocr")
    package.__path__ = [str(model_path)]
    sys.modules[package.__name__] = package
    return importlib.import_module("pinned_falcon_ocr.modeling_falcon_ocr")


def make_fixture(path):
    image = Image.new("RGB", (256, 128), "white")
    draw = ImageDraw.Draw(image)
    font_path = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 24)
    draw.text((12, 14), "Falcon OCR", font=font, fill=(0, 0, 0))
    draw.text((12, 50), "Hello, world!", font=font, fill=(0, 0, 0))
    draw.text((12, 86), "12345", font=font, fill=(0, 0, 0))
    image.save(path)
    return {"text": "Falcon OCR\nHello, world!\n12345", "font_path": str(font_path),
            "font_sha256": sha256(font_path), "font_size": 24}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=pathlib.Path, default=pathlib.Path("artifacts/model"))
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/smoke-fp32"))
    p.add_argument("--image", type=pathlib.Path)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-dimension", type=int, default=256)
    p.add_argument("--min-dimension", type=int, default=64)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--teacher-tokens", type=pathlib.Path, help="JSON list for identical-prefix comparisons")
    p.add_argument("--attention", choices=["flex", "dense", "blockwise"], default="flex")
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    model_module = import_model(args.model.resolve())
    # Explicitly lock FlexAttention's independent Triton matmul precision control.
    if args.attention == "flex":
        for name in ["compiled_flex_attn_prefill", "compiled_flex_attn_decode"]:
            setattr(model_module, name, functools.partial(getattr(model_module, name),
                                                          kernel_options={"FLOAT32_PRECISION": "'ieee'"}))
    elif args.attention == "blockwise":
        assert args.precision == "bf16", "Blockwise oracle describes the observed BF16 compiled kernels"
        from bf16_blockwise_attention import blockwise
        model_module.compiled_flex_attn_prefill = blockwise
        model_module.compiled_flex_attn_decode = blockwise
    else:
        def dense(q, k, v, *, block_mask, return_aux):
            del return_aux
            bi = torch.arange(q.shape[0], device=q.device)[:, None, None, None]
            hi = torch.arange(q.shape[1], device=q.device)[None, :, None, None]
            qi = torch.arange(q.shape[2], device=q.device)[None, None, :, None]
            ki = torch.arange(k.shape[2], device=q.device)[None, None, None, :]
            allowed = block_mask.mask_mod(bi, hi, qi, ki)
            scores = (q.float() @ k.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
            scores = scores.masked_fill(~allowed, -torch.inf)
            lse = scores.logsumexp(-1)
            output = torch.softmax(scores, -1).to(v.dtype) @ v
            return output, types.SimpleNamespace(lse=lse)
        model_module.compiled_flex_attn_prefill = dense
        model_module.compiled_flex_attn_decode = dense
    config = model_module.FalconOCRConfig.from_json_file(str(args.model / "config.json"))
    config._name_or_path = str(args.model.resolve())
    print("Loading pinned weights", flush=True)
    model = model_module.FalconOCRForCausalLM(config)
    model.load_state_dict(load_file(str(args.model / "model.safetensors")), strict=True, assign=True)
    golden_before_cast = model.freqs_cis_golden.detach().clone()
    model = model.to(device="cuda:0", dtype=dtype).eval()
    model._ensure_device_buffers()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=True)
    generated_fixture = None
    if args.image is None:
        args.image = args.output / "input.png"
        generated_fixture = make_fixture(args.image)
    image = Image.open(args.image).convert("RGB")
    image.save(args.output / "canonical-rgb.png")
    (args.output / "canonical-rgb.bin").write_bytes(image.tobytes())
    prompt = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
    batch = model_module.process_batch(tokenizer, config, [(image, prompt)], max_length=config.max_seq_len,
                                       min_dimension=args.min_dimension, max_dimension=args.max_dimension)
    tensors = {"tokens": batch["tokens"][0], "pos_t": batch["pos_t"][0], "pos_hw": batch["pos_hw"][0]}
    if args.precision == "bf16":
        tensors["buffers.golden_before_cast"] = golden_before_cast
        tensors["buffers.golden_after_cast"] = model.freqs_cis_golden.detach().cpu().contiguous()
        tensors["buffers.temporal_cis"] = torch.view_as_real(model.freqs_cis).detach().cpu().contiguous()
        tensors["weights.final_norm"] = model.norm.weight.detach().cpu().contiguous()
        for i, layer in enumerate(model.layers.values()):
            tensors[f"weights.layer.{i}.sinks"] = layer.attention.sinks.detach().cpu().contiguous()
    patches = E.rearrange(batch["pixel_values"], "n t (h ph) (w pw) c -> (n t h w) (ph pw c)", ph=16, pw=16)
    mask = E.reduce(batch["pixel_mask"], "n t (h ph) (w pw) -> (n t h w)", "any", ph=16, pw=16)
    tensors["patches"] = patches[mask.bool()]
    prefix_length = batch["tokens"].shape[1]
    if prefix_length + args.max_new_tokens > config.max_seq_len:
        raise ValueError("Input prefix plus requested output exceeds total model context")
    capacity = ((prefix_length + args.max_new_tokens + 127) // 128) * 128
    if capacity > config.max_seq_len:
        raise ValueError("Upstream cache rounding exceeds model context")
    model._pad_token_id = batch["pad_token_id"]
    batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}
    padded = torch.full((1, capacity), model._pad_token_id, device="cuda:0", dtype=torch.long)
    padded[:, :prefix_length] = batch["tokens"]
    attention_mask = model.get_attention_mask(padded, max_len=capacity)
    cache = model_module.KVCache(1, capacity, config.n_heads, config.head_dim, config.n_layers)
    state = {"prefix": "", "layer": 0}
    def capture(name, value):
        tensors[state["prefix"] + name] = value.detach().squeeze(0).to("cpu").contiguous().clone()
    def first_hook(_module, inputs):
        capture("embedding", inputs[0])
    model.layers["0"].register_forward_pre_hook(first_hook)
    if args.precision == "bf16":
        model.norm.register_forward_pre_hook(lambda m, inp: capture("final_norm.input", inp[0]))
        model.norm.register_forward_hook(lambda m, inp, out: capture("final_norm.output", out))
    for index, layer in enumerate(model.layers.values()):
        layer.register_forward_hook(lambda m, inp, out, i=index: capture(f"layer.{i}.hidden", out))
        layer.attention.wo.register_forward_pre_hook(lambda m, inp, i=index: capture(f"layer.{i}.attention", inp[0]))
        original = layer.attention._pre_attention_qkv
        def qkv(x, original=original, i=index):
            state["layer"] = i
            q, k, v = original(x)
            capture(f"layer.{i}.v", v)
            return q, k, v
        layer.attention._pre_attention_qkv = qkv
    original_rope = model_module.apply_3d_rotary_emb
    def rope(*a, **kw):
        q, k = original_rope(*a, **kw)
        capture(f"layer.{state['layer']}.q", q)
        capture(f"layer.{state['layer']}.k", k)
        return q, k
    model_module.apply_3d_rotary_emb = rope
    forced = json.loads(args.teacher_tokens.read_text()) if args.teacher_tokens else None
    chosen = []
    stop_ids = [config.eos_id, tokenizer.convert_tokens_to_ids("<|end_of_query|>")]
    timings = {}
    with torch.inference_mode():
        print(f"Prefill: {prefix_length} tokens; {args.attention} attention", flush=True)
        start = time.perf_counter()
        logits = model(tokens=batch["tokens"], attention_mask=attention_mask, kv_cache=cache,
                       rope_pos_t=batch["pos_t"], rope_pos_hw=batch["pos_hw"],
                       pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
        capture("logits", logits[:, -1])
        timings["prefill_seconds_with_capture_and_compile"] = time.perf_counter() - start
        for step in range(args.max_new_tokens):
            token = int(forced[step] if forced is not None else logits[0, -1].argmax())
            chosen.append(token)
            print(f"Step {step}: {token} {tokenizer.decode([token])!r}", flush=True)
            if forced is None and token in stop_ids:
                break
            if step + 1 == args.max_new_tokens or (forced is not None and step + 1 == len(forced)):
                break
            padded[:, cache.get_pos()] = token
            state["prefix"] = f"decode.{step}."
            logits = model(tokens=torch.tensor([[token]], device="cuda:0"), attention_mask=attention_mask, kv_cache=cache)
            capture("logits", logits[:, -1])
    tensors["teacher_tokens"] = torch.tensor(chosen, dtype=torch.long)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(args.output / "trace.safetensors"))
    (args.output / "teacher-tokens.json").write_text(json.dumps(chosen) + "\n")
    metadata = {"schema_version": 1, "model_revision": REVISION, "weights_sha256": WEIGHT_SHA256,
                "environment": environment, "precision": args.precision, "tf32": False,
                "flex_float32_precision": "ieee", "attention": args.attention,
                "rms_norm_implicit_epsilon": torch.finfo(torch.float32).eps, "compiled_blocks": False,
                "allow_bf16_reduced_precision_reduction": False,
                "cast_contract": {"model_conversion": f"model.to(device='cuda:0', dtype={dtype}) then _ensure_device_buffers()",
                                  "parameter_dtype": str(next(model.parameters()).dtype), "sinks_dtype": str(model.layers["0"].attention.sinks.dtype),
                                  "final_norm_weight_dtype": str(model.norm.weight.dtype), "final_norm_epsilon": config.norm_eps,
                                  "golden_frequency_dtype": str(model.freqs_cis_golden.dtype), "temporal_frequency_dtype": str(model.freqs_cis.dtype),
                                  "kv_cache_dtype": str(cache.kv_cache.dtype), "logits_dtype": str(logits.dtype)},
                "input_image": str(args.image), "image_sha256": sha256(args.image),
                "width": image.width, "height": image.height, "min_dimension": args.min_dimension,
                "max_dimension": args.max_dimension, "prompt": prompt, "prefix_length": prefix_length,
                "cache_capacity": capacity, "max_new_tokens": args.max_new_tokens,
                "teacher_forced": forced is not None, "token_ids": chosen,
                "text": tokenizer.decode(chosen, skip_special_tokens=False),
                "finish_reason": "eos" if chosen and chosen[-1] in stop_ids else "length",
                "generated_fixture": generated_fixture, "timings": timings,
                "trace_sha256": sha256(args.output / "trace.safetensors"),
                "note": "Capture and compilation timings are not inference benchmarks. Decode step 0 consumes teacher_tokens[0]."}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "text": metadata["text"], "trace_tensors": len(tensors)}), flush=True)


if __name__ == "__main__":
    main()
