#!/usr/bin/env python3
"""Measure Falcon-OCR KV cache precision sensitivity by layer and kind.

Run uses the checkpoint's FP32 PyTorch code on CUDA, with TF32 disabled and
IEEE FlexAttention. Each policy replays the reference's greedy tokens with
one cache format per layer and kind. Rank builds cumulative q8 curve sets;
summarize combines step-weighted KL and compares GPU and CPU reference tokens.
Requires Python 3.10+, PyTorch with CUDA, transformers, safetensors, Pillow,
and NumPy. Run from the repository root with CUBLAS_WORKSPACE_CONFIG=:4096:8
and HF_HUB_OFFLINE=1. Outputs resume by page and policy without replacement.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
LAYERS, HEADS, GROUPS, HEAD_DIM = 22, 16, 8, 64
BLOCK = 32
SCALE = 0.125
PROMPT = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
LEVELS = {"q16": 32767, "q8": 127, "q4": 7}
SAVED_BYTES = {"k": 768, "v": 512}
FULL_BYTES = 58080
JOURNAL_POSITIONS = 6544


def bf16_up(x: Any) -> Any:
    """The smallest BF16 value at or above each non-negative FP32 value."""
    import torch

    bits = x.contiguous().view(torch.int32)
    high = (bits >> 16) + ((bits & 0xFFFF) != 0).to(torch.int32)
    return (high << 16).view(torch.float32)


def block_quantize(x: Any, levels: int, midrise: bool = False) -> tuple[Any, Any]:
    """Dequantized absmax codes of every 32-value block of the last axis and
    each block's step. levels=127 is the runner's Q8 (codes rounded to nearest
    even); midrise uses the 2-bit levels +-0.5 and +-1.5 steps."""
    import torch

    shape = x.shape
    b = x.reshape(*shape[:-1], -1, BLOCK)
    top = b.abs().amax(-1, keepdim=True)
    limit = 1.5 if midrise else float(levels)
    step = (top.double() / limit).float().clamp_min(float(np.finfo(np.float32).smallest_subnormal))
    step = torch.where(top == 0, torch.zeros_like(step), bf16_up(step))
    safe = torch.where(step == 0, torch.ones_like(step), step).double()
    if midrise:
        codes = (torch.floor(b.double() / safe) + 0.5).clamp(-1.5, 1.5)
    else:
        codes = torch.round(b.double() / safe).clamp(-levels, levels)
    codes = torch.where(step == 0, torch.zeros_like(codes), codes)
    return (codes * step.double()).float().reshape(shape), step.squeeze(-1)


def load_model(model_dir: Path, device: str) -> tuple[Any, ModuleType, Any, Any]:
    """The checkpoint's own code (through transformers' remote-code loader)
    with export_reference.py's FP32 settings."""
    import functools

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    cls = get_class_from_dynamic_module(
        "modeling_falcon_ocr.FalconOCRForCausalLM", str(model_dir), local_files_only=True
    )
    mm = importlib.import_module(cls.__module__)
    for name in ["compiled_flex_attn_prefill", "compiled_flex_attn_decode"]:
        setattr(mm, name, functools.partial(getattr(mm, name), kernel_options={"FLOAT32_PRECISION": "'ieee'"}))
    config = mm.FalconOCRConfig.from_json_file(str(model_dir / "config.json"))
    config._name_or_path = str(model_dir)
    model = cls(config)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True, assign=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    model._ensure_device_buffers()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return model, mm, config, tokenizer


class FlexCheck:
    """Wraps the decode FlexAttention call; while active, rebuilds each
    layer's output from the softmax over the sink and the scores."""

    def __init__(self, inner: Any, sinks: list[Any]) -> None:
        self.inner, self.sinks = inner, sinks
        self.active, self.calls, self.results = False, 0, []

    def __call__(self, q: Any, k: Any, v: Any, **kwargs: Any) -> tuple[Any, Any]:
        import torch

        out, aux = self.inner(q, k, v, **kwargs)
        if self.active:
            sink = self.sinks[self.calls % LAYERS].view(1, HEADS, 1, 1)
            self.calls += 1
            s = (q @ k.transpose(-1, -2)) * SCALE
            p = torch.softmax(torch.cat([sink.expand(1, HEADS, 1, 1), s], -1), -1)
            mine = p[..., 1:] @ v
            theirs = out * torch.sigmoid(aux.lse - sink.view(1, HEADS, 1)).unsqueeze(-1)
            self.results.append(
                {
                    "max_abs": float((mine - theirs).abs().max()),
                    "max_ref": float(theirs.abs().max()),
                    "lse_max_abs": float((torch.logsumexp(s, -1) - aux.lse).abs().max()),
                    "sink_mass_mean": float(p[..., 0].mean()),
                }
            )
        return out, aux


@dataclass
class Page:
    path: Path
    page_id: str
    prefix: int
    img0: int
    img1: int
    grid_h: int
    grid_w: int
    tokens: list[int]
    finish: str
    pos_t_last: int
    k_prefix: object  # (layers, 16, prefix, 64): keys per query head
    v_prefix: object  # (layers, 8, prefix, 64): values per KV group
    ref_logp: object  # (decode steps, vocab)
    verify: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)

    @property
    def n_img(self) -> int:
        return self.img1 - self.img0

    @property
    def nb(self) -> int:
        return -(-self.n_img // BLOCK)

    @property
    def steps(self) -> int:
        return len(self.tokens) - 1


def reference_pass(
    model: Any, mm: ModuleType, config: Any, tokenizer: Any, path: Path, max_new: int, max_dim: int, check: FlexCheck
) -> Page:
    """Capture the FP32 greedy tokens, prefix cache and decode log-probabilities."""
    import torch
    from PIL import Image

    device = model.tok_embeddings.weight.device
    start = time.perf_counter()
    image = Image.open(path).convert("RGB")
    batch = mm.process_batch(
        tokenizer, config, [(image, PROMPT)], max_length=config.max_seq_len, min_dimension=64, max_dimension=max_dim
    )
    tokens = batch["tokens"]
    prefix = tokens.shape[1]
    img = (tokens[0] == config.img_id).nonzero().flatten()
    img0, img1 = int(img[0]), int(img[-1]) + 1
    assert img1 - img0 == img.numel(), "image tokens are not contiguous"
    mask_hw = batch["pixel_mask"][0, 0]
    grid_h, grid_w = int(mask_hw.any(1).sum()) // 16, int(mask_hw.any(0).sum()) // 16
    assert grid_h * grid_w == img.numel()
    capacity = ((prefix + max_new + 127) // 128) * 128
    assert capacity <= config.max_seq_len
    model._pad_token_id = batch["pad_token_id"]
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    padded = torch.full((1, capacity), model._pad_token_id, device=device, dtype=torch.long)
    padded[:, :prefix] = batch["tokens"]
    mask = model.get_attention_mask(padded, max_len=capacity)
    cache = mm.KVCache(1, capacity, config.n_heads, config.head_dim, config.n_layers)
    stop_ids = {config.eos_id, tokenizer.convert_tokens_to_ids("<|end_of_query|>")}
    chosen, finish, logps = [], "length", []
    with torch.inference_mode():
        logits = model(
            tokens=batch["tokens"],
            attention_mask=mask,
            kv_cache=cache,
            rope_pos_t=batch["pos_t"],
            rope_pos_hw=batch["pos_hw"],
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
        )
        k_prefix = cache.kv_cache[:, 0, 0, :, :prefix].clone()
        v_all = cache.kv_cache[:, 1, 0, :, :prefix]
        assert torch.equal(v_all[:, 0::2], v_all[:, 1::2]), "values are not repeated per KV group"
        v_prefix = v_all[:, 0::2].clone()
        pos_t_last = int(cache.pos_t.flatten()[0])
        prefill_seconds = time.perf_counter() - start
        for step in range(max_new):
            token = int(logits[0, -1].argmax())
            chosen.append(token)
            if token in stop_ids:
                finish = "eos"
                break
            if step + 1 == max_new:
                break
            padded[:, cache.get_pos()] = token
            # Decode call `step + 1` produces the logits of token `step + 1`.
            check.active = step == 0 or step + 2 == max_new
            check.calls = 0
            logits = model(tokens=torch.tensor([[token]], device=device), attention_mask=mask, kv_cache=cache)
            check.active = False
            logps.append(torch.log_softmax(logits[0, -1].float(), -1))
        torch.cuda.synchronize()
    ref_logp = torch.stack(logps) if logps else torch.empty(0, config.vocab_size, device=device)
    verify = (
        {}
        if not check.results
        else {
            "layers_checked": len(check.results),
            "max_abs": max(r["max_abs"] for r in check.results),
            "max_ref": max(r["max_ref"] for r in check.results),
            "lse_max_abs": max(r["lse_max_abs"] for r in check.results),
            "sink_mass_by_layer_first_step": [round(r["sink_mass_mean"], 4) for r in check.results[:LAYERS]],
        }
    )
    check.results = []
    return Page(
        path=path,
        page_id=path.parent.name,
        prefix=prefix,
        img0=img0,
        img1=img1,
        grid_h=grid_h,
        grid_w=grid_w,
        tokens=chosen,
        finish=finish,
        pos_t_last=pos_t_last,
        k_prefix=k_prefix,
        v_prefix=v_prefix,
        ref_logp=ref_logp,
        verify=verify,
        seconds={"prefill": prefill_seconds, "reference": time.perf_counter() - start},
    )


def softmax_with_sink(q: Any, keys: Any, sink: Any) -> tuple[Any, Any]:
    """Scores (16, n), probabilities over [sink, keys] (16, n + 1)."""
    import torch

    s = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1) * SCALE
    z = torch.cat([sink[:, None], s], 1)
    return s, torch.softmax(z, 1)


def weighted_values(p_keys: Any, values: Any) -> Any:
    """sum_j p_j v_j per query head: p_keys (16, n), values (8, n, 64)."""
    import torch

    return torch.bmm(p_keys.reshape(GROUPS, 2, -1), values).reshape(HEADS, HEAD_DIM)


@dataclass(frozen=True)
class Policy:
    """Store the format of each layer's keys and values."""

    name: str
    keys: tuple[str, ...]
    values: tuple[str, ...]
    changed: tuple[int, str] | None = None


def policy(name: str, entries: list[list[Any]] | None = None) -> Policy:
    """Expand one policy name into formats for every layer and kind."""
    if name == "fp32":
        return Policy(name, ("f32",) * LAYERS, ("f32",) * LAYERS)
    if name.startswith("all-"):
        fmt = name.removeprefix("all-")
        if fmt not in LEVELS:
            raise ValueError(f"unknown policy {name}")
        return Policy(name, (fmt,) * LAYERS, (fmt,) * LAYERS)
    baseline, changed = ("q16", "q8") if name.startswith("ne-") else ("q8", "q4")
    keys, values = [baseline] * LAYERS, [baseline] * LAYERS
    single = None
    if entries is None:
        family, layer, kind = name.split("-")
        if family not in ("ne", "fast") or kind not in ("k", "v"):
            raise ValueError(f"unknown policy {name}")
        single = (int(layer), kind)
        entries = [[int(layer), kind]]
    for layer, kind in entries:
        (keys if kind == "k" else values)[layer] = changed
    return Policy(name, tuple(keys), tuple(values), single)


def parse_policies(spec: str) -> list[Policy]:
    """Expand families and curve files in a comma-separated policy list."""
    result = []
    for name in spec.split(","):
        if name in ("ne", "fast"):
            result.extend(policy(f"{name}-{layer}-{kind}") for layer in range(LAYERS) for kind in ("k", "v"))
        elif name.startswith("sets:"):
            sets = json.loads(Path(name.removeprefix("sets:")).read_text(encoding="utf-8"))
            result.extend(policy(s["name"], s["entries"]) for s in sets if s["name"] != "all-q8")
        elif name:
            result.append(policy(name))
    return result


def quantized_prefixes(page: Page, formats: set[str]) -> dict[str, tuple[Any, Any]]:
    """Quantize the prefix once per requested format for one page."""
    import torch

    result = {"f32": (page.k_prefix, page.v_prefix)}
    for fmt in sorted(formats - {"f32"}):
        keys, values = torch.empty_like(page.k_prefix), torch.empty_like(page.v_prefix)
        for layer in range(LAYERS):
            keys[layer] = block_quantize(page.k_prefix[layer], LEVELS[fmt])[0]
            values[layer] = block_quantize(page.v_prefix[layer], LEVELS[fmt])[0]
        result[fmt] = keys, values
    return result


def teacher_forced(
    model: Any, mm: ModuleType, page: Page, selected: Policy, prefixes: dict[str, tuple[Any, Any]]
) -> dict:
    """Replay reference tokens using the selected key and value formats."""
    import torch

    device = page.k_prefix.device
    cap = page.prefix + len(page.tokens)
    keys_fp32 = torch.empty(LAYERS, HEADS, cap, HEAD_DIM, device=device)
    values_fp32 = torch.empty(LAYERS, GROUPS, cap, HEAD_DIM, device=device)
    keys_fp32[:, :, : page.prefix] = page.k_prefix
    values_fp32[:, :, : page.prefix] = page.v_prefix
    if selected.name == "fp32":
        keys, values = keys_fp32, values_fp32
    else:
        keys, values = torch.empty_like(keys_fp32), torch.empty_like(values_fp32)
        for layer in range(LAYERS):
            keys[layer, :, : page.prefix] = prefixes[selected.keys[layer]][0][layer]
            values[layer, :, : page.prefix] = prefixes[selected.values[layer]][1][layer]
    sinks = [layer.attention.sinks.detach() for layer in model.layers.values()]
    kl = torch.zeros(page.steps, device=device)
    flip = torch.zeros(page.steps, dtype=torch.bool, device=device)
    output_err = torch.zeros(page.steps, device=device) if selected.changed else None
    targets = torch.tensor(page.tokens, device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(1, len(page.tokens)):
            pos = page.prefix + i - 1
            h = model.tok_embeddings(targets[i - 1].view(1, 1))
            freqs = model.freqs_cis[torch.tensor([[page.pos_t_last + i]], device=device)]
            for layer_id, layer in enumerate(model.layers.values()):
                attn = layer.attention
                xq, xk, xv = attn._pre_attention_qkv(h)
                xq, xk = mm.apply_3d_rotary_emb(xq, xk, freqs, None, None)
                key, value = xk[0, 0], xv[0, 0, 0::2]
                keys_fp32[layer_id, :, pos] = key
                values_fp32[layer_id, :, pos] = value
                kfmt, vfmt = selected.keys[layer_id], selected.values[layer_id]
                keys[layer_id, :, pos] = key if kfmt == "f32" else block_quantize(key, LEVELS[kfmt])[0]
                values[layer_id, :, pos] = value if vfmt == "f32" else block_quantize(value, LEVELS[vfmt])[0]
                n, q = pos + 1, xq[0, 0]
                _, probs = softmax_with_sink(q, keys[layer_id, :, :n], sinks[layer_id])
                o = weighted_values(probs[:, 1:], values[layer_id, :, :n])
                if selected.changed and selected.changed[0] == layer_id:
                    _, p_ref = softmax_with_sink(q, keys_fp32[layer_id, :, :n], sinks[layer_id])
                    o_ref = weighted_values(p_ref[:, 1:], values_fp32[layer_id, :, :n])
                    output_err[i - 1] = (o - o_ref).norm() / o_ref.norm().clamp_min(1e-30)
                h = h + attn.wo(o.reshape(1, 1, -1))
                h = h + layer.feed_forward(h)
            logp = torch.log_softmax(model.output(model.norm(h))[0, -1], -1)
            ref = page.ref_logp[i - 1]
            kl[i - 1] = (ref.exp() * (ref - logp)).sum()
            flip[i - 1] = logp.argmax() != targets[i]
        torch.cuda.synchronize()
    return {
        "steps": page.steps,
        "flips": int(flip.sum()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "output_err_mean": float(output_err.mean()) if output_err is not None else None,
        "seconds": time.perf_counter() - start,
    }


def write_json(path: Path, data: Any) -> None:
    """Write a new JSON result without replacing previous evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def run(
    pages: list[Path], model_dir: Path, output: Path, spec: str, max_new_tokens: int, max_dimension: int, threads: int
) -> int:
    """Run missing policy passes on CUDA and retain per-page results."""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.set_num_threads(threads)
    selected = parse_policies(spec)
    output.mkdir(parents=True, exist_ok=True)
    model, mm, config, tokenizer = load_model(model_dir, "cuda:0")
    logger.warning(
        "CUDA device: %s; model device: %s; CPU threads: %d",
        torch.cuda.get_device_name(),
        model.tok_embeddings.weight.device,
        torch.get_num_threads(),
    )
    sinks = [layer.attention.sinks.detach() for layer in model.layers.values()]
    check = FlexCheck(mm.compiled_flex_attn_decode, sinks)
    mm.compiled_flex_attn_decode = check
    with torch.inference_mode():
        for path in pages:
            directory = output / path.parent.name
            directory.mkdir(parents=True, exist_ok=True)
            missing = [p for p in selected if not (directory / f"{p.name}.json").exists()]
            if not missing:
                continue
            page = reference_pass(model, mm, config, tokenizer, path, max_new_tokens, max_dimension, check)
            record = {
                "page": str(path),
                "page_id": page.page_id,
                "input_tokens": page.prefix,
                "token_ids": page.tokens,
                "finish": page.finish,
                "steps": page.steps,
                "flex_check": page.verify,
                "seconds": page.seconds,
                "device": torch.cuda.get_device_name(),
                "dtype": str(model.tok_embeddings.weight.dtype),
            }
            ref_path = directory / "reference.json"
            if not ref_path.exists():
                write_json(ref_path, record)
            else:
                previous = json.loads(ref_path.read_text(encoding="utf-8"))
                if previous["token_ids"] != record["token_ids"] or previous["input_tokens"] != page.prefix:
                    raise RuntimeError(f"reference changed on resume: {page.page_id}")
            logger.warning(
                "%s: %d input tokens, %d outputs, finish %s; flex max abs %.3g",
                page.page_id,
                page.prefix,
                len(page.tokens),
                page.finish,
                page.verify["max_abs"],
            )
            prefixes = quantized_prefixes(page, {f for p in missing for f in p.keys + p.values})
            for chosen in missing:
                result = teacher_forced(model, mm, page, chosen, prefixes)
                write_json(directory / f"{chosen.name}.json", result)
                logger.warning("%s %s: %s", page.page_id, chosen.name, result)
            del prefixes, page
            torch.cuda.empty_cache()
    return 0


def collect(directory: Path) -> tuple[list[dict], dict[str, dict]]:
    """Aggregate each policy by its number of forced decode steps."""
    references, passes = [], {}
    for ref_path in sorted(directory.glob("*/reference.json")):
        references.append(json.loads(ref_path.read_text(encoding="utf-8")))
        for path in ref_path.parent.glob("*.json"):
            if path.name == "reference.json":
                continue
            item = json.loads(path.read_text(encoding="utf-8"))
            total = passes.setdefault(
                path.stem,
                {
                    "pages": 0,
                    "steps": 0,
                    "flips": 0,
                    "kl_sum": 0.0,
                    "kl_max": -float("inf"),
                    "err_sum": 0.0,
                    "seconds": 0.0,
                },
            )
            total["pages"] += 1
            total["steps"] += item["steps"]
            total["flips"] += item["flips"]
            total["kl_sum"] += item["steps"] * item["kl_mean"]
            total["kl_max"] = max(total["kl_max"], item["kl_max"])
            total["err_sum"] += item["steps"] * (item["output_err_mean"] or 0.0)
            total["seconds"] += item["seconds"]
    for total in passes.values():
        total["kl_mean"] = total.pop("kl_sum") / total["steps"]
        total["output_err_mean"] = total.pop("err_sum") / total["steps"]
    return references, passes


def ranked_rows(passes: dict[str, dict], family: str) -> list[dict]:
    """Sort available single-pair policies and retain their page coverage."""
    rows = []
    for layer in range(LAYERS):
        for kind in ("k", "v"):
            name = f"{family}-{layer}-{kind}"
            if name not in passes:
                continue
            item = passes[name]
            saved = SAVED_BYTES[kind] if family == "ne" else SAVED_BYTES[kind] // 2
            rows.append(
                {
                    "name": name,
                    "layer": layer,
                    "kind": kind,
                    "saved_bytes": saved,
                    "saved_mb_journal": saved * JOURNAL_POSITIONS / 1e6,
                    **item,
                }
            )
    return sorted(rows, key=lambda row: row["kl_mean"])


def rank(directory: Path, output: Path) -> int:
    """Build shortest ranked prefixes reaching the registered byte targets."""
    references, passes = collect(directory)
    if len(references) != 10 or any(
        passes[f"ne-{layer}-{k}"]["pages"] != 10 for layer in range(LAYERS) for k in ("k", "v")
    ):
        raise ValueError("rank requires all 44 near-exact policies on all 10 pages")
    rows, sets = ranked_rows(passes, "ne"), []
    for target in (10, 15, 20, 30, 40):
        entries, saved = [], 0
        for row in rows:
            entries.append([row["layer"], row["kind"]])
            saved += row["saved_bytes"]
            if saved / FULL_BYTES >= target / 100:
                break
        sets.append(
            {
                "name": f"ne-set-{target}",
                "target": target,
                "entries": entries,
                "knob": ",".join(f"{layer}{k}" for layer, k in entries),
                "saved_share": saved / FULL_BYTES,
                "saved_mb_journal": saved * JOURNAL_POSITIONS / 1e6,
            }
        )
    saved = LAYERS * (SAVED_BYTES["k"] + SAVED_BYTES["v"])
    sets.append(
        {
            "name": "all-q8",
            "target": "all",
            "entries": [[layer, k] for layer in range(LAYERS) for k in ("k", "v")],
            "knob": ",".join(f"{layer}{k}" for layer in range(LAYERS) for k in ("k", "v")),
            "saved_share": saved / FULL_BYTES,
            "saved_mb_journal": saved * JOURNAL_POSITIONS / 1e6,
        }
    )
    write_json(output, sets)
    return 0


def summarize(directory: Path, sets_path: Path, cpu_reference: Path, output: Path) -> int:
    """Combine simulation tables and compare GPU reference tokens with CPU."""
    references, passes = collect(directory)
    ref = json.loads(cpu_reference.read_text(encoding="utf-8"))
    cpu = {
        Path(inp["path"]).parent.name: out["token_ids"]
        for inp, out in zip(ref["inputs"], ref["samples"][0]["outputs"], strict=True)
    }
    pages = []
    for record in references:
        gpu, anchor = record["token_ids"], cpu.get(record["page_id"], [])
        same = next(
            (i for i, (x, y) in enumerate(zip(gpu, anchor, strict=False)) if x != y), min(len(gpu), len(anchor))
        )
        pages.append(
            {
                **record,
                "cpu_exact_reference": {"gpu_tokens": len(gpu), "cpu_tokens": len(anchor), "common_prefix": same},
            }
        )
    sets = json.loads(sets_path.read_text(encoding="utf-8"))
    curve = [{**item, **passes[item["name"]]} for item in sets]
    eligible = [item for item in curve if item["kl_mean"] <= 1e-6 and item["flips"] <= passes["all-q16"]["flips"] + 1]
    chosen = max(eligible, key=lambda item: item["saved_share"]) if eligible else None
    write_json(
        output,
        {
            "pages": pages,
            "passes": passes,
            "near_exact_ranked": ranked_rows(passes, "ne"),
            "fast_ranked": ranked_rows(passes, "fast"),
            "curve": curve,
            "chosen": chosen,
            "stage1_qualifies": chosen is not None and chosen["saved_share"] >= 0.15,
        },
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the run, rank and summarize command-line interfaces."""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase diagnostic verbosity")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run missing policy passes on CUDA", formatter_class=formatter)
    r.add_argument("pages", nargs="+", type=Path, help="canonical RGB page images")
    r.add_argument("--model", type=Path, required=True, help="FP32 checkpoint and Python source directory")
    r.add_argument("--output", type=Path, required=True, help="per-page result directory")
    r.add_argument(
        "--policies", default="fp32,all-q16,all-q8,all-q4,ne", help="comma-separated policies, families or sets:<file>"
    )
    r.add_argument("--max-new-tokens", type=int, default=512, help="greedy reference output limit")
    r.add_argument("--max-dimension", type=int, default=1536, help="maximum page dimension")
    r.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    r.set_defaults(func=run)
    k = sub.add_parser("rank", help="build cumulative near-exact curve sets", formatter_class=formatter)
    k.add_argument("directory", type=Path, help="per-page simulation directory")
    k.add_argument("--output", type=Path, required=True, help="new sets JSON file")
    k.set_defaults(func=rank)
    s = sub.add_parser("summarize", help="combine sensitivity and curve tables", formatter_class=formatter)
    s.add_argument("directory", type=Path, help="per-page simulation directory")
    s.add_argument("--sets", type=Path, required=True, help="ranked curve sets JSON")
    s.add_argument("--cpu-reference", type=Path, required=True, help="FP32 CPU reference bench report")
    s.add_argument("--output", type=Path, required=True, help="new summary JSON")
    s.set_defaults(func=summarize)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected command and return its exit status."""
    args = parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), stream=sys.stderr, format="%(message)s"
    )
    try:
        if args.command == "run":
            return args.func(
                args.pages,
                args.model,
                args.output,
                args.policies,
                args.max_new_tokens,
                args.max_dimension,
                args.threads,
            )
        if args.command == "rank":
            return args.func(args.directory, args.output)
        return args.func(args.directory, args.sets, args.cpu_reference, args.output)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
