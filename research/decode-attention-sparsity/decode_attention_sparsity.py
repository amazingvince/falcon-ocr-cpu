"""How sparse is Falcon-OCR's decode attention over the image prefix?

GPU study with the checkpoint's own PyTorch code (FP32, TF32 off, FlexAttention
at IEEE precision, as scripts/export_reference.py sets it up). For each page:

1. Reference: greedy decode through the model's own forward (FlexAttention),
   up to --max-new-tokens or EOS. The log-probabilities of every decode step
   are kept, and at two steps every layer's FlexAttention call is checked
   against probabilities rebuilt here from its query and keys plus the learned
   sink (max abs error of the attention output).
2. Statistics pass: a teacher-forced re-run of the same tokens with attention
   computed here, exactly, from the cached keys and values. At every decode
   step, layer and query head the softmax over the sink and all keys is split
   into the image positions, grouped into consecutive blocks of 32 in cache
   order, and the rest (registers, prompt, generated text: always read). Per
   (layer, KV group) it records the oracle skip fraction (blocks whose mass,
   max over the group's two query heads, is below tau), the dropped mass, the
   relative error of each head's attention output when those blocks are
   dropped and the softmax renormalized, the same output's error from a Q8
   cache (absmax codes against a BF16 scale rounded up per 32 values, as
   src/quant/kv/codec.rs), and how many blocks cheap upper bounds on each
   block's max score would let a kernel skip (see BOUNDS).
3. Policy passes: teacher-forced runs that really apply a policy (Q8 cache,
   oracle drop, bound-based skip, or combinations) at every decode step in
   every layer, counting argmax flips and the mean KL divergence against the
   reference's log-probabilities.

A bound-based kernel reads the always-read positions first, then visits image
blocks in descending bound order and skips a block once
bound < lse_run + ln(tau / 32), where lse_run is the log-sum-exp of the sink
and every score read so far ("ball-max" uses the running max instead, the
weaker test exp(-delta) * 32 <= tau). A skipped block then holds at most tau of
the final softmax mass.

Usage (from the repository root; the GPU run needs CUDA):
    CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 python3 \\
        research/decode-attention-sparsity/decode_attention_sparsity.py run \\
        --model artifacts/model --output <new dir> <page.png> ...
    python3 research/decode-attention-sparsity/decode_attention_sparsity.py \\
        summarize <dir> ... --output <new summary.json>
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LAYERS, HEADS, GROUPS, HEAD_DIM = 22, 16, 8, 64
BLOCK = 32
SCALE = 0.125  # FlexAttention's default 1/sqrt(head_dim)
TAUS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5)
# Upper bounds on a block's max score, from a per-block summary or sketch:
#   ball-max  q.c + |q| r (c the BF16 mean key, r the max distance to it), running-max test
#   ball      the same bound, running log-sum-exp test (all the others use it too)
#   split     the smaller of ball and the bound from the temporal and spatial halves apart
#   box       sum_d max(q_d lo_d, q_d hi_d), per-dimension min/max rounded outward to BF16
#   sketch4   max_j q.k4_j + sum_halves |q_half|_1 step/2, k4 a 4-bit absmax sketch of every key
#   sketch2   the same with a 2-bit sketch (levels +-0.5, +-1.5 steps)
BOUNDS = ("ball-max", "ball", "split", "box", "sketch4", "sketch2")
MARGIN = 1e-3  # score units added to every bound against FP32 rounding
# Bytes per position and KV group of the Q8 split record (src/quant/kv.rs):
# temporal key half shared by the pair, two spatial halves, the value; 160
# codes and five BF16 scales.
RECORD_BYTES = 170
VALUE_BYTES = 68
# Summary bytes per 32-position block and KV group (temporal half shared):
SUMMARY_BYTES = {
    "ball-max": 96 * 2 + 2 * 4,  # BF16 centroid halves, two FP32 radii
    "ball": 96 * 2 + 2 * 4,
    "split": 96 * 2 + 5 * 4,  # plus a temporal radius and two spatial radii
    "box": 192 * 2,  # BF16 lo and hi of the three 32-value halves
    "sketch4": BLOCK * 3 * (16 + 2),  # three 4-bit halves with BF16 steps per position
    "sketch2": BLOCK * 3 * (8 + 2),
}
PROMPT = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
DEFAULT_POLICIES = (
    "exact",
    "q8",
    "oracle:1e-3",
    "oracle:3e-4",
    "oracle:1e-4",
    "values:1e-4",
    "ball-max:1e-3",
    "split:1e-3",
    "box:1e-3",
    "sketch4:1e-3",
    "q8+oracle:1e-4",
    "q8+sketch4:1e-3",
)


# ----------------------------------------------------------------------------
# Quantizers (torch; imported lazily so `summarize` needs only numpy)
# ----------------------------------------------------------------------------


def bf16_up(x):
    """The smallest BF16 value at or above each non-negative FP32 value."""
    import torch

    bits = x.contiguous().view(torch.int32)
    high = (bits >> 16) + ((bits & 0xFFFF) != 0).to(torch.int32)
    return (high << 16).view(torch.float32)


def bf16_outward(x, up: bool):
    """Each FP32 value rounded to BF16 toward +inf (up) or -inf (down)."""
    import torch

    bits = x.contiguous().view(torch.int32)
    low = bits & 0xFFFF
    trunc = bits - low  # magnitude truncated: toward zero
    away = (x < 0) if not up else (x >= 0)
    return torch.where(away & (low != 0), trunc + 0x10000, trunc).view(torch.float32)


def block_quantize(x, levels: int, midrise: bool = False):
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


def q8(x):
    return block_quantize(x, 127)[0]


# ----------------------------------------------------------------------------
# Model loading and the reference pass
# ----------------------------------------------------------------------------


def load_model(model_dir: Path, device: str):
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

    def __init__(self, inner, sinks):
        self.inner, self.sinks = inner, sinks
        self.active, self.calls, self.results = False, 0, []

    def __call__(self, q, k, v, **kwargs):
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


def reference_pass(model, mm, config, tokenizer, path: Path, max_new: int, max_dim: int, check: FlexCheck) -> Page:
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


# ----------------------------------------------------------------------------
# Block summaries and bounds
# ----------------------------------------------------------------------------


class Summaries:
    """Per-block summaries of one layer's image keys (16, n_img, 64)."""

    def __init__(self, keys, kinds: set[str]):
        import torch

        heads, n, dim = keys.shape
        nb = -(-n // BLOCK)
        pad = nb * BLOCK - n
        self.valid = torch.ones(nb * BLOCK, dtype=torch.bool, device=keys.device)
        if pad:
            self.valid[n:] = False
        kp = torch.nn.functional.pad(keys, (0, 0, 0, pad)).view(heads, nb, BLOCK, dim)
        valid = self.valid.view(1, nb, BLOCK, 1)
        count = valid.sum(2).float()
        if kinds & {"ball-max", "ball", "split"}:
            centroid = ((kp * valid).sum(2) / count).to(torch.bfloat16).float()
            diff = (kp - centroid[:, :, None]) * valid
            self.centroid = centroid
            self.radius = diff.norm(dim=-1).amax(-1)
            self.radius_t = diff[..., :32].norm(dim=-1).amax(-1)
            self.radius_s = diff[..., 32:].norm(dim=-1).amax(-1)
        if "box" in kinds:
            inf = torch.tensor(float("inf"), device=keys.device)
            self.lo = bf16_outward(torch.where(valid, kp, inf).amin(2), up=False)
            self.hi = bf16_outward(torch.where(valid, kp, -inf).amax(2), up=True)
        self.sketch = {}
        for kind, levels, midrise in (("sketch4", 7, False), ("sketch2", 1, True)):
            if kind in kinds:
                deq, step = block_quantize(kp.view(heads, nb * BLOCK, dim), levels, midrise)
                self.sketch[kind] = (deq, step / 2)  # (16, nb*32, 64), (16, nb*32, 2) max error

    def bound(self, kind: str, q):
        """Upper bound on every block's max score (16, nb) for queries q (16, 64)."""
        import torch

        qn = q.norm(dim=-1)
        if kind in ("ball-max", "ball", "split"):
            dot = torch.einsum("hd,hbd->hb", q, self.centroid)
            b = dot + qn[:, None] * self.radius
            if kind == "split":
                dot_t = torch.einsum("hd,hbd->hb", q[:, :32], self.centroid[..., :32])
                dot_s = dot - dot_t
                b_split = (
                    dot_t
                    + q[:, :32].norm(dim=-1)[:, None] * self.radius_t
                    + dot_s
                    + q[:, 32:].norm(dim=-1)[:, None] * self.radius_s
                )
                b = torch.minimum(b, b_split)
        elif kind == "box":
            b = torch.maximum(q[:, None] * self.lo, q[:, None] * self.hi).sum(-1)
        else:
            deq, err = self.sketch[kind]
            q1 = torch.stack([q[:, :32].abs().sum(-1), q[:, 32:].abs().sum(-1)], -1)  # (16, 2)
            pos = torch.bmm(deq, q.unsqueeze(-1)).squeeze(-1) + (err * q1[:, None]).sum(-1)
            pos = torch.where(self.valid[None], pos, torch.tensor(float("-inf"), device=q.device))
            b = pos.view(pos.shape[0], -1, BLOCK).amax(-1)
        return b * SCALE + MARGIN


def bound_reads(b, block_lse, block_max, seed_lse, seed_max, log_thr, use_max: bool):
    """Blocks a descending-bound scan reads (T, 16, nb) as 0/1 floats: block
    i is read while bound_i >= running(lse or max) + ln(tau / 32)."""
    import torch

    t = log_thr.numel()
    bs, idx = b.sort(1, descending=True)
    neg = torch.full_like(bs[:, :1], float("-inf"))
    if use_max:
        ms = block_max.gather(1, idx)
        prev = torch.cat([neg, ms.cummax(1).values[:, :-1]], 1)
        run = torch.maximum(prev, seed_max[:, None])
    else:
        ls = block_lse.gather(1, idx)
        prev = torch.cat([neg, torch.logcumsumexp(ls, 1)[:, :-1]], 1)
        run = torch.logaddexp(prev, seed_lse[:, None])
    cond = (bs[None] >= run[None] + log_thr[:, None, None]).float().cumprod(-1)
    return torch.zeros_like(cond).scatter_(-1, idx[None].expand(t, -1, -1), cond)


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------


def softmax_with_sink(q, keys, sink):
    """Scores (16, n), probabilities over [sink, keys] (16, n + 1)."""
    import torch

    s = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1) * SCALE
    z = torch.cat([sink[:, None], s], 1)
    return s, torch.softmax(z, 1)


def weighted_values(p_keys, values):
    """sum_j p_j v_j per query head: p_keys (16, n), values (8, n, 64)."""
    import torch

    return torch.bmm(p_keys.reshape(GROUPS, 2, -1), values).reshape(HEADS, HEAD_DIM)


@dataclass
class Geometry:
    img0: int
    img1: int
    nb: int

    def image_blocks(self, x, fill: float):
        """The image part of x (16, n) padded to (16, nb, 32)."""
        import torch

        part = x[:, self.img0 : self.img1]
        pad = self.nb * BLOCK - part.shape[1]
        return torch.nn.functional.pad(part, (0, pad), value=fill).view(x.shape[0], self.nb, BLOCK)

    def others(self, x):
        """The always-read positions of x (16, n): before and after the image."""
        import torch

        return torch.cat([x[:, : self.img0], x[:, self.img1 :]], 1)


def to_positions(block_keep, geo: Geometry, n: int):
    """Per-position keep mask (16, n) from a per-block one (16, nb)."""
    import torch

    keep = torch.ones(block_keep.shape[0], n, device=block_keep.device)
    keep[:, geo.img0 : geo.img1] = block_keep.repeat_interleave(BLOCK, 1)[:, : geo.img1 - geo.img0]
    return keep


@dataclass
class Policy:
    """A decode attention policy: q8 cache or not, then 'exact', 'oracle'
    (drop blocks with group mass < tau, renormalize), 'values' (drop their
    values only: the full denominator, as a two-pass kernel that reads every
    key), or a bound kind (skip what the descending-bound scan skips)."""

    spec: str
    q8: bool
    kind: str
    tau: float

    @classmethod
    def parse(cls, spec: str) -> Policy:
        body = spec.removeprefix("q8+")
        kind, _, tau = body.partition(":")
        kinds = ("exact", "q8", "oracle", "values", *BOUNDS)
        if kind not in kinds or (kind in ("exact", "q8")) != (tau == ""):
            raise ValueError(f"bad policy {spec!r}")
        return cls(spec, spec.startswith("q8+") or kind == "q8", kind, float(tau) if tau else 0.0)

    def attend(self, q, keys, values, sink, geo: Geometry, summ: Summaries | None):
        """Output (16, 64) and the fraction of image blocks skipped per group (8,)."""
        import torch

        s, p = softmax_with_sink(q, keys, sink)
        if self.kind in ("exact", "q8"):
            return weighted_values(p[:, 1:], values), None
        n = s.shape[1]
        if self.kind in ("oracle", "values"):
            mass = geo.image_blocks(p[:, 1:], 0.0).sum(-1)
            keep = (mass.view(GROUPS, 2, -1).amax(1) >= self.tau).float()
        else:
            b = summ.bound(self.kind, q)
            sb = geo.image_blocks(s, float("-inf"))
            seed = torch.cat([sink[:, None], geo.others(s)], 1)
            log_thr = torch.tensor([math.log(self.tau / BLOCK)], device=q.device)
            read = bound_reads(
                b,
                torch.logsumexp(sb, -1),
                sb.amax(-1),
                torch.logsumexp(seed, 1),
                seed.amax(1),
                log_thr,
                self.kind == "ball-max",
            )[0]
            keep = read.view(GROUPS, 2, -1).amax(1)
        pk = p[:, 1:] * to_positions(keep.repeat_interleave(2, 0), geo, n)
        if self.kind != "values":
            pk = pk / (p[:, :1] + pk.sum(1, keepdim=True))
        return weighted_values(pk, values), 1.0 - keep.mean(1)


# ----------------------------------------------------------------------------
# The statistics pass
# ----------------------------------------------------------------------------


class Stats:
    """Per-step statistics of exact decode attention (see the module doc)."""

    def __init__(self, page: Page, device):
        import torch

        s, nb, t, k = page.steps, page.nb, len(TAUS), len(BOUNDS)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        self.geo = Geometry(page.img0, page.img1, nb)
        self.grid = (page.grid_h, page.grid_w)
        self.log_thr = torch.tensor([math.log(x / BLOCK) for x in TAUS], device=device)
        self.taus = torch.tensor(TAUS, device=device)
        self.arrays = {
            "sink_mass": z(s, LAYERS, HEADS),
            "other_mass": z(s, LAYERS, HEADS),
            "oracle_skip": z(s, LAYERS, GROUPS, t),
            "oracle_drop": z(s, LAYERS, HEADS, t),
            "oracle_err": z(s, LAYERS, HEADS, t),
            "oracle_err_layer": z(s, LAYERS, t),
            "values_err_layer": z(s, LAYERS, t),
            "q8_err": z(s, LAYERS, HEADS),
            "q8_err_layer": z(s, LAYERS),
            "bound_skip": z(s, LAYERS, GROUPS, k, t),
            "bound_drop_max": z(s, LAYERS, k, t),
            "bound_err_layer": z(s, LAYERS, k, t),
            "top_block": z(s, LAYERS, GROUPS, dtype=torch.int16),
            "cover90": z(s, LAYERS, GROUPS, dtype=torch.int16),
            "cover99": z(s, LAYERS, GROUPS, dtype=torch.int16),
            "row_mass": z(s, LAYERS, GROUPS, page.grid_h, dtype=torch.float16),
        }
        self.violations = z(k, dtype=torch.int64)
        self.checked = z(k, dtype=torch.int64)

    def layer(self, step: int, layer: int, q, keys, values, keys_q8, values_q8, sink, summ: Summaries):
        """Records one (step, layer); returns the exact output (16, 64)."""
        import torch

        a, geo = self.arrays, self.geo
        s, p = softmax_with_sink(q, keys, sink)
        o = weighted_values(p[:, 1:], values)
        o_norm = o.norm(dim=-1).clamp_min(1e-30)
        o_total = o.norm().clamp_min(1e-30)
        pi = geo.image_blocks(p[:, 1:], 0.0)  # (16, nb, 32)
        bm = pi.sum(-1)  # (16, nb)
        a["sink_mass"][step, layer] = p[:, 0]
        a["other_mass"][step, layer] = 1.0 - p[:, 0] - bm.sum(-1)
        gm = bm.view(GROUPS, 2, -1).amax(1)  # (8, nb)
        # Each block's share of every head's output (16, nb, 64).
        vi = torch.nn.functional.pad(values[:, geo.img0 : geo.img1], (0, 0, 0, geo.nb * BLOCK - (geo.img1 - geo.img0)))
        ob = torch.einsum("gkbj,gbjd->gkbd", pi.view(GROUPS, 2, geo.nb, BLOCK), vi.view(GROUPS, geo.nb, BLOCK, -1))
        ob = ob.reshape(HEADS, geo.nb, HEAD_DIM)

        def dropped_error(drop_heads):  # drop_heads (T, 16, nb) 0/1
            mass = (drop_heads * bm).sum(-1)  # (T, 16)
            contrib = torch.einsum("thb,hbd->thd", drop_heads, ob)
            renorm = (o - contrib) / (1.0 - mass)[..., None]
            err = (renorm - o).norm(dim=-1) / o_norm
            err_layer = (renorm - o).flatten(1).norm(dim=-1) / o_total
            return mass, err, err_layer, contrib.flatten(1).norm(dim=-1) / o_total

        drop = (gm[None] < self.taus[:, None, None]).float()  # (T, 8, nb)
        a["oracle_skip"][step, layer] = drop.mean(-1).T
        mass, err, err_layer, err_values = dropped_error(drop.repeat_interleave(2, 1))
        a["oracle_drop"][step, layer] = mass.T
        a["oracle_err"][step, layer] = err.T
        a["oracle_err_layer"][step, layer] = err_layer
        a["values_err_layer"][step, layer] = err_values
        _, pq = softmax_with_sink(q, keys_q8, sink)
        oq = weighted_values(pq[:, 1:], values_q8)
        a["q8_err"][step, layer] = (oq - o).norm(dim=-1) / o_norm
        a["q8_err_layer"][step, layer] = (oq - o).norm() / o_total
        sb = geo.image_blocks(s, float("-inf"))
        block_lse, block_max = torch.logsumexp(sb, -1), sb.amax(-1)
        seed = torch.cat([sink[:, None], geo.others(s)], 1)
        seed_lse, seed_max = torch.logsumexp(seed, 1), seed.amax(1)
        for k, kind in enumerate(BOUNDS):
            b = summ.bound(kind, q)
            self.violations[k] += (block_max > b).sum()
            self.checked[k] += b.numel()
            read = bound_reads(b, block_lse, block_max, seed_lse, seed_max, self.log_thr, kind == "ball-max")
            gread = read.view(len(TAUS), GROUPS, 2, -1).amax(2)  # (T, 8, nb)
            a["bound_skip"][step, layer, :, k] = 1.0 - gread.mean(-1).T
            mass, _, err_layer, _ = dropped_error(1.0 - gread.repeat_interleave(2, 1))
            a["bound_drop_max"][step, layer, k] = mass.amax(1)
            a["bound_err_layer"][step, layer, k] = err_layer
        # Locality: the most attended block, how many blocks hold 90% / 99% of
        # the group's image mass, and the image mass per patch row.
        pair = bm.view(GROUPS, 2, -1).mean(1)
        a["top_block"][step, layer] = gm.argmax(-1).to(torch.int16)
        total = pair.sum(-1, keepdim=True).clamp_min(1e-30)
        cum = pair.sort(-1, descending=True).values.cumsum(-1) / total
        a["cover90"][step, layer] = ((cum < 0.9).sum(-1) + 1).to(torch.int16)
        a["cover99"][step, layer] = ((cum < 0.99).sum(-1) + 1).to(torch.int16)
        img = p[:, 1 + geo.img0 : 1 + geo.img1].view(GROUPS, 2, -1).mean(1)
        rows = img.view(GROUPS, *self.grid).sum(-1)
        a["row_mass"][step, layer] = (rows / total).to(torch.float16)
        return o


# ----------------------------------------------------------------------------
# Teacher-forced decode
# ----------------------------------------------------------------------------


def teacher_forced(model, mm, page: Page, policy: Policy | None, stats: Stats | None) -> dict:
    """Re-runs the reference tokens with attention computed here (the
    policy's, or exactly with statistics); returns flips and KL per step."""
    import torch

    device = page.k_prefix.device
    n_tok = len(page.tokens)
    cap = page.prefix + n_tok
    keys = torch.zeros(LAYERS, HEADS, cap, HEAD_DIM, device=device)
    values = torch.zeros(LAYERS, GROUPS, cap, HEAD_DIM, device=device)
    keys[:, :, : page.prefix] = page.k_prefix
    values[:, :, : page.prefix] = page.v_prefix
    use_q8 = stats is not None or policy.q8
    if use_q8:
        keys_q8, values_q8 = torch.zeros_like(keys), torch.zeros_like(values)
        for layer in range(LAYERS):
            keys_q8[layer, :, : page.prefix] = q8(page.k_prefix[layer])
            values_q8[layer, :, : page.prefix] = q8(page.v_prefix[layer])
    geo = Geometry(page.img0, page.img1, page.nb)
    if stats is not None:
        kinds = set(BOUNDS)
    else:
        kinds = {policy.kind} if policy.kind in BOUNDS else set()
    summary_keys = keys_q8 if (policy is not None and policy.q8) else keys
    summaries = [
        Summaries(summary_keys[layer, :, page.img0 : page.img1], kinds) if kinds else None for layer in range(LAYERS)
    ]
    sinks = [layer.attention.sinks.detach() for layer in model.layers.values()]
    steps = page.steps
    kl = torch.zeros(steps, device=device)
    flip = torch.zeros(steps, dtype=torch.bool, device=device)
    logit_diff = torch.zeros(steps, device=device)
    skip = torch.zeros(steps, LAYERS, GROUPS, device=device)
    targets = torch.tensor(page.tokens, device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(1, n_tok):
            pos = page.prefix + i - 1
            h = model.tok_embeddings(targets[i - 1].view(1, 1))
            freqs = model.freqs_cis[torch.tensor([[page.pos_t_last + i]], device=device)]
            for layer_id, layer in enumerate(model.layers.values()):
                attn = layer.attention
                xq, xk, xv = attn._pre_attention_qkv(h)
                xq, xk = mm.apply_3d_rotary_emb(xq, xk, freqs, None, None)
                keys[layer_id, :, pos] = xk[0, 0]
                values[layer_id, :, pos] = xv[0, 0, 0::2]
                if use_q8:
                    keys_q8[layer_id, :, pos] = q8(xk[0, 0])
                    values_q8[layer_id, :, pos] = q8(xv[0, 0, 0::2])
                n = pos + 1
                q = xq[0, 0]
                if stats is not None:
                    o = stats.layer(
                        i - 1,
                        layer_id,
                        q,
                        keys[layer_id, :, :n],
                        values[layer_id, :, :n],
                        keys_q8[layer_id, :, :n],
                        values_q8[layer_id, :, :n],
                        sinks[layer_id],
                        summaries[layer_id],
                    )
                else:
                    k_src, v_src = (keys_q8, values_q8) if policy.q8 else (keys, values)
                    o, skipped = policy.attend(
                        q, k_src[layer_id, :, :n], v_src[layer_id, :, :n], sinks[layer_id], geo, summaries[layer_id]
                    )
                    if skipped is not None:
                        skip[i - 1, layer_id] = skipped
                h = h + attn.wo(o.reshape(1, 1, -1))
                h = h + layer.feed_forward(h)
            logp = torch.log_softmax(model.output(model.norm(h))[0, -1], -1)
            ref = page.ref_logp[i - 1]
            kl[i - 1] = (ref.exp() * (ref - logp)).sum()
            flip[i - 1] = logp.argmax() != targets[i]
            logit_diff[i - 1] = (ref - logp).abs().max()
        torch.cuda.synchronize()
    return {
        "kl": kl.cpu().numpy(),
        "flip": flip.cpu().numpy(),
        "logp_max_abs_diff": logit_diff.cpu().numpy(),
        "skip": skip.cpu().numpy(),
        "seconds": time.perf_counter() - start,
    }


def run(args: argparse.Namespace) -> int:
    import torch

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "set CUBLAS_WORKSPACE_CONFIG=:4096:8"
    torch.set_num_threads(args.threads)
    policies = [Policy.parse(spec) for spec in args.policies]
    args.output.mkdir(parents=True, exist_ok=True)
    model, mm, config, tokenizer = load_model(args.model, "cuda:0")
    sinks = [layer.attention.sinks.detach() for layer in model.layers.values()]
    check = FlexCheck(mm.compiled_flex_attn_decode, sinks)
    mm.compiled_flex_attn_decode = check
    for path in args.pages:
        page_id = path.parent.name
        json_out, npz_out = args.output / f"{page_id}.json", args.output / f"{page_id}.npz"
        if json_out.exists() or npz_out.exists():
            raise FileExistsError(f"{json_out} or {npz_out} exists; write to a new directory")
        page = reference_pass(model, mm, config, tokenizer, path, args.max_new_tokens, args.max_dimension, check)
        print(
            f"{page_id}: {page.prefix} input tokens, {page.n_img} image ({page.grid_h}x{page.grid_w}, "
            f"{page.nb} blocks), {len(page.tokens)} generated ({page.finish}); flex check max abs "
            f"{page.verify['max_abs']:.2e} (output max {page.verify['max_ref']:.2e}); "
            f"{page.seconds['reference']:.1f} s",
            flush=True,
        )
        if page.steps < 1:
            print(f"{page_id}: no decode steps, skipped", flush=True)
            continue
        stats = Stats(page, page.k_prefix.device)
        exact = teacher_forced(model, mm, page, None, stats)
        page.seconds["stats"] = exact["seconds"]
        arrays = {name: value.cpu().numpy() for name, value in stats.arrays.items()}
        record = {
            "page": str(path),
            "page_id": page_id,
            "input_tokens": page.prefix,
            "image_tokens": page.n_img,
            "image_start": page.img0,
            "grid": [page.grid_h, page.grid_w],
            "image_blocks": page.nb,
            "token_ids": page.tokens,
            "finish": page.finish,
            "decode_steps": page.steps,
            "flex_check": page.verify,
            "bound_violations": dict(zip(BOUNDS, stats.violations.tolist(), strict=True)),
            "bound_blocks_checked": dict(zip(BOUNDS, stats.checked.tolist(), strict=True)),
            "taus": list(TAUS),
            "bounds": list(BOUNDS),
            "policies": {"stats-exact": summarize_pass(exact)},
        }
        arrays["tf_stats-exact_kl"] = exact["kl"]
        arrays["tf_stats-exact_flip"] = exact["flip"]
        print(f"  stats pass: {exact['seconds']:.1f} s, {record['policies']['stats-exact']}", flush=True)
        del stats
        for policy in policies:
            result = teacher_forced(model, mm, page, policy, None)
            record["policies"][policy.spec] = summarize_pass(result)
            arrays[f"tf_{policy.spec}_kl"] = result["kl"]
            arrays[f"tf_{policy.spec}_flip"] = result["flip"]
            if policy.kind not in ("exact", "q8"):
                arrays[f"tf_{policy.spec}_skip"] = result["skip"].astype(np.float16)
            print(f"  {policy.spec}: {result['seconds']:.1f} s, {record['policies'][policy.spec]}", flush=True)
        record["seconds"] = page.seconds
        np.savez_compressed(npz_out, **arrays)
        json_out.write_text(json.dumps(record) + "\n")
        del page
        torch.cuda.empty_cache()
    return 0


def summarize_pass(result: dict) -> dict:
    out = {
        "steps": int(result["flip"].size),
        "flips": int(result["flip"].sum()),
        "kl_mean": float(result["kl"].mean()),
        "kl_max": float(result["kl"].max()),
        "logp_max_abs_diff": float(result["logp_max_abs_diff"].max()),
        "seconds": round(result["seconds"], 1),
    }
    if result["skip"].any():
        out["skip_mean"] = float(result["skip"].mean())
    return out


# ----------------------------------------------------------------------------
# Summaries over pages (numpy only)
# ----------------------------------------------------------------------------


def load_pages(dirs: list[Path]) -> list[tuple[dict, dict]]:
    pages = []
    for d in dirs:
        for js in sorted(d.glob("*.json")):
            record = json.loads(js.read_text())
            if "page_id" not in record:
                continue
            with np.load(js.with_suffix(".npz")) as z:
                pages.append((record, {k: z[k] for k in z.files}))
    return pages


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=np.float64), q))


def summarize(args: argparse.Namespace) -> int:
    pages = load_pages(args.dirs)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists")
    cpu_ref = {}
    if args.cpu_reference:
        ref = json.loads(args.cpu_reference.read_text())
        for inp, out in zip(ref["inputs"], ref["samples"][0]["outputs"], strict=True):
            cpu_ref[Path(inp["path"]).parent.name] = out["token_ids"]
    t3, t4 = TAUS.index(1e-3), TAUS.index(1e-4)
    result = {"pages": [], "taus": list(TAUS), "bounds": list(BOUNDS)}
    skip_keys = ("tf_", "row_mass")
    cat = {k: np.concatenate([a[k] for _, a in pages]) for k in pages[0][1] if not k.startswith(skip_keys)}
    for record, a in pages:
        pid = record["page_id"]
        entry = {
            "page_id": pid,
            "input_tokens": record["input_tokens"],
            "image_blocks": record["image_blocks"],
            "decode_steps": record["decode_steps"],
            "finish": record["finish"],
            "flex_check_max_abs": record["flex_check"]["max_abs"],
            "flex_check_max_ref": record["flex_check"]["max_ref"],
            "flex_check_lse_max_abs": record["flex_check"]["lse_max_abs"],
            "bound_violations": record["bound_violations"],
            "oracle_skip": [float(a["oracle_skip"][..., t].mean()) for t in range(len(TAUS))],
            "bound_skip_1e-3": {b: float(a["bound_skip"][:, :, :, k, t3].mean()) for k, b in enumerate(BOUNDS)},
            "sink_mass": float(a["sink_mass"].mean()),
            "other_mass": float(a["other_mass"].mean()),
            "policies": record["policies"],
        }
        if pid in cpu_ref:
            gpu, cpu = record["token_ids"], cpu_ref[pid]
            same = next((i for i, (x, y) in enumerate(zip(gpu, cpu, strict=False)) if x != y), min(len(gpu), len(cpu)))
            entry["cpu_exact_reference"] = {"gpu_tokens": len(gpu), "cpu_tokens": len(cpu), "common_prefix": same}
        result["pages"].append(entry)
    # Oracle table: skip, dropped mass and output error versus Q8.
    result["oracle"] = [
        {
            "tau": tau,
            "skip_mean": float(cat["oracle_skip"][..., t].mean()),
            "dropped_mass_mean": float(cat["oracle_drop"][..., t].mean()),
            "dropped_mass_p99": pct(cat["oracle_drop"][..., t], 99),
            "head_err_mean": float(cat["oracle_err"][..., t].mean()),
            "head_err_median": pct(cat["oracle_err"][..., t], 50),
            "head_err_p99": pct(cat["oracle_err"][..., t], 99),
            "layer_err_mean": float(cat["oracle_err_layer"][..., t].mean()),
            "layer_err_p99": pct(cat["oracle_err_layer"][..., t], 99),
            "values_layer_err_mean": float(cat["values_err_layer"][..., t].mean()),
        }
        for t, tau in enumerate(TAUS)
    ]
    result["q8"] = {
        "head_err_mean": float(cat["q8_err"].mean()),
        "head_err_median": pct(cat["q8_err"], 50),
        "head_err_p99": pct(cat["q8_err"], 99),
        "layer_err_mean": float(cat["q8_err_layer"].mean()),
        "layer_err_p99": pct(cat["q8_err_layer"], 99),
    }
    # Bound table: skip versus the oracle and the net bytes of the image blocks.
    bounds = []
    for k, kind in enumerate(BOUNDS):
        for t, tau in enumerate(TAUS):
            skip = float(cat["bound_skip"][:, :, :, k, t].mean())
            summary_share = SUMMARY_BYTES[kind] / (BLOCK * RECORD_BYTES)
            bounds.append(
                {
                    "bound": kind,
                    "tau": tau,
                    "skip_mean": skip,
                    "oracle_skip_mean": result["oracle"][t]["skip_mean"],
                    "summary_bytes_share": summary_share,
                    "image_bytes_saved": skip - summary_share,
                    "dropped_mass_max_mean": float(cat["bound_drop_max"][..., k, t].mean()),
                    "layer_err_mean": float(cat["bound_err_layer"][..., k, t].mean()),
                }
            )
    result["bounds_table"] = bounds
    # Whole-cache byte shares per page and step (image blocks plus the rest).
    result["kv_bytes_saved_share"] = {}
    for t in (t3, t4):
        totals = {}
        for kind_name, k in [(b, i) for i, b in enumerate(BOUNDS)] + [("oracle", None), ("values", None)]:
            saved = []
            for record, a in pages:
                n_pos = record["input_tokens"] + np.arange(record["decode_steps"]) + 1
                n_img, nb = record["image_tokens"], record["image_blocks"]
                if k is None:
                    skip = a["oracle_skip"][..., t].mean(axis=(1, 2))
                    per_pos = RECORD_BYTES if kind_name == "oracle" else VALUE_BYTES
                    saved.append(skip * n_img * per_pos / (n_pos * RECORD_BYTES))
                else:
                    skip = a["bound_skip"][:, :, :, k, t].mean(axis=(1, 2))
                    cost = nb * SUMMARY_BYTES[kind_name]
                    saved.append((skip * n_img * RECORD_BYTES - cost) / (n_pos * RECORD_BYTES))
            totals[kind_name] = float(np.concatenate(saved).mean())
        result["kv_bytes_saved_share"][f"{TAUS[t]:.0e}"] = totals
    # Characterization.
    result["by_layer"] = [
        {
            "layer": layer,
            "sink_mass": float(cat["sink_mass"][:, layer].mean()),
            "other_mass": float(cat["other_mass"][:, layer].mean()),
            "oracle_skip_1e-3": float(cat["oracle_skip"][:, layer, :, t3].mean()),
            "oracle_skip_1e-4": float(cat["oracle_skip"][:, layer, :, t4].mean()),
            "oracle_skip_1e-3_group_min": float(cat["oracle_skip"][:, layer, :, t3].mean(0).min()),
            "oracle_skip_1e-3_group_max": float(cat["oracle_skip"][:, layer, :, t3].mean(0).max()),
            **{f"{b}_skip_1e-3": float(cat["bound_skip"][:, layer, :, k, t3].mean()) for k, b in enumerate(BOUNDS)},
            "q8_err_layer": float(cat["q8_err_layer"][:, layer].mean()),
            "oracle_err_layer_1e-3": float(cat["oracle_err_layer"][:, layer, t3].mean()),
            "cover90": float(cat["cover90"][:, layer].mean()),
            "cover99": float(cat["cover99"][:, layer].mean()),
        }
        for layer in range(LAYERS)
    ]
    buckets = [(0, 32), (32, 128), (128, 256), (256, 512)]
    result["by_step"] = []
    for lo, hi in buckets:
        rows = [a for _, a in pages if a["oracle_skip"].shape[0] > lo]
        if not rows:
            continue
        oracle = np.concatenate([r["oracle_skip"][lo:hi] for r in rows])
        bound = np.concatenate([r["bound_skip"][lo:hi] for r in rows])
        result["by_step"].append(
            {
                "steps": [lo, hi],
                "oracle_skip_1e-3": float(oracle[..., t3].mean()),
                "oracle_skip_1e-4": float(oracle[..., t4].mean()),
                **{f"{b}_skip_1e-3": float(bound[:, :, :, k, t3].mean()) for k, b in enumerate(BOUNDS)},
            }
        )
    result["locality"] = locality(pages)
    # Teacher-forced policies over all pages.
    specs = list(pages[0][0]["policies"])
    result["teacher_forced"] = []
    for spec in specs:
        steps = sum(r["policies"][spec]["steps"] for r, _ in pages)
        flips = sum(r["policies"][spec]["flips"] for r, _ in pages)
        kl = np.concatenate([a[f"tf_{spec}_kl"] for _, a in pages])
        entry = {"policy": spec, "steps": steps, "flips": flips, "kl_mean": float(kl.mean()), "kl_p99": pct(kl, 99)}
        if f"tf_{spec}_skip" in pages[0][1]:
            entry["skip_mean"] = float(
                np.concatenate([a[f"tf_{spec}_skip"] for _, a in pages]).astype(np.float64).mean()
            )
        entry["pages_with_flips"] = [r["page_id"] for r, _ in pages if r["policies"][spec]["flips"]]
        result["teacher_forced"].append(entry)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print_tables(result)
    return 0


def locality(pages) -> dict:
    """Does image attention follow the text being read? Per (layer, group):
    the Pearson correlation of its most attended patch row with the step
    (reading order; NaN when the row never moves), and the share of its image
    mass within +-2 rows of the step's pointer row (the most attended row of
    the best-correlated group of that page), against the uniform share."""
    corr_all, band_all, band_uniform = [], [], []
    for _, a in pages:
        rows = a["row_mass"].astype(np.float64)  # (S, L, 8, H)
        s, _, _, h = rows.shape
        if s < 16:
            continue
        top = rows.argmax(-1).astype(np.float64)  # (S, L, 8)
        dt = top - top.mean(0)
        ds = (np.arange(s) - (s - 1) / 2)[:, None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (dt * ds).sum(0) / np.sqrt((dt**2).sum(0) * (ds**2).sum(0))
        corr_all.append(corr)
        best = np.unravel_index(np.nanargmax(corr), corr.shape)
        pointer = top[:, best[0], best[1]]  # (S,)
        band = (np.abs(np.arange(h)[None, :] - pointer[:, None]) <= 2).astype(np.float64)  # (S, H)
        band_all.append((rows * band[:, None, None, :]).sum(-1).mean(0))  # (L, 8)
        band_uniform.append(band.sum(-1).mean() / h)
    corr = np.stack(corr_all)  # (pages, L, 8)
    band = np.stack(band_all)
    return {
        "reading_order_corr_by_layer": np.round(np.nanmean(corr, axis=(0, 2)), 3).tolist(),
        "reading_order_corr_best_per_page": np.round(np.nanmax(corr.reshape(len(corr), -1), 1), 3).tolist(),
        "band_mass_by_layer": np.round(band.mean(axis=(0, 2)), 3).tolist(),
        "band_mass_max_group_by_layer": np.round(band.mean(0).max(1), 3).tolist(),
        "band_mass_uniform": round(float(np.mean(band_uniform)), 4),
    }


def print_tables(r: dict) -> None:
    print("\nOracle (mean over pages, steps, layers, groups):")
    print("| tau | skip | dropped mass mean | p99 | head err mean | p99 | layer err mean | p99 | values-only |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for o in r["oracle"]:
        print(
            f"| {o['tau']:.0e} | {o['skip_mean']:.3f} | {o['dropped_mass_mean']:.2e} | {o['dropped_mass_p99']:.2e} "
            f"| {o['head_err_mean']:.2e} | {o['head_err_p99']:.2e} | {o['layer_err_mean']:.2e} "
            f"| {o['layer_err_p99']:.2e} | {o['values_layer_err_mean']:.2e} |"
        )
    q = r["q8"]
    print(
        f"| Q8 | - | - | - | {q['head_err_mean']:.2e} | {q['head_err_p99']:.2e} | {q['layer_err_mean']:.2e} "
        f"| {q['layer_err_p99']:.2e} | - |"
    )
    print("\nBounds at tau 1e-3 and 1e-2:")
    print("| bound | tau | skip | oracle skip | summary share | image bytes saved | layer err mean |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for b in r["bounds_table"]:
        if b["tau"] in (1e-3, 1e-2):
            print(
                f"| {b['bound']} | {b['tau']:.0e} | {b['skip_mean']:.3f} | {b['oracle_skip_mean']:.3f} "
                f"| {b['summary_bytes_share']:.3f} | {b['image_bytes_saved']:.3f} | {b['layer_err_mean']:.2e} |"
            )
    for tau, totals in r["kv_bytes_saved_share"].items():
        print(f"\nWhole-cache bytes saved at tau {tau}:", json.dumps({k: round(v, 4) for k, v in totals.items()}))
    print("\nTeacher-forced:")
    print("| policy | steps | flips | KL mean | KL p99 | skip |")
    print("|---|---:|---:|---:|---:|---:|")
    for t in r["teacher_forced"]:
        skip = f"{t['skip_mean']:.3f}" if "skip_mean" in t else "-"
        print(f"| {t['policy']} | {t['steps']} | {t['flips']} | {t['kl_mean']:.2e} | {t['kl_p99']:.2e} | {skip} |")
    print("\nBy layer (tau 1e-3 unless noted):")
    print("| layer | sink | other | image | oracle | 1e-4 | groups | split | sketch4 | cover90 | Q8 err | oracle err |")
    print("|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for x in r["by_layer"]:
        image = 1.0 - x["sink_mass"] - x["other_mass"]
        groups = f"{x['oracle_skip_1e-3_group_min']:.2f}-{x['oracle_skip_1e-3_group_max']:.2f}"
        print(
            f"| {x['layer']} | {x['sink_mass']:.3f} | {x['other_mass']:.3f} | {image:.3f} "
            f"| {x['oracle_skip_1e-3']:.3f} | {x['oracle_skip_1e-4']:.3f} | {groups} "
            f"| {x['split_skip_1e-3']:.4f} | {x['sketch4_skip_1e-3']:.4f} | {x['cover90']:.0f} "
            f"| {x['q8_err_layer']:.2e} | {x['oracle_err_layer_1e-3']:.2e} |"
        )
    print("\nBy step:")
    for x in r["by_step"]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    print("\nBy page:")
    print("| page | input | blocks | steps | CPU prefix | oracle 1e-3 | 1e-4 | sketch4 1e-3 | Q8 flips | o1e-4 flips |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in r["pages"]:
        cpu = p.get("cpu_exact_reference", {}).get("common_prefix", "-")
        pol = p["policies"]
        q8_flips = pol["q8"]["flips"] if "q8" in pol else "-"
        oracle_flips = pol["oracle:1e-4"]["flips"] if "oracle:1e-4" in pol else "-"
        print(
            f"| {p['page_id']} | {p['input_tokens']} | {p['image_blocks']} | {p['decode_steps']} | {cpu} "
            f"| {p['oracle_skip'][2]:.3f} | {p['oracle_skip'][4]:.3f} | {p['bound_skip_1e-3']['sketch4']:.4f} "
            f"| {q8_flips} | {oracle_flips} |"
        )
    print("\nLocality:", json.dumps(r["locality"]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="GPU: reference, statistics and policy passes per page")
    r.add_argument("pages", nargs="+", type=Path, help="canonical RGB page images (corpus/<id>/canonical-rgb.png)")
    r.add_argument("--model", type=Path, required=True, help="the FP32 checkpoint directory with its Python sources")
    r.add_argument("--output", type=Path, required=True, help="directory for <page id>.json and .npz (no overwrite)")
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--max-dimension", type=int, default=1536)
    r.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    r.add_argument(
        "--policies",
        type=lambda text: [spec for spec in text.split(",") if spec],
        default=list(DEFAULT_POLICIES),
        help="comma-separated teacher-forced policies (default: %(default)s)",
    )
    s = sub.add_parser("summarize", help="tables over the pages of one or more run directories")
    s.add_argument("dirs", nargs="+", type=Path)
    s.add_argument("--output", type=Path, required=True, help="new summary JSON")
    s.add_argument("--cpu-reference", type=Path, help="a falcon-ocr-eval FP32 report to compare tokens with")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
