#!/usr/bin/env python3
"""GPU harness for quantization fidelity: reference generation and fake-quant scoring.

Runs the pinned upstream model (artifacts/model) with PyTorch on CUDA, in the
reference environment (WSL: ~/falcon-ocr-rust-reference/.venv).

* `reference`: free-running greedy generation at `--precision bf16|fp32`
  (production serves BF16). Stores, per page, the tokens and the top-K next
  token log-probabilities of every step.
* `score`: teacher-forces the reference tokens through a candidate in ONE
  forward pass per page (prompt + reference tokens, text positions exactly as
  decode assigns them) and reports KL(reference || candidate) per step and
  greedy flips. Candidates: FP32 or BF16 weights, optionally with a W8
  overlay dequantized into the FP32 weights ("fake quant": the CPU W8 kernels
  are bitwise equal to FP32 math on the dequantized weights), with chosen
  matrices kept FP32 (`--keep-fp32`, `*` wildcards).
* `sweep`: `score` for many `--keep-fp32` sets with the model loaded once.

  python attempt3/gpu_harness.py reference --precision bf16 --pages P.txt --out DIR
  python attempt3/gpu_harness.py score --reference DIR --overlay W.safetensors --report R.json
"""
from __future__ import annotations

import argparse
import fnmatch
import functools
import importlib
import json
import math
import pathlib
import sys
import time
import types

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

PROMPT = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
TOP_K = 256
BODY = ("attention.wqkv", "attention.wo", "feed_forward.w13", "feed_forward.w2")


def import_model(model_path: pathlib.Path):
    package = types.ModuleType("pinned_falcon_ocr")
    package.__path__ = [str(model_path)]
    sys.modules[package.__name__] = package
    return importlib.import_module("pinned_falcon_ocr.modeling_falcon_ocr")


class Harness:
    def __init__(self, model_dir: pathlib.Path, precision: str, device: str, weights_bf16: bool = False):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.device = device
        self.module = import_model(model_dir.resolve())
        for name in ["compiled_flex_attn_prefill", "compiled_flex_attn_decode"]:
            setattr(self.module, name, functools.partial(getattr(self.module, name),
                                                         kernel_options={"FLOAT32_PRECISION": "'ieee'"}))
        config = self.module.FalconOCRConfig.from_json_file(str(model_dir / "config.json"))
        config._name_or_path = str(model_dir.resolve())
        self.config = config
        model = self.module.FalconOCRForCausalLM(config)
        self.source = load_file(str(model_dir / "model.safetensors"))
        if weights_bf16:
            # Production stores BF16 weights: FP32 math on BF16-rounded parameters.
            self.source = {k: v.to(torch.bfloat16).to(v.dtype) for k, v in self.source.items()}
        model.load_state_dict(self.source, strict=True, assign=True)
        self.dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.model._ensure_device_buffers()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()), local_files_only=True,
                                                       trust_remote_code=True)
        self.params = dict(self.model.named_parameters())
        self.stop_ids = [config.eos_id, self.tokenizer.convert_tokens_to_ids("<|end_of_query|>")]

    # ----- inputs
    def prepare(self, image_path: str, max_dimension: int = 1536, min_dimension: int = 64):
        image = Image.open(image_path).convert("RGB")
        batch = self.module.process_batch(self.tokenizer, self.config, [(image, PROMPT)],
                                          max_length=self.config.max_seq_len,
                                          min_dimension=min_dimension, max_dimension=max_dimension)
        self.model._pad_token_id = batch["pad_token_id"]
        return batch

    def _forward_prefix(self, batch, tokens, pos_t, pos_hw):
        """One prefill-style forward over `tokens` (1, S); returns logits (S, V)."""
        s = tokens.shape[1]
        capacity = ((s + 127) // 128) * 128
        padded = torch.full((1, capacity), self.model._pad_token_id, device=self.device, dtype=torch.long)
        padded[:, :s] = tokens
        mask = self.model.get_attention_mask(padded, max_len=capacity)
        cache = self.module.KVCache(1, capacity, self.config.n_heads, self.config.head_dim, self.config.n_layers)
        logits = self.model(tokens=tokens, attention_mask=mask, kv_cache=cache, rope_pos_t=pos_t, rope_pos_hw=pos_hw,
                            pixel_values=batch["pixel_values"].to(self.device),
                            pixel_mask=batch["pixel_mask"].to(self.device))
        return logits[0], mask, cache, capacity, padded

    # ----- free-running reference
    @torch.inference_mode()
    def generate(self, batch, max_new_tokens: int):
        tokens = batch["tokens"].to(self.device)
        prefix = tokens.shape[1]
        capacity = ((prefix + max_new_tokens + 127) // 128) * 128
        padded = torch.full((1, capacity), self.model._pad_token_id, device=self.device, dtype=torch.long)
        padded[:, :prefix] = tokens
        mask = self.model.get_attention_mask(padded, max_len=capacity)
        cache = self.module.KVCache(1, capacity, self.config.n_heads, self.config.head_dim, self.config.n_layers)
        logits = self.model(tokens=tokens, attention_mask=mask, kv_cache=cache,
                            rope_pos_t=batch["pos_t"].to(self.device), rope_pos_hw=batch["pos_hw"].to(self.device),
                            pixel_values=batch["pixel_values"].to(self.device),
                            pixel_mask=batch["pixel_mask"].to(self.device))[0, -1]
        chosen, tops = [], []
        for step in range(max_new_tokens):
            logp = torch.log_softmax(logits.float(), -1)
            values, ids = torch.topk(logp, TOP_K)
            tops.append((ids.to(torch.int32).cpu(), values.cpu()))
            token = int(logits.argmax())
            chosen.append(token)
            if token in self.stop_ids or step + 1 == max_new_tokens:
                break
            padded[:, cache.get_pos()] = token
            logits = self.model(tokens=torch.tensor([[token]], device=self.device), attention_mask=mask,
                                kv_cache=cache)[0, -1]
        return chosen, tops

    # ----- teacher-forced single pass
    @torch.inference_mode()
    def teacher_logprobs(self, batch, forced):
        """log-softmax of the next-token logits for every step of `forced`."""
        tokens = batch["tokens"]
        pos_t, pos_hw = batch["pos_t"], batch["pos_hw"]
        extra = torch.tensor([forced[:-1]], dtype=tokens.dtype)
        n = extra.shape[1]
        last = int(pos_t[0, -1])
        tokens_all = torch.cat([tokens, extra], 1).to(self.device)
        pos_t_all = torch.cat([pos_t, torch.arange(last + 1, last + 1 + n, dtype=pos_t.dtype)[None]], 1).to(self.device)
        pos_hw_all = torch.cat([pos_hw, torch.full((1, n, pos_hw.shape[2]), float("nan"), dtype=pos_hw.dtype)],
                               1).to(self.device)
        logits = self._forward_prefix(batch, tokens_all, pos_t_all, pos_hw_all)[0]
        start = tokens.shape[1] - 1
        return torch.log_softmax(logits[start:start + len(forced)].float(), -1)

    # ----- weights
    def set_weights(self, overlay: dict | None, keep_fp32: list[str]):
        """Body matrices from `overlay` (name -> dequantized FP32 tensor) unless kept FP32."""
        for name, param in self.params.items():
            if not name.endswith(".weight") or not any(b in name for b in BODY):
                continue
            use = overlay is not None and name in overlay and not any(fnmatch.fnmatchcase(name, p) for p in keep_fp32)
            source = overlay[name] if use else self.source[name]
            param.data.copy_(source.to(device=self.device, dtype=self.dtype))


def load_overlay(path: pathlib.Path) -> dict:
    out = {}
    with safe_open(str(path), framework="pt") as f:
        group = int(f.metadata()["group_size"])
        names = {k.rsplit(".__w8_", 1)[0] for k in f.keys()}
        for name in names:
            codes = f.get_tensor(name + ".__w8_codes").float()
            scales = f.get_tensor(name + ".__w8_scales").float()
            out[name] = codes * scales.repeat_interleave(group, dim=1)[:, : codes.shape[1]]
    return out


def kl_topk(ref_ids, ref_logp, cand_logp):
    """KL(P || Q) per step; P given by top-K (ids, log p), all other tokens pooled."""
    p = ref_logp.exp()
    q_top = cand_logp.gather(1, ref_ids.long())
    kl = (p * (ref_logp - q_top)).sum(1)
    p_tail = (1 - p.sum(1)).clamp_min(0)
    q_tail = (1 - q_top.exp().sum(1)).clamp_min(1e-30)
    tail = torch.where(p_tail > 0, p_tail * (p_tail.clamp_min(1e-30).log() - q_tail.log()), torch.zeros_like(p_tail))
    return (kl + tail).clamp_min(0)


def pages_of(path: pathlib.Path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def page_id(path: str) -> str:
    return pathlib.PurePosixPath(path.replace("\\", "/")).parent.name


def cmd_reference(a):
    h = Harness(a.model, a.precision, a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    for image in pages_of(a.pages):
        pid = page_id(image)
        target = a.out / f"{pid}.pt"
        if target.exists():
            continue
        t0 = time.time()
        batch = h.prepare(image, a.max_dimension)
        chosen, tops = h.generate(batch, a.max_new_tokens)
        torch.save({"page": pid, "image": image, "tokens": chosen,
                    "ids": torch.stack([t[0] for t in tops]), "logp": torch.stack([t[1] for t in tops]),
                    "finish": "eos" if chosen[-1] in h.stop_ids else "length",
                    "text": h.tokenizer.decode(chosen, skip_special_tokens=True),
                    "precision": a.precision}, target)
        print(f"{pid}: {len(chosen)} tokens in {time.time() - t0:.1f}s", flush=True)


def score_pages(h, refs, max_steps):
    rows, total_kl, total_steps, total_flips = [], 0.0, 0, 0
    for ref in refs:
        n = min(len(ref["tokens"]), max_steps)
        batch = h.prepare(ref["image"])
        logp = h.teacher_logprobs(batch, ref["tokens"][:n])
        kl = kl_topk(ref["ids"][:n].to(h.device), ref["logp"][:n].to(h.device), logp)
        flips = int((logp.argmax(1).cpu() != torch.tensor(ref["tokens"][:n])).sum())
        rows.append({"page": ref["page"], "steps": n, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                     "flips": flips})
        total_kl += float(kl.sum())
        total_steps += n
        total_flips += flips
    return {"steps": total_steps, "kl_mean": total_kl / max(1, total_steps), "flips": total_flips, "pages": rows}


def load_refs(directory: pathlib.Path, pages: pathlib.Path | None):
    wanted = None if pages is None else {page_id(p) for p in pages_of(pages)}
    refs = [torch.load(p) for p in sorted(directory.glob("*.pt"))]
    return [r for r in refs if wanted is None or r["page"] in wanted]


def cmd_score(a):
    h = Harness(a.model, a.precision, a.device, a.weights_bf16)
    overlay = load_overlay(a.overlay) if a.overlay else None
    h.set_weights(overlay, a.keep_fp32)
    t0 = time.time()
    result = score_pages(h, load_refs(a.reference, a.pages), a.max_steps)
    result.update({"precision": a.precision, "overlay": str(a.overlay), "keep_fp32": a.keep_fp32,
                   "reference": str(a.reference), "seconds": time.time() - t0})
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}), flush=True)
    if a.report:
        a.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def cmd_sweep(a):
    h = Harness(a.model, a.precision, a.device, a.weights_bf16)
    overlay = load_overlay(a.overlay) if a.overlay else None
    refs = load_refs(a.reference, a.pages)
    arms = json.loads(a.arms.read_text(encoding="utf-8"))  # {name: [patterns]}
    results = {}
    for name, keep in arms.items():
        h.set_weights(overlay, keep)
        t0 = time.time()
        r = score_pages(h, refs, a.max_steps)
        results[name] = {"keep_fp32": keep, "kl_mean": r["kl_mean"], "flips": r["flips"], "steps": r["steps"],
                         "pages": r["pages"], "seconds": time.time() - t0}
        print(f"{name:>24}: KL {r['kl_mean']:.4e} flips {r['flips']}/{r['steps']} ({time.time() - t0:.1f}s)",
              flush=True)
        if a.report:
            a.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["reference", "score", "sweep"]:
        s = sub.add_parser(name)
        s.add_argument("--model", type=pathlib.Path, default=pathlib.Path("artifacts/model"))
        s.add_argument("--precision", choices=["fp32", "bf16"], default="fp32" if name != "reference" else "bf16")
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--pages", type=pathlib.Path)
        s.add_argument("--max-dimension", type=int, default=1536)
    ref = sub.choices["reference"]
    ref.add_argument("--out", type=pathlib.Path, required=True)
    ref.add_argument("--max-new-tokens", type=int, default=4096)
    for name in ["score", "sweep"]:
        s = sub.choices[name]
        s.add_argument("--reference", type=pathlib.Path, required=True)
        s.add_argument("--overlay", type=pathlib.Path)
        s.add_argument("--max-steps", type=int, default=4096)
        s.add_argument("--report", type=pathlib.Path)
        s.add_argument("--weights-bf16", action="store_true",
                       help="round every parameter to BF16 first (FP32 math on production's BF16 weights)")
    sub.choices["score"].add_argument("--keep-fp32", nargs="*", default=[])
    sub.choices["sweep"].add_argument("--arms", type=pathlib.Path, required=True)
    a = ap.parse_args()
    {"reference": cmd_reference, "score": cmd_score, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
