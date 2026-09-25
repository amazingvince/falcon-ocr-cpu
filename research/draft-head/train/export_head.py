#!/usr/bin/env python3
"""Export a trained draft head for the Rust runner, plus a parity fixture.

Writes `<out>.safetensors` (FP32 weights of the text-only draft block, the
draft vocabulary and metadata) and `<out>.fixture.json`: a draft chain the
PyTorch head computes from fixed inputs (target features for a few verified
positions, their tokens), which the Rust `draft_head` must reproduce.

  python research/draft-head/train/export_head.py --checkpoint /mnt/d/falcon-draft/runs/text-twophase/draft.pt \
      --out /mnt/d/falcon-draft/heads/text-twophase
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "draft-head" / "train"))
from eagle3 import DIM, FFN, HEADS, HEAD_DIM, KV_HEADS, LAYERS, DraftHead, rope  # noqa: E402

FORMAT = "falcon-ocr-draft-head-v1"


def draft_chain(model: DraftHead, feats: torch.Tensor, tokens: list[int], steps: int):
    """Inference drafting from the last of `len(tokens)` verified positions:
    base keys for every position, then `steps` drafts, each fed the previous
    draft (argmax). Returns per step: drafted id, confidence, top-5 logits."""
    n = len(tokens)
    tok = torch.tensor(tokens)
    hidden = model.fuse(feats)
    q, k, v, _ = model.step(hidden, tok, torch.arange(n), None, None, None, None)
    keys, values = [k], [v]
    h, token, out = hidden[-1:], tok[-1:], []
    for j in range(1, steps + 1):
        pos = torch.tensor([n - 1 + j - 1])
        qj, kj, vj, _ = model.step(h, token, pos, None, None, None, None)
        if j > 1:
            keys.append(kj)
            values.append(vj)
        k_all, v_all = torch.cat(keys), torch.cat(values)
        mask = torch.ones(1, k_all.shape[0], dtype=torch.bool)
        att = model.attend(qj, k_all, v_all, mask)
        h, logits = model.finish(h, att, None)
        probs = logits[0].softmax(-1)
        best = int(logits[0].argmax())
        top = torch.topk(logits[0], 5)
        out.append({"token": int(model.vocab_ids[best]), "confidence": float(probs[best]),
                    "top_ids": [int(model.vocab_ids[i]) for i in top.indices],
                    "top_logits": [float(x) for x in top.values]})
        token = model.vocab_ids[best:best + 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state"]
    image = not ckpt["args"].get("no_image", False)
    if image:
        sys.exit("only text-only heads (--no-image) are exported for the Rust runner")
    target = load_file(str(ROOT / "artifacts" / "model" / "model.safetensors"))
    embedding = target["tok_embeddings.weight"].float()
    model = DraftHead(state["vocab_ids"], embedding, target["output.weight"].float(), image=False)
    factored = "head.0.weight" in state
    if factored:
        model.factorize_head(state["head.0.weight"].shape[0])
    model.load_state_dict(state)
    model.eval()
    tensors = {name: state[name].float().contiguous() for name in
               ["fc.weight", "qkv.weight", "o.weight", "w13.weight", "w2.weight", "norm.weight"]}
    if factored:
        tensors["head_down.weight"] = state["head.0.weight"].float().contiguous()
        tensors["head_up.weight"] = state["head.1.weight"].float().contiguous()
    else:
        tensors["head.weight"] = state["head.weight"].float().contiguous()
    tensors["vocab_ids"] = state["vocab_ids"].to(torch.int32).contiguous()
    meta = {"format": FORMAT, "layers": ",".join(map(str, LAYERS)), "dim": str(DIM), "heads": str(HEADS),
            "kv_heads": str(KV_HEADS), "head_dim": str(HEAD_DIM), "ffn": str(FFN), "rope_theta": "10000",
            "norm_eps": "1e-5", "draft_vocab": str(len(state["vocab_ids"])), "image": "false",
            "checkpoint": str(a.checkpoint), "steps_trained": str(ckpt["args"].get("steps"))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(a.out.with_suffix(".safetensors")), metadata=meta)
    # Parity fixture: realistic feature scales (per-layer RMS about 6.5 / 17 / 56).
    gen = torch.Generator().manual_seed(0)
    n = 6
    scales = torch.tensor([6.5, 17.0, 56.0]).repeat_interleave(DIM)
    feats = torch.randn(n, 3 * DIM, generator=gen) * scales
    tokens = [int(model.vocab_ids[i]) for i in torch.randint(0, len(model.vocab_ids), (n,), generator=gen)]
    with torch.no_grad():
        chain = draft_chain(model, feats, tokens, steps=4)
    fixture = {"features": feats.flatten().tolist(), "tokens": tokens, "chain": chain}
    a.out.with_suffix(".fixture.json").write_text(json.dumps(fixture))
    print(f"wrote {a.out.with_suffix('.safetensors')} ({sum(t.numel() for t in tensors.values()) / 1e6:.1f}M values) "
          f"and the fixture: chain {[c['token'] for c in chain]} confidences {[round(c['confidence'], 3) for c in chain]}")


if __name__ == "__main__":
    main()
