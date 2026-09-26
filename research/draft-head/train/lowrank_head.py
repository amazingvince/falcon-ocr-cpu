#!/usr/bin/env python3
"""Replace an exported draft head's vocabulary projection with a rank-r
factorization (truncated SVD), for a cheaper draft step on CPU.

`head.weight` (V x 768) becomes `head_down.weight` (r x 768) and
`head_up.weight` (V x r), each sqrt(S)-balanced so both quantize well. The
Rust runner computes `up(down(x))`, reading 768 r + V r weights instead of
768 V. Writes `<out>.safetensors` and a parity fixture (the chain the PyTorch
head drafts with the reconstructed head, same inputs as export_head.py).

  python research/draft-head/train/lowrank_head.py --head /mnt/d/falcon-draft/heads/text-twophase.safetensors \
      --rank 256 --out /mnt/d/falcon-draft/heads/text-twophase-r256
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "draft-head" / "train"))
from eagle3 import DraftHead  # noqa: E402
from export_head import draft_chain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=pathlib.Path, required=True, help="an export_head.py .safetensors")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    assert a.rank % 64 == 0, "rank must be a multiple of 64 (the int8 group)"
    tensors = load_file(str(a.head))
    with safe_open(str(a.head), "pt") as f:
        meta = dict(f.metadata())
    w = tensors.pop("head.weight").double()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    energy = (s[:a.rank] ** 2).sum() / (s ** 2).sum()
    root = s[:a.rank].sqrt()
    down = (root[:, None] * vh[:a.rank]).float().contiguous()
    up = (u[:, :a.rank] * root[None]).float().contiguous()
    tensors["head_down.weight"], tensors["head_up.weight"] = down, up
    meta["head_rank"] = str(a.rank)
    meta["source"] = str(a.head)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(a.out.with_suffix(".safetensors")), metadata=meta)

    # Parity fixture with the reconstructed head (same inputs as the source's).
    fixture = json.loads(a.head.with_suffix(".fixture.json").read_text())
    target = load_file(str(ROOT / "artifacts" / "model" / "model.safetensors"))
    vocab = tensors["vocab_ids"].long()
    model = DraftHead(vocab, target["tok_embeddings.weight"].float(), target["output.weight"].float(), image=False)
    state = {k: v for k, v in tensors.items() if k not in ("head_down.weight", "head_up.weight", "vocab_ids")}
    state["head.weight"] = up @ down
    model.load_state_dict(state, strict=False)
    model.eval()
    feats = torch.tensor(fixture["features"]).view(len(fixture["tokens"]), -1)
    with torch.no_grad():
        chain = draft_chain(model, feats, fixture["tokens"], steps=4)
    out_fixture = {"features": fixture["features"], "tokens": fixture["tokens"], "chain": chain}
    a.out.with_suffix(".fixture.json").write_text(json.dumps(out_fixture))
    ref = [c["token"] for c in fixture["chain"]]
    print(f"rank {a.rank}: {energy:.4f} of the squared singular mass; head weights {w.numel() / 1e6:.1f}M -> "
          f"{(down.numel() + up.numel()) / 1e6:.1f}M; fixture chain {[c['token'] for c in chain]} (full head {ref}) "
          f"confidences {[round(c['confidence'], 3) for c in chain]}")


if __name__ == "__main__":
    main()
