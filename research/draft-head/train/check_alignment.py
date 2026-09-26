#!/usr/bin/env python3
"""Sanity check for the draft trainer's feature alignment.

For a few generated pages: the target's own head on its final layer at text
position i must predict y_i (the vLLM greedy token placed there). Also
reports how much y_{i+1} is predictable from the step-1 inputs by the
target itself (the head at position i+1), and feature magnitudes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "phase4-hillclimb" / "attempt3"))
sys.path.insert(0, str(ROOT / "research" / "draft-head" / "train"))
from eagle3 import Pages, load_records, keep  # noqa: E402
from gpu_harness import Harness  # noqa: E402


def main():
    harness = Harness(ROOT / "artifacts" / "model", "bf16", "cuda")
    final = {}
    harness.model.layers["21"].register_forward_hook(lambda m, i, o: final.__setitem__("h", o[0]))
    pages = Pages(harness)
    head, norm = pages.head_weight, harness.model.norm
    records = [r for r in load_records(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])) if keep(r)][:6]
    for r in records:
        y = r["token_ids"]
        text, image = pages.features(r["path"], y)
        batch = harness.prepare(r["path"])
        prompt = batch["tokens"].shape[1]
        harness._batches.clear()
        h = final["h"][prompt - 1:prompt - 1 + len(y)]
        with torch.no_grad():
            pred = (norm(h) @ head.T).argmax(-1).cpu()
        agree = (pred == torch.tensor(y)).float().mean().item()
        parts = text.view(len(y), 3, -1)
        print(f"{r['id']}: n={len(y)} head(final)[i]==y_i: {100 * agree:.1f}%  "
              f"feature rms per layer {[round(v, 1) for v in parts.pow(2).mean((0, 2)).sqrt().tolist()]}  "
              f"max |f| {parts.abs().amax().item():.0f}  image rows {image.shape[0]}")


if __name__ == "__main__":
    main()
