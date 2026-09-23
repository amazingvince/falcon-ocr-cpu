#!/usr/bin/env python3
"""Activation-weighted output error of W8 overlays (offline, uses captured Grams).

  python attempt3/w8_output_error.py --gram-dir artifacts/phase4/gram OVERLAY [OVERLAY ...] [--top 10]

For each matrix: relative output error sqrt(tr(dW G dWᵀ) / tr(W G Wᵀ)), where
G = E[x xᵀ] is the captured mean Gram of the matrix input and dW the
dequantized-minus-source weights. Matrices absent from a partial overlay count
as exact. Prints the mean over matrices and the worst matrices.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("overlays", nargs="+", type=Path)
    ap.add_argument("--gram-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--top", type=int, default=0)
    a = ap.parse_args()
    grams = {}
    with safe_open(a.model / "model.safetensors", framework="numpy") as model:
        for overlay in a.overlays:
            tensors = load_file(overlay)
            group = int(safe_open(overlay, framework="numpy").metadata()["group_size"])
            errors = []
            for path in sorted(a.gram_dir.glob("*.gram.npy")):
                name = path.name[: -len(".gram.npy")]
                if name + ".__w8_codes" not in tensors:
                    errors.append((0.0, name))
                    continue
                if name not in grams:
                    grams[name] = np.load(path).astype(np.float64)
                g = grams[name]
                w = model.get_tensor(name).astype(np.float64)
                codes = tensors[name + ".__w8_codes"].astype(np.float32)
                scales = tensors[name + ".__w8_scales"]
                deq = (codes * np.repeat(scales, group, axis=1)[:, : w.shape[1]]).astype(np.float64)
                dw = deq - w
                num = float(((dw @ g) * dw).sum())
                den = float(((w @ g) * w).sum())
                errors.append((float(np.sqrt(max(num, 0.0) / den)), name))
            values = [e for e, _ in errors]
            print(f"{overlay.name}: mean {np.mean(values):.3e} max {max(values):.3e} "
                  f"(matrices {sum(v > 0 for v in values)})")
            for e, n in sorted(errors, reverse=True)[: a.top]:
                print(f"   {e:.3e} {n}")


if __name__ == "__main__":
    main()
