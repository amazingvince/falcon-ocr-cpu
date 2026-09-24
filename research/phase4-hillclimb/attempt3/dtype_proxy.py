#!/usr/bin/env python3
"""Activation-weighted output error of candidate storage types for every body matrix.

  python research/phase4-hillclimb/attempt3/dtype_proxy.py --gram-dir artifacts/phase4/gram [--overlay NAME=W8.safetensors ...] [--out R.json]

For each matrix and type: relative output error sqrt(tr(dW G dWᵀ) / tr(W G Wᵀ)),
with G the captured input Gram (an "importance matrix") and dW the stored
minus the FP32 weights. Round-to-nearest types: bf16, fp16, int16/int8/int4
with one absmax scale per 64 (or 32) input weights. `--overlay` adds existing
W8 overlays (for example GPTQ). Prints mean error and bytes per weight per
type, overall and by matrix kind.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file


def round_bf16(w: np.ndarray) -> np.ndarray:
    bits = w.astype(np.float32).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32)


def group_int(w: np.ndarray, bits: int, group: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group)
    scale = np.abs(g).max(axis=2, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    return (np.clip(np.rint(g / scale), -qmax, qmax) * scale).reshape(rows, cols)


TYPES = {
    "bf16": (2.0, round_bf16),
    "fp16": (2.0, lambda w: w.astype(np.float16).astype(np.float32)),
    "int16-g64": (2 + 4 / 64, lambda w: group_int(w, 16, 64)),
    "int8-g64": (1 + 4 / 64, lambda w: group_int(w, 8, 64)),
    "int4-g64": (0.5 + 4 / 64, lambda w: group_int(w, 4, 64)),
    "int4-g32": (0.5 + 4 / 32, lambda w: group_int(w, 4, 32)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gram-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    overlays = {}
    for spec in a.overlay:
        name, path = spec.split("=", 1)
        overlays[name] = (load_file(path), int(safe_open(path, framework="numpy").metadata()["group_size"]))
    kinds = list(TYPES) + list(overlays)
    result = {}
    with safe_open(a.model / "model.safetensors", framework="numpy") as model:
        for path in sorted(a.gram_dir.glob("*.gram.npy")):
            name = path.name[: -len(".gram.npy")]
            g = np.load(path).astype(np.float64)
            w32 = model.get_tensor(name).astype(np.float32)
            w = w32.astype(np.float64)
            den = float(((w @ g) * w).sum())
            row = {"params": int(w.size)}
            for kind in kinds:
                if kind in TYPES:
                    q = TYPES[kind][1](w32).astype(np.float64)
                else:
                    tensors, group = overlays[kind]
                    codes = tensors[name + ".__w8_codes"].astype(np.float32)
                    scales = tensors[name + ".__w8_scales"]
                    q = (codes * np.repeat(scales, group, axis=1)[:, : w.shape[1]]).astype(np.float64)
                dw = q - w
                row[kind] = float(np.sqrt(max(float(((dw @ g) * dw).sum()), 0.0) / den))
            result[name] = row
    by_kind = defaultdict(lambda: defaultdict(list))
    for name, row in result.items():
        site = name.split(".")[-2]
        for kind in kinds:
            by_kind[kind]["all"].append(row[kind])
            by_kind[kind][site].append(row[kind])
    sites = ["wqkv", "wo", "w13", "w2"]
    print(f"{'type':>14} {'bytes/wt':>8} {'mean':>10} {'max':>10}  " + " ".join(f"{s:>9}" for s in sites))
    for kind in kinds:
        b = TYPES[kind][0] if kind in TYPES else 1 + 4 / overlays[kind][1]
        v = by_kind[kind]
        print(f"{kind:>14} {b:>8.3f} {np.mean(v['all']):>10.3e} {max(v['all']):>10.3e}  "
              + " ".join(f"{np.mean(v[s]):>9.2e}" for s in sites))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
