#!/usr/bin/env python3
"""W8 overlay variants for the fidelity hill climb (same file format as convert_w8.py).

  python attempt3/w8_variants.py --output X.safetensors [--group 32|64] [--clip rtn|mse|wmse]
      [--include REGEX] [--exclude REGEX] [--gram-dir DIR] [--method rtn|gptq]

Every variant stores int8 codes in [-127, 127] and one FP32 scale per group;
the runner dequantizes fl(code * scale), so kernels are unchanged.
* --clip mse   per group, the scale factor in [0.80, 1.00] that minimizes the
               squared reconstruction error (plain absmax is factor 1.00).
* --clip wmse  the same, weighted by each input channel's mean squared
               activation (diagonal of the captured Gram, --gram-dir).
* --method gptq  GPTQ error compensation with the captured Gram (--gram-dir),
               per-group scales chosen as for --clip (rtn/mse) before each group.
* --include/--exclude select matrices; others are left out of the overlay and
               stay FP32 (the overlay is then marked partial).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_w8 import CONFIG_SHA256, FORMAT, REVISION, SOURCE_SHA256, inventory, sha256  # noqa: E402

FACTORS = np.linspace(0.80, 1.00, 41, dtype=np.float64)
SMALLEST = np.nextafter(np.float32(0), np.float32(1))


def absmax_scale(block: np.ndarray, factor: np.ndarray | float = 1.0) -> np.ndarray:
    """Scales for [..., G] blocks; zero rows keep scale 0."""
    maxima = np.max(np.abs(block), axis=-1).astype(np.float64)
    s = (maxima * factor / 127.0).astype(np.float32)
    return np.where(maxima == 0, np.float32(0), np.maximum(s, SMALLEST)).astype(np.float32)


def codes_for(block: np.ndarray, s: np.ndarray) -> np.ndarray:
    divisor = np.where(s == 0, np.float32(1), s).astype(np.float64)
    q = np.clip(np.rint(block.astype(np.float64) / divisor[..., None]), -127, 127).astype(np.int8)
    q[s == 0] = 0
    return q


def best_scales(block: np.ndarray, weight: np.ndarray | None, clip: str) -> np.ndarray:
    """block: [rows, G] float32; weight: [G] channel weights (wmse) or None."""
    if clip == "rtn":
        return absmax_scale(block)
    best = absmax_scale(block)
    best_err = np.full(block.shape[0], np.inf)
    w = np.ones(block.shape[1]) if weight is None else weight
    for f in FACTORS:
        s = absmax_scale(block, f)
        q = codes_for(block, s)
        err = (((q.astype(np.float32) * s[:, None]) - block).astype(np.float64) ** 2 * w).sum(axis=1)
        better = err < best_err
        best = np.where(better, s, best)
        best_err = np.where(better, err, best_err)
    return best


def quantize_rtn(w: np.ndarray, group: int, clip: str, diag: np.ndarray | None):
    n, k = w.shape
    groups = (k + group - 1) // group
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, groups), dtype=np.float32)
    for g in range(groups):
        a, b = g * group, min(k, (g + 1) * group)
        block = w[:, a:b]
        s = best_scales(block, None if diag is None else diag[a:b], clip)
        codes[:, a:b] = codes_for(block, s)
        scales[:, g] = s
    return codes, scales


def quantize_gptq(w: np.ndarray, gram: np.ndarray, group: int, clip: str, damp: float = 0.01,
                  act_order: bool = False):
    """GPTQ (Frantar et al.) with per-group scales, in float64.

    Columns are processed in order; each column's rounding error is spread over
    the remaining columns with the inverse-Hessian Cholesky factor. Without
    act_order, the scale of a group is fixed from the error-updated weights
    when the group starts. With act_order, columns are processed by
    descending activation energy and every group scale is fixed up front from
    the original weights (static groups), so the stored layout is unchanged.
    """
    if act_order:
        return quantize_gptq_act_order(w, gram, group, clip, damp)
    n, k = w.shape
    h = gram.astype(np.float64).copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    work = w.astype(np.float64).copy()
    work[:, dead] = 0.0
    h += np.eye(k) * damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T  # upper factor
    groups = (k + group - 1) // group
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, groups), dtype=np.float32)
    diag = np.diag(gram).astype(np.float64)
    for g in range(groups):
        a, b = g * group, min(k, (g + 1) * group)
        block32 = work[:, a:b].astype(np.float32)
        s = best_scales(block32, diag[a:b] if clip == "wmse" else None, "rtn" if clip == "rtn" else clip)
        scales[:, g] = s
        divisor = np.where(s == 0, 1.0, s.astype(np.float64))
        for j in range(a, b):
            col = work[:, j]
            q = np.clip(np.rint(col / divisor), -127, 127)
            q[s == 0] = 0
            codes[:, j] = q.astype(np.int8)
            deq = (q.astype(np.float32) * s).astype(np.float64)
            err = (col - deq) / hinv[j, j]
            work[:, j + 1:] -= np.outer(err, hinv[j, j + 1:])
    return codes, scales


def quantize_gptq_act_order(w: np.ndarray, gram: np.ndarray, group: int, clip: str, damp: float):
    n, k = w.shape
    diag = np.diag(gram).astype(np.float64)
    groups = (k + group - 1) // group
    scales = np.zeros((n, groups), dtype=np.float32)
    for g in range(groups):
        a, b = g * group, min(k, (g + 1) * group)
        scales[:, g] = best_scales(w[:, a:b].astype(np.float32),
                                   diag[a:b] if clip == "wmse" else None,
                                   "rtn" if clip == "rtn" else clip)
    column_scale = np.repeat(scales, group, axis=1)[:, :k].astype(np.float64)
    perm = np.argsort(-diag, kind="stable")
    h = gram.astype(np.float64)[np.ix_(perm, perm)].copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    work = w.astype(np.float64)[:, perm].copy()
    work[:, dead] = 0.0
    h += np.eye(k) * damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T
    s_perm = column_scale[:, perm]
    q_perm = np.zeros((n, k), dtype=np.int8)
    for j in range(k):
        s = s_perm[:, j]
        col = work[:, j]
        q = np.clip(np.rint(col / np.where(s == 0, 1.0, s)), -127, 127)
        q[s == 0] = 0
        q_perm[:, j] = q.astype(np.int8)
        deq = (q.astype(np.float32) * s.astype(np.float32)).astype(np.float64)
        err = (col - deq) / hinv[j, j]
        work[:, j + 1:] -= np.outer(err, hinv[j, j + 1:])
    codes = np.zeros((n, k), dtype=np.int8)
    codes[:, perm] = q_perm
    return codes, scales


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--group", type=int, choices=(32, 64), default=64)
    ap.add_argument("--clip", choices=("rtn", "mse", "wmse"), default="rtn")
    ap.add_argument("--method", choices=("rtn", "gptq"), default="rtn")
    ap.add_argument("--gram-dir", type=Path)
    ap.add_argument("--act-order", action="store_true", help="GPTQ: descending activation order, static groups")
    ap.add_argument("--damp", type=float, default=0.01, help="GPTQ: relative Hessian dampening")
    ap.add_argument("--source-bf16", action="store_true",
                    help="quantize the BF16-rounded weights (production serves BF16) instead of FP32")
    ap.add_argument("--include", default=".*")
    ap.add_argument("--exclude", default="$^")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output exists")
    if (args.clip == "wmse" or args.method == "gptq") and not args.gram_dir:
        ap.error("--clip wmse and --method gptq need --gram-dir")
    source = args.model / "model.safetensors"
    if sha256(source) != SOURCE_SHA256 or sha256(args.model / "config.json") != CONFIG_SHA256:
        ap.error("pinned source/config SHA-256 mismatch")
    include, exclude = re.compile(args.include), re.compile(args.exclude)
    names = [n for n in inventory(False) if include.search(n) and not exclude.search(n)]
    tensors = {}
    with safe_open(source, framework="numpy") as model:
        for name in names:
            w = model.get_tensor(name)
            if args.source_bf16:
                w = (w.view(np.uint32) + 0x7FFF + ((w.view(np.uint32) >> 16) & 1) & 0xFFFF0000).view(np.float32)
            gram = None
            if args.gram_dir:
                gram = np.load(args.gram_dir / f"{name}.gram.npy")
            if args.method == "gptq":
                codes, scales = quantize_gptq(w, gram, args.group, args.clip, args.damp, args.act_order)
            else:
                diag = None if gram is None else np.diag(gram).astype(np.float64)
                codes, scales = quantize_rtn(w, args.group, args.clip, diag)
            deq = codes.astype(np.float32) * np.repeat(scales, args.group, axis=1)[:, :w.shape[1]]
            rel = float(np.linalg.norm(deq - w) / np.linalg.norm(w))
            print(f"{name}: relative error {rel:.3e}", flush=True)
            tensors[name + ".__w8_codes"] = codes
            tensors[name + ".__w8_scales"] = scales
    partial = len(names) < len(inventory(False))
    metadata = {"format": FORMAT, "source_sha256": SOURCE_SHA256, "model_revision": REVISION,
                "include_head": "false", "group_size": str(args.group), "scale_dtype": "f32",
                "rounding": f"{args.method}-{args.clip}" + ("-actorder" if args.act_order else "")
                + ("-bf16source" if args.source_bf16 else ""),
                "activation_dtype": "f32",
                "partial": str(partial).lower()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    print(json.dumps({**metadata, "matrices": len(names), "output": str(args.output),
                      "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
