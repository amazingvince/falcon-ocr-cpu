#!/usr/bin/env python3
"""W8 overlay variants for the fidelity hill climb (same file format as convert_w8.py).

  python tools/w8_variants.py --output X.safetensors [--group 32|64] [--clip rtn|mse|wmse]
      [--include REGEX] [--exclude REGEX] [--gram-dir DIR] [--method rtn|gptq]
      [--exceptions N [--exceptions-include REGEX] [--exception-select energy|error]]

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
* --exceptions N  keeps N input columns of every matrix matching
               --exceptions-include (default all; 'feed_forward\\.w2', the
               squared-ReLU inputs with the most extreme channels, is the
               suggested start) unquantized in FP32 (experimental): the N largest
               activation energies diag(G), or with --exception-select error
               the largest diag(G)_j times the expected rounding error of
               column j. RTN computes scales and codes with those columns
               zeroed; GPTQ never quantizes them and processes them last, so
               they absorb every quantized column's error compensation. Codes
               there are 0; {name}.__w8_exc_cols (I32 [k]) and
               {name}.__w8_exc_vals (F32 [out, k]) hold the columns and their
               FP32 weights, in format v2 (older runners refuse it). Choose N with
               tools/w8_proxy.py first.
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
# FORMAT plus exception columns ({name}.__w8_exc_cols, {name}.__w8_exc_vals); written only when a
# matrix has some, so runners that predate them refuse the overlay instead of zeroing the columns.
FORMAT_EXCEPTIONS = "falcon-ocr-attempt3-w8g64-v2"
NO_EXCEPTIONS = np.zeros(0, dtype=np.int64)


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


def rounding_error_weights(w: np.ndarray, group: int) -> np.ndarray:
    """Per input column, the sum over rows of its RTN group scale squared: a weight's rounding error
    is about uniform over one step s (variance s^2 / 12), so this ranks the columns' expected error."""
    k = w.shape[1]
    weights = np.zeros(k, dtype=np.float64)
    for a in range(0, k, group):
        b = min(k, a + group)
        weights[a:b] = (absmax_scale(w[:, a:b]).astype(np.float64) ** 2).sum()
    return weights


def select_exceptions(w: np.ndarray, diag: np.ndarray, count: int, select: str, group: int) -> np.ndarray:
    """The `count` input columns to keep exact, ascending: the largest activation energy diag(G)
    ('energy'), or the largest diag(G)_j times the expected rounding error of column j ('error').
    A weight-only quantized product errs by about sum_j dw_j x_j, so a column whose input is huge
    multiplies its rounding error by |x_j|."""
    if count <= 0:
        return NO_EXCEPTIONS
    if count >= w.shape[1]:
        raise ValueError(f"{count} exception columns leave nothing to quantize in {w.shape[1]} inputs")
    score = diag.astype(np.float64)
    if select == "error":
        score = score * rounding_error_weights(w, group)
    return np.sort(np.argsort(-score, kind="stable")[:count]).astype(np.int64)


def quantize_rtn(w: np.ndarray, group: int, clip: str, diag: np.ndarray | None,
                 exceptions: np.ndarray = NO_EXCEPTIONS):
    """Codes, scales and the exact weights of the exception columns (their original values; the
    codes and scales are computed with those columns zeroed, so their codes are 0)."""
    n, k = w.shape
    source = w
    if len(exceptions):
        source = w.copy()
        source[:, exceptions] = 0
    groups = (k + group - 1) // group
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, groups), dtype=np.float32)
    for g in range(groups):
        a, b = g * group, min(k, (g + 1) * group)
        block = source[:, a:b]
        s = best_scales(block, None if diag is None else diag[a:b], clip)
        codes[:, a:b] = codes_for(block, s)
        scales[:, g] = s
    return codes, scales, np.ascontiguousarray(w[:, exceptions], dtype=np.float32)


def quantize_gptq(w: np.ndarray, gram: np.ndarray, group: int, clip: str, damp: float = 0.01,
                  act_order: bool = False, exceptions: np.ndarray = NO_EXCEPTIONS):
    """GPTQ (Frantar et al.) with per-group scales, in float64: codes, scales and the FP32
    weights of the exception columns (updated by GPTQ, below).

    Columns are processed in order; each column's rounding error is spread over
    the remaining columns with the inverse-Hessian Cholesky factor. Without
    act_order, the scale of a group is fixed from the error-updated weights
    when the group starts. With act_order, columns are processed by
    descending activation energy and every group scale is fixed up front from
    the original weights (static groups), so the stored layout is unchanged.

    Exception columns (as in OWQ/SpQR) go last in the order and are never
    quantized; group scales are computed with them zeroed and their codes are
    0. GPTQ's update extends to them unchanged: the update after fixing a
    column is the OBS step, which leaves the still-free columns at their
    least-squares optimum given every column fixed so far. After the last
    quantized column the exception columns therefore hold
    W_E - (Q - W_Q) H_QE H_EE^-1 (H the damped Gram), the best any exact
    weights can do for the chosen codes, and are returned as FP32.
    """
    if act_order:
        return quantize_gptq_act_order(w, gram, group, clip, damp, exceptions)
    n, k = w.shape
    # The quantized columns in natural order, then the exception columns (identity without any).
    kept = np.setdiff1d(np.arange(k), exceptions)
    perm = np.concatenate([kept, exceptions])
    position = np.argsort(perm)
    h = gram.astype(np.float64)[np.ix_(perm, perm)].copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    work = w.astype(np.float64)[:, perm].copy()
    work[:, dead] = 0.0
    h += np.eye(k) * damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T  # upper factor
    groups = (k + group - 1) // group
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, groups), dtype=np.float32)
    diag = np.diag(gram).astype(np.float64)
    for g in range(groups):
        a, b = g * group, min(k, (g + 1) * group)
        block = work[:, position[a:b]]
        block[:, np.isin(np.arange(a, b), exceptions)] = 0.0
        block32 = block.astype(np.float32)
        s = best_scales(block32, diag[a:b] if clip == "wmse" else None, "rtn" if clip == "rtn" else clip)
        scales[:, g] = s
        divisor = np.where(s == 0, 1.0, s.astype(np.float64))
        for j in kept[(kept >= a) & (kept < b)]:
            p = position[j]
            col = work[:, p]
            q = np.clip(np.rint(col / divisor), -127, 127)
            q[s == 0] = 0
            codes[:, j] = q.astype(np.int8)
            deq = (q.astype(np.float32) * s).astype(np.float64)
            err = (col - deq) / hinv[p, p]
            work[:, p + 1:] -= np.outer(err, hinv[p, p + 1:])
    return codes, scales, np.ascontiguousarray(work[:, len(kept):], dtype=np.float32)


def quantize_gptq_act_order(w: np.ndarray, gram: np.ndarray, group: int, clip: str, damp: float,
                            exceptions: np.ndarray = NO_EXCEPTIONS):
    n, k = w.shape
    diag = np.diag(gram).astype(np.float64)
    source = w
    if len(exceptions):
        source = w.copy()
        source[:, exceptions] = 0
    groups = (k + group - 1) // group
    scales = np.zeros((n, groups), dtype=np.float32)
    for g in range(groups):
        a, b = g * group, min(k, (g + 1) * group)
        scales[:, g] = best_scales(source[:, a:b].astype(np.float32),
                                   diag[a:b] if clip == "wmse" else None,
                                   "rtn" if clip == "rtn" else clip)
    column_scale = np.repeat(scales, group, axis=1)[:, :k].astype(np.float64)
    # Descending activation energy over the quantized columns, then the exception columns.
    kept = np.setdiff1d(np.arange(k), exceptions)
    perm = np.concatenate([kept[np.argsort(-diag[kept], kind="stable")], exceptions])
    h = gram.astype(np.float64)[np.ix_(perm, perm)].copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    work = w.astype(np.float64)[:, perm].copy()
    work[:, dead] = 0.0
    h += np.eye(k) * damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T
    s_perm = column_scale[:, perm]
    q_perm = np.zeros((n, k), dtype=np.int8)
    for j in range(len(kept)):
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
    return codes, scales, np.ascontiguousarray(work[:, len(kept):], dtype=np.float32)


def overlay_tensors(name: str, codes: np.ndarray, scales: np.ndarray, exceptions: np.ndarray,
                    values: np.ndarray) -> dict[str, np.ndarray]:
    """The overlay tensors of one matrix: codes and scales, and any exception columns and values."""
    tensors = {name + ".__w8_codes": codes, name + ".__w8_scales": scales}
    if len(exceptions):
        tensors[name + ".__w8_exc_cols"] = exceptions.astype(np.int32)
        tensors[name + ".__w8_exc_vals"] = values
    return tensors


def main(argv: list[str] | None = None) -> None:
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
    ap.add_argument("--exceptions", type=int, default=0, metavar="N",
                    help="input columns per matrix kept exact in FP32 (experimental; needs --gram-dir)")
    ap.add_argument("--exceptions-include", default=".*", metavar="REGEX",
                    help="matrices that get exception columns (suggested start: 'feed_forward\\.w2')")
    ap.add_argument("--exception-select", choices=("energy", "error"), default="energy",
                    help="rank columns by activation energy diag(G), or by it times the expected rounding error")
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error("output exists")
    if (args.clip == "wmse" or args.method == "gptq") and not args.gram_dir:
        ap.error("--clip wmse and --method gptq need --gram-dir")
    if args.exceptions < 0 or (args.exceptions and not args.gram_dir):
        ap.error("--exceptions takes a count >= 0 and needs --gram-dir")
    source = args.model / "model.safetensors"
    if sha256(source) != SOURCE_SHA256 or sha256(args.model / "config.json") != CONFIG_SHA256:
        ap.error("pinned source/config SHA-256 mismatch")
    include, exclude = re.compile(args.include), re.compile(args.exclude)
    exceptions_include = re.compile(args.exceptions_include)
    names = [n for n in inventory(False) if include.search(n) and not exclude.search(n)]
    if args.exceptions and not any(exceptions_include.search(n) for n in names):
        ap.error("--exceptions-include matches no matrix selected by --include/--exclude")
    if any(args.exceptions >= inventory(False)[n][1] for n in names if exceptions_include.search(n)):
        ap.error("--exceptions must stay below the input width of every matrix that gets them")
    tensors = {}
    exception_matrices = exception_columns = 0
    with safe_open(source, framework="numpy") as model:
        for name in names:
            w = model.get_tensor(name)
            if args.source_bf16:
                w = (w.view(np.uint32) + 0x7FFF + ((w.view(np.uint32) >> 16) & 1) & 0xFFFF0000).view(np.float32)
            gram = None
            if args.gram_dir:
                gram = np.load(args.gram_dir / f"{name}.gram.npy")
            exceptions = NO_EXCEPTIONS
            if args.exceptions and exceptions_include.search(name):
                exceptions = select_exceptions(w, np.diag(gram), args.exceptions, args.exception_select, args.group)
            if args.method == "gptq":
                codes, scales, values = quantize_gptq(w, gram, args.group, args.clip, args.damp, args.act_order,
                                                      exceptions)
            else:
                diag = None if gram is None else np.diag(gram).astype(np.float64)
                codes, scales, values = quantize_rtn(w, args.group, args.clip, diag, exceptions)
            deq = codes.astype(np.float32) * np.repeat(scales, args.group, axis=1)[:, :w.shape[1]]
            deq[:, exceptions] = values
            rel = float(np.linalg.norm(deq - w) / np.linalg.norm(w))
            note = f", exception columns {exceptions.tolist()}" if len(exceptions) else ""
            print(f"{name}: relative error {rel:.3e}{note}", flush=True)
            tensors.update(overlay_tensors(name, codes, scales, exceptions, values))
            exception_matrices += bool(len(exceptions))
            exception_columns += len(exceptions)
    partial = len(names) < len(inventory(False))
    metadata = {"format": FORMAT_EXCEPTIONS if exception_columns else FORMAT, "source_sha256": SOURCE_SHA256,
                "model_revision": REVISION,
                "include_head": "false", "group_size": str(args.group), "scale_dtype": "f32",
                "rounding": f"{args.method}-{args.clip}" + ("-actorder" if args.act_order else "")
                + ("-bf16source" if args.source_bf16 else ""),
                "activation_dtype": "f32",
                "partial": str(partial).lower()}
    if exception_columns:
        metadata.update({"exceptions": str(args.exceptions), "exceptions_include": args.exceptions_include,
                         "exception_select": args.exception_select, "exception_matrices": str(exception_matrices),
                         "exception_columns": str(exception_columns)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    print(json.dumps({**metadata, "matrices": len(names), "output": str(args.output),
                      "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
