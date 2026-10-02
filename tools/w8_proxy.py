#!/usr/bin/env python3
"""Output-error proxy of 8-bit body matrices with k exception columns, from captured Grams (offline).

  python tools/w8_proxy.py --gram-dir artifacts/w8/gram [--model artifacts/model] [--include REGEX]
      [--ks 0,1,2,4,8,16] [--methods rtn,gptq] [--overlay OVERLAY] [--jobs N] [--json OUT]

For every matrix with a Gram G = E[x x^T] of its input (tools/make_gptq_overlay.sh captures them):
* the share of tr(G) in its k highest-energy input channels;
* the relative output error sqrt(tr(dW G dW^T) / tr(W G W^T)) of round-to-nearest and of GPTQ with
  k exception columns (dW: reconstructed minus source weights), quantized exactly as
  tools/w8_variants.py does (defaults: the fast-mode recipe, G64, act-order);
* with --overlay, that overlay's error, its exception columns included (absent matrices are FP32).

A weight-only quantized product errs by about sum_j dw_j x_j, so a channel with a huge activation
multiplies its column's rounding error; the energy share says how much k exact columns can remove.
Use it to choose --exceptions for w8_variants.py before any model run: GPTQ takes seconds per
matrix and k, so start with --include 'feed_forward\\.w2' and use --jobs on a many-core machine.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_w8 import CONFIG_SHA256, SOURCE_SHA256, inventory, sha256  # noqa: E402
from w8_variants import quantize_gptq, quantize_rtn, select_exceptions  # noqa: E402


def relative_error(w: np.ndarray, reconstructed: np.ndarray, gram: np.ndarray) -> float:
    """sqrt(tr(dW G dW^T) / tr(W G W^T)) in float64."""
    w = w.astype(np.float64)
    dw = reconstructed.astype(np.float64) - w
    g = gram.astype(np.float64)
    return float(np.sqrt(max(float(((dw @ g) * dw).sum()), 0.0) / float(((w @ g) * w).sum())))


def energy_share(diag: np.ndarray, k: int) -> float:
    """Share of tr(G) in the k largest diagonal entries."""
    energy = np.sort(diag.astype(np.float64))[::-1]
    return float(energy[:k].sum() / energy.sum())


def reconstruct(codes: np.ndarray, scales: np.ndarray, group: int, exceptions: np.ndarray,
                values: np.ndarray) -> np.ndarray:
    """fl(code * scale), with the exact weights at the exception columns (what the runner uses)."""
    weights = codes.astype(np.float32) * np.repeat(scales, group, axis=1)[:, :codes.shape[1]]
    weights[:, exceptions] = values
    return weights


def evaluate(w: np.ndarray, gram: np.ndarray, ks: list[int], methods: list[str], group: int = 64,
             clip: str = "rtn", damp: float = 0.01, act_order: bool = True,
             select: str = "energy") -> dict[str, list[float]]:
    """Energy shares and, per method, the relative output error for every k."""
    diag = np.diag(gram).astype(np.float64)
    result = {"energy_share": [energy_share(diag, k) for k in ks]}
    for method in methods:
        errors = []
        for k in ks:
            exceptions = select_exceptions(w, diag, k, select, group)
            if method == "gptq":
                codes, scales, values = quantize_gptq(w, gram, group, clip, damp, act_order, exceptions)
            else:
                codes, scales, values = quantize_rtn(w, group, clip, diag, exceptions)
            errors.append(relative_error(w, reconstruct(codes, scales, group, exceptions, values), gram))
        result[method] = errors
    return result


def overlay_error(overlay: Path, name: str, w: np.ndarray, gram: np.ndarray) -> tuple[float, int] | None:
    """An overlay's relative output error for one matrix and its exception column count, or None
    when the overlay leaves the matrix in FP32."""
    with safe_open(overlay, framework="numpy") as f:
        if name + ".__w8_codes" not in f.keys():
            return None
        group = int(f.metadata()["group_size"])
        exceptions, values = np.zeros(0, dtype=np.int64), np.zeros((w.shape[0], 0), dtype=np.float32)
        if name + ".__w8_exc_cols" in f.keys():
            exceptions = f.get_tensor(name + ".__w8_exc_cols").astype(np.int64)
            values = f.get_tensor(name + ".__w8_exc_vals")
        weights = reconstruct(f.get_tensor(name + ".__w8_codes"), f.get_tensor(name + ".__w8_scales"), group,
                              exceptions, values)
    return relative_error(w, weights, gram), len(exceptions)


def evaluate_matrix(task: tuple) -> tuple[str, dict]:
    """One matrix's report (runs in a worker process with --jobs)."""
    name, model, gram_dir, overlay, options = task
    with safe_open(model / "model.safetensors", framework="numpy") as f:
        w = f.get_tensor(name)
    gram = np.load(gram_dir / f"{name}.gram.npy")
    report = evaluate(w, gram, **options)
    if overlay is not None:
        found = overlay_error(overlay, name, w, gram)
        report["overlay"] = None if found is None else found[0]
        report["overlay_exceptions"] = None if found is None else found[1]
    return name, report


def format_row(label: str, values: list[float], width: int = 10) -> str:
    return f"  {label:14s}" + "".join(f"{v:{width}.3e}" if label != "energy share" else f"{v:{width}.4f}"
                                      for v in values)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gram-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--include", default=".*", help="matrices to report (regex over names)")
    ap.add_argument("--ks", default="0,1,2,4,8,16", help="exception column counts (comma-separated)")
    ap.add_argument("--methods", default="rtn,gptq", help="rtn and/or gptq (comma-separated)")
    ap.add_argument("--group", type=int, choices=(32, 64), default=64)
    ap.add_argument("--clip", choices=("rtn", "mse", "wmse"), default="rtn")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--gptq-order", choices=("act", "natural"), default="act")
    ap.add_argument("--exception-select", choices=("energy", "error"), default="energy")
    ap.add_argument("--overlay", type=Path, help="also report this overlay's error")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (one matrix each)")
    ap.add_argument("--json", type=Path, help="write every number here (must not exist)")
    args = ap.parse_args(argv)
    ks = [int(k) for k in args.ks.split(",")]
    methods = args.methods.split(",")
    if any(k < 0 for k in ks) or not set(methods) <= {"rtn", "gptq"} or args.jobs < 1:
        ap.error("--ks takes counts >= 0, --methods rtn and/or gptq, --jobs at least 1")
    if args.json and args.json.exists():
        ap.error("--json output exists")
    if sha256(args.model / "model.safetensors") != SOURCE_SHA256 or sha256(args.model / "config.json") != CONFIG_SHA256:
        ap.error("pinned source/config SHA-256 mismatch")
    include = re.compile(args.include)
    names = [n for n in inventory(False) if include.search(n) and (args.gram_dir / f"{n}.gram.npy").is_file()]
    if not names:
        ap.error("no matrix matches --include with a Gram in --gram-dir")
    if any(k >= inventory(False)[n][1] for n in names for k in ks):
        ap.error("--ks must stay below every reported matrix's input width")
    options = {"ks": ks, "methods": methods, "group": args.group, "clip": args.clip, "damp": args.damp,
               "act_order": args.gptq_order == "act", "select": args.exception_select}
    tasks = [(name, args.model, args.gram_dir, args.overlay, options) for name in names]
    reports = {}
    print("k" + " " * 15 + "".join(f"{k:10d}" for k in ks))
    with contextlib.ExitStack() as stack:
        results = map(evaluate_matrix, tasks)
        if args.jobs > 1:
            # One BLAS thread per worker; spawned workers read this environment when they import numpy.
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(key, "1")
            context = multiprocessing.get_context("spawn")
            results = stack.enter_context(ProcessPoolExecutor(args.jobs, mp_context=context)).map(evaluate_matrix,
                                                                                                    tasks)
        for name, report in results:
            reports[name] = report
            print(name)
            print(format_row("energy share", report["energy_share"]))
            for method in methods:
                print(format_row(method, report[method]))
            if args.overlay is None:
                continue
            if report["overlay"] is None:
                print("  overlay       absent (FP32)")
            else:
                print(f"  overlay       {report['overlay']:.3e} ({report['overlay_exceptions']} exception columns)")
    mean = {method: [float(np.mean([r[method][i] for r in reports.values()])) for i in range(len(ks))]
            for method in methods}
    # Per exception column: an FP32 weight per output row and an I32 column index.
    extra = [sum(k * (4 * inventory(False)[n][0] + 4) for n in names) for k in ks]
    print(f"mean over {len(names)} matrices")
    for method in methods:
        print(format_row(method, mean[method]))
    print("  extra bytes   " + "".join(f"{b:10d}" for b in extra))
    if args.json:
        overlay = str(args.overlay) if args.overlay else None
        args.json.write_text(json.dumps({"ks": ks, "options": options, "overlay": overlay, "matrices": reports,
                                         "mean": mean, "exception_bytes": extra}, indent=1) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
