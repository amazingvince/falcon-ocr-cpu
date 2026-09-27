#!/usr/bin/env python3
"""Phase 1 data: thumbnails, cheap image statistics and routing labels for
the Phase 0 pages.

Per page: a 384-px grayscale thumbnail (long side 384, padded white to a
square) for the CNN; about twenty image statistics the runner can compute in
about a millisecond (size, ink, colour, a projection-profile estimate of line
height and count, column gaps, edge density); the labels (routable at 1024 and
768: text within the tolerance of the 1536 transcript and the same stop
reason), page type, source, tokens and image tokens for the cost model; and
the source document, so splits never put pages of one document on both sides.

  python research/resolution-router/build_dataset.py --router D:/falcon-draft/router
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

THUMB = 384
FEATURES = ["width", "height", "aspect", "megapixels", "gray_mean", "gray_std", "ink", "dark", "saturation",
            "edge", "row_coverage", "lines", "line_height", "line_height_p10", "line_gap", "line_height_px",
            "col_gaps", "col_coverage", "ink_rows_std", "small_blobs",
            "loss_1024", "loss_768", "lossy_ink_1024", "lossy_ink_768", "ink_edge_ratio", "laplacian"]
CROP, CROPS = 192, 4


def native_path(path: str) -> str:
    """Windows paths in the manifests, as seen from WSL."""
    if os.name != "nt" and len(path) > 2 and path[1] == ":":
        return "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
    return path


def resample_loss(g: np.ndarray, size: int) -> tuple[float, float]:
    """How much the inked neighbourhoods change when the page is downscaled so
    its long side is `size` (bicubic, as the model resizes) and scaled back."""
    h, w = g.shape
    scale = size / max(w, h)
    if scale >= 1:
        return 0.0, 0.0
    img = Image.fromarray(g.astype(np.uint8))
    down = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    up = np.asarray(down.resize((w, h), Image.BILINEAR), dtype=np.float32)
    ink = g < 160
    if not ink.any():
        return 0.0, 0.0
    diff = np.abs(g - up)[ink]
    return float(diff.mean()), float((diff > 60).mean())


def crops(g: np.ndarray) -> np.ndarray:
    """The CROPS densest-ink CROP x CROP tiles at native resolution."""
    h, w = g.shape
    ink = (g < 160).astype(np.float32)
    tiles = []
    for y in range(0, max(1, h - CROP + 1), CROP // 2):
        for x in range(0, max(1, w - CROP + 1), CROP // 2):
            tiles.append((ink[y:y + CROP, x:x + CROP].mean(), y, x))
    tiles.sort(reverse=True)
    out, used = [], []
    for _, y, x in tiles:
        if any(abs(y - uy) < CROP and abs(x - ux) < CROP for uy, ux in used):
            continue
        used.append((y, x))
        tile = np.full((CROP, CROP), 255, np.uint8)
        part = g[y:y + CROP, x:x + CROP].astype(np.uint8)
        tile[:part.shape[0], :part.shape[1]] = part
        out.append(tile)
        if len(out) == CROPS:
            break
    while len(out) < CROPS:
        out.append(np.full((CROP, CROP), 255, np.uint8))
    return np.stack(out)


def statistics(img: Image.Image) -> list[float]:
    """Image statistics on the page scaled to a 1024-px long side."""
    w, h = img.size
    scale = 1024 / max(w, h)
    g = np.asarray(img.convert("L").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR),
                   dtype=np.float32)
    hsv = np.asarray(img.resize((256, max(1, round(256 * h / w)))).convert("HSV"), dtype=np.float32)
    ink_mask = g < 160
    rows = ink_mask.mean(axis=1)
    text_rows = rows > 0.01
    # Runs of inked rows = text lines.
    runs, gaps, start, last_end = [], [], None, None
    for i, on in enumerate(np.append(text_rows, False)):
        if on and start is None:
            start = i
            if last_end is not None:
                gaps.append(i - last_end)
        elif not on and start is not None:
            runs.append(i - start)
            last_end, start = i, None
    runs = [r for r in runs if r >= 2] or [0]
    cols = ink_mask.mean(axis=0)
    band = cols > 0.005
    # Interior blank column gaps at least 1% of the width wide.
    col_gaps, run = 0, 0
    inside = np.nonzero(band)[0]
    if inside.size:
        for on in band[inside[0]:inside[-1] + 1]:
            run = 0 if on else run + 1
            if run == max(3, g.shape[1] // 100):
                col_gaps += 1
    gy, gx = np.abs(np.diff(g, axis=0)).mean(), np.abs(np.diff(g, axis=1)).mean()
    small = np.abs(np.diff(ink_mask.astype(np.int8), axis=1)).sum() / max(1, ink_mask.sum())
    height = g.shape[0]
    return [w, h, w / h, w * h / 1e6, g.mean(), g.std(), ink_mask.mean(), (g < 80).mean(), hsv[..., 1].mean(),
            gx + gy, text_rows.mean(), len(runs), float(np.median(runs)) / height,
            float(np.percentile(runs, 10)) / height, (float(np.median(gaps)) / height) if gaps else 0.0,
            float(np.median(runs)) / scale, col_gaps, band.mean(), rows.std(), small]


def native_statistics(g: np.ndarray) -> list[float]:
    """Resampling loss at 1024 and 768 and stroke statistics, at native resolution."""
    l1024, f1024 = resample_loss(g, 1024)
    l768, f768 = resample_loss(g, 768)
    ink = g < 160
    edges = (np.abs(np.diff(g, axis=1)) > 40).sum() + (np.abs(np.diff(g, axis=0)) > 40).sum()
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return [l1024, l768, f1024, f768, float(ink.sum() / max(1, edges)), float(lap[ink[1:-1, 1:-1]].mean()) if ink.any() else 0.0]


def thumbnail(img: Image.Image) -> np.ndarray:
    w, h = img.size
    scale = THUMB / max(w, h)
    t = img.convert("L").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    canvas = Image.new("L", (THUMB, THUMB), 255)
    canvas.paste(t, (0, 0))
    return np.asarray(canvas, dtype=np.uint8)


def one(path: str):
    with Image.open(native_path(path)) as img:
        img = img.convert("RGB")
        g = np.asarray(img.convert("L"), dtype=np.float32)
        return thumbnail(img), statistics(img) + native_statistics(g), crops(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (a.router / "phase0-rows.jsonl").read_text(encoding="utf-8").splitlines())}
    sample = [json.loads(l) for l in (a.router / "sample.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = [p for p in sample if p["id"] in rows]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        out = list(pool.map(one, [p["path"] for p in sample], chunksize=16))
    thumbs = np.stack([t for t, _, _ in out])
    feats = np.asarray([f for _, f, _ in out], dtype=np.float32)
    tiles = np.stack([c for _, _, c in out])
    meta = []
    for p in sample:
        r = rows[p["id"]]
        meta.append({"id": p["id"], "source": p["source"], "type": r["type"], "group": p.get("document") or p["id"],
                     "disagreement": r["disagreement"], "stop": r["stop"], "tokens": r["tokens"],
                     "image_tokens": r["image_tokens"]})
    np.savez_compressed(a.router / "phase1-data.npz", thumbs=thumbs, features=feats, crops=tiles)
    (a.router / "phase1-meta.json").write_text(json.dumps({"features": FEATURES, "pages": meta}), encoding="utf-8")
    print(f"{len(meta)} pages, thumbs {thumbs.shape}, features {feats.shape}")


if __name__ == "__main__":
    main()
