#!/usr/bin/env python3
"""The router's image statistics, version 2: the specification the Rust
runner reproduces bit for bit (src/router.rs).

Input is the page as the runner holds it before routing: the model's own
first resize capped at 1536 px (source mode, then RGB), i.e. what
`--max-dimension 1536` feeds the second resize. Every statistic is an integer
count or sum finished by a few IEEE float64 operations written out below in
the order Rust performs them; the only resampling is Pillow's 8-bit L-mode
BILINEAR and BICUBIC, which src/preprocess.rs implements exactly. Python ints
keep sums exact (numpy float reductions would not match a sequential sum).

The 26 statistics keep the names and meaning of build_dataset.py's first
version: size; grayscale mean, spread, ink, dark and edge density on the page
scaled to a 1024-px long side; mean saturation on a 4-px grid of the page;
a projection-profile estimate of text lines (count, median and 10th
percentile height, median gap, median height in page pixels), column gaps and
coverage, row-ink spread, horizontal ink transitions; and at page resolution
the resampling loss at 1024 and 768 (bicubic down, bilinear back up, over ink
pixels), ink per strong edge and the Laplacian over ink.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

FEATURES = ["width", "height", "aspect", "megapixels", "gray_mean", "gray_std", "ink", "dark", "saturation",
            "edge", "row_coverage", "lines", "line_height", "line_height_p10", "line_gap", "line_height_px",
            "col_gaps", "col_coverage", "ink_rows_std", "small_blobs",
            "loss_1024", "loss_768", "lossy_ink_1024", "lossy_ink_768", "ink_edge_ratio", "laplacian"]
CAP = 1536
ANALYSIS = 1024
INK, DARK, LOSS_DIFF, EDGE = 160, 80, 60, 40


def resize_image_if_necessary(image, shortest_dimension, longest_dimension):
    """The model processor's first resize (verbatim logic; default resample)."""
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if (shortest_dimension <= original_width <= longest_dimension
            and shortest_dimension <= original_height <= longest_dimension):
        return image
    is_vertical_image = original_width < original_height
    if original_width < shortest_dimension or original_height < shortest_dimension:
        if is_vertical_image:
            new_width = shortest_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = shortest_dimension
            new_width = int(new_height * aspect_ratio)
    else:
        if is_vertical_image:
            new_width = longest_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = longest_dimension
            new_width = int(new_height * aspect_ratio)
    if new_width > longest_dimension:
        new_width = longest_dimension
        new_height = int(new_width / aspect_ratio)
    if new_height > longest_dimension:
        new_height = longest_dimension
        new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height))


def capped_page(path: str) -> np.ndarray:
    """The runner's pre-routing page: first resize at the 1536 cap, then RGB."""
    with Image.open(path) as img:
        return np.asarray(resize_image_if_necessary(img, 64, CAP).convert("RGB"), dtype=np.uint8)


def gray(rgb: np.ndarray) -> np.ndarray:
    """Pillow's convert("L"): ITU-R 601-2 luma in 16-bit fixed point."""
    r, g, b = (rgb[..., c].astype(np.uint32) for c in range(3))
    return ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16).astype(np.uint8)


def resize(g: np.ndarray, width: int, height: int, resample) -> np.ndarray:
    return np.asarray(Image.fromarray(g, "L").resize((width, height), resample), dtype=np.uint8)


def median(sorted_values: list[int]) -> float:
    n = len(sorted_values)
    if n % 2:
        return float(sorted_values[n // 2])
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2


def percentile10(sorted_values: list[int]) -> float:
    position = 0.1 * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def total(a: np.ndarray) -> int:
    return int(a.sum(dtype=np.int64))


def std(count: int, s: int, s2: int) -> float:
    mean = s / count
    return math.sqrt(max(s2 / count - mean * mean, 0.0))


def resample_loss(g: np.ndarray, ink: np.ndarray, ink_count: int, size: int) -> tuple[float, float]:
    h, w = g.shape
    scale = size / max(w, h)
    if scale >= 1 or ink_count == 0:
        return 0.0, 0.0
    down = resize(g, max(1, int(w * scale)), max(1, int(h * scale)), Image.BICUBIC)
    up = resize(down, w, h, Image.BILINEAR)
    diff = np.abs(g.astype(np.int32) - up.astype(np.int32))[ink]
    return total(diff) / ink_count, total(diff > LOSS_DIFF) / ink_count


def features(rgb: np.ndarray) -> list[float]:
    h, w = rgb.shape[:2]
    page = gray(rgb)
    scale = ANALYSIS / max(w, h)
    a = resize(page, max(1, round(w * scale)), max(1, round(h * scale)), Image.BILINEAR)
    ah, aw = a.shape
    n = ah * aw
    ai = a.astype(np.int64)
    ink = a < INK
    ink_count = total(ink)

    sample = rgb[::4, ::4].astype(np.int32)
    hi, lo = sample.max(axis=2), sample.min(axis=2)
    saturation = total(np.where(hi > 0, (hi - lo) * 255 // np.maximum(hi, 1), 0)) / (sample.shape[0] * sample.shape[1])

    edge = total(np.abs(np.diff(ai, axis=1))) / (ah * (aw - 1)) if aw > 1 else 0.0
    edge += total(np.abs(np.diff(ai, axis=0))) / ((ah - 1) * aw) if ah > 1 else 0.0

    row_ink = [int(c) for c in ink.sum(axis=1)]
    text_rows = [c / aw > 0.01 for c in row_ink]
    runs, gaps, start, last_end = [], [], None, None
    for i, on in enumerate(text_rows + [False]):
        if on and start is None:
            start = i
            if last_end is not None:
                gaps.append(i - last_end)
        elif not on and start is not None:
            runs.append(i - start)
            last_end, start = i, None
    runs = sorted(r for r in runs if r >= 2) or [0]
    gaps.sort()
    col_ink = [int(c) for c in ink.sum(axis=0)]
    band = [c / ah > 0.005 for c in col_ink]
    col_gaps, run = 0, 0
    inside = [i for i, on in enumerate(band) if on]
    if inside:
        for on in band[inside[0]:inside[-1] + 1]:
            run = 0 if on else run + 1
            if run == max(3, aw // 100):
                col_gaps += 1
    transitions = total(ink[:, 1:] != ink[:, :-1])

    gi = page.astype(np.int32)
    page_ink = page < INK
    page_ink_count = total(page_ink)
    l1024, f1024 = resample_loss(page, page_ink, page_ink_count, 1024)
    l768, f768 = resample_loss(page, page_ink, page_ink_count, 768)
    edges = total(np.abs(np.diff(gi, axis=1)) > EDGE) + total(np.abs(np.diff(gi, axis=0)) > EDGE)
    lap = np.abs(4 * gi[1:-1, 1:-1] - gi[:-2, 1:-1] - gi[2:, 1:-1] - gi[1:-1, :-2] - gi[1:-1, 2:])
    inner = page_ink[1:-1, 1:-1]
    inner_count = total(inner)
    laplacian = total(lap[inner]) / inner_count if inner_count else 0.0

    line = median(runs)
    return [
        float(w), float(h), w / h, w * h / 1e6,
        total(ai) / n, std(n, total(ai), total(ai * ai)), ink_count / n, total(a < DARK) / n, saturation,
        edge, sum(text_rows) / ah, float(len(runs)), line / ah, percentile10(runs) / ah,
        median(gaps) / ah if gaps else 0.0, line / scale,
        float(col_gaps), sum(band) / aw, std(ah, sum(row_ink), sum(c * c for c in row_ink)) / aw,
        transitions / max(1, ink_count),
        l1024, l768, f1024, f768, page_ink_count / max(1, edges), laplacian,
    ]
