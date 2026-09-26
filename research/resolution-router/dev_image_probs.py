#!/usr/bin/env python3
"""Image-model routing probabilities for the router development set.

Trains the gradient-boosted image-statistics models on the Phase 0 pages with
the noise-aware labels of noise_aware.py (stable pages only), computes the
same statistics for the development pages (each capped at 1536 px first, as
the model's preprocessing does, since the training pages never exceeded it)
and writes {page: {"768": p, "1024": p}} for dev_policy.py --image-probs.

  python research/resolution-router/dev_image_probs.py --router /mnt/d/falcon-draft/router \\
      --lock reference/router-dev-v1-evaluation-lock.json --out /mnt/d/falcon-draft/router-dev/image-probs.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import native_path, native_statistics, statistics  # noqa: E402
from sample_pages import resize_image_if_necessary  # noqa: E402


def features(path: str) -> list[float]:
    with Image.open(native_path(path)) as img:
        img = resize_image_if_necessary(img.convert("RGB"), 64, 1536)
        g = np.asarray(img.convert("L"), dtype=np.float32)
        return statistics(img) + native_statistics(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, default=Path("/mnt/d/falcon-draft/outputs/stage3"))
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args()
    info = json.loads((a.router / "phase1-meta.json").read_text(encoding="utf-8"))
    meta, img = info["pages"], np.load(a.router / "phase1-data.npz")["features"]
    noise, stop_b = [], []
    for m in meta:
        name = m["id"].replace("/", "__") + ".json"
        first = json.loads((a.outputs / name).read_text(encoding="utf-8"))
        second = json.loads((a.router / "outputs-1536-retest" / name).read_text(encoding="utf-8"))
        noise.append(Levenshtein.distance(second["text"], first["text"]) / max(1, len(first["text"])))
        stop_b.append(second["finish_reason"])
    stable = np.array([nz <= a.tolerance and m["stop"]["1536"] == sb for m, nz, sb in zip(meta, noise, stop_b)])
    models = {}
    for s in ("768", "1024"):
        ok = np.array([m["disagreement"][s] <= max(a.tolerance, nz + a.margin) and m["stop"][s] in (m["stop"]["1536"], sb)
                       for m, nz, sb in zip(meta, noise, stop_b)])
        models[s] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(img[stable], ok[stable])
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    with concurrent.futures.ProcessPoolExecutor(8) as pool:
        dev = np.asarray(list(pool.map(features, [p["canonical_path"] for p in lock])), dtype=np.float32)
    out = {}
    for entry, f in zip(lock, dev):
        pid = Path(entry["canonical_path"]).parent.name
        out[pid] = {s: float(models[s].predict_proba(f[None])[0, 1]) for s in models}
        # For floor rules: the capped page's size and its median text-line height in its own pixels.
        out[pid]["width"], out[pid]["height"], out[pid]["line_height_px"] = float(f[0]), float(f[1]), float(f[15])
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} pages; mean p768 {np.mean([v['768'] for v in out.values()]):.2f}, "
          f"p1024 {np.mean([v['1024'] for v in out.values()]):.2f}")


if __name__ == "__main__":
    main()
