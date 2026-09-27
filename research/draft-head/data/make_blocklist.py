#!/usr/bin/env python3
"""Perceptual hashes of every evaluation page, so training pages near them are dropped.

Covers the full OmniDocBench release (all 1,651 pages, not only our subset)
and our local evaluation corpus (canonical renders and original sources).

  python research/draft-head/data/make_blocklist.py --omnidocbench D:/falcon-draft/raw/opendatalab__OmniDocBench \
      --corpus artifacts/corpus/v3 --out D:/falcon-draft/blocklist.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imagehash
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = [p for p in a.omnidocbench.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
             and "images" in p.parts]
    files += [p for p in a.corpus.rglob("*") if p.name in ("canonical-rgb.png", "source.jpg")]
    hashes = set()
    for p in files:
        image = Image.open(p).convert("RGB")
        hashes.add(str(imagehash.phash(image)))
        # Rotated copies too, so an augmented or rotated duplicate is caught.
        for angle in (90, 180, 270):
            hashes.add(str(imagehash.phash(image.rotate(angle, expand=True))))
    a.out.write_text(json.dumps(sorted(hashes)))
    print(f"{len(files)} evaluation images -> {len(hashes)} hashes (with rotations) in {a.out}")


if __name__ == "__main__":
    main()
