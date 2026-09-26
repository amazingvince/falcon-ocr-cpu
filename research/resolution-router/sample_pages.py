#!/usr/bin/env python3
"""Phase 0 of the resolution router: a stratified sample of stage-3 pages,
resized to lower maximum dimensions exactly as the model's preprocessing does.

Pages come from the draft-head stage-3 set (English, 1536-px transcripts from
the official vLLM image already exist), capped per source so every source has
enough pages for its own routable fraction; decontaminated pages are excluded.
Each page is resized with the model's `resize_image_if_necessary` (copied from
artifacts/model/processing_falcon_ocr.py: PIL's default bicubic, long side
capped at the target) and saved as PNG, so a vLLM server capped at 1536 only
applies the patch snapping the runner applies at `--max-dimension <target>`.
Writes one manifest per resolution for request_vllm_draftgen.py.

  python research/resolution-router/sample_pages.py --out D:/falcon-draft/router
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path

from PIL import Image

# Pages per source (all of the smaller sources).
CAPS = {"olmocr_documents": 2500, "pdfa": 2000, "idl": 1500, "olmocr_books": 1200, "doclaynet": 900,
        "olmocr_loc": 700, "olmocr_archives": 600, "loc_newspapers": 500, "notes_notesbank": 392,
        "zenodo_slides": 230, "notes_humyn": 169}
SEED = "falcon-router-phase0-v1"


def resize_image_if_necessary(image, shortest_dimension=224, longest_dimension=896):
    """Verbatim from artifacts/model/processing_falcon_ocr.py."""
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


def resize_one(job):
    source, target, size = job
    if target.exists():
        return target, None
    with Image.open(source) as img:
        out = resize_image_if_necessary(img.convert("RGB"), 64, size)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.save(target)
        return target, out.size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("D:/falcon-draft/pages/stage3/manifest.jsonl"))
    ap.add_argument("--outputs", type=Path, default=Path("D:/falcon-draft/outputs/stage3"))
    ap.add_argument("--exclusions", type=Path, default=Path("D:/falcon-draft/decontam-13gram.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 768])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    excluded = set(json.loads(a.exclusions.read_text()))
    pages = [json.loads(line) for line in a.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_source = collections.defaultdict(list)
    for p in pages:
        if p["id"] in excluded or not (a.outputs / (p["id"].replace("/", "__") + ".json")).exists():
            continue
        rank = hashlib.sha256(f"{SEED}\0{p['id']}".encode()).hexdigest()
        by_source[p["source"]].append((rank, p))
    sample = []
    for source, rows in sorted(by_source.items()):
        sample += [p for _, p in sorted(rows, key=lambda x: x[0])[:CAPS.get(source, 0)]]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sample.jsonl").write_text("".join(json.dumps(p) + "\n" for p in sample), encoding="utf-8")
    for size in a.sizes:
        jobs = [(Path(p["path"]), a.out / f"pages-{size}" / (p["id"].replace("/", "__") + ".png"), size) for p in sample]
        with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
            done = list(pool.map(resize_one, jobs, chunksize=16))
        manifest = [{**p, "path": str(target), "max_dimension": size} for p, (target, _) in zip(sample, done)]
        (a.out / f"manifest-{size}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
        print(f"{size}: {len(manifest)} pages", flush=True)
    counts = collections.Counter(p["source"] for p in sample)
    print(json.dumps({"sample": len(sample), "by_source": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
