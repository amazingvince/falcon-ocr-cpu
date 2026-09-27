#!/usr/bin/env python3
"""Figure-masked evaluation pages: paint white the regions of an OmniDocBench
page that the ground truth leaves out, so that no system is scored on content
the reference omits (a diagram transcribed as a table, a chart's labels).

A region is masked when its block contributes no ground-truth text (ignored,
no reading order, or empty text: the rule of prepare_corpus_smoke.ground_truth)
and its category is a figure or one of OmniDocBench's own masked regions.
Page furniture (headers, footers, page numbers) stays: it is small and hits
every system alike. Any part of a masked region that overlaps a block the
ground truth keeps (a caption, a table, a line of text) is left untouched.
The ground truth itself is unchanged.

Writes <output>/<page>/canonical-rgb.png and ground-truth.txt for every page
with something to mask, and a lock that points those pages at the masked copy
and every other page at its original image (same page directory names, so
reports of the original and masked sets can be merged page by page).

  python research/corpus-qualification/scripts/mask_excluded.py --lock reference/english-gate-v1-evaluation-lock.json \\
      --output artifacts/corpus/english-gate-v1-masked --masked-lock reference/english-gate-v1-masked-lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_corpus_smoke import ground_truth  # noqa: E402
from select_corpus import ANNOTATION_SHA256  # noqa: E402

MASKED = ("figure", "text_mask", "table_mask", "chart_mask", "unknown_mask", "need_mask",
          "organic_chemical_formula_mask", "algorithm_mask")


def contributes(block: dict) -> bool:
    """Whether ground_truth() takes text from this block."""
    return bool(ground_truth({"layout_dets": [block]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--masked-lock", type=Path, required=True)
    a = ap.parse_args()
    raw = a.annotations.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ANNOTATION_SHA256, "annotation source hash mismatch"
    rows = {r["page_info"]["image_path"]: r for r in json.loads(raw.decode("utf-8"))}
    lock = json.loads(a.lock.read_text(encoding="utf-8"))
    pages, masked_count = [], 0
    for entry in lock["pages"]:
        row = rows[entry["id"].split(":", 1)[1]]
        source = Path(entry["canonical_path"])
        truth = Path(entry["ground_truth_path"])
        # prepare_corpus.py writes the ground truth with a final newline.
        stored = truth.read_text(encoding="utf-8").rstrip("\n")
        assert stored == ground_truth(row), f"{entry['id']}: ground truth differs"
        regions = [b for b in row["layout_dets"] if b["category_type"] in MASKED and not contributes(b)]
        kept = [b for b in row["layout_dets"] if contributes(b)]
        page = dict(entry)
        if regions:
            with Image.open(source) as img:
                img = img.convert("RGB")
            w, h = row["page_info"]["width"], row["page_info"]["height"]
            assert img.size == (w, h), f"{entry['id']}: image {img.size} vs annotation {(w, h)}"
            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            for b in regions:
                draw.polygon([tuple(b["poly"][i:i + 2]) for i in range(0, len(b["poly"]), 2)], fill=255)
            for b in kept:
                draw.polygon([tuple(b["poly"][i:i + 2]) for i in range(0, len(b["poly"]), 2)], fill=0)
            share = sum(mask.histogram()[255:]) / (w * h)
            white = Image.new("RGB", img.size, (255, 255, 255))
            out_img = Image.composite(white, img, mask)
            difference = ImageChops.difference(out_img, img).convert("L").point(lambda v: 255 if v else 0)
            changed = sum(difference.histogram()[255:])
            out = a.output / source.parent.name
            out.mkdir(parents=True, exist_ok=True)
            out_img.save(out / "canonical-rgb.png")
            shutil.copyfile(truth, out / "ground-truth.txt")
            page["canonical_path"] = (out / "canonical-rgb.png").as_posix()
            page["ground_truth_path"] = (out / "ground-truth.txt").as_posix()
            page["canonical_png_sha256"] = hashlib.sha256((out / "canonical-rgb.png").read_bytes()).hexdigest()
            page["masked"] = {"regions": [b["category_type"] for b in regions], "page_share": round(share, 4),
                              "changed_pixels": changed, "source_canonical_path": entry["canonical_path"]}
            masked_count += 1
        pages.append(page)
    result = dict(lock)
    result["status"] = f"{lock.get('status', '')} figure-masked".strip()
    result["masking"] = ("mask_excluded.py: figure and *_mask regions without ground-truth text painted white, "
                         "minus every block with ground-truth text; ground truth unchanged; "
                         f"{masked_count} of {len(pages)} pages masked")
    result["pages"] = pages
    a.masked_lock.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shares = [p["masked"]["page_share"] for p in pages if "masked" in p]
    print(f"{masked_count} of {len(pages)} pages masked; masked share median {sorted(shares)[len(shares) // 2]:.1%}, "
          f"max {max(shares):.1%}" if shares else "nothing to mask")


if __name__ == "__main__":
    main()
