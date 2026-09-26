#!/usr/bin/env python3
"""Select a fresh English quality-gate set from the pinned OmniDocBench pages.

Pages: English (`page_attribute.language == "english"`), never in corpus v1,
v2 or v3, from no document family that corpus v3 uses, at most one page per
family. Categories: the corpus rules of select_corpus.py (`eligible`), one per
page by priority tiny_text > formulas > tables > multi_column > ordinary,
plus two English sources the v3 categories never covered: slides (PPT2PDF)
and exam papers. Order within a category: SHA-256 of
"falcon-cpu-english-gate-v1\\0<category>\\0<id>". Writes a manifest that
prepare_corpus.py materializes from the local image copy (no downloads).

  python research/corpus-qualification/scripts/select_english_gate.py \\
      --images D:/falcon-draft/raw/opendatalab__OmniDocBench/images --output reference/english-gate-v1-manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_corpus import ANNOTATION_SHA256, DATASET, REVISION, char_count, eligible, family, tiny_height  # noqa: E402
from prepare_corpus import stable_hash  # noqa: E402

SEED = "falcon-cpu-english-gate-v1"
QUOTAS = {"tiny_text": 7, "formulas": 25, "tables": 25, "multi_column": 25, "ordinary": 25, "slides": 15, "exam": 15}
PRIORITY = ["tiny_text", "formulas", "tables", "multi_column", "ordinary", "slides", "exam"]


def gate_family(name: str) -> str:
    """`family`, with slide decks' pages folded into one family."""
    base = family(name)
    deck = re.match(r"^(.*)_page_\d+\.[^.]+$", base)
    return deck[1] if deck else base


def category(row) -> str | None:
    attrs = row["page_info"]["page_attribute"]
    for name in PRIORITY:
        if name == "slides":
            ok = attrs["data_source"] == "PPT2PDF" and char_count(row) >= 100
        elif name == "exam":
            ok = attrs["data_source"] == "exam_paper" and char_count(row) >= 100
        else:
            ok = eligible(row, name)
        if ok:
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    raw = a.annotations.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ANNOTATION_SHA256, "annotation source hash mismatch"
    rows = json.loads(raw.decode("utf-8"))
    used, v3_families = set(), set()
    for version in (1, 2, 3):
        for page in json.loads(Path(f"reference/corpus-manifest-v{version}.json").read_text(encoding="utf-8"))["pages"]:
            name = Path(page["image_path"]).name
            used.add(name)
            if version == 3:
                v3_families.add(gate_family(name))
    pools: dict[str, list] = {name: [] for name in PRIORITY}
    for row in rows:
        name = row["page_info"]["image_path"]
        if row["page_info"]["page_attribute"].get("language") != "english" or name in used:
            continue
        if gate_family(name) in v3_families:
            continue
        cat = category(row)
        if cat:
            rank = hashlib.sha256(f"{SEED}\0{cat}\0omnidocbench:{name}".encode()).hexdigest()
            pools[cat].append((rank, row))
    pages, families = [], set()
    for cat in PRIORITY:
        chosen = 0
        for _, row in sorted(pools[cat], key=lambda x: x[0]):
            if chosen >= QUOTAS[cat]:
                break
            name = row["page_info"]["image_path"]
            fam = gate_family(name)
            if fam in families:
                continue
            source = a.images / name
            if not source.exists():
                continue
            families.add(fam)
            chosen += 1
            attrs = row["page_info"]["page_attribute"]
            pages.append({
                "id": f"omnidocbench:{name}", "split": "evaluation", "category": cat, "smoke": False,
                "image_path": f"images/{name}",
                "image_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/images/{name}",
                "source_family": fam, "width": row["page_info"]["width"], "height": row["page_info"]["height"],
                "attributes": attrs,
                "ground_truth": {"annotation_file": "OmniDocBench.json", "image_path": name,
                                 "page_sha256": stable_hash(row)},
                "annotation_text_characters": char_count(row),
                "median_line_height_ratio": tiny_height(row),
                "visual_review": "none: fresh English gate set, category from the corpus rules only",
                # prepare_corpus.py copies this local source (the image of the pinned revision).
                "reviewed_source_path": source.as_posix(),
                "reviewed_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
    counts = {cat: sum(p["category"] == cat for p in pages) for cat in PRIORITY}
    manifest = {
        "schema_version": 3, "status": "english-gate-v1", "dataset": DATASET, "revision": REVISION,
        "annotation_sha256": ANNOTATION_SHA256,
        "source_terms": "Research purposes only; not for commercial use. Do not redistribute source images in runner releases.",
        "evaluation_count": len(pages), "smoke_count": 0, "calibration_count": 0,
        "category_counts": counts,
        "selection": (f"English OmniDocBench pages outside corpus v1/v2/v3 and outside every v3 document family, one "
                      f"page per family; categories by priority {PRIORITY} (select_corpus.eligible plus PPT2PDF slides "
                      f"and exam papers with >= 100 characters); SHA-256 rank with seed {SEED}; quotas {QUOTAS}."),
        "pool_sizes": {cat: len(pools[cat]) for cat in PRIORITY},
        "pages": pages,
    }
    a.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": len(pages), "category_counts": counts, "pool_sizes": manifest["pool_sizes"]}, indent=2))


if __name__ == "__main__":
    main()
