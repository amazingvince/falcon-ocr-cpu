#!/usr/bin/env python3
"""The router's ground-truth development set: every English OmniDocBench page
that no corpus (v1, v2, v3) and not the English gate uses, with text ground
truth. Categories follow the English gate's rules (tiny_text > formulas >
tables > multi_column > ordinary > slides > exam, else "other"). Writes a
manifest that research/corpus-qualification/scripts/prepare_corpus.py
materializes from the local image copy.

  python research/resolution-router/select_dev_pages.py \\
      --images D:/falcon-draft/raw/opendatalab__OmniDocBench/images --output reference/router-dev-v1-manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "corpus-qualification" / "scripts"))
from select_corpus import ANNOTATION_SHA256, DATASET, REVISION, char_count, tiny_height  # noqa: E402
from select_english_gate import category, gate_family  # noqa: E402
from prepare_corpus import stable_hash  # noqa: E402  (also puts the frozen scripts/ on the path)
from prepare_corpus_smoke import ground_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    raw = a.annotations.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ANNOTATION_SHA256, "annotation source hash mismatch"
    used = set()
    for version in (1, 2, 3):
        for page in json.loads(Path(f"reference/corpus-manifest-v{version}.json").read_text(encoding="utf-8"))["pages"]:
            used.add(Path(page["image_path"]).name)
    for page in json.loads(Path("reference/english-gate-v1-evaluation-lock.json").read_text(encoding="utf-8"))["pages"]:
        used.add(page["id"].split(":", 1)[1])
    pages = []
    for row in json.loads(raw.decode("utf-8")):
        name = row["page_info"]["image_path"]
        attrs = row["page_info"]["page_attribute"]
        if attrs.get("language") != "english" or name in used:
            continue
        if not ground_truth(row):
            continue
        source = a.images / name
        pages.append({
            "id": f"omnidocbench:{name}", "split": "evaluation", "category": category(row) or "other", "smoke": False,
            "image_path": f"images/{name}",
            "image_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/images/{name}",
            "source_family": gate_family(name), "width": row["page_info"]["width"], "height": row["page_info"]["height"],
            "attributes": attrs,
            "ground_truth": {"annotation_file": "OmniDocBench.json", "image_path": name, "page_sha256": stable_hash(row)},
            "annotation_text_characters": char_count(row), "median_line_height_ratio": tiny_height(row),
            "visual_review": "none: router development set, category from the corpus rules only",
            "reviewed_source_path": source.as_posix(),
            "reviewed_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    counts = {}
    for p in pages:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    manifest = {
        "schema_version": 3, "status": "router-dev-v1", "dataset": DATASET, "revision": REVISION,
        "annotation_sha256": ANNOTATION_SHA256,
        "source_terms": "Research purposes only; not for commercial use. Do not redistribute source images in runner releases.",
        "evaluation_count": len(pages), "smoke_count": 0, "calibration_count": 0, "category_counts": counts,
        "selection": "Every English OmniDocBench page outside corpus v1/v2/v3 and the English gate v1 with text ground truth; "
                     "categories by the English gate's rules. Resolution-router development set: tuning allowed.",
        "pages": pages,
    }
    a.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": len(pages), "category_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
