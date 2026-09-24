#!/usr/bin/env python3
"""Download only the 24 selected smoke images; freeze bytes, RGB pixels and GT."""
import concurrent.futures
import hashlib
import json
import pathlib
import unicodedata
import urllib.request

from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256
from select_corpus import ANNOTATION_SHA256


def ground_truth(row):
    ordered = []
    for index, block in enumerate(row["layout_dets"]):
        order = block.get("order")
        if block.get("ignore", False) or not isinstance(order, (int, float)) or order < 0:
            continue
        kind = block["category_type"]
        if kind == "table":
            text = block.get("html") or block.get("latex") or block.get("text", "")
        elif kind.startswith("equation"):
            text = block.get("latex") or block.get("text", "")
        else:
            text = block.get("text", "")
        if text.strip():
            ordered.append((order, index, unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()))
    ordered.sort()
    return "\n\n".join(text for _, _, text in ordered)


def main():
    manifest_path = pathlib.Path("reference/corpus-manifest-v1.json")
    manifest = json.loads(manifest_path.read_text())
    annotations_path = pathlib.Path("artifacts/corpus-source-metadata/OmniDocBench.json")
    assert sha256(annotations_path) == ANNOTATION_SHA256
    annotation_rows = {r["page_info"]["image_path"]: r for r in json.loads(annotations_path.read_text())}
    def prepare(page):
        name = hashlib.sha256(page["id"].encode()).hexdigest()[:16]
        out = pathlib.Path("artifacts/corpus/smoke") / name
        out.mkdir(parents=True, exist_ok=True)
        source = out / ("source" + pathlib.Path(page["image_path"]).suffix)
        if not source.exists():
            temporary = source.with_suffix(source.suffix + ".partial")
            with urllib.request.urlopen(page["image_url"], timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(source)
        with Image.open(source) as loaded:
            source_mode = loaded.mode
            rgb = loaded.convert("RGB")
        canonical = out / "canonical-rgb.png"
        rgb.save(canonical)
        annotation = annotation_rows[page["ground_truth"]["image_path"]]
        payload_sha = hashlib.sha256(json.dumps(annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert payload_sha == page["ground_truth"]["page_sha256"]
        (out / "annotation.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n")
        gt = ground_truth(annotation)
        assert gt, f"No ordered text ground truth for {page['id']}"
        gt_path = out / "ground-truth.txt"
        gt_path.write_text(gt + "\n")
        result = {"id": page["id"], "category": page["category"], "source_family": page["source_family"],
                  "source_url": page["image_url"], "source_path": str(source), "source_sha256": sha256(source),
                  "source_bytes": source.stat().st_size, "source_mode": source_mode,
                  "canonical_path": str(canonical), "canonical_png_sha256": sha256(canonical),
                  "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "width": rgb.width, "height": rgb.height,
                  "ground_truth_path": str(gt_path), "ground_truth_sha256": sha256(gt_path),
                  "annotation_path": str(out / "annotation.json"), "annotation_page_sha256": payload_sha,
                  "attributes": page["attributes"], "visual_review": "pending"}
        print(json.dumps({"page": name, "category": page["category"], "bytes": result["source_bytes"], "width": rgb.width, "height": rgb.height}), flush=True)
        return result
    pages = [page for page in manifest["pages"] if page["smoke"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(prepare, pages))
    assert len(prepared) == 24
    rgb_hashes = [p["rgb_sha256"] for p in prepared]
    assert len(set(rgb_hashes)) == 24, "Duplicate canonical images in smoke selection"
    lock = {"schema_version": 1, "dataset": manifest["dataset"], "revision": manifest["revision"],
            "corpus_manifest_sha256": sha256(manifest_path), "pages": prepared,
            "source_terms": manifest["source_terms"],
            "rgb_policy": "Decode with pinned Pillow, then convert RGB before upstream preprocessing; no EXIF auto-orientation. For source RGB images this preserves decoded pixels.",
            "ground_truth_policy": "Nonignored blocks with numeric order >= 0, sorted by (order, original index). Text field for text; HTML then LaTeX for tables; LaTeX for equations. NFC, LF, trim block edges, join two newlines.",
            "metrics_note": "Assembled text permits diagnostic normalized CER only. Preserve original per-block annotations for official text/table/formula metrics; not a published OmniDocBench score.",
            "qualification": "24-page smoke subset; visual category review and complete 200-page evaluation remain pending"}
    lock_path = pathlib.Path("reference/corpus-smoke-lock-v1.json")
    if lock_path.exists():
        old = json.loads(lock_path.read_text())
        assert old == lock, "Existing smoke lock differs; do not silently alter a frozen dataset"
    else:
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"smoke_pages": len(prepared), "downloaded_bytes": sum(p["source_bytes"] for p in prepared), "lock": str(lock_path)}))


if __name__ == "__main__":
    main()
