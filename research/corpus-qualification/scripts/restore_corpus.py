#!/usr/bin/env python3
"""Rebuild the materialized pages of frozen OmniDocBench corpus locks from a
local copy of the dataset, verifying every hash the lock records (source
image, decoded RGB, canonical PNG, annotation page, ground truth). Locks are
never rewritten. Pages already present and matching are left alone.

The canonical PNG's bytes depend on the Pillow that wrote it (the lock's
`pillow_version`); run under that version for byte-identical files. With
another version the decoded pixels still match (`rgb_sha256`), which is what
the runner reads, and the mismatch is reported.

  python research/corpus-qualification/scripts/restore_corpus.py \\
      --images D:/falcon-draft/raw/opendatalab__OmniDocBench/images \\
      --lock reference/corpus-v3-evaluation-lock.json --lock reference/corpus-v3-calibration-lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import PIL
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_corpus import stable_hash  # noqa: E402  (also puts the frozen scripts/ on the path)
from prepare_corpus_smoke import ground_truth  # noqa: E402
from select_corpus import ANNOTATION_SHA256  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="the dataset's images/ folder")
    ap.add_argument("--annotations", type=Path, default=Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    ap.add_argument("--lock", type=Path, action="append", required=True)
    a = ap.parse_args()
    raw = a.annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANNOTATION_SHA256:
        sys.exit("annotation source hash mismatch")
    rows = {r["page_info"]["image_path"]: r for r in json.loads(raw.decode("utf-8"))}
    totals = {"restored": 0, "present": 0, "png_bytes_differ": 0}
    for lock_path in a.lock:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        wanted = lock.get("pillow_version")
        if wanted and wanted != PIL.__version__:
            print(f"{lock_path.name}: written with Pillow {wanted}, running {PIL.__version__}", flush=True)
        for page in lock["pages"]:
            dataset, name = page["id"].split(":", 1)
            if dataset != "omnidocbench":
                sys.exit(f"{page['id']}: not an OmniDocBench page")
            canonical = Path(page["canonical_path"])
            truth_path = Path(page["ground_truth_path"])
            if canonical.exists() and truth_path.exists() and sha256(canonical) == page["canonical_png_sha256"] \
                    and sha256(truth_path) == page["ground_truth_sha256"]:
                totals["present"] += 1
                continue
            out = canonical.parent
            out.mkdir(parents=True, exist_ok=True)
            source = Path(page["source_path"])
            shutil.copyfile(a.images / name, source)
            if sha256(source) != page["source_sha256"]:
                sys.exit(f"{page['id']}: source image hash differs from the lock")
            with Image.open(source) as loaded:
                rgb = loaded.convert("RGB")
            if hashlib.sha256(rgb.tobytes()).hexdigest() != page["rgb_sha256"]:
                sys.exit(f"{page['id']}: decoded RGB differs from the lock")
            rgb.save(canonical)
            if sha256(canonical) != page["canonical_png_sha256"]:
                totals["png_bytes_differ"] += 1
            row = rows[name]
            if stable_hash(row) != page["annotation_page_sha256"]:
                sys.exit(f"{page['id']}: annotation page hash differs from the lock")
            annotation = json.dumps(row, ensure_ascii=False, indent=2) + "\n"
            Path(page["annotation_path"]).write_text(annotation, encoding="utf-8")
            truth_path.write_text(ground_truth(row) + "\n", encoding="utf-8")
            if sha256(truth_path) != page["ground_truth_sha256"]:
                sys.exit(f"{page['id']}: ground truth differs from the lock")
            totals["restored"] += 1
        print(f"{lock_path.name}: done", flush=True)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
