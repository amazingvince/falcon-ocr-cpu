#!/usr/bin/env python3
"""Freeze a bounded serving sample from visual evidence, before serving outputs."""
import json
import pathlib

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

source = pathlib.Path("reference/corpus-v3-evaluation-lock.json")
target = pathlib.Path("reference/serving-fullpages-v1-lock.json")
manifest = json.loads(source.read_text(encoding="utf-8"))
choices = [
    ("3f294b5e60a0c2d4", "ordinary", "Single-column journal abstract/introduction prose."),
    ("2c243b31d36eb729", "tables", "Scientific page dominated by a long numeric comparison table."),
    ("f75c4a0b98e45ab3", "multi_column", "Game manual with two sustained parallel prose columns."),
]
pages = []
for sample, category, reason in choices:
    page = next(p for p in manifest["pages"] if pathlib.Path(p["canonical_path"]).parent.name == sample)
    assert page["category"] == category and page["visual_review"]["status"] == "supported"
    assert sha256(page["canonical_path"]) == page["canonical_png_sha256"]
    pages.append({**page, "serving_selection_reason": reason})
result = {**{k: v for k, v in manifest.items() if k != "pages"},
          "selection": "bounded_official_and_vllm_serving", "source_manifest": source.as_posix(),
          "source_manifest_sha256": sha256(source),
          "selection_policy": "Three named natural pages selected from existing visual category evidence, without selecting for any observed serving or HF output. Not statistically representative corpus evidence.",
          "max_dimension": 1536, "min_dimension": 64, "max_new_tokens": 4096,
          "pages": pages}
serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
if target.exists():
    assert target.read_text(encoding="utf-8") == serialized, "Frozen serving sample differs"
else:
    target.write_text(serialized, encoding="utf-8")
print(json.dumps({"manifest": target.as_posix(), "sha256": sha256(target), "samples": [s for s, _, _ in choices]}))
