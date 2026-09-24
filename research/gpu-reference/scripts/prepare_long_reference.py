#!/usr/bin/env python3
"""Wrap the original numerical stress page in the shared corpus runner contract."""
import hashlib
import json
import pathlib

from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

folder = pathlib.Path("artifacts/corpus/long-numeric-v1")
meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
image = folder / "canonical-rgb.png"
truth = folder / "ground-truth.txt"
assert sha256(image) == meta["canonical_png_sha256"]
assert sha256(truth) == meta["ground_truth_sha256"]
assert hashlib.sha256(Image.open(image).convert("RGB").tobytes()).hexdigest() == meta["rgb_sha256"]
manifest = {"schema_version": 1, "dataset": "original-numeric-stress", "revision": meta["generator_sha256"],
            "source_metadata_sha256": sha256(folder / "metadata.json"),
            "qualification": "Original deterministic numerical stress page; not natural-document OCR accuracy evidence. Measure actual EOS and budget truncation.",
            "pages": [{"id": "original:long-numeric-v1", "category": "long_numeric_stress", "source_family": "original-long-numeric-v1",
                       "canonical_path": image.as_posix(), "canonical_png_sha256": meta["canonical_png_sha256"],
                       "rgb_sha256": meta["rgb_sha256"], "ground_truth_path": truth.as_posix(),
                       "ground_truth_sha256": meta["ground_truth_sha256"], "width": meta["width"], "height": meta["height"],
                       "generator_sha256": meta["generator_sha256"], "plain_ground_truth_tokens": meta["plain_ground_truth_tokens"],
                       "requested_max_new_tokens": 8192, "requested_max_dimension": 1536}]}
target = pathlib.Path("reference/long-numeric-lock-v1.json")
serialized = json.dumps(manifest, indent=2) + "\n"
if target.exists() and target.read_text(encoding="utf-8") != serialized:
    raise FileExistsError("Refusing to change frozen long-output fixture contract")
target.write_text(serialized, encoding="utf-8")
print(json.dumps({"manifest": target.as_posix(), "sha256": sha256(target)}))
