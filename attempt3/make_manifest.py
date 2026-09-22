#!/usr/bin/env python3
"""Create explicit single-page cases and an optional joint batch from local images."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-dimension", type=int, default=1536)
    p.add_argument("--include-batch", action="store_true")
    a = p.parse_args()
    if a.output.exists() or not all(x.is_file() for x in a.images):
        p.error("select existing images and a new output path")
    if a.max_new_tokens <= 0 or a.max_dimension < 64 or a.max_dimension % 16:
        p.error("choose positive token budget and a max dimension >=64 divisible by 16")
    cases = [{"id": f"page-{i+1}", "images": [str(x.resolve())], "batch_size": 1,
              "max_new_tokens": a.max_new_tokens, "max_dimension": a.max_dimension} for i,x in enumerate(a.images)]
    if a.include_batch:
        if not 2 <= len(a.images) <= 8:
            p.error("batch example needs 2..8 images; prefer distinct long-output pages")
        cases.append({"id": "joint-batch", "images": [str(x.resolve()) for x in a.images], "batch_size": len(a.images),
                      "max_new_tokens": a.max_new_tokens, "max_dimension": a.max_dimension})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"schema":"falcon-ocr-attempt3-cases-v1", "cases":cases},indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
