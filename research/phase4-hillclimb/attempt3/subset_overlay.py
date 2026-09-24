#!/usr/bin/env python3
"""Drop matrices from a W8 overlay so they stay FP32 (mixed precision).

  python research/phase4-hillclimb/attempt3/subset_overlay.py --input artifacts/model/w8-gptq.safetensors \
      --output X.safetensors --exclude "feed_forward\\.w2"

Each body matrix in an overlay is quantized independently (RTN, or GPTQ from
FP32 activations), so removing some leaves the others unchanged. The output
is marked `partial`, which the loader accepts; omitted matrices use the FP32
checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exclude", required=True, help="regex over matrix names to keep in FP32")
    a = ap.parse_args()
    if a.output.exists():
        ap.error("output exists")
    exclude = re.compile(a.exclude)
    with safe_open(a.input, framework="numpy") as f:
        metadata = dict(f.metadata())
        names = list(f.keys())
        kept = {n: f.get_tensor(n) for n in names if not exclude.search(n.rsplit(".__w8_", 1)[0])}
    dropped = sorted({n.rsplit(".__w8_", 1)[0] for n in names} - {n.rsplit(".__w8_", 1)[0] for n in kept})
    metadata["partial"] = "true"
    save_file(kept, a.output, metadata=metadata)
    print(json.dumps({"output": str(a.output), "kept_matrices": len(kept) // 2, "fp32_matrices": len(dropped),
                      "fp32_params": None, "bytes": a.output.stat().st_size}))


if __name__ == "__main__":
    main()
