#!/usr/bin/env python3
"""Check exact BF16 storage representability of the pinned FP32 checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

WEIGHTS_SHA256 = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("artifacts/model/model.safetensors"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.weights) != WEIGHTS_SHA256:
        raise ValueError("Existing output or different checkpoint")
    with args.weights.open("rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
    results = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        elements = math.prod(meta["shape"])
        if meta["dtype"] != "F32" or end - begin != 4 * elements:
            raise ValueError("Unexpected pinned tensor representation: " + name)
        words = np.memmap(args.weights, dtype="<u4", mode="r", offset=8 + header_len + begin, shape=(elements,))
        inexact = nonfinite = 0
        for start in range(0, elements, 1 << 20):
            block = words[start:start + (1 << 20)]
            inexact += int(np.count_nonzero(block & 0xFFFF))
            nonfinite += int(np.count_nonzero((block & 0x7F800000) == 0x7F800000))
        results.append({"tensor": name, "shape": meta["shape"], "values": elements,
                        "not_bit_exact_bf16": inexact, "nonfinite": nonfinite})
        del block, words
    if sha(args.weights) != WEIGHTS_SHA256:
        raise ValueError("Checkpoint changed during scan")
    report = {"schema_version": 1, "weights_sha256": WEIGHTS_SHA256,
              "script_sha256": sha(Path(__file__)), "numpy_version": np.__version__,
              "method": "A finite FP32 bit pattern is exactly representable by BF16 widening iff its lower 16 bits are zero. No rounding or quantization is performed.",
              "tensor_count": len(results), "values": sum(r["values"] for r in results),
              "not_bit_exact_bf16": sum(r["not_bit_exact_bf16"] for r in results),
              "nonfinite": sum(r["nonfinite"] for r in results),
              "entirely_exact_tensor_count": sum(r["not_bit_exact_bf16"] == 0 and r["nonfinite"] == 0 for r in results),
              "tensors": results, "inference_executed": False, "performance_measured": False,
              "scope": "Storage feasibility only. BF16 conversion of these FP32 weights is a numerical change; no quality or speed conclusion."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}))


if __name__ == "__main__":
    main()
