#!/usr/bin/env python3
"""Offline W8G64 overlay. Source stays authoritative; this is not GGML Q8_0.
Run: python attempt3/convert_w8.py --model artifacts/model --output body.w8.safetensors
Requires numpy and safetensors. Never imports model-defined Python code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
SOURCE_SHA256 = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"
CONFIG_SHA256 = "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf"
FORMAT = "falcon-ocr-attempt3-w8g64-v1"

def sha256(path: Path) -> str:
    # Chunked rather than hashlib.file_digest so Python 3.10 also works.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def encode(values: np.ndarray, group_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Same FP64 scale division/round-to-even contract as uploaded q8_reference.rs."""
    w = np.asarray(values, dtype=np.float32)
    if w.ndim != 2 or min(w.shape) <= 0 or group_size != 64 or not np.isfinite(w).all():
        raise ValueError("requires a finite, nonempty matrix and group_size=64")
    n, k = w.shape
    groups = (k + group_size - 1) // group_size
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, groups), dtype=np.float32)
    smallest = np.nextafter(np.float32(0), np.float32(1))
    for a in range(0, k, group_size):
        block = w[:, a:a+group_size]
        maxima = np.max(np.abs(block), axis=1)
        s = (maxima.astype(np.float64) / 127.0).astype(np.float32)
        s = np.where(maxima == 0, np.float32(0), np.maximum(s, smallest)).astype(np.float32)
        divisor = np.where(s == 0, np.float32(1), s).astype(np.float64)
        q = np.clip(np.rint(block.astype(np.float64) / divisor[:, None]), -127, 127).astype(np.int8)
        q[s == 0] = 0
        with np.errstate(over="ignore"):
            reconstructed = q.astype(np.float32) * s[:, None]
        if not np.isfinite(reconstructed).all():
            raise ValueError("FP32 dequantization overflow")
        codes[:, a:a+group_size] = q
        scales[:, a // group_size] = s
    return codes, scales

def inventory(include_head: bool) -> dict[str, tuple[int, int]]:
    result = {}
    for i in range(22):
        result[f"layers.{i}.attention.wqkv.weight"] = (2048, 768)
        result[f"layers.{i}.attention.wo.weight"] = (768, 1024)
        result[f"layers.{i}.feed_forward.w13.weight"] = (4608, 768)
        result[f"layers.{i}.feed_forward.w2.weight"] = (768, 2304)
    if include_head:
        result["output.weight"] = (65536, 768)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-head", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("output or manifest exists; select a new path")
    source = args.model / "model.safetensors"
    if sha256(source) != SOURCE_SHA256 or sha256(args.model / "config.json") != CONFIG_SHA256:
        parser.error("pinned source/config SHA-256 mismatch")
    tensors = {}
    with safe_open(source, framework="numpy") as model:
        for name, shape in inventory(args.include_head).items():
            w = model.get_tensor(name)
            if w.dtype != np.float32 or w.shape != shape:
                raise ValueError(f"invalid source tensor {name}: {w.shape}/{w.dtype}")
            codes, scales = encode(w)
            tensors[name + ".__w8_codes"] = codes
            tensors[name + ".__w8_scales"] = scales
    metadata = {"format": FORMAT, "source_sha256": SOURCE_SHA256, "model_revision": REVISION,
                "include_head": str(args.include_head).lower(), "group_size": "64",
                "scale_dtype": "f32", "rounding": "ties_to_even", "activation_dtype": "f32"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        parser.error(f"stale temporary output: {temporary}")
    save_file(tensors, temporary, metadata=metadata)
    temporary.replace(args.output)
    manifest = {**metadata, "artifact_sha256": sha256(args.output),
                "payload_bytes": sum(t.nbytes for t in tensors.values()), "file_bytes": args.output.stat().st_size,
                "tensor_count": len(tensors), "calibration": "absmax RTN; not activation-calibrated",
                "quality_qualified": False, "requires_pinned_source_model_at_runtime": True}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
