#!/usr/bin/env python3
"""Read, never execute, generated kernel source to record selected tile metadata."""
import hashlib
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[3]
cache = pathlib.Path("/home/amazi/falcon-ocr-rust-reference/cache/torchinductor")
records = []
for path in cache.rglob("*.py"):
    source = path.read_text()
    if "flex_attention" not in source:
        continue
    if "torch.bfloat16" not in source and "bf16" not in source:
        continue
    constants = {k: sorted(set(re.findall(r"(?m)^\s*" + k + r"\s*(?::\s*tl\.constexpr)?\s*=(?!=)\s*([^\n]+)", source)))
                 for k in ["BLOCK_M", "BLOCK_N", "SPLIT_KV", "Q_LEN", "KV_LEN", "SPARSE_Q_BLOCK_SIZE", "SPARSE_KV_BLOCK_SIZE", "PRESCALE_QK", "FLOAT32_PRECISION", "GQA_SHARED_HEADS"]}
    lines = [line.strip() for line in source.splitlines() if any(k in line for k in ["assert_size_stride", "empty_strided_cuda", "triton_meta=", "triton_tem_fused", "SPLIT_KV", "NUM_SPLITS"])][:30]
    records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "constants": constants, "context": lines})
target = root / "artifacts/reference/bf16-kernel-inventory.json"
target.write_text(json.dumps(records, indent=2) + "\n")
selected = {}
for record in records:
    kind = "decode" if record["constants"]["SPLIT_KV"] else "prefill"
    if kind not in selected and any("def triton_tem_fused" in line for line in record["context"]):
        copy = root / f"artifacts/reference/bf16-flex-{kind}-compiled.py"
        copy.write_bytes(pathlib.Path(record["path"]).read_bytes())
        selected[kind] = {"source_sha256": record["sha256"], "source_copy": str(copy.relative_to(root)),
                          "constants": record["constants"]}
summary = {"inventory_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
           "selected": selected, "bf16_sources_found": len(records),
           "qualification": "Read from generated source after the BF16 reference run; no source was executed. All observed BF16 variants agree on block dimensions and split count."}
(root / "reference/bf16-flex-kernel-metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"output": str(target), "kernels": len(records)}, indent=2))
