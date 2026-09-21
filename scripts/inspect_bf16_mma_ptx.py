#!/usr/bin/env python3
"""Read existing Triton cache artifacts; never initialize CUDA or execute code."""
import collections
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
cache = Path("/home/amazi/falcon-ocr-rust-reference/cache/triton")
keys = ["DFQ7SOOKEVMW4ZDWRYRCN7HK4BXCKWGDN6JK7QY4D5QFYJ4KOYRA",
        "FUO54TKCDVFRKAWX6ZKL5V74LBTXCFBF4DBDYNI64DGVKFZP77JQ",
        "NUMXHWI6FORGSZJOV2LELDJQQ3KZBBRVQ3VZCPH25KZHKSJEJKIQ"]
out = root / "artifacts/reference/bf16-mma-ptx"
out.mkdir(parents=True, exist_ok=True)
records = []
for key in keys:
    base = cache / key / "triton_tem_fused_flex_attention_0"
    ptx_bytes = base.with_suffix(".ptx").read_bytes()
    ptx = ptx_bytes.decode()
    metadata = json.loads(base.with_suffix(".json").read_text())
    instructions = []
    for number, line in enumerate(ptx.splitlines(), 1):
        match = re.search(r"(mma\.sync\.\S+)\s*\{([^}]+)\}.*;", line)
        if match:
            instructions.append({"line": number, "opcode": match[1], "destination": match[2].strip(), "text": line.strip()})
    first = instructions[0]["destination"]
    first_chain = [entry for entry in instructions if entry["destination"] == first][:4]
    copied = out / (key + ".ptx")
    copied.write_bytes(ptx_bytes)
    records.append({"cache_key": key, "original_ptx": str(base.with_suffix(".ptx")),
                    "ptx_copy": str(copied.relative_to(root)), "ptx_sha256": hashlib.sha256(ptx_bytes).hexdigest(),
                    "target": metadata["target"], "triton_version": metadata["triton_version"],
                    "num_warps": metadata["num_warps"],
                    "opcodes": dict(collections.Counter(x["opcode"] for x in instructions)),
                    "first_accumulator_chain": first_chain})
report = {"schema_version": 1, "gpu_execution": False, "kernel_records": records,
          "interpretation": "Cached BF16 Flex kernels target sm89 and use dense m16n8k16 BF16-to-F32 MMA with four chained instructions per 64-wide dot. PTX does not expose the internal sum order or rounding of each MMA instruction."}
target = root / "reference/bf16-mma-ptx-inventory.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps([{k: v for k, v in record.items() if k in ["cache_key", "ptx_sha256", "target", "opcodes"]} for record in records], indent=2))
