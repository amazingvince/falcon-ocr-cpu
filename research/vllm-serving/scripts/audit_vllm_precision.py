#!/usr/bin/env python3
"""Inspect compiled Triton PTX from this fresh, single-request serving container."""
import hashlib
import json
import os
import pathlib
import re

import triton

cache = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", "/root/.triton/cache"))
records = []
for path in sorted(cache.rglob("*.ptx")):
    raw = path.read_bytes()
    relative = path.relative_to(cache)
    preserved = pathlib.Path("/out/ptx") / relative
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_bytes(raw)
    content = raw.decode()
    entries = re.findall(r"\.entry\s+([\w$]+)", content)
    tf32 = [line.strip() for line in content.splitlines() if ".tf32" in line and not line.strip().startswith("//")]
    records.append({"relative_path": str(relative), "preserved_path": str(pathlib.Path("ptx") / relative),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw), "entries": entries, "tf32_instruction_count": len(tf32), "tf32_examples": tf32[:3]})
attention = [r for r in records if any("attention" in name.lower() for name in r["entries"])]
report = {"triton_version": triton.__version__, "TRITON_F32_DEFAULT": os.environ["TRITON_F32_DEFAULT"],
          "runtime_fp32_default_knob": triton.knobs.language.fp32_default,
          "ptx_files": len(records), "attention_ptx_files": len(attention),
          "attention_contains_tf32_instructions": any(r["tf32_instruction_count"] for r in attention),
          "records": records,
          "qualification": "PTX evidence for compiled Triton kernels from the served request; CUDA-library GEMMs are governed separately by highest matmul precision and NVIDIA_TF32_OVERRIDE=0."}
assert report["runtime_fp32_default_knob"] == "ieee"
assert attention, "No compiled attention PTX found: precision remains unverified"
assert not report["attention_contains_tf32_instructions"]
pathlib.Path("/out/precision-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
