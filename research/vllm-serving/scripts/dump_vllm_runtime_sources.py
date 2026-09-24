#!/usr/bin/env python3
"""Preserve the specific installed serving model/backend sources from its image."""
import hashlib
import json
import pathlib

root = pathlib.Path("/usr/local/lib/python3.10/dist-packages/vllm")
paths = [root / "model_executor/models/falcon_ocr.py", root / "model_executor/models/falcon_ocr_multimodal.py",
         root / "transformers_utils/configs/falcon_ocr.py",
         root / "v1/attention/ops/triton_unified_attention.py", root / "v1/attention/ops/triton_decode_attention.py",
         root / "entrypoints/openai/chat_completion/protocol.py", root / "platforms/interface.py", root / "envs.py"]
paths += list(root.glob("v1/attention/backends/triton*.py"))
records = []
for source in paths:
    data = source.read_bytes()
    relative = source.relative_to(root)
    target = pathlib.Path("/out/sources") / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    records.append({"path": str(relative), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
pathlib.Path("/out/sources.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
