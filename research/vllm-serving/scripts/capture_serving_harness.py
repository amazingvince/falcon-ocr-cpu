#!/usr/bin/env python3
"""Preserve the full-page serving harness after launch, before HTTP inference."""
import datetime
import json
import pathlib

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

folder = pathlib.Path("artifacts/reference/vllm-fullpages-fp32-4096")
environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
if any(folder.glob("*-request.json")):
    raise ValueError("This capture must precede inference requests")
if sha256("research/vllm-serving/scripts/vllm_reference_entry.py") != environment["entrypoint_sha256"]:
    raise ValueError("Current entrypoint differs from the observed startup hash")
target = folder / "harness-source-capture"
target.mkdir(exist_ok=False)
names = ["scripts/" + name for name in ["capture_serving_harness.py", "run_reference_vllm.sh", "podman_reference.sh",
         "vllm_reference_entry.py", "request_vllm_fullpages.py", "dump_vllm_runtime_sources.py",
         "audit_vllm_precision.py", "finalize_vllm_reference.py", "fetch_reference.py", "validate_gpu_reference_record.py"]]
names += ["reference/serving-fullpages-v1-lock.json", "reference/manifest.json", "artifacts/model/artifact-manifest.json"]
records = {}
for name in names:
    source = pathlib.Path(name)
    preserved = target / name
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_bytes(source.read_bytes())
    records[name] = {"sha256": sha256(preserved), "bytes": preserved.stat().st_size}
record = {"captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "phase": "after container startup environment capture, before first HTTP inference request",
          "environment_sha256": sha256(folder / "environment.json"), "sources": records,
          "qualification": "The copied entrypoint matches its observed startup digest. Other harness sources are captured at the stated phase; this is not retroactive pre-startup attestation."}
(target / "capture.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"sources": len(records), "capture_sha256": sha256(target / "capture.json")}))
