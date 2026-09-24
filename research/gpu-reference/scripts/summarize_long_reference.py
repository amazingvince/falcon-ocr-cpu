#!/usr/bin/env python3
"""Freeze observed output/context boundaries without treating synthetic text as quality evidence."""
import json
import pathlib
import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
args = parser.parse_args()
folder = pathlib.Path(f"artifacts/reference/long-numeric-{args.precision}-8192")
page_path = folder / "long-numeric-v1.json"
page = json.loads(page_path.read_text(encoding="utf-8"))
run = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
assert run["status"] == "complete" and run["pages"] == 1
assert page["configuration"]["max_new_tokens"] == 8192
assert page["configuration"]["precision"] == args.precision
assert page["prefix_length"] + len(page["token_ids"]) <= 16384
summary = {"schema_version": 1, "configuration": page["configuration"], "environment": run["environment"],
           "id": page["id"], "source_record_sha256": sha256(page_path), "source_summary_sha256": sha256(folder / "summary.json"),
           "canonical_rgb_sha256": page["canonical_rgb_sha256"], "prefix_tokens": page["prefix_length"],
           "output_tokens": len(page["token_ids"]), "total_prompt_and_output_tokens": page["prefix_length"] + len(page["token_ids"]),
           "allocated_cache_capacity": page["cache_capacity"], "finish_reason": page["finish_reason"],
           "reached_8192_output_boundary": len(page["token_ids"]) == 8192,
           "diagnostic_cer": page["diagnostic_cer"], "peak_gpu_allocated_bytes": page["peak_gpu_allocated_bytes"],
           "elapsed_seconds_including_compile_and_diagnostics": page["elapsed_seconds"],
           "qualification": "Actual 8192-budget GPU execution on an original deterministic numerical stress page. Consult finish_reason and reached_8192_output_boundary for observed behavior. Poor transcription and synthetic data make this boundary evidence only, not natural OCR quality or a performance claim. CPU parity remains separate."}
target = pathlib.Path(f"reference/gpu-long-numeric-{args.precision}-summary.json")
target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output_tokens": summary["output_tokens"], "total_prompt_and_output_tokens": summary["total_prompt_and_output_tokens"], "finish_reason": summary["finish_reason"]}))
