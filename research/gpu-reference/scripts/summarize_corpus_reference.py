#!/usr/bin/env python3
"""Freeze a compact, auditable summary from completed full-page GPU records."""
import collections
import json
import pathlib
import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=pathlib.Path, default=pathlib.Path("artifacts/reference/corpus-smoke-fp32-4096"))
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/corpus-smoke-lock-v1.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference/gpu-corpus-smoke-fp32-summary.json"))
    args = parser.parse_args()
    folder = args.folder
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert run["configuration"]["manifest_sha256"] == sha256(args.manifest)
    pages = [json.loads((folder / (pathlib.Path(page["canonical_path"]).parent.name + ".json")).read_text(encoding="utf-8")) for page in manifest["pages"]]
    assert len(pages) == run["pages"] == len(manifest["pages"])
    assert all(page["configuration"] == run["configuration"] for page in pages)
    qualification = "Corrected frozen evaluation corpus. CER is assembled-ground-truth diagnostic, not official OmniDocBench scoring. CPU parity, structured quality metrics and performance qualification remain separate."
    if args.manifest.name == "corpus-smoke-lock-v1.json":
        qualification = "Qualitative v1 smoke only:24pages, limited visual review, notebook-family calibration overlap. CER is assembled-ground-truth diagnostic, not official OmniDocBench scoring. No8k-output qualification or standalone performance claim."
    compact = {"schema_version": 1, "configuration": run["configuration"], "environment": run["environment"],
               "source_run": str(folder / "summary.json"), "source_run_sha256": sha256(folder / "summary.json"),
               "pages": len(pages), "generated_tokens": sum(len(p["token_ids"]) for p in pages),
               "natural_max_output_tokens": max((len(p["token_ids"]) for p in pages if p["finish_reason"] == "eos"), default=None),
               "over_2048_output_pages": sum(len(p["token_ids"]) > 2048 for p in pages),
               "over_4096_output_pages": sum(len(p["token_ids"]) > 4096 for p in pages),
               "finish_reasons": dict(collections.Counter(p["finish_reason"] for p in pages)),
               "category_counts": dict(collections.Counter(p["category"] for p in pages)),
               "diagnostic_micro_cer": run["diagnostic_micro_cer"],
               "qualification": qualification,
               "zero_margin_decisions": sum(d["winner_margin"] == 0 for page in pages for d in page["logit_decisions"]),
               "page_records": [{"sample_id": p["sample_id"], "category": p["category"], "output_tokens": len(p["token_ids"]),
                                 "prefix_tokens": p["prefix_length"], "finish_reason": p["finish_reason"],
                                 "record_sha256": sha256(folder / (p["sample_id"] + ".json"))}
                                for p in sorted(pages, key=lambda p: (-len(p["token_ids"]), p["sample_id"]))]}
    args.output.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: compact[k] for k in ["pages", "generated_tokens", "natural_max_output_tokens", "over_2048_output_pages", "finish_reasons"]}))


if __name__ == "__main__":
    main()
