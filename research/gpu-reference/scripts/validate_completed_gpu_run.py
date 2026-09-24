#!/usr/bin/env python3
"""Validate and freeze a receipt for a completed new-style GPU corpus run."""
import argparse
import hashlib
import json
import pathlib

from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256
from reference_corpus_contract import read_snapshot, require_fresh_greedy, sample_key, selected_pages, validate_completed_page
from reference_run_identity import canonical_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--folder", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing validation receipts")
    manifest, manifest_hash = read_snapshot(args.manifest)
    run, run_hash = read_snapshot(args.folder / "run.json")
    summary, summary_hash = read_snapshot(args.folder / "summary.json")
    if run != summary or run["status"] != "complete":
        raise ValueError("Run and summary do not identify the same completed run")
    require_fresh_greedy(run, "run")
    config = run["configuration"]
    require_fresh_greedy(config, "configuration")
    pages = selected_pages(manifest, config["requested_limit"])
    if config["manifest_sha256"] != manifest_hash or config["selected_page_ids"] != [p["id"] for p in pages]:
        raise ValueError("Manifest/selection mismatch")
    startup_path = args.folder / "provenance/startup-identity.json"
    startup, startup_hash = read_snapshot(startup_path)
    if run["startup_identity"]["startup_identity_sha256"] != startup_hash:
        raise ValueError("Startup archive binding mismatch")
    identity = startup["identity"]
    digest = canonical_digest({k: v for k, v in identity.items() if k != "identity_sha256"})
    if digest != identity["identity_sha256"] or digest != config["runtime_identity_sha256"]:
        raise ValueError("Runtime identity mismatch")
    for key, value in identity["runtime"].items():
        if run["environment"][key] != value:
            raise ValueError("Recorded runtime mismatch: " + key)
    for name, expected in identity["source_files"].items():
        source = args.folder / "provenance/sources" / name
        if sha256(source) != expected["sha256"] or source.stat().st_size != expected["bytes"]:
            raise ValueError("Preserved source changed: " + name)
    records, record_hashes = [], {}
    for page in pages:
        sample = sample_key(page)
        record, record_hash = read_snapshot(args.folder / (sample + ".json"))
        validate_completed_page(record, page, config)
        for field, hash_field in [("canonical_path", "canonical_png_sha256"), ("ground_truth_path", "ground_truth_sha256")]:
            if sha256(page[field]) != page[hash_field]:
                raise ValueError("Input bytes changed: " + sample)
        with Image.open(page["canonical_path"]) as image:
            if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != page["rgb_sha256"]:
                raise ValueError("Decoded RGB bytes changed: " + sample)
        records.append(record)
        record_hashes[sample] = record_hash
    token_count = sum(len(record["token_ids"]) for record in records)
    eos_count = sum(record["finish_reason"] == "eos" for record in records)
    if run["pages"] != len(pages) or run["planned_pages"] != len(pages) or run["generated_tokens"] != token_count or run["eos_pages"] != eos_count:
        raise ValueError("Summary counts differ from saved records")
    length_ids = [record["sample_id"] for record in records if record["finish_reason"] == "length"]
    if run["length_limited_pages"] != length_ids:
        raise ValueError("Length-stop summary differs from saved records")
    if sha256(args.folder / "run.json") != run_hash or sha256(args.manifest) != manifest_hash:
        raise ValueError("Source changed while validating")
    report = {"schema_version": 1, "status": "complete_gpu_reference_validated", "pages": len(pages),
              "generated_tokens": token_count, "eos_pages": eos_count, "length_pages": len(length_ids),
              "configuration": config, "run_sha256": run_hash, "summary_sha256": summary_hash,
              "manifest_sha256": manifest_hash, "startup_identity_sha256": startup_hash,
              "runtime_identity_sha256": digest, "archived_source_file_count": len(identity["source_files"]),
              "record_sha256": record_hashes, "validation_script_sha256": sha256(__file__),
              "qualification": "Completed fresh greedy GPU records, input bytes and preserved startup identity validated. CPU/output parity, document quality and performance require separate evidence. Prepared image dimensions were not recorded by this driver; pixels, processing options and prefix lengths are bound."}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "pages", "generated_tokens", "eos_pages", "length_pages"]}))


if __name__ == "__main__":
    main()
