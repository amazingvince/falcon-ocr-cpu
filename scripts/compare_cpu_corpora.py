#!/usr/bin/env python3
"""Compare selected CPU outputs across platforms; validate text replay lineage.

No inference runs here. Timing, OS, threads and source versions may differ.
The old run contracts are never rewritten to pretend they used a subset lock.
"""
import argparse
import json
import pathlib

from fetch_reference import sha256
from corpus_comparison import (cpu_build_evidence, exact_output, manifest_scope,
    model_asset_evidence, read_object, read_snapshot, sample_key, semantic_contract,
    validate_cpu_result, validate_page_assets)
from validate_text_replay import (REPLAY_QUALIFICATION, canonical_sha256,
    validate_page_replay, validate_run_replay)


def compare(args):
    checks = []
    def check(name, actual, expected):
        checks.append({"name": name, "passed": actual == expected, "actual": actual, "expected": expected})
    runs, contracts, run_hashes = {}, {}, {}
    for side in ["left", "right"]:
        path = getattr(args, side) / "run.json"
        runs[side], run_hashes[side] = read_snapshot(path)
        contracts[side] = runs[side]["contract"]
    selected, scope = manifest_scope(args.manifest, {
        side: (getattr(args, side + "_manifest", None), contracts[side].get("manifest_sha256")) for side in runs}, check)
    assets = model_asset_evidence(args.model, check)
    context, evidence = {}, {}
    for side in runs:
        run, contract = runs[side], contracts[side]
        def side_check(name, actual, expected, side=side):
            check(side + "." + name, actual, expected)
        check(side + ".contract_sha256", canonical_sha256(contract), run.get("contract_sha256"))
        check(side + ".teacher_forced", run.get("teacher_forced"), False)
        context[side] = validate_run_replay(run, contract, side_check)
        evidence[side] = semantic_contract(contract, side, args.precision, check)
        evidence[side]["build"] = cpu_build_evidence(getattr(args, side + "_build", None), contract, check, side)
        evidence[side]["text_replay"] = context[side] is not None
        if context[side] is not None:
            evidence[side]["text_replay_qualification"] = REPLAY_QUALIFICATION
            evidence[side]["token_inference_source_run"] = contract["postprocessing_replay"].get("source_run_path")
    for key in ["options", "backend"]:
        check("platform_contract." + key, contracts["left"].get(key), contracts["right"].get(key))
    rows, missing, failed = [], {s: [] for s in runs}, {s: [] for s in runs}
    for page in selected["pages"]:
        sample = sample_key(page)
        validate_page_assets(page, check)
        records, hashes = {}, {}
        before = len(checks)
        for side in runs:
            path = getattr(args, side) / (sample + ".json")
            if not path.exists():
                missing[side].append({"sample_id": sample, "id": page["id"]})
                continue
            def side_check(name, actual, expected, side=side):
                check(side + "." + name, actual, expected)
            try:
                record, digest = read_snapshot(path)
            except (OSError, ValueError) as error:
                failed[side].append({"sample_id": sample, "id": page["id"], "error": str(error)})
                continue
            validate_page_replay(record, context[side], side_check, sample)
            for key, expected in [("id", page["id"]), ("category", page["category"]),
                                  ("contract_sha256", runs[side]["contract_sha256"]),
                                  ("input_sha256", page["canonical_png_sha256"]),
                                  ("ground_truth_sha256", page["ground_truth_sha256"])]:
                check(side + "." + sample + "." + key, record.get(key), expected)
            valid = "error" not in record and validate_cpu_result(record.get("result"), contracts[side], side_check, sample + ".result")
            if not valid:
                failed[side].append({"sample_id": sample, "id": page["id"], "record_sha256": digest,
                                     "error": record.get("error", "Missing or malformed inference result")})
                continue
            hashes[side] = digest
            records[side] = record["result"]
        if len(records) != 2:
            continue
        matches = exact_output(records["left"], records["right"])
        left_ids, right_ids = records["left"]["token_ids"], records["right"]["token_ids"]
        divergence = next((i for i, (a, b) in enumerate(zip(left_ids, right_ids)) if a != b), None)
        if divergence is None and len(left_ids) != len(right_ids):
            divergence = min(len(left_ids), len(right_ids))
        rows.append({"sample_id": sample, "id": page["id"], "category": page["category"],
                     **matches, "exact": all(matches.values()),
                     "provenance_passed": all(c["passed"] for c in checks[before:]),
                     "record_sha256": hashes, "left_output_tokens": len(left_ids), "right_output_tokens": len(right_ids),
                     "left_dimensions": [records["left"]["width"], records["left"]["height"]],
                     "right_dimensions": [records["right"]["width"], records["right"]["height"]],
                     "first_token_divergence": divergence})
    # Detect a run file being finalized or replaced during this snapshot.
    for side in runs:
        check(side + ".run_snapshot_unchanged", sha256(getattr(args, side) / "run.json"), run_hashes[side])
    complete = not any(missing.values()) and not any(failed.values()) and len(rows) == len(selected["pages"])
    provenance = all(c["passed"] for c in checks)
    exact = sum(r["exact"] for r in rows)
    passed = complete and provenance and exact == len(selected["pages"])
    report = {"schema_version": 1, "comparison_kind": "cpu_to_cpu_platform_outputs", "comparison_scope": scope,
              "status": "failed" if any(failed.values()) or not provenance or exact != len(rows) else ("complete" if complete else "partial"),
              "expected_pages": len(selected["pages"]), "compared_pages": len(rows), "exact_pages": exact,
              "compared_tokens": sum(r["left_output_tokens"] for r in rows), "missing_pages": missing, "failed_pages": failed,
              "selected_pages_complete": complete, "provenance_passed": provenance, "completed_output_parity_passed": passed,
              "run_files": {side: {"path": str(getattr(args, side) / "run.json"), "sha256": run_hashes[side]} for side in runs},
              "contracts": contracts, "contract_evidence": evidence, "checked_model_assets": assets,
              "startup_semantic_fields_complete": all(not item["missing_startup_fields"] for item in evidence.values()),
              "checks": checks, "pages": rows,
              "qualification": "Actual saved token IDs, literal text, stop reasons, counts and image dimensions on selected pages only. OS, thread counts, build/source versions and measured timing values may differ. Replay text remains derived from original inference. No hidden-tensor, OCR-quality, performance or full-parent-corpus qualification is implied."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True, help="Selected pages in comparison order")
    for side in ["left", "right"]:
        parser.add_argument("--" + side, type=pathlib.Path, required=True, help="CPU run directory")
        parser.add_argument("--" + side + "-manifest", type=pathlib.Path, help="Explicit original run manifest; default requires the selected manifest hash")
        parser.add_argument("--" + side + "-build", type=pathlib.Path, help="Optional preserved build.json, checked against startup contract")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("artifacts/model"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh report path")
    report = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "expected_pages", "compared_pages", "exact_pages", "compared_tokens", "provenance_passed", "completed_output_parity_passed"]}, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
