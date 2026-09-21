#!/usr/bin/env python3
"""Freeze file-identity checks for the completed three-page serving references."""
import json
import pathlib

from fetch_reference import sha256

folder = pathlib.Path("artifacts/reference/vllm-fullpages-fp32-4096")
target = pathlib.Path("reference/serving-fullpages-evidence-v1.json")
if target.exists():
    raise ValueError("Preserve the prior serving verification")


def check(value, message):
    if not value:
        raise ValueError(message)


manifest_path = pathlib.Path("reference/serving-fullpages-v1-lock.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
report_path = pathlib.Path("reference/vllm-fullpages-fp32.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
official_path = pathlib.Path("reference/official-fullpages-fp32.json")
official = json.loads(official_path.read_text(encoding="utf-8"))
check(len(manifest["pages"]) == len(report["pages"]) == len(official["pages"]) == 3, "Incomplete three-page evidence")
check(report["fullpages_passed"] and official["fullpages_passed"] and not report["precision_audit_pending"], "Serving gates incomplete")
check(report["manifest_sha256"] == official["configuration"]["manifest_sha256"] == sha256(manifest_path), "Manifest identity differs")
check(sha256(folder / "server.log") == report["server_log_sha256"], "Final server log changed")
capture_path = folder / "harness-source-capture/capture.json"
capture = json.loads(capture_path.read_text(encoding="utf-8"))
check(sha256(folder / "environment.json") == capture["environment_sha256"], "Environment changed")
for name, metadata in capture["sources"].items():
    file = folder / "harness-source-capture" / name
    check(sha256(file) == metadata["sha256"] and file.stat().st_size == metadata["bytes"], "Harness source changed: " + name)
for metadata in report["installed_source_files"]:
    file = folder / "sources" / metadata["path"]
    check(sha256(file) == metadata["sha256"] and file.stat().st_size == metadata["bytes"], "Installed source changed")
audit_path = folder / "precision-audit.json"
check(sha256(audit_path) == report["compiled_precision_evidence"]["record_sha256"], "Precision audit changed")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
for metadata in audit["records"]:
    file = folder / metadata["preserved_path"]
    check(sha256(file) == metadata["sha256"] and file.stat().st_size == metadata["bytes"], "Compiled PTX changed")
page_hashes, total_tokens = {}, 0
for page, row, official_row in zip(manifest["pages"], report["pages"], official["pages"]):
    sample = pathlib.Path(page["canonical_path"]).parent.name
    check(sample == row["sample_id"] == official_row["sample_id"], "Page order differs")
    record_path = folder / (sample + ".json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for suffix, field in [("-request.json", "request_sha256"), ("-response.json", "response_sha256")]:
        check(sha256(folder / (sample + suffix)) == record[field] == row[field], "HTTP evidence changed")
    golden_path = pathlib.Path("artifacts/reference/corpus-v3-fp32-4096") / (sample + ".json")
    check(sha256(golden_path) == record["golden_record_sha256"] == official_row["golden_record_sha256"], "Golden record changed")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    check(record["effective_token_ids_with_reported_stop"] == golden["token_ids"] and record["text"] == golden["text"], "Raw serving outputs differ")
    check(not record["reported_stop_appended"] and not record["unparsed_logprob_tokens"], "Unobserved serving token")
    check(record["usage"]["completion_tokens"] == len(golden["token_ids"]), "Output count differs")
    total_tokens += len(golden["token_ids"])
    page_hashes[sample] = sha256(record_path)
receipt = {"schema_version": 1, "status": "three_page_serving_evidence_verified", "pages": 3, "matching_token_ids": total_tokens,
           "vllm_report_sha256": sha256(report_path), "official_report_sha256": sha256(official_path),
           "manifest_sha256": sha256(manifest_path), "harness_capture_sha256": sha256(capture_path),
           "harness_capture_phase": capture["phase"], "installed_source_files": len(report["installed_source_files"]),
           "preserved_ptx_files": len(audit["records"]), "page_record_sha256": page_hashes, "script_sha256": sha256(__file__),
           "qualification": "Three-page output and preserved-file verification. Historical corpus startup limitations remain; copied harness sources are attested only at their recorded capture phase. No hidden-state, corpus-wide quality or performance qualification."}
with target.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
