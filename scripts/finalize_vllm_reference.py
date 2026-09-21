#!/usr/bin/env python3
"""Attach post-request precision/source/log evidence to the durable serving report."""
import argparse
import json
import pathlib

from fetch_reference import sha256

parser = argparse.ArgumentParser()
parser.add_argument("--full-pages", action="store_true")
args = parser.parse_args()
folder = pathlib.Path("artifacts/reference/vllm-fullpages-fp32-4096" if args.full_pages else "artifacts/reference/vllm-smoke-fp32")
target = pathlib.Path("reference/vllm-fullpages-fp32.json" if args.full_pages else "reference/vllm-smoke-fp32.json")
report = json.loads(target.read_text(encoding="utf-8"))
audit = json.loads((folder / "precision-audit.json").read_text(encoding="utf-8"))
assert audit["attention_ptx_files"] > 0 and not audit["attention_contains_tf32_instructions"]
assert audit["runtime_fp32_default_knob"] == "ieee"
report["compiled_precision_evidence"] = {k: v for k, v in audit.items() if k != "records"}
report["compiled_precision_evidence"]["record_sha256"] = sha256(folder / "precision-audit.json")
report["installed_source_files"] = json.loads((folder / "sources.json").read_text(encoding="utf-8"))
report["server_log_sha256"] = sha256(folder / "server.log")
report["server_shutdown"] = "Project-owned reference container stopped after the completed request and precision audit."
if args.full_pages:
    report["precision_audit_pending"] = False
    report["qualification"] = "Three actual direct-vLLM full-page HTTP requests with IEEE compiled-attention PTX evidence. This is not corpus-wide serving, tensor, quality, or performance qualification."
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not args.full_pages:
    attempt_path = pathlib.Path("reference/vllm-attempts.json")
    attempts = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempts["attempts"] = [a for a in attempts["attempts"] if a["attempt"] != 2] + [{
        "attempt": 2, "status": "serving_smoke_passed" if report["smoke_passed"] else "serving_smoke_failed", "report": target.as_posix(),
        "observed_tokens": len(report["effective_token_ids_with_reported_stop"]),
        "prompt_tokens": report.get("usage", {}).get("prompt_tokens"), "text_exact": report["text_exact"],
        "compiled_triton_attention_ieee_verified": True, "runtime_model_math_patched": False}]
    attempt_path.write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
gate = "fullpages_passed" if args.full_pages else "smoke_passed"
print(json.dumps({"report": target.as_posix(), "sha256": sha256(target), gate: report[gate]}))
