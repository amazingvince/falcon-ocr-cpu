#!/usr/bin/env python3
"""Freeze actual output count and cache cursor at the requested context boundary."""
import json
import pathlib

from fetch_reference import sha256
from validate_gpu_reference_record import validate_gpu_reference_record


def check(name, actual, expected):
    if actual != expected:
        raise ValueError(f"{name}: {actual!r} != {expected!r}")


folder = pathlib.Path("artifacts/reference/exact-context-boundary-fp32-8284")
target = pathlib.Path("reference/gpu-exact-context-boundary-fp32-summary.json")
receipt_path = pathlib.Path("reference/gpu-exact-context-boundary-fp32-validation.json")
if target.exists():
    raise ValueError("Preserve the prior boundary summary")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
page_path = folder / "long-numeric-v1.json"
page = json.loads(page_path.read_text(encoding="utf-8"))
check("validated record", sha256(page_path), receipt["record_sha256"]["long-numeric-v1"])
check("validated run", sha256(folder / "run.json"), receipt["run_sha256"])
check("prefix", page["prefix_length"], 8100)
check("cap", page["configuration"]["max_new_tokens"], 8284)
check("precision", page["configuration"]["precision"], "fp32")
validate_gpu_reference_record(page, page["configuration"], check, "boundary")
emitted = len(page["token_ids"])
cursor = page["final_kv_cache_position"]
check("actual cursor follows consumed tokens", cursor, page["prefix_length"] + emitted - 1)
check("cursor is an integer", type(cursor), int)
prior_path = pathlib.Path("artifacts/reference/long-numeric-fp32-8192/long-numeric-v1.json")
prior = json.loads(prior_path.read_text(encoding="utf-8"))
validate_gpu_reference_record(prior, prior["configuration"], check, "prior8192")
for key in ["model_revision", "weights_sha256", "precision", "tf32", "min_dimension", "max_dimension"]:
    check("prior." + key, prior["configuration"][key], page["configuration"][key])
check("prior image", prior["canonical_rgb_sha256"], page["canonical_rgb_sha256"])
check("prior prefix", prior["prefix_length"], page["prefix_length"])
reached = emitted == 8284 and page["finish_reason"] == "length"
summary = {"schema_version": 1, "status": "observed_context_boundary" if reached else "ended_before_requested_boundary",
           "configuration": page["configuration"], "source_record_sha256": sha256(page_path),
           "validation_receipt_sha256": sha256(receipt_path), "prior_8192_record_sha256": sha256(prior_path),
           "prefix_tokens": page["prefix_length"], "requested_output_cap": 8284, "emitted_tokens": emitted,
           "finish_reason": page["finish_reason"], "prompt_plus_emitted_tokens": page["prefix_length"] + emitted,
           "allocated_cache_capacity": page["cache_capacity"], "observed_final_kv_cache_position": cursor,
           "reached_requested_total_16384": reached, "final_emitted_token_is_unconsumed": True,
           "prior_8192_tokens_exact": emitted >= len(prior["token_ids"]) and page["token_ids"][:len(prior["token_ids"])] == prior["token_ids"],
           "script_sha256": sha256(__file__),
           "qualification": "Observed free-greedy GPU execution on the fixed original numeric stress image. Actual emitted tokens, stop reason and observed cache cursor are separate from requested capacity. CPU parity remains separate; synthetic transcription and concurrent execution do not establish natural OCR quality or performance."}
with target.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "configuration"}, indent=2))
