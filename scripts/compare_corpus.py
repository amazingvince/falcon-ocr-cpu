#!/usr/bin/env python3
"""Compare independently greedy CPU/GPU corpus outputs after checking provenance.

Partial runs are valid snapshots: missing pages remain explicit and cannot pass
the completed-corpus gate. CER is an assembled-ground-truth diagnostic only.
"""
import argparse
import hashlib
import json
import pathlib
import unicodedata

from rapidfuzz.distance import Levenshtein

from fetch_reference import sha256
from validate_text_replay import REPLAY_QUALIFICATION, validate_run_replay, validate_page_replay
from validate_gpu_reference_record import validate_gpu_reference_record
from corpus_comparison import (cpu_build_evidence, manifest_scope, model_asset_evidence,
    read_object, read_snapshot, sample_key, semantic_contract, validate_cpu_result, validate_page_assets)


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/corpus-smoke-lock-v1.json"))
    parser.add_argument("--gpu", type=pathlib.Path, default=pathlib.Path("artifacts/reference/corpus-smoke-fp32-4096"))
    parser.add_argument("--cpu", type=pathlib.Path, default=pathlib.Path("artifacts/cpu/corpus-smoke-fp32-4096"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference/windows-rust-corpus-smoke-fp32.json"))
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--gpu-manifest", type=pathlib.Path, help="Explicit original GPU manifest; selected pages must be exact members")
    parser.add_argument("--cpu-manifest", type=pathlib.Path, help="Explicit original CPU manifest; selected pages must be exact members")
    parser.add_argument("--cpu-build", type=pathlib.Path, help="Optional preserved build.json checked against CPU startup contract")
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("artifacts/model"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh comparison report path")
    gpu_run, gpu_run_sha = read_snapshot(args.gpu / "run.json")
    cpu_run, cpu_run_sha = read_snapshot(args.cpu / "run.json")
    gpu_config, cpu_config = gpu_run["configuration"], cpu_run["contract"]
    checks = []

    def check(name, actual, expected):
        checks.append({"name": name, "passed": actual == expected, "actual": actual, "expected": expected})

    run_hashes = {"gpu": gpu_run_sha, "cpu": cpu_run_sha}
    manifest, scope = manifest_scope(args.manifest, {
        "gpu": (args.gpu_manifest, gpu_config.get("manifest_sha256")),
        "cpu": (args.cpu_manifest, cpu_config.get("manifest_sha256"))}, check)
    evidence = {"gpu": semantic_contract(gpu_config, "gpu", args.precision, check, gpu=True),
                "cpu": semantic_contract(cpu_config, "cpu", args.precision, check)}
    evidence["cpu"]["build"] = cpu_build_evidence(args.cpu_build, cpu_config, check, "cpu")
    assets = model_asset_evidence(args.model, check)
    for key in ["max_dimension", "min_dimension", "max_new_tokens"]:
        check("options." + key, cpu_config["options"][key], gpu_config[key])
    check("gpu.tf32", gpu_config["tf32"], False)
    check("gpu.flex_float32_precision", gpu_config["flex_float32_precision"], "ieee")
    if args.precision == "bf16":
        check("gpu.cast_target", gpu_config.get("cast_target"), "HF eager model-wide BF16, including golden frequencies, then regenerate temporal complex64")
    check("cpu.teacher_forced", cpu_run["teacher_forced"], False)
    serialized = json.dumps(cpu_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    check("cpu.contract_sha256", hashlib.sha256(serialized).hexdigest(), cpu_run["contract_sha256"])
    replay_context = validate_run_replay(cpu_run, cpu_config, check)
    rows, missing_gpu, missing_cpu, failed_gpu, failed_cpu = [], [], [], [], []
    for page in manifest["pages"]:
        sample = sample_key(page)
        gpu_path, cpu_path = args.gpu / (sample + ".json"), args.cpu / (sample + ".json")
        if not gpu_path.exists():
            missing_gpu.append(sample)
        if not cpu_path.exists():
            missing_cpu.append(sample)
        def record(path):
            if not path.exists():
                return None, None
            try:
                return read_snapshot(path)
            except (OSError, ValueError) as error:
                return {"error": "Unreadable inference record: " + str(error)}, None
        (gpu, gpu_sha), (cpu, cpu_sha) = record(gpu_path), record(cpu_path)
        before = len(checks)
        cpu_valid = True
        if cpu is not None:
            validate_page_replay(cpu, replay_context, check, sample)
            if "error" not in cpu:
                cpu_valid = validate_cpu_result(cpu.get("result"), cpu_config, check, sample + ".cpu_result")
        cpu_error = cpu is not None and ("error" in cpu or not cpu_valid or not isinstance(cpu.get("result"), dict))
        gpu_error = gpu is not None and not validate_gpu_reference_record(gpu, gpu_config, check, sample + ".gpu_result")
        if gpu is not None:
            check(sample + ".gpu_not_replay", "postprocessing_replay" not in gpu and not gpu.get("derived_text_replay", False), True)
            check(sample + ".gpu_free_running_if_recorded", gpu.get("teacher_forced", False), False)
        if cpu_error:
            check(sample + ".failed_cpu_id", cpu.get("id"), page["id"])
            check(sample + ".failed_cpu_contract", cpu.get("contract_sha256"), cpu_run["contract_sha256"])
            check(sample + ".failed_cpu_input", cpu.get("input_sha256"), page["canonical_png_sha256"])
            failed_cpu.append({"sample_id": sample, "id": page["id"], "category": page["category"],
                               "record_sha256": cpu_sha, "error": cpu.get("error", "Missing or malformed inference result")})
        if gpu_error:
            check(sample + ".failed_gpu_id", gpu.get("id"), page["id"])
            failed_gpu.append({"sample_id": sample, "id": page["id"], "category": page["category"],
                               "record_sha256": gpu_sha, "error": gpu.get("error", "Missing or malformed GPU inference result")})
        if gpu is None or cpu is None or cpu_error or gpu_error:
            continue
        validate_page_assets(page, check)
        check(sample + ".gpu_configuration", gpu["configuration"], gpu_config)
        check(sample + ".cpu_contract", cpu["contract_sha256"], cpu_run["contract_sha256"])
        check(sample + ".gpu_id", gpu["id"], page["id"])
        check(sample + ".cpu_id", cpu["id"], page["id"])
        check(sample + ".gpu_category", gpu.get("category"), page["category"])
        check(sample + ".cpu_category", cpu.get("category"), page["category"])
        check(sample + ".gpu_sample_id", gpu.get("sample_id"), sample)
        check(sample + ".gpu_input", gpu["canonical_rgb_sha256"], page["rgb_sha256"])
        check(sample + ".cpu_input", cpu["input_sha256"], page["canonical_png_sha256"])
        check(sample + ".cpu_ground_truth", cpu["ground_truth_sha256"], page["ground_truth_sha256"])
        check(sample + ".cpu_teacher_forced", cpu["result"]["teacher_forced"], False)
        check(sample + ".prefix_tokens", cpu["result"]["input_tokens"], gpu["prefix_length"])
        if "width" in gpu or "height" in gpu:
            check(sample + ".processed_dimensions", [cpu["result"]["width"], cpu["result"]["height"]], [gpu.get("width"), gpu.get("height")])
        gt = normalized(pathlib.Path(page["ground_truth_path"]).read_text(encoding="utf-8"))
        gpu_text, cpu_text = gpu["text"], cpu["result"]["text"]
        gpu_ids, cpu_ids = gpu["token_ids"], cpu["result"]["token_ids"]
        mismatches = [i for i, (g, c) in enumerate(zip(gpu_ids, cpu_ids)) if g != c]
        first = mismatches[0] if mismatches else (min(len(gpu_ids), len(cpu_ids)) if len(gpu_ids) != len(cpu_ids) else None)
        gpu_dist = Levenshtein.distance(gt, normalized(gpu_text))
        cpu_dist = Levenshtein.distance(gt, normalized(cpu_text))
        check(sample + ".gpu_diagnostic_distance", gpu["diagnostic_character_edit_distance"], gpu_dist)
        row = {"sample_id": sample, "id": page["id"], "category": page["category"],
               "provenance_passed": all(c["passed"] for c in checks[before:]),
               "gpu_record_sha256": gpu_sha, "cpu_record_sha256": cpu_sha,
               "gpu_tokens": len(gpu_ids), "cpu_tokens": len(cpu_ids), "tokens_exact": gpu_ids == cpu_ids,
               "text_exact": gpu_text == cpu_text, "finish_reason_exact": gpu["finish_reason"] == cpu["result"]["finish_reason"],
               "gpu_finish_reason": gpu["finish_reason"], "cpu_finish_reason": cpu["result"]["finish_reason"],
               "cpu_dimensions": [cpu["result"]["width"], cpu["result"]["height"]],
               "processed_dimension_comparison": "compared" if "width" in gpu and "height" in gpu else "GPU record has prefix count only; processed width/height unavailable",
               "first_token_divergence": first, "matching_prefix_tokens": first if first is not None else len(gpu_ids),
               "diagnostic_ground_truth_characters": len(gt), "gpu_diagnostic_edit_distance": gpu_dist,
               "cpu_diagnostic_edit_distance": cpu_dist, "gpu_diagnostic_cer": gpu_dist / max(1, len(gt)),
               "cpu_diagnostic_cer": cpu_dist / max(1, len(gt))}
        if first is not None:
            row["first_divergence_gpu_decision"] = gpu["logit_decisions"][first] if first < len(gpu_ids) else None
            row["first_divergence_cpu_token"] = cpu_ids[first] if first < len(cpu_ids) else None
        rows.append(row)
    for side, directory in [("gpu", args.gpu), ("cpu", args.cpu)]:
        check(side + ".run_snapshot_unchanged", sha256(directory / "run.json"), run_hashes[side])
    provenance = all(c["passed"] for c in checks)
    complete = not missing_cpu and not missing_gpu
    matches = sum(r["tokens_exact"] and r["text_exact"] and r["finish_reason_exact"] for r in rows)
    failed = bool(failed_cpu or failed_gpu)
    report = {"schema_version": 2, "status": "failed" if failed or not provenance or matches != len(rows) else ("complete" if complete else "partial"),
              "manifest": str(args.manifest), "manifest_sha256": scope["selected_manifest"]["sha256"],
              "comparison_scope": scope, "contract_evidence": evidence, "checked_model_assets": assets,
              "startup_semantic_fields_complete": all(not item["missing_startup_fields"] for item in evidence.values()),
              "gpu_run": str(args.gpu / "run.json"), "gpu_run_sha256": run_hashes["gpu"],
              "cpu_run": str(args.cpu / "run.json"), "cpu_run_sha256": run_hashes["cpu"],
              "cpu_contract": cpu_config, "gpu_configuration": gpu_config,
              "expected_pages": len(manifest["pages"]), "compared_pages": len(rows), "exact_pages": matches,
              "compared_gpu_tokens": sum(r["gpu_tokens"] for r in rows), "missing_cpu": missing_cpu, "missing_gpu": missing_gpu,
              "missing_cpu_count": len(missing_cpu), "missing_gpu_count": len(missing_gpu),
              "failed_cpu_pages": failed_cpu, "failed_gpu_pages": failed_gpu,
              "failed_cpu_count": len(failed_cpu), "failed_gpu_count": len(failed_gpu),
              "provenance_passed": provenance, "completed_smoke_parity_passed": complete and not failed and provenance and matches == len(manifest["pages"]),
              "checks": checks, "pages": rows,
              "qualification": "Independent free greedy output comparison on selected pages only, with explicitly listed historical startup-attestation gaps. Larger parent completion, tensor parity, OCR accuracy qualification, and bare-metal performance are separate gates.",
              "metric": "NFC+whitespace-normalized assembled-ground-truth character edit distance, diagnostic only, not official OmniDocBench scoring."}
    if args.manifest.name == "corpus-smoke-lock-v1.json":
        report["v1_split_limitation"] = "v1 treats pages of the same notes_<hash> notebook as separate families, causing evaluation/calibration overlap. It is qualitative smoke only and must not calibrate quantization. v2 metadata selection fixes known notebook family assignment."
    elif "qualification" in manifest:
        report["corpus_qualification"] = manifest["qualification"]
    report["completed_output_parity_passed"] = report["completed_smoke_parity_passed"]
    report["derived_text_replay"] = replay_context is not None
    if replay_context is not None:
        report["qualification"] = REPLAY_QUALIFICATION + " Tensor parity, OCR accuracy qualification and bare-metal performance remain separate gates."
        report["token_inference_source_run"] = cpu_config["postprocessing_replay"]["source_run_path"]
        report["token_inference_source_run_sha256"] = cpu_config["postprocessing_replay"]["source_run_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "compared_pages", "exact_pages", "compared_gpu_tokens", "provenance_passed", "completed_smoke_parity_passed", "missing_cpu_count", "missing_gpu_count", "failed_cpu_count", "failed_gpu_count"]}, indent=2))
    if not provenance:
        print(json.dumps([c for c in checks if not c["passed"]], indent=2))
        raise SystemExit(1)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
