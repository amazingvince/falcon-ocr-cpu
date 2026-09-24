#!/usr/bin/env python3
"""Join completed CPU/GPU comparisons and directly compare common CPU outputs.

This is saved-output evidence, not new inference, tensor parity or timing.
Existing comparisons supply provenance qualification; their historical gaps
are retained. Every selected record is rehashed before and after use.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    script = Path(__file__)
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-comparison", type=Path, required=True)
    parser.add_argument("--linux-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing join reports")
    bound = {str(script): script_sha}

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None:
            require(digest == expected, "Changed input: " + str(path))
        require(str(path) not in bound or bound[str(path)] == digest, "Input changed during join")
        bound[str(path)] = digest
        return json.loads(raw)

    reports, manifests, runs = {}, {}, {}
    for platform, path in [("windows", args.windows_comparison), ("linux", args.linux_comparison)]:
        report = read(path)
        require(report["status"] == "complete" and report["provenance_passed"] is True
                and report["completed_output_parity_passed"] is True, "Incomplete or failed source comparison")
        require(report["expected_pages"] == report["compared_pages"] == report["exact_pages"]
                == len(report["pages"]), "Source comparison count differs")
        require(report["checks"] and all(c["passed"] is True for c in report["checks"]), "Failed provenance check")
        require(all(not report[k] for k in ["missing_cpu", "missing_gpu", "failed_cpu_pages", "failed_gpu_pages"]),
                "Missing/failed source outputs")
        require(all(r["provenance_passed"] and r["tokens_exact"] and r["text_exact"]
                    and r["finish_reason_exact"] for r in report["pages"]), "Nonexact source comparison")
        manifest = read(report["manifest"], report["manifest_sha256"])
        require(len({p["id"] for p in manifest["pages"]}) == len(manifest["pages"]), "Duplicate manifest ID")
        require([p["id"] for p in manifest["pages"]] == [p["id"] for p in report["pages"]], "Comparison manifest order")
        reports[platform], manifests[platform] = report, {p["id"]: p for p in manifest["pages"]}
        runs[platform] = read(report["cpu_run"], report["cpu_run_sha256"])
        require(runs[platform]["contract"] == report["cpu_contract"], "CPU contract differs from comparison")
        require(runs[platform]["teacher_forced"] is False, "Teacher-forced source")
        require(runs[platform]["os"] == platform, "Source runtime OS differs from platform label")
        replay = "postprocessing_replay" in runs[platform]["contract"]
        require(replay == (platform == "windows") and report["derived_text_replay"] is replay
                and runs[platform].get("derived_text_replay", False) is replay, "Unexpected or inconsistent replay origin")
        if platform == "linux":
            require("wsl" in runs[platform]["contract"]["environment_label"].lower(), "Linux run not labeled WSL")
    windows, linux = reports["windows"], reports["linux"]
    require(windows["gpu_run_sha256"] == linux["gpu_run_sha256"], "Different GPU reference run")
    read(windows["gpu_run"], windows["gpu_run_sha256"])
    read(linux["gpu_run"], linux["gpu_run_sha256"])
    for key in ["model_revision", "weights_sha256", "precision", "options"]:
        require(windows["cpu_contract"][key] == linux["cpu_contract"][key], "Different inference option: " + key)
    require(windows["expected_pages"] == 200 and linux["expected_pages"] == 24, "Expected fixed200/selected24 comparisons")
    win_rows = {r["id"]: r for r in windows["pages"]}
    fields = ["token_ids", "text", "finish_reason", "width", "height", "input_tokens", "output_tokens",
              "teacher_forced", "precision", "backend", "cache_layout", "weight_layout", "packed_weight_bytes"]
    rows, reasons = [], Counter()
    for row in linux["pages"]:
        page_id, key = row["id"], row["sample_id"]
        require(page_id in manifests["windows"] and manifests["linux"][page_id] == manifests["windows"][page_id],
                "Selected page differs from Windows parent")
        require(key and Path(key).name == key and key not in [".", ".."], "Unsafe record key")
        win_row = win_rows[page_id]
        require(win_row["sample_id"] == key and win_row["gpu_record_sha256"] == row["gpu_record_sha256"],
                "Different selected GPU record")
        records = {}
        for platform, comparison_row in [("windows", win_row), ("linux", row)]:
            report = reports[platform]
            record = read(Path(report["cpu_run"]).parent / (key + ".json"), comparison_row["cpu_record_sha256"])
            require(record["id"] == page_id and record["contract_sha256"] == runs[platform]["contract_sha256"],
                    "CPU record/run identity differs")
            records[platform] = record["result"]
        gpu = read(Path(linux["gpu_run"]).parent / (key + ".json"), row["gpu_record_sha256"])
        read(Path(windows["gpu_run"]).parent / (key + ".json"), win_row["gpu_record_sha256"])
        require(gpu["id"] == page_id, "GPU record ID differs")
        for field in fields:
            require(records["windows"][field] == records["linux"][field], "CPU disagreement: " + key + "/" + field)
        result = records["windows"]
        for field in ["token_ids", "text", "finish_reason"]:
            require(result[field] == gpu[field], "GPU disagreement: " + key + "/" + field)
        require(result["input_tokens"] == gpu["prefix_length"], "GPU prefix differs")
        require(result["output_tokens"] == len(result["token_ids"]), "Output count differs")
        require(result["teacher_forced"] is False and result["precision"] == "fp32", "Unexpected inference path")
        reasons[result["finish_reason"]] += 1
        rows.append({"id": page_id, "sample_id": key, "category": row["category"],
                     "generated_ids": result["output_tokens"], "finish_reason": result["finish_reason"],
                     "cpu_dimensions": [result["width"], result["height"]], "prefix_tokens": result["input_tokens"],
                     "windows_record_sha256": win_row["cpu_record_sha256"],
                     "linux_record_sha256": row["cpu_record_sha256"], "gpu_record_sha256": row["gpu_record_sha256"]})
    for path, expected in bound.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Input changed during join: " + path)
    report = {"schema_version": 1, "status": "complete_selected_platform_output_agreement",
              "pages": len(rows), "matching_token_ids": sum(r["generated_ids"] for r in rows),
              "compared_cpu_result_fields": fields, "finish_reasons": dict(reasons),
              "windows_text_replay": windows["derived_text_replay"], "linux_text_replay": linux["derived_text_replay"],
              "startup_semantic_fields_complete": {p: r["startup_semantic_fields_complete"] for p, r in reports.items()},
              "inference_contracts": {p: r["cpu_contract"] for p, r in reports.items()},
              "source_sha256": bound, "records": rows,
              "qualification": "Direct equality of saved outputs on the fixed24-page subset of the200-page corpus. Windows inference plus separately validated text replay is compared with fresh Linux-under-WSL inference and the identical strict FP32 GPU records. Existing comparison provenance qualifications remain in force. No inference, tensor comparison, fresh startup attestation, absolute OCR accuracy, throughput or bare-metal Linux measurement is performed. Historical GPU records do not contain processed width/height; only CPU dimensions and GPU prefix lengths are compared."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ["status", "pages", "matching_token_ids", "finish_reasons"]}))


if __name__ == "__main__":
    main()
