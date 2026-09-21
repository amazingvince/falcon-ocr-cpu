#!/usr/bin/env python3
"""Bind a corpus comparison to frozen inputs and report quality evidence only.

This consumes an independently validated comparison; it does not run inference
or substitute for its model/inference provenance checks. Literal equality can
prove zero differential without selecting a primary metric after the results.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import pathlib
import unicodedata

from rapidfuzz.distance import Levenshtein
from corpus_comparison import sample_key, validate_cpu_result, semantic_contract, REVISION, WEIGHT_SHA256
from validate_gpu_reference_record import validate_gpu_reference_record
from validate_text_replay import (resolve_recorded_path, canonical_sha256,
                                 validate_run_replay, validate_page_replay)

V3_SHA256 = "1a52a404670c48800eea9a275c8527bb35e800305d95061c85feaf3d5868204f"
V3_CATEGORIES = {"ordinary": 35, "formulas": 30, "tables": 30, "tiny_text": 30,
                 "degraded": 25, "handwriting": 25, "multi_column": 25}
NORMALIZATION = "NFC followed by Python split() whitespace collapse and single-space join; existing assembled-ground-truth diagnostic only."


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_metrics(value, label="metrics"):
    """Reject numeric NaN/Infinity anywhere; explicit absent values stay absent."""
    if isinstance(value, float):
        require(math.isfinite(value), label + ": nonfinite metric")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_metrics(item, label + "." + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_metrics(item, f"{label}[{index}]")


def normalized(text):
    # Deliberately identical to compare_corpus.normalized; do not clean markup,
    # punctuation, case or non-whitespace content to create apparent equality.
    return " ".join(unicodedata.normalize("NFC", text).split())


def aggregate(rows, *, words=False):
    unit = "words" if words else "characters"
    edits_key = "word_edits" if words else "character_edits"
    metric = "wer" if words else "cer"
    positive = [r for r in rows if r["ground_truth_" + unit] > 0]
    total = sum(r["ground_truth_" + unit] for r in positive)
    values = {}
    for side in ("gpu", "cpu"):
        edits = sum(r[side + "_" + edits_key] for r in positive)
        values[side] = {"micro_" + metric: edits / total if total else None,
                        "page_mean_" + metric: sum(r[side + "_" + edits_key] / r["ground_truth_" + unit] for r in positive) / len(positive) if positive else None,
                        "positive_denominator_" + edits_key: edits,
                        "all_page_" + edits_key: sum(r[side + "_" + edits_key] for r in rows),
                        "zero_truth_" + edits_key: sum(r[side + "_" + edits_key] for r in rows if r["ground_truth_" + unit] == 0),
                        "empty_truth_prediction_" + unit: sum(r[side + "_" + edits_key] for r in rows if r["ground_truth_" + unit] == 0)}
    return {"pages": len(rows), metric + "_pages": len(positive), "ground_truth_" + unit: total,
            "zero_denominator_pages": len(rows) - len(positive), **values,
            "micro_" + metric + "_cpu_minus_gpu_percentage_points": 100 * (values["cpu"]["micro_" + metric] - values["gpu"]["micro_" + metric]) if total else None,
            "page_mean_" + metric + "_cpu_minus_gpu_percentage_points": 100 * (values["cpu"]["page_mean_" + metric] - values["gpu"]["page_mean_" + metric]) if positive else None,
            "micro_rate_numerator": "positive_denominator_" + edits_key,
            "role": "descriptive assembled-text " + metric.upper() + "; neither aggregate is selected as the acceptance metric"}


def assess(manifest, rows, missing, *, expected_categories):
    """Pure decision layer; callers first bind source bytes and comparison checks."""
    finite_metrics(rows)
    expected = manifest["pages"]
    require(Counter(p["category"] for p in expected) == Counter(expected_categories), "Frozen category inventory differs")
    ids = {p["id"]: p for p in expected}
    require(len(ids) == len(expected), "Duplicate manifest page IDs")
    require(len({p["id"] for p in rows}) == len(rows), "Duplicate compared page IDs")
    for row in rows:
        require(row["id"] in ids and row["category"] == ids[row["id"]]["category"], "Unknown page or changed category")
    missing_ids = set(ids) - {p["id"] for p in rows}
    require(set(missing) == missing_ids, "Missing-page accounting differs")
    complete = not missing_ids and len(rows) == sum(expected_categories.values())
    exact = sum(r["literal_text_equal"] for r in rows)
    zero = complete and exact == len(expected)
    categories = {}
    for category, count in expected_categories.items():
        selected = [r for r in rows if r["category"] == category]
        category_complete = len(selected) == count
        category_zero = category_complete and all(r["literal_text_equal"] for r in selected)
        categories[category] = {"expected_pages": count, "compared_pages": len(selected),
            "complete": category_complete, "literal_exact_pages": sum(r["literal_text_equal"] for r in selected),
            "differential_percentage_points": 0.0 if category_zero else None,
            "zero_differential_proved": category_zero, "diagnostic_cer": aggregate(selected),
            "descriptive_wer": aggregate(selected, words=True)}
    return {"status": "zero_differential_proved" if zero else ("incomplete" if not complete else "nonidentical_metric_decision_unresolved"),
            "complete_corpus": complete, "expected_pages": len(expected), "compared_pages": len(rows),
            "literal_exact_pages": exact, "missing_pages": sorted(missing_ids), "by_category": categories,
            "nonquantized_differential_gate": {"passed": zero, "overall_budget_percentage_points": 0.25,
                "category_budget_percentage_points": 1.0, "overall_differential_percentage_points": 0.0 if zero else None,
                "basis": "Complete literal prediction identity implies equal inputs to every deterministic fixed quality metric with identical truth/evaluation settings." if zero else "No complete literal-identity proof. No primary metric was preselected; diagnostic scores do not silently define acceptance.",
                "primary_overall_metric": None, "official_overall_score": None},
            "diagnostic_cer": aggregate(rows), "descriptive_wer": aggregate(rows, words=True)}


def bind_official_predictions(official, roots, runs, comparison, expected_records, bound_file):
    """Join audited component inputs to these exact inference artifacts.

    A shared inference configuration does not identify a prediction. Bind both
    runs and every record/text hash, including replay records whose ancestry was
    separately validated by the caller. Keep all source text out of the report.
    """
    evidence = {}
    for side in ("gpu", "cpu"):
        provenance_path = bound_file(pathlib.Path(roots[side]) / "provenance.json",
                                     official["preparation_provenance_sha256"][side])
        provenance = read(provenance_path)
        # The attached scores depend on these files as well as the prediction
        # join. Retain the audited hashes in the caller's final source window.
        for filename, field in (("configuration.json", "configuration_sha256"),
                                ("ground-truth.json", "ground_truth_sha256")):
            bound_file(pathlib.Path(roots[side]) / filename, provenance[field])
        audit_path = bound_file(pathlib.Path(roots[side]) / "execution-audit.json",
                                official["execution_audit_sha256"][side])
        audit = read(audit_path)
        bound_file(pathlib.Path(roots[side]) / "execution.log", audit["execution_log_sha256"])
        result_hashes = {name: result[side + "_sha256"] for name, result in official["result_files"].items()
                         if result[side + "_sha256"] is not None}
        require(audit.get("result_file_sha256") == result_hashes,
                "Official audited result inventory differs: " + side)
        for filename, digest in result_hashes.items():
            require(pathlib.Path(filename).name == filename, "Official result filename is ambiguous")
            bound_file(pathlib.Path(roots[side]) / "result" / filename, digest)
        source_run = bound_file(provenance["run_path"], comparison[side + "_run_sha256"])
        require(source_run.resolve() == runs[side][0].resolve()
                and provenance.get("run_sha256") == comparison[side + "_run_sha256"],
                "Official source run differs from quality comparison: " + side)
        pages = provenance.get("pages")
        require(isinstance(pages, list) and all(isinstance(p, dict) and isinstance(p.get("id"), str) for p in pages),
                "Official prediction join lacks page identities: " + side)
        by_id = {p["id"]: p for p in pages}
        require(len(by_id) == len(pages) == len(expected_records)
                and set(by_id) == set(expected_records),
                "Official prediction join has missing, duplicate or unexpected pages: " + side)
        joined = []
        for page_id, expected in expected_records.items():
            actual = by_id[page_id]
            item = expected[side]
            require(actual.get("category") == expected["category"], "Official prediction category differs: " + page_id)
            require(actual.get("record_sha256") == item["record_sha256"],
                    "Official source record differs: " + side + ":" + page_id)
            record_path = bound_file(actual["record_path"], item["record_sha256"])
            require(record_path.resolve() == item["record_path"].resolve(),
                    "Official source record path differs: " + side + ":" + page_id)
            require(actual.get("prediction_sha256") == item["prediction_sha256"],
                    "Official literal prediction differs: " + side + ":" + page_id)
            filename = actual.get("prediction_filename")
            require(isinstance(filename, str) and pathlib.Path(filename).name == filename,
                    "Official prediction filename is missing or ambiguous: " + page_id)
            prediction = bound_file(pathlib.Path(roots[side]) / "predictions" / filename, item["prediction_sha256"])
            joined.append({"id": page_id, "category": expected["category"],
                           "record_path": str(record_path.resolve()), "record_sha256": item["record_sha256"],
                           "prediction_path": str(prediction.resolve()), "prediction_utf8_sha256": item["prediction_sha256"]})
        evidence[side] = {"source_run_path": str(source_run.resolve()),
                          "source_run_sha256": comparison[side + "_run_sha256"],
                          "preparation_provenance_sha256": official["preparation_provenance_sha256"][side],
                          "execution_audit_sha256": official["execution_audit_sha256"][side],
                          "result_file_sha256": result_hashes,
                          "joined_pages": len(joined), "pages": joined}
    return evidence


def report(manifest_path, comparison_path, *, expected_manifest_sha256=V3_SHA256,
           expected_categories=None, official_gpu=None, official_cpu=None):
    expected_categories = V3_CATEGORIES if expected_categories is None else expected_categories
    manifest_path, comparison_path = pathlib.Path(manifest_path), pathlib.Path(comparison_path)
    sources = {}
    validation_checks = []

    def check(name, actual, expected):
        require(actual == expected, "Inference/lineage validation failed: " + name)
        # Store no token sequences, source text, or copied record content.
        validation_checks.append({"name": name, "passed": True})

    def bound_file(path, expected=None):
        path = resolve_recorded_path(str(path))
        digest = sha(path)
        if expected is not None:
            require(digest == expected, "Source/hash changed: " + str(path))
        sources[str(path.resolve())] = digest
        return path

    script_dir = pathlib.Path(__file__).resolve().parent
    source_code_hashes = {}
    for name in ("report_quality_regression.py", "corpus_comparison.py", "validate_text_replay.py",
                 "validate_gpu_reference_record.py", "fetch_reference.py",
                 "compare_official_evaluation.py", "prepare_official_evaluation.py"):
        path = bound_file(script_dir / name)
        source_code_hashes[name] = sources[str(path.resolve())]
    bound_file(manifest_path, expected_manifest_sha256)
    bound_file(comparison_path)
    manifest, comparison = read(manifest_path), read(comparison_path)
    finite_metrics(comparison, "comparison")
    require(comparison.get("schema_version") == 2, "Validated comparison schema2 required")
    require(comparison.get("manifest_sha256") == expected_manifest_sha256, "Comparison manifest differs")
    require(comparison.get("expected_pages") == sum(expected_categories.values()), "Expected full-corpus count differs")
    require(comparison.get("provenance_passed") is True and comparison.get("checks") and all(c.get("passed") is True for c in comparison["checks"]), "Upstream comparison provenance failed or absent")
    require(not comparison.get("failed_cpu_pages") and not comparison.get("failed_gpu_pages") and comparison.get("failed_cpu_count") == 0 and comparison.get("failed_gpu_count") == 0, "Failed inference pages remain unresolved")
    runs = {}
    for side in ("gpu", "cpu"):
        path = bound_file(comparison[side + "_run"], comparison[side + "_run_sha256"])
        runs[side] = (path, read(path))
    gpu_config, cpu_config = runs["gpu"][1]["configuration"], runs["cpu"][1]["contract"]
    require(gpu_config == comparison["gpu_configuration"] and cpu_config == comparison["cpu_contract"], "Comparison run contract changed")
    require(gpu_config.get("precision") == cpu_config.get("precision") and cpu_config.get("precision") in ("fp32", "bf16"), "Only matched nonquantized modes qualify")
    semantic = {"cpu": semantic_contract(cpu_config, "cpu", cpu_config["precision"], check),
                "gpu": semantic_contract(gpu_config, "gpu", gpu_config["precision"], check, gpu=True)}
    check("gpu.tf32", gpu_config.get("tf32"), False)
    check("gpu.flex_float32_precision", gpu_config.get("flex_float32_precision"), "ieee")
    check("cpu.contract_hash", canonical_sha256(cpu_config), runs["cpu"][1].get("contract_sha256"))
    replay = validate_run_replay(runs["cpu"][1], cpu_config, check)
    if replay is not None:
        metadata = cpu_config["postprocessing_replay"]
        for name in ("source_run", "decoder_binary", "decoder_source"):
            bound_file(metadata[name + "_path"], metadata[name + "_sha256"])
        for path, digest in metadata["tokenizer_asset_sha256"].items():
            bound_file(path, digest)
    for config in (gpu_config, cpu_config):
        require(config.get("manifest_sha256") == expected_manifest_sha256, "Only the complete fixed evaluation manifest is accepted")
    for key in ("weights_sha256", "model_revision"):
        require(gpu_config.get(key) and gpu_config.get(key) == cpu_config.get(key), "CPU/GPU model contract differs")
    require(all(cpu_config["options"][key] == gpu_config[key] for key in ("min_dimension", "max_dimension", "max_new_tokens")), "Inference options differ")
    require(runs["cpu"][1].get("teacher_forced") is False, "Free-running CPU run required")
    rows_by_id = {p["id"]: p for p in comparison["pages"]}
    require(len(rows_by_id) == len(comparison["pages"]) == comparison.get("compared_pages"), "Comparison row count/duplicates")
    manifest_ids = {p["id"] for p in manifest["pages"]}
    require(set(rows_by_id) <= manifest_ids, "Unexpected comparison page")
    keys_by_id = {p["id"]: sample_key(p) for p in manifest["pages"]}
    require(len(set(keys_by_id.values())) == len(keys_by_id), "Duplicate manifest sample keys")
    missing_keys = set()
    for side in ("cpu", "gpu"):
        listed = comparison["missing_" + side]
        require(isinstance(listed, list) and len(listed) == len(set(listed))
                and set(listed) <= set(keys_by_id.values()), "Malformed missing-page inventory")
        require(comparison["missing_" + side + "_count"] == len(listed), "Missing-page count differs")
        missing_keys.update(listed)
    require(missing_keys == {keys_by_id[page_id] for page_id in manifest_ids - set(rows_by_id)}, "Missing inventories do not explain exactly the unpaired pages")
    for asset in comparison.get("checked_model_assets", {}).get("assets", {}).values():
        bound_file(asset["path"], asset["sha256"])
    rows, missing, official_records = [], [], {}
    for page in manifest["pages"]:
        key = sample_key(page)
        # Bind truth and source pixels for every page, including missing inference.
        for path_key, digest_key in (("ground_truth_path", "ground_truth_sha256"), ("canonical_path", "canonical_png_sha256"), ("source_path", "source_sha256")):
            if path_key == "source_path" and path_key not in page:
                continue
            bound_file(page[path_key], page[digest_key])
        if "annotation_path" in page:
            annotation_path = bound_file(page["annotation_path"])
            canonical = json.dumps(read(annotation_path), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            require(hashlib.sha256(canonical).hexdigest() == page["annotation_page_sha256"], "Source annotation changed")
        if page["id"] not in rows_by_id:
            missing.append(page["id"])
            require(key in set(comparison["missing_cpu"]) | set(comparison["missing_gpu"]), "Unexplained missing comparison row")
            continue
        prior = rows_by_id[page["id"]]
        require(prior.get("category") == page["category"] and prior.get("sample_id") == key and prior.get("provenance_passed") is True, "Page category/identity/provenance differs")
        records = {}
        for side in ("gpu", "cpu"):
            record_path = bound_file(runs[side][0].parent / (key + ".json"), prior[side + "_record_sha256"])
            records[side] = read(record_path)
            require(records[side].get("id") == page["id"] and records[side].get("category") == page["category"] and "error" not in records[side], "Changed inference identity/error")
        gpu, cpu = records["gpu"], records["cpu"].get("result")
        require(validate_cpu_result(cpu, cpu_config, check, key + ".cpu_result"), "Malformed CPU inference result")
        require(validate_gpu_reference_record(gpu, gpu_config, check, key + ".gpu_result"), "Malformed GPU inference result")
        check(key + ".prefix_tokens", cpu["input_tokens"], gpu["prefix_length"])
        if "width" in gpu or "height" in gpu:
            check(key + ".processed_dimensions", [cpu["width"], cpu["height"]], [gpu.get("width"), gpu.get("height")])
        check(key + ".gpu_not_replay", "postprocessing_replay" not in gpu and not gpu.get("derived_text_replay", False), True)
        check(key + ".gpu_free_running", gpu.get("teacher_forced", False), False)
        validate_page_replay(records["cpu"], replay, check, key)
        if replay is not None:
            metadata = records["cpu"]["postprocessing_replay"]
            bound_file(metadata["source_record_path"], metadata["source_record_sha256"])
        require(gpu.get("configuration") == gpu_config and records["cpu"].get("contract_sha256") == runs["cpu"][1]["contract_sha256"], "Page run binding differs")
        require(cpu.get("teacher_forced") is False, "Teacher-forced prediction cannot qualify")
        require(records["cpu"].get("ground_truth_sha256") == page["ground_truth_sha256"] and records["cpu"].get("input_sha256") == page["canonical_png_sha256"] and gpu.get("canonical_rgb_sha256") == page["rgb_sha256"], "Page input/truth binding differs")
        require(isinstance(cpu.get("text"), str) and isinstance(gpu.get("text"), str), "Literal string predictions required")
        truth = normalized(resolve_recorded_path(page["ground_truth_path"]).read_text(encoding="utf-8"))
        truth_words = truth.split(" ") if truth else []
        exact = cpu["text"] == gpu["text"]
        require(exact == prior["text_exact"], "Prior literal equality assertion differs")
        row = {"id": page["id"], "category": page["category"], "literal_text_equal": exact,
               "ground_truth_characters": len(truth), "ground_truth_words": len(truth_words), "gpu_text_utf8_sha256": hashlib.sha256(gpu["text"].encode()).hexdigest(),
               "cpu_text_utf8_sha256": hashlib.sha256(cpu["text"].encode()).hexdigest()}
        official_records[page["id"]] = {"category": page["category"], **{
            side: {"record_path": runs[side][0].parent / (key + ".json"),
                   "record_sha256": prior[side + "_record_sha256"],
                   "prediction_sha256": row[side + "_text_utf8_sha256"]}
            for side in ("gpu", "cpu")}}
        for side, record in (("gpu", gpu), ("cpu", cpu)):
            distance = Levenshtein.distance(truth, normalized(record["text"]))
            require(prior[side + "_diagnostic_edit_distance"] == distance and prior[side + "_diagnostic_cer"] == distance / max(1, len(truth)), "Diagnostic CER source/normalization differs")
            row[side + "_character_edits"] = distance
            prediction = normalized(record["text"])
            row[side + "_word_edits"] = Levenshtein.distance(truth_words, prediction.split(" ") if prediction else [])
        require(prior["diagnostic_ground_truth_characters"] == len(truth), "Ground-truth denominator differs")
        rows.append(row)
    decision = assess(manifest, rows, missing, expected_categories=expected_categories)
    components = {"status": "not_evaluated", "absolute_scores": None,
                  "reason": "Provide both audited official evaluation directories after completion; absent metrics are not zero."}
    require(bool(official_gpu) == bool(official_cpu), "Both official evaluation directories are required together")
    if official_gpu:
        require(decision["complete_corpus"], "Cannot attach complete official scores to partial predictions")
        from compare_official_evaluation import compare
        official = compare(official_gpu, official_cpu, manifest_path)
        finite_metrics(official, "official")
        require(official["inference_contracts"]["gpu"] == gpu_config and official["inference_contracts"]["cpu"] == cpu_config, "Official metrics belong to other inference runs")
        source_binding = bind_official_predictions(official, {"gpu": official_gpu, "cpu": official_cpu},
                                                  runs, comparison, official_records, bound_file)
        require(official["all_prediction_text_bytes_equal"] == (decision["literal_exact_pages"] == decision["expected_pages"]), "Official prediction equality differs")
        if official["all_prediction_text_bytes_equal"]:
            require(official["all_evaluator_outputs_equal"], "Identical predictions yielded inconsistent evaluator outputs")
        components = {"status": "audited_components_available", "report": official, "prediction_source_binding": source_binding,
                      "reason": "Component coverage/adaptations retained; rendered CDM and official Overall remain unavailable."}
    for path, digest in sources.items():
        require(sha(path) == digest, "Source changed during quality accounting: " + path)
    return {"schema_version": 1, **decision, "manifest_sha256": expected_manifest_sha256,
            "normalization": NORMALIZATION, "prediction_origin": "validated saved-token replay" if replay is not None else "original inference",
            "source_sha256": sources, "comparison_contract_evidence": comparison.get("contract_evidence"),
            "current_inference_and_lineage_checks": validation_checks, "current_semantic_contract_evidence": semantic,
            "startup_semantic_fields_complete": all(not evidence["missing_startup_fields"] for evidence in semantic.values()),
            "structure_metrics": components,
            "wer_definition": {"tokenization": "normalized(text).split(' ') when normalized text is nonempty; otherwise []",
                "introduced": "For this descriptive report after the82-page snapshot; not a preregistered primary gate.",
                "language_limit": "Unicode whitespace tokens are not linguistically segmented CJK words. No case, punctuation or markup cleanup.",
                "aggregation": "Edit totals, micro WER and unweighted page mean over positive denominators; empty-truth pages and emitted token counts are separate, not zero WER."},
            "reporting_conditions": {"full200_predictions": decision["complete_corpus"],
                "descriptive_cer_wer_available_for_compared_pages": True,
                "audited_structure_components_available": components["status"] == "audited_components_available",
                "fixed_corpus_quality_reporting_complete": decision["complete_corpus"] and components["status"] == "audited_components_available",
                "primary_metric_preselected": False, "primary_metric_needed_for_this_differential_decision": not decision["nonquantized_differential_gate"]["passed"],
                "planned_domain_coverage_complete_in_natural200": False,
                "absolute_accuracy_threshold": None},
            "coverage_limitations": ["Fixed200 natural pages only. Original receipts/blank/rotated/script supplements are separate evidence and cannot be folded into these category counts.", "No universal document-family independence claim; preserve reviewed v3 provenance/history limitations."],
            "limitations": ["Primary overall metric and aggregation were not fixed. Nonidentical predictions require an explicit prospective acceptance definition; diagnostic CER cannot resolve the gate silently.", "Zero differential is a prediction-identity proof, not a claim of high absolute OCR accuracy or complete domain coverage.", "The consumed comparison supplies inference/model provenance; source bytes are rechecked here, and historical startup gaps remain explicit.", "No tensor numerical-parity claim, performance claim, backend promotion, quantized acceptance or inference is performed."],
            "pages": rows, "source_code_sha256": source_code_hashes,
            "source_window": "Scripts/helpers, selected inputs and replay ancestry hashed before use and unchanged at end; no historical build attestation is fabricated.",
            "report_script_sha256": source_code_hashes["report_quality_regression.py"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/corpus-v3-evaluation-lock.json"))
    parser.add_argument("--comparison", type=pathlib.Path, required=True)
    parser.add_argument("--official-gpu", type=pathlib.Path)
    parser.add_argument("--official-cpu", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite a quality report")
    result = report(args.manifest, args.comparison, official_gpu=args.official_gpu, official_cpu=args.official_cpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "expected_pages", "compared_pages", "literal_exact_pages")}))
    if not result["nonquantized_differential_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
