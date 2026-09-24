#!/usr/bin/env python3
"""Compare original-fixture CPU/GPU outputs without hiding partial/failed runs.

CPU-only snapshots are valid reports, but cannot pass paired parity. Blank-page
quality has no CER/WER denominator. Derived variants retain parent grouping.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unicodedata

from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from validate_text_replay import REPLAY_QUALIFICATION, resolve_recorded_path, validate_run_replay, validate_page_replay

MANIFEST_SHA256 = "ff90af60b10c4bd411526326fed5a65ca820c4eef9d795fede808d80a1f7cfdf"
EXPECTED_OPTIONS = {"max_dimension": 1536, "min_dimension": 64, "max_new_tokens": 512}
STOP_IDS = {11, 263}


class TableTextProjection(HTMLParser):
    """Remove only known table tags; preserve content and non-table output."""
    tags = {"table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
        self.parts.append(" " if self.depth and tag in self.tags else self.get_starttag_text())

    def handle_endtag(self, tag):
        self.parts.append(" " if self.depth and tag in self.tags else "</" + tag + ">")
        if tag == "table":
            self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        self.parts.append(data)


def table_text_projection(text):
    if "<table" not in text.lower():
        return text, False
    parser = TableTextProjection()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts), True


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(left, right):
    """Exact unit-cost Levenshtein distance for strings or word sequences."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def intended_quality(truth, output):
    reference, prediction = normalized(truth), normalized(output)
    result = {"raw_text_exact": truth == output, "normalized_text_exact": reference == prediction}
    if not reference:
        result.update({"kind": "blank", "raw_output_empty": output == "", "normalized_output_empty": not prediction,
                       "raw_output_characters": len(output), "normalized_hallucinated_characters": len(prediction),
                       "nonwhitespace_hallucinated_characters": sum(not c.isspace() for c in prediction),
                       "cer": None, "wer": None})
    else:
        characters = edit_distance(reference, prediction)
        words = edit_distance(reference.split(), prediction.split())
        projected, projected_applied = table_text_projection(output)
        projected = normalized(projected)
        projected_characters = edit_distance(reference, projected)
        projected_words = edit_distance(reference.split(), projected.split())
        result.update({"kind": "nonblank", "reference_characters": len(reference), "reference_words": len(reference.split()),
                       "character_edit_distance": characters, "word_edit_distance": words,
                       "cer": characters / len(reference), "wer": words / len(reference.split()),
                       "secondary_table_text_projection_applied": projected_applied,
                       "secondary_content_normalized_exact": reference == projected,
                       "secondary_content_character_edit_distance": projected_characters,
                       "secondary_content_word_edit_distance": projected_words,
                       "secondary_content_cer": projected_characters / len(reference),
                       "secondary_content_wer": projected_words / len(reference.split())})
    return result


def aggregate(records, side):
    completed = [r[side] for r in records if r[side]["status"] == "complete" and r[side]["provenance_passed"]]
    nonblank = [r["quality"] for r in completed if r["quality"]["kind"] == "nonblank"]
    blank = [r["quality"] for r in completed if r["quality"]["kind"] == "blank"]
    characters = sum(r["reference_characters"] for r in nonblank)
    words = sum(r["reference_words"] for r in nonblank)
    char_edits = sum(r["character_edit_distance"] for r in nonblank)
    word_edits = sum(r["word_edit_distance"] for r in nonblank)
    return {"expected_pages": len(records), "valid_completed_pages": len(completed),
            "status_counts": dict(Counter(r[side]["status"] for r in records)),
            "invalid_provenance_pages": [r["id"] for r in records if r[side]["status"] == "complete" and not r[side]["provenance_passed"]],
            "nonblank_pages": len(nonblank), "nonblank_normalized_exact_pages": sum(r["normalized_text_exact"] for r in nonblank),
            "nonblank_reference_characters": characters, "nonblank_character_edit_distance": char_edits,
            "nonblank_micro_cer": char_edits / characters if characters else None,
            "nonblank_reference_words": words, "nonblank_word_edit_distance": word_edits,
            "nonblank_micro_wer": word_edits / words if words else None,
            "nonblank_macro_cer": sum(r["cer"] for r in nonblank) / len(nonblank) if nonblank else None,
            "secondary_content_normalized_exact_pages": sum(r["secondary_content_normalized_exact"] for r in nonblank),
            "secondary_content_micro_cer": sum(r["secondary_content_character_edit_distance"] for r in nonblank) / characters if characters else None,
            "secondary_content_micro_wer": sum(r["secondary_content_word_edit_distance"] for r in nonblank) / words if words else None,
            "blank_pages": len(blank), "blank_raw_empty_pages": sum(r["raw_output_empty"] for r in blank),
            "blank_normalized_empty_pages": sum(r["normalized_output_empty"] for r in blank),
            "blank_normalized_hallucinated_characters": sum(r["normalized_hallucinated_characters"] for r in blank),
            "generated_tokens": sum(r["output_tokens"] for r in completed),
            "finish_reason_counts": dict(Counter(r["finish_reason"] for r in completed))}


def outcome_status(cpu_complete, gpu_complete, exact_pages, expected_pages, provenance, missing, failed):
    paired = cpu_complete and gpu_complete and provenance and exact_pages == expected_pages
    if failed["cpu"] or failed["gpu"] or not provenance:
        return "failed", False
    if cpu_complete and gpu_complete:
        return ("complete" if paired else "parity_mismatch"), paired
    if cpu_complete and missing["gpu"]:
        return "cpu_complete_gpu_pending", False
    return "partial", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("reference/original-supplement-v1-lock.json"))
    parser.add_argument("--cpu", type=Path, default=Path("artifacts/cpu/original-supplement-fp32-512"))
    parser.add_argument("--gpu", type=Path, default=Path("artifacts/reference/original-supplement-fp32-512"))
    parser.add_argument("--output", type=Path, default=Path("reference/original-supplement-fp32-comparison.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks, file_errors = [], []

    def check(name, actual, expected):
        checks.append({"name": name, "passed": actual == expected, "actual": actual, "expected": expected})

    def read_optional(path):
        if not path.exists():
            return "missing", None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON record is not an object")
            return "present", value
        except (OSError, ValueError) as error:
            file_errors.append({"path": str(path), "error": str(error)})
            return "failed", None

    check("manifest.sha256", sha256(args.manifest), MANIFEST_SHA256)
    check("manifest.dataset", manifest.get("dataset"), "original-supplement-v1")
    cpu_run_state, cpu_run = read_optional(args.cpu / "run.json")
    gpu_run_state, gpu_run = read_optional(args.gpu / "run.json")
    cpu_contract = cpu_run.get("contract", {}) if cpu_run else {}
    gpu_config = gpu_run.get("configuration", {}) if gpu_run else {}
    replay_context = validate_run_replay(cpu_run, cpu_contract, check) if cpu_run else None
    snapshot_path = args.cpu / "binary-snapshot.json"
    snapshot_run = cpu_run
    if replay_context is not None:
        metadata = replay_context.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("source_run_path"), str):
            snapshot_path = resolve_recorded_path(metadata["source_run_path"]).parent / "binary-snapshot.json"
        snapshot_run = replay_context.get("source_run") or {}
    snapshot_state, snapshot = read_optional(snapshot_path)
    for side, run, config in [("cpu", cpu_run, cpu_contract), ("gpu", gpu_run, gpu_config)]:
        if run is None:
            continue
        for key, expected in [("model_revision", REVISION), ("weights_sha256", WEIGHT_SHA256),
                              ("manifest_sha256", MANIFEST_SHA256), ("precision", "fp32")]:
            check(side + "." + key, config.get(key), expected)
        options = config.get("options", {}) if side == "cpu" else config
        for key, expected in EXPECTED_OPTIONS.items():
            check(side + "." + key, options.get(key), expected)
    if cpu_run is not None:
        check("cpu.backend", cpu_contract.get("backend"), "avx2")
        check("cpu.threads", cpu_contract.get("threads"), 4)
        check("cpu.os", cpu_run.get("os"), "windows")
        check("cpu.teacher_forced", cpu_run.get("teacher_forced"), False)
        packed = json.dumps(cpu_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        check("cpu.contract_sha256", hashlib.sha256(packed).hexdigest(), cpu_run.get("contract_sha256"))
        check("cpu.binary_snapshot_present", snapshot_state, "present")
        if snapshot is not None:
            source_contract = snapshot_run.get("contract", {})
            check("cpu.snapshot_contract", snapshot.get("contract_sha256"), snapshot_run.get("contract_sha256"))
            check("cpu.snapshot_sources", snapshot.get("embedded_source_sha256"), source_contract.get("source_sha256"))
            check("cpu.snapshot_configuration", snapshot.get("contract"), source_contract)
            binary_path = resolve_recorded_path(snapshot.get("binary_path", ""))
            check("cpu.binary_unchanged", sha256(binary_path) if binary_path.is_file() else None, snapshot.get("binary_sha256"))
    if gpu_run is not None:
        check("gpu.tf32", gpu_config.get("tf32"), False)
        check("gpu.flex_float32_precision", gpu_config.get("flex_float32_precision"), "ieee")
        check("gpu.compiled_blocks", gpu_config.get("compiled_blocks"), False)
    run_provenance = {side: all(c["passed"] for c in checks if c["name"].startswith(("manifest.", side + "."))) for side in ["cpu", "gpu"]}
    rows = []
    for page in manifest["pages"]:
        name = Path(page["canonical_path"]).parent.name
        row = {"id": page["id"], "sample_id": name, "parent_id": page["parent_id"], "category": page["category"], "parity": None}
        before = len(checks)
        try:
            truth = Path(page["ground_truth_path"]).read_text(encoding="utf-8")
            check(name + ".ground_truth_sha256", sha256(page["ground_truth_path"]), page["ground_truth_sha256"])
            check(name + ".authored_text", truth, page["expected_text"])
            check(name + ".canonical_png_sha256", sha256(page["canonical_path"]), page["canonical_png_sha256"])
            with Image.open(page["canonical_path"]) as im:
                check(name + ".canonical_rgb_sha256", hashlib.sha256(im.convert("RGB").tobytes()).hexdigest(), page["rgb_sha256"])
        except (OSError, ValueError) as error:
            truth = page["expected_text"]
            check(name + ".input_read_error", str(error), None)
        row["expected_text"] = truth
        page_provenance = all(c["passed"] for c in checks[before:])
        outputs = {}
        for side, folder, run in [("cpu", args.cpu, cpu_run), ("gpu", args.gpu, gpu_run)]:
            path = folder / (name + ".json")
            state, record = read_optional(path)
            entry = {"status": "missing" if state == "missing" else "failed", "path": str(path)}
            row[side] = entry
            if record is None:
                if state == "failed":
                    entry["error"] = "Unreadable JSON record; see file_errors"
                continue
            entry["record_sha256"] = sha256(path)
            before = len(checks)
            check(name + "." + side + ".id", record.get("id"), page["id"])
            check(name + "." + side + ".run_present", run is not None, True)
            if side == "cpu":
                validate_page_replay(record, replay_context, check, name)
                check(name + ".cpu.contract", record.get("contract_sha256"), cpu_run.get("contract_sha256") if cpu_run else None)
                check(name + ".cpu.image", record.get("input_sha256"), page["canonical_png_sha256"])
                check(name + ".cpu.truth", record.get("ground_truth_sha256"), page["ground_truth_sha256"])
                result = record.get("result")
            else:
                check(name + ".gpu.configuration", record.get("configuration"), gpu_config)
                check(name + ".gpu.image", record.get("canonical_rgb_sha256"), page["rgb_sha256"])
                result = record
            if "error" in record or not isinstance(result, dict):
                entry["error"] = record.get("error", "Missing or invalid result object")
                continue
            ids, text, finish = result.get("token_ids"), result.get("text"), result.get("finish_reason")
            if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids) or not isinstance(text, str) or not isinstance(finish, str) or finish not in {"eos", "length"}:
                entry["error"] = "Invalid token_ids, text or finish_reason schema"
                continue
            prefix = result.get("input_tokens" if side == "cpu" else "prefix_length")
            if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 1:
                entry["error"] = "Missing or invalid input prefix length"
                continue
            check(name + "." + side + ".token_cap", len(ids) <= EXPECTED_OPTIONS["max_new_tokens"], True)
            check(name + "." + side + ".no_earlier_stop", any(i in STOP_IDS for i in ids[:-1]), False)
            check(name + "." + side + ".stop_reason", "eos" if ids[-1] in STOP_IDS else "length", finish)
            if finish == "length":
                check(name + "." + side + ".length_cap", len(ids), EXPECTED_OPTIONS["max_new_tokens"])
            if side == "cpu":
                check(name + ".cpu.output_tokens", result.get("output_tokens"), len(ids))
                check(name + ".cpu.teacher_forced", result.get("teacher_forced"), False)
                check(name + ".cpu.precision", result.get("precision"), "fp32")
            entry.update({"status": "complete", "provenance_passed": run_provenance[side] and page_provenance and all(c["passed"] for c in checks[before:]),
                          "output_tokens": len(ids), "terminal_token_id": ids[-1], "finish_reason": finish,
                          "quality": intended_quality(truth, text), "text": text})
            outputs[side] = (ids, text, finish, result)
        if len(outputs) == 2:
            cpu_ids, cpu_text, cpu_finish, cpu_result = outputs["cpu"]
            gpu_ids, gpu_text, gpu_finish, gpu_result = outputs["gpu"]
            first = next((i for i, (a, b) in enumerate(zip(cpu_ids, gpu_ids)) if a != b), None)
            if first is None and len(cpu_ids) != len(gpu_ids):
                first = min(len(cpu_ids), len(gpu_ids))
            row["parity"] = {"token_ids_exact": cpu_ids == gpu_ids, "text_exact": cpu_text == gpu_text,
                             "finish_reason_exact": cpu_finish == gpu_finish, "terminal_token_exact": cpu_ids[-1] == gpu_ids[-1],
                             "prefix_length_exact": cpu_result.get("input_tokens") == gpu_result.get("prefix_length"),
                             "first_token_divergence": first, "matching_prefix_tokens": first if first is not None else len(cpu_ids)}
            row["parity"]["exact"] = all(row["parity"][k] for k in ["token_ids_exact", "text_exact", "finish_reason_exact", "terminal_token_exact", "prefix_length_exact"])
            if first is not None:
                decisions = gpu_result.get("logit_decisions", [])
                row["parity"]["first_gpu_decision"] = decisions[first] if isinstance(decisions, list) and first < len(decisions) else None
                row["parity"]["first_cpu_token"] = cpu_ids[first] if first < len(cpu_ids) else None
        rows.append(row)

    parents, categories = defaultdict(list), defaultdict(list)
    for row in rows:
        parents[row["parent_id"]].append(row)
        categories[row["category"]].append(row)
    summaries = {side: aggregate(rows, side) for side in ["cpu", "gpu"]}
    parent_summaries = [{"parent_id": parent, "page_ids": [p["id"] for p in pages], **{side: aggregate(pages, side) for side in ["cpu", "gpu"]}} for parent, pages in parents.items()]
    for side in ["cpu", "gpu"]:
        parent_rates = [p[side]["nonblank_micro_cer"] for p in parent_summaries if p[side]["nonblank_micro_cer"] is not None]
        summaries[side]["nonblank_parent_balanced_macro_cer"] = sum(parent_rates) / len(parent_rates) if parent_rates else None
        summaries[side]["observed_nonblank_parents"] = len(parent_rates)
    missing = {side: [r["id"] for r in rows if r[side]["status"] == "missing"] for side in ["cpu", "gpu"]}
    failed = {side: [{"id": r["id"], "error": r[side]["error"]} for r in rows if r[side]["status"] == "failed"] for side in ["cpu", "gpu"]}
    cpu_complete = summaries["cpu"]["valid_completed_pages"] == len(rows)
    gpu_complete = summaries["gpu"]["valid_completed_pages"] == len(rows)
    exact_pages = sum(r["parity"] is not None and r["parity"]["exact"] for r in rows)
    provenance = all(c["passed"] for c in checks) and not file_errors
    status, parity_passed = outcome_status(cpu_complete, gpu_complete, exact_pages, len(rows), provenance, missing, failed)
    report = {"schema_version": 1, "status": status, "expected_pages": len(rows), "expected_parents": len(parents),
              "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest), "comparison_script_sha256": sha256(__file__),
              "cpu_run_state": cpu_run_state, "gpu_run_state": gpu_run_state,
              "cpu_run_sha256": sha256(args.cpu / "run.json") if cpu_run else None,
              "gpu_run_sha256": sha256(args.gpu / "run.json") if gpu_run else None,
              "cpu_contract": cpu_contract, "gpu_configuration": gpu_config, "cpu_binary_snapshot": snapshot,
              "cpu_binary_snapshot_path": str(snapshot_path),
              "cpu_prediction_origin": REPLAY_QUALIFICATION if replay_context is not None else "Original free-running inference and its original decoder.",
              "cpu_functional_complete": cpu_complete, "gpu_functional_complete": gpu_complete,
              "compared_pages": sum(r["parity"] is not None for r in rows), "exact_paired_pages": exact_pages,
              "completed_paired_parity_passed": parity_passed, "provenance_passed": provenance,
              "missing": missing, "failed": failed, "file_errors": file_errors,
              "summary": summaries, "categories": {category: {side: aggregate(pages, side) for side in ["cpu", "gpu"]} for category, pages in categories.items()},
              "parents": parent_summaries, "checks": checks, "pages": rows,
              "metrics": {"normalization": "NFC and whitespace collapse; case/punctuation preserved", "nonblank": "Primary: exact unit-cost character and whitespace-word Levenshtein distances on untouched output; HTML markup therefore counts as characters", "secondary_content": "Explicit secondary diagnostic: replace only known table tags with spaces and decode HTML entities before normalization; compare cell content in emitted order. No table-structure accuracy claim. Introduced after observing CPU table markup; raw metrics and exact parity remain unchanged.", "blank": "Raw/normalized empty output and hallucinated character counts; CER/WER are null; no marker stripping", "parent_weighting": "Mean of nonblank per-parent micro CER so derived rotations/contrast variants do not each receive a full parent weight"},
              "qualification": "Separate synthetic diagnostic results; CPU functional completion is distinct from paired GPU parity and intended-text quality. Concurrent load timings are not performance measurements. Missing, failed or invalid records cannot pass the complete paired parity gate."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "expected_pages", "cpu_functional_complete", "gpu_functional_complete", "compared_pages", "exact_paired_pages", "completed_paired_parity_passed", "provenance_passed"]}, indent=2))
    if not provenance or failed["cpu"] or failed["gpu"] or (cpu_complete and gpu_complete and not parity_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
