"""Small synthetic reports exercise completeness and fail-closed evaluation."""
from copy import deepcopy
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_official_evaluation import check_execution_audit, validate_results
from prepare_official_evaluation import EVALUATOR_REVISION, REVISION, WEIGHT_SHA256, prepare, validate_source_run, replay_check
from validate_text_replay import validate_page_replay

PREFIX = "predictions_quick_match_"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(image, edits, upper, **extra):
    return dict(image_name=image, img_id=image, gt="test", pred="text",
                upper_len=upper, Edit_num=edits, metric={"Edit_dist": edits / upper}, **extra)


def make_reports(root):
    text = [row("a.jpg", 1, 4), row("b.png", 1, 2)]
    table = row("b.png", 0, 5, gt_idx=[0])
    table["metric"].update(TEDS=0.5, TEDS_structure_only=0.75)
    samples = {"text_block": text, "display_formula": [], "table": [table], "reading_order": [row("a.jpg", 1, 4)]}
    metrics = {}
    for name, entries in samples.items():
        write(root / "result" / (PREFIX + name + "_result.json"), entries)
        if entries:
            scores = {r["image_name"]: r["Edit_num"] / r["upper_len"] for r in entries}
            write(root / "result" / (PREFIX + name + "_per_page_edit.json"), scores)
            metric = {"ALL_page_avg": sum(scores.values()) / len(scores), "edit_whole": sum(r["Edit_num"] for r in entries) / sum(r["upper_len"] for r in entries), "edit_sample_avg": sum(r["Edit_num"] / r["upper_len"] for r in entries) / len(entries)}
        else:
            metric = {"ALL_page_avg": "NaN"}
        metrics[name] = {"all": {"Edit_dist": metric}, "group": {}, "page": {}}
    metrics["table"]["all"].update(TEDS={"all": 0.5}, TEDS_structure_only={"all": 0.75})
    write(root / "result" / (PREFIX + "metric_result.json"), metrics)
    write(root / "result" / (PREFIX + "display_formula_formula.json"), [])
    write(root / "result" / (PREFIX + "table_per_table_TEDS.json"), {"b.png_[0]": {"TEDS": 0.5, "TEDS_structure_only": 0.75}})


class OfficialResultValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_reports(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return validate_results(self.root, ["a.jpg", "b.png"])

    def mutate(self, suffix, callback):
        path = self.root / "result" / (PREFIX + suffix)
        data = json.loads(path.read_text())
        callback(data)
        write(path, data)

    def test_valid_empty_component_and_exact_counts(self):
        result = self.validate()
        self.assertEqual(result["expected_file_count"], 10)
        self.assertEqual(result["components"]["text_block"]["page_count"], 2)
        self.assertEqual(result["components"]["display_formula"]["sample_count"], 0)
        self.assertEqual(result["components"]["display_formula"]["pages_without_component_score"], ["a.jpg", "b.png"])

    def test_silent_zero_exit_with_partial_pages_is_rejected(self):
        audit = {"status": "complete", "exit_code": 0, "preparation_provenance_sha256": "hash", "entered_pages": ["a.jpg", "b.png"], "completed_pages": ["a.jpg"]}
        with self.assertRaisesRegex(ValueError, "completed_pages"):
            check_execution_audit(audit, ["a.jpg", "b.png"], "hash")

    def test_duplicate_audited_page_is_rejected(self):
        audit = {"status": "complete", "exit_code": 0, "preparation_provenance_sha256": "hash", "entered_pages": ["a.jpg", "a.jpg"], "completed_pages": ["a.jpg", "a.jpg"]}
        with self.assertRaisesRegex(ValueError, "entered_pages"):
            check_execution_audit(audit, ["a.jpg", "b.png"], "hash")

    def test_missing_and_stale_result_files_are_rejected(self):
        path = self.root / "result" / (PREFIX + "reading_order_result.json")
        content = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        path.write_bytes(content)
        write(self.root / "result" / "stale.json", {})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.validate()

    def test_omitted_page_score_and_unknown_page_are_rejected(self):
        self.mutate("text_block_per_page_edit.json", lambda x: x.pop("b.png"))
        with self.assertRaisesRegex(ValueError, "per-page"):
            self.validate()
        make_reports(self.root)
        self.mutate("text_block_result.json", lambda x: x[0].update(image_name="unselected.jpg"))
        with self.assertRaisesRegex(ValueError, "image name"):
            self.validate()

    def test_nonfinite_and_inconsistent_aggregates_are_rejected(self):
        self.mutate("metric_result.json", lambda x: x["text_block"]["all"]["Edit_dist"].update(ALL_page_avg=float("nan")))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.validate()
        make_reports(self.root)
        self.mutate("metric_result.json", lambda x: x["text_block"]["all"]["Edit_dist"].update(edit_whole=0.01))
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            self.validate()

    def set_table_metric(self, metric, value):
        self.mutate("table_result.json", lambda x: x[0]["metric"].update({metric: value}))
        self.mutate("table_per_table_TEDS.json", lambda x: x["b.png_[0]"].update({metric: value}))
        self.mutate("metric_result.json", lambda x: x["table"]["all"][metric].update(all=value))

    def test_upstream_negative_teds_is_preserved_and_included_in_mean(self):
        # Actual pinned v1_5 result, independently replayed with apted 1.0.3:
        # distance 61.88801892551893 / max DOM descendants 54, without clamp.
        negative = -0.14607442454664676
        structure = 0.37037037037037035
        self.assertEqual(negative.hex(), "-0x1.2b291161adcf0p-3")
        self.set_table_metric("TEDS", negative)
        self.set_table_metric("TEDS_structure_only", structure)
        second = row("b.png", 0, 5, gt_idx=[1])
        second["metric"].update(TEDS=0.5, TEDS_structure_only=0.75)
        self.mutate("table_result.json", lambda x: x.append(second))
        self.mutate("table_per_table_TEDS.json", lambda x: x.update({
            "b.png_[1]": {"TEDS": 0.5, "TEDS_structure_only": 0.75}}))
        expected = {"TEDS": statistics.mean([negative, 0.5]),
                    "TEDS_structure_only": statistics.mean([structure, 0.75])}
        for metric, value in expected.items():
            self.mutate("metric_result.json", lambda x: x["table"]["all"][metric].update(all=value))
        before = {p: p.read_bytes() for p in (self.root / "result").iterdir()}
        result = self.validate()
        self.assertEqual(result["components"]["table"]["sample_count"], 2)
        for metric, value in expected.items():
            self.assertEqual(result["metrics"]["table"]["all"][metric]["all"].hex(), value.hex())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        # A mean that clamps the negative sample or excludes it is still invalid.
        for bad_mean in (statistics.mean([0, 0.5]), 0.5):
            self.mutate("metric_result.json", lambda x: x["table"]["all"]["TEDS"].update(all=bad_mean))
            with self.assertRaisesRegex(ValueError, "aggregate differs"):
                self.validate()

    def test_teds_finite_range_endpoints_and_negative_structure_are_valid(self):
        for metric in ("TEDS", "TEDS_structure_only"):
            for value in (-1, -0.14607442454664676, 0, 1):
                with self.subTest(metric=metric, value=value):
                    make_reports(self.root)
                    self.set_table_metric(metric, value)
                    self.assertEqual(self.validate()["metrics"]["table"]["all"][metric]["all"], value)

    def test_teds_out_of_range_and_nonfinite_values_are_rejected_at_every_level(self):
        values = (math.nextafter(-1.0, -math.inf), math.nextafter(1.0, math.inf),
                  float("nan"), float("inf"), -float("inf"), True, "0.5", None)
        for metric in ("TEDS", "TEDS_structure_only"):
            for value in values:
                for level in ("sample", "per-table", "mean"):
                    with self.subTest(metric=metric, value=value, level=level):
                        make_reports(self.root)
                        # At either boundary, close() alone would tolerate the
                        # next representable out-of-range per-table/mean score.
                        base = -1.0 if type(value) is float and value < -1 else 1.0
                        self.set_table_metric(metric, base)
                        if level == "sample":
                            self.mutate("table_result.json", lambda x: x[0]["metric"].update({metric: value}))
                        elif level == "per-table":
                            self.mutate("table_per_table_TEDS.json", lambda x: x["b.png_[0]"].update({metric: value}))
                        else:
                            self.mutate("metric_result.json", lambda x: x["table"]["all"][metric].update(all=value))
                        with self.assertRaisesRegex(ValueError, "Invalid " + metric + " " + level + " value"):
                            self.validate()

    def test_table_and_formula_pair_counts_are_rejected(self):
        self.mutate("table_per_table_TEDS.json", lambda x: x.clear())
        with self.assertRaisesRegex(ValueError, "TEDS"):
            self.validate()
        make_reports(self.root)
        self.mutate("display_formula_formula.json", lambda x: x.append({"img_id": "0"}))
        with self.assertRaisesRegex(ValueError, "formula pair count"):
            self.validate()

    def test_logged_error_is_rejected_even_with_all_pages(self):
        audit = {"status": "complete", "exit_code": 0, "preparation_provenance_sha256": "hash", "entered_pages": ["a.jpg", "b.png"], "completed_pages": ["a.jpg", "b.png"], "detected_error_lines": ["TEDS score error"]}
        with self.assertRaisesRegex(ValueError, "emitted an error"):
            check_execution_audit(audit, ["a.jpg", "b.png"], "hash")


class OfficialInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.predictions = self.root / "cpu"
        self.manifest = self.root / "manifest.json"
        image = self.root / "image" / "canonical.png"
        image.parent.mkdir()
        image.write_bytes(b"test image placeholder; image hash validation only")
        self.page = {"id": "sample", "category": "ordinary", "canonical_path": str(image), "canonical_png_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "ground_truth_sha256": "truth"}
        write(self.manifest, {"pages": [self.page]})
        self.contract = {"manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(), "model_revision": REVISION, "weights_sha256": WEIGHT_SHA256, "precision": "fp32", "options": {"min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 16}}
        self.contract_hash = hashlib.sha256(json.dumps(self.contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write(self.predictions / "run.json", {"contract": self.contract, "contract_sha256": self.contract_hash, "teacher_forced": False})
        self.record = {"id": "sample", "category": "ordinary", "contract_sha256": self.contract_hash, "input_sha256": self.page["canonical_png_sha256"], "ground_truth_sha256": "truth", "result": {"text": "test", "token_ids": [100, 11], "output_tokens": 2, "finish_reason": "eos", "precision": "fp32", "teacher_forced": False}}
        self.record["result"].update(width=64, height=64, input_tokens=32, timings={
            "image_decode_ms": 0.5, "preprocessing_ms": 1.0, "prefill_ms": 12.0,
            "decode_ms": 7.0, "total_ms": 20.5, "time_to_first_token_ms": 13.5})

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_record_prevents_preparation_before_writes(self):
        output = self.root / "prepared"
        args = SimpleNamespace(manifest=self.manifest, predictions=self.predictions, runtime="cpu", output=output, evaluator=self.root, drop_dangling_truncated_relations=False)
        with patch("prepare_official_evaluation.subprocess.check_output", side_effect=[EVALUATOR_REVISION, ""]):
            with self.assertRaisesRegex(ValueError, "missing, failed or invalid"):
                prepare(args)
        self.assertFalse(output.exists())

    def test_failed_or_changed_contract_cannot_pass(self):
        failed = dict(self.record, error="out of memory")
        write(self.predictions / "image.json", failed)
        with self.assertRaisesRegex(ValueError, "failed inference"):
            validate_source_run(self.manifest, self.predictions, "cpu")
        changed = deepcopy(self.record)
        changed["contract_sha256"] = "wrong"
        write(self.predictions / "image.json", changed)
        with self.assertRaisesRegex(ValueError, "contract differs"):
            validate_source_run(self.manifest, self.predictions, "cpu")

    def test_valid_source_and_bad_stop_contract(self):
        write(self.predictions / "image.json", self.record)
        self.assertEqual(len(validate_source_run(self.manifest, self.predictions, "cpu")[3]), 1)
        bad = deepcopy(self.record)
        bad["result"]["finish_reason"] = "length"
        write(self.predictions / "image.json", bad)
        with self.assertRaisesRegex(ValueError, "finish reason"):
            validate_source_run(self.manifest, self.predictions, "cpu")

    def gpu_fixture(self):
        page = dict(self.page, rgb_sha256="a" * 64)
        write(self.manifest, {"pages": [page]})
        configuration = {"manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                         "model_revision": REVISION, "weights_sha256": WEIGHT_SHA256,
                         "precision": "fp32", **self.contract["options"],
                         "tf32": False, "flex_float32_precision": "ieee"}
        run = {"configuration": configuration, "status": "complete", "pages": 1}
        record = {"id": page["id"], "category": page["category"], "configuration": configuration,
                  "canonical_rgb_sha256": page["rgb_sha256"], "text": "test", "token_ids": [100, 11],
                  "finish_reason": "eos", "prefix_length": 32, "cache_capacity": 128,
                  "logit_decisions": [{"step": i, "argmax": token, "runner_up": token + 1,
                                       "winner_logit": 2.0, "runner_up_logit": 1.0, "winner_margin": 1.0}
                                      for i, token in enumerate([100, 11])],
                  "prefill_seconds_including_compile": 0.0, "decode_seconds_including_python_diagnostics": 0.0,
                  "elapsed_seconds": 0.0, "peak_gpu_allocated_bytes": 0}
        write(self.predictions / "run.json", run)
        write(self.predictions / "image.json", record)
        return run, record

    def test_gpu_historical_schema_without_optional_dimensions_is_valid(self):
        self.gpu_fixture()
        self.assertEqual(len(validate_source_run(self.manifest, self.predictions, "gpu")[3]), 1)

    def test_gpu_missing_context_diagnostics_and_nonfinite_values_are_rejected(self):
        _, record = self.gpu_fixture()
        for mutation in [lambda r: r.pop("prefix_length"), lambda r: r.update(cache_capacity=256),
                         lambda r: r.pop("logit_decisions"),
                         lambda r: r["logit_decisions"][0].update(argmax=101),
                         lambda r: r["logit_decisions"][0].update(winner_margin=float("nan")),
                         lambda r: r.update(elapsed_seconds=float("inf"))]:
            changed = deepcopy(record); mutation(changed)
            write(self.predictions / "image.json", changed)
            with self.assertRaisesRegex(ValueError, "gpu_result"):
                validate_source_run(self.manifest, self.predictions, "gpu")

    def test_gpu_teacher_forcing_and_replay_markers_are_rejected(self):
        run, record = self.gpu_fixture()
        for target in ("run", "record", "configuration"):
            for field, value in [("teacher_forced", True), ("postprocessing_replay", {}), ("derived_text_replay", True),
                                 ("inference_reexecuted", False)]:
                with self.subTest(target=target, field=field):
                    changed_run, changed_record = deepcopy(run), deepcopy(record)
                    if target == "configuration":
                        changed_run["configuration"][field] = value
                        changed_record["configuration"][field] = value
                    else:
                        (changed_run if target == "run" else changed_record)[field] = value
                    write(self.predictions / "run.json", changed_run)
                    write(self.predictions / "image.json", changed_record)
                    with self.assertRaisesRegex(ValueError, "teacher-forced|replay"):
                        validate_source_run(self.manifest, self.predictions, "gpu")

    def test_saved_token_replay_preserves_every_nontext_result_field(self):
        source_path = self.predictions / "original.json"
        write(source_path, self.record)
        derived = deepcopy(self.record)
        derived["contract_sha256"] = "derived-contract"
        derived["postprocessing_replay"] = dict(source_record_path=str(source_path), source_record_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), original_text_sha256=hashlib.sha256(b"test").hexdigest(), original_text="test", source_contract_sha256=self.contract_hash, inference_reexecuted=False)
        derived["result"]["text"] = "corrected decode"
        context = {"source_contract_sha256": self.contract_hash, "source_run_path": self.predictions / "run.json", "source_run": {"contract": self.contract}}
        validate_page_replay(derived, context, replay_check, "sample")
        for field, value in [("token_ids", [101, 11]), ("finish_reason", "length"), ("timings", {"total_ms": 0})]:
            bad = deepcopy(derived)
            bad["result"][field] = value
            with self.assertRaisesRegex(ValueError, "inference_result_fields"):
                validate_page_replay(bad, context, replay_check, "sample")

    def test_saved_token_replay_rejects_one_ulp_timing_change(self):
        source_path = self.predictions / "original.json"
        write(source_path, self.record)
        derived = deepcopy(self.record)
        derived["contract_sha256"] = "derived-contract"
        derived["postprocessing_replay"] = dict(source_record_path=str(source_path), source_record_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), original_text_sha256=hashlib.sha256(b"test").hexdigest(), original_text="test", source_contract_sha256=self.contract_hash, inference_reexecuted=False)
        derived["result"]["text"] = "corrected decode"
        derived["result"]["timings"]["total_ms"] = math.nextafter(20.5, math.inf)
        context = {"source_contract_sha256": self.contract_hash, "source_run_path": self.predictions / "run.json", "source_run": {"contract": self.contract}}
        with self.assertRaisesRegex(ValueError, "inference_result_fields"):
            validate_page_replay(derived, context, replay_check, "sample")


if __name__ == "__main__":
    unittest.main()
