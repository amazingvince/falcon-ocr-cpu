"""Bounded saved-record tests: no model execution or evaluator invocation."""
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research/corpus-qualification/scripts"))  # report_quality_regression moved here in the 2026-09-24 archive

import report_quality_regression as quality


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class QualityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, categories=None, different=False, normalized_only=False):
        categories = categories or {"ordinary": 2, "tables": 2}
        pages, rows, records = [], [], []
        cpu_dir, gpu_dir = self.root / "cpu", self.root / "gpu"
        cpu_dir.mkdir(); gpu_dir.mkdir()
        for category, count in categories.items():
            for _ in range(count):
                i = len(pages); key = f"page{i:03}"
                directory = self.root / key; directory.mkdir()
                image = directory / "canonical.png"; image.write_bytes(b"synthetic source bytes" + str(i).encode())
                truth = directory / "truth.txt"; truth.write_text("Caf\u00e9 text", encoding="utf-8")
                page = {"id": key, "category": category, "canonical_path": str(image),
                        "canonical_png_sha256": quality.sha(image), "rgb_sha256": "a" * 64,
                        "ground_truth_path": str(truth), "ground_truth_sha256": quality.sha(truth)}
                pages.append(page)
                cpu_text = "Caf\u00e9 changed" if different and i == 0 else "Caf\u00e9 text"
                if normalized_only and i == 0:
                    cpu_text = "Cafe\u0301\n  text"
                records.append((page, cpu_text))
        manifest = self.root / "manifest.json"; write(manifest, {"pages": pages})
        manifest_hash = quality.sha(manifest)
        shared = {"precision": "fp32", "manifest_sha256": manifest_hash,
                  "weights_sha256": quality.WEIGHT_SHA256, "model_revision": quality.REVISION}
        options = {"min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}
        gpu_config = {**shared, **options, "tf32": False, "flex_float32_precision": "ieee"}; cpu_config = {**shared, "options": options}
        contract_hash = quality.canonical_sha256(cpu_config)
        cpu_run = {"contract": cpu_config, "contract_sha256": contract_hash, "teacher_forced": False}
        gpu_run = {"configuration": gpu_config}
        write(cpu_dir / "run.json", cpu_run); write(gpu_dir / "run.json", gpu_run)
        for page, cpu_text in records:
            key = page["id"]
            cpu = {"id": key, "category": page["category"], "contract_sha256": contract_hash,
                   "ground_truth_sha256": page["ground_truth_sha256"], "input_sha256": page["canonical_png_sha256"],
                   "result": {"text": cpu_text, "teacher_forced": False, "precision": "fp32", "width": 64, "height": 64,
                              "input_tokens": 32, "output_tokens": 2, "token_ids": [100, 263], "finish_reason": "eos",
                              "timings": {name: 0.0 for name in ("image_decode_ms", "preprocessing_ms", "prefill_ms", "decode_ms", "total_ms", "time_to_first_token_ms")}}}
            gpu = {"id": key, "category": page["category"], "configuration": gpu_config,
                   "canonical_rgb_sha256": page["rgb_sha256"], "text": "Caf\u00e9 text", "token_ids": [100, 263],
                   "finish_reason": "eos", "prefix_length": 32, "cache_capacity": 4224, "width": 64, "height": 64,
                   "logit_decisions": [{"step": i, "argmax": token, "runner_up": token + 1,
                                       "winner_logit": 2.0, "runner_up_logit": 1.0, "winner_margin": 1.0} for i, token in enumerate((100, 263))],
                   "prefill_seconds_including_compile": 0.0, "decode_seconds_including_python_diagnostics": 0.0,
                   "elapsed_seconds": 0.0, "peak_gpu_allocated_bytes": 0}
            write(cpu_dir / (key + ".json"), cpu); write(gpu_dir / (key + ".json"), gpu)
            edits = quality.Levenshtein.distance("Caf\u00e9 text", quality.normalized(cpu_text))
            rows.append({"id": key, "sample_id": key, "category": page["category"], "provenance_passed": True,
                "cpu_record_sha256": quality.sha(cpu_dir / (key + ".json")), "gpu_record_sha256": quality.sha(gpu_dir / (key + ".json")),
                "text_exact": cpu_text == gpu["text"], "tokens_exact": True,
                "diagnostic_ground_truth_characters": 9, "cpu_diagnostic_edit_distance": edits,
                "gpu_diagnostic_edit_distance": 0, "cpu_diagnostic_cer": edits / 9, "gpu_diagnostic_cer": 0.0})
        comparison = {"schema_version": 2, "status": "failed" if different or normalized_only else "complete",
            "manifest_sha256": manifest_hash, "expected_pages": len(pages), "compared_pages": len(pages),
            "provenance_passed": True, "checks": [{"passed": True}], "failed_cpu_pages": [], "failed_gpu_pages": [],
            "failed_cpu_count": 0, "failed_gpu_count": 0, "missing_cpu": [], "missing_gpu": [],
            "missing_cpu_count": 0, "missing_gpu_count": 0, "gpu_configuration": gpu_config, "cpu_contract": cpu_config,
            "cpu_run": str(cpu_dir / "run.json"), "gpu_run": str(gpu_dir / "run.json"),
            "cpu_run_sha256": quality.sha(cpu_dir / "run.json"), "gpu_run_sha256": quality.sha(gpu_dir / "run.json"), "pages": rows}
        path = self.root / "comparison.json"; write(path, comparison)
        return manifest, path, manifest_hash, categories

    def evaluate(self, fixture):
        manifest, comparison, digest, categories = fixture
        return quality.report(manifest, comparison, expected_manifest_sha256=digest, expected_categories=categories)

    def test_complete200_literal_identity_proves_zero_without_choosing_metric(self):
        result = self.evaluate(self.fixture(quality.V3_CATEGORIES))
        self.assertEqual(result["status"], "zero_differential_proved")
        self.assertEqual(result["compared_pages"], 200)
        self.assertTrue(result["nonquantized_differential_gate"]["passed"])
        self.assertIsNone(result["nonquantized_differential_gate"]["primary_overall_metric"])
        self.assertFalse(result["reporting_conditions"]["fixed_corpus_quality_reporting_complete"])
        self.assertEqual(result["descriptive_wer"]["cpu"]["micro_wer"], 0.0)
        self.assertIn("not a preregistered", result["wer_definition"]["introduced"])
        self.assertNotIn("Caf\u00e9", json.dumps(result, ensure_ascii=False))

    def test_complete_nonidentical_failed_parity_is_valid_unresolved_quality(self):
        result = self.evaluate(self.fixture(different=True))
        self.assertEqual(result["status"], "nonidentical_metric_decision_unresolved")
        self.assertFalse(result["nonquantized_differential_gate"]["passed"])
        self.assertGreater(result["diagnostic_cer"]["micro_cer_cpu_minus_gpu_percentage_points"], 0)
        self.assertGreater(result["descriptive_wer"]["micro_wer_cpu_minus_gpu_percentage_points"], 0)

    def test_normalized_equality_is_not_literal_metric_universal_proof(self):
        result = self.evaluate(self.fixture(normalized_only=True))
        self.assertEqual(result["diagnostic_cer"]["micro_cer_cpu_minus_gpu_percentage_points"], 0)
        self.assertFalse(result["nonquantized_differential_gate"]["passed"])

    def test_token_parity_failure_does_not_override_identical_text_quality(self):
        f = self.fixture(); comparison = quality.read(f[1])
        comparison["status"] = "failed"; comparison["pages"][0]["tokens_exact"] = False
        path = self.root / "cpu/page000.json"; record = quality.read(path)
        record["result"]["token_ids"] = [101, 263]; write(path, record)
        comparison["pages"][0]["cpu_record_sha256"] = quality.sha(path)
        write(f[1], comparison)
        self.assertTrue(self.evaluate(f)["nonquantized_differential_gate"]["passed"])

    def test_partial_with_entire_category_missing_cannot_pass(self):
        f = self.fixture(); comparison = quality.read(f[1])
        missing = [p for p in comparison["pages"] if p["category"] == "tables"]
        comparison["pages"] = [p for p in comparison["pages"] if p["category"] != "tables"]
        comparison["compared_pages"] = 2; comparison["missing_cpu"] = [p["sample_id"] for p in missing]
        comparison["missing_cpu_count"] = 2; comparison["status"] = "partial"
        write(f[1], comparison)
        result = self.evaluate(f)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["by_category"]["tables"]["complete"])
        self.assertIsNone(result["by_category"]["tables"]["differential_percentage_points"])
        self.assertIsNone(result["by_category"]["tables"]["diagnostic_cer"]["cpu"]["micro_cer"])

    def test_changed_prediction_record_or_source_is_rejected(self):
        f = self.fixture(); record = self.root / "cpu/page000.json"
        original = record.read_bytes(); record.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Source/hash changed"):
            self.evaluate(f)
        record.write_bytes(original)
        (self.root / "page000/truth.txt").write_text("changed truth")
        with self.assertRaisesRegex(ValueError, "Source/hash changed"):
            self.evaluate(f)

    def test_manifest_category_or_run_mutation_is_rejected(self):
        f = self.fixture(); manifest = quality.read(f[0]); manifest["pages"][0]["category"] = "other"
        write(f[0], manifest)
        with self.assertRaisesRegex(ValueError, "Source/hash changed"):
            self.evaluate(f)

    def test_nonfinite_comparison_metric_is_rejected(self):
        f = self.fixture(); comparison = quality.read(f[1])
        for value in (float("nan"), float("inf"), -float("inf")):
            comparison["pages"][0]["gpu_diagnostic_cer"] = value; write(f[1], comparison)
            with self.assertRaisesRegex(ValueError, "nonfinite metric"):
                self.evaluate(f)
        with self.assertRaisesRegex(ValueError, "nonfinite metric"):
            quality.finite_metrics({"official": {"TEDS": float("nan")}})

    def test_malformed_inference_cannot_pass_with_rehashed_comparison(self):
        f = self.fixture(); original_comparison = quality.read(f[1])
        for side in ("cpu", "gpu"):
            path = self.root / side / "page000.json"; original = quality.read(path)
            changed = copy.deepcopy(original)
            (changed["result"] if side == "cpu" else changed).pop("token_ids")
            write(path, changed)
            comparison = copy.deepcopy(original_comparison)
            comparison["pages"][0][side + "_record_sha256"] = quality.sha(path); write(f[1], comparison)
            with self.assertRaisesRegex(ValueError, "Inference/lineage validation"):
                self.evaluate(f)
            write(path, original)

    def test_forged_replay_flag_cannot_pass_with_rehashed_run(self):
        f = self.fixture(); comparison = quality.read(f[1]); path = self.root / "cpu/run.json"
        run = quality.read(path); run["derived_text_replay"] = True; write(path, run)
        comparison["cpu_run_sha256"] = quality.sha(path); write(f[1], comparison)
        with self.assertRaisesRegex(ValueError, "Inference/lineage validation"):
            self.evaluate(f)

    def test_changed_replay_ancestor_rejected_despite_consistent_new_run_hashes(self):
        f = self.fixture(); comparison = quality.read(f[1]); path = self.root / "cpu/run.json"
        run = quality.read(path); ancestor = self.root / "original-run.json"; write(ancestor, run)
        original_digest = quality.sha(ancestor)
        ancestor.write_bytes(ancestor.read_bytes() + b" ")
        run["contract"]["postprocessing_replay"] = {"policy": "synthetic text replay",
            "source_run_path": str(ancestor), "source_run_sha256": original_digest}
        run["contract_sha256"] = quality.canonical_sha256(run["contract"])
        run["derived_text_replay"] = True; write(path, run)
        comparison["cpu_run_sha256"] = quality.sha(path)
        comparison["cpu_contract"] = run["contract"]; write(f[1], comparison)
        with self.assertRaisesRegex(ValueError, "source_run.sha256"):
            self.evaluate(f)

    def test_consistently_relabelled_model_is_rejected_by_independent_pins(self):
        f = self.fixture(); comparison = quality.read(f[1])
        for side, field in (("cpu", "contract"), ("gpu", "configuration")):
            path = self.root / side / "run.json"; run = quality.read(path)
            run[field]["weights_sha256"] = "d" * 64
            if side == "cpu":
                run["contract_sha256"] = quality.canonical_sha256(run[field])
            write(path, run); comparison[side + "_run_sha256"] = quality.sha(path)
            comparison["cpu_contract" if side == "cpu" else "gpu_configuration"] = run[field]
        write(f[1], comparison)
        with self.assertRaisesRegex(ValueError, "weights_sha256"):
            self.evaluate(f)

    def test_summary_labels_do_not_override_fresh_semantic_evidence(self):
        f = self.fixture(); comparison = quality.read(f[1])
        comparison["derived_text_replay"] = True
        comparison["startup_semantic_fields_complete"] = True
        write(f[1], comparison)
        result = self.evaluate(f)
        self.assertEqual(result["prediction_origin"], "original inference")
        self.assertFalse(result["startup_semantic_fields_complete"])

    def test_empty_truth_metrics_are_absent_with_separate_emission_counts(self):
        rows = [{"ground_truth_characters": 0, "ground_truth_words": 0,
                 "cpu_character_edits": 14, "gpu_character_edits": 14,
                 "cpu_word_edits": 1, "gpu_word_edits": 1}]
        cer, wer = quality.aggregate(rows), quality.aggregate(rows, words=True)
        self.assertIsNone(cer["cpu"]["micro_cer"])
        self.assertIsNone(wer["cpu"]["page_mean_wer"])
        self.assertEqual(cer["cpu"]["empty_truth_prediction_characters"], 14)
        self.assertEqual(wer["cpu"]["empty_truth_prediction_words"], 1)
        self.assertEqual(cer["cpu"]["all_page_character_edits"], 14)
        self.assertEqual(cer["cpu"]["zero_truth_character_edits"], 14)
        self.assertEqual(cer["cpu"]["positive_denominator_character_edits"], 0)
        self.assertEqual(wer["cpu"]["all_page_word_edits"], 1)

    def test_failed_provenance_failed_page_or_missing_inventory_is_rejected(self):
        f = self.fixture(); original = quality.read(f[1])
        mutations = [lambda c: c["checks"][0].update(passed=False),
                     lambda c: c.update(failed_cpu_count=1),
                     lambda c: c.update(missing_cpu=["page000"], missing_cpu_count=1)]
        for mutate in mutations:
            value = copy.deepcopy(original); mutate(value); write(f[1], value)
            with self.assertRaises(ValueError):
                self.evaluate(f)

    def official_fixture(self, fixture):
        """Only the audited-comparator boundary is mocked; joins use real files."""
        comparison = quality.read(fixture[1])
        roots, digests, audits, result_files = {}, {}, {}, {}
        for side in ("gpu", "cpu"):
            root = self.root / ("official-" + side)
            (root / "predictions").mkdir(parents=True)
            (root / "result").mkdir()
            write(root / "configuration.json", {"synthetic": True})
            write(root / "ground-truth.json", [])
            write(root / "result/metric_result.json", {"synthetic_score": 0.0})
            (root / "execution.log").write_text("Synthetic audit boundary; evaluator never invoked.\n", encoding="utf-8")
            result_digest = quality.sha(root / "result/metric_result.json")
            result_files.setdefault("metric_result.json", {})[side + "_sha256"] = result_digest
            write(root / "execution-audit.json", {"execution_log_sha256": quality.sha(root / "execution.log"),
                  "result_file_sha256": {"metric_result.json": result_digest}})
            audits[side] = quality.sha(root / "execution-audit.json")
            pages = []
            for row in comparison["pages"]:
                record_path = self.root / side / (row["sample_id"] + ".json")
                record = quality.read(record_path)
                text = (record["result"] if side == "cpu" else record)["text"]
                filename = row["sample_id"] + ".md"
                prediction = root / "predictions" / filename
                prediction.write_bytes(text.encode("utf-8"))
                pages.append({"id": row["id"], "category": row["category"], "record_path": str(record_path),
                              "record_sha256": quality.sha(record_path), "prediction_filename": filename,
                              "prediction_sha256": quality.sha(prediction)})
            write(root / "provenance.json", {"run_path": comparison[side + "_run"],
                  "run_sha256": comparison[side + "_run_sha256"], "pages": pages,
                  "configuration_sha256": quality.sha(root / "configuration.json"),
                  "ground_truth_sha256": quality.sha(root / "ground-truth.json")})
            roots[side] = root
            digests[side] = quality.sha(root / "provenance.json")
        official = {"inference_contracts": {"gpu": comparison["gpu_configuration"], "cpu": comparison["cpu_contract"]},
                    "all_prediction_text_bytes_equal": True, "all_evaluator_outputs_equal": True,
                    "preparation_provenance_sha256": digests, "execution_audit_sha256": audits,
                    "result_files": result_files}
        return roots, official

    def evaluate_official(self, fixture, roots, official):
        with patch("compare_official_evaluation.compare", return_value=official):
            return quality.report(fixture[0], fixture[1], expected_manifest_sha256=fixture[2],
                                  expected_categories=fixture[3], official_gpu=roots["gpu"], official_cpu=roots["cpu"])

    def test_official_components_bind_exact_runs_records_and_prediction_bytes(self):
        fixture = self.fixture(); roots, official = self.official_fixture(fixture)
        result = self.evaluate_official(fixture, roots, official)
        self.assertTrue(result["reporting_conditions"]["fixed_corpus_quality_reporting_complete"])
        evidence = result["structure_metrics"]["prediction_source_binding"]
        for side in ("cpu", "gpu"):
            self.assertEqual(evidence[side]["joined_pages"], 4)
            self.assertIn(str((roots[side] / "provenance.json").resolve()), result["source_sha256"])
            self.assertIn(str((roots[side] / "predictions/page000.md").resolve()), result["source_sha256"])
            for filename in ("execution-audit.json", "execution.log", "configuration.json", "ground-truth.json", "result/metric_result.json"):
                self.assertIn(str((roots[side] / filename).resolve()), result["source_sha256"])
        self.assertNotIn("Caf\u00e9", json.dumps(result, ensure_ascii=False))

    def test_same_configuration_and_equality_flag_do_not_identify_official_predictions(self):
        fixture = self.fixture(); roots, official = self.official_fixture(fixture)
        for side in ("cpu", "gpu"):
            path = roots[side] / "provenance.json"; original = path.read_bytes()
            for kind in ("other_run", "record_hash", "record_path", "prediction_hash"):
                with self.subTest(side=side, kind=kind):
                    provenance = json.loads(original)
                    if kind == "other_run":
                        other = self.root / (side + "-other-run.json")
                        other.write_bytes(pathlib.Path(provenance["run_path"]).read_bytes())
                        provenance["run_path"] = str(other)  # Identical config and run bytes, different inference identity.
                    elif kind == "record_hash":
                        provenance["pages"][0]["record_sha256"] = "0" * 64
                    elif kind == "record_path":
                        other = self.root / (side + "-other-record.json")
                        other.write_bytes(pathlib.Path(provenance["pages"][0]["record_path"]).read_bytes())
                        provenance["pages"][0]["record_path"] = str(other)
                    else:
                        provenance["pages"][0]["prediction_sha256"] = hashlib.sha256(b"different literal prediction").hexdigest()
                    write(path, provenance)
                    official["preparation_provenance_sha256"][side] = quality.sha(path)
                    with self.assertRaisesRegex(ValueError, "Official (source run|source record|literal prediction)"):
                        self.evaluate_official(fixture, roots, official)
                    path.write_bytes(original)
                    official["preparation_provenance_sha256"][side] = quality.sha(path)

    def test_official_missing_duplicate_or_changed_identity_join_is_rejected(self):
        fixture = self.fixture(); roots, official = self.official_fixture(fixture)
        path = roots["gpu"] / "provenance.json"; original = path.read_bytes()
        for kind in ("missing", "duplicate", "unknown", "category"):
            provenance = json.loads(original)
            if kind == "missing": provenance["pages"].pop()
            elif kind == "duplicate": provenance["pages"][-1] = copy.deepcopy(provenance["pages"][0])
            elif kind == "unknown": provenance["pages"][0]["id"] = "unknown"
            else: provenance["pages"][0]["category"] = "changed"
            write(path, provenance)
            official["preparation_provenance_sha256"]["gpu"] = quality.sha(path)
            with self.assertRaisesRegex(ValueError, "Official prediction (join|category)"):
                self.evaluate_official(fixture, roots, official)

    def test_official_prediction_file_changed_after_comparison_is_rejected(self):
        fixture = self.fixture(); roots, official = self.official_fixture(fixture)
        (roots["cpu"] / "predictions/page000.md").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Source/hash changed"):
            self.evaluate_official(fixture, roots, official)

    def test_late_official_result_mutation_is_rejected_by_final_source_window(self):
        fixture = self.fixture(); roots, official = self.official_fixture(fixture)
        original_join = quality.bind_official_predictions
        def mutate_after_join(*args, **kwargs):
            result = original_join(*args, **kwargs)
            path = roots["gpu"] / "result/metric_result.json"
            path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(quality, "bind_official_predictions", side_effect=mutate_after_join):
            with self.assertRaisesRegex(ValueError, "Source changed during quality accounting"):
                self.evaluate_official(fixture, roots, official)


if __name__ == "__main__":
    unittest.main()
