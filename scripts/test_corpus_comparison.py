#!/usr/bin/env python3
"""Mutation tests for explicit subset provenance and cross-platform outputs."""
import copy
import hashlib
import json
import pathlib
import tempfile
import types
import unittest
import zipfile
import contextlib
import io
import sys
from unittest.mock import patch

from PIL import Image

from corpus_comparison import (ASSETS, GREEDY, PROMPT, manifest_scope, semantic_contract,
                               cpu_build_evidence, read_snapshot, validate_cpu_result)
from compare_cpu_corpora import compare
from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from validate_text_replay import canonical_sha256


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.pages = []
        for n in range(3):
            folder = self.root / f"image{n}"
            folder.mkdir()
            image = folder / "canonical.png"
            Image.new("RGB", (16, 32), (n, 1, 2)).save(image)
            truth = folder / "ground-truth.txt"
            truth.write_text("A", encoding="utf-8")
            annotation = folder / "annotation.json"
            write(annotation, {"text": "A", "number": n})
            self.pages.append({"id": f"page{n}", "category": "test", "canonical_path": str(image),
                "canonical_png_sha256": sha256(image), "rgb_sha256": hashlib.sha256(Image.open(image).tobytes()).hexdigest(),
                "width": 16, "height": 32, "ground_truth_path": str(truth), "ground_truth_sha256": sha256(truth),
                "annotation_path": str(annotation), "annotation_page_sha256": canonical_sha256({"text": "A", "number": n})})
        self.parent, self.selected = self.root / "parent.json", self.root / "selected.json"
        write(self.parent, {"dataset": "test", "revision": "fixed", "pages": self.pages})
        write(self.selected, {"dataset": "test", "revision": "fixed", "pages": self.pages[:2]})
        self.left, self.right = self.root / "linux", self.root / "windows"
        self.left.mkdir()
        self.right.mkdir()
        self.contract = {"model_revision": REVISION, "weights_sha256": WEIGHT_SHA256, "precision": "fp32",
            "prompt": PROMPT, "greedy_policy": GREEDY, "config_sha256": ASSETS["config.json"], "backend": "avx2",
            "options": {"max_dimension": 1536, "min_dimension": 64, "max_new_tokens": 2}}
        self.result = {"token_ids": [560, 263], "text": "A", "output_tokens": 2, "input_tokens": 18,
            "width": 16, "height": 32, "precision": "fp32", "teacher_forced": False, "finish_reason": "eos",
            "timings": {k: 1.0 for k in ["image_decode_ms", "preprocessing_ms", "prefill_ms", "decode_ms", "total_ms", "time_to_first_token_ms"]}}
        for directory, lock, os, threads in [(self.left, self.selected, "linux", 4), (self.right, self.parent, "windows", 16)]:
            contract = {**self.contract, "manifest_sha256": sha256(lock), "os": os, "threads": threads}
            digest = canonical_sha256(contract)
            write(directory / "run.json", {"contract": contract, "contract_sha256": digest, "teacher_forced": False})
            for n, page in enumerate(self.pages[:2]):
                result = copy.deepcopy(self.result)
                if directory == self.left:
                    result["timings"]["image_projection_ms"] = 0.25
                    result["timings"]["prefill_ms"] = 1.01
                write(directory / f"image{n}.json", {"id": page["id"], "category": "test", "contract_sha256": digest,
                    "input_sha256": page["canonical_png_sha256"], "ground_truth_sha256": page["ground_truth_sha256"], "result": result})
        self.args = types.SimpleNamespace(manifest=self.selected, left=self.left, right=self.right,
            left_manifest=None, right_manifest=self.parent, left_build=None, right_build=None,
            precision="fp32", model=self.root / "unused")

    def comparison(self):
        # Fixture tests exercise source membership/result/replay logic; actual
        # pinned model asset bytes are independently checked by integration runs.
        with patch("compare_cpu_corpora.model_asset_evidence", return_value={"scope": "unit test fixture"}):
            return compare(self.args)

    def scope_failures(self, path=None, digest=None):
        failures = []
        manifest_scope(self.selected, {"test": (path, digest)},
                       lambda name, actual, expected: failures.append(name) if actual != expected else None)
        return failures

    def test_explicit_subset_and_different_threads_timings_pass(self):
        report = self.comparison()
        self.assertTrue(report["completed_output_parity_passed"])
        self.assertEqual(report["exact_pages"], 2)
        self.assertFalse(report["comparison_scope"]["sources"]["right"]["entire_source_selected"])
        self.assertTrue(report["comparison_scope"]["sources"]["left"]["entire_source_selected"])

    def test_snapshot_hash_uses_the_parsed_bytes_once(self):
        payload = b'{"observed":"same bytes"}'
        with patch.object(pathlib.Path, "read_bytes", return_value=payload) as read:
            value, digest = read_snapshot(self.root / "not-a-real-file.json")
        read.assert_called_once()
        self.assertEqual(value, {"observed": "same bytes"})
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_defaults_remain_same_manifest_strict(self):
        self.args.right_manifest = None
        report = self.comparison()
        self.assertFalse(report["completed_output_parity_passed"])
        self.assertTrue(any(c["name"] == "right.source_manifest_sha256" and not c["passed"] for c in report["checks"]))

    def test_wrong_parent_hash_rejected(self):
        self.assertTrue(self.scope_failures(self.parent, "0" * 64))

    def test_wrong_parent_content_even_with_its_hash_rejected(self):
        value = json.loads(self.parent.read_text())
        value["pages"][0]["id"] = "unrelated"
        write(self.parent, value)
        self.assertTrue(self.scope_failures(self.parent, sha256(self.parent)))

    def test_changed_page_annotations_hashes_or_metadata_rejected(self):
        for field, value in [("annotation_page_sha256", "0" * 64), ("rgb_sha256", "0" * 64), ("category", "changed")]:
            with self.subTest(field=field):
                selected = {"dataset": "test", "revision": "fixed", "pages": copy.deepcopy(self.pages[:2])}
                selected["pages"][0][field] = value
                write(self.selected, selected)
                self.assertTrue(self.scope_failures(self.parent, sha256(self.parent)))

    def test_duplicate_id_and_duplicate_image_key_rejected(self):
        for change in ["id", "canonical_path"]:
            selected = {"pages": copy.deepcopy(self.pages[:2])}
            selected["pages"][1][change] = selected["pages"][0][change]
            write(self.selected, selected)
            self.assertTrue(self.scope_failures(self.selected, sha256(self.selected)))

    def test_empty_selection_cannot_qualify(self):
        write(self.selected, {"pages": []})
        self.assertTrue(self.scope_failures(self.selected, sha256(self.selected)))

    def test_missing_pages_are_partial_and_explicit(self):
        (self.left / "image1.json").unlink()
        report = self.comparison()
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["completed_output_parity_passed"])
        self.assertEqual(report["missing_pages"]["left"], [{"sample_id": "image1", "id": "page1"}])

    def test_malformed_results_cannot_be_complete(self):
        for value in [{}, {**self.result, "output_tokens": 1}, {**self.result, "token_ids": [263, 560]},
                      {**self.result, "timings": []}, {**self.result, "text": None}]:
            with self.subTest(value=value):
                path = self.left / "image0.json"
                record = json.loads(path.read_text())
                record["result"] = value
                write(path, record)
                report = self.comparison()
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["selected_pages_complete"])
                self.assertTrue(report["failed_pages"]["left"])

    def test_non_json_page_is_failed_not_missing(self):
        (self.left / "image0.json").write_text("[not-json")
        report = self.comparison()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["failed_pages"]["left"])
        self.assertFalse(report["missing_pages"]["left"])

    def test_text_stop_count_dimensions_are_compared(self):
        for field, value in [("text", "A "), ("width", 32), ("output_tokens", 1), ("finish_reason", "length")]:
            with self.subTest(field=field):
                path = self.left / "image0.json"
                record = json.loads(path.read_text())
                record["result"] = {**self.result, field: value}
                write(path, record)
                self.assertFalse(self.comparison()["completed_output_parity_passed"])

    def test_unbound_page_replay_is_rejected(self):
        path = self.right / "image0.json"
        record = json.loads(path.read_text())
        record["postprocessing_replay"] = {"inference_reexecuted": False}
        write(path, record)
        self.assertFalse(self.comparison()["completed_output_parity_passed"])

    def test_unbound_run_replay_is_rejected(self):
        path = self.right / "run.json"
        run = json.loads(path.read_text())
        run["derived_text_replay"] = True
        write(path, run)
        self.assertFalse(self.comparison()["completed_output_parity_passed"])

    def test_wrong_recorded_prompt_or_argmax_rejected(self):
        for key in ["prompt", "greedy_policy", "config_sha256"]:
            failures = []
            semantic_contract({**self.contract, key: "changed"}, "test", "fp32",
                lambda n, a, e: failures.append(n) if a != e else None)
            self.assertIn("test." + key, failures)

    def test_missing_historical_semantics_are_explicit_not_filled(self):
        old = {k: v for k, v in self.contract.items() if k not in ["prompt", "greedy_policy", "config_sha256"]}
        before = copy.deepcopy(old)
        evidence = semantic_contract(old, "old", "fp32", lambda *args: None)
        self.assertEqual(set(evidence["missing_startup_fields"]), {"prompt", "greedy_policy", "config_sha256"})
        self.assertEqual(old, before)

    def test_build_proof_rejects_changed_binary_or_archive(self):
        folder = self.root / "build"
        folder.mkdir()
        binary = folder / "corpus_eval"
        binary.write_bytes(b"preserved executable fixture")
        archive = folder / "source.zip"
        source = b"preserved source fixture"
        digest = hashlib.sha256(source).hexdigest()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("src/model.rs", source)
        build = {"status": "complete", "source_unchanged_during_build": True, "binary": binary.name,
                 "binary_sha256": sha256(binary), "source_archive": archive.name, "source_archive_sha256": sha256(archive),
                 "source_sha256": {"src/model.rs": digest}}
        path = folder / "build.json"
        write(path, build)
        contract = {"build_manifest_sha256": sha256(path), "binary_sha256": sha256(binary), "source_sha256": {"model": digest}}
        failures = []
        check = lambda n, a, e: failures.append(n) if a != e else None
        cpu_build_evidence(path, contract, check, "test")
        self.assertEqual(failures, [])
        binary.write_bytes(b"changed executable")
        cpu_build_evidence(path, contract, check, "test")
        self.assertIn("test.binary_file", failures)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("src/model.rs", b"changed source")
        failures.clear()
        cpu_build_evidence(path, contract, check, "test")
        self.assertIn("test.source_archive", failures)
        self.assertIn("test.archived_source.src/model.rs", failures)

    def gpu_fixture(self):
        directory = self.root / "gpu"
        directory.mkdir(exist_ok=True)
        config = {k: self.contract[k] for k in ["model_revision", "weights_sha256", "precision"]}
        config.update(self.contract["options"])
        config.update(manifest_sha256=sha256(self.parent), tf32=False, flex_float32_precision="ieee")
        write(directory / "run.json", {"configuration": config})
        for n, page in enumerate(self.pages[:2]):
            write(directory / f"image{n}.json", {"id": page["id"], "category": page["category"], "sample_id": f"image{n}",
                "configuration": config, "canonical_rgb_sha256": page["rgb_sha256"], "prefix_length": 18, "cache_capacity": 128,
                "token_ids": [560, 263], "text": "A", "finish_reason": "eos", "diagnostic_character_edit_distance": 0,
                "logit_decisions": [{"step": i, "argmax": token, "runner_up": 11, "winner_logit": 2.0,
                                     "runner_up_logit": 1.0, "winner_margin": 1.0} for i, token in enumerate([560, 263])],
                "prefill_seconds_including_compile": 1.0, "decode_seconds_including_python_diagnostics": 1.0,
                "elapsed_seconds": 2.0, "peak_gpu_allocated_bytes": 1234})
        return directory

    def gpu_compare(self, directory, explicit_parent):
        try:
            import compare_corpus
        except ModuleNotFoundError as error:
            if error.name == "rapidfuzz":
                self.skipTest("GPU comparator CLI tests require the pinned reference environment's rapidfuzz")
            raise
        output = self.root / f"gpu-report-{len(list(self.root.glob('gpu-report-*.json')))}.json"
        argv = ["compare_corpus", "--manifest", str(self.selected), "--cpu", str(self.left), "--gpu", str(directory), "--output", str(output)]
        if explicit_parent:
            argv += ["--gpu-manifest", str(self.parent)]
        with patch.object(sys, "argv", argv), patch("compare_corpus.model_asset_evidence", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.last_gpu_exit_code = 0
            try:
                compare_corpus.main()
            except SystemExit as error:
                self.last_gpu_exit_code = error.code
                self.assertEqual(error.code, 1)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_gpu_cli_explicit_parent_and_strict_default(self):
        directory = self.gpu_fixture()
        report = self.gpu_compare(directory, True)
        self.assertTrue(report["completed_output_parity_passed"])
        self.assertFalse(report["comparison_scope"]["sources"]["gpu"]["entire_source_selected"])
        self.assertFalse(report["startup_semantic_fields_complete"])
        self.assertFalse(self.gpu_compare(directory, False)["completed_output_parity_passed"])

    def test_gpu_cli_empty_ids_or_replay_cannot_qualify(self):
        directory = self.gpu_fixture()
        path = directory / "image0.json"
        original = json.loads(path.read_text())
        for mutation in [{"token_ids": []}, {"postprocessing_replay": {"inference_reexecuted": False}}]:
            write(path, {**original, **mutation})
            report = self.gpu_compare(directory, True)
            self.assertFalse(report["completed_output_parity_passed"])
            self.assertEqual(report["status"], "failed")

    def test_gpu_cli_partial_output_mismatch_exits_failure(self):
        directory = self.gpu_fixture()
        (self.left / "image1.json").unlink()
        path = self.left / "image0.json"
        record = json.loads(path.read_text())
        record["result"]["text"] = "A "
        write(path, record)
        report = self.gpu_compare(directory, True)
        self.assertEqual(report["compared_pages"], 1)
        self.assertEqual(report["missing_cpu_count"], 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.last_gpu_exit_code, 1)
        self.assertFalse(report["completed_output_parity_passed"])


if __name__ == "__main__":
    unittest.main()
