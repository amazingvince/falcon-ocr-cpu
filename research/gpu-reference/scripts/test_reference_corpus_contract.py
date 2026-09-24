#!/usr/bin/env python3
import copy
import pathlib
import tempfile
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from reference_corpus_contract import atomic_json, prepare_output, read_snapshot, selected_pages, validate_completed_page


class CorpusContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.page = {"id": "source-page", "canonical_path": "images/sample-a/rgb.png", "ground_truth_path": "images/sample-a/gt.txt",
                     "canonical_png_sha256": "a" * 64, "ground_truth_sha256": "b" * 64, "rgb_sha256": "c" * 64, "category": "ordinary"}
        self.config = {"max_new_tokens": 4, "precision": "fp32", "teacher_forced": False, "inference_reexecuted": True}
        self.record = {"id": self.page["id"], "sample_id": "sample-a", "category": self.page["category"],
                       "configuration": self.config, "teacher_forced": False, "inference_reexecuted": True,
                       "token_ids": [500, 263], "text": "example", "finish_reason": "eos", "prefix_length": 144,
                       "cache_capacity": 256, "canonical_rgb_sha256": self.page["rgb_sha256"],
                       "logit_decisions": [{"step": i, "argmax": token, "runner_up": 12, "winner_logit": 2.0,
                                            "runner_up_logit": 1.0, "winner_margin": 1.0} for i, token in enumerate([500, 263])],
                       "prefill_seconds_including_compile": 1.0, "decode_seconds_including_python_diagnostics": 2.0,
                       "elapsed_seconds": 4.0, "peak_gpu_allocated_bytes": 1000}

    def tearDown(self):
        self.temporary.cleanup()

    def test_bad_selection_and_output_collisions(self):
        for limit in [0, -1, 2, True]:
            with self.assertRaises(ValueError):
                selected_pages({"pages": [self.page]}, limit)
        for pages in [[], [self.page, self.page]]:
            with self.assertRaises(ValueError):
                selected_pages({"pages": pages})
        for name in ["run", "summary", "provenance", "../run", "file:stream"]:
            page = dict(self.page, canonical_path=f"images/{name}/rgb.png")
            with self.assertRaises(ValueError):
                selected_pages({"pages": [page]})

    def test_exact_selected_order(self):
        second = dict(self.page, id="source-two", canonical_path="images/sample-b/rgb.png")
        self.assertEqual(selected_pages({"pages": [self.page, second]}, 1), [self.page])

    def test_changed_configuration_rejected_before_archive(self):
        atomic_json(self.root / "run.json", {"configuration": self.config})
        with mock.patch("reference_corpus_contract.preserve_reference_identity") as archive:
            with self.assertRaisesRegex(ValueError, "different frozen"):
                prepare_output(self.root, dict(self.config, max_new_tokens=5), {}, {}, resume=True)
            archive.assert_not_called()

    def test_accidental_overwrite_and_missing_resume_rejected(self):
        with self.assertRaisesRegex(ValueError, "existing run"):
            prepare_output(self.root, self.config, {}, {}, resume=True)
        (self.root / "previous.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            prepare_output(self.root, self.config, {}, {}, resume=False)

    def test_resume_requires_identity_binding(self):
        atomic_json(self.root / "run.json", {"configuration": self.config, "teacher_forced": False, "inference_reexecuted": True})
        with mock.patch("reference_corpus_contract.preserve_reference_identity", return_value={"startup_identity_sha256": "a" * 64}):
            with self.assertRaisesRegex(ValueError, "bind its preserved"):
                prepare_output(self.root, self.config, {}, {}, resume=True)

    def test_nonfinite_json_does_not_replace_existing_record(self):
        target = self.root / "page.json"
        atomic_json(target, {"status": "preserved"})
        before = target.read_bytes()
        with self.assertRaises(ValueError):
            atomic_json(target, {"value": float("nan")})
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(target.with_name("page.json.tmp").exists())
        self.assertEqual(read_snapshot(target)[0], {"status": "preserved"})

    def test_malformed_record_cannot_resume_as_complete(self):
        with self.assertRaisesRegex(ValueError, "token_ids"):
            validate_completed_page({"configuration": {"max_new_tokens": 4}}, self.page, {"max_new_tokens": 4})

    def test_derived_or_teacher_forced_records_rejected_at_all_levels(self):
        validate_completed_page(self.record, self.page, self.config)
        for name, value in [("teacher_forced", True), ("inference_reexecuted", False),
                            ("postprocessing_replay", {}), ("derived_text_replay", True)]:
            record = copy.deepcopy(self.record)
            record[name] = value
            with self.assertRaises(ValueError):
                validate_completed_page(record, self.page, self.config)
            config = dict(self.config, **{name: value})
            record = dict(self.record, configuration=config)
            with self.assertRaises(ValueError):
                validate_completed_page(record, self.page, config)
            run = {"configuration": self.config, "teacher_forced": False, "inference_reexecuted": True, name: value}
            atomic_json(self.root / "run.json", run)
            with mock.patch("reference_corpus_contract.preserve_reference_identity") as archive:
                with self.assertRaises(ValueError):
                    prepare_output(self.root, self.config, {}, {}, resume=True)
                archive.assert_not_called()

    def test_stale_temporary_does_not_block_and_handled_failure_cleans_up(self):
        target = self.root / "page.json"
        stale = self.root / "page.json.tmp"
        stale.write_text("old interrupted write", encoding="utf-8")
        atomic_json(target, {"complete": True})
        self.assertEqual(read_snapshot(target)[0], {"complete": True})
        self.assertEqual(stale.read_text(encoding="utf-8"), "old interrupted write")
        with mock.patch("reference_corpus_contract.os.replace", side_effect=OSError("simulated rename failure")):
            with self.assertRaises(OSError):
                atomic_json(target, {"complete": False})
        self.assertEqual(read_snapshot(target)[0], {"complete": True})
        self.assertEqual(list(self.root.glob("page.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
