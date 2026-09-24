"""Targeted subset identity and comparison tests; no model inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research/corpus-qualification/scripts"))  # compare_tokenizer_regression moved here in the 2026-09-24 archive
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from compare_tokenizer_regression import compare_result, validate_subset
from fetch_reference import sha256


class TargetedRegressionTests(unittest.TestCase):
    def test_subset_binds_full_page_identity_not_just_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.json"
            full = {"pages": [{"id": "a", "canonical_png_sha256": "original", "category": "ordinary"}]}
            path.write_text(json.dumps(full), encoding="utf-8")
            subset = {"source_manifest": str(path), "source_manifest_sha256": sha256(path), "pages": deepcopy(full["pages"])}
            failures = []
            check = lambda name, actual, expected: failures.append(name) if actual != expected else None
            validate_subset(subset, full, path, check)
            self.assertFalse(failures)
            subset["pages"][0]["canonical_png_sha256"] = "changed"
            validate_subset(subset, full, path, check)
            self.assertEqual(failures, ["a.exact_selected_page"])

    def test_subset_rejects_duplicate_and_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.json"
            full = {"pages": [{"id": "a"}]}
            path.write_text(json.dumps(full), encoding="utf-8")
            subset = {"source_manifest": str(path), "source_manifest_sha256": sha256(path), "pages": [{"id": "x"}, {"id": "x"}]}
            failures = []
            validate_subset(subset, full, path, lambda name, actual, expected: failures.append(name) if actual != expected else None)
            self.assertIn("subset.nonempty_unique_ids", failures)
            self.assertIn("x.exact_selected_page", failures)

    def test_same_tokens_do_not_hide_text_or_stop_change(self):
        reference = {"token_ids": [5, 263], "text": "Date . . .", "finish_reason": "eos"}
        changed = dict(reference, text="Date...", finish_reason="length")
        result = compare_result(changed, reference)
        self.assertTrue(result["tokens_exact"])
        self.assertFalse(result["text_exact"])
        self.assertFalse(result["finish_reason_exact"])

    def test_appended_and_changed_tokens_have_exact_first_divergence(self):
        reference = {"token_ids": [5, 6], "text": "a", "finish_reason": "length"}
        appended = compare_result(dict(reference, token_ids=[5, 6, 7]), reference)
        self.assertFalse(appended["output_count_exact"])
        self.assertEqual(appended["first_token_divergence"], 2)
        changed = compare_result(dict(reference, token_ids=[5, 8]), reference)
        self.assertEqual(changed["first_token_divergence"], 1)


if __name__ == "__main__":
    unittest.main()
