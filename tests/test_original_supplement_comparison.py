"""Regression checks for diagnostic scoring and incomplete-run gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_original_supplement import aggregate, edit_distance, intended_quality, outcome_status


class SupplementalComparisonTests(unittest.TestCase):
    def test_known_character_and_word_distances(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance(["one", "two"], ["one", "new", "two"]), 1)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_blank_marker_is_not_hidden_or_assigned_cer(self):
        quality = intended_quality("", ">>UNUSED_261<<")
        self.assertFalse(quality["normalized_output_empty"])
        self.assertEqual(quality["normalized_hallucinated_characters"], 14)
        self.assertIsNone(quality["cer"])
        self.assertIsNone(quality["wer"])

    def test_whitespace_only_blank_is_distinguished_from_raw_empty(self):
        quality = intended_quality("", " \n ")
        self.assertFalse(quality["raw_output_empty"])
        self.assertTrue(quality["normalized_output_empty"])
        self.assertEqual(quality["normalized_hallucinated_characters"], 0)

    def test_nfc_and_whitespace_normalization_keeps_case(self):
        self.assertTrue(intended_quality("café room", "cafe\u0301\nroom")["normalized_text_exact"])
        self.assertFalse(intended_quality("Room", "room")["normalized_text_exact"])

    def test_html_projection_is_secondary_and_preserves_cell_order(self):
        quality = intended_quality("Tea 4.60 & cake", "<table><tr><td>Tea</td><td>4.60 &amp; cake</td></tr></table>")
        self.assertGreater(quality["cer"], 0)
        self.assertTrue(quality["secondary_content_normalized_exact"])
        self.assertTrue(quality["secondary_table_text_projection_applied"])
        swapped = intended_quality("Tea 4.60", "<table><tr><td>4.60</td><td>Tea</td></tr></table>")
        self.assertGreater(swapped["secondary_content_cer"], 0)

    def test_blanks_do_not_contaminate_nonblank_denominator(self):
        rows = []
        for name, truth, output in [("blank", "", "hallucination"), ("text", "abc", "axc")]:
            rows.append({"id": name, "cpu": {"status": "complete", "provenance_passed": True, "quality": intended_quality(truth, output), "output_tokens": 2, "finish_reason": "eos"}})
        result = aggregate(rows, "cpu")
        self.assertEqual(result["nonblank_reference_characters"], 3)
        self.assertEqual(result["nonblank_micro_cer"], 1 / 3)
        self.assertEqual(result["blank_normalized_hallucinated_characters"], 13)

    def test_missing_failed_and_mismatching_runs_cannot_pass(self):
        no_failures = {"cpu": [], "gpu": []}
        self.assertEqual(outcome_status(True, False, 0, 15, True, {"cpu": [], "gpu": ["page"]}, no_failures), ("cpu_complete_gpu_pending", False))
        self.assertEqual(outcome_status(True, True, 14, 15, True, {"cpu": [], "gpu": []}, no_failures), ("parity_mismatch", False))
        self.assertEqual(outcome_status(True, True, 15, 15, False, {"cpu": [], "gpu": []}, no_failures), ("failed", False))
        self.assertEqual(outcome_status(True, True, 15, 15, True, {"cpu": [], "gpu": []}, {"cpu": [], "gpu": ["error"]}), ("failed", False))
        self.assertEqual(outcome_status(True, True, 15, 15, True, {"cpu": [], "gpu": []}, no_failures), ("complete", True))


if __name__ == "__main__":
    unittest.main()
