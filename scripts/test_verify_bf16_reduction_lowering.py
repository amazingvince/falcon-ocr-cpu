import difflib
import pathlib
import tempfile
import unittest

from instrument_bf16_flex import build_instrumented
from verify_bf16_reduction_lowering import verify_lowering

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAVED = ROOT / "artifacts/reference/bf16-fused-substages-v4"


@unittest.skipUnless(SAVED.is_dir(), "Preserved pinned compiler artifacts required")
class LoweringTests(unittest.TestCase):
    def mirror_original(self, directory):
        for label in ["original", "instrumented"]:
            (directory / (label + ".py")).write_bytes((SAVED / "original.py").read_bytes())
            for kind in ["ptx", "ttgir"]:
                (directory / (label + "-0." + kind)).write_bytes((SAVED / ("original-0." + kind)).read_bytes())

    def test_original_is_positive_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            self.mirror_original(directory)
            self.assertTrue(verify_lowering(directory)["equal"])

    def test_rejected_v4_fails_both_structural_checks(self):
        result = verify_lowering(SAVED)
        self.assertFalse(result["equal"])
        self.assertFalse(result["checks"]["candidate_two_matching_MMA_add_reductions"])
        self.assertFalse(result["checks"]["candidate_identical_row_reduction_shapes"])

    def test_missing_artifact_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(verify_lowering(pathlib.Path(temporary))["equal"])

    def test_changed_shuffle_offset_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            self.mirror_original(directory)
            path = directory / "instrumented-0.ptx"
            text = path.read_text(encoding="utf-8").replace(", 2, 31, -1;", ", 4, 31, -1;")
            path.write_text(text, encoding="utf-8")
            self.assertFalse(verify_lowering(directory)["equal"])

    def test_partial_observer_changes_only_debug_sum_store(self):
        source = (ROOT / "artifacts/reference/bf16-flex-prefill-compiled.py").read_bytes()
        original, full = build_instrumented(source)
        second_original, partial = build_instrumented(source, omit_debug_sum=True)
        self.assertEqual(original, second_original)
        changes = [line for line in difflib.ndiff(full.splitlines(), partial.splitlines()) if line[:2] in ["- ", "+ "]]
        self.assertEqual(len(changes), 1)
        self.assertTrue(changes[0].startswith("-     tl.store(debug_observations"))
        self.assertIn("tl.sum(p, 1)", changes[0])
        self.assertEqual(partial.count("l_i = l_i * alpha + tl.sum(p, 1)"), 1)


if __name__ == "__main__":
    unittest.main()
