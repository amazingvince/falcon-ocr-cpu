"""Independent bounded reporting tests; never run inference or an evaluator."""
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import report_quality_regression as quality
from test_quality_regression import QualityRegressionTests


class LateOfficialBindingTests(unittest.TestCase):
    def test_files_changed_after_successful_join_are_rejected_at_final_closure(self):
        for target in ["run", "record", "prediction", "provenance"]:
            with self.subTest(target=target):
                helper = QualityRegressionTests()
                helper.setUp()
                try:
                    fixture = helper.fixture()
                    roots, official = helper.official_fixture(fixture)
                    paths = {"run": helper.root / "cpu/run.json",
                             "record": helper.root / "cpu/page000.json",
                             "prediction": roots["cpu"] / "predictions/page000.md",
                             "provenance": roots["cpu"] / "provenance.json"}
                    original = quality.bind_official_predictions

                    def join_then_change(*args, **kwargs):
                        evidence = original(*args, **kwargs)
                        path = paths[target]
                        path.write_bytes(path.read_bytes() + b" ")
                        return evidence

                    with patch.object(quality, "bind_official_predictions", side_effect=join_then_change):
                        with self.assertRaisesRegex(ValueError, "Source changed during quality accounting"):
                            helper.evaluate_official(fixture, roots, official)
                finally:
                    helper.tearDown()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LateOfficialBindingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
