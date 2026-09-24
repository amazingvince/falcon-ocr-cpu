"""Host-only correction tests using existing synthetic records; no inference."""
import copy
import inspect
import unittest

import compare_temporal_model_saved_v1 as saved
import capture_temporal_model_v2 as original
from test_temporal_model_capture_v2 import fixture_reports


def fixture():
    reports = fixture_reports()
    for report in reports.values():
        report["canonical"]["result"]["finish_reason"] = "length"
    return reports


class SavedTemporalTests(unittest.TestCase):
    def test_only_teacher_stop_validator_changes(self):
        self.assertEqual(inspect.getsource(saved.compare_reports), inspect.getsource(original.compare_reports))
        expected = inspect.getsource(original.validate_record).replace(
            'record["finish_reason"] == "eos"', 'record["finish_reason"] == ("length" if teacher else "eos")').replace(
            '    require(ids[-1]', '    require(not teacher or len(ids) == 17, "Teacher trace length must be exactly seventeen")\n    require(ids[-1]')
        self.assertEqual(inspect.getsource(saved.validate_record), expected)

    def test_actual_teacher_length_semantics_and_free_eos_pass(self):
        reports = fixture()
        result = saved.compare_reports(reports, [1] * 16 + [263])
        self.assertEqual(result["allocation_intervals"], 10)
        self.assertEqual(result["canonical_tensor_count"], 1904)
        with self.assertRaisesRegex(ValueError, "Count/stop"):
            original.compare_reports(reports, [1] * 16 + [263])

    def test_wrong_teacher_stop_length_or_free_stop_rejected(self):
        for kind in ["teacher-eos", "teacher-count", "free-length", "free-teacher"]:
            reports = fixture()
            candidate = reports["candidate"]
            if kind == "teacher-eos": candidate["canonical"]["result"]["finish_reason"] = "eos"
            if kind == "teacher-count":
                candidate["canonical"]["result"]["token_ids"].pop(0)
                candidate["canonical"]["result"]["output_tokens"] = 16
            if kind == "free-length": candidate["independent_free_single"][0]["finish_reason"] = "length"
            if kind == "free-teacher": candidate["independent_free_single"][0]["teacher_forced"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                saved.compare_reports(reports, [1] * 16 + [263])

    def test_tensor_argmax_and_allocation_guards_unchanged(self):
        for kind in ["tensor", "argmax", "allocation"]:
            reports = fixture()
            candidate = reports["candidate"]
            if kind == "tensor": candidate["mixed_trace"]["tensors"]["request.0.prefill.logits"]["sha256"] = "b" * 64
            if kind == "argmax": candidate["canonical"]["trace"]["logits_argmax"][0][1][0] = 2
            if kind == "allocation": candidate["allocation"]["mixed"]["allocation_calls"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                saved.compare_reports(reports, [1] * 16 + [263])


if __name__ == "__main__":
    unittest.main()
