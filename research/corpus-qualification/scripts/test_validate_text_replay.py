#!/usr/bin/env python3
"""Meaningful rejection checks and optional actual native replay integration."""
import argparse
import copy
import json
import math
import pathlib
import unittest

from validate_text_replay import validate_inference_result, validate_run_replay, validate_page_replay

REPLAY = None


def collector():
    failures = []

    def check(name, actual, expected):
        if actual != expected:
            failures.append(name)
    return failures, check


class SavedResultTests(unittest.TestCase):
    contract = {"precision": "fp32", "options": {"max_new_tokens": 2, "max_dimension": 256}}
    result = {"token_ids": [561, 263], "output_tokens": 2, "text": "Hello", "teacher_forced": False,
              "precision": "fp32", "finish_reason": "eos", "width": 256, "height": 128, "input_tokens": 144,
              "timings": {"image_decode_ms": 0.0, "preprocessing_ms": 1.0, "prefill_ms": 2.0,
                          "decode_ms": 3.0, "total_ms": 6.0, "time_to_first_token_ms": 3.0}}

    def test_valid_saved_result(self):
        failures, check = collector()
        self.assertTrue(validate_inference_result(self.result, self.contract, check, "test"))
        self.assertEqual(failures, [])

    def test_empty_result_rejected(self):
        failures, check = collector()
        self.assertFalse(validate_inference_result({}, self.contract, check, "test"))
        self.assertTrue(failures)

    def test_malformed_fields_rejected(self):
        for field, value in [("output_tokens", 1), ("token_ids", [561, 65536]), ("token_ids", [True, 263]),
                             ("teacher_forced", True), ("text", None), ("input_tokens", 145),
                             ("height", 129), ("precision", "bf16"), ("finish_reason", "length")]:
            with self.subTest(field=field, value=value):
                result = {**self.result, field: value}
                failures, check = collector()
                self.assertFalse(validate_inference_result(result, self.contract, check, "test"))
                self.assertTrue(failures)

    def test_nonfinite_timing_rejected(self):
        for value in [float("nan"), float("inf"), -1.0, True, None]:
            with self.subTest(value=value):
                result = copy.deepcopy(self.result)
                result["timings"]["prefill_ms"] = value
                failures, check = collector()
                self.assertFalse(validate_inference_result(result, self.contract, check, "test"))

    def test_length_stop_requires_cap(self):
        result = {**self.result, "token_ids": [561, 562], "finish_reason": "length"}
        failures, check = collector()
        self.assertTrue(validate_inference_result(result, self.contract, check, "test"))
        result.update(token_ids=[561], output_tokens=1)
        self.assertFalse(validate_inference_result(result, self.contract, check, "test"))


class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        if REPLAY is None:
            self.skipTest("Pass --replay for actual native-Windows-produced artifact validation")
        self.run = json.loads((REPLAY / "run.json").read_text(encoding="utf-8"))
        self.pages = []
        for path in sorted(REPLAY.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "result" in value:
                self.pages.append((path.stem, value))
        self.assertTrue(self.pages)

    def context(self):
        failures, check = collector()
        context = validate_run_replay(self.run, self.run["contract"], check)
        self.assertIsNotNone(context)
        self.assertEqual(failures, [])
        return context

    def test_all_actual_native_pages_preserve_inference(self):
        context = self.context()
        failures, check = collector()
        for sample, page in self.pages:
            validate_page_replay(page, context, check, sample)
        self.assertEqual(failures, [])

    def test_one_ulp_timing_mutation_rejected(self):
        context = self.context()
        sample, page = self.pages[0]
        altered = copy.deepcopy(page)
        value = altered["result"]["timings"]["prefill_ms"]
        altered["result"]["timings"]["prefill_ms"] = math.nextafter(value, math.inf)
        failures, check = collector()
        validate_page_replay(altered, context, check, sample)
        self.assertTrue(any("all_inference_result_fields_sha256" in name for name in failures))

    def test_token_and_original_text_mutations_rejected(self):
        context = self.context()
        sample, page = self.pages[0]
        altered = copy.deepcopy(page)
        altered["result"]["token_ids"][0] ^= 1
        altered["postprocessing_replay"]["original_text"] += " changed"
        failures, check = collector()
        validate_page_replay(altered, context, check, sample)
        self.assertTrue(any("all_inference_result_fields_sha256" in name for name in failures))
        self.assertTrue(any("original_text_matches" in name for name in failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=pathlib.Path)
    options = parser.parse_args()
    REPLAY = options.replay
    unittest.main(argv=[__file__])
