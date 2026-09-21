#!/usr/bin/env python3
import copy
import unittest
from validate_gpu_reference_record import validate_gpu_reference_record


class GpuRecordTests(unittest.TestCase):
    def setUp(self):
        self.config = {"max_new_tokens": 4, "precision": "fp32"}
        self.page = {"configuration": self.config, "token_ids": [500, 263], "text": "example", "finish_reason": "eos",
                     "prefix_length": 144, "cache_capacity": 256, "canonical_rgb_sha256": "a" * 64,
                     "logit_decisions": [{"step": i, "argmax": token, "runner_up": 12, "winner_logit": 2.0,
                                          "runner_up_logit": 1.0, "winner_margin": 1.0} for i, token in enumerate([500, 263])],
                     "prefill_seconds_including_compile": 1.0, "decode_seconds_including_python_diagnostics": 2.0,
                     "elapsed_seconds": 4.0, "peak_gpu_allocated_bytes": 1000}

    def valid(self, page):
        return validate_gpu_reference_record(page, self.config, lambda *args: None, "page")

    def test_valid_record(self):
        self.assertTrue(self.valid(self.page))

    def test_empty_and_malformed_tokens(self):
        for ids in [[], None, [True], [65536], [11, 500, 263]]:
            page = copy.deepcopy(self.page)
            page["token_ids"] = ids
            self.assertFalse(self.valid(page))

    def test_false_length_stop(self):
        self.page["finish_reason"] = "length"
        self.assertFalse(self.valid(self.page))

    def test_impossible_context(self):
        self.page["cache_capacity"] = 128
        self.assertFalse(self.valid(self.page))

    def test_decisions_reject_corruption(self):
        for key, value in [("step", 1), ("argmax", 99), ("runner_up", 500),
                           ("winner_margin", 0.9), ("winner_logit", float("nan"))]:
            page = copy.deepcopy(self.page)
            page["logit_decisions"][0][key] = value
            self.assertFalse(self.valid(page), key)

    def test_missing_timing_or_object(self):
        del self.page["elapsed_seconds"]
        self.assertFalse(self.valid(self.page))
        self.assertFalse(self.valid(None))


if __name__ == "__main__":
    unittest.main()
