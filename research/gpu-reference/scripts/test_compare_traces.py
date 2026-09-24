"""Regression checks for numerical comparison decisions, without a GPU."""
import unittest

import torch

from compare_traces import logit_decision, normalized, stats, validate_policy


class TraceComparisonTests(unittest.TestCase):
    def test_tie_report_uses_first_argmax_index(self):
        logits = torch.tensor([0.0, 2.0, 2.0, 2.0])
        decision = logit_decision(logits, logits.clone(), 0.0)
        self.assertEqual(decision["reference_argmax"], 1)
        self.assertEqual(decision["candidate_argmax"], 1)
        self.assertTrue(decision["argmax_matches"])
        self.assertTrue(decision["near_tie"])

    def test_margin_gate_distinguishes_stable_and_near_tie(self):
        logits = torch.tensor([1.0, 3.0, 2.0])
        self.assertTrue(logit_decision(logits, logits, 0.49)["must_match"])
        self.assertFalse(logit_decision(logits, logits, 0.5)["must_match"])

    def test_nonfinite_logits_are_rejected_even_when_both_match(self):
        for bad in [float("nan"), float("inf"), -float("inf")]:
            logits = torch.tensor([0.0, bad])
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                logit_decision(logits, logits, 0.0)

    def test_alias_collision_is_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            normalized({"layer.0.q": 1, "prefill.layer.0.q": 2})
        self.assertEqual(normalized({"prefill.layer.0.q": 2}), {"layer.0.q": 2})

    def test_policy_bounds_must_be_finite_nonnegative_numbers(self):
        for bad in [None, True, "0.1", -1.0, float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "Invalid"):
                validate_policy({"stages": {"layer.0.q": {"absolute_tolerance": bad}}})
        validate_policy({"stages": {"layer.0.q": {"absolute_tolerance": 0.0}}})
        with self.assertRaisesRegex(ValueError, "no stage"):
            validate_policy({"stages": {}})

    def test_stats_rejects_shape_and_nonfinite_sign_mismatches(self):
        self.assertIn("shape_mismatch", stats(torch.zeros(2), torch.zeros(3)))
        self.assertIn("nonfinite_mismatch", stats(torch.tensor([float("inf")]),
                                                torch.tensor([-float("inf")])))


if __name__ == "__main__":
    unittest.main()
