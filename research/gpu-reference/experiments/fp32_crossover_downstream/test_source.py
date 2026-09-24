"""Prospective host-only checks; do not import/run during the quiet benchmark."""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from contract import (ROOT, KIND, PINS, RUNTIME, STAGES, CONTROLS, ENDPOINT, STATE_HASHES,
                      SOURCE_FILES, OLD_EXPORT, OLD_EXPORT_SHA, validate_shape_contract)
from source_guard import verify_text, verify_sources
from evidence import saved_name, validate_reported_controls
from export_gpu import check_environment
from compare import fixed_endpoint


def synthetic_plan():
    return {"kind": KIND, "runtime": copy.deepcopy(RUNTIME), "stages": copy.deepcopy(STAGES),
            "control_stages": list(CONTROLS), "branches": ["control_c", "substitution_s"],
            "endpoint": copy.deepcopy(ENDPOINT), "state_raw_sha256": dict(STATE_HASHES),
            "model_directory": "artifacts/model",
            "inputs": {k: {"path": p, "sha256": h} for k, (p, h) in PINS.items()},
            "source_sha256": {**{k: "0" * 64 for k in SOURCE_FILES}, OLD_EXPORT: OLD_EXPORT_SHA}}


class SourceTests(unittest.TestCase):
    def test_copied_native_fragments_and_call_sites(self):
        proof = verify_sources()
        self.assertEqual(proof["native_fragments_exact"], 4)
        self.assertEqual(proof["native_call_sites"], 2)

    def test_guard_rejects_arithmetic_mask_or_capture_change(self):
        original = (ROOT / OLD_EXPORT).read_text(encoding="utf-8")
        candidate = (Path(__file__).parent / "native_segment.py").read_text(encoding="utf-8")
        for old, new in [('mask.seq_lengths = (144, 144)', 'mask.seq_lengths = (1, 144)'),
                         ('q, k = original_rope(*a, **kw)', 'q, k = original_rope(*a, **kw); q = q + 1'),
                         ('cache = module.KVCache(1, 256,', 'cache = module.KVCache(1, 144,'),
                         ('capture("layer.9.v", v)', 'capture("layer.9.v", v.clone())')]:
            self.assertEqual(candidate.count(old), 1)
            with self.assertRaises(ValueError):
                verify_text(original, candidate.replace(old, new))

    def test_fixed_inputs_control_order_and_endpoint_fail_closed(self):
        base = synthetic_plan()
        validate_shape_contract(base)
        self.assertEqual(len(STAGES), 17)
        self.assertEqual(CONTROLS, list(STAGES))
        for change in [lambda p: p["branches"].reverse(),
                       lambda p: p["control_stages"].pop(),
                       lambda p: p["stages"].pop("input"),
                       lambda p: p["runtime"].update(gpu_cache_capacity=144),
                       lambda p: p["endpoint"].update(coordinate=[112, 249]),
                       lambda p: p["endpoint"].update(original_absolute_bound=1.0),
                       lambda p: p["inputs"]["layer7_gpu_tensors"].update(sha256="0" * 64),
                       lambda p: p["source_sha256"].pop("research/gpu-reference/experiments/fp32_crossover_downstream/native_segment.py")]:
            other = copy.deepcopy(base)
            change(other)
            with self.assertRaises(ValueError):
                validate_shape_contract(other)

    def test_environment_checked_without_importing_torch(self):
        env = {"CUDA_VISIBLE_DEVICES": RUNTIME["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        with patch.dict(os.environ, env, clear=True):
            check_environment()
        for field in env:
            with patch.dict(os.environ, {k: v for k, v in env.items() if k != field}, clear=True):
                with self.assertRaises(ValueError):
                    check_environment()

    def test_missing_or_ambiguous_original_alias_rejected(self):
        self.assertEqual(saved_name({"prefill.layer.7.hidden"}, "layer.7.hidden"), "prefill.layer.7.hidden")
        for keys in (set(), {"layer.7.hidden", "prefill.layer.7.hidden"}):
            with self.assertRaises(ValueError):
                saved_name(keys, "layer.7.hidden")

    def test_reported_controls_do_not_accept_partial_or_failed_inventory(self):
        controls = {"controls": [{"stage": s, "bit_exact": True, "mismatched_elements": 0} for s in CONTROLS]}
        validate_reported_controls(controls, CONTROLS, True)
        for change in [lambda p: p["controls"].pop(),
                       lambda p: p["controls"][5].update(bit_exact=False),
                       lambda p: p["controls"][5].update(mismatched_elements=1)]:
            bad = copy.deepcopy(controls)
            change(bad)
            with self.assertRaises(ValueError):
                validate_reported_controls(bad, CONTROLS, True)

    def test_signed_telescoping_retains_cancellation_and_zero_total(self):
        # Exact binary integer values: total 2 = downstream 1 + layer7 -3 + earlier 4.
        result = fixed_endpoint(1.0, 5.0, 2.0, 3.0)
        self.assertEqual(result["D_minus_A_original"], 2.0)
        self.assertEqual(result["C_minus_B_layer7_engine_propagated"], -3.0)
        self.assertEqual(result["signed_fractions_of_original_total"]["earlier_state_propagated"], 2.0)
        self.assertEqual(result["fp64_telescoping_residual"], 0.0)
        zero = fixed_endpoint(1.0, 5.0, 2.0, 1.0)
        self.assertTrue(all(v is None for v in zero["signed_fractions_of_original_total"].values()))


if __name__ == "__main__":
    unittest.main()

