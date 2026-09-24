"""Host-only protocol/source tests. No tensors, model imports, or artifact preparation."""
import ast
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import contract as c
from compare import signed_terms, fixed_endpoint, validate_capture_header
from export_gpu import check_environment
from source_guard import verify_sources, verify_text


def plan():
    return {"kind": c.KIND, "runtime": copy.deepcopy(c.RUNTIME),
            "branches": c.BRANCHES.copy(), "control_branches": c.CONTROL_BRANCHES.copy(),
            "stages": copy.deepcopy(c.STAGES), "start_layer": c.START_LAYER.copy(),
            "entry_state": c.ENTRY_STATE.copy(), "endpoint": copy.deepcopy(c.ENDPOINT),
            "inputs": {k: {"path": p, "sha256": h} for k, (p, h) in c.PINS.items()},
            "state_raw_sha256": {k: "0" * 64 for k in c.ENTRY_KEYS},
            "source_sha256": {**{s: "0" * 64 for s in c.SOURCE_FILES}, **c.PRESERVED_SOURCE_PINS},
            "model_directory": "artifacts/model"}


def capture_header():
    return {"kind": c.KIND + "-gpu", "status": "three_controls_exact_eight_endpoints_captured",
            "plan_sha256": "a" * 64, "runtime": copy.deepcopy(c.RUNTIME),
            "controls_passed": c.CONTROL_BRANCHES.copy(), "completed_branches": c.BRANCHES.copy(),
            "source_and_input_closure": True, "effective_compiled_blocks": False,
            "native_block_calls": 54, "native_layer9_qkv_calls": 11,
            "controls": [{"branch": b, "stages": [{"stage": s, "bit_exact": True,
                 "mismatched_elements": 0} for s in c.STAGES[b]]} for b in c.CONTROL_BRANCHES]}


class FixedProtocol(unittest.TestCase):
    def test_budget_and_complete_inventory(self):
        self.assertEqual(c.BRANCHES, ["control_gpu", "E8", "E7", "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E9"])
        self.assertEqual(sum(9 - c.START_LAYER[b] for b in c.BRANCHES), 54)
        self.assertEqual(len(c.BRANCHES), 11)
        self.assertEqual([len(c.STAGES[b]) for b in c.CONTROL_BRANCHES], [47, 17, 30])
        self.assertEqual(sum(map(len, c.STAGES.values())), 110)
        self.assertEqual(len(c.ENTRY_KEYS), 11)
        self.assertEqual(c.ENDPOINT["coordinate"], [112, 14, 2])
        self.assertEqual(c.ENDPOINT["original_absolute_bound"], 0.005096435546875)
        self.assertTrue(all(c.STAGES[b]["input"] == [144, 768] for b in c.BRANCHES))
        self.assertTrue(all(c.STAGES[b]["layer.9.v"] == [144, 16, 64] for b in c.BRANCHES))

    def test_contract_rejects_drift(self):
        c.validate_shape_contract(plan())
        changes = [lambda p: p["runtime"].update(gpu_cache_capacity=144),
                   lambda p: p["runtime"].update(native_block_calls=53),
                   lambda p: p["branches"].reverse(),
                   lambda p: p["control_branches"].pop(),
                   lambda p: p["stages"]["E7"].pop("layer.7.hidden"),
                   lambda p: p["entry_state"].update(E7="C8"),
                   lambda p: p["endpoint"].update(coordinate=[112, 14, 3]),
                   lambda p: p["endpoint"].update(original_absolute_bound=0.1),
                   lambda p: p["state_raw_sha256"].pop("C9"),
                   lambda p: p["state_raw_sha256"].update(C0="g" * 64),
                   lambda p: p["inputs"]["cpu_trace"].update(sha256="0" * 64),
                   lambda p: p["source_sha256"].pop(c.HERE + "compare.py"),
                   lambda p: p["source_sha256"].update({next(iter(c.PRESERVED_SOURCE_PINS)): "0" * 64})]
        for change in changes:
            with self.subTest(change=change):
                altered = plan()
                change(altered)
                with self.assertRaises(ValueError):
                    c.validate_shape_contract(altered)

    def test_ordered_controls_fail_closed(self):
        for i, branch in enumerate(c.BRANCHES):
            c.require_next_branch(branch, c.BRANCHES[:i], c.CONTROL_BRANCHES[:min(i, 3)])
        invalid = [("E0", [], []), ("E8", ["control_gpu"], []),
                   ("E0", c.BRANCHES[:3], c.CONTROL_BRANCHES[:2]),
                   ("E1", ["E0", "E8", "E7", "control_gpu"], c.CONTROL_BRANCHES),
                   ("E9", c.BRANCHES, c.CONTROL_BRANCHES)]
        for args in invalid:
            with self.assertRaises(ValueError):
                c.require_next_branch(*args)

    def test_capture_header_requires_all_controls(self):
        validate_capture_header(capture_header(), "a" * 64)
        mutations = [lambda x: x.update(native_block_calls=55),
                     lambda x: x.update(native_layer9_qkv_calls=10),
                     lambda x: x.update(source_and_input_closure=False),
                     lambda x: x.update(effective_compiled_blocks=True),
                     lambda x: x["controls_passed"].pop(),
                     lambda x: x["completed_branches"].reverse(),
                     lambda x: x["controls"][0]["stages"].pop(),
                     lambda x: x["controls"][2]["stages"][0].update(bit_exact=False),
                     lambda x: x["controls"][1]["stages"][0].update(mismatched_elements=1)]
        for mutate in mutations:
            value = capture_header()
            mutate(value)
            with self.assertRaises(ValueError):
                validate_capture_header(value, "a" * 64)

    def test_preimport_environment_is_exact(self):
        environment = {"CUDA_VISIBLE_DEVICES": c.UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        with patch.dict(os.environ, environment, clear=True):
            check_environment()
        for key in environment:
            altered = environment.copy()
            altered.pop(key)
            with patch.dict(os.environ, altered, clear=True), self.assertRaises(ValueError):
                check_environment()
        with patch.dict(os.environ, {**environment, "CUDA_VISIBLE_DEVICES": "0,1"}, clear=True), self.assertRaises(ValueError):
            check_environment()

    def test_source_native_guards_and_mutations(self):
        proof = verify_sources()  # Small Python sources only, never model/trace assets.
        self.assertTrue(proof["native_block_call_exact_modulo_local_name"])
        original = (c.ROOT / c.PRIOR / "native_segment.py").read_text(encoding="utf-8")
        candidate = (c.ROOT / c.HERE / "native_segment.py").read_text(encoding="utf-8")
        mutations = [("mask.seq_lengths = (144, 144)", "mask.seq_lengths = (143, 144)"),
                     ("module.KVCache(1, 256,", "module.KVCache(1, 144,"),
                     ("hidden = layer(x,", "hidden = layer(x * 1.0,"),
                     ("_pre_attention_qkv(x)", "_pre_attention_qkv(x[:, :1])"),
                     ("x = hidden", "x = hidden + 0.0"),
                     ('entries[ENTRY_STATE[branch]].to("cuda:0")[None]', 'entries[ENTRY_STATE[branch]][:1].to("cuda:0")[None]'),
                     ("return q, k, v", "return q + 0.0, k, v"),
                     ("return q, k\n", "return q, k + 0.0\n")]
        for before, after in mutations:
            self.assertIn(before, candidate)
            with self.subTest(mutation=before), self.assertRaises(ValueError):
                verify_text(original, candidate.replace(before, after, 1))

    def test_telescope_signs_cancellation_and_zero_total(self):
        endpoints = [2.0, 5.0, -4.0, 3.0, 3.0, 8.0, 1.0, 7.0, 6.0, 9.0]
        terms = signed_terms(1.0, endpoints, 10.0)
        self.assertEqual(len(terms), 11)
        self.assertEqual(terms["E2_minus_E1_block1_conditional"], -9.0)
        self.assertEqual(sum(terms.values()), 9.0)
        fixed = fixed_endpoint(1.0, endpoints, 10.0)
        self.assertEqual(fixed["fp64_telescoping_residual"], 0.0)
        self.assertGreater(fixed["sum_abs_terms_over_abs_total"], 1)
        self.assertLess(fixed["signed_fractions_of_original_total"]["E2_minus_E1_block1_conditional"], 0)
        zero = fixed_endpoint(1.0, endpoints, 1.0)
        self.assertTrue(all(v is None for v in zero["signed_fractions_of_original_total"].values()))
        self.assertIsNone(zero["sum_abs_terms_over_abs_total"])
        with self.assertRaises(ValueError):
            signed_terms(1.0, endpoints[:-1], 10.0)

    def test_sources_syntax_and_no_eager_tensor_imports(self):
        for name in c.SOURCE_FILES:
            if not name.startswith(c.HERE) or not name.endswith(".py"):
                continue
            source = (c.ROOT / name).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=name)
            for item in tree.body:
                if isinstance(item, ast.Import):
                    self.assertFalse(any(n.name.split(".")[0] in ("torch", "numpy", "safetensors") for n in item.names), name)
                if isinstance(item, ast.ImportFrom):
                    self.assertNotIn((item.module or "").split(".")[0], ("torch", "numpy", "safetensors"), name)


if __name__ == "__main__":
    unittest.main()
