"""Small source/contract tests. Run only after the quiet-window release."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from adapter import make_adapter, replace_one
from capture import selected_artifact, MODULE
from compare import saved_name
from contract import (ROOT, KIND, PINS, METADATA, RUNTIME, STAGES, CONTROL_STAGES,
                      PRODUCTION_MODEL_SHA, PRODUCTION_KERNELS_SHA, REQUIRED_SOURCES, validate_plan)


class SourceTests(unittest.TestCase):
    def test_adapter_retains_numerical_calls_and_stops_before_head_norm_on_layer9(self):
        adapter, original = make_adapter((ROOT / "src/model.rs").read_text(encoding="utf-8"))
        body = original[original.index("        let c = &self.config;"):original.index("        session.len += rows;")]
        for operator in ("kernels::rms_norm(", "kernels::linear_with_simd(", "kernels::squared_relu_gate(",
                         "rotary_factors(", "cache.append(", "cache.attention("):
            self.assertEqual(adapter.count(operator), body.count(operator))
        self.assertIn("for i in 8..=9 {", adapter)
        self.assertNotIn("session.len += rows", adapter)
        self.assertNotIn("work.logits", adapter)
        self.assertLess(adapter.index("if i == 9 {"), adapter.index("kernels::rms_norm(&work.q"))
        for line in ("vector[p] = a * cos - b * sin;", "vector[p + 1] = a * sin + b * cos;",
                     "*x += a;", "let cache = &mut session.layers[i];"):
            self.assertEqual(adapter.count(line), body.count(line))
        self.assertEqual(MODULE.count("#[cfg(test)]"), 1)

    def test_changed_or_ambiguous_anchor_rejects(self):
        for source in ("missing anchor", "old old"):
            with self.assertRaises(ValueError):
                replace_one(source, "old", "new")
        self.assertEqual(replace_one("one old anchor", "old", "new"), "one new anchor")

    def test_saved_prefix_resolution_is_unambiguous(self):
        self.assertEqual(saved_name({"layer.7.hidden"}, "layer.7.hidden"), "layer.7.hidden")
        self.assertEqual(saved_name({"prefill.layer.7.hidden"}, "layer.7.hidden"), "prefill.layer.7.hidden")
        for keys in (set(), {"layer.7.hidden", "prefill.layer.7.hidden"}):
            with self.assertRaises(ValueError):
                saved_name(keys, "layer.7.hidden")

    def test_frozen_runtime_and_stage_inventory_reject_mutation(self):
        plan = {"kind": KIND, "runtime": copy.deepcopy(RUNTIME), "stages": copy.deepcopy(STAGES),
                "control_stages": list(CONTROL_STAGES), "model_directory": "artifacts/model",
                "original_state_key": "layer.7.hidden", "original_state_shape": [144, 768],
                "branches": {"gpu": ["gpu_state", "cpu_state"], "rust": ["cpu_state"]},
                "inputs": {role: {"path": path, "sha256": digest} for role, (path, digest) in PINS.items()},
                "source_sha256": {name: "0" * 64 for name in REQUIRED_SOURCES}}
        plan["source_sha256"].update({"src/model.rs": PRODUCTION_MODEL_SHA, "src/kernels.rs": PRODUCTION_KERNELS_SHA})
        plan["inputs"].update({role: {"path": path, "sha256": "0" * 64} for role, path in METADATA.items()})
        validate_plan(plan, check_files=False)
        mutations = [lambda p: p["runtime"].update(rust_cache_capacity=256),
                     lambda p: p["stages"].pop("input"),
                     lambda p: p["control_stages"].pop(),
                     lambda p: p["source_sha256"].pop("research/gpu-reference/experiments/fp32_crossover/compare.py"),
                     lambda p: p.update(original_state_key="layer.8.hidden"),
                     lambda p: p["inputs"]["cpu_trace"].update(sha256="0" * 64),
                     lambda p: p["source_sha256"].update({"src/kernels.rs": "0" * 64})]
        for mutate in mutations:
            other = copy.deepcopy(plan)
            mutate(other)
            with self.assertRaises(ValueError):
                validate_plan(other, check_files=False)

    def test_cargo_artifact_rejects_shared_stale_or_wrong_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            project, target = base / "project", base / "target"
            row = {"reason": "compiler-artifact", "executable": str(target / "release/test.exe"),
                   "profile": {"test": True}, "fresh": False,
                   "manifest_path": str(project / "Cargo.toml"),
                   "target": {"src_path": str(project / "src/lib.rs"), "kind": ["lib"]}}
            self.assertEqual(selected_artifact(json.dumps(row), project, target)[1], target / "release/test.exe")
            for key, value in [("fresh", True), ("manifest_path", str(base / "other/Cargo.toml")),
                               ("executable", str(base / "shared/test.exe"))]:
                bad = copy.deepcopy(row)
                bad[key] = value
                with self.assertRaises(ValueError):
                    selected_artifact(json.dumps(bad), project, target)
            with self.assertRaises(ValueError):
                selected_artifact(json.dumps(row) + "\n" + json.dumps(row), project, target)


if __name__ == "__main__":
    unittest.main()
