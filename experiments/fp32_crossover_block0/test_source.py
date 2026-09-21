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
                      BUDGET, FAILURE_CONTEXT, FROZEN_TOOLCHAIN,
                      PRODUCTION_MODEL_SHA, PRODUCTION_KERNELS_SHA, REQUIRED_SOURCES,
                      validate_plan, validate_cpu_control)
from source_guard import check_sources, check_text, ENTRY_CHECK
from capture import copied_name


class SourceTests(unittest.TestCase):
    def test_adapter_retains_numerical_calls_and_executes_block0_only(self):
        adapter, original = make_adapter((ROOT / "src/model.rs").read_text(encoding="utf-8"))
        body = original[original.index("        let c = &self.config;"):original.index("        session.len += rows;")]
        for operator in ("kernels::rms_norm(", "kernels::linear_with_simd(", "kernels::squared_relu_gate(",
                         "rotary_factors(", "cache.append(", "cache.attention("):
            self.assertEqual(adapter.count(operator), body.count(operator))
        self.assertIn("for i in 0..=0 {", adapter)
        self.assertNotIn("session.len += rows", adapter)
        self.assertNotIn("work.logits", adapter)
        self.assertNotIn("if i ==", adapter)
        self.assertNotIn("layer.8", adapter)
        self.assertNotIn("layer.9", adapter)
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
        self.assertEqual(saved_name({"embedding"}, "embedding"), "embedding")
        self.assertEqual(saved_name({"prefill.embedding"}, "embedding"), "prefill.embedding")
        for keys in (set(), {"embedding", "prefill.embedding"}):
            with self.assertRaises(ValueError):
                saved_name(keys, "embedding")

    def test_frozen_runtime_and_stage_inventory_reject_mutation(self):
        plan = {"kind": KIND, "runtime": copy.deepcopy(RUNTIME), "stages": copy.deepcopy(STAGES),
                "control_stages": list(CONTROL_STAGES), "model_directory": "artifacts/model",
                "original_state_key": "embedding", "original_state_shape": [144, 768],
                "segment_endpoint": "layer.0.hidden", "original_failure_context": copy.deepcopy(FAILURE_CONTEXT),
                "evaluation_budget": copy.deepcopy(BUDGET),
                "branches": {"gpu": ["gpu_state", "cpu_state"], "rust": ["cpu_state"]},
                "inputs": {role: {"path": path, "sha256": digest} for role, (path, digest) in PINS.items()},
                "source_sha256": {name: "0" * 64 for name in REQUIRED_SOURCES}}
        plan["source_sha256"].update({"src/model.rs": PRODUCTION_MODEL_SHA, "src/kernels.rs": PRODUCTION_KERNELS_SHA})
        plan["inputs"].update({role: {"path": path, "sha256": "0" * 64} for role, path in METADATA.items()})
        self.assertEqual(len(STAGES), 14)
        self.assertEqual(len(CONTROL_STAGES), 5)
        validate_plan(plan, check_files=False)
        mutations = [lambda p: p["original_failure_context"].update(coordinate=[112, 14, 3]),
                     lambda p: p["evaluation_budget"].update(gpu_native_blocks=3),
                     lambda p: p["runtime"].update(rust_toolchain="1.94.0"),
                     lambda p: p["runtime"].update(rust_cache_capacity=256),
                     lambda p: p["stages"].pop("input"),
                     lambda p: p["control_stages"].pop(),
                     lambda p: p["source_sha256"].pop("experiments/fp32_crossover_block0/compare.py"),
                     lambda p: p.update(original_state_key="layer.0.hidden"),
                     lambda p: p["inputs"]["cpu_trace"].update(sha256="0" * 64),
                     lambda p: p["source_sha256"].update({"src/kernels.rs": "0" * 64})]
        for mutate in mutations:
            other = copy.deepcopy(plan)
            mutate(other)
            with self.assertRaises(ValueError):
                validate_plan(other, check_files=False)

    def test_native_gpu_and_adapter_source_guard(self):
        self.assertTrue(check_sources()["gpu_native_fragment_preserved"])
        old = ROOT / "experiments/fp32_crossover_layer7"
        new = ROOT / "experiments/fp32_crossover_block0"
        args = [(old / "adapter.py").read_text(), (new / "adapter.py").read_text(),
                (old / "export_gpu.py").read_text(), (new / "export_gpu.py").read_text()]
        mutations = [lambda s: s.replace('model.layers["0"]', 'model.layers["1"]'),
                     lambda s: s.replace('hidden = block0(x,', 'hidden = block0(x * 1.0,'),
                     lambda s: s.replace('return q, k, v', 'return q, k, v.clone()'),
                     lambda s: s.replace('cache = module.KVCache(1, 256,', 'cache = module.KVCache(1, 161,'),
                     lambda s: s.replace(ENTRY_CHECK, ""),
                     lambda s: s.replace('cpu_state = saved(c, "embedding")', 'cpu_state = saved(c, "layer.0.hidden")')]
        for mutate in mutations:
            other = args.copy()
            other[3] = mutate(other[3])
            self.assertNotEqual(other[3], args[3])
            with self.assertRaises(ValueError):
                check_text(*other)

    def test_cpu_prelaunch_control_claims_fail_closed(self):
        h = "a" * 64
        execution = {"kind": "fp32-crossover-block0-execution-v1", "status": "cpu_control_exact",
                     "plan_sha256": h, "source_and_input_closure": True}
        cpu = {"kind": "fp32-crossover-block0-rust-arm-v1", "status": "control_exact",
               "plan_sha256": h, "branch": "cpu_state", "arithmetic_changed": False,
               "native_block0_calls": 1, "input_bit_exact": True, "input_raw_sha256": h,
               "runtime": {"threads": 4, "backend": "avx2", "precision": "fp32", "cache_capacity": 161},
               "controls": [{"stage": k, "passed": True, "bit_mismatches": 0,
                             "actual_raw_sha256": h, "expected_raw_sha256": h} for k in CONTROL_STAGES],
               "tensors": {"cpu_state." + k: {"shape": v, "dtype": "F32", "raw_sha256": h}
                           for k, v in STAGES.items()}}
        validate_cpu_control(execution, cpu, h)
        mutations = [lambda c: c.update(native_block0_calls=2), lambda c: c.update(input_bit_exact=False),
                     lambda c: c["controls"][0].update(bit_mismatches=1),
                     lambda c: c["controls"][0].update(actual_raw_sha256="b" * 64),
                     lambda c: c["controls"].pop(), lambda c: c["tensors"].pop("cpu_state.layer.0.hidden"),
                     lambda c: c.update(input_raw_sha256="b" * 64),
                     lambda c: c["runtime"].update(threads=16)]
        for mutate in mutations:
            other = copy.deepcopy(cpu)
            mutate(other)
            with self.assertRaises(ValueError):
                validate_cpu_control(execution, other, h)
        for key, value in (("source_and_input_closure", False), ("plan_sha256", "b" * 64)):
            with self.assertRaises(ValueError):
                validate_cpu_control({**execution, key: value}, cpu, h)

    def test_frozen_toolchain_copy_does_not_use_live_pin(self):
        self.assertEqual(copied_name(ROOT / FROZEN_TOOLCHAIN), "rust-toolchain.toml")
        source = (ROOT / "experiments/fp32_crossover_block0/capture.py").read_text()
        self.assertIn('RUSTUP_TOOLCHAIN="1.92.0"', source)
        self.assertNotIn('("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")', source)

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
