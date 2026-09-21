"""Source/proof checks only; no model, build, GPU or subprocess execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import capture_observed_w2_intervention as capture


class InterventionTests(unittest.TestCase):
    def test_successful_probe_arithmetic_is_copied_exactly(self):
        source = capture.PROBE.read_text(encoding="utf-8")
        self.assertEqual(capture.sha(capture.PROBE), capture.PROBE_SHA)
        copied = source[source.index("fn gather("):source.index("fn compare(")]
        module, digest = capture.module_source(source)
        self.assertEqual(module.count(copied), 1)
        self.assertEqual(digest, hashlib.sha256(copied.encode()).hexdigest())
        self.assertIn("rows != 144 || width != 2304 || channels != 768", module)
        self.assertIn("for slot in 0..14", module)
        self.assertIn("output.fill(0.0_f32)", module)
        self.assertIn("*dst += value", module)

    def test_only_copy_is_patched_and_controls_explicit(self):
        files = ["src/numerical_diagnostics.rs", "src/model_diagnostics.rs"]
        original = {name: (capture.ROOT / name).read_bytes() for name in files}
        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            (project / "src").mkdir()
            for name, data in original.items():
                (project / name).write_bytes(data)
            patches, _ = capture.patch_copy(project, capture.PROBE.read_text(encoding="utf-8"))
            self.assertEqual(set(patches), set(files))
            self.assertEqual({p.name for p in (project / "src").iterdir()}, {"model_diagnostics.rs", "numerical_diagnostics.rs", "observed_w2.rs"})
            model = (project / "src/model_diagnostics.rs").read_text()
            self.assertIn('["production", "observed_split_w2", "production_after"]', model)
            self.assertIn('if intervention == "observed_split_w2" { 22 } else { 0 }', model)
            self.assertIn('"cuda_rms" => 1', model)  # existing dispatch is unchanged and never selected
            self.assertIn('"cuda_rms_rounded_rsqrt" => 2', model)
            with self.assertRaises(ValueError):
                capture.patch_copy(project, capture.PROBE.read_text())
        self.assertEqual(original, {name: (capture.ROOT / name).read_bytes() for name in files})

    def test_complete_prerequisite_required(self):
        self.assertEqual(capture.sha(capture.PROOF), capture.PROOF_SHA)
        proof = json.loads(capture.PROOF.read_bytes())
        capture.validate_proof(proof)
        mutations = [lambda p: p.update(artifact_closure_unchanged=False),
                     lambda p: p["fixed_k_ranges"][0].__setitem__(1, 164),
                     lambda p: p["platforms"].pop("linux"),
                     lambda p: p["platforms"]["windows"]["all_partials_vs_gpu"].update(bit_mismatches=1),
                     lambda p: p["platforms"]["linux"]["ascending_fold_vs_gpu_output"].update(elements=110591)]
        for mutate in mutations:
            changed = copy.deepcopy(proof)
            mutate(changed)
            with self.assertRaises(ValueError):
                capture.validate_proof(changed)


if __name__ == "__main__":
    unittest.main()
