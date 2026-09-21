"""Bounded host-only independent review of three staged GPU contract fixes."""
import builtins
import datetime
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_corpus_contract import atomic_json, read_snapshot
from reference_run_identity import capture_reference_identity
import test_reference_corpus_contract
import test_reference_run_identity


class IndependentTargetedTests(unittest.TestCase):
    def test_staged_preflight_rejects_before_torch_import(self):
        path = ROOT / "artifacts/reference-source-preparation-v2/reference_preflight.py"
        spec = importlib.util.spec_from_file_location("staged_reference_preflight_review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                self.fail("Invalid cuBLAS environment reached Torch import")
            return original_import(name, *args, **kwargs)

        for value in (None, ":16:8"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
                with mock.patch("builtins.__import__", side_effect=guarded_import):
                    with self.assertRaisesRegex(RuntimeError, "cuBLAS workspace"):
                        module.preflight(ROOT)

    def test_captured_environment_cannot_disagree_with_actual(self):
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            for environment in ({}, {"cublas_workspace_config": ":16:8"}):
                with self.subTest(environment=environment):
                    with self.assertRaisesRegex(ValueError, "Actual cuBLAS"):
                        capture_reference_identity(ROOT, environment, "fp32")

    def test_interrupted_unique_temporary_does_not_block_next_publish(self):
        with tempfile.TemporaryDirectory() as name:
            folder = pathlib.Path(name)
            target = folder / "run.json"
            atomic_json(target, {"generation": 1})
            stale = folder / "run.json.interrupted123.tmp"
            stale.write_bytes(b"{partial")
            atomic_json(target, {"generation": 2})
            self.assertEqual(read_snapshot(target)[0], {"generation": 2})
            self.assertEqual(stale.read_bytes(), b"{partial")
            self.assertEqual(list(folder.glob("*.tmp")), [stale])


def main():
    preparation_path = ROOT / "artifacts/reference-source-preparation-v2/preparation.json"
    preparation = json.loads(preparation_path.read_bytes())
    expected = dict(preparation["original_sources_sha256"])
    expected.update({"artifacts/reference-source-preparation-v2/" + name: digest
                     for name, digest in preparation["candidate_sources_sha256"].items()})
    expected.update(preparation["helper_sources_sha256"])
    paths = sorted(set(expected) | {
        "artifacts/reference-source-preparation-v2/preparation.json",
        "scripts/test_reference_corpus_contract.py", "scripts/test_reference_run_identity.py",
        "scripts/validate_gpu_reference_record.py", "reference/future-gpu-contract-review-v1.py",
    })
    snapshot = lambda: {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    before = snapshot()
    for name, digest in expected.items():
        if before[name] != digest:
            raise RuntimeError("Prepared/original source binding changed: " + name)
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromModule(test_reference_corpus_contract),
        unittest.defaultTestLoader.loadTestsFromModule(test_reference_run_identity),
        unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTargetedTests),
    ])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    after = snapshot()
    if before != after:
        raise RuntimeError("Reviewed source bytes changed during tests")
    report = {
        "schema_version": 1, "status": "closed" if result.wasSuccessful() else "failed",
        "reviewed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Only the three previously reported future-run contract defects; no broad audit",
        "findings": {
            "teacher_or_replay_resume_acceptance": "closed: strict explicit booleans and replay-marker rejection at configuration/run/page levels",
            "unverified_cublas_workspace_claim": "closed: actual environment checked before preflight Torch import and rechecked/bound during identity capture",
            "crash_left_temporary_blocks_resume": "closed: unique sibling temporary files, atomic replacement, handled-failure cleanup; interrupted leftovers do not block publication",
        },
        "tests_run": result.testsRun, "test_failures": len(result.failures), "test_errors": len(result.errors),
        "host_only": True, "model_inference_executed": False, "cuda_called": False,
        "source_before_sha256": before, "source_after_sha256": after,
        "source_unchanged": before == after,
        "qualification": "Checks the held, unused staged source and host helpers. It does not attest historical GPU startup metadata or execute the candidate model/runtime path.",
    }
    destination = ROOT / "reference/future-gpu-contract-review-v1.json"
    log_path = destination.with_suffix(".txt")
    with log_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(log)
    report["test_log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(log)
    print(json.dumps({"status": report["status"], "receipt": str(destination),
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
