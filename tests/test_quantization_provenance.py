"""Negative provenance checks independent of quantized model arithmetic."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quant_checker", ROOT / "experiments/quantization/check_probe.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class QuantProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.binary = self.root / "quant_probe.exe"
        self.binary.write_bytes(b"preserved diagnostic executable")
        self.archive = self.root / "source.zip"
        sources = {path: ("frozen " + key).encode() for key, path in checker.COMPILED_SOURCES.items()}
        sources["experiments/quantization/check_probe.py"] = b"historical checker"
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, value in sources.items():
                archive.writestr(name, value)
        self.hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
        self.build = {"status": "complete", "example": "quant_probe", "build_exit_code": 0, "source_unchanged_during_build": True,
                      "binary": self.binary.name, "binary_sha256": checker.sha(self.binary),
                      "source_archive_sha256": checker.sha(self.archive), "source_sha256": self.hashes}
        write(self.root / "build.json", self.build)
        self.report = {"binary_sha256": checker.sha(self.binary),
                       "source_sha256": {key: self.hashes[path] for key, path in checker.COMPILED_SOURCES.items()}}

    def tearDown(self):
        self.directory.cleanup()

    def validate(self, report=None):
        return checker.validate_preserved_build(report or self.report, self.root / "operators.json")

    def test_valid_historical_checker_is_distinct_from_current_checker(self):
        result = self.validate()
        self.assertEqual(result["archived_original_checker_sha256"], self.hashes["experiments/quantization/check_probe.py"])
        self.assertNotEqual(result["archived_original_checker_sha256"], result["current_checker_sha256"])

    def test_embedded_source_hash_mutation_rejected(self):
        report = deepcopy(self.report)
        report["source_sha256"]["q4"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Compiled source binding"):
            self.validate(report)

    def test_executable_substitution_rejected(self):
        self.binary.write_bytes(b"different executable")
        with self.assertRaisesRegex(ValueError, "binary hash"):
            self.validate()

    def test_archive_bytes_checked_even_if_container_hash_is_updated(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name in self.hashes:
                archive.writestr(name, b"mutated source")
        self.build["source_archive_sha256"] = checker.sha(self.archive)
        write(self.root / "build.json", self.build)
        with self.assertRaisesRegex(ValueError, "Archived source bytes"):
            self.validate()

    def test_source_changed_build_cannot_pass(self):
        self.build["source_unchanged_during_build"] = False
        write(self.root / "build.json", self.build)
        with self.assertRaisesRegex(ValueError, "source-before/after"):
            self.validate()

    def test_self_declared_input_hash_cannot_replace_fixed_pin(self):
        for field in ("checkpoint", "fixture", "fixture_sidecar"):
            report = {key + "_sha256": value for key, value in (("checkpoint", checker.CHECKPOINT_SHA256), ("fixture", checker.FIXTURE_SHA256), ("fixture_sidecar", checker.SIDECAR_SHA256))}
            report[field + "_sha256"] = "0" * 64
            # No model I/O is needed: preceding fields retain their known pinned
            # values and a mocked digest isolates the identity comparison itself.
            for key in ("checkpoint", "fixture", "fixture_sidecar"):
                report[key] = key
            from unittest.mock import patch
            def digest(path):
                return {"checkpoint": checker.CHECKPOINT_SHA256, "fixture": checker.FIXTURE_SHA256, "fixture_sidecar": checker.SIDECAR_SHA256}[Path(path).name]
            with self.subTest(field=field), patch.object(checker, "sha", side_effect=digest):
                with self.assertRaisesRegex(ValueError, "Pinned " + field + " identity"):
                    checker.validate_fixed_inputs(report)


if __name__ == "__main__":
    unittest.main()
