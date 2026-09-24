#!/usr/bin/env python3
"""Exercise source-resume rejection without any GPU work or historical edits."""
import copy
import hashlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from reference_run_identity import capture_reference_identity, preserve_reference_identity, canonical_digest


class StartupIdentityTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parents[3]
        run = json.loads((self.root / "artifacts/reference/corpus-v3-fp32-4096/run.json").read_text(encoding="utf-8"))
        # This is a synthetic unit-test environment, not startup attestation of
        # the historical run that predates the Torch UUID metadata field.
        self.environment = dict(run["environment"], torch_device_uuid=run["environment"]["gpu_uuid"], cublas_workspace_config=":4096:8")
        environment_patch = mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        self.identity, self.contents = capture_reference_identity(self.root, self.environment, "fp32")
        self.temporary = tempfile.TemporaryDirectory()
        self.output = pathlib.Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_new_run_and_identical_resume(self):
        initial = preserve_reference_identity(self.output, self.identity, self.contents, new_run=True)
        resumed = preserve_reference_identity(self.output, self.identity, self.contents, new_run=False)
        self.assertEqual(initial, resumed)

    def test_old_run_without_archive_cannot_be_retrofitted(self):
        with self.assertRaisesRegex(ValueError, "do not retrofit"):
            preserve_reference_identity(self.output, self.identity, self.contents, new_run=False)
        self.assertFalse((self.output / "provenance").exists())

    def test_source_mutation_rejected(self):
        preserve_reference_identity(self.output, self.identity, self.contents, new_run=True)
        source = self.output / "provenance/sources/scripts/run_corpus_reference.py"
        source.write_bytes(source.read_bytes() + b"\n# injected change\n")
        with self.assertRaisesRegex(ValueError, "Preserved source archive changed"):
            preserve_reference_identity(self.output, self.identity, self.contents, new_run=False)

    def test_new_source_identity_cannot_resume_old_run(self):
        preserve_reference_identity(self.output, self.identity, self.contents, new_run=True)
        contents = dict(self.contents)
        name = "scripts/run_corpus_reference.py"
        contents[name] += b"\n# changed source\n"
        identity = copy.deepcopy(self.identity)
        identity["source_files"][name] = {"sha256": hashlib.sha256(contents[name]).hexdigest(), "bytes": len(contents[name])}
        identity["identity_sha256"] = canonical_digest({k: v for k, v in identity.items() if k != "identity_sha256"})
        with self.assertRaisesRegex(ValueError, "Preserved startup identity changed"):
            preserve_reference_identity(self.output, identity, contents, new_run=False)

    def test_source_path_injection_rejected(self):
        contents = dict(self.contents)
        contents["../unexpected.py"] = b"unexpected"
        with self.assertRaisesRegex(ValueError, "exactly the declared"):
            preserve_reference_identity(self.output, self.identity, contents, new_run=True)

    def test_unset_or_changed_actual_cublas_environment_rejected(self):
        for value in [None, ":16:8"]:
            with mock.patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
                with self.assertRaisesRegex(ValueError, "Actual cuBLAS"):
                    capture_reference_identity(self.root, self.environment, "fp32")


if __name__ == "__main__":
    unittest.main()
