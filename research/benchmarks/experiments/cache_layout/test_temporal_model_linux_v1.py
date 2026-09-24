"""Host-only exact-source and platform-observation checks; no build/inference."""
import copy
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import capture_temporal_model_linux_v1 as linux
from test_temporal_model_capture_v2 import fixture_reports


class LinuxModelTests(unittest.TestCase):
    def test_exact_native_sources_and_unchanged_integration_test(self):
        plan = json.loads((linux.NATIVE_DIR / "plan.json").read_bytes())
        raw = (linux.NATIVE_DIR / "source.zip").read_bytes()
        members = linux.source_members(raw, plan)
        self.assertEqual(len(members), 60)
        self.assertEqual(members["control/tests/temporal_model_qualification.rs"], members["candidate/tests/temporal_model_qualification.rs"])
        for name, contents in members.items():
            self.assertEqual(contents, (linux.NATIVE_DIR / name).read_bytes())

    def test_changed_missing_or_added_source_rejected(self):
        plan = json.loads((linux.NATIVE_DIR / "plan.json").read_bytes())
        with zipfile.ZipFile(linux.NATIVE_DIR / "source.zip") as z:
            members = {n: z.read(n) for n in z.namelist()}
        for kind in ["changed", "missing", "added"]:
            altered = dict(members)
            if kind == "changed": altered["candidate/src/temporal_candidate.rs"] += b"\n"
            if kind == "missing": del altered["control/src/runner.rs"]
            if kind == "added": altered["candidate/src/extra.rs"] = b"// extra"
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w") as z:
                for n, data in altered.items(): z.writestr(n, data)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                linux.source_members(out.getvalue(), plan)

    def test_windows_evidence_paths_map_under_wsl(self):
        with patch.object(linux.sys, "platform", "linux"):
            self.assertEqual(linux.recorded_path('C:\\Users\\amazi\\file.json').as_posix(), '/mnt/c/Users/amazi/file.json')

    def test_cross_platform_difference_is_explicit_not_an_acceptance_gate(self):
        reports = fixture_reports()
        windows = copy.deepcopy(reports)
        windows["candidate"]["mixed_trace"]["tensors"]["request.0.prefill.logits"]["sha256"] = "b" * 64
        result = linux.cross_platform_observation(reports, windows)
        self.assertFalse(result["per_invocation"]["candidate"]["mixed_trace"])
        self.assertTrue(result["per_invocation"]["candidate"]["independent_free_mixed"])


if __name__ == "__main__":
    unittest.main()
