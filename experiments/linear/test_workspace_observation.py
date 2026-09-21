"""Host-only diagnostic tests; never import Torch or initialize CUDA."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import workspace_observation as w
import export_owned_workspace as exporter

ROOT = Path(__file__).resolve().parents[2]


class ObservationTests(unittest.TestCase):
    def test_fixed_basis_covers_all_k_exactly(self):
        covered = []
        for group in range(3):
            x, weight, expected = w.basis_operands(group)
            keys = weight.argmax(axis=1)
            self.assertEqual(int(np.count_nonzero(weight)), w.M)
            codes = np.arange(1, w.M + 1, dtype=np.float32)
            self.assertTrue(np.array_equal(weight[np.arange(w.M), keys], codes))
            self.assertTrue(np.array_equal(keys, np.arange(group * w.M, (group + 1) * w.M)))
            # Independent one-nonzero dot oracle; no matrix multiply is needed.
            oracle = (x[:, keys].astype(np.float64) * codes.astype(np.float64)[None, :]).astype(np.float32)
            self.assertTrue(np.array_equal(oracle.view(np.uint32), expected.view(np.uint32)))
            self.assertEqual(float(expected.max()), 110592.0)
            self.assertTrue(all(len(set(row)) == w.M for row in expected))
            covered.extend(int(k) for k in keys)
        self.assertEqual(covered, list(range(w.K)))

    def test_reject_invalid_group(self):
        for value in [-1, 3, True, 1.0, "1"]:
            with self.assertRaises(RuntimeError):
                w.basis_operands(value)

    def test_existing_w2_logs(self):
        directory = ROOT / "artifacts/reference/linear-logger-fp32"
        classic = (directory / "cublas.log").read_text()
        lt = (directory / "cublaslt_6443.log").read_text()
        # Preserved files contain four W13 calls followed by four W2 calls.
        # Keep only W2 helper/call records, beginning at the fifth SetStream.
        import re
        starts = [m.start() for m in re.finditer(r"I! cuBLAS .* function cublasStatus_t cublasSetStream_v2", classic)]
        classic = classic[starts[4]:]
        lines = lt.splitlines()
        start = next(i for i, line in enumerate(lines) if "[Api][cublasLtSSSMatmulAlgoGetHeuristic]" in line and "rows=2304 cols=768" in line)
        lt = "\n".join(lines[start:])
        self.assertEqual(w.parse_logs(classic, lt, 4)["split_count"], 14)
        changes = [(classic.replace("val=33554432", "val=6193152"), lt),
                   (classic.replace("val=2304", "val=2303"), lt),
                   (classic, lt.replace("numSplitsK=14", "numSplitsK=13")),
                   (classic, lt.replace("minBytesAlignmentA=16", "minBytesAlignmentA=8")),
                   (classic, lt.replace("stream=0X0", "stream=0X1")),
                   (classic, lt.replace("beta=0", "beta=1"))]
        for c, l in changes:
            with self.assertRaises(RuntimeError):
                w.parse_logs(c, l, 4)

    def _fake(self, directory, membership, mutate=None):
        paths = []
        for group in range(3):
            raw = np.zeros((w.SPLITS, w.N, w.M), dtype=np.float32)
            for channel in range(w.M):
                raw[membership[group * w.M + channel], :, channel] = np.arange(1, w.N + 1, dtype=np.float32) * (channel + 1)
            if mutate:
                mutate(group, raw)
            path = Path(directory) / f"{group}.bin"
            path.write_bytes(raw.tobytes() + bytes([w.CANARY_BYTE]) * (w.WORKSPACE_BYTES - w.SELECTED_BYTES))
            paths.append(path)
        return paths

    def test_complete_hypothesis_reports_noncontiguous_membership(self):
        # Tiny synthetic dimensions exercise the real full-array validator cheaply.
        with patch.multiple(w, M=5, N=3, K=15, SPLITS=2, SELECTED_BYTES=120, WORKSPACE_BYTES=160):
            membership = [k % 2 for k in range(15)]
            with tempfile.TemporaryDirectory() as d:
                result = w.observe_layout(self._fake(d, membership))
            self.assertTrue(result["complete_basis_agreement"])
            self.assertEqual(result["conditional_k_membership"], membership)
            self.assertTrue(all(not p["contiguous"] for p in result["conditional_partitions"]))
            self.assertIn("conditional", result["limitation"])

    def test_late_group_bad_value_no_membership(self):
        with patch.multiple(w, M=5, N=3, K=15, SPLITS=2, SELECTED_BYTES=120, WORKSPACE_BYTES=160):
            def mutate(group, raw):
                if group == 2:
                    raw[0, 2, 4] = 999
            with tempfile.TemporaryDirectory() as d:
                result = w.observe_layout(self._fake(d, [k % 2 for k in range(15)], mutate))
            self.assertFalse(result["complete_basis_agreement"])
            self.assertIsNone(result["conditional_k_membership"])

    def test_duplicate_partial_and_row_disagreement_rejected(self):
        with patch.multiple(w, M=5, N=3, K=15, SPLITS=2, SELECTED_BYTES=120, WORKSPACE_BYTES=160):
            for kind in ("duplicate", "row"):
                def mutate(group, raw):
                    if kind == "duplicate":
                        raw[1, 0, 0] = 1
                    else:
                        raw[0, 0, 0] = 0
                        raw[1, 0, 0] = 1
                with tempfile.TemporaryDirectory() as d:
                    result = w.observe_layout(self._fake(d, [0] * 15, mutate))
                self.assertFalse(result["complete_basis_agreement"])

    def test_missing_group_and_wrong_size_rejected(self):
        with self.assertRaises(RuntimeError):
            w.observe_layout([])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "short.bin"
            path.write_bytes(bytes(3))
            with self.assertRaises(RuntimeError):
                w.observe_layout([path] * 3)

    def test_hash_pin_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "source"
            path.write_bytes(b"original")
            with patch.object(w, "PINS", {"source": w.sha256(path)}):
                w.verify_pins(Path(d))
                path.write_bytes(b"changed")
                with self.assertRaises(RuntimeError):
                    w.verify_pins(Path(d))

    def test_changed_channel_code_rejected(self):
        with patch.multiple(w, M=5, N=3, K=15, SPLITS=2, SELECTED_BYTES=120, WORKSPACE_BYTES=160):
            def mutate(group, raw):
                raw[:, :, [0, 4]] = raw[:, :, [4, 0]]
            with tempfile.TemporaryDirectory() as d:
                result = w.observe_layout(self._fake(d, [0] * 15, mutate))
            self.assertFalse(result["complete_basis_agreement"])

    def test_failed_control_process_never_starts_basis(self):
        with tempfile.TemporaryDirectory() as d, patch.object(exporter, "SOURCE_NAMES", []), \
                patch.object(exporter.platform, "platform", return_value="synthetic-host-test"), \
                patch.object(exporter, "source_state", return_value={}), \
                patch.object(exporter, "verify_pins", return_value={}), \
                patch.object(exporter.subprocess, "run", return_value=SimpleNamespace(returncode=9)) as run:
            output = Path(d) / "attempt"
            self.assertEqual(exporter.orchestrate(output), 1)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "basis-0").exists())
            self.assertEqual(w.read_json(output / "report.json")["status"], "execution_failed")

    def test_failed_control_log_gate_never_starts_basis(self):
        with tempfile.TemporaryDirectory() as d, patch.object(exporter, "SOURCE_NAMES", []), \
                patch.object(exporter.platform, "platform", return_value="synthetic-host-test"), \
                patch.object(exporter, "source_state", return_value={}), \
                patch.object(exporter, "verify_pins", return_value={}), \
                patch.object(exporter.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                patch.object(exporter, "validate_worker", side_effect=RuntimeError("changed split count")):
            output = Path(d) / "attempt"
            self.assertEqual(exporter.orchestrate(output), 1)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "basis-0").exists())
            self.assertIn("split count", w.read_json(output / "report.json")["failure"]["message"])

    def test_final_pin_change_leaves_failed_receipt(self):
        with tempfile.TemporaryDirectory() as d, patch.object(exporter, "SOURCE_NAMES", []), \
                patch.object(exporter.platform, "platform", return_value="synthetic-host-test"), \
                patch.object(exporter, "source_state", return_value={}), \
                patch.object(exporter, "verify_pins", side_effect=[{}, RuntimeError("changed final fixture")]), \
                patch.object(exporter.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            output = Path(d) / "attempt"
            self.assertEqual(exporter.orchestrate(output), 1)
            report = w.read_json(output / "report.json")
            self.assertFalse(report["source_and_artifact_closure_unchanged"])
            self.assertIn("changed final fixture", report["closure_errors"])

    def test_mutation_immediately_after_gate_is_rejected(self):
        def gate_then_mutate(directory, mode, sources):
            artifact = directory / "workspace.bin"
            artifact.write_bytes(b"validated")
            digest = w.sha256(artifact)
            artifact.write_bytes(b"changed-after-gate")
            return {"status": "passed", "mode": mode, "validated_files": {"workspace.bin": digest}}
        with tempfile.TemporaryDirectory() as d, patch.object(exporter, "SOURCE_NAMES", []), \
                patch.object(exporter.platform, "platform", return_value="synthetic-host-test"), \
                patch.object(exporter, "source_state", return_value={}), \
                patch.object(exporter, "verify_pins", return_value={}), \
                patch.object(exporter.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                patch.object(exporter, "validate_worker", side_effect=gate_then_mutate):
            output = Path(d) / "attempt"
            self.assertEqual(exporter.orchestrate(output), 1)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "basis-0").exists())
            report = w.read_json(output / "report.json")
            self.assertIn("changed before closure", report["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
