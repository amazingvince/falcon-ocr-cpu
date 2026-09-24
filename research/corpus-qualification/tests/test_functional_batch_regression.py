"""Functional harness tests use synthetic JSON only; no inference."""
import copy
import json
import pathlib
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research/corpus-qualification/scripts"))  # functional_batch_regression moved here in the 2026-09-24 archive

import functional_batch_regression as h


class FunctionalBatchTests(unittest.TestCase):
    def setUp(self):
        self.workload = {
            "runtime": {"precision": "fp32", "backend": "avx2", "threads": 4, "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096},
            "model": {"directory": "artifacts/model"},
            "cases": {"mixed-b4": ["prose", "blank", "sparse", "receipt"]},
            "inputs": {key: {"canonical_path": key + ".png", "prepared_dimensions_expected": [1024, 1024], "input_tokens_expected": 4112}
                       for key in ["prose", "blank", "sparse", "receipt"]}}
        self.jobs = h.jobs_for(self.workload, "mixed-b4")

    def result(self, key, mode):
        count = {"prose": 8, "blank": 1, "sparse": 3, "receipt": 5}[key]
        return {"token_ids": [100] * (count - 1) + [263], "text": key, "finish_reason": "eos",
                "output_tokens": count, "input_tokens": 4112, "width": 1024, "height": 1024,
                "precision": "fp32", "teacher_forced": False, "backend": "rust-gemm/avx2",
                "cache_layout": mode["cache_layout"], "weight_layout": mode["weight_layout"].replace("-", "_"),
                "packed_weight_bytes": 1234 if mode["weight_layout"] == "phase-packed" else 0, "weight_packing_ms": 0.0,
                "timings": {key: 0.0 for key in ["image_decode_ms", "preprocessing_ms", "prefill_ms", "decode_ms", "total_ms", "time_to_first_token_ms"]}}

    def test_all_layouts_and_original_order_are_frozen(self):
        self.assertEqual(len(self.jobs), 5)
        self.assertEqual(self.jobs[0]["batch_size"], 1)
        self.assertEqual({(j["mode"]["cache_layout"], j["mode"]["weight_layout"]) for j in self.jobs[1:]},
                         {("expanded", "unpacked"), ("compact", "unpacked"), ("expanded", "phase-packed"), ("compact", "phase-packed")})
        for job in self.jobs:
            self.assertEqual(job["request_keys"], ["prose", "blank", "sparse", "receipt"])
            command = h.command("fresh-cli", self.workload, job)
            self.assertIn("4096", command)
            self.assertNotIn("--text", command)
            self.assertNotIn("trace", command)
            self.assertEqual([pathlib.Path(p).name for p in command[-4:]], ["prose.png", "blank.png", "sparse.png", "receipt.png"])

    def test_candidate_outputs_must_be_valid_and_match_declared_layout(self):
        for job in self.jobs:
            outputs = [self.result(key, job["mode"]) for key in job["request_keys"]]
            h.validate_results(outputs, self.workload, job)
            for field, value in [("token_ids", []), ("output_tokens", 1), ("text", None), ("finish_reason", "length"),
                                 ("width", 512), ("input_tokens", 4113), ("teacher_forced", True), ("precision", "bf16"),
                                 ("backend", "rust-gemm/scalar"), ("cache_layout", "wrong"), ("weight_layout", "wrong")]:
                with self.subTest(job=job["id"], field=field):
                    changed = copy.deepcopy(outputs)
                    changed[0][field] = value
                    with self.assertRaises(ValueError):
                        h.validate_results(changed, self.workload, job)
            with self.assertRaises(ValueError):
                h.validate_results(outputs[:-1], self.workload, job)

    def test_literal_text_ids_stops_dimensions_and_order_compared(self):
        control = {key: self.result(key, self.jobs[0]["mode"]) for key in self.jobs[0]["request_keys"]}
        job = self.jobs[-1]
        results = [self.result(key, job["mode"]) for key in job["request_keys"]]
        self.assertTrue(all(row["exact"] for row in h.compare_results(control, results, job["request_keys"])))
        for field, value in [("text", "prose "), ("token_ids", [101, 263]), ("finish_reason", "length"), ("width", 512)]:
            changed = copy.deepcopy(results)
            changed[0][field] = value
            self.assertFalse(h.compare_results(control, changed, job["request_keys"])[0]["exact"])
        self.assertFalse(all(row["exact"] for row in h.compare_results(control, list(reversed(results)), job["request_keys"])))
        with self.assertRaises(ValueError):
            h.compare_results(control, results[:-1], job["request_keys"])

    def run_synthetic(self, folder, *, low_memory=False, fail_process=False, mismatch=False, missing=False,
                      tamper=None, no_mixed_eos=False):
        plan_path = folder / "plan.json"
        plan_path.write_text("{}", encoding="utf8")
        record = {"jobs": copy.deepcopy(self.jobs), "binary": "synthetic", "binary_sha256": "unused",
                  "case": "mixed-b4", "qualification": "synthetic", "minimum_available_physical_bytes": h.MEMORY_FLOOR}
        for job in record["jobs"]:
            job["command"] = [job["id"]]
        launches = []
        def fake_process(command, cwd, stdout, stderr):
            job = next(j for j in record["jobs"] if j["id"] == command[0])
            launches.append(job["id"])
            results = [self.result(key, job["mode"]) for key in job["request_keys"]]
            if no_mixed_eos:
                for result in results:
                    result.update(token_ids=[263], output_tokens=1)
            if mismatch and job["id"] == "joint-compact":
                results[0]["text"] += " "
            if missing:
                results.pop()
            for result in results:
                stdout.write((json.dumps(result) + "\n").encode())
            if tamper and len(launches) == 5:
                target = folder / tamper
                target.write_bytes(target.read_bytes() + b" ")
            return types.SimpleNamespace(returncode=1 if fail_process else 0)
        original_verify = h.verify
        def verify(path, digest):
            if str(path) != "synthetic":
                original_verify(path, digest)
        with patch.object(h, "load_plan", return_value=(record, self.workload)), \
             patch.object(h, "execution_identity", return_value={"platform": "synthetic", "cli_executable_sha256": "unused"}), \
             patch.object(h, "available_physical_memory", return_value={"available_bytes": 0 if low_memory else h.MEMORY_FLOOR}), \
             patch.object(h, "verify", side_effect=verify), patch.object(h.subprocess, "run", side_effect=fake_process):
            try:
                report = h.run(plan_path)
            except ValueError as error:
                return None, launches, str(error)
        return report, launches, None

    def test_complete_outputs_qualify_only_when_all_candidates_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, launches, error = self.run_synthetic(pathlib.Path(tmp))
            self.assertIsNone(error)
            self.assertEqual(len(launches), 5)
            self.assertTrue(report["functional_gate_passed"])
            self.assertTrue(report["mixed_eos_completion_observed"])
            self.assertEqual(report["live_requests_after_prefill"], 3)

    def test_memory_guard_prevents_any_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, launches, error = self.run_synthetic(pathlib.Path(tmp), low_memory=True)
            self.assertIsNone(report)
            self.assertEqual(launches, [])
            self.assertIn("physical memory", error)

    def test_process_failure_or_missing_requests_cannot_qualify(self):
        for key in ["fail_process", "missing"]:
            with tempfile.TemporaryDirectory() as tmp:
                report, launches, error = self.run_synthetic(pathlib.Path(tmp), **{key: True})
                self.assertIsNone(report)
                self.assertEqual(len(launches), 1)
                self.assertTrue(error)
                self.assertFalse((pathlib.Path(tmp) / "report.json").exists())
                self.assertFalse(h.read(pathlib.Path(tmp) / "execution-final.json")["functional_gate_passed"])

    def test_mismatch_preserves_failed_comparison_and_fails_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            report, launches, error = self.run_synthetic(folder, mismatch=True)
            self.assertIsNone(report)
            self.assertEqual(len(launches), 5)
            saved = h.read(folder / "report.json")
            self.assertFalse(saved["functional_gate_passed"])
            self.assertFalse(saved["comparisons"][1]["all_requests_exact"])

    def test_nonfinite_timing_and_unobserved_packing_rejected(self):
        job = self.jobs[-1]
        results = [self.result(key, job["mode"]) for key in job["request_keys"]]
        results[0]["weight_packing_ms"] = float("nan")
        with self.assertRaises(ValueError):
            h.validate_results(results, self.workload, job)
        results[0]["weight_packing_ms"] = 0.0
        results[0]["packed_weight_bytes"] = 0
        with self.assertRaises(ValueError):
            h.validate_results(results, self.workload, job)

    def test_replay_markers_rejected(self):
        job = self.jobs[0]
        for marker, value in [("postprocessing_replay", {}), ("derived_text_replay", False), ("inference_reexecuted", False)]:
            results = [self.result(key, job["mode"]) for key in job["request_keys"]]
            results[0][marker] = value
            with self.assertRaisesRegex(ValueError, "Replayed"):
                h.validate_results(results, self.workload, job)

    def test_late_plan_or_result_artifact_changes_cannot_qualify(self):
        for name in ["plan.json", "sequential-control.jsonl", "sequential-control.stderr.txt",
                     "sequential-control.invocation.json", "joint-compact.comparison.json", "execution.json"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                report, launches, error = self.run_synthetic(pathlib.Path(tmp), tamper=name)
                self.assertIsNone(report)
                self.assertIn("Changed file", error)
                self.assertEqual(len(launches), 5)
                self.assertFalse((pathlib.Path(tmp) / "report.json").exists())

    def test_equal_outputs_without_mixed_completion_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, launches, error = self.run_synthetic(pathlib.Path(tmp), no_mixed_eos=True)
            self.assertIsNone(report)
            saved = h.read(pathlib.Path(tmp) / "report.json")
            self.assertTrue(saved["all_outputs_exact"])
            self.assertFalse(saved["functional_gate_passed"])
            self.assertFalse(saved["multirow_decode_eligible_by_output_lengths"])

    def test_frozen_selection_rejects_changed_order_and_metadata(self):
        frozen = h.read(h.ROOT / "reference/functional-batch-v1-lock.json")
        self.assertEqual(h.canonical_sha256(frozen), h.WORKLOAD_CANONICAL_SHA256)
        for field in ["order", "metadata"]:
            changed = copy.deepcopy(frozen)
            if field == "order":
                changed["cases"]["mixed-b4"].reverse()
            else:
                changed["inputs"]["prose"]["category"] = "substituted"
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "changed.json"
                h.write_new(path, changed)
                with self.assertRaisesRegex(ValueError, "selection changed"):
                    h.validate_manifest(path)

    def test_build_binary_archive_and_live_source_binding(self):
        for changed in [None, "binary", "archive", "source", "source_map"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                source = root / "src/main.rs"
                source.parent.mkdir()
                source.write_bytes(b"original source")
                binary, archive = root / "falcon-ocr.exe", root / "source.zip"
                binary.write_bytes(b"original binary")
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("src/main.rs", source.read_bytes())
                record = {"target_kind": "bin", "target_name": "falcon-ocr", "status": "complete",
                          "source_unchanged_during_build": True, "build_exit_code": 0, "cargo_emitted_executable": "original-bin",
                          "command": ["cargo", "build", "--locked", "--release"], "binary": binary.name,
                          "binary_sha256": h.sha(binary), "binary_bytes": binary.stat().st_size,
                          "source_archive": archive.name, "source_archive_sha256": h.sha(archive),
                          "source_sha256": {"src/main.rs": h.sha(source)}}
                if changed == "binary":
                    binary.write_bytes(b"changed binary")
                elif changed == "archive":
                    archive.write_bytes(b"changed archive")
                elif changed == "source":
                    source.write_bytes(b"changed source")
                elif changed == "source_map":
                    record["source_sha256"] = {}
                manifest = root / "build.json"
                h.write_new(manifest, record)
                with patch.object(h, "ROOT", root), patch.object(h.capture, "source_paths", return_value=[source]):
                    if changed:
                        with self.assertRaises(ValueError):
                            h.validate_build(manifest)
                    else:
                        self.assertEqual(h.validate_build(manifest)[1], binary)

    def test_foreign_execution_platform_rejected_before_assets_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "plan.json"
            h.write_new(path, {"kind": "functional-batch-regression-v1", "platform": "different platform"})
            with self.assertRaisesRegex(ValueError, "platform differs"):
                h.load_plan(path)


if __name__ == "__main__":
    unittest.main()
