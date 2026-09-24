"""Protocol regression tests only: synthetic JSON, no inference or measured timings."""
import copy
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from realistic_benchmark import DEFAULT_WORKLOAD, jobs_for, read, validate_report
from realistic_benchmark import sha, write_new
from compare_realistic_benchmarks import compare


class ReportValidation(unittest.TestCase):
    def setUp(self):
        self.workload = read(DEFAULT_WORKLOAD)
        self.plan = {"repetitions": 3, "cpu_label": "synthetic-test", "environment_label": "synthetic-test"}
        self.job = jobs_for(self.workload, ["mixed-sparse-first"], [2])[0]
        self.build = {"binary_sha256": "not-a-real-binary", "source_sha256": {"Cargo.lock": "not-a-real-lock", "examples/ocr_bench.rs": "test-source"}}
        self.report = {"schema_version": 2, "threads": 16, "backend": "avx2", "precision": "fp32",
            "warmup": 2, "repetitions": 3, "cpu_label": "synthetic-test", "environment_label": "synthetic-test",
            "model_revision": self.workload["model"]["revision"],
            "weights_sha256": self.workload["model"]["assets"]["model.safetensors"]["sha256"],
            "binary_sha256": "not-a-real-binary", "cargo_lock_sha256": "not-a-real-lock",
            "cache_layout": "expanded", "weight_layout": "unpacked", "source_sha256": {"harness": "test-source"},
            "options": {"min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096},
            "packed_weight_bytes": 0, "weight_packing_ms": 0, "read_decode_ms": 0, "verified_model_load_ms": 0,
            "images": [], "cases": []}
        from realistic_benchmark import ROOT
        for key in self.job["image_keys"]:
            item = self.workload["inputs"][key]
            self.report["images"].append({"path": str(ROOT / item["canonical_path"]), "sha256": item["canonical_png_sha256"],
                "rgb_sha256": item["rgb_sha256"], "width": item["width"], "height": item["height"]})
        requests = []
        for key in self.job["request_keys"]:
            item = self.workload["inputs"][key]
            requests.append({"text": key, "input_tokens": item["input_tokens_expected"], "output_tokens": 2,
                "finish_reason": "eos", "teacher_forced": False, "precision": "fp32",
                "width": item["prepared_dimensions_expected"][0], "height": item["prepared_dimensions_expected"][1],
                "timings": {name: 0 for name in ("image_decode_ms", "preprocessing_ms", "image_projection_ms", "transformer_prefill_ms", "prefill_ms", "decode_ms", "total_ms", "time_to_first_token_ms")}})
        self.report["cases"].append({"batch_size": 2, "active_batch_size": 1, "image_indices": [0, 1],
            "execution": "independent_sequential", "median_ms": 100, "median_pages_per_second": 20,
            "token_ids": [[42, 263], [77, 11]], "samples": [{"wall_ms": 100, "pages_per_second": 20,
                "emitted_tokens": 4, "per_request": copy.deepcopy(requests)} for _ in range(3)]})

    def valid(self):
        return validate_report(self.report, self.job, self.plan, self.workload, self.build)

    def test_literal_text_and_stops_preserved(self):
        self.assertEqual([r["text"] for r in self.valid()], ["sparse-room", "table"])

    def test_reject_missing_text(self):
        del self.report["cases"][0]["samples"][1]["per_request"][0]["text"]
        with self.assertRaises(ValueError): self.valid()

    def test_reject_unmeasured_or_impossible_prefill_stages(self):
        timings = self.report["cases"][0]["samples"][0]["per_request"][0]["timings"]
        for value in (None, float("nan"), -1, 1):
            timings["image_projection_ms"] = value
            with self.assertRaises(ValueError): self.valid()

    def test_reject_changed_text_between_repetitions(self):
        self.report["cases"][0]["samples"][1]["per_request"][0]["text"] = "changed"
        with self.assertRaises(ValueError): self.valid()

    def test_reject_interior_eos(self):
        self.report["cases"][0]["token_ids"][0] = [263, 42]
        with self.assertRaises(ValueError): self.valid()

    def test_reject_early_length_stop(self):
        self.report["cases"][0]["token_ids"][0] = [42, 51]
        with self.assertRaises(ValueError): self.valid()

    def test_reject_stop_label_mismatch(self):
        self.report["cases"][0]["samples"][0]["per_request"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError): self.valid()

    def test_reject_missing_request(self):
        self.report["cases"][0]["samples"][0]["per_request"].pop()
        with self.assertRaises(ValueError): self.valid()

    def test_reject_wrong_order(self):
        self.report["cases"][0]["image_indices"] = [1, 0]
        with self.assertRaises(ValueError): self.valid()

    def test_reject_changed_build(self):
        self.report["binary_sha256"] = "different"
        with self.assertRaises(ValueError): self.valid()

    def test_reject_false_median(self):
        self.report["cases"][0]["median_ms"] = 90
        with self.assertRaises(ValueError): self.valid()

    def test_reject_teacher_forced(self):
        self.report["cases"][0]["samples"][0]["per_request"][0]["teacher_forced"] = True
        with self.assertRaises(ValueError): self.valid()

    def test_reject_mode_substitution(self):
        self.report["cache_layout"] = "compact"
        with self.assertRaises(ValueError): self.valid()

    def test_workload_allows_fixed_prefixes_and_cycles(self):
        full = jobs_for(self.workload, ["fullpages"], [1, 8])
        self.assertEqual(full[0]["request_keys"], ["prose"])
        self.assertEqual(full[6]["request_keys"], ["prose", "table", "columns", "prose", "table", "columns", "prose", "table"])
        self.assertEqual([j["mode"]["name"] for j in full[:6]], ["sequential-a", "expanded-a", "compact", "phase-packed", "expanded-b", "sequential-b"])

    def synthetic_comparison(self, changed_text=False, control_drift=False):
        with tempfile.TemporaryDirectory(prefix="benchmark-protocol-test-") as temporary:
            folder = pathlib.Path(temporary)
            jobs = jobs_for(self.workload, ["mixed-sparse-first"], [2])
            plan = dict(self.plan, jobs=jobs, coverage="selected_groups_only", workload_sha256="synthetic",
                        build_manifest_sha256="synthetic", binary_sha256="synthetic", source_archive_sha256="synthetic")
            plan_path = folder / "plan.json"
            write_new(plan_path, plan)
            write_new(folder / "execution-start.json", {"plan_sha256": sha(plan_path), "quiet_attestation": "synthetic test; no model was run"})
            receipts = []
            for job in jobs:
                report = copy.deepcopy(self.report)
                report.update(os="synthetic", arch="synthetic", logical_cpus=16, loaded_memory={"unavailable": True})
                report["cache_layout"] = job["mode"]["cache_layout"]
                report["weight_layout"] = job["mode"]["weight_layout"].replace("-", "_")
                if report["weight_layout"] == "phase_packed": report["packed_weight_bytes"] = 100
                case = report["cases"][0]
                case["memory_after"] = {"unavailable": True}
                if job["mode"]["execution"] == "joint":
                    case["active_batch_size"] = 2
                    case["execution"] = "independent_prefill_joint_decode"
                if changed_text and job["mode"]["name"] == "compact":
                    for sample in case["samples"]: sample["per_request"][0]["text"] = "different literal text"
                if control_drift and job["mode"]["name"] == "sequential-b":
                    case["median_ms"] = 120
                    case["median_pages_per_second"] = 2000 / 120
                    for sample in case["samples"]:
                        sample["wall_ms"] = 120
                        sample["pages_per_second"] = 2000 / 120
                path = folder / f"{job['id']}.json"
                write_new(path, report)
                receipts.append({"job": job["id"], "sha256": sha(path)})
            write_new(folder / "execution-complete.json", {"plan_sha256": sha(plan_path), "reports": receipts})
            with patch("compare_realistic_benchmarks.validate_plan", return_value=(plan, self.workload, self.build)):
                return compare(plan_path)

    def test_comparison_rejects_cross_mode_text_change(self):
        with self.assertRaises(ValueError): self.synthetic_comparison(changed_text=True)

    def test_comparison_accepts_exact_outputs_without_promoting(self):
        report = self.synthetic_comparison()
        self.assertTrue(report["groups"][0]["outputs_exact_across_all_modes_and_repetitions"])
        self.assertFalse(report["default_promotion"])

    def test_control_drift_disables_performance_gate(self):
        group = self.synthetic_comparison(control_drift=True)["groups"][0]
        self.assertFalse(group["control_stability_pass"])
        self.assertTrue(all(not c["meets_target_in_this_group"] for c in group["candidates"].values()))


if __name__ == "__main__":
    unittest.main()
