"""Offline protocol regression tests: no subprocesses, model loads, or asset hashes."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("candidate_protocol", Path(__file__).with_name("benchmark_candidate.py"))
bc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bc)


class FakeChild:
    pid = 12345

    def __init__(self, first_wait=None):
        self.first_wait = first_wait
        self.waits = 0
        self.killed = False
        self.terminated = False

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1 and self.first_wait is not None:
            raise self.first_wait
        return -9 if self.killed or self.terminated else 0

    def poll(self):
        return -9 if self.killed or self.terminated else None

    def kill(self):
        self.killed = True

    def terminate(self):
        self.terminated = True


class CandidateProtocolTests(unittest.TestCase):
    def protocol(self, folder):
        plan = {
            "kind": "single-page-isolated-candidate-bracket-v1",
            "output_directory": str(folder), "host_platform": platform.platform(),
            "files_sha256": {}, "workload": "unused-workload.json",
            "builds": {key: str(folder / key / "build.json") for key in ("control", "candidate")},
            "repetitions": 3, "max_process_seconds": 900,
            "control_drift_limit_percent": 5, "target_latency_reduction_percent": 5,
            "default_promotion": False, "historical_baseline": str(folder / "historical.json"),
            "expected_signatures": [],
            "jobs": [{"name": "control-before", "build": "control"},
                     {"name": "candidate", "build": "candidate"},
                     {"name": "control-after", "build": "control"}],
        }
        workload = {"runtime": {"warmup": 2, "threads": 16, "backend": "avx2"}}
        builds = {key: ({"binary_sha256": key, "source_archive": "source.zip"}, folder / key / "unused.exe") for key in ("control", "candidate")}
        required = {str(Path(bc.__file__).resolve()), str(bc.ROOT / "research/benchmarks/scripts/realistic_benchmark.py"),
                    str(folder / "protocol-source.zip"), plan["workload"], plan["historical_baseline"], *plan["builds"].values()}
        for label, (build, binary) in builds.items():
            required.update((str(binary), str(Path(plan["builds"][label]).parent / build["source_archive"])))
        plan["files_sha256"] = {name: "f" * 64 for name in required}
        return plan, workload, builds

    def save(self, path, plan):
        raw = json.dumps(plan).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_valid_fixed_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            plan, workload, builds = self.protocol(path.parent)
            digest = self.save(path, plan)
            with mock.patch.object(bc.rb, "read", return_value=workload), \
                 mock.patch.object(bc.rb, "verify"), mock.patch.object(bc.rb, "validate_inputs"), \
                 mock.patch.object(bc.rb, "validate_build", side_effect=[builds["control"], builds["candidate"]]):
                self.assertEqual(bc.validate(path, digest)[0], plan)

    def test_wrong_supplied_plan_hash_rejected_before_input_work(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            plan, _, _ = self.protocol(path.parent)
            self.save(path, plan)
            with mock.patch.object(bc.rb, "validate_inputs") as check:
                with self.assertRaisesRegex(ValueError, "Plan hash mismatch"):
                    bc.validate(path, "0" * 64)
                check.assert_not_called()

    def test_empty_bound_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            plan, workload, builds = self.protocol(path.parent)
            plan["files_sha256"] = {}
            digest = self.save(path, plan)
            with mock.patch.object(bc.rb, "read", return_value=workload), \
                 mock.patch.object(bc.rb, "verify"), mock.patch.object(bc.rb, "validate_inputs"), \
                 mock.patch.object(bc.rb, "validate_build", side_effect=[builds["control"], builds["candidate"]]):
                with self.assertRaisesRegex(ValueError, "Bound file inventory changed"):
                    bc.validate(path, digest)

    def test_changed_job_order_or_repetitions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            original, workload, builds = self.protocol(path.parent)
            for changed in ("jobs", "repetitions"):
                with self.subTest(changed=changed):
                    plan = copy.deepcopy(original)
                    if changed == "jobs":
                        plan["jobs"][1]["build"] = "control"
                    else:
                        plan["repetitions"] = 1
                    digest = self.save(path, plan)
                    with mock.patch.object(bc.rb, "read", return_value=workload), \
                         mock.patch.object(bc.rb, "verify"), mock.patch.object(bc.rb, "validate_inputs"), \
                         mock.patch.object(bc.rb, "validate_build", return_value=builds["control"]):
                        with self.assertRaisesRegex(ValueError, "Fixed bracket changed"):
                            bc.validate(path, digest)

    def lifecycle(self, kind):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            plan, workload, builds = self.protocol(folder)
            error = KeyboardInterrupt() if kind == "interrupt" else (
                subprocess.TimeoutExpired(["unused.exe"], 900) if kind == "timeout" else None)
            child = FakeChild(error)
            args = argparse.Namespace(plan=folder / "plan.json", quiet_attestation="mocked offline test",
                                      plan_sha256="f" * 64)
            actual_write = bc.rb.write_new

            def write(path, value):
                if kind == "start_receipt" and str(path).endswith(".start.json"):
                    raise OSError("Synthetic start receipt failure")
                actual_write(path, value)

            with mock.patch.object(bc, "validate", return_value=(plan, workload, builds)), \
                 mock.patch.object(bc, "baseline_signatures", return_value=[]), \
                 mock.patch.object(bc.rb, "read", return_value={}), \
                 mock.patch.object(bc.rb, "sha", return_value="f" * 64), \
                 mock.patch.object(bc.rb, "command_for", return_value=["unused.exe"]), \
                 mock.patch.object(bc.rb, "write_new", side_effect=write), \
                 mock.patch.object(bc.subprocess, "Popen", return_value=child):
                with self.assertRaises((KeyboardInterrupt, OSError, ValueError)):
                    bc.run(args)
            self.assertTrue(child.killed or child.terminated, "Failed bracket left benchmark child alive")
            self.assertGreaterEqual(child.waits, 1, "Child was not reaped")
            if kind in ("interrupt", "timeout"):
                self.assertGreaterEqual(child.waits, 2, "Interrupted/timed-out child was not reaped")

    def test_interrupt_cleans_up_child(self):
        self.lifecycle("interrupt")

    def test_start_receipt_failure_cleans_up_child(self):
        self.lifecycle("start_receipt")

    def test_timeout_cleans_up_child(self):
        self.lifecycle("timeout")


if __name__ == "__main__":
    unittest.main()
