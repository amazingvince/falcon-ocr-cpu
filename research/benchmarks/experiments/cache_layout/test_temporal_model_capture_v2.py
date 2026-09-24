"""Bounded host-only preparation/comparison guards; no build or model execution."""
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

import capture_temporal_model_v2 as capture


def fixture_reports():
    tensor = {"dtype": "F32-le", "shape": [1], "elements": 1, "sha256": "a" * 64}
    names = ["prefill.logits"] + [f"decode.{n}.logits" for n in range(16)]
    smoke_ids = [1] * 16 + [263]
    tensors = {n: {**tensor, "shape": [65536], "elements": 65536} for n in names}
    tensors.update({f"fixture.layer.{n}": dict(tensor) for n in range(1904 - len(names))})
    canonical = {"tensors": tensors, "logits_argmax": [[n, [token]] for n, token in zip(names, smoke_ids)], "decode_rows": [],
                 "active_request_indices": [], "duplicate_head_tensors_checked": 748}
    def record(count, teacher=False):
        return {"teacher_forced": teacher, "precision": "fp32", "backend": "rust-gemm/avx2",
                "weight_layout": "unpacked", "packed_weight_bytes": 0, "token_ids": [1] * (count - 1) + [263],
                "output_tokens": count, "finish_reason": "eos", "input_tokens": 144,
                "text": "synthetic", "width": 0 if teacher else 256, "height": 0 if teacher else 128}
    results = [record(n) for n in [17, 2, 6]]
    mixed = {"tensors": {}, "logits_argmax": [], "decode_rows": [],
             "active_request_indices": [], "duplicate_head_tensors_checked": 1}
    for i in range(3):
        name = f"request.{i}.prefill.logits"
        mixed["tensors"][name] = {**tensor, "shape": [65536], "elements": 65536}
        mixed["logits_argmax"].append([name, [results[i]["token_ids"][0]]])
    for step in range(16):
        active = [i for i, value in enumerate(results) if value["output_tokens"] > step + 1]
        name = f"batch.0.decode.{step}.logits"
        mixed["tensors"][name] = {**tensor, "shape": [len(active), 65536], "elements": len(active) * 65536}
        mixed["logits_argmax"].append([name, [results[i]["token_ids"][step + 1] for i in active]])
        mixed["decode_rows"].append(len(active))
        mixed["active_request_indices"].append([f"batch.0.decode.{step}.request_indices", active])
    report = {"status": "completed", "cache_layout": "expanded", "runtime": capture.RUNTIME,
              "model_revision": capture.MODEL_REVISION, "weights_sha256": capture.WEIGHTS_SHA,
              "performance_measurement": False, "inputs": [{"key": "synthetic"}],
              "canonical": {"path_kind": "teacher_forced_same_prefix", "result": record(17, True), "trace": canonical},
              "independent_free_single": results, "independent_free_mixed": copy.deepcopy(results),
              "mixed_trace": mixed, "allocation": {name: {"decode_starts": 1, "decode_ends": 1,
                    "allocation_calls": 0, "requested_bytes": 0} for name in ["single", "mixed"]}}
    reports = {}
    for name, _, layout in capture.JOBS:
        reports[name] = copy.deepcopy(report)
        reports[name]["cache_layout"] = layout
    return reports


class TemporalModelPreparationTests(unittest.TestCase):
    def test_separate_target_and_distinct_binary_guards(self):
        targets = {label: str((capture.TARGET_ROOT / label).resolve()) for label in ["control", "candidate"]}
        capture.validate_targets(targets)
        with self.assertRaises(ValueError):
            capture.validate_targets({"control": targets["control"], "candidate": targets["control"]})
        built = {"projects": {label: {"target_directory": target, "binary_sha256": label} for label, target in targets.items()}}
        plan = {"build_target_directories": targets}
        capture.validate_build_identity(built, plan)
        built["projects"]["candidate"]["binary_sha256"] = "control"
        with self.assertRaises(ValueError):
            capture.validate_build_identity(built, plan)

    def test_cargo_manifest_source_freshness_and_target_binding(self):
        project = (capture.ROOT / "synthetic-copy").resolve()
        target = (capture.TARGET_ROOT / "control").resolve()
        library = {"reason": "compiler-artifact", "manifest_path": str(project / "Cargo.toml"), "fresh": False,
                   "target": {"name": "falcon_ocr", "src_path": str(project / "src/lib.rs"), "kind": ["lib"]},
                   "profile": {"test": False}, "filenames": [str(target / "lib.rlib")], "executable": None}
        test = copy.deepcopy(library)
        test.update(target={"name": capture.TEST_NAME, "src_path": str(project / "tests" / (capture.TEST_NAME + ".rs")), "kind": ["test"]},
                    profile={"test": True}, filenames=[str(target / "test.exe")], executable=str(target / "test.exe"))
        def check(items):
            return capture.checked_emitted_executable("\n".join(json.dumps(x) for x in items), project, target)
        self.assertEqual(check([library, test]), target / "test.exe")
        for kind in ["manifest", "source", "fresh", "target", "library"]:
            items = copy.deepcopy([library, test])
            if kind == "manifest": items[1]["manifest_path"] = str(project.parent / "other/Cargo.toml")
            if kind == "source": items[0]["target"]["src_path"] = str(project.parent / "other/src/lib.rs")
            if kind == "fresh": items[0]["fresh"] = True
            if kind == "target": items[1]["executable"] = str(target.parent / "other/test.exe")
            if kind == "library": items.pop(0)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check(items)

    def test_frozen_candidate_and_genuine_original_source(self):
        native, _ = capture.read_json(capture.NATIVE / "build.json")
        archive = capture.checked_bytes(capture.NATIVE / "source.zip", capture.PINS[
            "artifacts/diagnostics/prefix-temporal-candidate-windows-v1/source.zip"])
        members = capture.native_members(archive, native)
        self.assertEqual(len(members), 30)
        changed = set()
        for name, raw in members.items():
            if name not in native["original_source_sha256"]:
                changed.add(name)
            elif capture.checked_bytes(capture.ROOT / name, native["original_source_sha256"][name]) != raw:
                changed.add(name)
        self.assertEqual(changed, {"src/config.rs", "src/lib.rs", "src/model.rs", "src/kernels.rs",
                                   "src/temporal_candidate.rs", "src/temporal_model_tests.rs"})

    def test_changed_missing_and_unlisted_archive_entries_rejected(self):
        native, _ = capture.read_json(capture.NATIVE / "build.json")
        raw = (capture.NATIVE / "source.zip").read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = {name: z.read(name) for name in z.namelist()}
        for kind in ["changed", "missing", "unlisted"]:
            with self.subTest(kind=kind):
                altered = dict(members)
                if kind == "changed":
                    altered["src/kernels.rs"] += b"\n"
                elif kind == "missing":
                    del altered["src/kernels.rs"]
                else:
                    altered["src/not-frozen.rs"] = b"// extra"
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w") as z:
                    for name, data in altered.items():
                        z.writestr(name, data)
                with self.assertRaises(ValueError):
                    capture.native_members(output.getvalue(), native)

    def test_complete_equal_comparison(self):
        result = capture.compare_reports(fixture_reports(), [1] * 16 + [263])
        self.assertEqual(result["canonical_tensor_count"], 1904)
        self.assertEqual(result["canonical_actual_argmax_decisions"], 17)
        self.assertEqual(result["allocation_intervals"], 10)
        self.assertFalse(result["performance_claim"])

    def test_missing_invocation_and_missing_tensor_rejected(self):
        for kind in ["invocation", "tensor", "logits"]:
            reports = fixture_reports()
            if kind == "invocation":
                del reports["expanded-after"]
            elif kind == "tensor":
                del reports["candidate"]["canonical"]["trace"]["tensors"]["prefill.logits"]
            else:
                reports["candidate"]["canonical"]["trace"]["logits_argmax"].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                capture.compare_reports(reports, [1] * 16 + [263])

    def test_tensor_shape_dtype_bits_and_nonfinite_dimension_rejected(self):
        for field, value in [("shape", [2]), ("dtype", "BF16"), ("sha256", "b" * 64), ("shape", [float("nan")])]:
            reports = fixture_reports()
            reports["candidate"]["canonical"]["trace"]["tensors"]["prefill.logits"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                capture.compare_reports(reports, [1] * 16 + [263])

    def test_free_prediction_order_stop_and_teacher_path_rejected(self):
        for kind in ["id", "text", "stop", "order", "teacher", "count", "dimension"]:
            reports = fixture_reports()
            group = reports["candidate"]["independent_free_mixed"]
            if kind == "id": group[0]["token_ids"][0] = 2
            if kind == "text": group[0]["text"] = "changed"
            if kind == "stop": group[0]["finish_reason"] = "length"
            if kind == "order": group.reverse()
            if kind == "teacher": group[0]["teacher_forced"] = True
            if kind == "count": group.pop()
            if kind == "dimension": group[0]["width"] = 128
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                capture.compare_reports(reports, [1] * 16 + [263])

    def test_trace_argmax_compaction_and_nonzero_allocations_rejected(self):
        for kind in ["argmax", "rows", "schedule", "calls", "callbacks"]:
            reports = fixture_reports()
            candidate = reports["candidate"]
            if kind == "argmax": candidate["canonical"]["trace"]["logits_argmax"][0][1][0] = 2
            if kind == "rows": candidate["mixed_trace"]["decode_rows"] = [3] * 16
            if kind == "schedule": candidate["mixed_trace"]["active_request_indices"].pop()
            if kind == "calls": candidate["allocation"]["mixed"]["allocation_calls"] = 1
            if kind == "callbacks": candidate["allocation"]["mixed"]["decode_starts"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                capture.compare_reports(reports, [1] * 16 + [263])


if __name__ == "__main__":
    unittest.main()
