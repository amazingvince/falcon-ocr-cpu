#!/usr/bin/env python3
"""CPU-only validation and one fixed ascending-slot FP32 fold of observed scratch."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "artifacts/reference/linear-owned-workspace-fp32-v1"
REPORT_SHA = "697b717ac9949c1f90bf94eaca8f074164ec40d10f7917ca8bd4ecf3c5a7a7f4"
FIXTURE_SHA = "a8b9c60e8f5da012efca80fa6955edff814616f6372668229e0aba74e6f3d338"
M, N, K, SPLITS = 768, 144, 2304, 14
BYTES = M * N * SPLITS * 4


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(path, name, shape):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        entry = header[name]
        require(entry["dtype"] == "F32" and entry["shape"] == shape, "Fixture tensor schema changed")
        start, end = entry["data_offsets"]
        require(end - start == int(np.prod(shape)) * 4, "Invalid tensor size")
        f.seek(8 + header_len + start)
        result = f.read(end - start)
        require(len(result) == end - start, "Truncated tensor")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sys.byteorder == "little", "Little-endian artifact interpretation required")
    source = Path(__file__).read_bytes()
    report_bytes = (CAPTURE / "report.json").read_bytes()
    require(digest(report_bytes) == REPORT_SHA, "Capture report pin differs")
    report = json.loads(report_bytes)
    require(report["status"] == "captured_layout_observed_conditionally" and report["source_and_artifact_closure_unchanged"], "Capture did not pass")
    require([g["mode"] for g in report["gates"]] == ["control", "basis-0", "basis-1", "basis-2"], "Incomplete capture modes")
    for name, item in report["artifacts"].items():
        path = CAPTURE / name
        require(path.stat().st_size == item["bytes"] and hash_file(path) == item["sha256"], "Capture artifact changed: " + name)
    for gate in report["gates"]:
        require(gate["status"] == "passed", "Capture gate failed")
        for name, value in gate["validated_files"].items():
            require(hash_file(CAPTURE / gate["mode"] / name) == value, "Validated artifact differs")
    with zipfile.ZipFile(CAPTURE / "sources.zip") as archive:
        sources = report["startup"]["source_hashes"]
        require(set(archive.namelist()) == set(sources), "Source archive inventory differs")
        require(all(digest(archive.read(name)) == value for name, value in sources.items()), "Archived source bytes differ")
    fixture = ROOT / "artifacts/reference/linear-operators.safetensors"
    require(hash_file(fixture) == FIXTURE_SHA, "Original operand fixture changed")
    joins = [("input.f32le", "prefill.layer.12.w2.input", [N, K]),
             ("weight.f32le", "weights.layers.12.feed_forward.w2.weight", [M, K]),
             ("expected.f32le", "prefill.layer.12.w2.expected", [N, M])]
    for filename, key, shape in joins:
        require((CAPTURE / "control" / filename).read_bytes() == tensor_bytes(fixture, key, shape), "Control does not join exact original tensor: " + key)
    expected_bytes = (CAPTURE / "control/expected.f32le").read_bytes()
    require((CAPTURE / "control/output.f32le").read_bytes() == expected_bytes, "Captured control output differs")
    # Independently verify all basis locations using coordinate loops and bitwise
    # label/zero checks, rather than importing the original observer's reductions.
    membership = []
    for group in range(3):
        folder = CAPTURE / f"basis-{group}"
        workspace = (folder / "workspace.bin").read_bytes()
        raw = np.frombuffer(workspace[:BYTES], dtype="<u4").reshape(SPLITS, N, M)
        output = np.frombuffer((folder / "output.f32le").read_bytes(), dtype="<f4").reshape(N, M)
        weights = np.frombuffer((folder / "weight.f32le").read_bytes(), dtype="<f4").reshape(M, K)
        inputs = np.frombuffer((folder / "input.f32le").read_bytes(), dtype="<f4").reshape(N, K)
        require(np.array_equal(inputs, np.broadcast_to(np.arange(1, N + 1, dtype=np.float32)[:, None], (N, K))), "Basis row labels differ")
        require(np.count_nonzero(weights) == M, "Basis weight sparsity differs")
        for channel in range(M):
            k = group * M + channel
            require(weights[channel, k] == channel + 1, "Basis channel code differs")
            codes = (np.arange(1, N + 1, dtype=np.int64) * (channel + 1)).astype(np.float32)
            require(np.array_equal(output[:, channel].view(np.uint32), codes.view(np.uint32)), "Analytic basis output differs")
            candidates = [slot for slot in range(SPLITS) if np.array_equal(raw[slot, :, channel], codes.view(np.uint32))]
            require(len(candidates) == 1, "No unique matching partial slot")
            selected = candidates[0]
            require(all(np.count_nonzero(raw[s, :, channel]) == 0 for s in range(SPLITS) if s != selected), "Unexpected partial bytes outside selected slot")
            membership.append(selected)
    require(membership == report["layout_observation"]["conditional_k_membership"], "Independent K membership differs")
    partitions = []
    for slot in range(SPLITS):
        keys = [i for i, value in enumerate(membership) if value == slot]
        partitions.append({"slot": slot, "count": len(keys), "k_min": min(keys) if keys else None,
                           "k_end_exclusive": max(keys) + 1 if keys else None,
                           "contiguous": not keys or keys == list(range(min(keys), max(keys) + 1))})
    control_bytes = (CAPTURE / "control/workspace.bin").read_bytes()[:BYTES]
    partials = np.frombuffer(control_bytes, dtype="<f4").reshape(SPLITS, N, M)
    expected = np.frombuffer(expected_bytes, dtype="<f4").reshape(N, M)
    require(np.isfinite(partials).all() and np.isfinite(expected).all(), "Nonfinite real control data")
    folded = np.zeros((N, M), dtype=np.float32)
    for slot in range(SPLITS):
        np.add(folded, partials[slot], out=folded, dtype=np.float32)
    different = folded.view(np.uint32) != expected.view(np.uint32)
    delta = folded.astype(np.float64) - expected.astype(np.float64)
    flat_worst = int(np.abs(delta).argmax())
    worst = np.unravel_index(flat_worst, expected.shape)
    stats = {"elements": N * M, "bit_mismatches": int(np.count_nonzero(different)),
             "max_abs_error": float(np.abs(delta).max()), "rms_error": float(np.sqrt(np.mean(delta * delta))),
             "worst_flat_index": flat_worst, "worst_token_channel": [int(v) for v in worst],
             "worst_candidate": float(folded[worst]), "worst_reference": float(expected[worst]),
             "first_mismatching_flat_indices": np.flatnonzero(different)[:32].tolist()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(source)
    (output / "control-partials.f32le").write_bytes(control_bytes)
    (output / "ascending-fold.f32le").write_bytes(folded.tobytes())
    # The exact read artifacts are rehashed at closure; no input is rewritten.
    require((CAPTURE / "report.json").read_bytes() == report_bytes, "Capture report changed during replay")
    for name, item in report["artifacts"].items():
        require(hash_file(CAPTURE / name) == item["sha256"], "Capture artifact changed during replay: " + name)
    require(Path(__file__).read_bytes() == source and hash_file(fixture) == FIXTURE_SHA, "Replay source/fixture changed")
    result = {"schema_version": 1, "status": "replayed_one_fixed_fp32_fold", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "capture_report_sha256": REPORT_SHA, "fixture_sha256": FIXTURE_SHA,
              "source_sha256": digest(source), "numpy_version": np.__version__, "python": sys.version,
              "capture_artifacts_rehashed_before_after": len(report["artifacts"]), "archived_sources_verified": sources,
              "original_control_operand_output_joins_exact": True, "independent_all_basis_elements_agree": True,
              "conditional_partitions": partitions, "fold": "FP32 +0 accumulator; np.add dtype=float32 in slots0,1,...,13; one rounding per addition; no BLAS/matmul",
              "comparison": stats, "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": hash_file(p)} for p in output.iterdir()},
              "qualification": "Observed-function comparison only. Even bit equality does not establish the actual cuBLAS instruction/reduction order. K membership remains conditional on the explicitly tested scratch layout; no layout/permutation/accumulation sweep or production change."}
    with (output / "report.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
