"""Read-only bitwise audit of a possible split prefix-K representation.

This does not implement an attention kernel, execute a model, or measure speed.
It reconstructs existing post-RoPE K/V tensors from a smaller byte representation.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct
import sys


ROOT = Path(__file__).resolve().parents[2]
TRACE_SHA = "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309"
CONFIG_SHA = "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct(data, rows, prefix, key):
    # Exact bit copies, including signed zeros: no floating-point arithmetic.
    output = bytearray()
    stored_bytes = 0
    distinct_spatial_pairs = 0
    for row in range(rows):
        for group in range(8):
            start = (row * 16 + 2 * group) * 64 * 4
            first = data[start:start + 256]
            second = data[start + 256:start + 512]
            if prefix and key:
                temporal, spatial_a, spatial_b = first[:128], first[128:], second[128:]
                output.extend(temporal + spatial_a)
                output.extend(temporal + spatial_b)
                stored_bytes += 384
                distinct_spatial_pairs += spatial_a != spatial_b
            else:
                output.extend(first)
                output.extend(first)
                stored_bytes += 256
    return bytes(output), stored_bytes, distinct_spatial_pairs


def main():
    if sys.flags.optimize:
        raise RuntimeError("This audit requires enabled Python assertions; do not use -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing report")
    trace = ROOT / "artifacts/diagnostics/stage-timing-windows-v1/trace.safetensors"
    config = ROOT / "artifacts/model/config.json"
    source_paths = [Path(__file__).resolve(), ROOT / "src/model.rs", ROOT / "src/kernels.rs", config, trace]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert sha(trace) == TRACE_SHA and sha(config) == CONFIG_SHA, "fixed fixture/config mismatch"
    c = json.loads(config.read_text(encoding="utf-8"))
    assert (c["n_heads"], c["n_kv_heads"], c["head_dim"], c["n_layers"]) == (16, 8, 64, 22)
    records = []
    with trace.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        header_len = struct.unpack_from("<Q", mapped)[0]
        header = json.loads(mapped[8:8 + header_len])
        offset = 8 + header_len
        pattern = re.compile(r"(prefill|decode\.\d+)\.layer\.(\d+)\.([kv])")
        selected = {name: value for name, value in header.items() if pattern.fullmatch(name)}
        steps = sorted({pattern.fullmatch(name)[1] for name in selected})
        assert "prefill" in steps and len(steps) > 1
        expected = {f"{step}.layer.{layer}.{kind}" for step in steps for layer in range(22) for kind in "kv"}
        assert set(selected) == expected, "missing layer or K/V tensor"
        decode_indices = sorted(int(s.split(".")[1]) for s in steps if s != "prefill")
        assert decode_indices == list(range(len(decode_indices))), "incomplete decode sequence"
        for name, item in sorted(selected.items()):
            step, _, kind = pattern.fullmatch(name).groups()
            rows, heads, dims = item["shape"]
            assert item["dtype"] == "F32" and heads == 16 and dims == 64
            assert rows == (144 if step == "prefill" else 1)
            lo, hi = item["data_offsets"]
            assert hi - lo == rows * heads * dims * 4 and offset + hi <= len(mapped)
            data = mapped[offset + lo:offset + hi]
            restored, stored, distinct = reconstruct(data, rows, step == "prefill", kind == "k")
            assert restored == data, f"not lossless: {name}"
            records.append({"tensor": name, "shape": item["shape"], "original_bytes": len(data),
                            "proposed_stored_bytes": stored, "reconstructed_sha256": hashlib.sha256(restored).hexdigest(),
                            "distinct_paired_spatial_halves": distinct, "bitwise_reconstruction_passed": True})
    # Sensitivity check: a differing temporal copy must fail reconstruction.
    altered = bytearray(512)
    altered[256] = 1
    padded = bytes(altered) + bytes(7 * 512)
    assert reconstruct(padded, 1, True, True)[0] != padded
    after = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert before == after, "input or source changed during audit"
    prefix = [r for r in records if r["tensor"].startswith("prefill.")]
    distinct = sum(r["distinct_paired_spatial_halves"] for r in prefix)
    assert distinct > 0, "fixture must exercise spatially distinct paired heads"
    report = {
        "schema_version": 1, "kind": "prefix-temporal-sharing-byte-audit-v1", "status": "passed",
        "source_and_input_sha256": before, "python": sys.version, "tensors_checked": len(records),
        "decode_steps": len(decode_indices), "prefix_rows_per_layer": 144,
        "current_compact_prefix_f32_per_token_layer": 1536,
        "proposed_prefix_f32_per_token_layer": 1280,
        "current_compact_generated_f32_per_token_layer": 1024,
        "prefix_payload_reduction_fraction": 1 / 6,
        "distinct_paired_prefix_spatial_halves": distinct,
        "duplicate_mutation_rejected": True,
        "all_reconstructions_bitwise_equal": True,
        "records": records,
        "qualification": "Byte reconstruction of one existing 144-token prefix and its saved decode steps only. No new attention kernel, full-page inference, numerical-gate qualification, memory measurement, speed measurement, or default promotion. Existing dot-product FMA/reduction order must survive any future split-memory implementation."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "tensors_checked", "decode_steps", "distinct_paired_prefix_spatial_halves")}))


if __name__ == "__main__":
    main()
