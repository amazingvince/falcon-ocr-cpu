"""Observed CUDA rsqrt table and exhaustive normal-domain rescaling diagnostic.

Run only through the GPU owner's isolated, pinned reference environment.
No model integration, tuning, timing qualification or inferred GPU variance.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from reference_preflight import preflight

TABLE_BEGIN = 0x3F800000
TABLE_END = 0x40800000
TABLE_LENGTH = 1 << 24
MANTISSA_LENGTH = 1 << 23
EXPONENTS = range(104, 255)
BOUNDARIES = (0, 1, 0x1FFFFF, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF)
SOURCE_PATHS = (
    "research/gpu-reference/experiments/rms_norm/export_rsqrt_table.py",
    "research/gpu-reference/experiments/rms_norm/rsqrt_lookup.rs",
    "research/gpu-reference/experiments/rms_norm/rsqrt_lookup_probe.rs",
    "research/gpu-reference/experiments/rms_norm/test_export_rsqrt_table.py",
    "scripts/reference_preflight.py",
    "scripts/fetch_reference.py",
    "requirements/reference-lock.txt",
    "reference/manifest.json",
    "artifacts/model/artifact-manifest.json",
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_chunk_size(size):
    if type(size) is not int or not 4096 <= size <= (1 << 20) or size & (size - 1):
        raise ValueError("chunk size must be a power of two in [4096,1048576]")


def exponent_parameters(biased):
    if type(biased) is not int or not 104 <= biased <= 254:
        raise ValueError("unsupported exponent")
    exponent = biased - 127
    parity = exponent % 2
    scale_exponent = -(exponent // 2)
    return parity * MANTISSA_LENGTH, scale_exponent


def execute(args, source_hashes):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    environment = preflight(ROOT)
    if torch.__version__ != "2.11.0+cu130" or torch.version.git_version != "70d99e998b4955e0049d13a98d77ae1b14db1f45":
        raise ValueError("Torch implementation identity differs")
    report = {
        "schema_version": 1, "status": "running", "environment": environment,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_git_revision": torch.version.git_version,
        "torch_build_configuration": torch.__config__.show(),
        "invocation": [sys.executable, *sys.argv], "source_sha256": source_hashes,
        "table": {"input_begin_bits": TABLE_BEGIN, "input_end_bits_exclusive": TABLE_END,
                  "entries": TABLE_LENGTH, "format": "ascending-input-bits, raw little-endian F32 outputs"},
        "domain": {"biased_exponent_begin": 104, "biased_exponent_end_inclusive": 254,
                   "mantissas_per_exponent": MANTISSA_LENGTH,
                   "arguments": len(EXPONENTS) * MANTISSA_LENGTH,
                   "minimum_argument_bits": 0x34000000, "maximum_argument_bits": 0x7F7FFFFF},
        "chunk_size": args.chunk_size, "boundary_mantissas": list(BOUNDARIES),
        "boundary_format": "little-endian U32 argument bits followed by observed GPU output bits; exponent-major ascending boundary order",
        "mapping": "k=e-127; index=((k mod 2)<<23)|mantissa; multiply canonical rsqrt by2^(-floor(k/2))",
        "comparison": "exact F32 bits; GPU FP32 power-of-two multiplication independently cross-checks integer output-exponent adjustment",
        "records": [], "timing_qualification": False, "model_integration": False,
        "qualification": "Observed-function diagnostic only. No shipping design, speed claim, inferred native RMS variance or tolerance adjustment.",
    }
    json_new(args.output / "startup.json", report)
    artifacts = {}
    with torch.inference_mode():
        table = torch.empty(TABLE_LENGTH, device="cuda:0", dtype=torch.float32)
        with (args.output / "canonical-inputs.f32le").open("xb") as input_file, (args.output / "rsqrt-table.f32le").open("xb") as table_file:
            for offset in range(0, TABLE_LENGTH, args.chunk_size):
                argument_bits = torch.arange(TABLE_BEGIN + offset, TABLE_BEGIN + offset + args.chunk_size,
                                             device="cuda:0", dtype=torch.int32)
                observed = torch.rsqrt(argument_bits.view(torch.float32))
                bits = observed.view(torch.int32)
                if not bool(((bits >= 0x3F000000) & (bits <= 0x3F800000)).all().item()):
                    raise ValueError("Canonical observed result is outside positive normal [0.5,1]")
                table[offset:offset + args.chunk_size].copy_(observed)
                input_file.write(argument_bits.cpu().numpy().astype("<i4", copy=False).tobytes())
                table_file.write(observed.cpu().numpy().astype("<f4", copy=False).tobytes())
        for filename in ("canonical-inputs.f32le", "rsqrt-table.f32le"):
            path = args.output / filename
            if path.stat().st_size != TABLE_LENGTH * 4:
                raise ValueError("Incomplete canonical artifact")
            artifacts[filename] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        print(json.dumps({"canonical_entries_exported": TABLE_LENGTH, "table_sha256": artifacts["rsqrt-table.f32le"]["sha256"]}), flush=True)
        total_mismatches = total_mapping_mismatches = total_checked = 0
        with (args.output / "boundary-pairs.u32le").open("xb") as boundary_file:
            for biased in EXPONENTS:
                base, scale_exponent = exponent_parameters(biased)
                scale = math.ldexp(1.0, scale_exponent)
                record = {"biased_exponent": biased, "canonical_base_index": base,
                          "scale_exponent": scale_exponent, "checked": 0, "mismatches": 0,
                          "bit_mapping_mismatches": 0, "max_positive_f32_bit_steps": 0, "first_mismatches": []}
                for start in range(0, MANTISSA_LENGTH, args.chunk_size):
                    end = start + args.chunk_size
                    argument_bits = torch.arange((biased << 23) + start, (biased << 23) + end,
                                                 device="cuda:0", dtype=torch.int32)
                    observed = torch.rsqrt(argument_bits.view(torch.float32))
                    canonical = table[base + start:base + end]
                    # This FP32 power-of-two multiply is independent of the
                    # Rust bit adjustment; all its results remain normal.
                    expected = canonical * scale
                    mapped_bits = canonical.view(torch.int32) + scale_exponent * (1 << 23)
                    expected_bits, observed_bits = expected.view(torch.int32), observed.view(torch.int32)
                    mapping_mismatches = int((mapped_bits != expected_bits).sum().item())
                    differing = observed_bits != expected_bits
                    mismatches = int(differing.sum().item())
                    steps = int((observed_bits - expected_bits).abs().max().item())
                    if mismatches and len(record["first_mismatches"]) < 8:
                        indices = differing.nonzero(as_tuple=False).flatten()[:8-len(record["first_mismatches"])]
                        for local in indices.cpu().tolist():
                            record["first_mismatches"].append({"argument_bits": (biased << 23) + start + local,
                                "observed_bits": int(observed_bits[local].item()), "rescaled_bits": int(expected_bits[local].item()),
                                "canonical_output_bits": int(canonical[local].view(torch.int32).item())})
                    for mantissa in BOUNDARIES:
                        if start <= mantissa < end:
                            boundary_file.write(struct.pack("<II", (biased << 23) | mantissa,
                                                           int(observed_bits[mantissa - start].item())))
                    record["checked"] += args.chunk_size
                    record["mismatches"] += mismatches
                    record["bit_mapping_mismatches"] += mapping_mismatches
                    record["max_positive_f32_bit_steps"] = max(record["max_positive_f32_bit_steps"], steps)
                if record["checked"] != MANTISSA_LENGTH:
                    raise ValueError("Incomplete exponent domain")
                report["records"].append(record)
                total_checked += record["checked"]
                total_mismatches += record["mismatches"]
                total_mapping_mismatches += record["bit_mapping_mismatches"]
                if (biased - 104) % 16 == 0 or biased == 254:
                    print(json.dumps({"completed_biased_exponent": biased, "arguments_checked": total_checked,
                                      "exact_mismatches": total_mismatches, "bit_mapping_mismatches": total_mapping_mismatches}), flush=True)
    if total_checked != report["domain"]["arguments"] or len(report["records"]) != 151:
        raise ValueError("Incomplete exhaustive coverage")
    boundary_path = args.output / "boundary-pairs.u32le"
    if boundary_path.stat().st_size != 151 * len(BOUNDARIES) * 8:
        raise ValueError("Incomplete observed boundary fixture")
    artifacts[boundary_path.name] = {"bytes": boundary_path.stat().st_size, "sha256": sha(boundary_path)}
    for name, expected in artifacts.items():
        if sha(args.output / name) != expected["sha256"]:
            raise ValueError("Captured artifact changed: " + name)
    after = {name: sha(ROOT / name) for name in SOURCE_PATHS}
    if after != source_hashes:
        raise ValueError("Source closure changed during exhaustive capture")
    report.update(status="passed_exact" if total_mismatches == total_mapping_mismatches == 0 else "failed_exact",
                  total_checked=total_checked, total_mismatches=total_mismatches,
                  total_bit_mapping_mismatches=total_mapping_mismatches, artifacts=artifacts,
                  source_after_sha256=after, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json_new(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "total_checked": total_checked,
                      "report_sha256": sha(args.output / "report.json")}), flush=True)
    return 0 if report["status"] == "passed_exact" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    args = parser.parse_args()
    validate_chunk_size(args.chunk_size)
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in SOURCE_PATHS:
        data = (ROOT / name).read_bytes()
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        source_hashes[name] = hashlib.sha256(data).hexdigest()
    try:
        return execute(args, source_hashes)
    except Exception as error:
        json_new(args.output / "failed.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                 "source_sha256": source_hashes, "qualification": "Incomplete or invalid diagnostic; cannot qualify the table."})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
