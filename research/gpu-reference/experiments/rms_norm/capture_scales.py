"""Capture CPU RMS scales and constrain them using saved GPU normalized outputs.

The inferred scale set is an output-consistency result, NOT an actual CUDA rstd
capture or an inferred variance. The queued fused-rstd export must confirm it.
No production arithmetic, tolerance, model output, or timing is changed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
FIXTURE_SHA = "ce7345c219d8923182ff66e9aad2f4d6bad3c193a19c3445a850c2d3a90e5417"
METADATA_SHA = "148b9d4286bf413b7c9c693b920290298d5f46cd137449ac4e394430495a8962"
VARIANTS = ("production", "production_rounded_rsqrt", "cuda_shaped", "cuda_shaped_rounded_rsqrt")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(raw):
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def extract_function(source, name):
    start = source.index("fn " + name + "(")
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def scale_candidates(inputs, expected):
    # For one normal, nonzero output, its rounding interval divided by the
    # corresponding input brackets all possible positive FP32 multiplier bits.
    # F32 midpoint arithmetic is exact in F64. Widen each quotient by one F32
    # neighbor to cover F64 division/endpoint rounding; verify every candidate
    # against every output bit (including signed zero) afterward.
    index = max(range(len(inputs)), key=lambda i: abs(inputs[i]))
    x, y = abs(inputs[index]), abs(expected[index])
    y_bits = bits(y)
    require(x > 0 and 0x00800000 < y_bits < 0x7f7fffff, "non-normal anchor requires separate analysis")
    lower = (value(y_bits - 1) + y) * 0.5 / x
    upper = (value(y_bits + 1) + y) * 0.5 / x
    lo, hi = max(1, bits(lower) - 1), min(0x7f7fffff, bits(upper) + 1)
    require(0 <= hi - lo <= 8, "unexpectedly broad multiplier interval")
    targets = [bits(y) for y in expected]
    matches = []
    for candidate in range(lo, hi + 1):
        scale = value(candidate)
        if all(bits(x * scale) == target for x, target in zip(inputs, targets)):
            matches.append(candidate)
    return {"anchor_index": index, "candidate_bits_range": [lo, hi], "consistent_scale_bits": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / "artifacts/reference/layer-operators-fp32.safetensors"
    metadata_path = fixture.with_suffix(".json")
    kernel = ROOT / "src/kernels.rs"
    diagnostic = ROOT / "src/numerical_diagnostics.rs"
    sources = [Path(__file__).resolve(), Path(__file__).with_name("rstd_probe.rs"), kernel, diagnostic]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [fixture, metadata_path]}
    require(sha(fixture) == FIXTURE_SHA and sha(metadata_path) == METADATA_SHA, "fixed reference changed")
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for path in sources:
        dest = out / "sources" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    production = extract_function(kernel.read_text(encoding="utf-8"), "sum_squares_pairwise")
    generated = out / "sources/experiments/rms_norm/production_pairwise.rs"
    generated.write_text(production, encoding="utf-8", newline="\n")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(len(metadata["cases"]) == 7, "expected seven frozen cases")
    inventory = []
    with fixture.open("rb") as stream, (out / "input.bin").open("xb") as input_file, (out / "expected.bin").open("xb") as expected_file:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size)); offset = size + 8
        for case in metadata["cases"]:
            for kind, input_suffix, expected_suffix in (("attention_norm", "input", "attention_norm.expected"),
                                                         ("ffn_norm", "attention_residual.expected", "ffn_norm.expected")):
                for suffix, target in ((input_suffix, input_file), (expected_suffix, expected_file)):
                    entry = header[case["name"] + "." + suffix]
                    require(entry["dtype"] == "F32" and entry["shape"] == [case["rows"], 768], "unexpected RMS shape/dtype")
                    lo, hi = entry["data_offsets"]
                    require(hi - lo == case["rows"] * 768 * 4, "invalid tensor extent")
                    stream.seek(offset + lo); data = stream.read(hi - lo)
                    require(len(data) == hi - lo, "truncated tensor")
                    target.write(data)
                inventory.append({"name": case["name"] + "." + kind, "rows": case["rows"], "width": 768})
    env = dict(os.environ)
    compiler = shutil.which("rustc", path=env.get("PATH"))
    require(compiler is not None, "rustc must be available in PATH")
    compiler_version = subprocess.check_output([compiler, "-Vv"], env=env, text=True)
    binary = out / ("rstd_probe.exe" if sys.platform == "win32" else "rstd_probe")
    command = [compiler, "--edition=2024", "-C", "opt-level=3", str(out / "sources/experiments/rms_norm/rstd_probe.rs"), "-o", str(binary)]
    subprocess.run(command, env=env, check=True)
    source_unchanged = all(sha(ROOT / p) == digest for p, digest in source_hashes.items())
    require(source_unchanged, "source changed during build")
    subprocess.run([str(binary), str(out / "input.bin"), str(out / "scales.bin")], env=env, check=True)
    total_rows = sum(case["rows"] for case in inventory)
    require((out / "scales.bin").stat().st_size == total_rows * 4 * 12, "scale output extent mismatch")
    counts = Counter()
    variants = {name: {"output_consistent_rows": 0, "different_output_elements": 0, "max_abs": 0.0} for name in VARIANTS}
    records = []
    with (out / "input.bin").open("rb") as inp, (out / "expected.bin").open("rb") as exp, (out / "scales.bin").open("rb") as scales:
        for case in inventory:
            for row in range(case["rows"]):
                inputs = struct.unpack("<768f", inp.read(768 * 4))
                expected = struct.unpack("<768f", exp.read(768 * 4))
                require(all(math.isfinite(x) for x in (*inputs, *expected)), "nonfinite input/reference")
                consistent = scale_candidates(inputs, expected)
                counts[len(consistent["consistent_scale_bits"])] += 1
                per_variant = {}
                for name in VARIANTS:
                    raw = struct.unpack("<3I", scales.read(12))
                    s = value(raw[2]); require(math.isfinite(s), "nonfinite captured scale")
                    actual = [value(bits(x * s)) for x in inputs]
                    different = sum(bits(a) != bits(b) for a, b in zip(actual, expected))
                    error = max(abs(a - b) for a, b in zip(actual, expected))
                    require((different == 0) == (raw[2] in consistent["consistent_scale_bits"]), "candidate interval does not cover exact scale")
                    variants[name]["output_consistent_rows"] += different == 0
                    variants[name]["different_output_elements"] += different
                    variants[name]["max_abs"] = max(variants[name]["max_abs"], error)
                    per_variant[name] = {"sum_bits": raw[0], "variance_bits": raw[1], "scale_bits": raw[2]}
                records.append({"case": case["name"], "row": row, **consistent, "variants": per_variant})
    require(all(sha(ROOT / p) == digest for p, digest in {**source_hashes, **input_hashes}.items()), "bound source/input changed")
    report = {"schema_version": 1, "status": "diagnostic_capture_complete", "source_sha256": source_hashes,
              "input_sha256": input_hashes, "compiler_version": compiler_version, "compile_command": command,
              "python_version": sys.version, "platform": sys.platform, "inventory": inventory,
              "total_rows": total_rows, "elements_per_variant": total_rows * 768,
              "consistent_scale_count_histogram": dict(counts), "variants": variants, "rows": records,
              "artifact_sha256": {p.name: sha(p) for p in [out / "input.bin", out / "expected.bin", out / "scales.bin", binary]},
              "generated_production_function_sha256": sha(generated),
              "qualification": "CPU-only width768 equal-input diagnostic. Scale candidates are inferred output-consistency intervals under FP32 multiply/RNE, not observed CUDA rstd. Actual queued fused-rstd capture must confirm this inference. No GPU variance is inferred, no model inference/timing occurs, and no kernel or frozen acceptance bound changes."}
    with (out / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: report[k] for k in ["status", "total_rows", "consistent_scale_count_histogram", "variants"]}))


if __name__ == "__main__":
    main()
