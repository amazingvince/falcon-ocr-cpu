"""Read-only independent audit of the two preserved v1 RMS scale captures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import test_capture_scales as oracle

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = oracle.capture.VARIANTS


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            result.update(part)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to replace a review")
    bound = {}
    def bind(path, expected=None):
        path = path.resolve()
        raw = digest(path)
        require(expected is None or raw == expected, "hash mismatch: " + str(path))
        bound[path.relative_to(ROOT).as_posix()] = raw
        return raw
    for name in ("capture_scales.py", "rstd_probe.rs", "test_capture_scales.py", "review_saved_scales.py"):
        bind(Path(__file__).with_name(name))
    directories = [ROOT / "artifacts/diagnostics" / name for name in
                   ("rms-scales-windows-v1", "rms-scales-wsl-v1")]
    reports, identities = [], []
    for directory in directories:
        report_path = directory / "report.json"
        report_hash = bind(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["status"] == "diagnostic_capture_complete", "incomplete capture")
        for name, expected in {**report["source_sha256"], **report["input_sha256"]}.items():
            bind(ROOT / name.replace("\\", "/"), expected)
        for name, expected in report["source_sha256"].items():
            bind(directory / "sources" / name.replace("\\", "/"), expected)
        for name, expected in report["artifact_sha256"].items():
            bind(directory / name, expected)
        generated = directory / "sources/experiments/rms_norm/production_pairwise.rs"
        bind(generated, report["generated_production_function_sha256"])
        preserved_kernel = directory / "sources/src/kernels.rs"
        require(generated.read_text(encoding="utf-8") == oracle.capture.extract_function(
            preserved_kernel.read_text(encoding="utf-8"), "sum_squares_pairwise"),
            "generated function differs from preserved production source")
        reports.append(report)
        identities.append({"report": report_path.relative_to(ROOT).as_posix(),
                           "report_sha256": report_hash,
                           "compiler_version": report["compiler_version"],
                           "artifact_sha256": report["artifact_sha256"]})
    windows, linux = reports
    for key in ("inventory", "total_rows", "elements_per_variant", "rows", "variants",
                "consistent_scale_count_histogram", "generated_production_function_sha256"):
        require(windows[key] == linux[key], "platform report difference: " + key)
    for name in ("input.bin", "expected.bin", "scales.bin"):
        require(windows["artifact_sha256"][name] == linux["artifact_sha256"][name],
                "platform binary data difference: " + name)

    fixture = ROOT / "artifacts/reference/layer-operators-fp32.safetensors"
    metadata = json.loads(fixture.with_suffix(".json").read_text(encoding="utf-8"))
    require(digest(fixture) == oracle.capture.FIXTURE_SHA and
            digest(fixture.with_suffix(".json")) == oracle.capture.METADATA_SHA, "fixed fixture")
    inputs, expected = [], []
    inventory = []
    with fixture.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        for case in metadata["cases"]:
            require(case["norm_epsilon"] == 2 ** -23, "fixed epsilon differs")
            for kind, source, target in (("attention_norm", "input", "attention_norm.expected"),
                                         ("ffn_norm", "attention_residual.expected", "ffn_norm.expected")):
                for suffix, parts in ((source, inputs), (target, expected)):
                    entry = header[case["name"] + "." + suffix]
                    require(entry["dtype"] == "F32" and entry["shape"] == [case["rows"], 768], "fixture shape")
                    lo, hi = entry["data_offsets"]
                    stream.seek(8 + size + lo)
                    parts.append(stream.read(hi - lo))
                inventory.append({"name": case["name"] + "." + kind, "rows": case["rows"], "width": 768})
    require(inventory == windows["inventory"], "stage inventory differs")
    for name, parts in (("input.bin", inputs), ("expected.bin", expected)):
        require(hashlib.sha256(b"".join(parts)).hexdigest() == windows["artifact_sha256"][name],
                "captured operand differs from actual pinned stage: " + name)

    rows_checked, output_bits_checked = 0, 0
    counts = Counter()
    with (directories[0] / "input.bin").open("rb") as inp, \
         (directories[0] / "expected.bin").open("rb") as exp, \
         (directories[0] / "scales.bin").open("rb") as scale_stream:
        for record in windows["rows"]:
            x = struct.unpack("<768I", inp.read(768 * 4))
            y = struct.unpack("<768I", exp.read(768 * 4))
            anchor = max(range(768), key=lambda i: x[i] & 0x7fffffff)
            require(anchor == record["anchor_index"], "anchor identity")
            lo, hi = oracle.anchor_interval(x[anchor], y[anchor])
            require(0 <= hi - lo <= 8, "unexpected exact interval width")
            require(record["candidate_bits_range"][0] <= lo <= hi <= record["candidate_bits_range"][1],
                    "reported search does not cover exact anchor interval")
            exact_matches = []
            for scale in range(lo, hi + 1):
                products = [oracle.exact_product_bits(v, scale) for v in x]
                output_bits_checked += len(products)
                if products == list(y):
                    exact_matches.append(scale)
            require(exact_matches == record["consistent_scale_bits"], "exact-integer full-row candidate mismatch")
            counts[len(exact_matches)] += 1
            for variant in VARIANTS:
                triple = struct.unpack("<3I", scale_stream.read(12))
                require(triple == tuple(record["variants"][variant][field] for field in
                                        ("sum_bits", "variance_bits", "scale_bits")), "captured triple/report mismatch")
            rows_checked += 1
        require(not inp.read(1) and not exp.read(1) and not scale_stream.read(1), "trailing captured data")
    require(rows_checked == windows["total_rows"] == 1444, "row count")
    require(dict(counts) == {1: 1444}, "one-scale result differs")
    tests = subprocess.run([sys.executable, str(Path(__file__).with_name("test_capture_scales.py")), "-v"],
                           capture_output=True, text=True, check=True)
    require(all(digest(ROOT / name) == expected for name, expected in bound.items()), "bound source/artifact changed")
    result = {"schema_version": 1, "status": "independent_scale_review_passed",
              "captures": identities, "checked_sha256": bound,
              "independent_integer_oracle_rows": rows_checked,
              "candidate_output_bits_checked": output_bits_checked,
              "cross_platform_scale_triples_equal": rows_checked * 4,
              "consistent_scale_count_histogram": dict(counts),
              "cpu_variants": windows["variants"],
              "synthetic_tests": {"passed": 7, "log": tests.stdout + tests.stderr},
              "confirmed_arithmetic_defects": [],
              "qualification": "The positive finite FP32 scale search is complete for the accepted normal anchors in these pinned rows under one FP32 RNE multiplication. The exact-integer oracle independently checks all anchor intervals and every candidate against all768 output bits. No actual CUDA rstd or variance is inferred as observed; the queued CUDA capture is required. No inference, new probe execution, or timing measurements occurred during this review.",
              "provenance_limits": ["Original capture source/input hashes and preserved source copies are verified; source/function/operand/scale bytes agree across platforms. This review is after-the-run evidence, not retroactive startup attestation.",
                                     "The original capture records compiler version/command and final executable hash, not a hermetic compiler/linker dependency closure or an executable pre-run hash.",
                                     "The f64 sqrt/reciprocal then F32 cast variant is an explicit CPU numerical candidate, not a proof of correctly rounded mathematical reciprocal square root or CUDA rsqrt behavior."]}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "rows": rows_checked,
                      "output_bits_checked": output_bits_checked, "report_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
