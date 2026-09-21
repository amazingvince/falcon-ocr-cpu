"""Independent NumPy reconstruction and math.fsum oracles; no model/timing execution."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import struct
import sys
import zipfile

# This checker only uses elementwise NumPy; prevent implicit external thread teams.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"
FIXTURE_SHA256 = "ce7345c219d8923182ff66e9aad2f4d6bad3c193a19c3445a850c2d3a90e5417"
SIDECAR_SHA256 = "148b9d4286bf413b7c9c693b920290298d5f46cd137449ac4e394430495a8962"
COMPILED_SOURCES = {"probe": "examples/quant_probe.rs", "q4": "experiments/quantization/q4_reference.rs",
                    "q8": "experiments/quantization/q8_reference.rs", "cargo_lock": "Cargo.lock"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def validate_preserved_build(report, report_path, build_path=None):
    """Bind old or new reports to immutable evidence, never mutable workspace code."""
    build_path = pathlib.Path(build_path) if build_path else pathlib.Path(report_path).parent / "build.json"
    build = load(build_path)
    require(build.get("status") == "complete" and build.get("build_exit_code") == 0 and build.get("source_unchanged_during_build") is True, "Preserved build did not pass source-before/after checks")
    require(build.get("example") == "quant_probe", "Preserved build is not quant_probe")
    binary_name = build.get("binary", "")
    archive_name = build.get("source_archive", "source.zip")
    require(binary_name and pathlib.Path(binary_name).name == binary_name, "Unsafe preserved binary name")
    require(archive_name and pathlib.Path(archive_name).name == archive_name, "Unsafe source archive name")
    binary, archive = build_path.parent / binary_name, build_path.parent / archive_name
    require(sha(binary) == build.get("binary_sha256") == report.get("binary_sha256"), "Preserved executable/report binary hash differs")
    require(sha(archive) == build.get("source_archive_sha256"), "Preserved source archive hash differs")
    hashes = build.get("source_sha256", {})
    require(isinstance(hashes, dict) and hashes, "Missing archived source inventory")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        require(len(names) == len(set(names)) and set(names) == set(hashes), "Source ZIP inventory is missing, duplicated or unexpected")
        for name, digest in hashes.items():
            require(hashlib.sha256(source.read(name)).hexdigest() == digest, "Archived source bytes differ: " + name)
    embedded = report.get("source_sha256", {})
    for key, path in COMPILED_SOURCES.items():
        require(path in hashes and embedded.get(key) == hashes[path], "Compiled source binding differs: " + key)
    checker_key = "experiments/quantization/check_probe.py"
    require(checker_key in hashes, "Archived original checker source missing")
    return {"build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
            "binary": str(binary), "binary_sha256": sha(binary), "source_archive": str(archive),
            "source_archive_sha256": sha(archive), "source_entry_count": len(hashes),
            "compiled_source_bindings": {key: hashes[path] for key, path in COMPILED_SOURCES.items()},
            "archived_original_checker_sha256": hashes[checker_key],
            "current_checker_sha256": sha(__file__),
            "qualification": "Current strict checker may postdate the captured build; its hash is separate from the preserved original checker. No original report is rewritten."}


def validate_fixed_inputs(report):
    for field, expected in (("checkpoint", CHECKPOINT_SHA256), ("fixture", FIXTURE_SHA256), ("fixture_sidecar", SIDECAR_SHA256)):
        require(report.get(field + "_sha256") == expected, "Pinned " + field + " identity differs")
        require(sha(ROOT / report[field]) == expected, "Pinned " + field + " file bytes differ")


class Tensors:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        with self.path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(size))
        self.offset = size + 8

    def get(self, name):
        record = self.header[name]
        require(record["dtype"] == "F32", "Expected F32 tensor")
        a, b = record["data_offsets"]
        shape = record["shape"]
        require(b - a == math.prod(shape) * 4, "Tensor byte length")
        return np.memmap(self.path, mode="r", dtype="<f4", offset=self.offset + a, shape=tuple(shape))


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def quantize(weights, bits, group):
    """Independent array formulation, including odd K and partial groups."""
    weights = np.asarray(weights, dtype="<f4")
    require(weights.ndim == 2 and min(weights.shape) > 0 and np.isfinite(weights).all(), "Finite nonempty matrix")
    require(bits in (4, 8) and group in (64, 128), "Format")
    n, k = weights.shape
    groups = (k + group - 1) // group
    padded = np.pad(weights, ((0, 0), (0, groups * group - k))).reshape(n, groups, group)
    maximum = np.abs(padded).max(axis=2)
    qmax = (1 << (bits - 1)) - 1
    scale = (maximum.astype(np.float64) / qmax).astype("<f4")
    scale = np.where(maximum == 0, np.float32(0), np.maximum(scale, np.nextafter(np.float32(0), np.float32(1)))).astype("<f4")
    divisor = np.where(scale == 0, np.float32(1), scale).astype(np.float64)
    signed = np.clip(np.rint(padded.astype(np.float64) / divisor[:, :, None]), -qmax, qmax).astype(np.int8)
    dequant = (signed.astype(np.float32) * scale[:, :, None]).reshape(n, groups * group)[:, :k].copy()
    require(np.isfinite(dequant).all(), "Reconstruction overflow")
    signed = signed.reshape(n, groups * group)[:, :k]
    if bits == 4:
        nibble = np.pad(signed, ((0, 0), (0, k % 2))).astype(np.uint8) & 15
        codes = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    else:
        codes = signed.astype(np.int8)
    return codes, scale, dequant


def samples(rows, channels, qkv):
    rr = sorted({0, min(1, rows - 1), rows // 4, rows // 2, 3 * rows // 4, max(0, rows - 2), rows - 1})
    cc = set(range(min(4, channels))) | set(range(max(0, channels - 4), channels))
    for f in (1, 2, 3):
        even = (f * channels // 4) // 2 * 2
        cc |= {even, even + 1}
    if qkv:
        cc |= {1022, 1023, 1024, 1025, 1534, 1535, 1536, 1537}
    return rr, sorted(cc)


def oracle(input_, weights, rows, channels):
    values, absolute = [], []
    for row in rows:
        for channel in channels:
            # Each FP32×FP32 product is exact in FP64 for these normal fixtures;
            # fsum then avoids the Rust oracle's ascending-F64 reduction order.
            products = [float(x) * float(w) for x, w in zip(input_[row], weights[channel])]
            values.append(math.fsum(products))
            absolute.append(math.fsum(map(abs, products)))
    return np.asarray(values, dtype=np.float64), np.asarray(absolute, dtype=np.float64)


def difference(candidate, expected):
    candidate = np.asarray(candidate, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    delta = candidate - expected
    rms = float(np.sqrt(np.mean(delta * delta)))
    ref_rms = float(np.sqrt(np.mean(expected * expected)))
    return {"elements": int(delta.size), "max_abs": float(np.max(np.abs(delta))), "rms_abs": rms,
            "reference_rms": ref_rms, "relative_l2": rms / ref_rms if ref_rms else None}


def check(report_path, build_path=None):
    report = load(report_path)
    candidate_digest = sha(report_path)
    require(report["timing"] is None and report["model_integration"] is False and report["quality_qualification"] is False, "Diagnostic scope")
    validate_fixed_inputs(report)
    provenance = validate_preserved_build(report, report_path, build_path)
    model = Tensors(ROOT / report["checkpoint"])
    fixture = Tensors(ROOT / report["fixture"])
    sidecar = load(ROOT / report["fixture_sidecar"])
    require(report["original_trace_sha256"] == sidecar["source_trace_sha256"], "Source trace identity")
    expected_cases = {(c["name"], op) for c in sidecar["cases"] for op in ("qkv", "wo", "w13", "w2")}
    observed_cases = [(c["case"], c["operator"]) for c in report["operators"]]
    require(len(observed_cases) == 28 and set(observed_cases) == expected_cases, "All seven cases/four operators required")
    checks = []
    case_metadata = {c["name"]: c for c in sidecar["cases"]}
    mappings = {
        "qkv": ("attention.wqkv.weight", "attention_norm.expected", "qkv.expected"),
        "wo": ("attention.wo.weight", "attention.scaled", "attention_projection.expected"),
        "w13": ("feed_forward.w13.weight", "ffn_norm.expected", "w13.expected"),
        "w2": ("feed_forward.w2.weight", "gate.expected", "w2.expected"),
    }
    for record in report["operators"]:
        weight_suffix, input_suffix, expected_suffix = mappings[record["operator"]]
        case = case_metadata[record["case"]]
        require(record["weight_key"] == case["weight_prefix"] + weight_suffix, "Wrong original checkpoint matrix")
        require(record["input_key"] == record["case"] + "." + input_suffix, "Wrong equal-input substage")
        require(record["expected_key"] == record["case"] + "." + expected_suffix, "Wrong GPU projection output")
        weight = model.get(record["weight_key"])
        input_ = fixture.get(record["input_key"])
        expected = fixture.get(record["expected_key"])
        n, k = weight.shape
        input_ = input_.reshape(-1, k)
        require(record["shape"] == {"source_rows": len(input_), "out_dim": n, "in_dim": k}, "Shape")
        for array, field in ((weight, "weight_f32le_sha256"), (input_, "input_f32le_sha256"), (expected, "gpu_expected_f32le_sha256")):
            require(array_sha(array) == record[field], field)
        rows, channels = samples(len(input_), n, record["operator"] == "qkv")
        require(rows == record["sample_rows"] and channels == record["sample_channels"], "Outcome-independent sampling")
        gpu = expected[np.ix_(rows, channels)].reshape(-1).astype(np.float64)
        require(np.array_equal(gpu, record["sampled_gpu"]), "Sampled GPU values")
        original64, original_abs = oracle(input_, weight, rows, channels)
        double_bound = (k * np.finfo(np.float64).eps) * original_abs
        require(np.all(np.abs(np.asarray(record["sampled_original_f64"]) - original64) <= double_bound), "Original F64 oracle")
        baseline = np.asarray(record["sampled_original_fp32_ascending_fma"])
        ku = k * np.finfo(np.float32).eps / 2
        require(np.all(np.abs(baseline - original64) <= ku / (1 - ku) * original_abs), "Original FP32 accumulation bound")
        require({(v["bits"], v["group_size"]) for v in record["variants"]} == {(4,64),(4,128),(8,64),(8,128)} and len(record["variants"]) == 4, "All four formats required")
        for variant in record["variants"]:
            bits, group = variant["bits"], variant["group_size"]
            codes, scales, restored = quantize(weight, bits, group)
            require(variant["format"] == f"w{bits}a32_g{group}", "Format label")
            require(array_sha(codes) == variant["storage"]["codes_sha256"], "All code bytes differ")
            require(array_sha(scales) == variant["storage"]["scales_f32le_sha256"], "All scale bits differ")
            require(array_sha(restored) == variant["reconstructed_f32le_sha256"], "Full reconstructed matrix differs")
            require(codes.nbytes + scales.nbytes == variant["storage"]["payload_bytes"], "Payload byte count")
            reconstructed = difference(restored, weight)
            for field in ("max_abs", "rms_abs", "relative_l2"):
                require(math.isclose(reconstructed[field], variant["weight_reconstruction_error_full_matrix"][field], rel_tol=2e-11, abs_tol=1e-15), f"Reconstruction statistic {field}")
            dequant64, absolute = oracle(input_, restored, rows, channels)
            double_bound = (k * np.finfo(np.float64).eps) * absolute
            require(np.all(np.abs(np.asarray(variant["sampled_output_f64_dequantized"]) - dequant64) <= double_bound), "Dequantized F64 oracle")
            candidate = np.asarray(variant["sampled_output_fp32"], dtype=np.float64)
            require(np.array_equal(candidate.astype(np.float32).astype(np.float64), candidate), "Candidate must be exact finite FP32 values")
            bound = ku / (1 - ku) * absolute
            error = np.abs(candidate - dequant64)
            require(np.all(error <= bound), "Candidate FP32 accumulation bound")
            require(variant["fp32_accumulation_bound_violations"] == 0, "Rust accumulation status")
            require(np.allclose(bound, variant["fp32_accumulation_error_bound"], rtol=2e-12, atol=0), "Reported gamma-K bound")
            ratios = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0)
            checks.append({"case": record["case"], "operator": record["operator"], "format": variant["format"],
                "all_codes_scales_reconstructed_bits_exact": True, "sampled_elements": len(candidate),
                "f64_oracle_pass": True, "fp32_accumulation_bound_violations": 0,
                "max_fraction_of_fp32_accumulation_bound": float(ratios.max()),
                "weight_reconstruction": reconstructed,
                "quantization_only_f64_vs_original_f64": difference(dequant64, original64),
                "candidate_vs_gpu_sampled": difference(candidate, gpu)})
        print(f"checked {record['case']}.{record['operator']}", file=sys.stderr)
    for field in ("checkpoint", "fixture", "fixture_sidecar"):
        require(sha(ROOT / report[field]) == report[field + "_sha256"], f"{field} changed during check")
    require(sha(report_path) == candidate_digest, "Candidate report changed during check")
    for path_field, digest_field in (("build_manifest", "build_manifest_sha256"), ("binary", "binary_sha256"), ("source_archive", "source_archive_sha256")):
        require(sha(provenance[path_field]) == provenance[digest_field], "Preserved evidence changed during check: " + path_field)
    return {"schema_version": 1, "status": "independent_arithmetic_checks_passed", "candidate_report": str(report_path),
        "candidate_report_sha256": sha(report_path), "checkpoint_sha256": report["checkpoint_sha256"],
        "fixture_sha256": report["fixture_sha256"], "checker_sha256": sha(__file__),
        "numpy_version": np.__version__, "python": sys.version,
        "strict_provenance": provenance,
        "qualification": "Exact reconstruction-format checks and sampled F64 arithmetic bounds only; no model integration, quality selection, corpus evaluation or speed claim.",
        "float_statistic_comparison": "Full reconstruction statistics allow 2e-11 relative/1e-15 absolute for different FP64 reduction orders; codes/scales/reconstruction compare SHA256 exactly. This does not change any model parity policy.",
        "checks": checks}


def self_test():
    for bits in (4, 8):
        for group in (64, 128):
            qmax = (1 << (bits - 1)) - 1
            x = np.zeros((3, 129), dtype=np.float32)
            x[0, :8] = [2*qmax, -2*qmax, 1, 3, 5, -1, -3, -5]
            x[1, -1] = np.nextafter(np.float32(0), np.float32(1))
            codes, scales, restored = quantize(x, bits, group)
            require(np.array_equal(restored[0, :8], [2*qmax,-2*qmax,0,4,4,0,-4,-4]), "Independent ties fixture")
            require(restored[1, -1] == x[1, -1] and np.all(restored[2] == 0), "Tiny/zero fixture")
            if bits == 4: require(np.all(codes[:, -1] & 0xf0 == 0), "Odd row nibble tail")
    for invalid in (np.nan, np.inf, -np.inf):
        try:
            quantize(np.array([[invalid]]), 8, 64)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted nonfinite weight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path, nargs="?")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--build-manifest", type=pathlib.Path, help="Preserved quant_probe build manifest; default is build.json beside the report")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Independent quantization reference self-tests passed")
        return
    require(args.report and args.output, "Report and --output required")
    require(not args.output.exists(), "Refusing to overwrite evidence")
    result = check(args.report, args.build_manifest)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
