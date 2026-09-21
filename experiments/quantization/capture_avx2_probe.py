"""Capture/check the std-only AVX2 experiment; no inference or speed measurements."""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import zipfile

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from check_probe import array_sha, difference, oracle, quantize, require, sha

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_NAMES = ["q4_reference.rs", "q4_avx2.rs", "q4_avx2_probe.rs", "check_probe.py", "capture_avx2_probe.py"]
WIDTHS = [1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 768, 1024, 2304]
QUALIFICATION = "Synthetic operator functionality only. No model weights/GPU fixtures, model integration, OCR quality, or performance qualification. No timings are collected."


def source_hashes():
    return {"experiments/quantization/" + name: sha(pathlib.Path(__file__).parent / name) for name in SOURCE_NAMES}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_new(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_arithmetic(report):
    require(report.get("schema") == 1 and report.get("scope") == "synthetic operator only", "Scope")
    require(report.get("timing") is None and report.get("model_integration") is False and report.get("quality_qualification") is False, "No inference/performance qualification")
    require(report.get("avx2_fma_available") is True, "AVX2/FMA execution required")
    cases = report["cases"]
    require([(c["k"], c["group_size"]) for c in cases] == [(k, g) for k in WIDTHS for g in (64, 128)], "Exact predetermined width/group inventory")
    details = []
    for case in cases:
        k, n, group = case["k"], case["n"], case["group_size"]
        require(n == 7, "Predetermined odd output channel count")
        # Independent elementwise FP32 construction; no result-based sampling.
        idx = np.arange(n * k, dtype=np.int64)
        weights = (((idx * 67 % 199).astype(np.float32) - np.float32(99)) / np.float32(31)).reshape(n, k)
        observed_weights = np.asarray(case["original_weights"], dtype="<f4").reshape(n, k)
        require(array_sha(observed_weights) == array_sha(weights), "Predetermined original weights")
        packed, scales, restored = quantize(weights, 4, group)
        require(array_sha(np.asarray(case["packed_codes"], dtype=np.uint8).reshape(packed.shape)) == array_sha(packed), "Full packed Q4 bytes")
        require(array_sha(np.asarray(case["scales"], dtype="<f4").reshape(scales.shape)) == array_sha(scales), "Full scale bytes")
        require([b["rows"] for b in case["batches"]] == [1, 2, 4, 8], "All four live row counts")
        for batch in case["batches"]:
            rows = batch["rows"]
            idx = np.arange(rows * k, dtype=np.int64)
            input_ = (((idx * 101 % 257).astype(np.float32) - np.float32(128)) / np.float32(37)).reshape(rows, k)
            observed_input = np.asarray(batch["input"], dtype="<f4").reshape(rows, k)
            require(array_sha(observed_input) == array_sha(input_), "Predetermined activations")
            original64, _ = oracle(input_, weights, range(rows), range(n))
            restored64, absolute = oracle(input_, restored, range(rows), range(n))
            ku = k * 2.0 ** -24
            bound = ku / (1 - ku) * absolute
            metrics = {}
            for backend in ("scalar", "avx2"):
                # Restore FP32 bits from Rust's shortest FP32 decimal strings.
                candidate = np.asarray(batch[backend], dtype="<f4").astype(np.float64)
                require(candidate.shape == (rows * n,) and np.isfinite(candidate).all(), "Finite full output shape")
                error = np.abs(candidate - restored64)
                require(np.all(error <= bound), f"{backend} frozen gamma_K bound K={k} rows={rows} group={group}")
                ratio = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0)
                metrics[backend] = {"arithmetic_vs_fp64_dequantized": difference(candidate, restored64),
                                    "max_fraction_of_frozen_arithmetic_bound": float(ratio.max()),
                                    "total_error_vs_fp64_original": difference(candidate, original64),
                                    "output_f32le_sha256": array_sha(candidate.astype("<f4"))}
            details.append({"k": k, "n": n, "group_size": group, "rows": rows,
                            "outputs_checked_per_backend": rows * n,
                            "quantization_only_fp64_dequantized_minus_original": difference(restored64, original64),
                            "scalar": metrics["scalar"], "avx2": metrics["avx2"],
                            "avx2_vs_scalar": difference(np.asarray(batch["avx2"], dtype="<f4"), np.asarray(batch["scalar"], dtype="<f4"))})
    return {"passed": True, "cases": len(cases), "batch_cases": len(details),
            "outputs_per_backend": sum(d["outputs_checked_per_backend"] for d in details),
            "weights_independently_reconstructed": sum(c["k"] * c["n"] for c in cases),
            "arithmetic_bound": "unchanged gamma_K * sum(abs(FP32_activation * FP32_dequantized_weight)); gamma_K=K*2^-24/(1-K*2^-24)",
            "oracle": "Independent NumPy full code/scale reconstruction and math.fsum FP64 sums for every output; no sampled output omissions",
            "details": details}


def validate_build(directory, report):
    build_path = directory / "build.json"
    build = json.loads(build_path.read_text())
    require(build.get("status") == "complete" and build.get("source_unchanged_during_build_and_run") is True, "Successful source window")
    sources = build["source_sha256"]
    require(set(sources) == {"experiments/quantization/" + name for name in SOURCE_NAMES}, "Exact source inventory")
    require(report["compiled_source_inventory_sha256"] == canonical_sha(sources), "Embedded startup source map")
    for field in ("binary", "test_binary", "source_archive", "report", "test_log"):
        name = build[field]
        require(pathlib.Path(name).name == name, "Basename artifact path")
        require(sha(directory / name) == build[field + "_sha256"], field + " hash")
    with zipfile.ZipFile(directory / build["source_archive"]) as archive:
        require(len(archive.namelist()) == len(sources) and set(archive.namelist()) == set(sources), "Archive inventory")
        for name, digest in sources.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Archived source bytes")
    require(build["unit_tests_passed"] == 9, "Expected nine tests, including zero allocations")
    return {"build_manifest_sha256": sha(build_path), "binary_sha256": build["binary_sha256"],
            "source_archive_sha256": build["source_archive_sha256"],
            "startup_source_inventory_sha256": canonical_sha(sources),
            "unit_tests_passed": build["unit_tests_passed"],
            "current_checker_sha256": sha(__file__), "shared_numpy_checker_sha256": sha(pathlib.Path(__file__).with_name("check_probe.py"))}


def check(directory):
    report_path = directory / "operators.json"
    report = json.loads(report_path.read_text())
    original_hash = sha(report_path)
    proof = validate_build(directory, report)
    arithmetic = validate_arithmetic(report)
    negative = []
    for mutation in ("changed AVX2 output", "changed packed code"):
        corrupted = copy.deepcopy(report)
        if mutation == "changed AVX2 output":
            corrupted["cases"][0]["batches"][0]["avx2"][0] += 1.0
        else:
            corrupted["cases"][0]["packed_codes"][0] ^= 1
        try:
            validate_arithmetic(corrupted)
        except ValueError:
            negative.append(mutation + " rejected")
        else:
            raise AssertionError(mutation + " was accepted")
    require(sha(report_path) == original_hash, "Report changed during independent check")
    require(validate_build(directory, report) == proof, "Preserved evidence changed during check")
    return {"schema": 1, "qualification": QUALIFICATION, "timing": None,
            "report_sha256": original_hash, "provenance": proof, "arithmetic": arithmetic,
            "negative_checks": negative, "python": sys.version, "numpy": np.__version__}


def capture(directory):
    directory.mkdir(parents=True, exist_ok=False)
    start = source_hashes()
    rustc = shutil.which("rustc")
    require(rustc is not None, "rustc required")
    suffix = ".exe" if os.name == "nt" else ""
    binary, tests = directory / ("q4_avx2_probe" + suffix), directory / ("q4_avx2_tests" + suffix)
    env = dict(os.environ, FOCR_AVX2_SOURCE_SHA256=canonical_sha(start))
    source = ROOT / "experiments/quantization/q4_avx2_probe.rs"
    flags = ["--edition", "2024", "-C", "opt-level=3"]
    commands = [[rustc, str(source), *flags, "-o", str(binary)],
                [rustc, str(source), *flags, "--test", "-o", str(tests)]]
    archive_path = directory / "source.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in start:
            archive.write(ROOT / name, name)
    with (directory / "build.log").open("x", encoding="utf-8") as log:
        for command in commands:
            subprocess.run(command, check=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    require(source_hashes() == start, "Source changed during compile")
    executable_hash, tests_hash = sha(binary), sha(tests)
    test_command = [str(tests), "--test-threads", "1"]
    tested = subprocess.run(test_command, check=True, cwd=ROOT, capture_output=True, text=True)
    (directory / "tests.log").write_text(tested.stdout + tested.stderr, encoding="utf-8")
    require(re.search(r"test result: ok\. 9 passed; 0 failed; 0 ignored", tested.stdout), "Expected all nine tests")
    run_command = [str(binary), "--output", str(directory / "operators.json")]
    subprocess.run(run_command, check=True, cwd=ROOT)
    require(sha(binary) == executable_hash and sha(tests) == tests_hash, "Executable changed during execution")
    require(source_hashes() == start, "Source changed during run")
    build = {"schema": 1, "status": "complete", "qualification": QUALIFICATION,
             "source_unchanged_during_build_and_run": True, "source_sha256": start,
             "source_archive": archive_path.name, "source_archive_sha256": sha(archive_path),
             "binary": binary.name, "binary_sha256": executable_hash,
             "test_binary": tests.name, "test_binary_sha256": tests_hash,
             "report": "operators.json", "report_sha256": sha(directory / "operators.json"),
             "test_log": "tests.log", "test_log_sha256": sha(directory / "tests.log"), "unit_tests_passed": 9,
             "build_log_sha256": sha(directory / "build.log"), "build_commands": commands,
             "test_command": test_command, "run_command": run_command,
             "rustc": subprocess.run([rustc, "-vV"], check=True, capture_output=True, text=True).stdout,
             "platform": platform.platform(), "machine": platform.machine(),
             "target_features": "Compiler baseline unchanged; AVX2/FMA target_feature only inside runtime-guarded function. No target-cpu=native.",
             "snapshot_limits": "Source window and executables preserved; not a hermetic compiler/OS attestation. Test harness prints its ordinary elapsed line; it is not operator timing evidence."}
    write_new(directory / "build.json", build)
    return check(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=pathlib.Path, help="New capture directory")
    group.add_argument("--check", type=pathlib.Path, help="Read-only recheck of preserved capture")
    parser.add_argument("--report", type=pathlib.Path, help="New independent check report path")
    args = parser.parse_args()
    if args.output:
        directory = args.output.resolve()
        result = capture(directory)
        write_new(args.report or directory / "independent-check.json", result)
    else:
        result = check(args.check.resolve())
        if args.report:
            write_new(args.report, result)
    print(json.dumps({"passed": result["arithmetic"]["passed"], "cases": result["arithmetic"]["cases"],
                      "outputs_per_backend": result["arithmetic"]["outputs_per_backend"],
                      "unit_tests_passed": result["provenance"]["unit_tests_passed"], "timing": None}))


if __name__ == "__main__":
    main()
