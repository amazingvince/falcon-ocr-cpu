#!/usr/bin/env python3
"""Compare captured Windows/Linux AOCL output digests against one GPU fixture."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(windows, linux):
    reports = {"windows": read(windows), "linux": read(linux)}
    for name, report in reports.items():
        require(report["status"] == "complete" and report["case_count"] == 7 and report["operator_count"] == 28, name + ": incomplete probe")
        require(report["execution"]["os"] == name and report["execution"]["rust_threads"] == 4, name + ": execution contract differs")
    first, second = reports["windows"], reports["linux"]
    for key in ("fixture_sha256", "manifest_sha256", "source_trace_sha256", "weights_sha256", "config_sha256"):
        require(first["reference"][key] == second["reference"][key], "Reference differs: " + key)
    for key in ("source_sha256", "ffi_source_sha256", "rust_kernels_source_sha256", "cargo_lock_sha256", "cargo_manifest_sha256"):
        require(first["probe"][key] == second["probe"][key], "Probe compiled source differs: " + key)
    for key in ("revision", "gemm_header_sha256", "base_types_header_sha256"):
        require(first["aocl_source"][key] == second["aocl_source"][key], "AOCL source ABI differs: " + key)
    require(first["known_answers"] == second["known_answers"], "Known-answer results differ")
    require(first["output_digest_encoding"] == second["output_digest_encoding"], "Output digest encoding differs")
    by_key = {name: {(row["case"], row["operator"]): row for row in report["operators"]} for name, report in reports.items()}
    require(len(by_key["windows"]) == 28 and set(by_key["windows"]) == set(by_key["linux"]), "Missing, duplicate or unexpected operator keys")
    rows = []
    for case, operator in sorted(by_key["windows"]):
        left, right = by_key["windows"][(case, operator)], by_key["linux"][(case, operator)]
        for key in ("shape", "layer", "input_key", "weight_key", "gpu_expected_key"):
            require(left[key] == right[key], f"{case}/{operator}: {key} differs")
        require(left["output_sha256"]["gpu"] == right["output_sha256"]["gpu"], f"{case}/{operator}: GPU array differs")
        rows.append({"case": case, "operator": operator, "shape": left["shape"],
                     "aocl_arrays_bit_exact": left["output_sha256"]["aocl"] == right["output_sha256"]["aocl"],
                     "rust_arrays_bit_exact": left["output_sha256"]["rust"] == right["output_sha256"]["rust"],
                     "output_sha256": {"windows": left["output_sha256"], "linux": right["output_sha256"]},
                     "aocl_vs_gpu": {"windows": left["aocl_vs_gpu"], "linux": right["aocl_vs_gpu"]},
                     "rust_vs_gpu": {"windows": left["rust_vs_gpu"], "linux": right["rust_vs_gpu"]}})
    counts = {}
    for name, report in reports.items():
        counts[name] = {}
        for metric in ("max_abs", "rms_abs"):
            pairs = [(row["aocl_vs_gpu"][metric], row["rust_vs_gpu"][metric]) for row in report["operators"]]
            counts[name][metric] = {"aocl_lower": sum(a < b for a, b in pairs), "equal": sum(a == b for a, b in pairs), "aocl_higher": sum(a > b for a, b in pairs)}
    return {"schema_version": 1, "status": "complete", "scope": "Cross-platform functionality on 28 identical FP32 operator inputs; no performance or full-model parity claim.",
            "report_paths": {"windows": str(windows), "linux": str(linux)}, "report_sha256": {"windows": sha(windows), "linux": sha(linux)},
            "libraries": {name: value["library"] for name, value in reports.items()},
            "reference": first["reference"], "operator_count": len(rows),
            "aocl_bit_exact_operators": sum(row["aocl_arrays_bit_exact"] for row in rows),
            "rust_bit_exact_operators": sum(row["rust_arrays_bit_exact"] for row in rows),
            "aocl_vs_rust_error_to_same_gpu": counts, "operators": rows,
            "comparison_script_sha256": sha(Path(__file__)),
            "limitations": ["Array SHA-256 equality tests exact FP32 byte identity; unequal hashes alone do not quantify cross-platform error magnitude. Each platform's full maximum/RMS errors to the identical GPU fixture are retained.", "Windows and Linux compiler/ABI/library builds may choose different reductions or kernels.", "Linux ran under WSL with concurrent functional jobs; neither platform run records throughput or latency."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing report")
    result = compare(args.windows, args.linux)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "operator_count", "aocl_bit_exact_operators", "rust_bit_exact_operators")}))


if __name__ == "__main__":
    main()
