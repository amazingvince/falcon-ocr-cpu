"""Full-channel, single-thread Q4 arithmetic checks on pinned real projection operands."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from check_probe import (Tensors, array_sha, difference, oracle, quantize, require, sha,
                         validate_fixed_inputs, validate_preserved_build, samples)

ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE_NAMES = ["experiments/quantization/" + name for name in
                ("q4_reference.rs", "q4_avx2.rs", "q4_avx2_probe.rs", "q4_real_avx2_probe.rs",
                 "check_probe.py", "capture_real_avx2_probe.py", "test_real_avx2_probe.py")] + ["rust-toolchain.toml"]
PRIOR = "artifacts/quantization/operator-probe-v2/operators.json"
PRIOR_SHA = "6b65d9b377ee310a30e43909ca326e975bab310f1960f4dd3381abc91987f18a"
CASES = ["prefill.layer.9", "prefill.layer.12", "prefill.layer.13", "prefill.layer.17", "prefill.layer.18",
         "decode.2.layer.21", "decode.6.layer.19"]
MAPPINGS = [("qkv", "attention.wqkv.weight", "attention_norm.expected", "qkv.expected", 2048, 768),
            ("wo", "attention.wo.weight", "attention.scaled", "attention_projection.expected", 768, 1024),
            ("w13", "feed_forward.w13.weight", "ffn_norm.expected", "w13.expected", 4608, 768),
            ("w2", "feed_forward.w2.weight", "gate.expected", "w2.expected", 768, 2304)]
QUALIFICATION = "Isolated real-weight/equal-input arithmetic only; saved synthetic diagnostic activations are not calibration or held-out quality data. No model integration, group/layer selection, OCR quality, GPU parity, or timing qualification."


def load(path):
    return json.loads(pathlib.Path(path).read_bytes())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_new(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_bytes(path, data):
    with pathlib.Path(path).open("xb") as f:
        f.write(data)


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_NAMES}


def fixed_inputs():
    require(sha(ROOT / PRIOR) == PRIOR_SHA, "Pinned earlier real-weight report changed")
    prior = load(ROOT / PRIOR)
    validate_fixed_inputs(prior)
    proof = validate_preserved_build(prior, ROOT / PRIOR)
    model, fixture = Tensors(ROOT / prior["checkpoint"]), Tensors(ROOT / prior["fixture"])
    sidecar = load(ROOT / prior["fixture_sidecar"])
    require([c["name"] for c in sidecar["cases"]] == CASES, "Exact seven-case fixture inventory")
    require(prior["original_trace_sha256"] == sidecar["source_trace_sha256"], "Original GPU trace binding")
    return prior, proof, model, fixture, sidecar


def row_indices(source_rows, rows):
    require(rows in (1, 2, 4, 8) and rows <= source_rows, "Insufficient real source rows")
    return [0] if rows == 1 else [i * (source_rows - 1) // (rows - 1) for i in range(rows)]


def descriptors(prior, model, fixture, sidecar):
    result = []
    for case in sidecar["cases"]:
        for op, weight_suffix, input_suffix, expected_suffix, n, k in MAPPINGS:
            index = len(result)
            previous = prior["operators"][index]
            weight_key = case["weight_prefix"] + weight_suffix
            input_key, expected_key = case["name"] + "." + input_suffix, case["name"] + "." + expected_suffix
            weights, inputs, expected = model.get(weight_key), fixture.get(input_key), fixture.get(expected_key)
            rows = case["rows"]
            require(tuple(weights.shape) == (n, k) and inputs.size == rows * k and expected.size == rows * n, "Real tensor shapes")
            require(rows == (144 if case["name"].startswith("prefill.") else 1), "Real fixture row count")
            require(np.isfinite(weights).all() and np.isfinite(inputs).all() and np.isfinite(expected).all(), "Finite real operands")
            item = {"index": index, "case": case["name"], "operator": op, "n": n, "k": k, "source_rows": rows,
                    "weight_key": weight_key, "input_key": input_key, "gpu_expected_key": expected_key,
                    "weight_f32le_sha256": array_sha(weights), "full_input_f32le_sha256": array_sha(inputs),
                    "full_gpu_expected_f32le_sha256": array_sha(expected)}
            require(previous["case"] == item["case"] and previous["operator"] == op
                    and previous["weight_key"] == weight_key and previous["input_key"] == input_key,
                    "Earlier operator mapping differs")
            for field, old in (("weight_f32le_sha256", "weight_f32le_sha256"),
                               ("full_input_f32le_sha256", "input_f32le_sha256"),
                               ("full_gpu_expected_f32le_sha256", "gpu_expected_f32le_sha256")):
                require(item[field] == previous[old], "Earlier operand identity differs: " + field)
            item["formats"] = [{"group_size": g, **next(v["storage"] for v in previous["variants"] if v["bits"] == 4 and v["group_size"] == g),
                                "reconstructed_f32le_sha256": next(v["reconstructed_f32le_sha256"] for v in previous["variants"] if v["bits"] == 4 and v["group_size"] == g)} for g in (64, 128)]
            item["omitted_batch_sizes"] = [b for b in (1, 2, 4, 8) if b > rows]
            item["omission_reason"] = "Larger batches lack enough real source rows; no row replication" if rows == 1 else None
            result.append(item)
    require(len(result) == 28, "All actual projection cases")
    return result


def prepare_plan(directory, inputs):
    prior, proof, model, fixture, sidecar = inputs
    operators = descriptors(prior, model, fixture, sidecar)
    (directory / "operands").mkdir()
    for item in operators:
        index, k = item["index"], item["k"]
        weights = model.get(item["weight_key"])
        item["weight_file"] = f"operands/{index:03}.weights.f32"
        write_bytes(directory / item["weight_file"], np.ascontiguousarray(weights).tobytes())
        original = fixture.get(item["input_key"]).reshape(item["source_rows"], k)
        item["batches"] = []
        for rows in (1, 2, 4, 8):
            if rows > item["source_rows"]:
                continue
            indices = row_indices(item["source_rows"], rows)
            selected = np.ascontiguousarray(original[indices])
            path = f"operands/{index:03}.b{rows}.input.f32"
            write_bytes(directory / path, selected.tobytes())
            item["batches"].append({"rows": rows, "source_row_indices": indices, "input_file": path,
                                    "input_f32le_sha256": array_sha(selected),
                                    "distinct_input_rows_by_bytes": len({row.tobytes() for row in selected})})
    plan = {"schema": 1, "qualification": QUALIFICATION, "threads": 1, "timing": None,
            "row_selection": "B=1 selects index0; otherwise floor(i*(source_rows-1)/(B-1)); distinct original positions, never replicated within a batch.",
            "output_channel_sampling": False, "groups": [64, 128], "operators": operators,
            "checkpoint": prior["checkpoint"], "checkpoint_sha256": prior["checkpoint_sha256"],
            "fixture": prior["fixture"], "fixture_sha256": prior["fixture_sha256"],
            "fixture_sidecar": prior["fixture_sidecar"], "fixture_sidecar_sha256": prior["fixture_sidecar_sha256"],
            "original_trace_sha256": prior["original_trace_sha256"],
            "prior_report": PRIOR, "prior_report_sha256": PRIOR_SHA, "prior_build_proof": proof}
    write_new(directory / "plan.json", plan)
    lines = ["FOCR_Q4_REAL_V1\t" + sha(directory / "plan.json")]
    for item in operators:
        batches = ";".join(str(b["rows"]) + ":" + b["input_file"] for b in item["batches"])
        lines.append("\t".join(map(str, [item["index"], item["n"], item["k"], item["source_rows"], item["weight_file"], batches])))
    write_bytes(directory / "jobs.tsv", ("\n".join(lines) + "\n").encode())
    return plan


def checked_array(path, shape, dtype="<f4"):
    array = np.fromfile(path, dtype=dtype)
    require(array.size == int(np.prod(shape)), "Captured array shape: " + str(path))
    return array.reshape(shape)


def validate_capture(directory):
    build = load(directory / "build.json")
    require(build.get("status") == "complete" and build.get("source_unchanged_during_build_and_run") is True, "Successful captured source window")
    require(set(build["source_sha256"]) == set(SOURCE_NAMES), "Exact captured source inventory")
    for field in ("binary", "test_binary"):
        require(pathlib.Path(build[field]).name == build[field], "Preserved executable basename")
    require({"source.zip", "plan.json", "jobs.tsv", "outputs/operators.json", "tests.log", "build.log", "probe.log",
             build["binary"], build["test_binary"]} <= set(build["artifact_sha256"]), "Incomplete captured artifact inventory")
    for name, digest in build["artifact_sha256"].items():
        require(not pathlib.Path(name).is_absolute() and ".." not in pathlib.Path(name).parts, "Artifact path outside capture")
        require(sha(directory / name) == digest, "Preserved capture changed: " + name)
    with zipfile.ZipFile(directory / "source.zip") as source:
        require(len(source.namelist()) == len(build["source_sha256"]) and set(source.namelist()) == set(build["source_sha256"]), "Source ZIP inventory")
        for name, digest in build["source_sha256"].items():
            require(hashlib.sha256(source.read(name)).hexdigest() == digest, "Source ZIP bytes")
    report = load(directory / "outputs/operators.json")
    require(report["compiled_source_inventory_sha256"] == canonical_sha(build["source_sha256"]), "Compiled source binding")
    require(report["compiled_plan_sha256"] == sha(directory / "plan.json"), "Compiled plan binding")
    require(report["timing"] is None and report["threads"] == 1 and report["avx2_fma_available"] is True
            and report["output_channel_sampling"] is False, "Scope/backend changed")
    require(build["unit_tests_passed"] == 9, "Existing nine arithmetic tests must pass")
    return build, report, load(directory / "plan.json")


def fp64_dots(input_, weight):
    x, w = input_.astype(np.float64), weight.astype(np.float64)
    return x @ w.T, np.abs(x) @ np.abs(w).T


def arithmetic_metrics(candidate, reference, absolute, k):
    require(candidate.shape == reference.shape and np.isfinite(candidate).all(), "Finite full candidate output")
    gamma32 = (k * 2.0**-24) / (1 - k * 2.0**-24)
    gamma64 = (k * 2.0**-53) / (1 - k * 2.0**-53)
    error = np.abs(candidate.astype(np.float64) - reference)
    # A sufficient condition for the UNCHANGED gamma32*S_exact gate, accounting
    # conservatively for both F64 dot and F64 sum(abs) rounding. It tightens the
    # check; it does not add an arbitrary tolerance to the FP32 bound.
    upper_error = error + gamma64 * absolute / (1 - gamma64)
    lower_bound = gamma32 * absolute / (1 + gamma64)
    require(np.all(upper_error <= lower_bound), "Full-output frozen gamma_K arithmetic gate")
    ratio = np.divide(upper_error, lower_bound, out=np.zeros_like(error), where=lower_bound != 0)
    return {"error_vs_fp64_dequantized": difference(candidate, reference),
            "max_fraction_of_frozen_bound_including_fp64_uncertainty": float(ratio.max()),
            "bound_violations": 0, "output_f32le_sha256": array_sha(candidate)}


def check(directory):
    build, report, plan = validate_capture(directory)
    build_hash = sha(directory / "build.json")
    prior, proof, model, fixture, sidecar = fixed_inputs()
    expected = descriptors(prior, model, fixture, sidecar)
    require(plan["prior_report_sha256"] == PRIOR_SHA and plan["prior_build_proof"] == proof, "Earlier real-weight provenance changed")
    for field in ("checkpoint", "checkpoint_sha256", "fixture", "fixture_sha256", "fixture_sidecar", "fixture_sidecar_sha256", "original_trace_sha256"):
        require(plan[field] == prior[field], "Plan input identity differs: " + field)
    require(plan["groups"] == [64, 128] and plan["threads"] == 1 and plan["timing"] is None
            and plan["output_channel_sampling"] is False, "Frozen real-operator scope differs")
    require(len(plan["operators"]) == len(expected) == report["operators"] == 28, "Operator inventory")
    details, expected_files = [], {"outputs/operators.json"}
    sampled_fsum_elements = 0
    for item, reference_item in zip(plan["operators"], expected):
        require({k: item[k] for k in reference_item} == reference_item, "Pinned operator descriptor differs")
        n, k, index = item["n"], item["k"], item["index"]
        weights = model.get(item["weight_key"])
        inputs = fixture.get(item["input_key"]).reshape(item["source_rows"], k)
        gpu = fixture.get(item["gpu_expected_key"]).reshape(item["source_rows"], n)
        require(item["weight_file"] in build["artifact_sha256"] and all(b["input_file"] in build["artifact_sha256"] for b in item["batches"]), "Operand omitted from final captured source window")
        require(sha(directory / item["weight_file"]) == item["weight_f32le_sha256"], "Preserved real weight bytes")
        require([b["rows"] for b in item["batches"]] == [b for b in (1, 2, 4, 8) if b <= item["source_rows"]], "Real batch inventory")
        for fmt in item["formats"]:
            group = fmt["group_size"]
            packed, scales, restored = quantize(weights, 4, group)
            prefix = f"outputs/{index:03}.g{group}"
            code_file, scale_file = prefix + ".codes.bin", prefix + ".scales.f32"
            expected_files.update([code_file, scale_file])
            require(sha(directory / code_file) == array_sha(packed) == fmt["codes_sha256"], "Full real Q4 code bytes")
            require(sha(directory / scale_file) == array_sha(scales) == fmt["scales_f32le_sha256"], "Full real Q4 scale bytes")
            require(array_sha(restored) == fmt["reconstructed_f32le_sha256"], "Full reconstructed matrix bytes")
            for batch in item["batches"]:
                rows = batch["rows"]
                indices = row_indices(item["source_rows"], rows)
                require(batch["source_row_indices"] == indices and len(set(indices)) == rows, "No fabricated source row diversity")
                x = np.ascontiguousarray(inputs[indices])
                require(sha(directory / batch["input_file"]) == array_sha(x) == batch["input_f32le_sha256"], "Actual selected input bytes")
                require(batch["distinct_input_rows_by_bytes"] == len({row.tobytes() for row in x}), "Observed row diversity metadata")
                restored64, absolute = fp64_dots(x, restored)
                original64, _ = fp64_dots(x, weights)
                _, channels = samples(rows, n, item["operator"] == "qkv")
                # Sparse math.fsum cross-checks of the full F64 oracle, fixed by
                # output-channel geometry; these do not sample candidate gates.
                sampled, sampled_absolute = oracle(x, restored, range(rows), channels)
                sampled_fsum_elements += len(sampled)
                gamma64 = (k * 2.0**-53) / (1 - k * 2.0**-53)
                require(np.all(np.abs(restored64[:, channels].reshape(-1) - sampled) <= gamma64 * sampled_absolute), "F64 GEMM differs from independent math.fsum beyond F64 bound")
                metrics, arrays = {}, {}
                for backend in ("scalar", "avx2"):
                    output_file = f"{prefix}.b{rows}.{backend}.f32"
                    expected_files.add(output_file)
                    arrays[backend] = checked_array(directory / output_file, (rows, n))
                    metrics[backend] = {**arithmetic_metrics(arrays[backend], restored64, absolute, k),
                                        "total_error_vs_fp64_original": difference(arrays[backend], original64),
                                        "diagnostic_error_vs_saved_gpu_fp32": difference(arrays[backend], gpu[indices])}
                details.append({"case": item["case"], "operator": item["operator"], "n": n, "k": k,
                                "group_size": group, "rows": rows, "source_row_indices": indices,
                                "outputs_checked_per_backend": rows * n, "quantization_only_fp64": difference(restored64, original64),
                                **metrics, "avx2_vs_scalar": difference(arrays["avx2"], arrays["scalar"])})
        print("checked " + item["case"] + "." + item["operator"], flush=True)
    actual_files = {p.relative_to(directory).as_posix() for p in (directory / "outputs").iterdir() if p.is_file()}
    require(actual_files == expected_files, "Missing/extra captured output/bitstream files")
    require(expected_files <= set(build["artifact_sha256"]), "Output omitted from final captured source window")
    outputs = sum(d["outputs_checked_per_backend"] for d in details)
    require(len(details) == report["batch_cases"] == 176 and outputs == report["elements_per_backend"] == 1261568, "Full selected-output coverage")
    validate_capture(directory)
    validate_fixed_inputs(prior)
    require(sha(directory / "build.json") == build_hash, "Build identity changed during check")
    return {"schema": 1, "status": "independent_full_output_arithmetic_checks_passed", "qualification": QUALIFICATION,
            "threads": 1, "timing": None, "model_integration": False, "quality_qualification": False,
            "build_manifest_sha256": build_hash, "plan_sha256": sha(directory / "plan.json"),
            "source_archive_sha256": build["artifact_sha256"]["source.zip"], "binary_sha256": build["artifact_sha256"][build["binary"]],
            "current_checker_sha256": sha(__file__), "shared_independent_checker_sha256": sha(pathlib.Path(__file__).with_name("check_probe.py")),
            "python": sys.version, "numpy": np.__version__, "operator_cases": 28, "matrix_group_cases": 56,
            "operator_batch_cases": len(details), "outputs_checked_per_backend": outputs, "output_channel_sampling": False,
            "math_fsum_cross_checked_outputs": sampled_fsum_elements,
            "arithmetic_bound": "Unchanged gamma_K*sum(abs(FP32 activation*FP32 reconstructed weight)); gamma_K=K*2^-24/(1-K*2^-24). Conservative FP64 oracle uncertainty tightens, never widens, the gate.",
            "row_scope": "All channels of predetermined 1/2/4/8 subsets of real prefill rows; one real row only for saved decode cases. Not all144 prefill rows, not a joint model decode.",
            "details": details}


def capture(directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = source_hashes()
    inputs = fixed_inputs()
    prepare_plan(directory, inputs)
    with zipfile.ZipFile(directory / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in start:
            archive.write(ROOT / name, name)
    suffix = ".exe" if os.name == "nt" else ""
    rustc = shutil.which("rustc")
    require(rustc is not None, "Rust compiler required")
    binary, tests = directory / ("q4_real_avx2_probe" + suffix), directory / ("q4_existing_tests" + suffix)
    env = dict(os.environ, FOCR_REAL_Q4_SOURCE_SHA256=canonical_sha(start), FOCR_REAL_Q4_PLAN_SHA256=sha(directory / "plan.json"))
    commands = [[rustc, str(ROOT / "research/quantization-feasibility/experiments/quantization/q4_real_avx2_probe.rs"), "--edition", "2024", "-C", "opt-level=3", "-o", str(binary)],
                [rustc, str(ROOT / "research/quantization-feasibility/experiments/quantization/q4_avx2_probe.rs"), "--edition", "2024", "-C", "opt-level=3", "--test", "-o", str(tests)]]
    with (directory / "build.log").open("x", encoding="utf8") as log:
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    require(source_hashes() == start, "Source changed during build")
    identities = {p.name: sha(p) for p in (binary, tests)}
    test_command = [str(tests), "--test-threads", "1"]
    tested = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True, check=True)
    require("test result: ok. 9 passed; 0 failed; 0 ignored" in tested.stdout, "All existing nine tests must pass")
    write_bytes(directory / "tests.log", (tested.stdout + tested.stderr).encode())
    run_command = [str(binary), "--plan", str(directory / "jobs.tsv"), "--output", str(directory / "outputs")]
    with (directory / "probe.log").open("x", encoding="utf8") as log:
        subprocess.run(run_command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    require(source_hashes() == start and all(sha(directory / p) == digest for p, digest in identities.items()), "Sources/binaries changed during run")
    artifacts = {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob("*") if p.is_file()}
    build = {"schema": 1, "status": "complete", "started_utc": started, "source_unchanged_during_build_and_run": True,
             "source_sha256": start, "artifact_sha256": artifacts, "binary": binary.name, "test_binary": tests.name,
             "build_commands": commands, "run_command": run_command, "test_command": test_command, "unit_tests_passed": 9,
             "rustc": subprocess.check_output([rustc, "-vV"], text=True), "platform": platform.platform(),
             "environment": {k: env[k] for k in ("FOCR_REAL_Q4_SOURCE_SHA256", "FOCR_REAL_Q4_PLAN_SHA256", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
             "qualification": QUALIFICATION, "timing": None, "limitations": "Non-hermetic std-only rustc capture. Compiler baseline unchanged; AVX2/FMA only in existing runtime-guarded kernel. Test-harness elapsed text is not operator timing evidence."}
    write_new(directory / "build.json", build)
    result = check(directory)
    require(source_hashes() == start, "Sources changed during independent check")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=pathlib.Path)
    mode.add_argument("--check", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    directory = (args.output or args.check).resolve()
    result = capture(directory) if args.output else check(directory)
    write_new(args.report or directory / "independent-check.json", result)
    print(json.dumps({k: result[k] for k in ("status", "operator_batch_cases", "outputs_checked_per_backend", "timing")}))


if __name__ == "__main__":
    main()
