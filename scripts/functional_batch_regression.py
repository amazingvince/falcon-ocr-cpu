#!/usr/bin/env python3
"""Freeze and run exact-output batch/layout checks; planning never runs inference."""
import argparse
import ctypes
import datetime
import hashlib
import json
import math
import pathlib
import platform
import subprocess
import sys
import zipfile

from PIL import Image

import capture_functional_cli as capture
from corpus_comparison import ASSETS, PROMPT, GREEDY, exact_output, validate_cpu_result
from fetch_reference import REVISION, WEIGHT_SHA256
from validate_text_replay import canonical_sha256

ROOT = capture.ROOT
MEMORY_FLOOR = 12 * 1024**3
WORKLOAD_CANONICAL_SHA256 = "1950b969d19f69eb987ac53118fcb2d9ba47ab4cb268e1df13f17c594a6a79b9"
MODES = [
    {"id": "joint-expanded", "cache_layout": "expanded", "weight_layout": "unpacked"},
    {"id": "joint-compact", "cache_layout": "compact", "weight_layout": "unpacked"},
    {"id": "joint-phase-packed-expanded", "cache_layout": "expanded", "weight_layout": "phase-packed"},
    {"id": "joint-phase-packed-compact", "cache_layout": "compact", "weight_layout": "phase-packed"},
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_bytes())


def write_new(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify(path, digest):
    require(sha(path) == digest, "Changed file: " + str(path))


def available_physical_memory():
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in ["total", "available", "total_pagefile", "available_pagefile", "total_virtual", "available_virtual", "available_extended"]]
        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        function = ctypes.windll.kernel32.GlobalMemoryStatusEx
        function.argtypes = [ctypes.POINTER(MemoryStatus)]
        function.restype = ctypes.c_int
        require(bool(function(ctypes.byref(state))), "GlobalMemoryStatusEx failed")
        return {"available_bytes": state.available, "total_bytes": state.total, "source": "GlobalMemoryStatusEx"}
    if sys.platform.startswith("linux"):
        fields = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in pathlib.Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemAvailable:", "MemTotal:"))}
        return {"available_bytes": fields["MemAvailable"], "total_bytes": fields["MemTotal"], "source": "/proc/meminfo MemAvailable (guest budget under WSL)"}
    raise ValueError("No checked available-physical-memory provider on this platform")


def validate_manifest(path):
    workload = read(path)
    require(canonical_sha256(workload) == WORKLOAD_CANONICAL_SHA256, "Frozen v1 workload selection changed")
    runtime = workload["runtime"]
    for key, expected in {"precision": "fp32", "backend": "avx2", "threads": 4, "min_dimension": 64,
                          "max_dimension": 1536, "max_new_tokens": 4096}.items():
        require(runtime.get(key) == expected, "Frozen functional runtime differs: " + key)
    require(runtime.get("minimum_available_physical_bytes", MEMORY_FLOOR) >= MEMORY_FLOOR, "Memory floor is below 12 GiB")
    for source, digest in workload["source_locks"].items():
        verify(ROOT / source, digest)
    if "selection_manifest" in workload:
        verify(ROOT / workload["selection_manifest"]["path"], workload["selection_manifest"]["sha256"])
    for name, item in workload["inputs"].items():
        require(item["source_lock"] in workload["source_locks"], name + ": unbound source lock")
        pages = read(ROOT / item["source_lock"])["pages"]
        matches = [page for page in pages if page["id"] == item["id"]]
        require(len(matches) == 1 and canonical_sha256(matches[0]) == item["source_page_sha256"], name + ": source membership changed")
        for key in ["source_path", "source_sha256", "canonical_path", "canonical_png_sha256", "rgb_sha256", "width", "height", "category"]:
            require(matches[0][key] == item[key], name + ": changed source page field " + key)
        verify(ROOT / item["source_path"], item["source_sha256"])
        verify(ROOT / item["canonical_path"], item["canonical_png_sha256"])
        with Image.open(ROOT / item["canonical_path"]) as image:
            require(image.mode == "RGB" and image.format == "PNG", name + ": expected RGB PNG")
            require(list(image.size) == [item["width"], item["height"]], name + ": original dimensions")
            require(hashlib.sha256(image.tobytes()).hexdigest() == item["rgb_sha256"], name + ": pixels changed")
        width, height = item["prepared_dimensions_expected"]
        require(all(type(x) is int and 0 < x <= 1536 and x % 16 == 0 for x in [width, height]), name + ": expected prepared dimensions")
        require(item["input_tokens_expected"] == width // 16 * (height // 16) + 16, name + ": expected prefix")
        require(item["input_tokens_expected"] + 4096 <= 16384, name + ": context budget")
    for name, order in workload["cases"].items():
        require(len(order) in [2, 4, 8] and all(key in workload["inputs"] for key in order), "Invalid case order: " + name)
    model = workload["model"]
    require(model["revision"] == REVISION, "Model revision differs")
    expected_assets = {**ASSETS, "model.safetensors": WEIGHT_SHA256}
    require(set(model["assets"]) == set(expected_assets), "Model asset inventory differs")
    for name, expected in expected_assets.items():
        item = model["assets"][name]
        require(item["sha256"] == expected, "Unpinned model asset: " + name)
        asset = ROOT / model["directory"] / name
        require(asset.stat().st_size == item["bytes"], "Model asset size differs: " + name)
        verify(asset, expected)
    return workload


def validate_build(path, current=True):
    path = pathlib.Path(path).resolve()
    build = read(path)
    require(build.get("target_kind") == "bin" and build.get("target_name") == "falcon-ocr" and build.get("status") == "complete", "Need a complete functional CLI capture")
    require(build.get("source_unchanged_during_build") is True and build.get("build_exit_code") == 0, "Build failed or source changed")
    require(build.get("cargo_emitted_executable") and "--locked" in build["command"] and "--release" in build["command"], "Missing locked release/Cargo executable proof")
    binary, archive = path.parent / build["binary"], path.parent / build["source_archive"]
    require(binary.parent.resolve() == path.parent and archive.parent.resolve() == path.parent, "Binary/archive must be in preserved build directory")
    verify(binary, build["binary_sha256"])
    verify(archive, build["source_archive_sha256"])
    require(binary.stat().st_size == build["binary_bytes"], "Binary size differs")
    hashes = build["source_sha256"]
    require({p.relative_to(ROOT).as_posix() for p in capture.source_paths()} <= hashes.keys(), "Source inventory incomplete")
    with zipfile.ZipFile(archive) as source:
        require(len(source.namelist()) == len(hashes) and set(source.namelist()) == set(hashes), "Source archive/map differs")
        for name, digest in hashes.items():
            require(hashlib.sha256(source.read(name)).hexdigest() == digest, "Archived source differs: " + name)
            if current:
                verify(ROOT / name, digest)
    return build, binary


def command(binary, workload, job):
    runtime = workload["runtime"]
    mode = job["mode"]
    return [str(binary), "--model", str(ROOT / workload["model"]["directory"]), "--mode", "exact",
            "--threads", str(runtime["threads"]), "--backend", "avx2", "--batch-size", str(job["batch_size"]),
            "--cache-layout", mode["cache_layout"], "--weight-layout", mode["weight_layout"], "run",
            "--min-dimension", "64", "--max-dimension", "1536", "--max-new-tokens", "4096"] + [
            str(ROOT / workload["inputs"][key]["canonical_path"]) for key in job["request_keys"]]


def jobs_for(workload, case):
    require(case in workload["cases"], "Unknown frozen case")
    order = workload["cases"][case]
    # One same-binary sequential control per distinct input; never reuse a cap512
    # result or a different inference build as this control.
    jobs = [{"id": "sequential-control", "batch_size": 1, "request_keys": list(dict.fromkeys(order)),
             "mode": {"cache_layout": "expanded", "weight_layout": "unpacked"}}]
    jobs += [{"id": mode["id"], "batch_size": len(order), "request_keys": list(order), "mode": mode} for mode in MODES]
    return jobs


def validate_results(results, workload, job):
    require(isinstance(results, list) and len(results) == len(job["request_keys"]), "Missing/extra request outputs")
    config = {"precision": "fp32", "options": {k: workload["runtime"][k] for k in ["min_dimension", "max_dimension", "max_new_tokens"]}}
    def check(name, actual, expected):
        require(actual == expected, "Invalid saved output: " + name)
    for index, (result, key) in enumerate(zip(results, job["request_keys"])):
        require(isinstance(result, dict) and not any(marker in result for marker in ["postprocessing_replay", "derived_text_replay"])
                and result.get("inference_reexecuted") is not False, "Replayed output is not fresh CLI inference")
        require(validate_cpu_result(result, config, check, str(index)), "Malformed inference result")
        item = workload["inputs"][key]
        require([result["width"], result["height"]] == item["prepared_dimensions_expected"], key + ": prepared dimensions differ")
        require(result["input_tokens"] == item["input_tokens_expected"], key + ": prefix differs")
        require(result.get("backend") in ("avx2", "rust-gemm/avx2"), key + ": wrong backend")
        require(result.get("cache_layout") == job["mode"]["cache_layout"], key + ": wrong cache layout")
        require(result.get("weight_layout") == job["mode"]["weight_layout"].replace("-", "_"), key + ": wrong weight layout")
        packed = result.get("packed_weight_bytes")
        require(type(packed) is int and (packed > 0 if job["mode"]["weight_layout"] == "phase-packed" else packed == 0), key + ": packed weight payload differs")
        milliseconds = result.get("weight_packing_ms")
        require(type(milliseconds) in [int, float] and math.isfinite(milliseconds) and milliseconds >= 0, key + ": malformed packing metadata")


def compare_results(control, results, request_keys):
    require(len(results) == len(request_keys), "Missing/extra candidate requests")
    return [{"request_index": i, "input_key": key, **exact_output(control[key], result),
             "exact": all(exact_output(control[key], result).values()), "output_tokens": result["output_tokens"],
             "finish_reason": result["finish_reason"], "input_tokens": result["input_tokens"]}
            for i, (key, result) in enumerate(zip(request_keys, results))]


def plan(manifest_path, build_path, case, output):
    workload = validate_manifest(manifest_path)
    build, binary = validate_build(build_path)
    jobs = jobs_for(workload, case)
    for job in jobs:
        job["command"] = command(binary, workload, job)
    output = pathlib.Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "workload.json", workload)
    record = {"schema_version": 1, "kind": "functional-batch-regression-v1", "created_utc": utc(), "case": case,
              "workload_source": str(pathlib.Path(manifest_path).resolve()), "workload_source_sha256": sha(manifest_path),
              "workload_snapshot": "workload.json", "workload_snapshot_sha256": sha(output / "workload.json"),
              "build_manifest": str(pathlib.Path(build_path).resolve()), "build_manifest_sha256": sha(build_path),
              "binary": str(binary), "binary_sha256": build["binary_sha256"], "platform": platform.platform(),
              "prompt": PROMPT, "greedy_policy": GREEDY, "teacher_forced": False,
              "minimum_available_physical_bytes": workload["runtime"].get("minimum_available_physical_bytes", MEMORY_FLOOR),
              "jobs": jobs, "qualification": "Functional stress only: once-per-layout free greedy output comparison to fresh same-binary sequential controls. No quality sample, performance gate or default promotion."}
    write_new(output / "plan.json", record)
    return output / "plan.json"


def load_plan(path):
    path = pathlib.Path(path).resolve()
    record = read(path)
    require(record.get("kind") == "functional-batch-regression-v1", "Wrong plan kind")
    require(record.get("platform") == platform.platform(), "Execution platform differs from planning platform")
    verify(record["workload_source"], record["workload_source_sha256"])
    workload_path = path.parent / record["workload_snapshot"]
    require(workload_path.parent.resolve() == path.parent, "Workload snapshot must stay in plan directory")
    verify(workload_path, record["workload_snapshot_sha256"])
    workload = validate_manifest(workload_path)
    require(workload == read(record["workload_source"]), "Frozen workload snapshot differs")
    verify(record["build_manifest"], record["build_manifest_sha256"])
    build, binary = validate_build(record["build_manifest"])
    require(str(binary) == record["binary"] and build["binary_sha256"] == record["binary_sha256"], "Binary identity differs")
    expected = jobs_for(workload, record["case"])
    for job in expected:
        job["command"] = command(binary, workload, job)
    require(expected == record["jobs"], "Plan job order/options changed")
    require(record["minimum_available_physical_bytes"] >= MEMORY_FLOOR, "Memory floor weakened")
    require(record["prompt"] == PROMPT and record["greedy_policy"] == GREEDY and record["teacher_forced"] is False, "Plan prompt/greedy contract changed")
    return record, workload


def execution_identity(binary):
    return {"platform": platform.platform(), "sys_platform": sys.platform, "python_version": sys.version,
            "python_executable": str(pathlib.Path(sys.executable).resolve()), "python_executable_sha256": sha(sys.executable),
            "cli_executable": str(pathlib.Path(binary).resolve()), "cli_executable_sha256": sha(binary)}


def run(path):
    path = pathlib.Path(path).resolve()
    plan_hash = sha(path)
    record, workload = load_plan(path)
    output = path.parent
    identity = execution_identity(record["binary"])
    write_new(output / "execution.json", {"plan_sha256": plan_hash, "started_utc": utc(), "identity": identity,
                                         "qualification": record["qualification"]})
    try:
        report = run_jobs(path, plan_hash, record, workload, identity)
    except BaseException as error:
        try:
            end_identity = execution_identity(record["binary"])
        except Exception as identity_error:
            end_identity = {"identity_error": str(identity_error)}
        write_new(output / "execution-final.json", {"plan_sha256": plan_hash, "finished_utc": utc(),
                  "status": "failed_or_interrupted", "functional_gate_passed": False, "identity_at_end": end_identity,
                  "error_type": type(error).__name__, "error": str(error),
                  "retry_policy": "No automatic retry, dimensions/cap reduction, or reuse. Preserve this directory and create a fresh plan."})
        raise
    write_new(output / "execution-final.json", {"plan_sha256": plan_hash, "finished_utc": utc(), "status": "passed",
              "functional_gate_passed": True, "identity_at_end": execution_identity(record["binary"]),
              "report_sha256": sha(output / "report.json")})
    return report


def run_jobs(path, plan_hash, record, workload, identity):
    output = path.parent
    artifact_hashes = {"execution.json": sha(output / "execution.json")}
    control, comparisons, completed = {}, [], []
    for job in record["jobs"]:
        verify(path, plan_hash)
        memory = available_physical_memory()
        require(memory["available_bytes"] >= record["minimum_available_physical_bytes"], "Insufficient available physical memory; no next process launched")
        verify(record["binary"], record["binary_sha256"])
        raw, error = output / (job["id"] + ".jsonl"), output / (job["id"] + ".stderr.txt")
        with raw.open("xb") as stdout, error.open("xb") as stderr:
            process = subprocess.run(job["command"], cwd=ROOT, stdout=stdout, stderr=stderr)
        raw_bytes = raw.read_bytes()
        receipt = {"job": job, "plan_sha256": plan_hash, "memory_before_launch": memory,
                   "exit_code": process.returncode, "stdout_sha256": hashlib.sha256(raw_bytes).hexdigest(), "stderr_sha256": sha(error)}
        invocation = output / (job["id"] + ".invocation.json")
        write_new(invocation, receipt)
        artifact_hashes.update({raw.name: receipt["stdout_sha256"], error.name: receipt["stderr_sha256"], invocation.name: sha(invocation)})
        require(process.returncode == 0, "CLI failed; preserve stdout/stderr and use a fresh plan directory")
        lines = raw_bytes.decode("utf-8").splitlines()
        require(all(line.strip() for line in lines), "Blank/non-JSON CLI stdout line")
        results = [json.loads(line) for line in lines]
        validate_results(results, workload, job)
        if job["id"] == "sequential-control":
            control = dict(zip(job["request_keys"], results))
        else:
            rows = compare_results(control, results, job["request_keys"])
            comparison = {"job": job["id"], "all_requests_exact": all(row["exact"] for row in rows), "requests": rows}
            comparison_path = output / (job["id"] + ".comparison.json")
            write_new(comparison_path, comparison)
            artifact_hashes[comparison_path.name] = sha(comparison_path)
            comparisons.append(comparison)
        completed.append(receipt)
    counts = [control[key]["output_tokens"] for key in workload["cases"][record["case"]]]
    early_eos = any(control[key]["finish_reason"] == "eos" and control[key]["output_tokens"] < max(counts) for key in workload["cases"][record["case"]])
    live_decode_rows = sum(count > 1 for count in counts)
    long_input = any(control[key]["input_tokens"] >= 4096 for key in workload["cases"][record["case"]])
    exact = len(comparisons) == 4 and all(item["all_requests_exact"] for item in comparisons)
    # Repeat all source/build/input checks after this multi-process run.
    verify(path, plan_hash)
    load_plan(path)
    for name, digest in artifact_hashes.items():
        verify(output / name, digest)
    require(execution_identity(record["binary"]) == identity, "Execution identity changed")
    passed = exact and early_eos and live_decode_rows >= 2 and long_input
    report = {"schema_version": 1, "status": "passed" if passed else "failed_or_coverage_incomplete",
              "plan_sha256": plan_hash, "case": record["case"], "all_outputs_exact": exact,
              "mixed_eos_completion_observed": early_eos, "observed_output_lengths": counts,
              "live_requests_after_prefill": live_decode_rows, "long_input_observed": long_input,
              "multirow_decode_eligible_by_output_lengths": live_decode_rows >= 2,
              "all_four_layouts_completed": len(comparisons) == 4, "functional_gate_passed": passed,
              "invocations": completed, "comparisons": comparisons, "artifact_sha256": artifact_hashes,
              "qualification": "Functional output parity on this frozen case only. Different observed output lengths exercise mixed completion; no internal attention trace or allocation proof is inferred. Raw per-request timings are preserved but not compared or aggregated; batch timing attribution is shared. No performance/quality qualification or backend promotion."}
    write_new(output / "report.json", report)
    require(report["functional_gate_passed"], "Functional outputs differ or mixed-EOS coverage was not observed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--build", type=pathlib.Path)
    parser.add_argument("--case")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--plan", type=pathlib.Path, help="Validate/replay a previously frozen plan")
    parser.add_argument("--run", action="store_true", help="Explicitly launch sequential control and four layout processes")
    args = parser.parse_args()
    if args.plan is not None:
        require(args.build is None and args.output is None and args.manifest is None and args.case is None,
                "Existing plan cannot accept build/output/manifest/case overrides")
        path = args.plan
        load_plan(path)
    else:
        require(args.build is not None and args.output is not None, "Planning requires --build and a fresh --output directory")
        path = plan(args.manifest or ROOT / "reference/functional-batch-v1-lock.json", args.build, args.case or "mixed-b4", args.output)
    if args.run:
        result = run(path)
        print(json.dumps({"report": str(path.parent / "report.json"), "functional_gate_passed": result["functional_gate_passed"]}))
    else:
        print(json.dumps({"plan": str(path), "inference_executed": False, "next_step": "Review frozen inputs/order and use --plan <path> --run only when coordinated."}))


if __name__ == "__main__":
    main()
