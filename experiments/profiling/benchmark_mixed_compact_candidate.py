#!/usr/bin/env python3
"""One fixed mixed-B2 compact candidate between unchanged compact CPU controls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import realistic_benchmark as rb

GPU_RECORDS = {
    "sparse-room": ROOT / "artifacts/reference/functional-originals-fp32-4096/sparse-room.json",
    "table": ROOT / "artifacts/reference/corpus-v3-fp32-4096/2c243b31d36eb729.json",
}
# These identities were accepted before this experiment, in the completed
# functional-original and 200-page corpus comparisons respectively.
GPU_RECORD_HASHES = {
    "sparse-room": "a9a77300c5f25f03d420603b2172eeeb4f35c09b215bf0d793aadca44148dd0c",
    "table": "5434fc7bd87510672eb81c7227412cb43f370977c99d8e4d3575612304e361d3",
}
KEYS = list(GPU_RECORDS)
JOBS = [{"name": "control-before", "build": "control"},
        {"name": "candidate", "build": "candidate"},
        {"name": "control-after", "build": "control"}]


def job(name):
    return {"id": name, "profile": "mixed-sparse-first", "batch": 2,
            "mode": {"execution": "joint", "cache_layout": "compact", "weight_layout": "unpacked"},
            "image_keys": KEYS, "request_keys": KEYS}


def validate(plan_path, expected_sha):
    plan_bytes = plan_path.read_bytes()
    rb.require(hashlib.sha256(plan_bytes).hexdigest() == expected_sha, "Plan hash mismatch")
    plan = json.loads(plan_bytes)
    rb.require(plan["kind"] == "mixed-b2-compact-candidate-bracket-v1", "Wrong plan kind")
    rb.require(plan["output_directory"] == str(plan_path.parent), "Plan moved")
    rb.require(plan["host_platform"] == platform.platform(), "Host changed")
    rb.require(plan["jobs"] == JOBS and plan["repetitions"] == 3, "Fixed bracket changed")
    rb.require(set(plan["builds"]) == {"control", "candidate"}, "Build labels changed")
    rb.require(plan["max_process_seconds"] == 1800 and plan["control_drift_limit_percent"] == 5
               and plan["target_latency_reduction_percent"] == 5
               and plan["maximum_latency_regression_percent"] == 5 and plan["default_promotion"] is False,
               "Benchmark limits changed")
    required = {str(Path(__file__).resolve()), str(ROOT / "scripts/realistic_benchmark.py"),
                str(plan_path.parent / "protocol-source.zip"), plan["workload"], *map(str, GPU_RECORDS.values()),
                *plan["builds"].values()}
    for name, digest in plan["files_sha256"].items():
        rb.verify(name, digest)
    workload = rb.read(plan["workload"])
    rb.require(workload["runtime"]["warmup"] == 2 and workload["runtime"]["threads"] == 16
               and workload["runtime"]["backend"] == "avx2", "Runtime changed")
    rb.require(plan["gpu_records"] == {key: str(path) for key, path in GPU_RECORDS.items()}, "GPU records changed")
    rb.require(expected_signatures(workload) == plan["expected_signatures"], "Expected GPU signatures changed")
    rb.validate_inputs(workload)
    builds = {}
    for label, path in plan["builds"].items():
        builds[label] = rb.validate_build(path, check_current=False)
        build, binary = builds[label]
        required.update((str(binary), str(Path(path).parent / build["source_archive"])))
    rb.require(set(plan["files_sha256"]) == required, "Bound file inventory changed")
    return plan, workload, builds


def read_gpu_record(key, path):
    raw = path.read_bytes()
    rb.require(hashlib.sha256(raw).hexdigest() == GPU_RECORD_HASHES[key], "Accepted GPU record changed")
    return json.loads(raw)


def expected_signatures(workload):
    profile = next(p for p in workload["profiles"] if p["name"] == "mixed-sparse-first")
    rb.require(profile["order"][:2] == KEYS, "Frozen mixed request order changed")
    signatures = []
    for key, path in GPU_RECORDS.items():
        record = read_gpu_record(key, path)
        item = workload["inputs"][key]
        cfg = record["configuration"]
        rb.require(record["id"] == item["id"] and record["canonical_rgb_sha256"] == item["rgb_sha256"],
                   f"{key}: GPU input identity differs")
        for field in ("max_new_tokens", "min_dimension", "max_dimension", "precision"):
            rb.require(cfg[field] == workload["runtime"][field], f"{key}: GPU {field} differs")
        rb.require(cfg["model_revision"] == workload["model"]["revision"] and
                   cfg["weights_sha256"] == workload["model"]["assets"]["model.safetensors"]["sha256"],
                   f"{key}: GPU model identity differs")
        rb.require(cfg["tf32"] is False and cfg["flex_float32_precision"] == "ieee" and
                   cfg["compiled_blocks"] is False, f"{key}: GPU strict FP32 contract differs")
        for scope in (record, cfg):
            rb.require("error" not in scope and "postprocessing_replay" not in scope,
                       f"{key}: error or replay record")
            rb.require("teacher_forced" not in scope or scope["teacher_forced"] is False,
                       f"{key}: teacher forcing")
            rb.require("inference_reexecuted" not in scope or scope["inference_reexecuted"] is True,
                       f"{key}: inference explicitly not executed")
        ids = record["token_ids"]
        rb.require(ids and all(type(i) is int and 0 <= i < 65536 for i in ids), "Invalid GPU IDs")
        rb.require(not any(i in (11, 263) for i in ids[:-1]), "Interior GPU EOS")
        rb.require(record["finish_reason"] == "eos" and ids[-1] in (11, 263) and
                   len(ids) == {"sparse-room": 6, "table": 2280}[key], "Frozen GPU output changed")
        rb.require(record["prefix_length"] == item["input_tokens_expected"], "GPU prefix differs")
        rb.require(isinstance(record["text"], str), "Missing GPU literal text")
        signatures.append({"token_ids": ids, "text": record["text"], "finish_reason": record["finish_reason"],
                           "output_tokens": len(ids), "input_tokens": record["prefix_length"],
                           "width": item["prepared_dimensions_expected"][0],
                           "height": item["prepared_dimensions_expected"][1]})
    return signatures


def prepare(args):
    workload_path = rb.DEFAULT_WORKLOAD
    workload = rb.read(workload_path)
    rb.validate_inputs(workload)
    builds = {label: rb.validate_build(path, check_current=False)
              for label, path in (("control", args.control), ("candidate", args.candidate))}
    control, candidate = (builds[k][0] for k in ("control", "candidate"))
    for name in ("rustc_version", "cargo_version"):
        rb.require(control[name] == candidate[name], f"Build tool differs: {name}")
    # The copied project and fresh Cargo target necessarily have different paths.
    for build in (control, candidate):
        rb.require("build" in build["command"], "Missing Cargo build command")
    rb.require(control["command"][control["command"].index("build"):] ==
               candidate["command"][candidate["command"].index("build"):], "Cargo build arguments differ")
    rb.require({k: v for k, v in control["environment_overrides"].items() if k != "CARGO_TARGET_DIR"} ==
               {k: v for k, v in candidate["environment_overrides"].items() if k != "CARGO_TARGET_DIR"},
               "Build environment differs beyond isolated target directory")
    rb.require(control["source_sha256"].keys() == candidate["source_sha256"].keys(), "Source inventory changed")
    changed = [name for name in control["source_sha256"]
               if control["source_sha256"][name] != candidate["source_sha256"][name]]
    rb.require(changed == ["src/kernels.rs"], f"Expected isolated kernel source difference, got {changed}")
    rb.require(workload["runtime"]["warmup"] == 2, "Expected baseline two warmups")
    rb.require(workload["runtime"]["threads"] == 16 and workload["runtime"]["backend"] == "avx2", "Runtime changed")
    signatures = expected_signatures(workload)
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), ROOT / "scripts/realistic_benchmark.py"]
    protocol = folder / "protocol-source.zip"
    with zipfile.ZipFile(protocol, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    bound = source_paths + [protocol, workload_path, *GPU_RECORDS.values(), args.control.resolve(), args.candidate.resolve()]
    for path, build in ((args.control, control), (args.candidate, candidate)):
        bound += [path.parent / build["binary"], path.parent / build["source_archive"]]
    plan = {"kind": "mixed-b2-compact-candidate-bracket-v1", "created_utc": rb.utc(),
            "output_directory": str(folder), "host_platform": platform.platform(),
            "workload": str(workload_path), "cpu_label": "AMD Ryzen 9 7950X",
            "environment_label": "Native Windows", "repetitions": 3,
            "builds": {"control": str(args.control.resolve()), "candidate": str(args.candidate.resolve())},
            "source_changes": changed, "expected_signatures": signatures,
            "gpu_records": {key: str(path) for key, path in GPU_RECORDS.items()},
            "files_sha256": {str(p.resolve()): rb.sha(p) for p in bound},
            "max_process_seconds": 1800, "control_drift_limit_percent": 5,
            "target_latency_reduction_percent": 5, "maximum_latency_regression_percent": 5,
            "default_promotion": False,
            "jobs": JOBS,
            "limitations": ["One fixed mixed pair and three processes; no broad batch/platform promotion.",
                           "This pair has only a short multirow decode interval and a long singleton tail; it does not measure sustained B2 throughput.",
                           "GPU records lack observed prepared dimensions; expected dimensions come from the frozen workload and are checked on every CPU output.",
                           "The historical table GPU record lacks complete startup attestation; saved output agreement does not fill those missing fields.",
                           "Stage timings are wall intervals, not hardware counters.",
                           "Model load and file decode precede warmed measured recognitions."]}
    rb.write_new(folder / "plan.json", plan)
    print(json.dumps({"status": "planned", "plan": str(folder / "plan.json"), "sha256": rb.sha(folder / "plan.json")}))


def run(args):
    plan_path = args.plan.resolve()
    plan, workload, builds = validate(plan_path, args.plan_sha256)
    rb.require(expected_signatures(workload) == plan["expected_signatures"], "Frozen expected signatures differ")
    folder = plan_path.parent
    rb.require(args.quiet_attestation and args.quiet_attestation.strip(), "Quiet attestation required")
    rb.require(not any((folder / f"{item['name']}.{extension}").exists()
               for item in plan["jobs"] for extension in ("json", "log", "execution.json")), "Partial bracket exists")
    rb.write_new(folder / "execution-start.json", {"started_utc": rb.utc(), "plan_sha256": rb.sha(plan_path),
                 "quiet_attestation": args.quiet_attestation})
    reports = []
    try:
        for item in plan["jobs"]:
            build, binary = builds[item["build"]]
            name = item["name"]
            output = folder / f"{name}.json"
            command = rb.command_for(dict(plan, binary=str(binary)), workload, job(name), output)
            execution = {"command": command, "cwd": str(ROOT), "started_utc": rb.utc()}
            print(f"Running {name}", flush=True)
            with (folder / f"{name}.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    execution["pid"] = child.pid
                    rb.write_new(folder / f"{name}.start.json", execution)
                    code = child.wait(timeout=plan["max_process_seconds"])
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
                    execution["timed_out"] = True
                except BaseException:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
                    raise
                execution.update(exit_code=code, finished_utc=rb.utc())
                rb.write_new(folder / f"{name}.execution.json", execution)
            rb.require(code == 0 and not execution.get("timed_out"), f"{name} failed")
            report = rb.read(output)
            signatures = rb.validate_report(report, job(name), plan, workload, build)
            rb.require(signatures == plan["expected_signatures"], f"{name}: output differs from frozen GPU records")
            case = report["cases"][0]
            reports.append({"name": name, "report_sha256": rb.sha(output), "binary_sha256": build["binary_sha256"],
                            "all_measured_outputs_exact": True, "median_ms": case["median_ms"],
                            "samples_ms": [sample["wall_ms"] for sample in case["samples"]],
                            "median_pages_per_second": case["median_pages_per_second"],
                            "per_request_stage_medians_ms": [
                                {key: statistics.median(s["per_request"][index]["timings"][key]
                                                       for s in case["samples"]) for key in
                                 ("preprocessing_ms", "prefill_ms", "decode_ms", "total_ms")}
                                for index in range(2)]})
        validate(plan_path, args.plan_sha256)  # Bulk checks only outside the measured process bracket.
        before, candidate, after = [r["median_ms"] for r in reports]
        drift = 100 * (after / before - 1)
        gains = {"control_before": 100 * (1 - candidate / before), "control_after": 100 * (1 - candidate / after)}
        result = {"status": "complete", "finished_utc": rb.utc(), "plan_sha256": rb.sha(plan_path),
                  "reports": reports, "control_drift_percent": drift,
                  "control_stability_pass": abs(drift) <= plan["control_drift_limit_percent"],
                  "candidate_latency_reduction_percent": gains,
                  "target_met_in_this_pair": abs(drift) <= plan["control_drift_limit_percent"]
                       and min(gains.values()) >= plan["target_latency_reduction_percent"],
                  "no_regression_in_this_pair": abs(drift) <= plan["control_drift_limit_percent"]
                       and min(gains.values()) >= -plan["maximum_latency_regression_percent"],
                  "default_promotion": False}
        rb.write_new(folder / "comparison.json", result)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        rb.write_new(folder / "execution-failed.json", {"finished_utc": rb.utc(), "error": str(error), "completed_reports": reports})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet-attestation")
    args = parser.parse_args()
    if args.run:
        rb.require(args.plan and args.plan_sha256, "--run requires --plan and --plan-sha256")
        run(args)
    elif args.plan:
        rb.require(args.plan_sha256, "Validation requires --plan-sha256")
        validate(args.plan.resolve(), args.plan_sha256)
        print("Plan validated; no inference")
    else:
        rb.require(all((args.control, args.candidate, args.output)), "Planning needs both builds and output")
        prepare(args)


if __name__ == "__main__":
    main()
