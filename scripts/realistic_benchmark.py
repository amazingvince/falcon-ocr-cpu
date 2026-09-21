#!/usr/bin/env python3
"""Freeze/validate a benchmark plan; inference requires --run and a quiet-host attestation."""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import platform
import statistics
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD = ROOT / "reference/benchmarks/realistic-fp32-v1-workloads.json"
SCRIPT_FILES = ["scripts/realistic_benchmark.py", "scripts/compare_realistic_benchmarks.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify(path, expected):
    require(sha(path) == expected, f"SHA256 mismatch: {path}")


def validate_inputs(workload):
    # Only lossless decoding for byte verification. No resize or model execution.
    import PIL
    from PIL import Image
    for path, digest in workload["source_locks"].items():
        verify(ROOT / path, digest)
    for name, item in workload["inputs"].items():
        source = next(p for p in read(ROOT / item["source_lock"])["pages"] if p["id"] == item["id"])
        for key in ("source_path", "source_sha256", "canonical_path", "canonical_png_sha256",
                    "rgb_sha256", "width", "height"):
            require(source[key] == item[key], f"{name}: source-lock mismatch in {key}")
        verify(ROOT / item["source_path"], item["source_sha256"])
        verify(ROOT / item["canonical_path"], item["canonical_png_sha256"])
        with Image.open(ROOT / item["canonical_path"]) as image:
            require(image.format == "PNG" and image.mode == "RGB", f"{name}: expected RGB PNG")
            require(image.size == (item["width"], item["height"]), f"{name}: dimensions")
            require(hashlib.sha256(image.tobytes()).hexdigest() == item["rgb_sha256"], f"{name}: pixels")
    model = ROOT / workload["model"]["directory"]
    for name, item in workload["model"]["assets"].items():
        require((model / name).stat().st_size == item["bytes"], f"model asset size: {name}")
        verify(model / name, item["sha256"])
    return {"input_count": len(workload["inputs"]), "lossless_verifier_pillow": PIL.__version__,
            "model_assets_verified": list(workload["model"]["assets"])}


def validate_build(path, check_current=True):
    path = pathlib.Path(path).resolve()
    build = read(path)
    require(build.get("example") == "ocr_bench" and build.get("status") == "complete",
            "Expected a complete capture_rust_build ocr_bench build")
    require(build.get("source_unchanged_during_build") is True and build.get("build_exit_code") == 0,
            "Source changed or build failed")
    require(build.get("cargo_emitted_executable"), "Build must record Cargo's emitted executable")
    require("--locked" in build["command"] and "--release" in build["command"], "Require locked release build")
    binary = path.parent / build["binary"]
    archive = path.parent / build["source_archive"]
    require(binary.parent.resolve() == path.parent and archive.parent.resolve() == path.parent,
            "Preserved binary/archive must be in build directory")
    verify(binary, build["binary_sha256"])
    require(binary.stat().st_size == build["binary_bytes"], "Binary byte count")
    verify(archive, build["source_archive_sha256"])
    hashes = build["source_sha256"]
    required = {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "examples/ocr_bench.rs",
                "scripts/capture_rust_build.py", "scripts/build_windows.ps1", "scripts/build_linux.sh"}
    if check_current:
        required |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.rs")}
        required |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "examples/support").rglob("*.rs")}
        if (ROOT / ".cargo/config.toml").exists():
            required.add(".cargo/config.toml")
    require(required <= hashes.keys(), f"Source archive missing: {sorted(required - hashes.keys())}")
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(hashes) and set(z.namelist()) == set(hashes), "Archive/source map differs")
        for name, digest in hashes.items():
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, f"Archived source hash: {name}")
            if check_current:
                verify(ROOT / name, digest)
        # Schema 1 omitted actual decoder text; historical binaries cannot run this plan.
        harness = z.read("examples/ocr_bench.rs").decode("utf-8")
        require('"schema_version":2' in harness and '"text":r.text' in harness,
                "Benchmark source lacks schema2 literal text reporting")
    return build, binary


def jobs_for(workload, profiles, batches):
    jobs = []
    for profile in workload["profiles"]:
        if profile["name"] not in profiles:
            continue
        for batch in batches:
            keys = profile["order"][:batch]
            requests = [keys[i % len(keys)] for i in range(batch)]
            for mode in workload["schedule"]:
                jobs.append({"id": f"{profile['name']}-b{batch}-{mode['name']}",
                             "profile": profile["name"], "batch": batch, "mode": mode,
                             "image_keys": keys, "request_keys": requests})
    return jobs


def command_for(plan, workload, job, output):
    r = workload["runtime"]
    command = [plan["binary"], "--model", str(ROOT / workload["model"]["directory"]),
               "--threads", str(r["threads"]), "--backend", r["backend"],
               "--execution", job["mode"]["execution"], "--cache-layout", job["mode"]["cache_layout"],
               "--weight-layout", job["mode"]["weight_layout"], "--batches", str(job["batch"]),
               "--warmup", str(r["warmup"]), "--repetitions", str(plan["repetitions"]),
               "--min-dimension", str(r["min_dimension"]), "--max-dimension", str(r["max_dimension"]),
               "--max-new-tokens", str(r["max_new_tokens"]), "--cpu-label", plan["cpu_label"],
               "--environment-label", plan["environment_label"], "--output", str(output)]
    return command + [str(ROOT / workload["inputs"][key]["canonical_path"]) for key in job["image_keys"]]


def validate_plan(path, current=True, inputs=True):
    path = pathlib.Path(path).resolve()
    plan = read(path)
    require(plan.get("kind") == "realistic-fp32-benchmark-plan-v1", "Wrong plan kind")
    verify(plan["workload"], plan["workload_sha256"])
    workload = read(plan["workload"])
    verify(plan["build_manifest"], plan["build_manifest_sha256"])
    build, binary = validate_build(plan["build_manifest"], check_current=current)
    require(str(binary) == plan["binary"] and build["binary_sha256"] == plan["binary_sha256"], "Plan binary binding")
    for name, digest in plan["script_sha256"].items():
        verify(ROOT / name, digest)
    protocol = path.parent / "protocol-source.zip"
    verify(protocol, plan["protocol_archive_sha256"])
    with zipfile.ZipFile(protocol) as archive:
        require(set(archive.namelist()) == set(plan["protocol_source_sha256"]), "Protocol archive closure")
        for name, digest in plan["protocol_source_sha256"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, f"Protocol archive hash: {name}")
    require(plan["repetitions"] >= workload["runtime"]["minimum_repetitions"], "Too few repetitions")
    require(plan["jobs"] == jobs_for(workload, plan["profiles"], plan["batches"]), "Plan job order changed")
    require(plan["output_directory"] == str(path.parent), "Plan moved from its frozen output directory")
    if inputs:
        validate_inputs(workload)
    return plan, workload, build


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"Invalid {name}: {value}")


def validate_report(report, job, plan, workload, build):
    r = workload["runtime"]
    require(report["schema_version"] == 2, "Schema2 literal-text report required")
    for name, expected in {"threads": r["threads"], "backend": r["backend"], "precision": "fp32",
                           "warmup": r["warmup"], "repetitions": plan["repetitions"],
                           "cpu_label": plan["cpu_label"], "environment_label": plan["environment_label"],
                           "model_revision": workload["model"]["revision"],
                           "weights_sha256": workload["model"]["assets"]["model.safetensors"]["sha256"],
                           "binary_sha256": build["binary_sha256"], "cargo_lock_sha256": build["source_sha256"]["Cargo.lock"],
                           "cache_layout": job["mode"]["cache_layout"],
                           "weight_layout": job["mode"]["weight_layout"].replace("-", "_")}.items():
        require(report.get(name) == expected, f"{job['id']}: {name} mismatch")
    require(report["options"] == {k: r[k] for k in ("min_dimension", "max_dimension", "max_new_tokens")}, "Options mismatch")
    for name, digest in report["source_sha256"].items():
        source = "examples/ocr_bench.rs" if name == "harness" else f"src/{name}.rs"
        require(build["source_sha256"].get(source) == digest, f"Embedded source not in preserved build: {name}")
    require(len(report["images"]) == len(job["image_keys"]), "Input image count")
    for observed, key in zip(report["images"], job["image_keys"]):
        expected = workload["inputs"][key]
        for dest, src in [("sha256", "canonical_png_sha256"), ("rgb_sha256", "rgb_sha256"), ("width", "width"), ("height", "height")]:
            require(observed[dest] == expected[src], f"{key}: report input {dest}")
        require(pathlib.Path(observed["path"]).resolve() == (ROOT / expected["canonical_path"]).resolve(), "Input path/order")
    require(len(report["cases"]) == 1, "Exactly one batch per fresh process is required")
    case = report["cases"][0]
    batch = job["batch"]
    sequential = job["mode"]["execution"] == "sequential"
    require(case["batch_size"] == batch and case["active_batch_size"] == (1 if sequential else batch), "Batch dispatch")
    require(case["execution"] == ("independent_sequential" if sequential else "independent_prefill_joint_decode"), "Execution mismatch")
    require(case["image_indices"] == [i % len(job["image_keys"]) for i in range(batch)], "Request order")
    require(len(case["token_ids"]) == batch, "Missing per-request token IDs")
    signatures = []
    for index, (ids, key) in enumerate(zip(case["token_ids"], job["request_keys"])):
        require(ids and all(type(i) is int and 0 <= i < 65536 for i in ids), "Invalid token IDs")
        require(len(ids) <= r["max_new_tokens"] and not any(i in (11, 263) for i in ids[:-1]), "Output length/interior EOS")
        eos = ids[-1] in (11, 263)
        require(eos or len(ids) == r["max_new_tokens"], "Length stop before cap")
        item = workload["inputs"][key]
        signatures.append({"token_ids": ids, "finish_reason": "eos" if eos else "length", "output_tokens": len(ids),
                           "input_tokens": item["input_tokens_expected"], "width": item["prepared_dimensions_expected"][0],
                           "height": item["prepared_dimensions_expected"][1]})
    require(len(case["samples"]) == plan["repetitions"], "Sample count")
    for sample in case["samples"]:
        number(sample["wall_ms"], "wall time", positive=True)
        require(math.isclose(sample["pages_per_second"], batch * 1000 / sample["wall_ms"], rel_tol=1e-12), "Throughput mismatch")
        require(sample["emitted_tokens"] == sum(map(len, case["token_ids"])), "Emitted count")
        require(len(sample["per_request"]) == batch, "Missing per-request sample")
        for request, expected in zip(sample["per_request"], signatures):
            for name in ("finish_reason", "output_tokens", "input_tokens", "width", "height"):
                require(request[name] == expected[name], f"Request {name} mismatch")
            require(request.get("teacher_forced") is False and request.get("precision") == "fp32", "Expected free FP32 generation")
            require(isinstance(request.get("text"), str), "Missing actual decoder text")
            if "text" in expected:
                require(request["text"] == expected["text"], "Text changed between repetitions")
            expected["text"] = request["text"]
            for name in ("image_decode_ms", "preprocessing_ms", "image_projection_ms", "transformer_prefill_ms", "prefill_ms", "decode_ms", "total_ms", "time_to_first_token_ms"):
                number(request["timings"].get(name), name)
            require(request["timings"]["image_projection_ms"] + request["timings"]["transformer_prefill_ms"] <= request["timings"]["prefill_ms"] + 1e-6,
                    "Projection/transformer subintervals exceed enclosing prefill")
            require(request["timings"]["image_decode_ms"] == 0, "Warm RGB input unexpectedly includes file decode")
    med = statistics.median(s["wall_ms"] for s in case["samples"])
    require(math.isclose(case["median_ms"], med, rel_tol=1e-12), "Median mismatch")
    require(math.isclose(case["median_pages_per_second"], batch * 1000 / med, rel_tol=1e-12), "Median throughput mismatch")
    for name in ("read_decode_ms", "verified_model_load_ms", "weight_packing_ms", "packed_weight_bytes"):
        number(report[name], name)
    require((report["packed_weight_bytes"] > 0) == (job["mode"]["weight_layout"] == "phase-packed"), "Packed memory/layout mismatch")
    return signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=pathlib.Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--build", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="New plan directory; never overwritten")
    parser.add_argument("--profiles", help="Comma-separated frozen profile names; default all")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cpu-label")
    parser.add_argument("--environment-label")
    parser.add_argument("--validate-inputs-only", action="store_true", help="No binary/run plan qualification")
    parser.add_argument("--plan", type=pathlib.Path, help="Validate or execute an existing immutable plan")
    parser.add_argument("--run", action="store_true", help="Actually execute, only with --plan and quiet attestation")
    parser.add_argument("--quiet-attestation", help="Record who verified other CPU/GPU project work was paused")
    args = parser.parse_args()
    if args.run:
        require(args.plan and args.quiet_attestation and args.quiet_attestation.strip(), "--run requires --plan and --quiet-attestation")
        plan, workload, build = validate_plan(args.plan)
        require(plan["host_platform"] == platform.platform(), "Execution host differs from plan")
        folder = args.plan.resolve().parent
        paths = [folder / f"{j['id']}.{ext}" for j in plan["jobs"] for ext in ("json", "log", "command.json")]
        require(not any(p.exists() for p in paths), "Refusing partial/existing results; create a new plan for the whole bracket")
        write_new(folder / "execution-start.json", {"started_utc": utc(), "plan_sha256": sha(args.plan),
                  "quiet_attestation": args.quiet_attestation, "platform": platform.platform()})
        receipts = []
        for job in plan["jobs"]:
            target = folder / f"{job['id']}.json"
            command = command_for(plan, workload, job, target)
            write_new(folder / f"{job['id']}.command.json", {"command": command, "cwd": str(ROOT), "started_utc": utc()})
            print(f"Running {job['id']}", flush=True)
            with (folder / f"{job['id']}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_report(read(target), job, plan, workload, build)
            receipts.append({"job": job["id"], "sha256": sha(target)})
        # Recheck file identity after the timed window, not between measurements.
        validate_plan(args.plan)
        write_new(folder / "execution-complete.json", {"finished_utc": utc(), "plan_sha256": sha(args.plan), "reports": receipts})
        print("Execution complete; run compare_realistic_benchmarks.py for independent mode comparisons.")
        return
    if args.plan:
        plan, _, _ = validate_plan(args.plan)
        print(json.dumps({"status": "validated_plan_no_execution", "jobs": len(plan["jobs"]), "performance_run": False}))
        return
    workload = read(args.workload)
    verified = validate_inputs(workload)
    if args.validate_inputs_only:
        print(json.dumps({"status": "inputs_only_not_build_qualified", "performance_run": False, **verified}, indent=2))
        return
    require(args.build and args.output and args.cpu_label and args.environment_label, "Planning requires --build --output --cpu-label --environment-label")
    build, binary = validate_build(args.build)
    require(args.repetitions >= workload["runtime"]["minimum_repetitions"], "Minimum three repetitions")
    profiles = args.profiles.split(",") if args.profiles else [p["name"] for p in workload["profiles"]]
    require(profiles and len(profiles) == len(set(profiles)) and set(profiles) <= {p["name"] for p in workload["profiles"]}, "Unknown/duplicate profile")
    batches = list(map(int, args.batches.split(",")))
    require(batches and batches == sorted(set(batches)) and set(batches) <= set(workload["runtime"]["batches"]), "Batches must be an increasing subset of 1,2,4,8")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    protocol_files = {name: (ROOT / name).read_bytes() for name in SCRIPT_FILES + list(workload["source_locks"])}
    protocol_files["workload.json"] = args.workload.read_bytes()
    with zipfile.ZipFile(folder / "protocol-source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(protocol_files.items()):
            archive.writestr(name, contents)
    plan = {"kind": "realistic-fp32-benchmark-plan-v1", "created_utc": utc(), "performance_run": False,
            "workload": str(args.workload.resolve()), "workload_sha256": sha(args.workload),
            "build_manifest": str(args.build.resolve()), "build_manifest_sha256": sha(args.build),
            "binary": str(binary), "binary_sha256": build["binary_sha256"], "source_archive_sha256": build["source_archive_sha256"],
            "script_sha256": {name: sha(ROOT / name) for name in SCRIPT_FILES}, "output_directory": str(folder),
            "protocol_archive_sha256": sha(folder / "protocol-source.zip"),
            "protocol_source_sha256": {name: hashlib.sha256(contents).hexdigest() for name, contents in protocol_files.items()},
            "host_platform": platform.platform(), "cpu_label": args.cpu_label, "environment_label": args.environment_label,
            "repetitions": args.repetitions, "profiles": profiles, "batches": batches, "verified_inputs": verified,
            "coverage": "complete_matrix" if len(profiles) == len(workload["profiles"]) and batches == workload["runtime"]["batches"] else "selected_groups_only",
            "jobs": jobs_for(workload, profiles, batches)}
    write_new(folder / "plan.json", plan)
    write_new(folder / "commands.json", [{"id": j["id"], "command": command_for(plan, workload, j, folder / f"{j['id']}.json")} for j in plan["jobs"]])
    print(json.dumps({"status": "planned_only", "plan": str(folder / "plan.json"), "jobs": len(plan["jobs"]), "performance_run": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
