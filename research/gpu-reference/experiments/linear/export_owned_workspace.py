#!/usr/bin/env python3
"""Prospective direct-cuBLAS diagnostic; GPU execution belongs to the reference owner.

Each subprocess exits before its logger files are checked. No basis worker starts
until the real-input control and all preceding workers have passed their gates.
Only new, caller-owned allocations are read. This is not a runner backend.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
import zipfile

from workspace_observation import (CASE, M, N, K, WORKSPACE_BYTES, CANARY_BYTE, PINS,
                                   require, sha256, read_json, write_json, verify_pins,
                                   signature, parse_logs, basis_operands, observe_layout)

ROOT = Path(__file__).resolve().parents[4]
SOURCE_NAMES = ["research/gpu-reference/experiments/linear/export_owned_workspace.py", "research/gpu-reference/experiments/linear/workspace_observation.py",
                "research/gpu-reference/experiments/linear/test_workspace_observation.py", "research/gpu-reference/experiments/linear/OWNED-WORKSPACE.md",
                "scripts/reference_preflight.py", "scripts/fetch_reference.py", "requirements/reference-lock.txt",
                "reference/manifest.json", "artifacts/model/artifact-manifest.json"]


def source_state():
    return {name: sha256(ROOT / name) for name in SOURCE_NAMES}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def save_array(path, value):
    with path.open("xb") as f:
        f.write(value.tobytes(order="C"))
    return {"sha256": sha256(path), "bytes": path.stat().st_size,
            "shape": list(value.shape), "dtype": str(value.dtype)}


def bind(lib, name, arguments):
    function = getattr(lib, name)
    function.argtypes, function.restype = arguments, ctypes.c_int
    return function


def checked(function, *args):
    status = function(*args)
    require(status == 0, f"{function.__name__} failed with status {status}")


def loaded_libraries(expected):
    maps = Path("/proc/self/maps").read_text()
    paths = {line.split()[-1] for line in maps.splitlines() if line.split()[-1].startswith("/")}
    found = {name: sha256(name) for name in paths if any(part in name for part in ("libcublas", "libtorch_cuda", "libcudart"))}
    require(found == expected, "Actually loaded CUDA/BLAS/Torch library bytes differ")
    return found


def worker(output, mode):
    # Every worker starts a new interpreter with log paths already in its environment.
    require(platform.system() == "Linux", "Pinned diagnostic requires the reference Linux environment")
    start_sources = source_state()
    inputs = verify_pins(ROOT)
    expected_logs = {"CUBLAS_LOGINFO_DBG": "1", "CUBLAS_LOGDEST_DBG": str(output / "cublas.log"),
                     "CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_MASK": "31",
                     "CUBLASLT_LOG_FILE": str(output / "cublaslt_%i.log")}
    require(all(os.environ.get(k) == v for k, v in expected_logs.items()), "Logger environment changed")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Workspace environment differs")
    sys.path.insert(0, str(ROOT / "scripts"))
    from reference_preflight import preflight
    environment = preflight(ROOT)
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    original = read_json(ROOT / "artifacts/reference/linear-logger-fp32/report.json")
    expected_record = next(r for r in original["records"] if r["name"] == CASE)
    meta = read_json(ROOT / "artifacts/reference/linear-operators.json")
    entry = next(r for r in meta["fixtures"] if r["name"] == CASE)
    require(entry["weight_key"] == "weights.layers.12.feed_forward.w2.weight", "Wrong fixture weight mapping")
    if mode == "control":
        with safe_open(str(ROOT / "artifacts/reference/linear-operators.safetensors"), framework="np") as f:
            x, w, expected = (f.get_tensor(name).copy() for name in (CASE + ".input", entry["weight_key"], CASE + ".expected"))
        require(array_sha(x) == expected_record["input_sha256"] and array_sha(w) == expected_record["weight_sha256"],
                "Original operand bytes differ")
    else:
        x, w, expected = basis_operands(int(mode.removeprefix("basis-")))
    require(x.dtype == w.dtype == expected.dtype == np.dtype("float32"), "Wrong dtype")
    require(x.shape == (N, K) and w.shape == (M, K) and expected.shape == (N, M), "Wrong shape")
    artifacts = {"input.f32le": save_array(output / "input.f32le", x),
                 "weight.f32le": save_array(output / "weight.f32le", w),
                 "expected.f32le": save_array(output / "expected.f32le", expected)}
    a = torch.from_numpy(w).cuda()
    b = torch.from_numpy(x).cuda()
    c = torch.empty((N, M), dtype=torch.float32, device="cuda")
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    require(workspace.data_ptr() % 256 == 0 and all(t.data_ptr() % 16 == 0 for t in (a, b, c)), "Allocation alignment differs")
    torch.cuda.synchronize()
    library_path = next(p for p in original["loaded_libraries"] if p.endswith("/libcublas.so.13"))
    require(sha256(library_path) == original["loaded_libraries"][library_path], "cuBLAS binary changed")
    lib = ctypes.CDLL(library_path)
    p, i = ctypes.c_void_p, ctypes.c_int
    create = bind(lib, "cublasCreate_v2", [ctypes.POINTER(p)])
    destroy = bind(lib, "cublasDestroy_v2", [p])
    set_stream = bind(lib, "cublasSetStream_v2", [p, p])
    set_workspace = bind(lib, "cublasSetWorkspace_v2", [p, p, ctypes.c_size_t])
    set_math = bind(lib, "cublasSetMathMode", [p, i])
    get_pointer = bind(lib, "cublasGetPointerMode_v2", [p, ctypes.POINTER(i)])
    get_version = bind(lib, "cublasGetVersion_v2", [p, ctypes.POINTER(i)])
    gemm = bind(lib, "cublasSgemm_v2", [p, i, i, i, i, i, p, p, i, p, i, p, p, i])
    handle = p()
    checked(create, ctypes.byref(handle))
    pointer_mode, version = i(), i()
    try:
        checked(get_pointer, handle, ctypes.byref(pointer_mode))
        checked(get_version, handle, ctypes.byref(version))
        require(pointer_mode.value == 0, "New handle does not use host scalar pointer mode")
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)

        def invoke():
            # Match the observed Torch helper ordering on every call.
            checked(set_stream, handle, p(0))
            checked(set_workspace, handle, p(workspace.data_ptr()), WORKSPACE_BYTES)
            checked(set_math, handle, 0)
            checked(gemm, handle, 1, 0, M, N, K, ctypes.byref(alpha), p(a.data_ptr()), K,
                    p(b.data_ptr()), K, ctypes.byref(beta), p(c.data_ptr()), M)

        warmups = 3 if mode == "control" else 0
        for _ in range(warmups):
            workspace.fill_(CANARY_BYTE)
            c.zero_()
            torch.cuda.synchronize()
            invoke()
            torch.cuda.synchronize()
        workspace.fill_(CANARY_BYTE)
        c.zero_()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, with_stack=False) as profile:
            with torch.profiler.record_function("owned-workspace-" + mode):
                invoke()
            torch.cuda.synchronize()
        # No cuBLAS call or workspace reinitialization may precede these copies.
        candidate = c.cpu().numpy().copy()
        scratch = workspace.cpu().numpy().copy()
        trace_path = output / "trace.json"
        profile.export_chrome_trace(str(trace_path))
        kernels = [{"name": e.get("name"), "arguments": e.get("args", {})}
                   for e in read_json(trace_path).get("traceEvents", []) if e.get("cat") == "kernel"]
        artifacts["output.f32le"] = save_array(output / "output.f32le", candidate)
        artifacts["workspace.bin"] = save_array(output / "workspace.bin", scratch)
        artifacts["trace.json"] = {"sha256": sha256(trace_path), "bytes": trace_path.stat().st_size}
        mismatches = np.flatnonzero(candidate.view(np.uint32).ravel() != expected.view(np.uint32).ravel())
        libraries = loaded_libraries(original["loaded_libraries"])
        report = {"schema_version": 1, "mode": mode, "pid": os.getpid(), "created_utc": utc(),
                  "environment": environment, "logger_environment": expected_logs,
                  "precision_environment": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_EMULATION_STRATEGY", "CUBLAS_WORKSPACE_CONFIG")},
                  "source_hashes": start_sources, "pinned_inputs": inputs,
                  "loaded_libraries": libraries, "cublas_version_integer": version.value,
                  "pointer_mode": pointer_mode.value, "alpha_bits": "3f800000", "beta_bits": "00000000",
                  "workspace": {"owned": True, "address": workspace.data_ptr(), "bytes": WORKSPACE_BYTES,
                                "canary_byte": CANARY_BYTE, "initial_sha256": hashlib.sha256(bytes([CANARY_BYTE]) * WORKSPACE_BYTES).hexdigest(),
                                "changed_bytes": int(np.count_nonzero(scratch != CANARY_BYTE))},
                  "operand_addresses": {"A": a.data_ptr(), "B": b.data_ptr(), "C": c.data_ptr()},
                  "warmups": warmups, "profiled_calls": 1, "kernels": kernels,
                  "kernel_launches_match_original": signature(kernels) == signature(expected_record["kernels"]),
                  "output_bit_mismatches": int(mismatches.size), "first_output_mismatches": mismatches[:16].tolist(),
                  "artifacts": artifacts, "qualification": "Owned scratch bytes only; no layout interpretation in worker."}
        require(source_state() == start_sources, "Worker source closure changed")
        write_json(output / "worker.json", report)
    finally:
        checked(destroy, handle)


def validate_worker(directory, mode, sources):
    report_bytes = (directory / "worker.json").read_bytes()
    report = json.loads(report_bytes)
    require(report["mode"] == mode and report["source_hashes"] == sources, "Worker/source identity mismatch")
    require(report["output_bit_mismatches"] == 0, "Direct SGEMM output differs from exact expected bits")
    require(report["kernel_launches_match_original"], "Kernel names/grids/blocks differ")
    original = read_json(ROOT / "artifacts/reference/linear-logger-fp32/report.json")
    require(report["loaded_libraries"] == original["loaded_libraries"], "Loaded libraries differ")
    validated_files = {"worker.json": hashlib.sha256(report_bytes).hexdigest()}
    comparison_bytes = {}
    for name, meta in report["artifacts"].items():
        path = directory / name
        data = path.read_bytes()
        require(len(data) == meta["bytes"] and hashlib.sha256(data).hexdigest() == meta["sha256"], "Captured worker artifact changed: " + name)
        validated_files[name] = meta["sha256"]
        if name in ("output.f32le", "expected.f32le", "trace.json"):
            comparison_bytes[name] = data
    require(comparison_bytes["output.f32le"] == comparison_bytes["expected.f32le"],
            "Preserved output bytes differ from exact expected bytes")
    expected_record = next(r for r in original["records"] if r["name"] == CASE)
    kernels = [{"name": e.get("name"), "arguments": e.get("args", {})}
               for e in json.loads(comparison_bytes["trace.json"]).get("traceEvents", []) if e.get("cat") == "kernel"]
    require(signature(kernels) == signature(expected_record["kernels"]), "Preserved kernel launches differ")
    logs = list(directory.glob("cublaslt_*.log"))
    require(len(logs) == 1 and logs[0].name == f"cublaslt_{report['pid']}.log", "Unexpected Lt logger files/PID")
    classic_bytes, lt_bytes = (directory / "cublas.log").read_bytes(), logs[0].read_bytes()
    classic, lt = (data.decode().replace("\r\n", "\n") for data in (classic_bytes, lt_bytes))
    require(set(int(x) for x in re.findall(r"Process=(\d+);", classic)) == {report["pid"]}, "Classic logger PID differs")
    config = parse_logs(classic, lt, 4 if mode == "control" else 1)
    validated_files.update({"cublas.log": hashlib.sha256(classic_bytes).hexdigest(), logs[0].name: hashlib.sha256(lt_bytes).hexdigest()})
    gate = {"status": "passed", "mode": mode, "worker_sha256": validated_files["worker.json"],
            "execution": config, "validated_files": validated_files,
            "logs": {name: validated_files[name] for name in ["cublas.log", logs[0].name]}}
    write_json(directory / "gate.json", gate)
    return gate


def orchestrate(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = source_state()
    inputs = verify_pins(ROOT)
    with zipfile.ZipFile(output / "sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_NAMES:
            data = (ROOT / name).read_bytes()
            require(hashlib.sha256(data).hexdigest() == sources[name], "Source changed while archiving")
            archive.writestr(name, data)
    start = {"created_utc": utc(), "python": sys.executable, "platform": platform.platform(),
             "source_hashes": sources, "sources_zip_sha256": sha256(output / "sources.zip"),
             "pinned_inputs": inputs, "ordered_modes": ["control", "basis-0", "basis-1", "basis-2"],
             "diagnostic_only": True}
    write_json(output / "startup.json", start)
    gates, failure, closed_phase_files = [], None, {}
    try:
        for mode in start["ordered_modes"]:
            require(source_state() == sources, "Source changed before next worker")
            directory = output / mode
            directory.mkdir()
            env = dict(os.environ)
            env.update({"CUBLAS_LOGINFO_DBG": "1", "CUBLAS_LOGDEST_DBG": str(directory / "cublas.log"),
                        "CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_MASK": "31",
                        "CUBLASLT_LOG_FILE": str(directory / "cublaslt_%i.log")})
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--output", str(directory)]
            write_json(directory / "invocation.json", {"command": command, "cwd": str(ROOT),
                       "source_hashes": sources, "logger_environment": {k: env[k] for k in env if k.startswith("CUBLAS")},
                       "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")})
            with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
            write_json(directory / "exit.json", {"returncode": result.returncode})
            require(result.returncode == 0, f"{mode} worker failed: {result.returncode}")
            gate = validate_worker(directory, mode, sources)
            gates.append(gate)
            for name, digest in gate["validated_files"].items():
                closed_phase_files[f"{mode}/{name}"] = digest
            for path in directory.iterdir():
                if path.is_file():
                    name, digest = path.relative_to(output).as_posix(), sha256(path)
                    require(name not in closed_phase_files or closed_phase_files[name] == digest,
                            "Validated phase artifact changed before closure: " + name)
                    closed_phase_files[name] = digest
            print(f"{mode}: exact outputs, finalized execution logs and kernel launches passed", flush=True)
        layout = observe_layout([output / f"basis-{g}" / "workspace.bin" for g in range(3)])
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        layout = None
    # Revalidate captured bytes and source pins after every completed mode.
    closure_errors = []
    try:
        require(source_state() == sources, "Final source closure differs")
        require(verify_pins(ROOT) == inputs, "Final input pins differ")
        require(sha256(output / "sources.zip") == start["sources_zip_sha256"], "Source archive changed")
        for name, digest in closed_phase_files.items():
            require(sha256(output / name) == digest, "Closed phase artifact changed: " + name)
    except Exception as error:
        closure_errors.append(str(error))
    closure_ok = not closure_errors
    captured = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            captured[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    status = "execution_failed" if failure or not closure_ok else "captured_layout_observed_conditionally" if layout["complete_basis_agreement"] else "captured_layout_not_established"
    report = {"schema_version": 1, "status": status, "created_utc": utc(), "startup": start,
              "source_and_artifact_closure_unchanged": bool(closure_ok), "closure_errors": closure_errors,
              "gates": gates, "failure": failure,
              "layout_observation": layout, "artifacts": captured,
              "qualification": "No production/quality/performance claim. Complete basis checks concern one explicit scratch layout; failed interpretation is preserved rather than searched or repaired."}
    write_json(output / "report.json", report)
    print(status, flush=True)
    return 1 if failure or not closure_ok else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=["control", "basis-0", "basis-1", "basis-2"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.byteorder == "little", "Artifact format requires little endian")
    require(not sys.flags.optimize, "Run without Python optimization")
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker)
        return 0
    return orchestrate(output)


if __name__ == "__main__":
    raise SystemExit(main())
