#!/usr/bin/env python3
"""Replay the two original prefill projections with supported BLAS logging only."""
import hashlib
import importlib.metadata
import json
import os
import pathlib

import torch
from safetensors.torch import load_file, save_file

from fetch_reference import sha256
from reference_preflight import preflight


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    output = root / "artifacts/reference/linear-logger-fp32"
    expected_env = {"CUBLAS_LOGINFO_DBG": "1", "CUBLAS_LOGDEST_DBG": str(output / "cublas.log"),
                    "CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_MASK": "31",
                    "CUBLASLT_LOG_FILE": str(output / "cublaslt_%i.log")}
    if any(os.environ.get(key) != value for key, value in expected_env.items()):
        raise ValueError("Logger environment differs from the prepared startup configuration")
    report_path = output / "report.json"
    script_copy = output / pathlib.Path(__file__).name
    if report_path.exists() or script_copy.exists():
        raise ValueError("Preserve previous logging attempts")
    script_copy.write_bytes(pathlib.Path(__file__).read_bytes())
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source = pathlib.Path("artifacts/reference/linear-operators.safetensors")
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    original_path = root / "reference/linear-cuda-kernel-identities-fp32.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if sha256(source) != metadata["output_sha256"] or sha256(source) != original["source_fixture_sha256"]:
        raise ValueError("Frozen linear-operator fixture changed")
    values = load_file(str(source))
    names = ["prefill.layer.12.w13", "prefill.layer.12.w2"]
    fixtures = {entry["name"]: entry for entry in metadata["fixtures"]}
    originals = {record["name"]: record for record in original["records"]}
    records, outputs = [], {}
    with torch.inference_mode():
        for name in names:
            fixture = fixtures[name]
            value = values[name + ".input"].cuda()
            weight = values[fixture["weight_key"]].cuda()
            expected = values[name + ".expected"]
            for _ in range(3):
                actual = torch.nn.functional.linear(value, weight)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, with_stack=False) as profile:
                with torch.profiler.record_function(name):
                    actual = torch.nn.functional.linear(value, weight)
                torch.cuda.synchronize()
            trace_path = output / (name + ".json")
            profile.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            kernels = [{"name": entry.get("name"), "arguments": entry.get("args", {})}
                       for entry in trace.get("traceEvents", []) if entry.get("cat") == "kernel"]
            signature = lambda entries: [(item["name"], item["arguments"].get("grid"), item["arguments"].get("block")) for item in entries]
            candidate = actual.cpu()
            outputs[name + ".output"] = candidate
            records.append({"name": name, "input_shape": list(value.shape), "weight_shape": list(weight.shape),
                            "input_sha256": hashlib.sha256(values[name + ".input"].numpy().tobytes()).hexdigest(),
                            "weight_sha256": hashlib.sha256(values[fixture["weight_key"]].numpy().tobytes()).hexdigest(),
                            "warmup_calls": 3, "profiled_calls": 1, "kernels": kernels,
                            "matches_original_kernel_names_grids_blocks": signature(kernels) == signature(originals[name]["kernels"]),
                            "output_matches_frozen_bits": torch.equal(candidate.view(torch.int32), expected.view(torch.int32)),
                            "trace_sha256": sha256(trace_path)})
    save_file(outputs, str(output / "outputs.safetensors"))
    libraries = {}
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and any(part in path for part in ["libcublas", "libtorch_cuda", "libcudart"]):
            libraries[path] = sha256(path)
    report = {"environment": environment, "pid": os.getpid(), "logger_environment": expected_env,
              "source_fixture_sha256": sha256(source), "source_metadata_sha256": sha256(source.with_suffix(".json")),
              "original_profiler_sha256": sha256(original_path), "script_sha256": sha256(script_copy),
              "outputs_sha256": sha256(output / "outputs.safetensors"), "loaded_libraries": libraries,
              "cublas_package_version": importlib.metadata.version("nvidia-cublas"), "records": records,
              "qualification": "One supported logging replay of the original two prefill projections. No backend/algorithm override, descriptor manufacture, selector search or performance claim. Logs are finalized and hashed after process exit."}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
