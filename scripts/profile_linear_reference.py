#!/usr/bin/env python3
"""Bounded CUDA kernel identity capture for existing equal-input FP32 projections."""
import json
import pathlib

import torch
from safetensors.torch import load_file

from fetch_reference import sha256
from reference_preflight import preflight


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source = pathlib.Path("artifacts/reference/linear-operators.safetensors")
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(source) != metadata["output_sha256"]:
        raise ValueError("Frozen linear-operator fixture changed")
    values = load_file(str(source))
    names = ["prefill.layer.12.w13", "prefill.layer.12.w2", "decode.2.layer.12.w13", "decode.2.layer.12.w2"]
    fixtures = {entry["name"]: entry for entry in metadata["fixtures"]}
    output = pathlib.Path("artifacts/reference/linear-profiler-fp32")
    report_path = pathlib.Path("reference/linear-cuda-kernel-identities-fp32.json")
    if output.exists() or report_path.exists():
        raise ValueError("Profiler output already exists; preserve previous attempts")
    output.mkdir(parents=True)
    script_copy = output / pathlib.Path(__file__).name
    script_copy.write_bytes(pathlib.Path(__file__).read_bytes())
    records = []
    with torch.inference_mode():
        for name in names:
            fixture = fixtures[name]
            value = values[name + ".input"].cuda()
            weight = values[fixture["weight_key"]].cuda()
            expected = values[name + ".expected"]
            for _ in range(3):
                actual = torch.nn.functional.linear(value, weight)
            torch.cuda.synchronize()
            record = {"name": name, "input_shape": list(value.shape), "weight_shape": list(weight.shape),
                      "expected_sha256": sha256(source), "profiled_calls": 1, "unprofiled_warmup_calls": 3}
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, with_stack=False) as profile:
                    with torch.profiler.record_function(name):
                        actual = torch.nn.functional.linear(value, weight)
                    torch.cuda.synchronize()
                trace_path = output / (name + ".json")
                profile.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                kernels = [entry for entry in trace.get("traceEvents", []) if entry.get("cat") == "kernel"]
                record["trace_path"] = trace_path.as_posix()
                record["trace_sha256"] = sha256(trace_path)
                record["kernels"] = [{"name": event.get("name"), "arguments": event.get("args", {})} for event in kernels]
                record["status"] = "kernel_identity_captured" if kernels else "cuda_kernel_events_unavailable"
            except Exception as error:
                record["status"] = "profiler_unavailable"
                record["error"] = type(error).__name__ + ": " + str(error)
            candidate = actual.cpu()
            record["output_matches_saved_gpu_exactly"] = torch.equal(candidate, expected)
            record["max_output_difference"] = float((candidate - expected).abs().max())
            records.append(record)
    report = {"schema_version": 1, "environment": environment, "source_fixture_sha256": sha256(source),
              "source_metadata_sha256": sha256(source.with_suffix(".json")), "script_sha256": sha256(script_copy),
              "precision": "fp32", "tf32": False, "records": records,
              "split_k_algorithm_id": None,
              "qualification": "Observed kernel names and launch metadata from bounded same-input CUDA calls. No split-K algorithm ID is inferred from names, no CPU selector search or tolerance change, and profiler timings are not a performance benchmark."}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
