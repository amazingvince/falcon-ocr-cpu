#!/usr/bin/env python3
"""Replay exact argument bits with CUDA ex2.approx.ftz and torch.exp2.

An accepted fused observation may contribute arguments and native expected
results. The independent oracle/CPU/counterfactual inputs are always retained.
"""
import argparse
import json
import pathlib
import struct

import numpy as np
import torch
import triton
import triton.language as tl
from safetensors.torch import load_file, save_file

from fetch_reference import sha256
from reference_preflight import preflight


@triton.jit
def native_exp2(Input, Output, Rounded, Count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Input + index, index < Count, other=0.0)
    result = tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[value],
                                     dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Output + index, result, index < Count)
    tl.store(Rounded + index, result.to(tl.bfloat16), index < Count)


def words(tensor):
    integer = torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32
    mask = 0xffff if integer == torch.int16 else 0xffffffff
    return [int(x) & mask for x in tensor.detach().cpu().contiguous().view(integer).reshape(-1).tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arguments", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-exp2-arguments-v1"))
    parser.add_argument("--fused", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-fused-substages-v2"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-exp2-gpu-replay-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior replay evidence; output already exists")
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    frozen = json.loads((args.arguments / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in frozen["outputs"].items():
        if sha256(args.arguments / name) != digest:
            raise ValueError("Frozen argument evidence changed: " + name)
    payload = json.loads((args.arguments / "arguments.json").read_text(encoding="utf-8"))
    rust = {int(row["argument_bits"], 16): row for row in payload["distinct_arguments"]}
    all_words = set(rust)
    fused_cases = []
    fused_evidence = {"status": "no_accepted_fused_capture", "native_intermediate_claim": False}
    report_path = args.fused / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fused_evidence["report_sha256"] = sha256(report_path)
        fused_evidence["observation_status"] = report["status"]
        if report.get("accepted") is True:
            if not report["cases"] or not all(case["accepted"] and all(c["equal"] for c in case["checks"].values()) for case in report["cases"]):
                raise ValueError("Fused observation acceptance is inconsistent")
            capture = args.fused / "capture.safetensors"
            if sha256(capture) != report["capture_sha256"]:
                raise ValueError("Accepted fused observation bytes changed")
            values = load_file(str(capture))
            for name, value in values.items():
                expected_name = None
                if name.endswith(".exp2_argument_f32"):
                    expected_name = name.removesuffix(".exp2_argument_f32") + ".exp2_result_f32"
                elif name.endswith(".alpha_argument_f32"):
                    expected_name = name.removesuffix(".alpha_argument_f32") + ".alpha_result_f32"
                if expected_name is None:
                    continue
                arguments = words(value)
                all_words.update(arguments)
                fused_cases.append({"name": name, "arguments": arguments, "expected": words(values[expected_name])})
            fused_evidence.update(status="accepted_fused_capture_included", native_intermediate_claim=True, capture_sha256=sha256(capture))
    ordered = sorted(all_words)
    args.output.mkdir(parents=True)
    script_copy = args.output / pathlib.Path(__file__).name
    script_copy.write_bytes(pathlib.Path(__file__).read_bytes())
    inputs = torch.from_numpy(np.asarray(ordered, dtype=np.uint32).view(np.float32)).cuda()
    outputs = torch.empty_like(inputs)
    rounded = torch.empty(inputs.shape, dtype=torch.bfloat16, device=inputs.device)
    with torch.inference_mode():
        binary = native_exp2[(triton.cdiv(len(ordered), 256),)](inputs, outputs, rounded, len(ordered), 256)
        torch_outputs = torch.exp2(inputs)
        torch_rounded = torch_outputs.to(torch.bfloat16)
        torch.cuda.synchronize()
    native_bits, native_bf16 = words(outputs), words(rounded)
    torch_bits, torch_bf16 = words(torch_outputs), words(torch_rounded)
    lookup = {word: index for index, word in enumerate(ordered)}
    rows = []
    for index, argument in enumerate(ordered):
        row = {"argument_bits": f"{argument:08x}", "native_ex2_bits": f"{native_bits[index]:08x}",
               "native_bf16_bits": f"{native_bf16[index]:04x}", "torch_exp2_bits": f"{torch_bits[index]:08x}",
               "torch_bf16_bits": f"{torch_bf16[index]:04x}"}
        if argument in rust:
            row.update({k: v for k, v in rust[argument].items() if k != "argument_bits"})
        rows.append(row)
    fused_checks = []
    for case in fused_cases:
        differing = [index for index, (argument, expected) in enumerate(zip(case["arguments"], case["expected"]))
                     if native_bits[lookup[argument]] != expected]
        fused_checks.append({"name": case["name"], "elements": len(case["arguments"]), "different_indices": differing})
    (args.output / "arguments-u32-le.bin").write_bytes(b"".join(struct.pack("<I", word) for word in ordered))
    save_file({"arguments": inputs.cpu(), "native_ex2": outputs.cpu(), "native_bf16": rounded.cpu(),
               "torch_exp2": torch_outputs.cpu(), "torch_bf16": torch_rounded.cpu()}, str(args.output / "outputs.safetensors"))
    artifacts = {}
    for name in ["ptx", "cubin", "ttir", "ttgir", "llir"]:
        value = binary.asm.get(name)
        if value is not None:
            path = args.output / ("native-exp2." + name)
            path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
            artifacts[name] = sha256(path)
    if "ex2.approx.ftz.f32" not in binary.asm["ptx"] or "cvt.rn.bf16" not in binary.asm["ptx"]:
        raise ValueError("Compiled replay lacks the requested native exp2/cast instructions")
    result = {"schema_version": 1, "environment": environment, "script_sha256": sha256(script_copy),
              "argument_manifest_sha256": sha256(args.arguments / "manifest.json"), "fused_evidence": fused_evidence,
              "distinct_arguments": len(ordered), "original_rust_arguments": len(rust), "rows": rows,
              "native_vs_torch_f32_differences": sum(a != b for a, b in zip(native_bits, torch_bits)),
              "native_vs_torch_bf16_differences": sum(a != b for a, b in zip(native_bf16, torch_bf16)),
              "native_vs_rust_f32_differences": sum(row["native_ex2_bits"] != row["rust_exp2_bits"] for row in rows if "rust_exp2_bits" in row),
              "native_vs_rust_bf16_differences": sum(row["native_bf16_bits"] != row["rust_bf16_bits"] for row in rows if "rust_bf16_bits" in row),
              "fused_same_argument_checks": fused_checks, "compiled_artifacts": artifacts,
              "outputs_sha256": sha256(args.output / "outputs.safetensors"),
              "qualification": "Same exact FP32 arguments replayed on the GPU, including saved CPU/counterfactual arguments. Native fused intermediates are included only if the separate complete raw/LSE observation acceptance passed. This replay is not a replacement for that acceptance and changes no arithmetic or bounds."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["distinct_arguments", "original_rust_arguments", "native_vs_torch_f32_differences", "native_vs_torch_bf16_differences", "native_vs_rust_f32_differences", "native_vs_rust_bf16_differences", "fused_evidence"]}, indent=2))


if __name__ == "__main__":
    main()
