#!/usr/bin/env python3
"""Replay frozen BF16 sink failures with independent raw/LSE interventions.

Uses stdlib only and standalone Rust. No inference, GPU use, production changes,
tolerance changes, or performance measurement. Fails on stale failure records,
shape ambiguity, or inability to reproduce saved CPU scaled values exactly.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def independent_bf16_bits(bits):
    # Explicit discarded-half comparison, independent of Rust's rounding add.
    kept, discarded = divmod(bits, 65536)
    return kept + int(discarded > 32768 or (discarded == 32768 and kept % 2))


class Tensors:
    def __init__(self, path):
        raw = path.read_bytes()
        size = struct.unpack_from("<Q", raw)[0]
        self.header = json.loads(raw[8:8 + size])
        self.data = raw[8 + size:]

    def shape(self, key):
        return self.header[key]["shape"]

    def value(self, key, index):
        info = self.header[key]
        shape = info["shape"]
        if len(shape) != len(index) or any(i < 0 or i >= n for i, n in zip(index, shape)):
            raise ValueError((key, shape, index))
        flat = 0
        for i, n in zip(index, shape):
            flat = flat * n + i
        start, end = info["data_offsets"]
        if info["dtype"] == "F32":
            assert end - start == math.prod(shape) * 4
            return struct.unpack_from("<f", self.data, start + flat * 4)[0]
        if info["dtype"] == "BF16":
            assert end - start == math.prod(shape) * 2
            return f32_from_bits(struct.unpack_from("<H", self.data, start + flat * 2)[0] << 16)
        raise ValueError(info["dtype"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    policy_path = ROOT / "reference/bf16-local-contract-v1.json"
    details_path = ROOT / "reference/bf16-local-failure-details.json"
    policy, details = read_json(policy_path), read_json(details_path)
    assert sha(policy_path) == details["contract_sha256"]
    gpu_path = ROOT / "artifacts/reference/attention-operators-bf16.safetensors"
    bounds_path = ROOT / "artifacts/reference/bf16-local-contract-v1.safetensors"
    for path in [gpu_path, bounds_path]:
        assert sha(path) == policy["sources"][path.relative_to(ROOT).as_posix()]
    gpu, bounds = Tensors(gpu_path), Tensors(bounds_path)
    cpu_paths = {b: ROOT / f"artifacts/cpu/bf16-native-attention-{b}.safetensors"
                 for b in ["avx512", "scalar"]}
    candidates = {b: Tensors(p) for b, p in cpu_paths.items()}
    failures = [x for x in details["violating_elements"]
                if x["kind"] == "attention" and x["case"].endswith(".scaled")]
    assert failures
    inputs = [policy_path, details_path, gpu_path, bounds_path, *cpu_paths.values()]
    source_paths = [Path(__file__), Path(__file__).with_name("sink_probe.rs"),
                    ROOT / "src/bf16_attention.rs"]
    input_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    for path in source_paths:
        dest = out / "sources" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    compiler = subprocess.check_output(["rustc", "-vV"], text=True, env=dict(os.environ))
    exe = out / ("sink_probe.exe" if os.name == "nt" else "sink_probe")
    command = ["rustc", str(out / "sources/experiments/bf16_attention/sink_probe.rs"),
               "--edition=2021", "-C", "opt-level=3", "-o", str(exe)]
    build = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ))
    (out / "build.log").write_text(build.stdout + build.stderr, encoding="utf-8")
    build.check_returncode()
    heads = sorted({(x["backend"], x["case"].rsplit(".", 1)[0], *x["index"][:2])
                    for x in failures})
    requests, head_info = [], []
    variants = [("cpu_raw_cpu_lse", "cpu", "cpu"),
                ("gpu_raw_cpu_lse", "gpu", "cpu"),
                ("cpu_raw_gpu_lse", "cpu", "gpu"),
                ("gpu_raw_gpu_lse", "gpu", "gpu")]
    for backend, case, row, head in heads:
        cpu = candidates[backend]
        entry = next(x for x in policy["attention_cases"] if x["name"] == case)
        shape = gpu.shape(entry["raw"]["expected_key"])
        assert shape == cpu.shape(case + ".raw") == cpu.shape(case + ".scaled")
        rows, nheads, width = shape
        assert width == 64
        assert gpu.shape(case + ".lse") == cpu.shape(case + ".lse") == [nheads, rows]
        assert gpu.shape(case + ".sink_scale_f32") == [nheads, rows, 1]
        sink = gpu.value(case + ".sinks", [head])
        lse = {"cpu": cpu.value(case + ".lse", [head, row]),
               "gpu": gpu.value(case + ".lse", [head, row])}
        raw = {"cpu": [cpu.value(case + ".raw", [row, head, d]) for d in range(width)],
               "gpu": [gpu.value(entry["raw"]["expected_key"], [row, head, d]) for d in range(width)]}
        info = {"backend": backend, "case": case, "row": row, "head": head,
                "sink": sink, "lse": lse, "requests": {}}
        for label, raw_source, lse_source in variants:
            info["requests"][label] = len(requests)
            for value in raw[raw_source]:
                requests.append(f"{f32_bits(value):08x} {f32_bits(lse[lse_source]):08x} {f32_bits(sink):08x}")
        head_info.append(info)
    request_text = "\n".join(requests) + "\n"
    (out / "input.txt").write_text(request_text, encoding="ascii")
    run = subprocess.run([str(exe)], input=request_text, text=True, capture_output=True,
                         check=True, env=dict(os.environ))
    (out / "output.txt").write_text(run.stdout, encoding="ascii")
    outputs = [[int(s, 16) for s in line.split()] for line in run.stdout.splitlines()]
    assert len(outputs) == len(requests)
    assert all(len(x) == 3 and independent_bf16_bits(x[1]) == x[2] for x in outputs)
    records, head_records = [], []
    for info in head_info:
        backend, case, row, head = (info[k] for k in ["backend", "case", "row", "head"])
        cpu = candidates[backend]
        entry = next(x for x in policy["attention_cases"] if x["name"] == case)
        info["gpu_scale"] = gpu.value(case + ".sink_scale_f32", [head, row, 0])
        gpu_same_args_differences = {"scale_f32": 0, "precast_f32": 0, "scaled_bf16": 0}
        for dim in range(64):
            index = [row, head, dim]
            replay = {label: outputs[start + dim] for label, start in info["requests"].items()}
            actual = cpu.value(case + ".scaled", index)
            expected = gpu.value(entry["scaled"]["expected_key"], index)
            assert f32_from_bits(replay["cpu_raw_cpu_lse"][2] << 16) == actual
            same = replay["gpu_raw_gpu_lse"]
            gpu_same_args_differences["scale_f32"] += int(same[0] != f32_bits(info["gpu_scale"]))
            gpu_same_args_differences["precast_f32"] += int(same[1] != f32_bits(gpu.value(case + ".post_sink_before_output_cast_f32", index)))
            gpu_same_args_differences["scaled_bf16"] += int(f32_from_bits(same[2] << 16) != expected)
            failure = next((x for x in failures if x["backend"] == backend and x["case"] == case + ".scaled" and x["index"] == index), None)
            if failure is None:
                continue
            bound = bounds.value(entry["scaled"]["bound_key"], index)
            assert (expected, actual, bound) == (failure["gpu"], failure["cpu"], failure["bound"])
            assert abs(actual - expected) > bound
            record = {"backend": backend, "case": case, "index": index,
                      "gpu_raw": gpu.value(entry["raw"]["expected_key"], index),
                      "cpu_raw": cpu.value(case + ".raw", index),
                      "gpu_expected": expected, "saved_cpu": actual, "frozen_bound": bound,
                      "variants": {}}
            for label, bits in replay.items():
                value = f32_from_bits(bits[2] << 16)
                record["variants"][label] = {"scale": f32_from_bits(bits[0]),
                    "precast": f32_from_bits(bits[1]), "rounded": value,
                    "error": abs(value - expected), "passes_frozen_bound": abs(value - expected) <= bound}
            records.append(record)
        info.pop("requests")
        info["cpu_reproduced_exactly_count"] = 64
        info["gpu_same_argument_cpu_replay_difference_counts"] = gpu_same_args_differences
        head_records.append(info)
    assert len(records) == len(failures)
    assert input_hashes == {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    assert source_hashes == {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    result = {"schema_version": 1,
              "scope": "Frozen scaled-output failure heads, all 64 dimensions per selected head; CPU-only counterfactual replay, not model qualification",
              "gpu_execution": False, "bounds_changed": False, "production_changed": False,
              "inputs_sha256": input_hashes, "sources_sha256": source_hashes,
              "source_capture": "Sources preserved before compilation; sources and input artifacts unchanged after replay",
              "rustc": compiler, "build_command": command, "executable_sha256": sha(exe),
              "request_sha256": sha(out / "input.txt"), "output_sha256": sha(out / "output.txt"),
              "replay_count": len(outputs), "independent_rounding_checks": len(outputs),
              "heads": head_records, "scaled_failures": records,
              "limitations": ["Exact checks cover selected failure heads, not all attention elements.",
                  "Substituting raw values diagnoses the sink boundary; it does not identify or fix the upstream QK/PV cause.",
                  "This CPU expression is not a capture of GPU sigmoid internals.",
                  "No numerical gate or production backend is promoted by counterfactual replay."]}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"heads": len(heads), "scaled_failures": len(records),
        "replay_count": len(outputs), "gpu_raw_restores": sum(x["variants"]["gpu_raw_cpu_lse"]["passes_frozen_bound"] for x in records),
        "gpu_lse_restores": sum(x["variants"]["cpu_raw_gpu_lse"]["passes_frozen_bound"] for x in records),
        "report": str(out / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
