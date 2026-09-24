#!/usr/bin/env python3
"""Freeze exact saved GPU/CPU/counterfactual exp2 arguments for later GPU replay.

This preparation is CPU-only. The source evidence is the independent tile oracle,
not native fused Flex intermediates; a later accepted fused capture is separate.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess

import numpy as np


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/bf16-exp2-arguments-v1"))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    if args.output.exists():
        raise ValueError("Refusing to overwrite preserved argument evidence")
    analysis_path = root / "research/bf16-graph/experiments/bf16_attention/analyze_crossings.py"
    report_path = root / "reference/bf16-attention-crossing-causality-v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert digest(analysis_path) == report["source_sha256"]["analysis"]
    spec = importlib.util.spec_from_file_location("saved_bf16_crossing_analysis", analysis_path)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    gpu_path = root / "artifacts/reference/bf16-attention-tiles.safetensors"
    gpu = analysis.tensors(gpu_path)
    metadata = json.loads(gpu_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert digest(gpu_path) == metadata["output_sha256"] == "1de7df7665034a58e2647c9ac840decad24e9aca11c81ff6d4e4101189440e23"
    arguments, records = set(), []
    inputs = {gpu_path.as_posix(): digest(gpu_path), report_path.as_posix(): digest(report_path), analysis_path.as_posix(): digest(analysis_path)}
    for backend in ["avx512", "scalar"]:
        cpu_path = root / f"artifacts/cpu/bf16-attention-tiles-{backend}.safetensors"
        prior = json.loads((root / f"reference/bf16-attention-tile-differences-{backend}.json").read_text(encoding="utf-8"))
        assert digest(cpu_path) == prior["candidate_sha256"]
        inputs[cpu_path.as_posix()] = digest(cpu_path)
        cpu = analysis.tensors(cpu_path)
        for probe in metadata["probes"]:
            for tile in probe["tiles"]:
                prefix = probe["name"] + f".tile{tile['index']}"
                for score_label, score_source in [("gpu", gpu), ("cpu", cpu)]:
                    for max_label, max_source in [("gpu", gpu), ("cpu", cpu)]:
                        values = (score_source[prefix + ".scores_log2"] - np.float32(max_source[prefix + ".maximum"][0])).astype(np.float32)
                        words = [analysis.bits(v) for v in values]
                        arguments.update(words)
                        records.append({"backend": backend, "probe": probe["name"], "tile": tile,
                                        "condition": score_label + "_score_" + max_label + "_max",
                                        "argument_bits": [f"{word:08x}" for word in words]})
    assert len(arguments) == report["rust_exp2_distinct_arguments"]
    executable = root / "artifacts/bf16-attention-crossings/exp2-probe.exe"
    assert digest(executable) == report["rust_exp2_binary_sha256"]
    request = "".join(f"{word:08x}\n" for word in sorted(arguments))
    completed = subprocess.run([str(executable.resolve())], input=request, text=True, capture_output=True, check=True)
    observed = {}
    for line in completed.stdout.splitlines():
        argument, answer, rounded = line.split()
        observed[int(argument, 16)] = {"argument_bits": argument, "rust_exp2_bits": answer, "rust_bf16_bits": rounded}
    assert set(observed) == arguments
    args.output.mkdir(parents=True)
    (args.output / "arguments-hex.txt").write_text(request, encoding="ascii")
    (args.output / "arguments-u32-le.bin").write_bytes(b"".join(struct.pack("<I", word) for word in sorted(arguments)))
    (args.output / "rust-results.txt").write_text(completed.stdout, encoding="ascii")
    target = args.output / "arguments.json"
    target.write_text(json.dumps({"records": records, "distinct_arguments": [observed[word] for word in sorted(arguments)]}, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "source_sha256": inputs, "script_sha256": digest(__file__),
                "rust_binary_sha256": digest(executable), "original_causality_report_sha256": digest(report_path),
                "distinct_arguments": len(arguments), "records": len(records), "gpu_execution": False,
                "outputs": {p.name: digest(p) for p in args.output.iterdir()},
                "qualification": "Exact input bits from the saved independent GPU/CPU tile oracle and all four score/max substitutions, plus the preserved native Windows Rust exp2 results. Native fused Flex arguments and actual GPU exp2 at these bits remain pending."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arguments": len(arguments), "records": len(records), "output": str(args.output), "gpu_execution": False}))


if __name__ == "__main__":
    main()
