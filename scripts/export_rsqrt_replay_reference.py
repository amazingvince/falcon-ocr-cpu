#!/usr/bin/env python3
"""Replay saved CPU variance arguments through CUDA rsqrt; GPU variance is unknown."""
import json
import pathlib

import torch
from safetensors.torch import load_file, save_file

from fetch_reference import sha256
from reference_preflight import preflight

INFERRED = "artifacts/diagnostics/rms-scales-windows-v1/report.json"
INFERRED_SHA = "9dca1f5a3e236eae60bd9aabb4370dd11a8b70cad5ad05bdff43f111a541e848"
OBSERVED = "artifacts/reference/rms-rstd-fp32.safetensors"
OBSERVED_SHA = "be4b3c78fc81b9d04d8082a796d57827dda758f9faf6191d88c181d836f4a948"


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    inferred_path, observed_path = root / INFERRED, root / OBSERVED
    if sha256(inferred_path) != INFERRED_SHA or sha256(observed_path) != OBSERVED_SHA:
        raise ValueError("Frozen replay source changed")
    inferred = json.loads(inferred_path.read_text(encoding="utf-8"))
    observed_metadata = json.loads(observed_path.with_suffix(".json").read_text(encoding="utf-8"))
    if observed_metadata["output_sha256"] != OBSERVED_SHA or not observed_metadata["cpu_inferred_scale_comparison"]["all_fused_outputs_match_original_bits"]:
        raise ValueError("Observed rstd is not qualified against original fused outputs")
    observed = load_file(str(observed_path))
    output = root / "artifacts/reference/rms-rsqrt-replay-fp32"
    if output.exists():
        raise ValueError("Preserve the previous replay attempt")
    output.mkdir()
    for source in [pathlib.Path(__file__), inferred_path, observed_path.with_suffix(".json")]:
        (output / source.name).write_bytes(source.read_bytes())
    rows = inferred["rows"]
    variants = list(rows[0]["variants"])
    expected_bits = [int(observed[row["case"] + ".rstd"].view(torch.int32).reshape(-1)[row["row"]]) & 0xffffffff for row in rows]
    if len(rows) != 1444 or any(row["consistent_scale_bits"] != [bits] for row, bits in zip(rows, expected_bits)):
        raise ValueError("Observed/inferred row identity changed")
    tensors, results, details = {}, {}, []
    with torch.inference_mode():
        for variant in variants:
            bits = [row["variants"][variant]["variance_bits"] for row in rows]
            signed = [value if value < 2**31 else value - 2**32 for value in bits]
            argument = torch.tensor(signed, dtype=torch.int32).view(torch.float32)
            replay = torch.rsqrt(argument.cuda()).cpu()
            replay_bits = [value & 0xffffffff for value in replay.view(torch.int32).tolist()]
            tensors[variant + ".argument"] = argument
            tensors[variant + ".cuda_rsqrt"] = replay
            results[variant] = {
                "rows": len(rows),
                "cuda_replay_matches_observed_rstd": sum(a == b for a, b in zip(replay_bits, expected_bits)),
                "cpu_scale_matches_observed_rstd": sum(row["variants"][variant]["scale_bits"] == bits for row, bits in zip(rows, expected_bits)),
                "cuda_replay_matches_cpu_scale": sum(bits == row["variants"][variant]["scale_bits"] for bits, row in zip(replay_bits, rows)),
                "maximum_positive_f32_bit_distance_to_observed": max(abs(a - b) for a, b in zip(replay_bits, expected_bits)),
            }
            for row, arg, replay_bit, expected in zip(rows, bits, replay_bits, expected_bits):
                details.append({"case": row["case"], "row": row["row"], "variant": variant,
                                "argument_bits": arg, "cuda_rsqrt_bits": replay_bit,
                                "observed_rstd_bits": expected, "cpu_scale_bits": row["variants"][variant]["scale_bits"]})
    target = output / "tensors.safetensors"
    save_file(tensors, str(target))
    report = {"schema_version": 1, "environment": environment, "source_inferred_sha256": INFERRED_SHA,
              "source_observed_sha256": OBSERVED_SHA, "source_observed_metadata_sha256": sha256(observed_path.with_suffix(".json")),
              "script_sha256": sha256(output / pathlib.Path(__file__).name), "output_sha256": sha256(target),
              "variants": results, "rows": details,
              "qualification": "Observed CUDA torch.rsqrt applied to exact saved CPU variance-plus-epsilon arguments. Equality establishes argument/intrinsic compatibility with observed fused rstd, not the unobserved CUDA variance or its reduction order. No production arithmetic or bounds changed."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
