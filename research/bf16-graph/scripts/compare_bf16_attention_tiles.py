#!/usr/bin/env python3
"""Locate equal-input BF16 tile differences without changing acceptance gates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/reference/bf16-attention-tiles.safetensors"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    expected, actual = load_file(str(args.reference)), load_file(str(args.candidate))
    meta = json.loads(args.reference.with_suffix(".json").read_text())
    operands = load_file("artifacts/reference/attention-operators-bf16.safetensors")
    records = []
    for probe in meta["probes"]:
        prefix = probe["name"]
        case, row, head = probe["case"], probe["query_row"], probe["head"]
        q = operands[case + ".q"][row, head].double()
        k = operands[case + ".k"][:, head].double()
        tiles = []
        for tile in probe["tiles"]:
            stem = prefix + ".tile" + str(tile["index"])
            stages = {}
            for stage in ["qk", "scores_log2", "exp2", "probabilities_bf16", "pv", "maximum", "denominator", "accumulator", "alpha"]:
                a, b = expected[stem + "." + stage].flatten(), actual[stem + "." + stage].flatten()
                assert a.shape == b.shape, (stem, stage, a.shape, b.shape)
                different = (a != b).nonzero().flatten()
                delta = (a.double() - b.double()).abs()
                delta[torch.isinf(a) & (a == b)] = 0
                stages[stage] = {"different_elements": len(different), "max_abs": delta.max().item(),
                                 "different_indices": different.tolist()}
            changes = []
            gpu_p, cpu_p = expected[stem + ".probabilities_bf16"].flatten(), actual[stem + ".probabilities_bf16"].flatten()
            for index in (gpu_p != cpu_p).nonzero().flatten().tolist():
                key = tile["key_start"] + index
                exact_qk = (q * k[key]).sum().item()
                changes.append({"key": key, "tile_lane": index, "exact_f64_qk": exact_qk,
                    **{label + "_" + stage: tensors[stem + "." + stage].flatten()[index].item()
                       for label, tensors in [("gpu", expected), ("cpu", actual)]
                       for stage in ["qk", "scores_log2", "exp2", "probabilities_bf16"]}})
            tiles.append({**tile, "stages": stages, "probability_rounding_differences": changes})
        records.append({"name": prefix, "tiles": tiles})
    report = {"candidate_sha256": digest(args.candidate), "reference_sha256": digest(args.reference),
              "script_sha256": digest(__file__), "bounds_changed": False,
              "interpretation": "Diagnostic equality and exact BF16-operand F64 QK only. This does not redefine or pass any frozen gate.", "probes": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{ "name": p["name"], "probability_changes": sum(len(t["probability_rounding_differences"]) for t in p["tiles"])} for p in records], indent=2))


if __name__ == "__main__":
    main()
