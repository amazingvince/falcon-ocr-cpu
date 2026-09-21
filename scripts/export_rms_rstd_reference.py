#!/usr/bin/env python3
"""Capture actual fused CUDA RMSNorm inverse standard deviations, no guessed variance."""
import json
import pathlib
import torch
from safetensors.torch import load_file, save_file
from fetch_reference import sha256
from reference_preflight import preflight

INFERRED_REPORT = "artifacts/diagnostics/rms-scales-windows-v1/report.json"
INFERRED_REPORT_SHA256 = "9dca1f5a3e236eae60bd9aabb4370dd11a8b70cad5ad05bdff43f111a541e848"


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    source = root / "artifacts/reference/layer-operators-fp32.safetensors"
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(source) != metadata["output_sha256"]:
        raise ValueError("Frozen layer-operator fixture changed")
    target = root / "artifacts/reference/rms-rstd-fp32.safetensors"
    if target.exists() or target.with_suffix(".json").exists():
        raise ValueError("RMS rstd output already exists; preserve previous attempts")
    source_copy = target.with_suffix(".source.py")
    if source_copy.exists():
        raise ValueError("RMS rstd source archive already exists")
    source_copy.write_bytes(pathlib.Path(__file__).read_bytes())
    inferred_path = root / INFERRED_REPORT
    if sha256(inferred_path) != INFERRED_REPORT_SHA256:
        raise ValueError("Frozen CPU scale-inference report changed")
    inferred = json.loads(inferred_path.read_text(encoding="utf-8"))
    inferred_copy = target.with_suffix(".inferred-scales.json")
    inferred_copy.write_bytes(inferred_path.read_bytes())
    inferred_source = {name.replace("\\", "/"): value for name, value in inferred["input_sha256"].items()}
    if inferred_source["artifacts/reference/layer-operators-fp32.safetensors"] != sha256(source):
        raise ValueError("CPU scale inference uses a different input fixture")
    values = load_file(str(source))
    result, records = {}, []
    eps = torch.finfo(torch.float32).eps
    with torch.inference_mode():
        for case in metadata["cases"]:
            for kind, input_suffix, expected_suffix in [("attention_norm", "input", "attention_norm.expected"),
                                                        ("ffn_norm", "attention_residual.expected", "ffn_norm.expected")]:
                name = case["name"] + "." + kind
                value = values[case["name"] + "." + input_suffix].cuda()
                output, rstd = torch.ops.aten._fused_rms_norm(value, [value.shape[-1]], None, eps)
                expected = values[case["name"] + "." + expected_suffix]
                result[name + ".input"] = value.cpu().contiguous()
                result[name + ".expected"] = output.cpu().contiguous()
                result[name + ".rstd"] = rstd.cpu().contiguous()
                records.append({"name": name, "rstd_shape": list(rstd.shape), "rstd_dtype": str(rstd.dtype),
                                "output_matches_original_layer_capture": torch.equal(output.cpu().view(torch.int32), expected.view(torch.int32)),
                                "output_max_error": float((output.cpu()-expected).abs().max())})
    save_file(result, str(target))
    comparison = {"inferred_report_sha256": sha256(inferred_copy),
                  "all_fused_outputs_match_original_bits": all(record["output_matches_original_layer_capture"] for record in records),
                  "qualification": "CPU consistent_scale_bits are inferred from output rounding; rstd tensors are newly observed CUDA results. Compare them only if all actual fused outputs equal the original layer capture bit for bit."}
    if comparison["all_fused_outputs_match_original_bits"]:
        inferred_rows = {(row["case"], row["row"]): row for row in inferred["rows"]}
        if len(inferred_rows) != len(inferred["rows"]):
            raise ValueError("Duplicate inferred scale rows")
        seen, comparisons = set(), []
        for record in records:
            name = record["name"]
            scales = result[name + ".rstd"]
            if scales.dtype != torch.float32:
                raise ValueError("Observed rstd is not FP32")
            for row_index, bits in enumerate(scales.contiguous().view(torch.int32).reshape(-1).tolist()):
                row_key = (name, row_index)
                seen.add(row_key)
                row = inferred_rows[row_key]
                candidates = row["consistent_scale_bits"]
                if len(candidates) != 1:
                    raise ValueError("CPU inference did not establish one consistent scale")
                observed = bits & 0xffffffff
                comparisons.append({"case": name, "row": row_index, "observed_rstd_bits": observed,
                                    "inferred_consistent_scale_bits": candidates[0], "exact": observed == candidates[0],
                                    "variant_scale_matches": {variant: value["scale_bits"] == observed for variant, value in row["variants"].items()}})
        if seen != set(inferred_rows):
            raise ValueError("Observed/inferred row sets differ")
        comparison.update(status="compared_observed_rstd_to_inferred_scales", rows=len(comparisons),
                          exact_rows=sum(row["exact"] for row in comparisons),
                          differences=[row for row in comparisons if not row["exact"]], row_comparisons=comparisons)
    else:
        comparison["status"] = "not_compared_fused_outputs_differ"
    report = {"environment": environment, "source_sha256": sha256(source), "output_sha256": sha256(target),
              "source_metadata_sha256": sha256(source.with_suffix(".json")), "script_sha256": sha256(source_copy),
              "operator_schema": str(torch.ops.aten._fused_rms_norm.default._schema), "epsilon": eps, "cases": records,
              "cpu_inferred_scale_comparison": comparison,
              "qualification": "Actual CUDA fused RMSNorm output and rstd; variance is not exported by this operator and is not inferred by inverting a rounded reciprocal square root."}
    target.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cpu_inferred_scale_comparison"}
                     | {"cpu_inferred_scale_comparison": {k: v for k, v in comparison.items() if k != "row_comparisons"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
