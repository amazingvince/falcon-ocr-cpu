#!/usr/bin/env python3
"""Expose independent GPU blockwise intermediates at fixed failing query/heads."""
import json
import pathlib
import torch
from safetensors.torch import load_file, save_file
from fetch_reference import sha256
from reference_preflight import preflight


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source = root / "artifacts/reference/attention-operators-bf16.safetensors"
    fixtures = load_file(str(source))
    result, probes = {}, []
    requests = {"prefill.layer.0": [(13, 1), (13, 2), (53, 8)], "prefill.layer.17": [(54, 10), (124, 11)]}
    with torch.inference_mode():
        for case, selections in requests.items():
            q, k, v = [fixtures[case + "." + key].cuda().transpose(0, 1) for key in ["q", "k", "v"]]
            allowed = fixtures[case + ".allowed"].cuda().bool()
            # All selected rows belong to the first observed128-query tile.
            q = q[:, :128]
            maximum = torch.full(q.shape[:2], -torch.inf, device=q.device)
            den = torch.zeros_like(maximum)
            acc = torch.zeros_like(q, dtype=torch.float32)
            schedule = [(128, False), (0, True), (64, True)]
            for tile, (start, full) in enumerate(schedule):
                stop = min(start + 64, k.shape[1])
                dot = torch.bmm(q, k[:, start:stop].transpose(1, 2), out_dtype=torch.float32)
                scaled = dot * 0.125
                if not full:
                    scaled = scaled.masked_fill(~allowed[:128, start:stop][None], -torch.inf)
                scores = scaled * 1.44269504
                new_max = torch.maximum(maximum, scores.amax(-1))
                safe_max = torch.where(new_max == -torch.inf, 0, new_max)
                alpha = torch.exp2(maximum - safe_max)
                p = torch.exp2(scores-safe_max[..., None])
                p_bf16 = p.bfloat16()
                pv = torch.bmm(p_bf16, v[:, start:stop], out_dtype=torch.float32)
                new_den = den * alpha + p.sum(-1)
                new_acc = acc * alpha[..., None] + pv
                for row, head in selections:
                    name = f"{case}.row{row}.head{head}"
                    key = name + f".tile{tile}"
                    values = {"qk": dot[head,row], "scaled_scores": scaled[head,row], "scores_log2": scores[head,row],
                              "exp2": p[head,row], "probabilities_bf16": p_bf16[head,row], "pv": pv[head,row],
                              "maximum_before": maximum[head,row].reshape(1), "maximum": new_max[head,row].reshape(1),
                              "alpha": alpha[head,row].reshape(1), "denominator_before": den[head,row].reshape(1),
                              "denominator": new_den[head,row].reshape(1), "accumulator_before": acc[head,row], "accumulator": new_acc[head,row]}
                    for suffix, value in values.items():
                        value = value.float()
                        if suffix in ["qk", "scaled_scores", "scores_log2", "exp2", "probabilities_bf16"] and len(value) < 64:
                            pad_value = -torch.inf if suffix in ["scaled_scores", "scores_log2"] else 0
                            value = torch.cat([value, torch.full((64-len(value),), pad_value, device=value.device)])
                        result[key + "." + suffix] = value.cpu().contiguous().clone()
                maximum, den, acc = new_max, new_den, new_acc
            raw = (acc/den[...,None]).bfloat16()
            lse = (maximum+torch.log2(den))*0.6931471805599453
            sinks = fixtures[case + ".sinks"].cuda()
            scaled_out = (raw*torch.sigmoid(lse-sinks[:,None])[...,None]).bfloat16()
            for row, head in selections:
                name = f"{case}.row{row}.head{head}"
                result[name + ".raw"] = raw[head,row].float().cpu().contiguous()
                result[name + ".scaled"] = scaled_out[head,row].float().cpu().contiguous()
                result[name + ".lse"] = lse[head,row].reshape(1).cpu().contiguous()
                expected = fixtures[case + ".sink_free_expected"][row,head]
                probes.append({"name":name, "case":case, "query_row":row, "head":head,
                               "tiles":[{"index": i, "key_start": start, "valid_keys": min(64,k.shape[1]-start), "full": full} for i,(start,full) in enumerate(schedule)],
                               "oracle_raw_matches_flex": torch.equal(raw[head,row].cpu(), expected)})
    target = root / "artifacts/reference/bf16-attention-tiles.safetensors"
    save_file(result, str(target))
    metadata = {"environment":environment,"fixture_sha256":sha256(source),"output_sha256":sha256(target),"probes":probes,
                "qualification":"Independent GPU blockwise oracle substages, not captures from inside the fused Flex kernel. Same observed full query-tile matrix shapes and sparse ordering as the validated oracle. All storage F32; probabilities_bf16/raw/scaled values preserve BF16 rounding. Tail tile computed at16valid keys as in the validated oracle, then diagnostic vectors padded to64 with zeros/masked infinities. No policy modification."}
    target.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == "__main__":
    main()
