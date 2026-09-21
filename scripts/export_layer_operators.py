#!/usr/bin/env python3
"""Capture real strict-FP32 layer boundaries on fixed GPU reference inputs."""
import functools
import json
import pathlib
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from export_reference import import_model
from fetch_reference import sha256
from reference_preflight import preflight


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = root / "artifacts/reference/smoke-fp32/trace.safetensors"
    trace = load_file(str(source))
    module = import_model(root / "artifacts/model")
    config = module.FalconOCRConfig.from_json_file(str(root / "artifacts/model/config.json"))
    model = module.FalconOCRForCausalLM(config)
    model.load_state_dict(load_file(str(root / "artifacts/model/model.safetensors")), strict=True, assign=True)
    model = model.cuda().float().eval()
    model._ensure_device_buffers()
    pad = Tokenizer.from_file(str(root / "artifacts/model/tokenizer.json")).token_to_id("<|pad|>")
    model._pad_token_id = pad
    tokens = trace["tokens"].cuda()
    padded = torch.full((1, 256), pad, dtype=torch.long, device="cuda:0")
    padded[:, :len(tokens)] = tokens
    mask = model.get_attention_mask(padded, max_len=256)
    pos_t, pos_hw = trace["pos_t"].cuda()[None], trace["pos_hw"].cuda()[None]
    spatial = module.apply_golden_freqs_cis_to_visual_pos(model.freqs_cis_golden, pos_hw)
    result, records = {}, []
    state = {"name": "", "live": {}}

    def capture(suffix, value):
        state["live"][suffix] = value
        result[state["name"] + "." + suffix] = value.detach().squeeze(0).cpu().contiguous().clone()

    original_rope = module.apply_3d_rotary_emb
    def rope(*args, **kwargs):
        output = original_rope(*args, **kwargs)
        capture("post_rope.q", output[0])
        capture("post_rope.k", output[1])
        return output
    module.apply_3d_rotary_emb = rope
    for function in ["compiled_flex_attn_prefill", "compiled_flex_attn_decode"]:
        original = getattr(module, function)
        def flex(q, k, v, original=original, **kwargs):
            raw, aux = original(q, k, v, kernel_options={"FLOAT32_PRECISION": "'ieee'"}, **kwargs)
            capture("attention.q", q.transpose(1, 2))
            capture("attention.k", k.transpose(1, 2))
            capture("attention.v", v.transpose(1, 2))
            capture("attention.raw", raw.transpose(1, 2))
            capture("attention.lse", aux.lse)
            return raw, aux
        setattr(module, function, flex)
    with torch.inference_mode():
        for step, layer_id in [(None, i) for i in [9, 12, 13, 17, 18]] + [(2, 21), (6, 19)]:
            prefix = "" if step is None else f"decode.{step}."
            name = (prefix or "prefill.") + f"layer.{layer_id}"
            state.update(name=name, live={})
            x = trace[prefix + f"layer.{layer_id-1}.hidden"].cuda()[None]
            capture("input", x)
            cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)
            cache.set_pos_t(pos_t[:, -1:])
            offset = 0 if step is None else len(tokens) + step
            if step is not None:
                cache.kv_cache = torch.empty(cache.kv_shape, device="cuda:0", dtype=torch.float32)
                for index, kind in enumerate(["k", "v"]):
                    past = torch.cat([trace[f"layer.{layer_id}.{kind}"]] + [trace[f"decode.{j}.layer.{layer_id}.{kind}"] for j in range(step)])
                    cache.kv_cache[layer_id, index, 0, :, :offset] = past.cuda().transpose(0, 1)
                cache.pos = offset
                this_mask = mask[:, :, offset // mask.BLOCK_SIZE[0]]
                this_mask.seq_lengths = (1, offset + 1)
                this_mask.mask_mod = module.offset_mask_mod(mask.mask_mod, offset)
                temporal = model.freqs_cis[pos_t[:, -1:] + step + 1]
            else:
                this_mask = mask
                this_mask.seq_lengths = (len(tokens), len(tokens))
                temporal = model.freqs_cis[pos_t]
            capture("temporal_cis_real", torch.view_as_real(temporal))
            if step is None:
                result[name + ".spatial_cis_real"] = torch.view_as_real(spatial).cpu().contiguous()
            layer = model.layers[str(layer_id)]
            handles = []
            for submodule, input_key, output_key in [
                (layer.attention.wqkv, "attention_norm.expected", "qkv.expected"),
                (layer.attention.wo, "attention.scaled", "attention_projection.expected"),
                (layer.feed_forward.w13, "ffn_norm.expected", "w13.expected"),
                (layer.feed_forward.w2, "gate.expected", "w2.expected")]:
                handles.append(submodule.register_forward_pre_hook(lambda mod, inp, key=input_key: capture(key, inp[0])))
                handles.append(submodule.register_forward_hook(lambda mod, inp, out, key=output_key: capture(key, out)))
            handles.append(layer.feed_forward.register_forward_pre_hook(lambda mod, inp: capture("attention_residual.expected", inp[0])))
            original_qkv = layer.attention._pre_attention_qkv
            def qkv(value):
                q, k, v = original_qkv(value)
                capture("q_norm.expected", q)
                capture("k_norm_repeated.expected", k)
                return q, k, v
            layer.attention._pre_attention_qkv = qkv
            out = layer(x, freqs_cis=temporal, freqs_cis_2d=spatial if step is None else None,
                        pos_hw=pos_hw if step is None else None, attention_masks=this_mask, kv_cache=cache)
            capture("hidden.expected", out)
            for handle in handles:
                handle.remove()
            layer.attention._pre_attention_qkv = original_qkv
            expected = trace[prefix + f"layer.{layer_id}.hidden"]
            observed = out[0].cpu()
            errors = (observed-expected).abs()
            record = {"name": name, "layer": layer_id, "decode_step": step, "query_offset": offset,
                      "source_hidden_bit_exact": torch.equal(expected, observed), "source_hidden_max_abs": float(errors.max()),
                      "rows": x.shape[1], "weight_prefix": f"layers.{layer_id}.",
                      "norm_epsilon": torch.finfo(torch.float32).eps,
                      "attention_sink_weight_key": f"layers.{layer_id}.attention.sinks"}
            records.append(record)
            print(json.dumps(record), flush=True)
    result["tokens"] = trace["tokens"]
    result["pos_hw"] = trace["pos_hw"]
    target = root / "artifacts/reference/layer-operators-fp32.safetensors"
    save_file(result, str(target))
    metadata = {"environment": environment, "source_trace_sha256": sha256(source), "output_sha256": sha256(target),
                "cases": records, "all_source_hidden_bit_exact": all(r["source_hidden_bit_exact"] for r in records),
                "schema": "Each <case>.input is the fixed original previous-layer hidden; substage tensors come from actual module hooks, not reconstructed CPU arithmetic. Attention K/V include the fixed original reference cache for decode. Shared weights remain in the pinned checkpoint.",
                "qualification": "Equal-input strict FP32 block/substage references for locating drift; no frozen tolerance change."}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(target), "bytes": target.stat().st_size, "sha256": metadata["output_sha256"], "all_source_hidden_bit_exact": metadata["all_source_hidden_bit_exact"]}), flush=True)


if __name__ == "__main__":
    main()
