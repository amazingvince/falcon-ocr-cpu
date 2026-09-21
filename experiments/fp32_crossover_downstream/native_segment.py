"""Native segment copied from the pinned exporter; checked before future use."""
import functools

from contract import STAGES as SHAPES, require


def run(torch, import_model, load_file, Tokenizer, model_dir, tokens, pos_t, pos_hw,
        c7, substitution, expected, report, tensors, stable):
    # The guarded fragments below retain the prior native expressions exactly.
    module = import_model(model_dir)
    for name in ("compiled_flex_attn_prefill", "compiled_flex_attn_decode"):
        setattr(module, name, functools.partial(getattr(module, name), kernel_options={"FLOAT32_PRECISION": "'ieee'"}))
    config = module.FalconOCRConfig.from_json_file(str(model_dir / "config.json"))
    model = module.FalconOCRForCausalLM(config)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True, assign=True)
    model = model.to(device="cuda:0", dtype=torch.float32).eval()
    model._ensure_device_buffers()
    report["effective_compiled_blocks"] = model._is_compiled
    require(model._is_compiled is False, "Native transformer blocks must remain uncompiled")
    pad = Tokenizer.from_file(str(model_dir / "tokenizer.json")).token_to_id("<|pad|>")
    require(type(pad) is int, "Missing original pad token")
    model._pad_token_id = pad
    padded = torch.full((1, 256), pad, dtype=torch.long, device="cuda:0")
    padded[:, :144] = tokens.to("cuda:0")
    mask = model.get_attention_mask(padded, max_len=256)
    mask.seq_lengths = (144, 144)
    temporal_positions = pos_t.to("cuda:0")[None]
    spatial_positions = pos_hw.to("cuda:0")[None]
    temporal = model.freqs_cis[temporal_positions]
    spatial = module.apply_golden_freqs_cis_to_visual_pos(model.freqs_cis_golden, spatial_positions)
    state = {"branch": None}

    def capture(key, value):
        name = state["branch"] + "." + key
        require(name not in tensors, "Duplicate stage capture")
        value = value.detach().squeeze(0).to("cpu").contiguous().clone()
        require(value.dtype == torch.float32 and list(value.shape) == SHAPES[key] and bool(torch.isfinite(value).all()), "Unexpected native stage shape/dtype/value")
        tensors[name] = value

    handles = []
    layer8, layer9 = model.layers["8"], model.layers["9"]
    for submodule, before, after in [
        (layer8.attention.wqkv, "layer.8.attention_norm", "layer.8.qkv"),
        (layer8.attention.wo, "layer.8.attention", "layer.8.wo"),
        (layer8.feed_forward.w13, "layer.8.ffn_norm", "layer.8.w13"),
        (layer8.feed_forward.w2, "layer.8.gate", "layer.8.w2"),
        (layer9.attention.wqkv, "layer.9.attention_norm", "layer.9.qkv")]:
        handles.append(submodule.register_forward_pre_hook(lambda m, args, key=before: capture(key, args[0])))
        handles.append(submodule.register_forward_hook(lambda m, args, out, key=after: capture(key, out)))
    handles.append(layer8.feed_forward.register_forward_pre_hook(lambda m, args: capture("layer.8.attention_residual", args[0])))
    original_qkv = layer8.attention._pre_attention_qkv
    original_rope = module.apply_3d_rotary_emb

    def qkv(*a, **kw):
        q, k, v = original_qkv(*a, **kw)
        capture("layer.8.v", v)
        return q, k, v

    def rope(*a, **kw):
        q, k = original_rope(*a, **kw)
        capture("layer.8.q", q)
        capture("layer.8.k", k)
        return q, k

    layer8.attention._pre_attention_qkv = qkv
    module.apply_3d_rotary_emb = rope
    try:
        with torch.inference_mode():
            for branch, input_state in (("control_c", c7), ("substitution_s", substitution)):
                require(branch == "control_c" or report["control_passed"] is True,
                        "No substitution before all control stages pass")
                stable()
                state["branch"] = branch
                x = input_state.to("cuda:0")[None]
                capture("input", x)
                cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)
                cache.set_pos_t(temporal_positions[:, -1:])
                report["native_layer8_calls"] += 1
                hidden = layer8(x, freqs_cis=temporal, freqs_cis_2d=spatial,
                                pos_hw=spatial_positions, attention_masks=mask, kv_cache=cache)
                capture("layer.8.hidden", hidden)
                report["native_layer9_qkv_calls"] += 1
                _, _, v = layer9.attention._pre_attention_qkv(hidden)
                capture("layer.9.v", v)
                require({k[len(branch) + 1:] for k in tensors if k.startswith(branch + ".")} == set(SHAPES), "Incomplete stage capture")
                if branch == "control_c":
                    require(set(expected) == set(SHAPES), "All 17 historical control stages required")
                    for key in SHAPES:
                        got, want = tensors[branch + "." + key], expected[key]
                        equal = got.shape == want.shape and torch.equal(got.view(torch.int32), want.view(torch.int32))
                        report["controls"].append({"stage": key, "bit_exact": equal,
                            "mismatched_elements": int((got.view(torch.int32) != want.view(torch.int32)).sum()) if got.shape == want.shape else None})
                    report["control_passed"] = len(report["controls"]) == 17 and all(c["bit_exact"] for c in report["controls"])
                    require(report["control_passed"], "Fresh F(C7) differs; stop before substitution, no retry")
                else:
                    report["substitution_executed"] = True
    finally:
        for handle in handles:
            handle.remove()
        layer8.attention._pre_attention_qkv = original_qkv
        module.apply_3d_rotary_emb = original_rope

