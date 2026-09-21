"""One fixed native prefix/suffix run; no arithmetic alternatives or retries."""
import functools

from contract import BRANCHES, CONTROL_BRANCHES, START_LAYER, ENTRY_STATE, STAGES, require, require_next_branch


def run(torch, import_model, load_file, Tokenizer, model_dir, tokens, pos_t, pos_hw,
        entries, expected, report, tensors, stable):
    # This setup fragment is checked byte-for-byte after dedenting against the
    # preserved downstream implementation. Only the observation loop is new.
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
        if key not in STAGES[state["branch"]]:
            return
        name = state["branch"] + "." + key
        require(name not in tensors, "Duplicate stage capture")
        value = value.detach().squeeze(0).to("cpu").contiguous().clone()
        require(value.dtype == torch.float32 and list(value.shape) == STAGES[state["branch"]][key]
                and bool(torch.isfinite(value).all()), "Unexpected native stage shape/dtype/value")
        tensors[name] = value

    handles, original_qkv = [], {}
    original_rope = module.apply_3d_rotary_emb

    def observe_qkv(layer, original):
        def qkv(*a, **kw):
            q, k, v = original(*a, **kw)
            capture(f"layer.{layer}.v", v)
            return q, k, v
        return qkv

    def rope(*a, **kw):
        q, k = original_rope(*a, **kw)
        capture(f"layer.{state['layer']}.q", q)
        capture(f"layer.{state['layer']}.k", k)
        return q, k

    try:
        for i in range(10):
            layer = model.layers[str(i)]
            observations = [(layer.attention.wqkv, "attention_norm", "qkv")]
            if i < 9:
                observations += [(layer.attention.wo, "attention", "wo"),
                    (layer.feed_forward.w13, "ffn_norm", "w13"),
                    (layer.feed_forward.w2, "gate", "w2")]
                handles.append(layer.feed_forward.register_forward_pre_hook(
                    lambda m, args, i=i: capture(f"layer.{i}.attention_residual", args[0])))
                original_qkv[i] = layer.attention._pre_attention_qkv
                layer.attention._pre_attention_qkv = observe_qkv(i, original_qkv[i])
            for submodule, before, after in observations:
                handles.append(submodule.register_forward_pre_hook(
                    lambda m, args, key=f"layer.{i}.{before}": capture(key, args[0])))
                handles.append(submodule.register_forward_hook(
                    lambda m, args, out, key=f"layer.{i}.{after}": capture(key, out)))
        module.apply_3d_rotary_emb = rope
        layer9 = model.layers["9"]
        with torch.inference_mode():
            for branch in BRANCHES:
                require_next_branch(branch, report["completed_branches"], report["controls_passed"])
                stable()
                state["branch"] = branch
                x = entries[ENTRY_STATE[branch]].to("cuda:0")[None]
                capture("input", x)
                cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)
                cache.set_pos_t(temporal_positions[:, -1:])
                for layer_index in range(START_LAYER[branch], 9):
                    state["layer"] = layer_index
                    layer = model.layers[str(layer_index)]
                    report["native_block_calls"] += 1
                    hidden = layer(x, freqs_cis=temporal, freqs_cis_2d=spatial,
                                   pos_hw=spatial_positions, attention_masks=mask, kv_cache=cache)
                    capture(f"layer.{layer_index}.hidden", hidden)
                    x = hidden
                report["native_layer9_qkv_calls"] += 1
                _, _, v = layer9.attention._pre_attention_qkv(x)
                capture("layer.9.v", v)
                require({k[len(branch) + 1:] for k in tensors if k.startswith(branch + ".")}
                        == set(STAGES[branch]), "Incomplete branch capture")
                if branch in CONTROL_BRANCHES:
                    require(set(expected[branch]) == set(STAGES[branch]), "Missing expected control stage")
                    checks = []
                    for key in STAGES[branch]:
                        got, want = tensors[branch + "." + key], expected[branch][key]
                        equal = got.shape == want.shape and torch.equal(got.view(torch.int32), want.view(torch.int32))
                        checks.append({"stage": key, "bit_exact": equal,
                            "mismatched_elements": int((got.view(torch.int32) != want.view(torch.int32)).sum()) if got.shape == want.shape else None})
                    report["controls"].append({"branch": branch, "stages": checks})
                    require(all(c["bit_exact"] for c in checks), "Exact control failed; stop without retry: " + branch)
                    report["controls_passed"].append(branch)
                report["completed_branches"].append(branch)
    finally:
        for handle in handles:
            handle.remove()
        for i, original in original_qkv.items():
            model.layers[str(i)].attention._pre_attention_qkv = original
        module.apply_3d_rotary_emb = original_rope
