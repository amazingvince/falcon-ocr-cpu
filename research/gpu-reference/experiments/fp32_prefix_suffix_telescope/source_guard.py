"""Bounded text/AST guards, not a machine-code or universal numerical proof."""
import ast
import hashlib
import textwrap

from contract import ROOT, HERE, PRIOR, PRESERVED_SOURCE_PINS, require


def fragment(source, first, last):
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == first]
    ends = [i for i, line in enumerate(lines) if line.strip() == last]
    require(len(starts) == len(ends) == 1 and starts[0] <= ends[0], "Ambiguous source anchors")
    return textwrap.dedent("\n".join(lines[starts[0]:ends[0] + 1]))


def assignment_call(source, target):
    nodes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and ast.unparse(n.targets[0]) == target
             and isinstance(n.value, ast.Call)]
    require(len(nodes) == 1, "Missing/ambiguous native call")
    return nodes[0].value


class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id={"layer8": "layer", "hidden": "x"}.get(node.id, node.id),
                                          ctx=node.ctx), node)


def verify_text(original, candidate):
    require(fragment(original, "module = import_model(model_dir)", 'state = {"branch": None}')
            == fragment(candidate, "module = import_model(model_dir)", 'state = {"branch": None}'),
            "Native model/mask/position/precision setup changed")
    require(ast.dump(Rename().visit(assignment_call(original, "hidden")))
            == ast.dump(assignment_call(candidate, "hidden")), "Native full-block call changed")
    require(ast.dump(Rename().visit(assignment_call(original, "(_, _, v)")))
            == ast.dump(assignment_call(candidate, "(_, _, v)")), "Native complete layer9 pre-QKV call changed")
    expected = '''for layer_index in range(START_LAYER[branch], 9):
    state["layer"] = layer_index
    layer = model.layers[str(layer_index)]
    report["native_block_calls"] += 1
    hidden = layer(x, freqs_cis=temporal, freqs_cis_2d=spatial,
                   pos_hw=spatial_positions, attention_masks=mask, kv_cache=cache)
    capture(f"layer.{layer_index}.hidden", hidden)
    x = hidden'''
    got = fragment(candidate, "for layer_index in range(START_LAYER[branch], 9):", "x = hidden")
    require(ast.dump(ast.parse(got)) == ast.dump(ast.parse(expected)), "Native suffix recurrence changed")
    cache = fragment(original, "cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)",
                     "cache.set_pos_t(temporal_positions[:, -1:])")
    require(cache == fragment(candidate, "cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)",
                             "cache.set_pos_t(temporal_positions[:, -1:])"), "Fresh native cache setup changed")
    expected_observers = '''def qkv(*a, **kw):
    q, k, v = original(*a, **kw)
    capture(f"layer.{layer}.v", v)
    return q, k, v
def rope(*a, **kw):
    q, k = original_rope(*a, **kw)
    capture(f"layer.{state['layer']}.q", q)
    capture(f"layer.{state['layer']}.k", k)
    return q, k'''
    tree = ast.parse(candidate)
    for expected_observer in ast.parse(expected_observers).body:
        found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == expected_observer.name]
        require(len(found) == 1 and ast.dump(found[0]) == ast.dump(expected_observer),
                "QKV/RoPE observer does not pass through native outputs")
    require('for branch in BRANCHES:\n                require_next_branch(branch, report["completed_branches"], report["controls_passed"])' in candidate,
            "Ordered control gate missing")
    require('x = entries[ENTRY_STATE[branch]].to("cuda:0")[None]' in candidate, "Entry sliced/transformed")
    return {"model_setup_exact": True, "native_block_call_exact_modulo_local_name": True,
            "native_layer9_qkv_call_exact_modulo_local_name": True, "suffix_recurrence_guarded": True,
            "fresh_cache_exact": True, "observers_pass_through": True,
            "limits": "Source guards and actual finite controls do not prove identical emitted kernels or allocation/observation histories."}


def verify_sources():
    original = {}
    for path, expected in PRESERVED_SOURCE_PINS.items():
        raw = (ROOT / path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "Preserved downstream source changed")
        original[path] = raw
    require((ROOT / HERE / "legacy_contract.py").read_bytes() == original[PRIOR + "contract.py"],
            "Copied historical contract changed")
    expected_loader = original[PRIOR + "evidence.py"].replace(b"from contract import (", b"from legacy_contract import (")
    require((ROOT / HERE / "legacy_evidence.py").read_bytes().replace(b"\r\n", b"\n")
            == expected_loader.replace(b"\r\n", b"\n"), "Historical loader differs beyond local import")
    current = (ROOT / HERE / "native_segment.py").read_text(encoding="utf-8")
    proof = verify_text(original[PRIOR + "native_segment.py"].decode("utf-8"), current)
    export = (ROOT / HERE / "export_gpu.py").read_text(encoding="utf-8")
    old_export = original[PRIOR + "export_gpu.py"].decode("utf-8")
    start, end = 'torch.set_float32_matmul_precision("highest")', "torch.manual_seed(42)"
    require(fragment(export, start, end) == fragment(old_export, start, end), "Precision/thread recording changed")
    a, b = (ast.parse(s) for s in (old_export, export))
    environment = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_environment")
    require(ast.dump(environment(a)) == ast.dump(environment(b)), "Pre-import isolation guard changed")
    return {**proof, "preserved_source_sha256": PRESERVED_SOURCE_PINS,
            "historical_loader_only_import_redirected": True, "effective_runtime_setup_exact": True}
