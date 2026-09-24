"""Compare native source fragments only when explicitly called; no imports run it."""
import ast
import hashlib
import textwrap

from contract import ROOT, OLD_EXPORT, OLD_EXPORT_SHA, require


def fragment(source, first, last):
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == first]
    ends = [i for i, line in enumerate(lines) if line.strip() == last]
    require(len(starts) == len(ends) == 1 and starts[0] <= ends[0], "Ambiguous native fragment anchors")
    return textwrap.dedent("\n".join(lines[starts[0]:ends[0] + 1]))


def verify_text(original, candidate):
    regions = [
        ("module = import_model(model_dir)", 'state = {"branch": None}'),
        ("def capture(key, value):", "module.apply_3d_rotary_emb = rope"),
        ("cache = module.KVCache(1, 256, config.n_heads, config.head_dim, config.n_layers)",
         'capture("layer.9.v", v)'),
        ("for handle in handles:", "module.apply_3d_rotary_emb = original_rope"),
    ]
    for start, end in regions:
        require(fragment(original, start, end) == fragment(candidate, start, end),
                "Native arithmetic/setup/observation fragment changed: " + start)
    tree = ast.parse(candidate)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    require(sum(isinstance(n.func, ast.Name) and n.func.id == "layer8" for n in calls) == 1,
            "Expected one layer8 call site")
    require(sum(isinstance(n.func, ast.Attribute) and n.func.attr == "_pre_attention_qkv"
                for n in calls) == 1, "Expected one layer9 native QKV call site")
    require('(("control_c", c7), ("substitution_s", substitution))' in candidate,
            "Fixed ordered branches changed")
    return {"native_fragments_exact": len(regions), "native_call_sites": 2,
            "limits": "Text/AST guards cover listed native fragments and call sites; they do not prove machine-code identity. All 17 actual control outputs remain mandatory."}


def verify_sources():
    original_bytes = (ROOT / OLD_EXPORT).read_bytes()
    require(hashlib.sha256(original_bytes).hexdigest() == OLD_EXPORT_SHA, "Old source pin differs")
    candidate = (ROOT / "research/gpu-reference/experiments/fp32_crossover_downstream/native_segment.py").read_text(encoding="utf-8")
    return verify_text(original_bytes.decode("utf-8"), candidate)
