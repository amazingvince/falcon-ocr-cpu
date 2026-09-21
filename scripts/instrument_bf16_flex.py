#!/usr/bin/env python3
"""Add observation stores to the pinned generated Flex source, without executing it.

The output is only a candidate instrumented kernel. No captured intermediate is
qualified until its complete raw output and LSE equal the unmodified kernel and
the frozen fixture bit for bit. This module imports no Torch/CUDA code.
"""
import ast
import hashlib
import pathlib

SOURCE_SHA256 = "df8bf27f03ed01178889a917b328d377e3c797336d007e82a391d69c992cd5c7"
FIELDS = (
    "qk_unscaled_f32", "scores_scaled_f32", "post_mod_scores_log2_f32",
    "exp2_argument_f32", "exp2_result_f32", "probabilities_bf16_exact_promoted_f32",
    "accumulator_before_f32", "accumulator_after_alpha_f32", "accumulator_after_fused_dot_f32",
    "maximum_before_f32", "maximum_after_f32", "alpha_argument_f32", "alpha_result_f32",
    "sum_unrounded_probabilities_f32", "denominator_before_f32", "denominator_after_f32",
    "validity_mask", "key_start", "loop_kind", "loop_ordinal", "query_block", "head",
)
PROBE_ROWS = {1: 13, 2: 13, 5: 41, 6: 41, 8: 53, 10: 54, 11: 124}
FINAL_FIELDS = ("accumulator_before_division_f32", "denominator_before_division_f32",
                "quotient_before_output_cast_f32", "output_bf16_exact_promoted_f32")
SLOTS_PER_LOOP = 3


def replace_exact(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise ValueError(f"Expected {count} source anchors, found {actual}: {before!r}")
    return source.replace(before, after)


def _store(field, value, *, matrix=False):
    index = FIELDS.index(field)
    base = f"debug_base + {index} * 64"
    if matrix:
        return (f"    tl.store(debug_observations + {base} + tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], [BLOCK_M, BLOCK_N]), "
                f"({value}).to(tl.float32), debug_probe & (offs_m == debug_row))\n")
    return (f"    tl.store(debug_observations + {base} + tl.zeros([BLOCK_M], tl.int32), "
            f"({value}).to(tl.float32), debug_probe & (tl.reshape(offs_m, [BLOCK_M]) == debug_row))\n")


def build_instrumented(source_bytes, *, omit_debug_sum=False):
    """Return importable original/candidate modules; preserve all loop bodies."""
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned generated prefill source hash mismatch")
    source = source_bytes.decode("utf-8")
    original = source.split("class Runner:", 1)[0]
    candidate = original
    # Append the observation pointer; original argument positions and specialization
    # metadata remain unchanged. The pointer is threaded through both helpers.
    common = "out_ptr0, ks0, ks1, ks2, ks3, ks4"
    candidate = replace_exact(candidate, common, common + ", debug_observations", 7)
    candidate = replace_exact(candidate, "ks4, debug_observations):", "ks4, debug_observations, debug_final):")
    candidate = replace_exact(candidate, "'ks4': 'i32'", "'ks4': 'i32', 'debug_observations': '*fp32', 'debug_final': '*fp32'")
    # Forward the actual original loop ordinal, rather than a reconstructed live
    # tile list. In particular the original all-masked key192 tile is retained.
    candidate = replace_exact(candidate, "\n    kv_offset,\n", "\n    kv_offset, debug_loop_ordinal,\n")
    candidate = replace_exact(candidate, "\n                kv_offset,\n", "\n                kv_offset, start_n,\n", 2)
    setup = """    # Diagnostic address calculations and stores only; complete matrices unchanged.
    debug_row = tl.where((off_h == 1) | (off_h == 2), 13, tl.where((off_h == 5) | (off_h == 6), 41, tl.where(off_h == 8, 53, tl.where(off_h == 10, 54, 124))))
    debug_probe = (off_h == 1) | (off_h == 2) | (off_h == 5) | (off_h == 6) | (off_h == 8) | (off_h == 10) | (off_h == 11)
    debug_kind = tl.full((), 0, tl.int32)
    if IS_FULL_BLOCKS:
        debug_kind += 1
    debug_base = ((off_h * 2 + debug_kind) * 3 + debug_loop_ordinal) * DEBUG_FIELDS * 64
""".replace("DEBUG_FIELDS", str(len(FIELDS)))
    candidate = replace_exact(candidate, "    # -- load k --\n", setup + "    # -- load k --\n")
    anchor = "    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.\n"
    candidate = replace_exact(candidate, anchor, anchor + _store("qk_unscaled_f32", "qk", matrix=True))
    anchor = "        qk *= SM_SCALE\n"
    candidate = replace_exact(candidate, anchor, anchor + _store("scores_scaled_f32", "qk", matrix=True))
    anchor = "    # -- compute scaling constant ---\n"
    insert = _store("post_mod_scores_log2_f32", "post_mod_scores", matrix=True)
    insert += _store("validity_mask", 'post_mod_scores != float("-inf")', matrix=True)
    for key, value in [("key_start", "kv_base_offset"), ("loop_kind", "debug_kind"),
                       ("loop_ordinal", "debug_loop_ordinal"), ("query_block", "tl.program_id(0)"), ("head", "off_h")]:
        insert += _store(key, value)
    insert += _store("maximum_before_f32", "m_i")
    insert += _store("denominator_before_f32", "l_i")
    insert += _store("accumulator_before_f32", "acc", matrix=True)
    candidate = replace_exact(candidate, anchor, insert + anchor)
    anchor = "    alpha = tl.math.exp2(m_i - m_ij_masked)\n"
    candidate = replace_exact(candidate, anchor,
                              _store("maximum_after_f32", "m_ij")
                              + _store("alpha_argument_f32", "m_i - m_ij_masked")
                              + anchor + _store("alpha_result_f32", "alpha"))
    anchor = "    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])\n"
    candidate = replace_exact(candidate, anchor,
                              _store("exp2_argument_f32", "post_mod_scores - m_ij_masked[:, None]", matrix=True)
                              + anchor + _store("exp2_result_f32", "p", matrix=True)
                              + _store("probabilities_bf16_exact_promoted_f32", "p.to(MATMUL_PRECISION)", matrix=True)
                              + ("" if omit_debug_sum else _store("sum_unrounded_probabilities_f32", "tl.sum(p, 1)")))
    anchor = "    l_i = l_i * alpha + tl.sum(p, 1)\n"
    candidate = replace_exact(candidate, anchor, anchor + _store("denominator_after_f32", "l_i"))
    anchor = "    acc = acc * alpha[:, None]\n"
    candidate = replace_exact(candidate, anchor, anchor + _store("accumulator_after_alpha_f32", "acc", matrix=True))
    anchor = "    acc = tl.dot(p.to(MATMUL_PRECISION), v.to(q.dtype), acc, input_precision=FLOAT32_PRECISION)\n"
    candidate = replace_exact(candidate, anchor, anchor + _store("accumulator_after_fused_dot_f32", "acc", matrix=True))
    anchor = "    acc = acc / l_i[:, None]\n"
    before = """    final_head = tl.program_id(2)
    final_row = tl.where((final_head == 1) | (final_head == 2), 13, tl.where((final_head == 5) | (final_head == 6), 41, tl.where(final_head == 8, 53, tl.where(final_head == 10, 54, 124))))
    final_probe = (final_head == 1) | (final_head == 2) | (final_head == 5) | (final_head == 6) | (final_head == 8) | (final_head == 10) | (final_head == 11)
    final_base = final_head * 4 * 64
    final_columns = tl.broadcast_to(tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :], [BLOCK_M, V_HEAD_DIM_ROUNDED])
    final_mask = final_probe & (offs_m[:, None] == final_row)
    tl.store(debug_final + final_base + final_columns, acc, final_mask)
    tl.store(debug_final + final_base + 64 + tl.zeros([BLOCK_M], tl.int32), l_i, final_probe & (offs_m == final_row))
"""
    after = """    tl.store(debug_final + final_base + 128 + final_columns, acc, final_mask)
    tl.store(debug_final + final_base + 192 + final_columns, acc.to(MATMUL_PRECISION).to(tl.float32), final_mask)
"""
    candidate = replace_exact(candidate, anchor, before + anchor + after)
    for text in [original, candidate]:
        parsed = ast.parse(text)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "triton":
                ast.parse(ast.literal_eval(node.args[1]))
    return original, candidate


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    original, candidate = build_instrumented((root / "artifacts/reference/bf16-flex-prefill-compiled.py").read_bytes())
    print(f"Source-only validation complete: {len(FIELDS)} fields; original {len(original)} bytes; candidate {len(candidate)} bytes. No CUDA execution.")
