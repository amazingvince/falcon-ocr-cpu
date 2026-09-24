"""Generate a test-only segment by retaining original forward operation text.

No file I/O or build occurs on import. The only live model change in a future
copy is a cfg(test) module declaration; this generator never edits live files.
"""


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Original forward anchor changed: " + old[:100])
    return text.replace(old, new)


def make_adapter(model_source):
    begin = model_source.index("    pub(crate) fn forward<'s>(")
    end = model_source.index("    /// Advance one generated token", begin)
    original = model_source[begin:end]
    body_start = original.index("        let c = &self.config;")
    body_end = original.index("        session.len += rows;")
    body = original[body_start:body_end]
    # All numerical call/loop text is retained. Extra statements only capture
    # FP32 buffers; the loop selects layer8 and stops layer9 after V expansion.
    body = replace_one(body, '            trace.tensor(&format!("{phase}.embedding"), &[rows, c.dim], h)?;',
                       '            trace.tensor(&format!("{phase}.input"), &[rows, c.dim], h)?;')
    body = replace_one(body, "        for (i, layer) in self.layers.iter().enumerate() {",
                       "        for i in 8..=9 {\n            let layer = &self.layers[i];")
    body = replace_one(body,
        "            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.qkv),",
        '            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n'
        '            trace.tensor(&format!("{phase}.layer.{i}.attention_norm"), &[rows, c.dim], &work.normalized)?;\n'
        '            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.qkv),')
    marker = "            // Normalize each original K head before GQA expansion."
    body = replace_one(body, marker,
        '            trace.tensor(&format!("{phase}.layer.{i}.qkv"), &[rows, qkv_width], &work.qkv)?;\n' + marker)
    marker = "            // Normalize all heads in two calls; avoid creating a Rayon operation"
    body = replace_one(body, marker,
        '            if i == 9 {\n'
        '                trace.tensor(&format!("{phase}.layer.9.v"), &[rows, c.n_heads, c.head_dim], &work.v)?;\n'
        '                break;\n            }\n' + marker)
    body = replace_one(body,
        "            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n            kernels::rms_norm",
        '            trace.tensor(&format!("{phase}.layer.8.wo"), &[rows, c.dim], &work.projected)?;\n'
        '            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n'
        '            trace.tensor(&format!("{phase}.layer.8.attention_residual"), &[rows, c.dim], h)?;\n'
        '            kernels::rms_norm')
    body = replace_one(body,
        "            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.w13),",
        '            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n'
        '            trace.tensor(&format!("{phase}.layer.8.ffn_norm"), &[rows, c.dim], &work.normalized)?;\n'
        '            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.w13),')
    body = replace_one(body, "            kernels::squared_relu_gate(&work.ffn_packed, &mut work.gated);",
        '            trace.tensor(&format!("{phase}.layer.8.w13"), &[rows, 2 * c.ffn_dim], &work.ffn_packed)?;\n'
        '            kernels::squared_relu_gate(&work.ffn_packed, &mut work.gated);\n'
        '            trace.tensor(&format!("{phase}.layer.8.gate"), &[rows, c.ffn_dim], &work.gated)?;')
    body = replace_one(body,
        "            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n            if trace.enabled() {",
        '            trace.tensor(&format!("{phase}.layer.8.w2"), &[rows, c.dim], &work.projected)?;\n'
        '            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n'
        '            if trace.enabled() {')
    prefix = '''// Generated from the pinned production forward; no replacement arithmetic.
impl Model {
    fn crossover_segment(&self, h: &mut [f32], positions: &[usize], positions_hw: &[[f32; 2]],
                         session: &mut Session, trace: &mut dyn Trace, phase: &str) -> Result<()> {
        ensure!(positions.len() == 144 && h.len() == 144 * 768, "fixed segment shape");
        ensure!(session.len == 0 && session.capacity == 161, "fresh original CPU prefill capacity");
        ensure!(trace.enabled() && phase == "cpu_state", "capture original CPU arm only");
'''
    return prefix + body + "        Ok(())\n    }\n}\n", original

