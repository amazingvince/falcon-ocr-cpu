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
    # FP32 buffers; the loop selects block0 only.
    body = replace_one(body, '            trace.tensor(&format!("{phase}.embedding"), &[rows, c.dim], h)?;',
                       '            trace.tensor(&format!("{phase}.input"), &[rows, c.dim], h)?;')
    body = replace_one(body, "        for (i, layer) in self.layers.iter().enumerate() {",
                       "        for i in 0..=0 {\n            let layer = &self.layers[i];")
    body = replace_one(body,
        "            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.qkv),",
        '            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n'
        '            trace.tensor(&format!("{phase}.layer.{i}.attention_norm"), &[rows, c.dim], &work.normalized)?;\n'
        '            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.qkv),')
    marker = "            // Normalize each original K head before GQA expansion."
    body = replace_one(body, marker,
        '            trace.tensor(&format!("{phase}.layer.{i}.qkv"), &[rows, qkv_width], &work.qkv)?;\n' + marker)
    body = replace_one(body,
        "            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n            kernels::rms_norm",
        '            trace.tensor(&format!("{phase}.layer.0.wo"), &[rows, c.dim], &work.projected)?;\n'
        '            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n'
        '            trace.tensor(&format!("{phase}.layer.0.attention_residual"), &[rows, c.dim], h)?;\n'
        '            kernels::rms_norm')
    body = replace_one(body,
        "            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.w13),",
        '            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);\n'
        '            trace.tensor(&format!("{phase}.layer.0.ffn_norm"), &[rows, c.dim], &work.normalized)?;\n'
        '            kernels::linear_with_simd(\n                &work.normalized,\n                rows,\n                c.dim,\n                self.w(&layer.w13),')
    body = replace_one(body, "            kernels::squared_relu_gate(&work.ffn_packed, &mut work.gated);",
        '            trace.tensor(&format!("{phase}.layer.0.w13"), &[rows, 2 * c.ffn_dim], &work.ffn_packed)?;\n'
        '            kernels::squared_relu_gate(&work.ffn_packed, &mut work.gated);\n'
        '            trace.tensor(&format!("{phase}.layer.0.gate"), &[rows, c.ffn_dim], &work.gated)?;')
    body = replace_one(body,
        "            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n            if trace.enabled() {",
        '            trace.tensor(&format!("{phase}.layer.0.w2"), &[rows, c.dim], &work.projected)?;\n'
        '            for (x, a) in h.iter_mut().zip(&work.projected) {\n                *x += a;\n            }\n'
        '            if trace.enabled() {')
    # Mechanical observer removal must reproduce every original body byte except
    # the deliberately bounded layer-loop selection and embedding label.
    observed_stage_suffixes = (".attention_norm", ".qkv", ".wo", ".attention_residual",
                               ".ffn_norm", ".w13", ".gate", ".w2")
    restored = "".join(line for line in body.splitlines(keepends=True)
                       if not ("trace.tensor" in line and
                               any(suffix + '\"' in line for suffix in observed_stage_suffixes)))
    restored = replace_one(restored, 'trace.tensor(&format!("{phase}.input"),',
                            'trace.tensor(&format!("{phase}.embedding"),')
    restored = replace_one(restored, "        for i in 0..=0 {\n            let layer = &self.layers[i];",
                            "        for (i, layer) in self.layers.iter().enumerate() {")
    if restored != original[body_start:body_end]:
        raise ValueError("Observer removal did not restore original numerical body text")
    prefix = '''// Generated from the pinned production forward; no replacement arithmetic.
impl Model {
    fn crossover_segment(&self, h: &mut [f32], positions: &[usize], positions_hw: &[[f32; 2]],
                         session: &mut Session, trace: &mut dyn Trace, phase: &str) -> Result<()> {
        ensure!(positions.len() == 144 && h.len() == 144 * 768, "fixed segment shape");
        ensure!(session.len == 0 && session.capacity == 161, "fresh original CPU prefill capacity");
        ensure!(trace.enabled() && phase == "cpu_state", "capture original CPU arm only");
'''
    return prefix + body + "        Ok(())\n    }\n}\n", original

