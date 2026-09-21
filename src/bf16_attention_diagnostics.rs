//! Diagnostic-only intermediate capture. Each reconstructed step is checked
//! bit-for-bit against the actual update function; production dispatch is unchanged.
use super::*;
use crate::trace::Trace;
use anyhow::{Result, ensure};

#[allow(clippy::too_many_arguments)]
pub fn trace_prefill_head(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    p: Parameters,
    row: usize,
    head: usize,
    backend: Backend,
    trace: &mut dyn Trace,
    prefix: &str,
) -> Result<()> {
    ensure!(
        p.query_len > 1 && row < p.query_len && head < p.heads,
        "prefill diagnostic shape"
    );
    let query = p.query_offset + row;
    let q = &q[(row * p.heads + head) * 64..(row * p.heads + head + 1) * 64];
    let schedule = schedule(query, p);
    let blocks = schedule.partial[..schedule.partial_len]
        .iter()
        .map(|&start| (start, false))
        .chain(
            schedule.full[..schedule.full_len]
                .iter()
                .map(|&start| (start, true)),
        );
    let mut state = Partial::new();
    for (tile, (start, full)) in blocks.filter(|&(start, _)| start < p.kv_len).enumerate() {
        let count = 64.min(p.kv_len - start);
        let mut qk = [0.; 64];
        let mut scores = [f32::NEG_INFINITY; 64];
        for lane in 0..count {
            let key = start + lane;
            let offset = (key * p.heads + head) * 64;
            qk[lane] = bf16_kernels::dot(q, &k[offset..offset + 64], backend);
            if full
                || key <= query
                || ((p.image_start..p.image_end).contains(&query)
                    && (p.image_start..p.image_end).contains(&key))
            {
                scores[lane] = (qk[lane] * 0.125) * 1.442_695_04_f32;
            }
        }
        let maximum = scores.iter().copied().fold(state.maximum, f32::max);
        let safe_max = if maximum == f32::NEG_INFINITY {
            0.
        } else {
            maximum
        };
        let alpha = (state.maximum - safe_max).exp2();
        let exp2 = scores.map(|v| (v - safe_max).exp2());
        let probabilities = exp2.map(bf16::from_f32);
        let denominator = state.denominator * alpha + sum(exp2);
        let mut pv = [0.; 64];
        let mut accumulator = [0.; 64];
        let mut column = [bf16::ZERO; 64];
        for dim in 0..64 {
            for lane in 0..count {
                column[lane] = v[((start + lane) * p.heads + head) * 64 + dim];
            }
            pv[dim] = bf16_kernels::dot(&probabilities, &column, backend);
            accumulator[dim] = state.value[dim] * alpha + pv[dim];
        }
        update(&mut state, q, k, v, head, query, start, full, p, backend);
        ensure!(
            state.maximum.to_bits() == maximum.to_bits()
                && state.denominator.to_bits() == denominator.to_bits()
                && state
                    .value
                    .iter()
                    .zip(&accumulator)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "diagnostic differs from CPU attention update"
        );
        let prefix = format!("{prefix}.tile{tile}");
        for (name, data) in [
            ("qk", qk),
            ("scores_log2", scores),
            ("exp2", exp2),
            ("probabilities_bf16", probabilities.map(|v| v.to_f32())),
            ("pv", pv),
            ("accumulator", accumulator),
        ] {
            trace.tensor(&format!("{prefix}.{name}"), &[64], &data)?;
        }
        for (name, value) in [
            ("key_start", start as f32),
            ("maximum", maximum),
            ("denominator", denominator),
            ("alpha", alpha),
        ] {
            trace.tensor(&format!("{prefix}.{name}"), &[1], &[value])?;
        }
    }
    Ok(())
}
