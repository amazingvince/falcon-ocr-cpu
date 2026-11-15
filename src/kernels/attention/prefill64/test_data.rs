//! Shared inputs and high-precision reference for the static-softmax tests.
use super::{CompactKv, Geometry, HEAD_DIM};

/// Deterministic inputs in [-range, range).
pub(super) fn values(n: usize, seed: u32, range: f32) -> Vec<f32> {
    let mut values = crate::kernels::test_values(n, seed);
    for value in &mut values {
        *value *= range;
    }
    values
}

/// Random heads scaled to norm 8, as the model's per-head RMS norms leave them.
pub(super) fn normalized(n: usize, seed: u32) -> Vec<f32> {
    let mut x = values(n, seed, 1.0);
    for head in x.chunks_exact_mut(HEAD_DIM) {
        let norm = head.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in head {
            *v *= 8.0 / norm;
        }
    }
    x
}

/// Prefill operands (q, k, v, sinks) for 16 query and 8 KV heads.
pub(super) fn operands(queries: usize, total: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let seed = 17 + total as u32;
    (
        normalized(queries * 16 * HEAD_DIM, seed),
        normalized(total * 16 * HEAD_DIM, seed + 2),
        values(total * 8 * HEAD_DIM, seed + 6, 3.0),
        values(16, 29, 2.0),
    )
}

/// Maximum absolute errors of the static and online outputs against f64
/// attention with the exact exp, on the supplied (possibly BF16-rounded) inputs.
pub(super) fn reference_errors(
    q: &[f32],
    cache: &CompactKv<'_>,
    geometry: Geometry,
    sinks: &[f32],
    output: &[f32],
    online: &[f32],
    logit_limit: f64,
) -> (f64, f64) {
    let width = cache.query_width();
    let (mut worst, mut worst_online) = (0.0_f64, 0.0_f64);
    for (head, &sink) in sinks.iter().enumerate() {
        let kv_head = head / cache.repeat();
        for (row, queries) in q.chunks_exact(width).enumerate() {
            let absolute = geometry.query_offset + row;
            let query = &queries[head * HEAD_DIM..][..HEAD_DIM];
            let logits: Vec<(usize, f64)> = (0..cache.total_len)
                .filter(|&key| !geometry.masked(absolute, key))
                .map(|key| {
                    let dot: f64 = query
                        .iter()
                        .zip(cache.key(key, head))
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    (key, dot / 8.0)
                })
                .collect();
            assert!(logits.iter().all(|l| l.1.abs() <= logit_limit));
            let denominator: f64 = logits.iter().map(|l| l.1.exp()).sum();
            let sink_scale = 1.0 / (1.0 + (sink as f64 - denominator.ln()).exp());
            for d in 0..HEAD_DIM {
                let sum: f64 = logits
                    .iter()
                    .map(|&(key, logit)| logit.exp() * cache.value(key, kv_head)[d] as f64)
                    .sum();
                let expected = sum / denominator * sink_scale;
                let at = row * width + head * HEAD_DIM + d;
                worst = worst.max((output[at] as f64 - expected).abs());
                worst_online = worst_online.max((online[at] as f64 - expected).abs());
            }
        }
    }
    (worst, worst_online)
}
