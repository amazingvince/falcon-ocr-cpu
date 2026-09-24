//! The generic per-head online-softmax attention loops with function-pointer
//! dot/AXPY kernels: every shape or backend the fixed-width kernels do not
//! cover, and the bit-exact oracle for their tests.
use rayon::prelude::*;

use crate::kernels::{Simd, axpy_kernel, dot_kernel, elements};

/// The generic per-head online-softmax loop with function-pointer kernels.
///
/// It serves every shape or backend the fixed-width kernel does not cover and
/// is the bit-exact oracle for [`attention64`]'s tests. `selected` must already
/// be resolved and validated by the public entry point.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_online_softmax(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    selected: Simd,
) {
    let token_width = elements(n_heads, head_dim);
    let dot = dot_kernel(selected);
    let axpy = axpy_kernel(selected);
    let scale = (head_dim as f32).sqrt().recip();
    output.par_chunks_mut(head_dim).enumerate().for_each(|(qh, out)| {
        let query = qh / n_heads;
        let head = qh % n_heads;
        let absolute_query = query_offset + query;
        let query_in_image = absolute_query >= image_start && absolute_query < image_end;
        let visible_end = if query_in_image { image_end } else { absolute_query + 1 };
        let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
        out.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        const TILE: usize = 128;
        let mut logits = [0.0_f32; TILE];
        for start in (0..visible_end).step_by(TILE) {
            let len = (visible_end - start).min(TILE);
            let mut block_max = f32::NEG_INFINITY;
            for (j, logit) in logits[..len].iter_mut().enumerate() {
                let key = start + j;
                let begin = key * token_width + head * head_dim;
                *logit = dot(qvec, &k[begin..begin + head_dim]) * scale;
                block_max = block_max.max(*logit);
            }
            let new_max = running_max.max(block_max);
            let rescale = if running_max == f32::NEG_INFINITY {
                0.0
            } else {
                (running_max - new_max).exp()
            };
            for value in out.iter_mut() {
                *value *= rescale;
            }
            denominator *= rescale;
            for (j, logit) in logits[..len].iter().enumerate() {
                let probability = (*logit - new_max).exp();
                denominator += probability;
                let begin = (start + j) * token_width + head * head_dim;
                axpy(probability, &v[begin..begin + head_dim], out);
            }
            running_max = new_max;
        }
        let logsumexp = running_max + denominator.ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in out {
            *value = (*value / denominator) * sink_scale;
        }
    });
}

/// The generic compact-cache online-softmax loop; see [`attention_online_softmax`].
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_compact_online_softmax(
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    prefix_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    selected: Simd,
) {
    let query_width = elements(n_heads, head_dim);
    let kv_width = elements(n_kv_heads, head_dim);
    let dot = dot_kernel(selected);
    let axpy = axpy_kernel(selected);
    let scale = (head_dim as f32).sqrt().recip();
    let repeat = n_heads / n_kv_heads;
    output.par_chunks_mut(head_dim).enumerate().for_each(|(qh, out)| {
        let query = qh / n_heads;
        let head = qh % n_heads;
        let kv_head = head / repeat;
        let absolute_query = query_offset + query;
        let query_in_image = absolute_query >= image_start && absolute_query < image_end;
        let visible_end = if query_in_image { image_end } else { absolute_query + 1 };
        let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
        out.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        const TILE: usize = 128;
        let mut logits = [0.0_f32; TILE];
        for start in (0..visible_end).step_by(TILE) {
            let len = (visible_end - start).min(TILE);
            let mut block_max = f32::NEG_INFINITY;
            for (j, logit) in logits[..len].iter_mut().enumerate() {
                let key = start + j;
                let kvec = if key < prefix_len {
                    let begin = key * query_width + head * head_dim;
                    &prefix_k[begin..begin + head_dim]
                } else {
                    let begin = (key - prefix_len) * kv_width + kv_head * head_dim;
                    &generated_k[begin..begin + head_dim]
                };
                *logit = dot(qvec, kvec) * scale;
                block_max = block_max.max(*logit);
            }
            let new_max = running_max.max(block_max);
            let rescale = if running_max == f32::NEG_INFINITY {
                0.0
            } else {
                (running_max - new_max).exp()
            };
            for value in out.iter_mut() {
                *value *= rescale;
            }
            denominator *= rescale;
            for (j, logit) in logits[..len].iter().enumerate() {
                let probability = (*logit - new_max).exp();
                denominator += probability;
                let begin = (start + j) * kv_width + kv_head * head_dim;
                axpy(probability, &v[begin..begin + head_dim], out);
            }
            running_max = new_max;
        }
        let logsumexp = running_max + denominator.ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in out {
            *value = (*value / denominator) * sink_scale;
        }
    });
}
