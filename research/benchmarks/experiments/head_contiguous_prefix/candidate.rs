//! Exact frozen fixed64 arithmetic with prefix K/V addressing changes only.
use rayon::prelude::*;
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn attention(
    q: &[f32], prefix_k: &[f32], prefix_v: &[f32], generated_k: &[f32], generated_v: &[f32],
    prefix_len: usize, n_heads: usize, n_kv_heads: usize, query_offset: usize,
    image_start: usize, image_end: usize, sinks: &[f32], output: &mut [f32],
) {
    let scale = (64_f32).sqrt().recip();
    output.par_chunks_mut(64).enumerate().for_each(|(qh, out)| {
        // SAFETY: The safe adapter checked complete fixed shapes and AVX2/FMA.
        unsafe { prefix_head(qh, q, prefix_k, prefix_v, generated_k, generated_v,
            prefix_len, n_heads, n_kv_heads, query_offset, image_start, image_end,
            sinks, scale, out); }
    });
}

#[target_feature(enable="avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn prefix_head(
    qh: usize, q: &[f32], prefix_k: &[f32], prefix_v: &[f32], generated_k: &[f32],
    generated_v: &[f32], prefix_len: usize, n_heads: usize, n_kv_heads: usize,
    query_offset: usize, image_start: usize, image_end: usize, sinks: &[f32],
    scale: f32, out: &mut [f32],
) {
    let head_dim = 64;
    let kv_width = n_kv_heads * head_dim;
    let repeat = n_heads / n_kv_heads;
    unsafe {
        // GENERATED_ORIGINAL_COMPACT_HEAD_BODY
    }
}

// Copied byte-for-byte from the frozen combined attention module.
// GENERATED_FIXED_HELPERS
