//! Isolated copied-project experiment. No live runner or default changes.
use rayon::prelude::*;
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn attention(
    q: &[f32], temporal_k: &[f32], spatial_k: &[f32], generated_k: &[f32], v: &[f32],
    prefix_len: usize, n_heads: usize, n_kv_heads: usize, query_offset: usize,
    image_start: usize, image_end: usize, sinks: &[f32], output: &mut [f32],
) {
    let scale = (64_f32).sqrt().recip();
    output.par_chunks_mut(64).enumerate().for_each(|(qh, out)| {
        // SAFETY: The safe adapter checked the fixed shape and AVX2/FMA.
        unsafe { temporal_head(qh, q, temporal_k, spatial_k, generated_k, v,
            prefix_len, n_heads, n_kv_heads, query_offset, image_start, image_end,
            sinks, scale, out); }
    });
}

#[target_feature(enable="avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn temporal_head(
    qh: usize, q: &[f32], temporal_k: &[f32], spatial_k: &[f32], generated_k: &[f32],
    v: &[f32], prefix_len: usize, n_heads: usize, n_kv_heads: usize,
    query_offset: usize, image_start: usize, image_end: usize, sinks: &[f32],
    scale: f32, out: &mut [f32],
) {
    let head_dim=64;
    let kv_width=n_kv_heads*head_dim;
    let repeat=n_heads/n_kv_heads;
    // The frozen compact_head body is copied exactly except its K load/dot
    // block. Temporal and spatial halves update the SAME four accumulators.
    unsafe {
        // GENERATED_ORIGINAL_COMPACT_HEAD_BODY
    }
}

#[inline(always)]
unsafe fn dot64_split(a: &[f32], temporal: &[f32], spatial: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 64);
    debug_assert_eq!(temporal.len(), 32);
    debug_assert_eq!(spatial.len(), 32);
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        // Same operand order and chronological FMAs as frozen dot64.
        for (i, key) in [(0, temporal), (32, spatial)] {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)),
                                 _mm256_loadu_ps(key.as_ptr()), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)),
                                 _mm256_loadu_ps(key.as_ptr().add(8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)),
                                 _mm256_loadu_ps(key.as_ptr().add(16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)),
                                 _mm256_loadu_ps(key.as_ptr().add(24)), acc3);
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let pairs = _mm_hadd_ps(halves, halves);
        _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
    }
}

// Copied byte-for-byte from the frozen combined attention module.
// GENERATED_FIXED_HELPERS

#[cfg(test)]
mod tests {
    // GENERATED_TESTS
}
