//! Copied-project experiment only. The head body is extracted from the frozen
//! original attention implementation by patch.py, not independently rewritten.
use rayon::prelude::*;
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn attention(
    q: &[f32], k: &[f32], v: &[f32], n_heads: usize, query_offset: usize,
    image_start: usize, image_end: usize, sinks: &[f32], output: &mut [f32],
) {
    let scale = (64_f32).sqrt().recip();
    output.par_chunks_mut(64).enumerate().for_each(|(qh, out)| {
        // SAFETY: Parent checks AVX2/FMA and all shapes before entering here.
        // The target-feature function encloses the complete per-head loop;
        // Rayon closures do not need to inherit the caller's target features.
        unsafe { head(qh, q, k, v, n_heads, query_offset, image_start, image_end,
                      sinks, scale, out); }
    });
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn head(
    qh: usize, q: &[f32], k: &[f32], v: &[f32], n_heads: usize,
    query_offset: usize, image_start: usize, image_end: usize,
    sinks: &[f32], scale: f32, out: &mut [f32],
) {
    let head_dim = 64;
    let token_width = n_heads * head_dim;
    // SAFETY: Shapes/features were checked by the original safe entry point.
    unsafe {
        // GENERATED_ORIGINAL_HEAD_BODY
    }
}

// These helpers are deliberately inline(always), with the feature-qualified
// head as their caller. No function-pointer call remains inside the key loops.
#[inline(always)]
unsafe fn dot64(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 64);
    debug_assert_eq!(b.len(), 64);
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for i in [0, 32] {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)),
                                 _mm256_loadu_ps(b.as_ptr().add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)),
                                 _mm256_loadu_ps(b.as_ptr().add(i + 8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)),
                                 _mm256_loadu_ps(b.as_ptr().add(i + 16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)),
                                 _mm256_loadu_ps(b.as_ptr().add(i + 24)), acc3);
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let pairs = _mm_hadd_ps(halves, halves);
        _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
    }
}

#[inline(always)]
unsafe fn axpy64(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), 64);
    debug_assert_eq!(y.len(), 64);
    unsafe {
        let factor = _mm256_set1_ps(a);
        for i in [0, 8, 16, 24, 32, 40, 48, 56] {
            let value = _mm256_fmadd_ps(factor, _mm256_loadu_ps(x.as_ptr().add(i)),
                                      _mm256_loadu_ps(y.as_ptr().add(i)));
            _mm256_storeu_ps(y.as_mut_ptr().add(i), value);
        }
    }
}

#[cfg(test)]
#[path = "attention64_tests.rs"]
mod tests;
