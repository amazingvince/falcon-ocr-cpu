use rayon::prelude::*;
use std::arch::x86_64::*;

// Five actual Falcon row-one shapes, expressed as (input, output).
pub(super) fn supported_shape(input: usize, output: usize) -> bool {
    matches!((input, output), (768, 2048) | (1024, 768) | (768, 4608)
        | (2304, 768) | (768, 65536))
}

pub(super) unsafe fn linear(input: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert!(supported_shape(input.len(), out.len()));
    debug_assert_eq!(weight.len(), input.len() * out.len());
    // The scheduling unit remains 32 output channels, not 32 channel pairs.
    // One feature-qualified direct call handles each disjoint block. No
    // weight packing, temporary vector or independent worker pool is created.
    out.par_chunks_mut(32).enumerate().for_each(|(block, dst)| {
        let start = block * 32 * input.len();
        let weights = &weight[start..start + dst.len() * input.len()];
        // SAFETY: Parent checked complete shapes and resolved AVX2/FMA.
        unsafe { channel_block(input, weights, dst); }
    });
}

#[target_feature(enable = "avx2,fma")]
unsafe fn channel_block(input: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(input.len() % 32, 0);
    debug_assert_eq!(out.len() % 2, 0);
    debug_assert_eq!(weight.len(), input.len() * out.len());
    // SAFETY: Every guarded shape has even output width and K divisible by 32.
    unsafe {
        for (pair_index, dst) in out.chunks_exact_mut(2).enumerate() {
            let begin = pair_index * 2 * input.len();
            let first = &weight[begin..begin + input.len()];
            let second = &weight[begin + input.len()..begin + 2 * input.len()];
            let (left, right) = dot_pair(input, first, second);
            dst[0] = left;
            dst[1] = right;
        }
    }
}

// These helpers are inlined only into the AVX2/FMA-qualified channel block.
// Each output keeps the original four accumulators and chronological FMA
// order. Only loaded input vectors are shared; the reductions never mix.
#[inline(always)]
unsafe fn dot_pair(input: &[f32], first: &[f32], second: &[f32]) -> (f32, f32) {
    debug_assert_eq!(input.len(), first.len());
    debug_assert_eq!(input.len(), second.len());
    debug_assert_eq!(input.len() % 32, 0);
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();
        let mut i = 0;
        while i < input.len() {
            let x0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let x1 = _mm256_loadu_ps(input.as_ptr().add(i + 8));
            let x2 = _mm256_loadu_ps(input.as_ptr().add(i + 16));
            let x3 = _mm256_loadu_ps(input.as_ptr().add(i + 24));
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(first.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(first.as_ptr().add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(x2, _mm256_loadu_ps(first.as_ptr().add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(x3, _mm256_loadu_ps(first.as_ptr().add(i + 24)), a3);
            b0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(second.as_ptr().add(i)), b0);
            b1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(second.as_ptr().add(i + 8)), b1);
            b2 = _mm256_fmadd_ps(x2, _mm256_loadu_ps(second.as_ptr().add(i + 16)), b2);
            b3 = _mm256_fmadd_ps(x3, _mm256_loadu_ps(second.as_ptr().add(i + 24)), b3);
            i += 32;
        }
        (finish(a0, a1, a2, a3), finish(b0, b1, b2, b3))
    }
}

#[inline(always)]
unsafe fn finish(acc0: __m256, acc1: __m256, acc2: __m256, acc3: __m256) -> f32 {
    unsafe {
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let pairs = _mm_hadd_ps(halves, halves);
        _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
    }
}

#[cfg(test)]
mod tests {
    // GENERATED_TESTS
}
