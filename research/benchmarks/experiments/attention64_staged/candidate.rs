//! Isolated compact AVX2 attention experiment. Wrapper, head and dot64 are
//! extracted from the pinned combined-attention source by patch.py.
use rayon::prelude::*;
use std::arch::x86_64::*;

// GENERATED_WRAPPER

// GENERATED_HEAD

// GENERATED_DOT64

// Scalar exp and denominator additions have finished before this function.
// Eight independent value-lane accumulators remain live through the entire
// key-ordered PV loop. LLVM register residency is a compiled-code question.
#[inline(always)]
unsafe fn pv_tile(
    probabilities: &[f32], values: &[f32], first: usize, stride: usize,
    rescale: f32, output: &mut [f32],
) {
    debug_assert_eq!(output.len(), 64);
    debug_assert!(probabilities.len() <= 128);
    debug_assert!(probabilities.is_empty()
        || first + (probabilities.len() - 1) * stride + 64 <= values.len());
    // SAFETY: The checked public attention entry point and unchanged key
    // addressing establish every 64-element value/output range. The caller
    // has AVX2/FMA enabled. Loads/stores deliberately permit unaligned slices.
    unsafe {
        let scale = _mm256_set1_ps(rescale);
        let out = output.as_mut_ptr();
        // Separate rounded multiplication, matching `*value *= rescale`.
        let mut y0 = _mm256_mul_ps(_mm256_loadu_ps(out), scale);
        let mut y1 = _mm256_mul_ps(_mm256_loadu_ps(out.add(8)), scale);
        let mut y2 = _mm256_mul_ps(_mm256_loadu_ps(out.add(16)), scale);
        let mut y3 = _mm256_mul_ps(_mm256_loadu_ps(out.add(24)), scale);
        let mut y4 = _mm256_mul_ps(_mm256_loadu_ps(out.add(32)), scale);
        let mut y5 = _mm256_mul_ps(_mm256_loadu_ps(out.add(40)), scale);
        let mut y6 = _mm256_mul_ps(_mm256_loadu_ps(out.add(48)), scale);
        let mut y7 = _mm256_mul_ps(_mm256_loadu_ps(out.add(56)), scale);
        for (j, probability) in probabilities.iter().enumerate() {
            let factor = _mm256_set1_ps(*probability);
            let value = values.as_ptr().add(first + j * stride);
            y0 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value), y0);
            y1 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(8)), y1);
            y2 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(16)), y2);
            y3 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(24)), y3);
            y4 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(32)), y4);
            y5 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(40)), y5);
            y6 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(48)), y6);
            y7 = _mm256_fmadd_ps(factor, _mm256_loadu_ps(value.add(56)), y7);
        }
        _mm256_storeu_ps(out, y0);
        _mm256_storeu_ps(out.add(8), y1);
        _mm256_storeu_ps(out.add(16), y2);
        _mm256_storeu_ps(out.add(24), y3);
        _mm256_storeu_ps(out.add(32), y4);
        _mm256_storeu_ps(out.add(40), y5);
        _mm256_storeu_ps(out.add(48), y6);
        _mm256_storeu_ps(out.add(56), y7);
    }
}

#[cfg(test)]
mod tests {
    // GENERATED_TESTS
}
