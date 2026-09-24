//! Experimental phase-packed FP32 small-batch projections.
//!
//! The AVX2 arithmetic is identical to `kernels::linear_with_simd(Avx2)`:
//! four independent eight-lane FMA accumulators, followed by the same sum
//! tree, vector remainder, horizontal reduction, and scalar tail. Only the
//! scheduling of independent accumulators changes. Weights are persisted as
//! `[output_channel, accumulator_phase, K_block, lane]` with an unpacked tail.
//! This lets batches of four/eight share each weight load without retaining
//! sixteen/thirty-two live vector accumulators. These kernels are not wired
//! into the runner and have no measured performance promotion.

use rayon::prelude::*;
use std::ptr::NonNull;

use crate::kernels::Simd;

fn elements(rows: usize, width: usize) -> usize {
    rows.checked_mul(width).expect("packed FP32 shape overflow")
}

/// A reusable copy of HF `[output_dim,input_dim]` FP32 weights. Construction
/// owns exactly `output_dim * input_dim * 4` tensor bytes, with no padding.
pub struct PhasePackedLinear {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
}

impl PhasePackedLinear {
    pub fn new(weights: &[f32], input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            weights.len(),
            elements(input_dim, output_dim),
            "packed FP32 weight shape"
        );
        let blocks = input_dim / 32;
        let phase_stride = blocks * 8;
        let mut packed = vec![0.0; weights.len()];
        for channel in 0..output_dim {
            let src = &weights[channel * input_dim..(channel + 1) * input_dim];
            let dst = &mut packed[channel * input_dim..(channel + 1) * input_dim];
            for phase in 0..4 {
                for block in 0..blocks {
                    dst[phase * phase_stride + block * 8..phase * phase_stride + (block + 1) * 8]
                        .copy_from_slice(
                            &src[block * 32 + phase * 8..block * 32 + (phase + 1) * 8],
                        );
                }
            }
            dst[blocks * 32..].copy_from_slice(&src[blocks * 32..]);
        }
        Self {
            input_dim,
            output_dim,
            weights: packed,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    pub fn packed_weight_bytes(&self) -> usize {
        self.weights.len() * size_of::<f32>()
    }

    /// Projects one to eight rows, with allocation performed only at construction.
    /// Requires AVX2 and FMA at runtime; there is deliberately no implicit ISA or
    /// reduction-order fallback inside this isolated experimental interface.
    pub fn linear_avx2(&self, input: &[f32], rows: usize, output: &mut [f32]) {
        assert!(rows <= 8, "phase-packed FP32 supports at most eight rows");
        assert_eq!(
            input.len(),
            elements(rows, self.input_dim),
            "packed FP32 input shape"
        );
        assert_eq!(
            output.len(),
            elements(rows, self.output_dim),
            "packed FP32 output shape"
        );
        Simd::Avx2
            .validate()
            .expect("phase-packed FP32 requires AVX2/FMA");
        if output.is_empty() {
            return;
        }
        if self.input_dim == 0 {
            output.fill(0.0);
            return;
        }
        let destinations = OutputColumns {
            ptr: NonNull::new(output.as_mut_ptr()).unwrap(),
            width: self.output_dim,
            rows,
        };
        // Each Rayon item owns one output column across all request rows.
        // No two items address the same output element. The exclusive output
        // borrow remains live until this synchronous parallel operation joins.
        (0..self.output_dim)
            .into_par_iter()
            .with_min_len(8)
            .for_each(|channel| {
                let weight =
                    &self.weights[channel * self.input_dim..(channel + 1) * self.input_dim];
                let mut values = [0.0; 8];
                #[cfg(target_arch = "x86_64")]
                // SAFETY: Feature checks and exact shapes bound every SIMD load;
                // const batch instantiations match the validated request count.
                unsafe {
                    match rows {
                        1 => x86::column::<1, 4>(input, self.input_dim, weight, &mut values),
                        2 => x86::column::<2, 4>(input, self.input_dim, weight, &mut values),
                        3 => x86::column::<3, 2>(input, self.input_dim, weight, &mut values),
                        4 => x86::column::<4, 2>(input, self.input_dim, weight, &mut values),
                        5 => x86::column::<5, 1>(input, self.input_dim, weight, &mut values),
                        6 => x86::column::<6, 1>(input, self.input_dim, weight, &mut values),
                        7 => x86::column::<7, 1>(input, self.input_dim, weight, &mut values),
                        8 => x86::column::<8, 1>(input, self.input_dim, weight, &mut values),
                        _ => unreachable!("validated nonempty small batch"),
                    }
                }
                // SAFETY: This item is the sole writer of this column; every
                // index is bounded by the caller's validated row-major output.
                unsafe {
                    destinations.store(channel, &values[..rows]);
                }
            });
    }
}

struct OutputColumns {
    ptr: NonNull<f32>,
    width: usize,
    rows: usize,
}

// SAFETY: OutputColumns is used only while the owning exclusive slice borrow
// is live. The caller gives each parallel item a distinct column, with no
// reads until joining. Moving/sharing the descriptor does not move the slice.
unsafe impl Send for OutputColumns {}
unsafe impl Sync for OutputColumns {}

impl OutputColumns {
    /// Caller must own this column exclusively throughout the operation.
    unsafe fn store(&self, column: usize, values: &[f32]) {
        debug_assert!(column < self.width && values.len() == self.rows);
        for (row, &value) in values.iter().enumerate() {
            // SAFETY: Safe entry point checked the full matrix dimensions,
            // and caller promises exclusive ownership of this column.
            unsafe {
                self.ptr
                    .as_ptr()
                    .add(row * self.width + column)
                    .write(value);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    // Accumulator arrays are indexed by the row/phase loop variables on purpose:
    // the const-generic bounds keep every index in registers.
    #[allow(clippy::needless_range_loop)]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn column<const B: usize, const P: usize>(
        input: &[f32],
        width: usize,
        weight: &[f32],
        output: &mut [f32; 8],
    ) {
        debug_assert!(B > 0 && B <= 8 && [1, 2, 4].contains(&P));
        debug_assert_eq!(input.len(), B * width);
        debug_assert_eq!(weight.len(), width);
        let blocks = width / 32;
        let phase_stride = blocks * 8;
        let mut partials = [[0.0_f32; 32]; B];
        // SAFETY: The caller checked the exact shapes and feature support.
        // Full-width loads are only used for full 32/8-value input blocks.
        unsafe {
            for first_phase in (0..4).step_by(P) {
                let mut accumulators = [[_mm256_setzero_ps(); P]; B];
                for block in 0..blocks {
                    for phase in 0..P {
                        let w = _mm256_loadu_ps(
                            weight
                                .as_ptr()
                                .add((first_phase + phase) * phase_stride + block * 8),
                        );
                        for row in 0..B {
                            let x = _mm256_loadu_ps(
                                input
                                    .as_ptr()
                                    .add(row * width + block * 32 + (first_phase + phase) * 8),
                            );
                            accumulators[row][phase] =
                                _mm256_fmadd_ps(x, w, accumulators[row][phase]);
                        }
                    }
                }
                for row in 0..B {
                    for phase in 0..P {
                        _mm256_storeu_ps(
                            partials[row].as_mut_ptr().add((first_phase + phase) * 8),
                            accumulators[row][phase],
                        );
                    }
                }
            }
            for row in 0..B {
                let p = partials[row].as_ptr();
                let mut sum = _mm256_add_ps(
                    _mm256_add_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))),
                    _mm256_add_ps(_mm256_loadu_ps(p.add(16)), _mm256_loadu_ps(p.add(24))),
                );
                let x = &input[row * width..(row + 1) * width];
                let mut i = blocks * 32;
                while i + 8 <= width {
                    sum = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.as_ptr().add(i)),
                        _mm256_loadu_ps(weight.as_ptr().add(i)),
                        sum,
                    );
                    i += 8;
                }
                let halves =
                    _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
                let pairs = _mm_hadd_ps(halves, halves);
                let mut value = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
                while i < width {
                    value += x[i] * weight[i];
                    i += 1;
                }
                output[row] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;

    fn values(len: usize, mut state: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn compare(rows: usize, input_dim: usize, output_dim: usize) {
        let x = values(rows * input_dim, 91);
        let w = values(output_dim * input_dim, 117);
        let packed = PhasePackedLinear::new(&w, input_dim, output_dim);
        assert_eq!(packed.packed_weight_bytes(), w.len() * 4);
        let mut actual = vec![f32::NAN; rows * output_dim];
        let mut expected = actual.clone();
        kernels::linear_with_simd(
            &x,
            rows,
            input_dim,
            &w,
            output_dim,
            &mut expected,
            Simd::Avx2,
        );
        packed.linear_avx2(&x, rows, &mut actual);
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "rows={rows}, K={input_dim}, N={output_dim}, index={index}"
            );
        }
    }

    #[test]
    fn all_batches_and_vector_tails_are_bit_identical() {
        if Simd::Avx2.validate().is_err() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            for rows in 0..=8 {
                for width in (0..=65).chain([127, 128, 129, 767, 768, 769, 1024, 2304]) {
                    compare(rows, width, 19);
                }
            }
        });
    }

    #[test]
    fn real_qkv_attention_and_ffn_projection_shapes_are_bit_identical() {
        if Simd::Avx2.validate().is_err() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            for rows in [1, 2, 4, 8] {
                for (input, output) in [(768, 2048), (1024, 768), (768, 4608), (2304, 768)] {
                    compare(rows, input, output);
                }
            }
        });
    }

    #[test]
    fn signed_zero_and_large_cancellation_preserve_the_reduction() {
        if Simd::Avx2.validate().is_err() {
            return;
        }
        let width = 129;
        let x: Vec<_> = (0..8 * width)
            .map(|i| [1.0e15, -1.0e15, -0.0, 0.0, 1.0, -2.0, 0.125][i % 7])
            .collect();
        let w: Vec<_> = (0..19 * width)
            .map(|i| [1.0, -1.0, 0.0, -0.0, 0.25][i % 5])
            .collect();
        let mut expected = vec![0.0; 8 * 19];
        let mut actual = expected.clone();
        kernels::linear_with_simd(&x, 8, width, &w, 19, &mut expected, Simd::Avx2);
        PhasePackedLinear::new(&w, width, 19).linear_avx2(&x, 8, &mut actual);
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "phase-packed FP32 supports at most eight rows")]
    fn rejects_unsupported_batch_before_dispatch() {
        PhasePackedLinear::new(&[1.0], 1, 1).linear_avx2(&[1.0; 9], 9, &mut [0.0; 9]);
    }
}
