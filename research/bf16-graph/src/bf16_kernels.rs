//! Experimental BF16-operand CPU building blocks, separate from the FP32 runner.
//!
//! Both activations and weights are explicitly BF16; accumulation and outputs
//! are FP32. This is neither weight-only mixed precision (FP32 activations with
//! BF16 weights) nor the full Hugging Face BF16 graph: HF linear outputs need
//! an additional BF16 cast, and RMSNorm, attention, residual, and Triton gate
//! cast boundaries still need independent GPU qualification. Nothing here is
//! wired into `Runner` or advertised as GPU BF16 parity.
//!
//! AVX-512BF16 dot instructions flush BF16 subnormal operands and have hardware
//! underflow semantics. The scalar reference uses ordinary FP32 arithmetic.
//! Their mathematical comparisons target finite, normal-range operands/results;
//! explicit subnormal tests document the difference instead of concealing it.

use half::bf16;
use rayon::prelude::*;

/// Experimental CPU dispatch; all choices remain unqualified for model use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Scalar,
    Avx512Bf16,
}

impl Backend {
    pub fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Auto | Self::Scalar => Ok(()),
            Self::Avx512Bf16 if avx512_bf16_available() => Ok(()),
            Self::Avx512Bf16 => Err("AVX-512F and AVX-512BF16 are required"),
        }
    }

    pub fn resolved(self) -> Self {
        self.validate()
            .expect("unsupported experimental BF16 backend");
        match self {
            Self::Auto if avx512_bf16_available() => Self::Avx512Bf16,
            Self::Auto => Self::Scalar,
            explicit => explicit,
        }
    }
}

pub fn avx512_bf16_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn elements(rows: usize, width: usize) -> usize {
    rows.checked_mul(width)
        .expect("BF16 tensor shape overflows usize")
}

/// Explicit round-to-nearest-even conversion, preserving a visible dtype boundary.
pub fn convert_f32_to_bf16(input: &[f32], output: &mut [bf16]) {
    assert_eq!(input.len(), output.len(), "BF16 conversion shape");
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(dst, src)| *dst = bf16::from_f32(*src));
}

/// BF16×BF16 dot product, returning FP32 without rounding the output to BF16.
pub fn dot(input: &[bf16], weight: &[bf16], backend: Backend) -> f32 {
    assert_eq!(input.len(), weight.len(), "BF16 dot shape");
    dot_kernel(backend.resolved())(input, weight)
}

type Dot = fn(&[bf16], &[bf16]) -> f32;

fn dot_kernel(backend: Backend) -> Dot {
    match backend {
        Backend::Scalar => dot_scalar,
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512Bf16 => |a, b| {
            // SAFETY: Every caller resolves and validates the CPU features,
            // and shape validation gives equal slice lengths.
            unsafe { x86::dot_bf16(a, b) }
        },
        _ => unreachable!("unresolved or unsupported BF16 backend"),
    }
}

fn dot_scalar(input: &[bf16], weight: &[bf16]) -> f32 {
    input
        .iter()
        .zip(weight)
        .map(|(a, b)| a.to_f32() * b.to_f32())
        .sum()
}

/// Unpacked reference/GEMV path with HF row-major `[output_dim,input_dim]` weights.
/// Rows represent independent input vectors. All output values remain FP32.
pub fn linear(
    input: &[bf16],
    rows: usize,
    input_dim: usize,
    weight: &[bf16],
    output_dim: usize,
    output: &mut [f32],
    backend: Backend,
) {
    assert_eq!(
        input.len(),
        elements(rows, input_dim),
        "BF16 linear input shape"
    );
    assert_eq!(
        weight.len(),
        elements(output_dim, input_dim),
        "BF16 linear weight shape"
    );
    assert_eq!(
        output.len(),
        elements(rows, output_dim),
        "BF16 linear output shape"
    );
    let dot = dot_kernel(backend.resolved());
    if output.is_empty() {
        return;
    }
    if input_dim == 0 {
        output.fill(0.0);
        return;
    }
    output
        .par_iter_mut()
        .enumerate()
        .with_min_len(32)
        .for_each(|(index, dst)| {
            let row = index / output_dim;
            let channel = index % output_dim;
            *dst = dot(
                &input[row * input_dim..(row + 1) * input_dim],
                &weight[channel * input_dim..(channel + 1) * input_dim],
            );
        });
}

/// Reusable BF16 weight packing for sixteen output channels per dot instruction.
///
/// Layout is `[output_block, input_pair, output_lane, pair_member]`. Each packed
/// vector contains two weights for each of sixteen outputs. An input pair is
/// broadcast to all lanes; VDPBF16PS updates sixteen FP32 output accumulators.
/// Construction allocates; forward calls use the caller's current Rayon pool
/// and have no tensor scratch allocation. Packing is experimental and has not
/// passed whole-model speed or quality promotion gates.
#[derive(Debug)]
pub struct PackedLinear {
    input_dim: usize,
    output_dim: usize,
    pairs: usize,
    weights: Vec<bf16>,
}

impl PackedLinear {
    pub fn new(weight: &[bf16], input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            weight.len(),
            elements(output_dim, input_dim),
            "BF16 packed weight shape"
        );
        let pairs = input_dim.div_ceil(2);
        let blocks = output_dim.div_ceil(16);
        let mut weights = vec![bf16::ZERO; elements(elements(blocks, pairs), 32)];
        for channel in 0..output_dim {
            for inner in 0..input_dim {
                let index =
                    ((channel / 16) * pairs + inner / 2) * 32 + (channel % 16) * 2 + inner % 2;
                weights[index] = weight[channel * input_dim + inner];
            }
        }
        Self {
            input_dim,
            output_dim,
            pairs,
            weights,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    pub fn packed_weight_bytes(&self) -> usize {
        self.weights.len() * std::mem::size_of::<bf16>()
    }

    /// Multiply BF16 input rows by packed BF16 weights, writing FP32 outputs.
    pub fn linear(&self, input: &[bf16], rows: usize, output: &mut [f32], backend: Backend) {
        assert_eq!(
            input.len(),
            elements(rows, self.input_dim),
            "BF16 packed input shape"
        );
        assert_eq!(
            output.len(),
            elements(rows, self.output_dim),
            "BF16 packed output shape"
        );
        let selected = backend.resolved();
        if output.is_empty() {
            return;
        }
        if self.input_dim == 0 {
            output.fill(0.0);
            return;
        }
        output
            .par_chunks_mut(self.output_dim)
            .zip(input.par_chunks(self.input_dim))
            .for_each(|(dst, src)| {
                // Nested work stays inside the same Rayon pool; it creates no
                // additional OS thread pool or native-library worker population.
                dst.par_chunks_mut(16).enumerate().for_each(|(block, out)| {
                    let packed =
                        &self.weights[block * self.pairs * 32..(block + 1) * self.pairs * 32];
                    match selected {
                        Backend::Scalar => {
                            for (lane, value) in out.iter_mut().enumerate() {
                                let mut sum = 0.0_f32;
                                for (inner, x) in src.iter().enumerate() {
                                    sum += x.to_f32()
                                        * packed[(inner / 2) * 32 + lane * 2 + inner % 2].to_f32();
                                }
                                *value = sum;
                            }
                        }
                        #[cfg(target_arch = "x86_64")]
                        Backend::Avx512Bf16 => {
                            // SAFETY: CPU features and tensor shapes were checked;
                            // packing pads both input pairs and output lanes.
                            unsafe {
                                x86::packed_block(src, packed, out);
                            }
                        }
                        _ => unreachable!("unresolved BF16 packed backend"),
                    }
                });
            });
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use half::bf16;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn load_bf16(ptr: *const bf16) -> __m512bh {
        // SAFETY: Caller provides at least 32 BF16 elements. The unaligned
        // integer load reads their bits; transmute changes only vector type.
        unsafe { std::mem::transmute(_mm512_loadu_si512(ptr.cast())) }
    }

    fn hardware_operand(value: bf16) -> f32 {
        let bits = value.to_bits();
        if bits & 0x7f80 == 0 {
            f32::from_bits(((bits & 0x8000) as u32) << 16)
        } else {
            value.to_f32()
        }
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn dot_bf16(input: &[bf16], weight: &[bf16]) -> f32 {
        debug_assert_eq!(input.len(), weight.len());
        unsafe {
            let mut a0 = _mm512_setzero_ps();
            let mut a1 = _mm512_setzero_ps();
            let mut a2 = _mm512_setzero_ps();
            let mut a3 = _mm512_setzero_ps();
            let mut index = 0;
            while index + 128 <= input.len() {
                a0 = _mm512_dpbf16_ps(
                    a0,
                    load_bf16(input.as_ptr().add(index)),
                    load_bf16(weight.as_ptr().add(index)),
                );
                a1 = _mm512_dpbf16_ps(
                    a1,
                    load_bf16(input.as_ptr().add(index + 32)),
                    load_bf16(weight.as_ptr().add(index + 32)),
                );
                a2 = _mm512_dpbf16_ps(
                    a2,
                    load_bf16(input.as_ptr().add(index + 64)),
                    load_bf16(weight.as_ptr().add(index + 64)),
                );
                a3 = _mm512_dpbf16_ps(
                    a3,
                    load_bf16(input.as_ptr().add(index + 96)),
                    load_bf16(weight.as_ptr().add(index + 96)),
                );
                index += 128;
            }
            let mut accumulated = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
            while index + 32 <= input.len() {
                accumulated = _mm512_dpbf16_ps(
                    accumulated,
                    load_bf16(input.as_ptr().add(index)),
                    load_bf16(weight.as_ptr().add(index)),
                );
                index += 32;
            }
            let mut sum = _mm512_reduce_add_ps(accumulated);
            while index < input.len() {
                // Keep BF16 input-subnormal treatment consistent in the tail.
                sum += hardware_operand(input[index]) * hardware_operand(weight[index]);
                index += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn pair(input: &[bf16], index: usize) -> __m512bh {
        let low = input[index].to_bits() as u32;
        let high = input
            .get(index + 1)
            .map_or(0, |value| value.to_bits() as u32);
        // All sixteen 32-bit lanes receive the same pair of BF16 bit patterns.
        unsafe { std::mem::transmute(_mm512_set1_epi32((low | (high << 16)) as i32)) }
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn packed_block(input: &[bf16], weights: &[bf16], output: &mut [f32]) {
        debug_assert!(output.len() <= 16);
        debug_assert_eq!(weights.len(), input.len().div_ceil(2) * 32);
        unsafe {
            let mut a0 = _mm512_setzero_ps();
            let mut a1 = _mm512_setzero_ps();
            let mut a2 = _mm512_setzero_ps();
            let mut a3 = _mm512_setzero_ps();
            let pairs = input.len().div_ceil(2);
            let mut p = 0;
            while p + 4 <= pairs {
                a0 = _mm512_dpbf16_ps(
                    a0,
                    pair(input, 2 * p),
                    load_bf16(weights.as_ptr().add(p * 32)),
                );
                a1 = _mm512_dpbf16_ps(
                    a1,
                    pair(input, 2 * (p + 1)),
                    load_bf16(weights.as_ptr().add((p + 1) * 32)),
                );
                a2 = _mm512_dpbf16_ps(
                    a2,
                    pair(input, 2 * (p + 2)),
                    load_bf16(weights.as_ptr().add((p + 2) * 32)),
                );
                a3 = _mm512_dpbf16_ps(
                    a3,
                    pair(input, 2 * (p + 3)),
                    load_bf16(weights.as_ptr().add((p + 3) * 32)),
                );
                p += 4;
            }
            let mut sum = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
            while p < pairs {
                sum = _mm512_dpbf16_ps(
                    sum,
                    pair(input, 2 * p),
                    load_bf16(weights.as_ptr().add(p * 32)),
                );
                p += 1;
            }
            let mask = ((1_u32 << output.len()) - 1) as __mmask16;
            _mm512_mask_storeu_ps(output.as_mut_ptr(), mask, sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<bf16> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bf16::from_f32(((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32)
            })
            .collect()
    }

    fn backends() -> Vec<Backend> {
        [Backend::Scalar, Backend::Auto, Backend::Avx512Bf16]
            .into_iter()
            .filter(|backend| backend.validate().is_ok())
            .collect()
    }

    fn linear_f64(
        input: &[bf16],
        rows: usize,
        input_dim: usize,
        weight: &[bf16],
        output_dim: usize,
    ) -> Vec<f64> {
        (0..rows * output_dim)
            .map(|index| {
                let row = index / output_dim;
                let channel = index % output_dim;
                (0..input_dim)
                    .map(|inner| {
                        input[row * input_dim + inner].to_f64()
                            * weight[channel * input_dim + inner].to_f64()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a as f64 - e).abs() <= 1.5e-4 + 5e-6 * e.abs(),
                "index {index}: {a} versus {e}"
            );
        }
    }

    #[test]
    fn explicit_conversion_uses_nearest_even_and_does_not_hide_output_casts() {
        let source = [
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f81_8000),
            -2.75,
        ];
        let mut result = [bf16::ZERO; 3];
        convert_f32_to_bf16(&source, &mut result);
        assert_eq!(
            result.map(bf16::to_bits),
            [0x3f80, 0x3f82, bf16::from_f32(-2.75).to_bits()]
        );
        let x = [bf16::from_f32(1.0078125)];
        let output = dot(&x, &x, Backend::Scalar);
        assert_eq!(output, 1.01568603515625);
        assert_ne!(
            output,
            bf16::from_f32(output).to_f32(),
            "FP32 output must not be silently BF16-rounded"
        );
    }

    #[test]
    fn dot_handles_every_vector_tail_against_f64_quantized_operands() {
        for len in [
            0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 767, 768, 769, 1024,
            2304,
        ] {
            let input = values(len, 71);
            let weight = values(len, 117);
            let expected = input
                .iter()
                .zip(&weight)
                .map(|(a, b)| a.to_f64() * b.to_f64())
                .sum::<f64>();
            for backend in backends() {
                assert_close(&[dot(&input, &weight, backend)], &[expected]);
            }
        }
    }

    #[test]
    fn actual_falcon_dimensions_and_small_batches_match_f64_oracle() {
        for (rows, input_dim, output_dim) in [
            (1, 768, 2048),
            (2, 768, 4608),
            (1, 2304, 768),
            (4, 768, 128),
            (8, 768, 128),
            (17, 67, 97),
            (3, 1, 19),
            (1, 129, 33),
        ] {
            let input = values(rows * input_dim, 31);
            let weight = values(output_dim * input_dim, 43);
            let expected = linear_f64(&input, rows, input_dim, &weight, output_dim);
            let packed = PackedLinear::new(&weight, input_dim, output_dim);
            for backend in backends() {
                let mut plain = vec![f32::NAN; expected.len()];
                let mut tiled = vec![f32::NAN; expected.len()];
                linear(
                    &input, rows, input_dim, &weight, output_dim, &mut plain, backend,
                );
                packed.linear(&input, rows, &mut tiled, backend);
                assert_close(&plain, &expected);
                assert_close(&tiled, &expected);
                if backend == Backend::Scalar {
                    assert_eq!(plain, tiled);
                }
            }
        }
    }

    #[test]
    fn packed_lane_order_padding_and_zero_dimensions() {
        let input: Vec<_> = [1.0, 2.0, 3.0].map(bf16::from_f32).to_vec();
        let weight: Vec<_> = (0..19)
            .flat_map(|channel| [channel as f32, 10.0, 100.0].map(bf16::from_f32))
            .collect();
        let packed = PackedLinear::new(&weight, 3, 19);
        assert_eq!(packed.packed_weight_bytes(), 2 * 2 * 32 * 2);
        for backend in backends() {
            let mut out = [f32::NAN; 19];
            packed.linear(&input, 1, &mut out, backend);
            for (channel, value) in out.into_iter().enumerate() {
                assert_eq!(value, 320.0 + channel as f32);
            }
            let empty = PackedLinear::new(&[], 0, 3);
            let mut zeros = [f32::NAN; 6];
            empty.linear(&[], 2, &mut zeros, backend);
            assert_eq!(zeros, [0.0; 6]);
            PackedLinear::new(&[], 3, 0).linear(&input, 1, &mut [], backend);
        }
    }

    #[test]
    fn avx512_input_subnormal_difference_is_explicit() {
        if !avx512_bf16_available() {
            return;
        }
        // A BF16 subnormal times a large normal gives a normal mathematical
        // product. VDPBF16PS instead treats that BF16 operand as signed zero.
        let x = vec![bf16::from_bits(1); 33];
        let w = vec![bf16::from_f32(1e30); 33];
        assert!(dot(&x, &w, Backend::Scalar) > 0.0);
        assert_eq!(dot(&x, &w, Backend::Avx512Bf16), 0.0);
        let packed = PackedLinear::new(&w, 33, 1);
        let mut out = [f32::NAN];
        packed.linear(&x, 1, &mut out, Backend::Avx512Bf16);
        assert_eq!(out, [0.0]);
    }

    #[test]
    #[should_panic(expected = "BF16 linear weight shape")]
    fn invalid_weight_shape_is_rejected_before_simd() {
        linear(
            &[bf16::ONE; 3],
            1,
            3,
            &[bf16::ONE],
            1,
            &mut [0.0],
            Backend::Auto,
        );
    }
}
