//! Integrated W8A32 experiment, derived from q8_reference.rs: signed [-127,127] row-major bytes, FP32
//! absmax/127 scale per K group, ties-to-even, FP32 dequant and ascending-K FMA.
//! Same rules as q4_reference except the code range and byte storage.
//! Neither activations nor accumulators are integer/BF16; this is not W8A8.

use anyhow::ensure;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Q8Linear {
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    codes: Vec<i8>,
    scales: Vec<f32>,
}

impl Q8Linear {
    pub fn quantize(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
    ) -> Result<Self, &'static str> {
        if out_dim == 0 || in_dim == 0 || ![64, 128].contains(&group_size) {
            return Err("nonzero shape and group size 64/128 required");
        }
        if out_dim.checked_mul(in_dim) != Some(weights.len()) {
            return Err("weight shape overflow or mismatch");
        }
        if weights.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite weight");
        }
        let groups = in_dim.div_ceil(group_size);
        let mut result = Self {
            out_dim,
            in_dim,
            group_size,
            codes: vec![0; weights.len()],
            scales: vec![0.; out_dim * groups],
        };
        for row in 0..out_dim {
            for group in 0..groups {
                let start = group * group_size;
                let end = (start + group_size).min(in_dim);
                let values = &weights[row * in_dim + start..row * in_dim + end];
                let maximum = values.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let scale = if maximum == 0.0 {
                    0.0
                } else {
                    ((maximum as f64 / 127.0) as f32).max(f32::from_bits(1))
                };
                result.scales[row * groups + group] = scale;
                for (i, &value) in values.iter().enumerate() {
                    let code = if scale == 0.0 {
                        0
                    } else {
                        (value as f64 / scale as f64)
                            .round_ties_even()
                            .clamp(-127., 127.) as i8
                    };
                    if !(code as f32 * scale).is_finite() {
                        return Err("dequantized value overflows FP32");
                    }
                    result.codes[row * in_dim + start + i] = code;
                }
            }
        }
        Ok(result)
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.out_dim, self.in_dim)
    }
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
    /// Tensor payload only, excluding headers/alignment/allocator overhead.
    pub fn payload_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * size_of::<f32>()
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        assert!(row < self.out_dim);
        assert_eq!(output.len(), self.in_dim);
        let groups = self.in_dim.div_ceil(self.group_size);
        for (column, value) in output.iter_mut().enumerate() {
            *value = self.codes[row * self.in_dim + column] as f32
                * self.scales[row * groups + column / self.group_size];
        }
    }

    pub fn linear_f32(
        &self,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<(), &'static str> {
        if rows.checked_mul(self.in_dim) != Some(input.len())
            || rows.checked_mul(self.out_dim) != Some(output.len())
        {
            return Err("linear shape overflow or mismatch");
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite activation");
        }
        let groups = self.in_dim.div_ceil(self.group_size);
        for row in 0..rows {
            for channel in 0..self.out_dim {
                let mut sum = 0.0f32;
                for k in 0..self.in_dim {
                    let w = self.codes[channel * self.in_dim + k] as f32
                        * self.scales[channel * groups + k / self.group_size];
                    sum = input[row * self.in_dim + k].mul_add(w, sum);
                }
                if !sum.is_finite() {
                    return Err("nonfinite accumulation");
                }
                output[row * self.out_dim + channel] = sum;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_ties_zero_and_signed_endpoints() {
        let mut weights = vec![0.; 128];
        weights[..8].copy_from_slice(&[254., -254., 1., 3., 5., -1., -3., -5.]);
        let q = Q8Linear::quantize(&weights, 1, 128, 64).unwrap();
        assert_eq!(q.scales(), &[2., 0.]);
        assert_eq!(&q.codes()[..8], &[127, -127, 0, 2, 2, 0, -2, -2]);
        let mut restored = vec![0.; 128];
        q.dequantize_row(0, &mut restored);
        assert_eq!(&restored[..8], &[254., -254., 0., 4., 4., 0., -4., -4.]);
        assert!(restored[8..].iter().all(|&x| x == 0.));
    }

    #[test]
    fn odd_rows_partial_groups_and_error_bound() {
        for k in [1usize, 31, 65, 127, 129, 257] {
            for g in [64, 128] {
                let weights: Vec<_> = (0..3 * k)
                    .map(|i| ((i * 137 % 127) as f32 - 63.) / 19.)
                    .collect();
                let q = Q8Linear::quantize(&weights, 3, k, g).unwrap();
                assert_eq!(q.payload_bytes(), 3 * k + 12 * k.div_ceil(g));
                for row in 0..3 {
                    let mut restored = vec![0.; k];
                    q.dequantize_row(row, &mut restored);
                    for (column, &w) in restored.iter().enumerate() {
                        let scale = q.scales()[row * k.div_ceil(g) + column / g] as f64;
                        let original = weights[row * k + column] as f64;
                        assert!(
                            (w as f64 - original).abs()
                                <= scale / 2. + 2. * f32::EPSILON as f64 * original.abs()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn real_reduction_widths_batches_and_f64_bound() {
        for k in [768usize, 1024, 2304] {
            for g in [64, 128] {
                let weights: Vec<_> = (0..7 * k)
                    .map(|i| ((i * 67 % 199) as f32 - 99.) / 31.)
                    .collect();
                let q = Q8Linear::quantize(&weights, 7, k, g).unwrap();
                for rows in [1, 2, 4, 8] {
                    let x: Vec<_> = (0..rows * k)
                        .map(|i| ((i * 101 % 257) as f32 - 128.) / 37.)
                        .collect();
                    let mut y = vec![0.; rows * 7];
                    q.linear_f32(&x, rows, &mut y).unwrap();
                    for channel in 0..7 {
                        let mut restored = vec![0.; k];
                        q.dequantize_row(channel, &mut restored);
                        for row in 0..rows {
                            let products: Vec<_> = x[row * k..(row + 1) * k]
                                .iter()
                                .zip(&restored)
                                .map(|(&a, &b)| a as f64 * b as f64)
                                .collect();
                            let sum: f64 = products.iter().sum();
                            let absolute_sum: f64 = products.iter().map(|x| x.abs()).sum();
                            let ku = k as f64 * f32::EPSILON as f64 / 2.;
                            assert!(
                                (y[row * 7 + channel] as f64 - sum).abs()
                                    <= ku / (1. - ku) * absolute_sum
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn finite_checks_subnormals_and_overflow() {
        let tiny = f32::from_bits(1);
        let q = Q8Linear::quantize(&[tiny, -tiny], 1, 2, 64).unwrap();
        let mut restored = [0.; 2];
        q.dequantize_row(0, &mut restored);
        assert_eq!(restored, [tiny, -tiny]);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(Q8Linear::quantize(&[value], 1, 1, 64).is_err());
        }
        assert!(Q8Linear::quantize(&[], usize::MAX, 2, 64).is_err());
        assert!(Q8Linear::quantize(&[], 0, 0, 64).is_err());
        assert!(Q8Linear::quantize(&[1.], 1, 1, 32).is_err());
        let q = Q8Linear::quantize(&[2., 2.], 1, 2, 64).unwrap();
        assert!(q.linear_f32(&[f32::NAN, 1.], 1, &mut [0.]).is_err());
        assert!(q.linear_f32(&[f32::MAX, f32::MAX], 1, &mut [0.]).is_err());
        assert!(q.linear_f32(&[], usize::MAX, &mut []).is_err());
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    pub dense: Vec<f32>,
    pub channel_major: Vec<f32>,
}
impl Scratch {
    pub fn reserve_decode(&mut self, rows: usize, vocabulary: usize) {
        self.channel_major.resize(rows * vocabulary, 0.0);
    }
}
impl Q8Linear {
    /// Import the exact custom W8G64 encoding, not GGML Q8_0/Q8_K.
    pub fn from_parts(
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        codes: Vec<i8>,
        scales: Vec<f32>,
    ) -> anyhow::Result<Self> {
        ensure!(
            out_dim > 0 && in_dim > 0 && group_size == 64,
            "invalid W8G64 shape/group"
        );
        ensure!(
            out_dim.checked_mul(in_dim) == Some(codes.len()),
            "W8 codes shape"
        );
        ensure!(
            out_dim.checked_mul(in_dim.div_ceil(group_size)) == Some(scales.len()),
            "W8 scales shape"
        );
        ensure!(
            codes.iter().all(|&v| v != -128),
            "W8 code -128 is outside [-127,127]"
        );
        ensure!(
            scales
                .iter()
                .all(|&s| s.is_finite() && s >= 0.0 && (127.0 * s).is_finite()),
            "invalid W8 scale"
        );
        for row in 0..out_dim {
            for g in 0..in_dim.div_ceil(group_size) {
                if scales[row * in_dim.div_ceil(group_size) + g] == 0.0 {
                    let a = row * in_dim + g * group_size;
                    let b = (a + group_size).min((row + 1) * in_dim);
                    ensure!(
                        codes[a..b].iter().all(|&v| v == 0),
                        "nonzero code in zero-scale group"
                    );
                }
            }
        }
        Ok(Self {
            out_dim,
            in_dim,
            group_size,
            codes,
            scales,
        })
    }

    /// Same reconstructed weights at every phase. Large-M uses one reusable
    /// dense scratch matrix, never the original full-precision weights.
    pub(crate) fn linear(
        &self,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
        scratch: &mut Scratch,
        simd: crate::kernels::Simd,
    ) -> anyhow::Result<()> {
        ensure!(
            rows.checked_mul(self.in_dim) == Some(input.len()),
            "W8 input shape"
        );
        ensure!(
            rows.checked_mul(self.out_dim) == Some(output.len()),
            "W8 output shape"
        );
        ensure!(
            input.iter().all(|x| x.is_finite()),
            "nonfinite W8 activation"
        );
        simd.validate().map_err(anyhow::Error::msg)?;
        if rows == 0 {
            return Ok(());
        }
        if rows > 8 {
            scratch.dense.resize(self.codes.len(), 0.0);
            scratch
                .dense
                .par_chunks_mut(self.in_dim)
                .enumerate()
                .for_each(|(row, dst)| self.dequantize_row(row, dst));
            crate::kernels::linear_with_simd(
                input,
                rows,
                self.in_dim,
                &scratch.dense,
                self.out_dim,
                output,
                simd,
            );
        } else {
            let selected = simd.resolved();
            #[cfg(target_arch = "x86_64")]
            let use_avx2 = selected != crate::kernels::Simd::Scalar
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma");
            #[cfg(not(target_arch = "x86_64"))]
            let use_avx2 = {
                let _ = selected;
                false
            };
            scratch.channel_major.resize(rows * self.out_dim, 0.0);
            // A task owns all active rows for one output channel. Weight values
            // are unpacked once per channel and reused across up to eight rows.
            scratch
                .channel_major
                .par_chunks_mut(rows)
                .enumerate()
                .with_min_len(16)
                .for_each(|(channel, dst)| {
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        // SAFETY: feature and shape checks above bound all loads.
                        unsafe {
                            self.channel_avx2(input, rows, channel, dst);
                        }
                        return;
                    }
                    let _ = use_avx2;
                    self.channel_scalar(input, rows, channel, dst);
                });
            output
                .par_chunks_mut(self.out_dim)
                .enumerate()
                .for_each(|(r, dst)| {
                    for (c, y) in dst.iter_mut().enumerate() {
                        *y = scratch.channel_major[c * rows + r];
                    }
                });
        }
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "nonfinite W8 accumulation"
        );
        Ok(())
    }
    fn channel_scalar(&self, input: &[f32], rows: usize, channel: usize, output: &mut [f32]) {
        output.fill(0.0);
        let groups = self.in_dim.div_ceil(self.group_size);
        for k in 0..self.in_dim {
            let w = self.codes[channel * self.in_dim + k] as f32
                * self.scales[channel * groups + k / self.group_size];
            for r in 0..rows {
                output[r] = input[r * self.in_dim + k].mul_add(w, output[r]);
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn channel_avx2(&self, input: &[f32], rows: usize, channel: usize, output: &mut [f32]) {
        use std::arch::x86_64::*;
        // Every eight-byte code load is inside a row/group; tails use scalar
        // reads. No global target-cpu=native or integer activation approximation.
        unsafe {
            let mut accum = [_mm256_setzero_ps(); 8];
            let mut tails = [0.0_f32; 8];
            let groups = self.in_dim.div_ceil(self.group_size);
            for group in 0..groups {
                let scale = self.scales[channel * groups + group];
                let sv = _mm256_set1_ps(scale);
                let mut k = group * self.group_size;
                let end = (k + self.group_size).min(self.in_dim);
                while k + 8 <= end {
                    let packed =
                        _mm_loadl_epi64(self.codes.as_ptr().add(channel * self.in_dim + k).cast());
                    let w = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed)), sv);
                    for r in 0..rows {
                        let x = _mm256_loadu_ps(input.as_ptr().add(r * self.in_dim + k));
                        accum[r] = _mm256_fmadd_ps(x, w, accum[r]);
                    }
                    k += 8;
                }
                for k in k..end {
                    let w = self.codes[channel * self.in_dim + k] as f32 * scale;
                    for r in 0..rows {
                        tails[r] = input[r * self.in_dim + k].mul_add(w, tails[r]);
                    }
                }
            }
            for r in 0..rows {
                let mut lane = [0.0_f32; 8];
                _mm256_storeu_ps(lane.as_mut_ptr(), accum[r]);
                let a = (lane[0] + lane[1]) + (lane[2] + lane[3]);
                let b = (lane[4] + lane[5]) + (lane[6] + lane[7]);
                output[r] = (a + b) + tails[r];
            }
        }
    }
}

#[cfg(test)]
mod integrated_tests {
    use super::*;
    use crate::kernels::Simd;
    #[test]
    fn integrated_shapes_and_phase_values() {
        for k in [31, 64, 65, 768, 1024, 2304] {
            let n = 9;
            let w: Vec<_> = (0..n * k)
                .map(|i| ((i * 137 % 193) as f32 - 96.0) / 113.0)
                .collect();
            let q = Q8Linear::quantize(&w, n, k, 64).unwrap();
            for rows in [1, 2, 4, 8, 9, 17] {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 31 % 151) as f32 - 75.0) / 97.0)
                    .collect();
                let mut dense = vec![0.0; n * k];
                for r in 0..n {
                    q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
                }
                for backend in [Simd::Scalar, Simd::Auto] {
                    let mut out = vec![0.0; rows * n];
                    let mut scratch = Scratch::default();
                    q.linear(&x, rows, &mut out, &mut scratch, backend).unwrap();
                    for r in 0..rows {
                        for c in 0..n {
                            let mut expected = 0.0_f64;
                            let mut magnitude = 0.0_f64;
                            for j in 0..k {
                                let p = x[r * k + j] as f64 * dense[c * k + j] as f64;
                                expected += p;
                                magnitude += p.abs();
                            }
                            assert!(
                                (out[r * n + c] as f64 - expected).abs()
                                    <= 4.0 * k as f64 * f32::EPSILON as f64 * magnitude + 1e-6
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn artifact_validation() {
        assert!(Q8Linear::from_parts(1, 1, 64, vec![-128], vec![1.0]).is_err());
        assert!(Q8Linear::from_parts(1, 1, 64, vec![1], vec![0.0]).is_err());
        assert!(Q8Linear::from_parts(1, 1, 64, vec![0], vec![f32::NAN]).is_err());
    }
}
