//! Standalone research oracle; not compiled into the production runner.
//! W4A32: row-major signed [-7,7] weights, adjacent low/high nibbles,
//! one FP32 absmax/7 scale per K group, ties-to-even, FP32 dequant and FMA.
//! This is deliberately NOT the GGML Q4_0/Q4_K bitstream or an HF dtype.

#[derive(Debug)]
pub struct Q4Linear {
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
}

impl Q4Linear {
    pub fn quantize(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
    ) -> Result<Self, &'static str> {
        if out_dim == 0 || in_dim == 0 || ![32, 64, 128, 256].contains(&group_size) {
            return Err("nonzero shape and group size 32/64/128/256 required");
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
            packed: vec![0; out_dim * in_dim.div_ceil(2)],
            scales: vec![0.0; out_dim * groups],
        };
        for row in 0..out_dim {
            for group in 0..groups {
                let lo = group * group_size;
                let hi = (lo + group_size).min(in_dim);
                let values = &weights[row * in_dim + lo..row * in_dim + hi];
                let maximum = values.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let scale = if maximum == 0.0 {
                    0.0
                } else {
                    // F64 division avoids a reciprocal overflow for tiny values.
                    ((maximum as f64 / 7.0) as f32).max(f32::from_bits(1))
                };
                result.scales[row * groups + group] = scale;
                for (i, &value) in values.iter().enumerate() {
                    let code = if scale == 0.0 {
                        0i8
                    } else {
                        (value as f64 / scale as f64)
                            .round_ties_even()
                            .clamp(-7.0, 7.0) as i8
                    };
                    if !(code as f32 * scale).is_finite() {
                        return Err("dequantized value overflows FP32");
                    }
                    let column = lo + i;
                    let offset = row * in_dim.div_ceil(2) + column / 2;
                    result.packed[offset] |= ((code as u8) & 15) << (4 * (column % 2));
                }
            }
        }
        Ok(result)
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.out_dim, self.in_dim)
    }

    /// Read-only layout metadata for isolated checked operator experiments.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Payload only; excludes Vec headers, allocator padding and file metadata.
    pub fn payload_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * size_of::<f32>()
    }

    /// Diagnostic serialization accessors; do not change the settled format.
    pub fn packed_codes(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    fn code(&self, row: usize, column: usize) -> i8 {
        let byte = self.packed[row * self.in_dim.div_ceil(2) + column / 2];
        let unsigned = ((byte >> (4 * (column % 2))) & 15) as i8;
        if unsigned >= 8 {
            unsigned - 16
        } else {
            unsigned
        }
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        assert!(row < self.out_dim);
        assert_eq!(output.len(), self.in_dim);
        let groups = self.in_dim.div_ceil(self.group_size);
        for (column, value) in output.iter_mut().enumerate() {
            *value = self.code(row, column) as f32
                * self.scales[row * groups + column / self.group_size];
        }
    }

    /// Allocation-free scalar operator; deliberately no SIMD or thread claims.
    pub fn linear_f32(&self, input: &[f32], rows: usize, output: &mut [f32]) {
        assert_eq!(rows.checked_mul(self.in_dim), Some(input.len()));
        assert_eq!(rows.checked_mul(self.out_dim), Some(output.len()));
        let groups = self.in_dim.div_ceil(self.group_size);
        for row in 0..rows {
            for channel in 0..self.out_dim {
                let mut sum = 0.0f32;
                for k in 0..self.in_dim {
                    let weight = self.code(channel, k) as f32
                        * self.scales[channel * groups + k / self.group_size];
                    sum = input[row * self.in_dim + k].mul_add(weight, sum);
                }
                output[row * self.out_dim + channel] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_known_codes_ties_and_zero_groups() {
        let mut weights = vec![0.0; 64];
        weights[..8].copy_from_slice(&[14.0, -14.0, 1.0, 3.0, 5.0, -1.0, -3.0, -5.0]);
        let q = Q4Linear::quantize(&weights, 1, 64, 32).unwrap();
        assert_eq!(q.scales, [2.0, 0.0]);
        assert_eq!(&q.packed[..4], &[0x97, 0x20, 0x02, 0xee]);
        let mut restored = vec![0.0; 64];
        q.dequantize_row(0, &mut restored);
        assert_eq!(
            &restored[..8],
            &[14.0, -14.0, 0.0, 4.0, 4.0, 0.0, -4.0, -4.0]
        );
        assert!(restored[8..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn tails_do_not_cross_rows_and_group_error_is_bounded() {
        for width in [1, 31, 33, 65, 257] {
            for group in [32, 64, 128, 256] {
                let weights: Vec<f32> = (0..3 * width)
                    .map(|i| ((i * 137 % 127) as f32 - 63.0) / 19.0)
                    .collect();
                let q = Q4Linear::quantize(&weights, 3, width, group).unwrap();
                assert_eq!(
                    q.payload_bytes(),
                    3 * width.div_ceil(2) + 12 * width.div_ceil(group)
                );
                for row in 0..3 {
                    let mut restored = vec![0.0; width];
                    q.dequantize_row(row, &mut restored);
                    for k in 0..width {
                        let scale = q.scales[row * width.div_ceil(group) + k / group] as f64;
                        let original = weights[row * width + k] as f64;
                        let error = (original - restored[k] as f64).abs();
                        let bound = scale / 2.0 + 2.0 * f32::EPSILON as f64 * original.abs();
                        assert!(error <= bound, "quantization error {error} > {bound}");
                    }
                    if width % 2 == 1 {
                        assert_eq!(q.packed[(row + 1) * width.div_ceil(2) - 1] & 0xf0, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn falcon_reduction_widths_and_batches_match_independent_f64_dequantized_oracle() {
        // Real K dimensions, with odd output counts to cover future SIMD tails.
        // Quantization error is separate from this arithmetic error comparison.
        for width in [768usize, 1024, 2304] {
            for group in [32, 64, 128, 256] {
                let weights: Vec<f32> = (0..7 * width)
                    .map(|i| ((i * 67 % 199) as f32 - 99.0) / 31.0)
                    .collect();
                let q = Q4Linear::quantize(&weights, 7, width, group).unwrap();
                for rows in [1, 2, 4, 8] {
                    let input: Vec<f32> = (0..rows * width)
                        .map(|i| ((i * 101 % 257) as f32 - 128.0) / 37.0)
                        .collect();
                    let mut actual = vec![0.0; rows * 7];
                    q.linear_f32(&input, rows, &mut actual);
                    for channel in 0..7 {
                        let mut restored = vec![0.0; width];
                        q.dequantize_row(channel, &mut restored);
                        for row in 0..rows {
                            let products: Vec<f64> = input[row * width..(row + 1) * width]
                                .iter()
                                .zip(&restored)
                                .map(|(&a, &b)| a as f64 * b as f64)
                                .collect();
                            let expected: f64 = products.iter().sum();
                            let absolute_sum: f64 = products.iter().map(|x| x.abs()).sum();
                            let nu = width as f64 * (f32::EPSILON as f64 / 2.0);
                            let bound = nu / (1.0 - nu) * absolute_sum;
                            assert!((actual[row * 7 + channel] as f64 - expected).abs() <= bound);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn finite_subnormals_and_invalid_inputs_are_explicit() {
        let tiny = f32::from_bits(1);
        let q = Q4Linear::quantize(&[tiny, -tiny], 1, 2, 64).unwrap();
        let mut out = [0.0; 2];
        q.dequantize_row(0, &mut out);
        assert_eq!(out, [tiny, -tiny]);
        assert!(Q4Linear::quantize(&[f32::NAN], 1, 1, 64).is_err());
        assert!(Q4Linear::quantize(&[f32::INFINITY], 1, 1, 64).is_err());
        assert!(Q4Linear::quantize(&[], usize::MAX, 2, 64).is_err());
        assert!(Q4Linear::quantize(&[], 0, 0, 64).is_err());
        assert!(Q4Linear::quantize(&[1.0], 1, 1, 63).is_err());
    }
}
