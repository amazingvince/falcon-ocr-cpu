//! Integrated W8A32 / W16A32 experiment, derived from q8_reference.rs: signed
//! row-major codes in [-127,127] (8-bit) or [-32767,32767] (16-bit), an FP32
//! absmax/qmax scale per K group, ties-to-even, FP32 dequant and ascending-K
//! FMA. Neither activations nor accumulators are integer/BF16; this is not
//! W8A8. 16-bit codes are effectively lossless against the FP32 weights
//! (activation-weighted output error about 2.5e-5 relative).

use anyhow::ensure;
use rayon::prelude::*;

/// Row-major weight codes of one matrix (owned, or mapped from a
/// kernel-ready model file).
#[derive(Debug)]
enum Codes {
    I8(crate::buf::Buf<i8>),
    I16(crate::buf::Buf<i16>),
}

/// A quantized linear layer with 8- or 16-bit codes (the name predates
/// 16-bit support).
#[derive(Debug)]
pub struct Q8Linear {
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    codes: Codes,
    scales: crate::buf::Buf<f32>,
}

impl Q8Linear {
    /// 8-bit codes; see [`Q8Linear::quantize_bits`].
    pub fn quantize(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
    ) -> Result<Self, &'static str> {
        Self::quantize_bits(weights, out_dim, in_dim, group_size, 8)
    }

    /// Round-to-nearest (ties to even) codes of `bits` = 8 or 16 with one
    /// absmax scale per `group_size` inputs of each output row.
    pub fn quantize_bits(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        bits: u32,
    ) -> Result<Self, &'static str> {
        if bits != 8 && bits != 16 {
            return Err("8- or 16-bit codes required");
        }
        let qmax = if bits == 8 { 127.0 } else { 32767.0 };
        if out_dim == 0 || in_dim == 0 || ![32, 64, 128].contains(&group_size) {
            return Err("nonzero shape and group size 32/64/128 required");
        }
        if out_dim.checked_mul(in_dim) != Some(weights.len()) {
            return Err("weight shape overflow or mismatch");
        }
        if weights.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite weight");
        }
        let groups = in_dim.div_ceil(group_size);
        let mut scales = vec![0.; out_dim * groups];
        let mut codes = vec![0i32; weights.len()];
        for row in 0..out_dim {
            for group in 0..groups {
                let start = group * group_size;
                let end = (start + group_size).min(in_dim);
                let values = &weights[row * in_dim + start..row * in_dim + end];
                let maximum = values.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let scale = if maximum == 0.0 {
                    0.0
                } else {
                    ((maximum as f64 / qmax) as f32).max(f32::from_bits(1))
                };
                scales[row * groups + group] = scale;
                for (i, &value) in values.iter().enumerate() {
                    let code = if scale == 0.0 {
                        0
                    } else {
                        (value as f64 / scale as f64)
                            .round_ties_even()
                            .clamp(-qmax, qmax) as i32
                    };
                    if !(code as f32 * scale).is_finite() {
                        return Err("dequantized value overflows FP32");
                    }
                    codes[row * in_dim + start + i] = code;
                }
            }
        }
        let codes = if bits == 8 {
            Codes::I8(crate::buf::Buf::Owned(codes.into_iter().map(|c| c as i8).collect()))
        } else {
            Codes::I16(crate::buf::Buf::Owned(codes.into_iter().map(|c| c as i16).collect()))
        };
        Ok(Self {
            out_dim,
            in_dim,
            group_size,
            codes,
            scales: crate::buf::Buf::Owned(scales),
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.out_dim, self.in_dim)
    }
    /// The 8-bit codes (panics for a 16-bit matrix).
    pub fn codes(&self) -> &[i8] {
        match &self.codes {
            Codes::I8(codes) => codes,
            Codes::I16(_) => panic!("16-bit codes"),
        }
    }
    /// Bits per code: 8 or 16.
    pub fn bits(&self) -> u32 {
        match self.codes {
            Codes::I8(_) => 8,
            Codes::I16(_) => 16,
        }
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
    /// Tensor payload only, excluding headers/alignment/allocator overhead.
    pub fn payload_bytes(&self) -> usize {
        let codes = match &self.codes {
            Codes::I8(codes) => codes.len(),
            Codes::I16(codes) => 2 * codes.len(),
        };
        codes + self.scales.len() * size_of::<f32>()
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        assert!(row < self.out_dim);
        assert_eq!(output.len(), self.in_dim);
        let groups = self.in_dim.div_ceil(self.group_size);
        let scales = &self.scales[row * groups..(row + 1) * groups];
        let span = row * self.in_dim..(row + 1) * self.in_dim;
        match &self.codes {
            Codes::I8(codes) => fill_row(&codes[span], scales, self.group_size, output),
            Codes::I16(codes) => fill_row(&codes[span], scales, self.group_size, output),
        }
    }

    /// `fl(code * scale)` for element `index` (row-major).
    fn weight(&self, index: usize) -> f32 {
        let groups = self.in_dim.div_ceil(self.group_size);
        let (row, column) = (index / self.in_dim, index % self.in_dim);
        let code = match &self.codes {
            Codes::I8(codes) => codes[index] as f32,
            Codes::I16(codes) => codes[index] as f32,
        };
        code * self.scales[row * groups + column / self.group_size]
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
        for row in 0..rows {
            for channel in 0..self.out_dim {
                let mut sum = 0.0f32;
                for k in 0..self.in_dim {
                    let w = self.weight(channel * self.in_dim + k);
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
        assert!(Q8Linear::quantize(&[1.], 1, 1, 16).is_err());
        let q = Q8Linear::quantize(&[2., 2.], 1, 2, 64).unwrap();
        assert!(q.linear_f32(&[f32::NAN, 1.], 1, &mut [0.]).is_err());
        assert!(q.linear_f32(&[f32::MAX, f32::MAX], 1, &mut [0.]).is_err());
        assert!(q.linear_f32(&[], usize::MAX, &mut []).is_err());
    }
}

/// Large-M expansion target; decode (1..=8 rows) writes outputs directly.
#[derive(Default)]
pub(crate) struct Scratch {
    pub dense: Vec<f32>,
    pub bf16: crate::kernels::panel_bf16::Panels,
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
            out_dim > 0 && in_dim > 0 && [32, 64].contains(&group_size),
            "invalid W8 shape/group (32 or 64)"
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
            codes: Codes::I8(crate::buf::Buf::Owned(codes)),
            scales: crate::buf::Buf::Owned(scales),
        })
    }

    /// A matrix whose codes (`bits` = 8 or 16) and FP32 scales are byte
    /// ranges of a kernel-ready model file, used in place. Shapes, sizes and
    /// alignment are checked; code values are trusted (the file was written
    /// by `Model::write_packed` from a validated matrix).
    pub(crate) fn from_mapped(
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        bits: u32,
        map: &std::sync::Arc<memmap2::Mmap>,
        codes: std::ops::Range<usize>,
        scales: std::ops::Range<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            out_dim > 0 && in_dim > 0 && [32, 64].contains(&group_size),
            "invalid mapped matrix shape/group"
        );
        let elements = out_dim * in_dim;
        let codes = match bits {
            8 => Codes::I8(crate::buf::Buf::mapped(map, codes, elements)?),
            16 => Codes::I16(crate::buf::Buf::mapped(map, codes, elements)?),
            _ => anyhow::bail!("mapped codes must be 8 or 16 bits"),
        };
        let scales = crate::buf::Buf::mapped(map, scales, out_dim * in_dim.div_ceil(group_size))?;
        Ok(Self {
            out_dim,
            in_dim,
            group_size,
            codes,
            scales,
        })
    }

    /// Group size of the scales.
    pub(crate) fn group_size(&self) -> usize {
        self.group_size
    }

    /// Raw code bytes (i8 or little-endian i16) and scale bytes.
    pub(crate) fn raw_parts(&self) -> (&[u8], &[u8]) {
        let codes = match &self.codes {
            Codes::I8(c) => c.bytes(),
            Codes::I16(c) => c.bytes(),
        };
        (codes, self.scales.bytes())
    }

    /// Same reconstructed weights at every phase. Large-M uses one reusable
    /// dense scratch matrix, never the original full-precision weights.
    ///
    /// 1..=8 rows compute each output with the FP32 GEMV kernel's exact
    /// operation sequence over `fl(code * scale)` weights (four phase
    /// accumulators, the same reduction tree and scalar tail on AVX2; the same
    /// iterator sum on the scalar path). The result is therefore bitwise equal
    /// to `kernels::linear_with_simd` on the dequantized matrix, which also
    /// makes large-M (dequantize, then the same GEMM) consistent by
    /// construction. NaN/Inf are not scanned here: they propagate exactly as in
    /// the FP32 graph and are rejected by greedy selection.
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
        simd.validate().map_err(anyhow::Error::msg)?;
        if rows == 0 {
            return Ok(());
        }
        if rows > 8 && self.panel_gemm(simd) {
            // Prefill: dequantize once into cache-resident panels (see
            // `kernels::panel_gemm`); rounding-level different from the path below.
            use crate::kernels::panel_gemm;
            panel_gemm::pack_panels(
                self.out_dim,
                self.in_dim,
                |r, dst| self.dequantize_row(r, dst),
                &mut scratch.dense,
            );
            panel_gemm::gemm(
                input,
                rows,
                self.in_dim,
                &scratch.dense,
                self.out_dim,
                None,
                panel_gemm::Epilogue::Store(output),
            );
            return Ok(());
        }
        if rows > 8 {
            scratch.dense.resize(self.out_dim * self.in_dim, 0.0);
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
            return Ok(());
        }
        let output = &OutputCell(output.as_mut_ptr());
        self.small_m(input, rows, simd, |channel, row, value| {
            // SAFETY: (row, channel) lies inside `rows * out_dim`; one task
            // writes each channel for all rows.
            unsafe { *output_ptr(output).add(row * self.out_dim + channel) = value };
        });
        Ok(())
    }

    /// Fused W13 projection and squared-ReLU gate for 1..=8 rows on the
    /// interleaved `[gate_i, up_i]` layout: `gated[row][i]` uses the same two
    /// dot products and gate expression as `linear` followed by
    /// `kernels::squared_relu_gate`, bit for bit.
    pub(crate) fn linear_glu(
        &self,
        input: &[f32],
        rows: usize,
        gated: &mut [f32],
        scratch: &mut Scratch,
        simd: crate::kernels::Simd,
    ) -> anyhow::Result<bool> {
        ensure!(
            rows.checked_mul(self.in_dim) == Some(input.len()),
            "W8 input shape"
        );
        ensure!(self.out_dim % 2 == 0, "GLU needs interleaved gate/up rows");
        let ffn = self.out_dim / 2;
        ensure!(rows.checked_mul(ffn) == Some(gated.len()), "W8 GLU shape");
        simd.validate().map_err(anyhow::Error::msg)?;
        if rows > 8 && self.panel_gemm(simd) {
            // Prefill: the gate runs in the GEMM epilogue, so the interleaved
            // intermediate is never stored.
            use crate::kernels::panel_gemm;
            panel_gemm::pack_panels(
                self.out_dim,
                self.in_dim,
                |r, dst| self.dequantize_row(r, dst),
                &mut scratch.dense,
            );
            panel_gemm::gemm(
                input,
                rows,
                self.in_dim,
                &scratch.dense,
                self.out_dim,
                None,
                panel_gemm::Epilogue::Glu(gated),
            );
            return Ok(true);
        }
        if rows == 0 || rows > 8 {
            return Ok(false);
        }
        let out = OutputPtr(gated.as_mut_ptr());
        let in_dim = self.in_dim;
        let dot = self.dot_fn(simd);
        let dot_rows = if rows > 1 { self.dot_rows_fn(simd) } else { DotRows::None };
        let tasks = balanced_tasks(ffn);
        let block = ffn.div_ceil(tasks);
        crate::team::for_each(ffn.div_ceil(block), |b| {
            let out = out.get();
            let (mut gates, mut ups) = ([0.0_f32; 8], [0.0_f32; 8]);
            for i in b * block..((b + 1) * block).min(ffn) {
                let (gate_row, up_row) = (2 * i, 2 * i + 1);
                if self.rows_dot(dot_rows, input, rows, gate_row, &mut gates) {
                    self.rows_dot(dot_rows, input, rows, up_row, &mut ups);
                    for row in 0..rows {
                        // SAFETY: (row, i) lies inside `rows * ffn`; disjoint tasks.
                        unsafe { *out.add(row * ffn + i) = crate::kernels::squared_relu_glu(gates[row], ups[row]) };
                    }
                    continue;
                }
                for row in 0..rows {
                    let x = &input[row * in_dim..(row + 1) * in_dim];
                    let gate = self.row_dot(dot, x, gate_row);
                    let up = self.row_dot(dot, x, up_row);
                    // SAFETY: (row, i) lies inside `rows * ffn`; disjoint tasks.
                    unsafe { *out.add(row * ffn + i) = crate::kernels::squared_relu_glu(gate, up) };
                }
            }
        });
        Ok(true)
    }

    /// Prefill product through `kernels::panel_gemm` with a folded row scale
    /// (RMS norm) and a fused epilogue (store, gate or residual add). The
    /// caller checks `panel_gemm` first.
    pub(crate) fn prefill(
        &self,
        input: &[f32],
        rows: usize,
        row_scale: Option<&[f32]>,
        epilogue: crate::kernels::panel_gemm::Epilogue<'_>,
        scratch: &mut Scratch,
    ) {
        use crate::kernels::{panel_bf16, panel_gemm};
        assert_eq!(input.len(), rows * self.in_dim, "prefill input shape");
        if let Codes::I8(codes) = &self.codes
            && self.group_size == 64
            && self.out_dim % panel_bf16::NR == 0
            && panel_bf16::projections()
        {
            // 8-bit codes are exact in BF16: `vdpbf16ps` doubles FMA
            // throughput and only the activations round (see `panel_bf16`).
            panel_bf16::pack(self.out_dim, self.in_dim, codes, &self.scales, &mut scratch.bf16);
            panel_bf16::gemm(input, rows, self.in_dim, &scratch.bf16, self.out_dim, row_scale, epilogue);
            return;
        }
        panel_gemm::pack_panels(
            self.out_dim,
            self.in_dim,
            |r, dst| self.dequantize_row(r, dst),
            &mut scratch.dense,
        );
        panel_gemm::gemm(input, rows, self.in_dim, &scratch.dense, self.out_dim, row_scale, epilogue);
    }

    /// Whether large-M products use `kernels::panel_gemm` here.
    pub(crate) fn panel_gemm(&self, simd: crate::kernels::Simd) -> bool {
        crate::kernels::panel_gemm::available(simd)
            && self.out_dim % crate::kernels::panel_gemm::NR == 0
    }

    /// The dot kernel for this matrix's code type, group size and backend.
    fn dot_fn(&self, simd: crate::kernels::Simd) -> Dot {
        match self.codes {
            Codes::I8(_) => Dot::I8(select_dot::<i8>(self.group_size, simd)),
            Codes::I16(_) => Dot::I16(select_dot::<i16>(self.group_size, simd)),
        }
    }

    /// The multi-row dot kernel for this matrix (`DotRows::None` where the
    /// backend has none).
    fn dot_rows_fn(&self, simd: crate::kernels::Simd) -> DotRows {
        match self.codes {
            Codes::I8(_) => select_dot_rows::<i8>(self.group_size, simd).map_or(DotRows::None, DotRows::I8),
            Codes::I16(_) => select_dot_rows::<i16>(self.group_size, simd).map_or(DotRows::None, DotRows::I16),
        }
    }

    /// `row_dot` of every row of `input` (`rows <= 8`) with output row
    /// `channel` into `out`, bitwise per row; false if `dot` is `None`.
    #[inline(always)]
    fn rows_dot(&self, dot: DotRows, input: &[f32], rows: usize, channel: usize, out: &mut [f32; 8]) -> bool {
        let (in_dim, groups) = (self.in_dim, self.in_dim.div_ceil(self.group_size));
        let scales = &self.scales[channel * groups..(channel + 1) * groups];
        let span = channel * in_dim..(channel + 1) * in_dim;
        match (dot, &self.codes) {
            (DotRows::I8(f), Codes::I8(codes)) => f(input, in_dim, rows, &codes[span], scales, out),
            (DotRows::I16(f), Codes::I16(codes)) => f(input, in_dim, rows, &codes[span], scales, out),
            (DotRows::None, _) => return false,
            _ => unreachable!("dot kernel selected for another code type"),
        }
        true
    }

    /// `dot` of `x` with output row `channel`.
    #[inline(always)]
    fn row_dot(&self, dot: Dot, x: &[f32], channel: usize) -> f32 {
        let (in_dim, groups) = (self.in_dim, self.in_dim.div_ceil(self.group_size));
        let scales = &self.scales[channel * groups..(channel + 1) * groups];
        let span = channel * in_dim..(channel + 1) * in_dim;
        match (dot, &self.codes) {
            (Dot::I8(f), Codes::I8(codes)) => f(x, &codes[span], scales),
            (Dot::I16(f), Codes::I16(codes)) => f(x, &codes[span], scales),
            _ => unreachable!("dot kernel selected for another code type"),
        }
    }

    /// Every (channel, row) output of a 1..=8-row product, in balanced
    /// channel blocks; a task reuses a channel's codes across rows.
    fn small_m(
        &self,
        input: &[f32],
        rows: usize,
        simd: crate::kernels::Simd,
        write: impl Fn(usize, usize, f32) + Sync,
    ) {
        let (in_dim, out_dim) = (self.in_dim, self.out_dim);
        let dot = self.dot_fn(simd);
        // Several rows (draft verification) decode each weight chunk once.
        let dot_rows = if rows > 1 { self.dot_rows_fn(simd) } else { DotRows::None };
        let block = out_dim.div_ceil(balanced_tasks(out_dim));
        crate::team::for_each(out_dim.div_ceil(block), |b| {
            let mut values = [0.0_f32; 8];
            for channel in b * block..((b + 1) * block).min(out_dim) {
                if self.rows_dot(dot_rows, input, rows, channel, &mut values) {
                    for (row, &value) in values[..rows].iter().enumerate() {
                        write(channel, row, value);
                    }
                    continue;
                }
                for row in 0..rows {
                    write(
                        channel,
                        row,
                        self.row_dot(dot, &input[row * in_dim..(row + 1) * in_dim], channel),
                    );
                }
            }
        });
    }
}

type DotQ<C> = fn(&[f32], &[C], &[f32]) -> f32;
/// `out[r] = dot(x[r * stride..], codes, scales)` for `rows <= 8`, bitwise
/// the single-row [`DotQ`] per row, decoding the weights once.
type DotRowsQ<C> = fn(&[f32], usize, usize, &[C], &[f32], &mut [f32; 8]);

/// A multi-row dot kernel, where the backend has one.
#[derive(Clone, Copy)]
enum DotRows {
    I8(DotRowsQ<i8>),
    I16(DotRowsQ<i16>),
    None,
}

#[derive(Clone, Copy)]
enum Dot {
    I8(DotQ<i8>),
    I16(DotQ<i16>),
}

/// `output[k] = fl(codes[k] * scales[k / group])`.
fn fill_row<C: crate::simd::QCode>(codes: &[C], scales: &[f32], group: usize, output: &mut [f32]) {
    for (k, (value, &code)) in output.iter_mut().zip(codes).enumerate() {
        *value = code.to_f32() * scales[k / group];
    }
}

/// The dot kernel for code type `C`, `group` and the selected backend.
fn select_dot<C: crate::simd::QCode>(group: usize, simd: crate::kernels::Simd) -> DotQ<C> {
    let selected = simd.resolved();
    #[cfg(target_arch = "x86_64")]
    if selected != crate::kernels::Simd::Scalar
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        return match group {
            // SAFETY (all): AVX2/FMA detected above.
            32 => |x, c, s| unsafe { dot_q_avx2::<C, 32>(x, c, s) },
            64 => |x, c, s| unsafe { dot_q_avx2::<C, 64>(x, c, s) },
            _ => |x, c, s| unsafe { dot_q_avx2::<C, 128>(x, c, s) },
        };
    }
    #[cfg(target_arch = "aarch64")]
    if selected == crate::kernels::Simd::Neon {
        return match group {
            // SAFETY (all): NEON is baseline on aarch64.
            32 => |x, c, s| unsafe { crate::simd::dot_q::<crate::simd::Neon, C, 32>(x, c, s) },
            64 => |x, c, s| unsafe { crate::simd::dot_q::<crate::simd::Neon, C, 64>(x, c, s) },
            _ => |x, c, s| unsafe { crate::simd::dot_q::<crate::simd::Neon, C, 128>(x, c, s) },
        };
    }
    let _ = selected;
    match group {
        32 => dot_q_scalar::<C, 32>,
        64 => dot_q_scalar::<C, 64>,
        _ => dot_q_scalar::<C, 128>,
    }
}

/// The multi-row dot for code type `C`, `group` and the selected backend
/// (vector backends only; the scalar path keeps per-row dots).
fn select_dot_rows<C: crate::simd::QCode>(group: usize, simd: crate::kernels::Simd) -> Option<DotRowsQ<C>> {
    let selected = simd.resolved();
    #[cfg(target_arch = "x86_64")]
    if selected != crate::kernels::Simd::Scalar
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        return Some(match group {
            // SAFETY (all): AVX2/FMA detected above.
            32 => |x, st, n, c, s, o| unsafe { dot_rows_avx2::<C, 32>(x, st, n, c, s, o) },
            64 => |x, st, n, c, s, o| unsafe { dot_rows_avx2::<C, 64>(x, st, n, c, s, o) },
            _ => |x, st, n, c, s, o| unsafe { dot_rows_avx2::<C, 128>(x, st, n, c, s, o) },
        });
    }
    #[cfg(target_arch = "aarch64")]
    if selected == crate::kernels::Simd::Neon {
        return Some(match group {
            // SAFETY (all): NEON is baseline on aarch64.
            32 => |x, st, n, c, s, o| unsafe { dot_rows::<crate::simd::Neon, C, 32>(x, st, n, c, s, o) },
            64 => |x, st, n, c, s, o| unsafe { dot_rows::<crate::simd::Neon, C, 64>(x, st, n, c, s, o) },
            _ => |x, st, n, c, s, o| unsafe { dot_rows::<crate::simd::Neon, C, 128>(x, st, n, c, s, o) },
        });
    }
    let _ = (group, selected);
    None
}

/// Rows in groups of up to three (12 accumulators fit AVX2's registers).
#[inline(always)]
unsafe fn dot_rows<S: crate::simd::Simd, C: crate::simd::QCode, const G: usize>(
    x: &[f32],
    stride: usize,
    rows: usize,
    codes: &[C],
    scales: &[f32],
    out: &mut [f32; 8],
) {
    use crate::simd::dot_q_rows;
    debug_assert!(rows <= 8);
    let mut r = 0;
    unsafe {
        while r < rows {
            let x = &x[r * stride..];
            match rows - r {
                1 => {
                    out[r] = dot_q_rows::<S, C, G, 1>(x, stride, codes, scales)[0];
                    r += 1;
                }
                2 => {
                    out[r..r + 2].copy_from_slice(&dot_q_rows::<S, C, G, 2>(x, stride, codes, scales));
                    r += 2;
                }
                _ => {
                    out[r..r + 3].copy_from_slice(&dot_q_rows::<S, C, G, 3>(x, stride, codes, scales));
                    r += 3;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_rows_avx2<C: crate::simd::QCode, const G: usize>(
    x: &[f32],
    stride: usize,
    rows: usize,
    codes: &[C],
    scales: &[f32],
    out: &mut [f32; 8],
) {
    unsafe { dot_rows::<crate::simd::Avx2, C, G>(x, stride, rows, codes, scales, out) }
}

/// About two equal channel blocks per pool thread (multiples of 4 channels):
/// small decode matrices finish in one or two rounds without idle threads.
fn balanced_tasks(channels: usize) -> usize {
    let target = 2 * crate::team::threads();
    let per = channels.div_ceil(target).next_multiple_of(4).max(4);
    channels.div_ceil(per)
}

/// Output base pointer shared by tasks that write disjoint elements.
#[derive(Clone, Copy)]
struct OutputPtr(*mut f32);
// SAFETY: tasks write disjoint (row, channel) elements of one exclusive borrow.
unsafe impl Send for OutputPtr {}
unsafe impl Sync for OutputPtr {}
impl OutputPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Address of an exclusive output buffer for disjoint writes from tasks.
fn output_ptr(output: &OutputCell) -> *mut f32 {
    output.0
}
/// `Sync` wrapper so the write closure can be shared across tasks.
struct OutputCell(*mut f32);
// SAFETY: see OutputPtr.
unsafe impl Send for OutputCell {}
unsafe impl Sync for OutputCell {}

/// `kernels::dot_scalar` over `fl(code * scale)` weights, same iterator sum.
fn dot_q_scalar<C: crate::simd::QCode, const G: usize>(x: &[f32], codes: &[C], scales: &[f32]) -> f32 {
    x.iter()
        .zip(codes)
        .enumerate()
        .map(|(k, (value, &code))| value * (code.to_f32() * scales[k / G]))
        .sum()
}

/// `kernels::x86::dot_avx2` over `fl(code * scale)` weights: the generic
/// `simd::dot_q` (same phase accumulators, tree and tails) under AVX2/FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q_avx2<C: crate::simd::QCode, const G: usize>(
    x: &[f32],
    codes: &[C],
    scales: &[f32],
) -> f32 {
    unsafe { crate::simd::dot_q::<crate::simd::Avx2, C, G>(x, codes, scales) }
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
    fn dot_rows_are_bitwise_single_dots() {
        for (k, group) in [(768, 64), (1024, 64), (2304, 64), (65, 64), (31, 32), (100, 32), (768, 128)] {
            let x: Vec<f32> = (0..8 * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
            for bits in [8, 16] {
                let w: Vec<f32> = (0..3 * k).map(|i| ((i * 13 % 89) as f32 - 44.0) / 23.0).collect();
                let q = Q8Linear::quantize_bits(&w, 3, k, group, bits).unwrap();
                for simd in [crate::kernels::Simd::Auto, crate::kernels::Simd::Scalar] {
                    let (dot, dot_rows) = (q.dot_fn(simd), q.dot_rows_fn(simd));
                    for rows in 1..=8 {
                        for channel in 0..3 {
                            let mut values = [f32::NAN; 8];
                            if !q.rows_dot(dot_rows, &x[..rows * k], rows, channel, &mut values) {
                                continue;
                            }
                            for row in 0..rows {
                                let single = q.row_dot(dot, &x[row * k..(row + 1) * k], channel);
                                assert_eq!(values[row].to_bits(), single.to_bits(), "k {k} bits {bits} rows {rows}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_m_is_bitwise_fp32_on_dequantized_weights() {
        // The oracle for every W8 decode result: the FP32 GEMV on fl(code*scale).
        for (n, k, group) in [
            (37, 768, 64),
            (5, 1024, 64),
            (3, 2304, 64),
            (4, 65, 64),
            (2, 31, 64),
            (37, 768, 32),
            (3, 2304, 32),
            (4, 65, 32),
        ] {
            let w: Vec<_> = (0..n * k)
                .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 977.0)
                .collect();
            let q = Q8Linear::quantize(&w, n, k, group).unwrap();
            let mut dense = vec![0.0; n * k];
            for r in 0..n {
                q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
            }
            for backend in [Simd::Scalar, Simd::Auto, Simd::Avx2]
                .into_iter()
                .filter(|s| s.validate().is_ok())
            {
                for rows in 1..=8 {
                    let x: Vec<_> = (0..rows * k)
                        .map(|i| ((i * 104729 % 1009) as f32 - 504.0) / 311.0)
                        .collect();
                    let mut expected = vec![0.0; rows * n];
                    crate::kernels::linear_with_simd(
                        &x,
                        rows,
                        k,
                        &dense,
                        n,
                        &mut expected,
                        backend,
                    );
                    let mut out = vec![f32::NAN; rows * n];
                    q.linear(&x, rows, &mut out, &mut Scratch::default(), backend)
                        .unwrap();
                    for (a, b) in out.iter().zip(&expected) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{backend:?} n{n} k{k} g{group} rows{rows}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn w16_small_m_is_bitwise_fp32_on_dequantized_weights() {
        for (n, k) in [(37, 768), (5, 1024), (3, 2304), (4, 65)] {
            let w: Vec<_> = (0..n * k)
                .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 977.0)
                .collect();
            let q = Q8Linear::quantize_bits(&w, n, k, 64, 16).unwrap();
            assert_eq!(q.bits(), 16);
            assert_eq!(q.payload_bytes(), 2 * n * k + 4 * n * k.div_ceil(64));
            let mut dense = vec![0.0; n * k];
            for r in 0..n {
                q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
            }
            // 16-bit codes: every weight within half a step of 1/32767 of its group's absmax.
            for (a, b) in dense.iter().zip(&w) {
                assert!((a - b).abs() <= 1.0 / 32767.0 * 1.03 + 1e-7, "{a} vs {b}");
            }
            for backend in [Simd::Scalar, Simd::Auto, Simd::Avx2]
                .into_iter()
                .filter(|s| s.validate().is_ok())
            {
                for rows in 1..=8 {
                    let x: Vec<_> = (0..rows * k)
                        .map(|i| ((i * 104729 % 1009) as f32 - 504.0) / 311.0)
                        .collect();
                    let mut expected = vec![0.0; rows * n];
                    crate::kernels::linear_with_simd(&x, rows, k, &dense, n, &mut expected, backend);
                    let mut out = vec![f32::NAN; rows * n];
                    q.linear(&x, rows, &mut out, &mut Scratch::default(), backend)
                        .unwrap();
                    for (a, b) in out.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{backend:?} n{n} k{k} rows{rows}");
                    }
                }
            }
        }
    }
    #[test]
    fn fused_glu_is_bitwise_linear_then_gate() {
        for (ffn, k) in [(19, 768), (5, 65), (3, 2304)] {
            let n = 2 * ffn;
            let w: Vec<_> = (0..n * k)
                .map(|i| ((i * 6007 % 1999) as f32 - 999.0) / 613.0)
                .collect();
            let q = Q8Linear::quantize(&w, n, k, 64).unwrap();
            for backend in [Simd::Scalar, Simd::Auto] {
                for rows in 1..=8 {
                    let x: Vec<_> = (0..rows * k)
                        .map(|i| ((i * 7727 % 997) as f32 - 498.0) / 211.0)
                        .collect();
                    let mut packed = vec![0.0; rows * n];
                    q.linear(&x, rows, &mut packed, &mut Scratch::default(), backend)
                        .unwrap();
                    let mut expected = vec![0.0; rows * ffn];
                    crate::kernels::squared_relu_gate(&packed, &mut expected);
                    let mut fused = vec![f32::NAN; rows * ffn];
                    assert!(q.linear_glu(&x, rows, &mut fused, &mut Scratch::default(), backend).unwrap());
                    for (a, b) in fused.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{backend:?} ffn{ffn} rows{rows}");
                    }
                }
            }
        }
    }
    /// Decode-shaped GEMV throughput, cold in cache (working set > L3).
    #[test]
    #[ignore = "timing probe; run in release with --nocapture"]
    fn decode_gemv_throughput_probe() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        for (name, n, k) in [
            ("wo", 768, 1024),
            ("qkv", 2048, 768),
            ("w13", 4608, 768),
            ("w2", 768, 2304),
        ] {
            let copies = (96 << 20) / (n * k) + 1;
            let mats: Vec<Q8Linear> = (0..copies)
                .map(|c| {
                    let w: Vec<f32> = (0..n * k)
                        .map(|i| (((i + c) * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
                        .collect();
                    Q8Linear::quantize(&w, n, k, 64).unwrap()
                })
                .collect();
            let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 / 13.0).collect();
            let mut out = vec![0.0; n];
            pool.install(|| {
                for m in &mats {
                    m.linear(&x, 1, &mut out, &mut Scratch::default(), Simd::Auto)
                        .unwrap();
                }
                let rounds = 5;
                let t = std::time::Instant::now();
                for _ in 0..rounds {
                    for m in &mats {
                        m.linear(&x, 1, &mut out, &mut Scratch::default(), Simd::Auto)
                            .unwrap();
                    }
                }
                let calls = (rounds * mats.len()) as f64;
                let us = t.elapsed().as_secs_f64() * 1e6 / calls;
                let bytes = (n * k + n * k.div_ceil(64) * 4) as f64;
                println!("{name}: {us:.1} us/call, {:.1} GB/s", bytes / (us * 1e3));
            });
        }
    }
    #[test]
    fn artifact_validation() {
        assert!(Q8Linear::from_parts(1, 1, 64, vec![-128], vec![1.0]).is_err());
        assert!(Q8Linear::from_parts(1, 1, 64, vec![1], vec![0.0]).is_err());
        assert!(Q8Linear::from_parts(1, 1, 64, vec![0], vec![f32::NAN]).is_err());
    }
}
