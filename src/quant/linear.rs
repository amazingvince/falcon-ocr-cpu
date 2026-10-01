//! Quantized linear layers (W8A32 / W16A32): signed row-major codes in
//! [-127,127] (8-bit) or [-32767,32767] (16-bit), an FP32 absmax/qmax scale
//! per K group, ties-to-even, FP32 dequant and ascending-K FMA. Neither
//! activations nor accumulators are integer/BF16; this is not W8A8. 16-bit
//! codes are effectively lossless against the FP32 weights
//! (activation-weighted output error about 2.5e-5 relative).

use anyhow::ensure;
use rayon::prelude::*;

mod dot;
#[cfg(test)]
mod integrated_tests;
#[cfg(test)]
mod tests;

use dot::{Dot, DotRows, OutputCell, OutputPtr, balanced_tasks, fill_row, output_ptr, select_dot, select_dot_rows};

/// Row-major weight codes of one matrix (owned, or mapped from a
/// kernel-ready model file).
#[derive(Debug)]
enum Codes {
    I8(crate::buf::Buf<i8>),
    I16(crate::buf::Buf<i16>),
}

/// A quantized linear layer with 8- or 16-bit codes.
#[derive(Debug)]
pub struct QuantLinear {
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    codes: Codes,
    scales: crate::buf::Buf<f32>,
}

impl QuantLinear {
    /// 8-bit codes; see [`QuantLinear::quantize_bits`].
    pub fn quantize(weights: &[f32], out_dim: usize, in_dim: usize, group_size: usize) -> Result<Self, &'static str> {
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
                        (value as f64 / scale as f64).round_ties_even().clamp(-qmax, qmax) as i32
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

    pub fn linear_f32(&self, input: &[f32], rows: usize, output: &mut [f32]) -> Result<(), &'static str> {
        if rows.checked_mul(self.in_dim) != Some(input.len()) || rows.checked_mul(self.out_dim) != Some(output.len()) {
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

/// Large-M expansion target; decode (1..=8 rows) writes outputs directly.
#[derive(Default)]
pub(crate) struct Scratch {
    pub dense: Vec<f32>,
    #[cfg(target_arch = "x86_64")]
    pub bf16: crate::kernels::panel_bf16::Panels,
}
impl QuantLinear {
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
        ensure!(out_dim.checked_mul(in_dim) == Some(codes.len()), "W8 codes shape");
        ensure!(
            out_dim.checked_mul(in_dim.div_ceil(group_size)) == Some(scales.len()),
            "W8 scales shape"
        );
        ensure!(codes.iter().all(|&v| v != -128), "W8 code -128 is outside [-127,127]");
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
                    ensure!(codes[a..b].iter().all(|&v| v == 0), "nonzero code in zero-scale group");
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
        ensure!(rows.checked_mul(self.in_dim) == Some(input.len()), "W8 input shape");
        ensure!(rows.checked_mul(self.out_dim) == Some(output.len()), "W8 output shape");
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
            crate::kernels::linear_with_simd(input, rows, self.in_dim, &scratch.dense, self.out_dim, output, simd);
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
        ensure!(rows.checked_mul(self.in_dim) == Some(input.len()), "W8 input shape");
        ensure!(self.out_dim.is_multiple_of(2), "GLU needs interleaved gate/up rows");
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
        let dot_rows = if rows > 1 {
            self.dot_rows_fn(simd)
        } else {
            DotRows::None
        };
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
    /// caller checks `panel_gemm` first. With `bf16_projections`, 8-bit
    /// matrices use the BF16 panels where the CPU has AVX512-BF16.
    pub(crate) fn prefill(
        &self,
        input: &[f32],
        rows: usize,
        row_scale: Option<&[f32]>,
        epilogue: crate::kernels::panel_gemm::Epilogue<'_>,
        scratch: &mut Scratch,
        bf16_projections: bool,
    ) {
        use crate::kernels::panel_gemm;
        assert_eq!(input.len(), rows * self.in_dim, "prefill input shape");
        #[cfg(target_arch = "x86_64")]
        {
            use crate::kernels::panel_bf16;
            if let Codes::I8(codes) = &self.codes
                && self.group_size == 64
                && self.out_dim.is_multiple_of(panel_bf16::NR)
                && bf16_projections
                && panel_bf16::available()
            {
                // 8-bit codes are exact in BF16: `vdpbf16ps` doubles FMA
                // throughput and only the activations round (see `panel_bf16`).
                panel_bf16::pack(self.out_dim, self.in_dim, codes, &self.scales, &mut scratch.bf16);
                panel_bf16::gemm(
                    input,
                    rows,
                    self.in_dim,
                    &scratch.bf16,
                    self.out_dim,
                    row_scale,
                    epilogue,
                );
                return;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = bf16_projections;
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
            row_scale,
            epilogue,
        );
    }

    /// Whether large-M products use `kernels::panel_gemm` here.
    pub(crate) fn panel_gemm(&self, simd: crate::kernels::Simd) -> bool {
        crate::kernels::panel_gemm::available(simd) && self.panel_shaped()
    }
    /// Whether the output dimension fills whole panels (`kernels::panel_gemm::NR`).
    pub(crate) fn panel_shaped(&self) -> bool {
        self.out_dim.is_multiple_of(crate::kernels::panel_gemm::NR)
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
        let dot_rows = if rows > 1 {
            self.dot_rows_fn(simd)
        } else {
            DotRows::None
        };
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
