//! Quantized linear layers (W8A32 / W16A32): signed row-major codes in
//! [-127,127] (8-bit) or [-32767,32767] (16-bit), an FP32 absmax/qmax scale
//! per K group, ties-to-even, FP32 dequant and ascending-K FMA. Neither
//! activations nor accumulators are integer/BF16; this is not W8A8. 16-bit
//! codes are effectively lossless against the FP32 weights
//! (activation-weighted output error about 2.5e-5 relative). A matrix may
//! also keep a few input columns unquantized in FP32
//! ([`QuantLinear::with_exceptions`]).

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

/// Exception columns: input columns whose weights are stored unquantized in
/// FP32 while the codes there are zero. A weight-only quantized product errs
/// by about `Σ_j Δw_j x_j`, so an input channel with a huge activation `x_c`
/// multiplies its column's rounding error by `|x_c|`; an unquantized column
/// has no rounding error. The values are the overlay's: the checkpoint's
/// weights after round to nearest, or after GPTQ (which quantizes the other
/// columns first and moves their error compensation onto these) weights that
/// differ from the checkpoint's. Every product uses them (see
/// [`QuantLinear::linear`]).
#[derive(Debug)]
struct Exceptions {
    /// Sorted, unique input columns, each below `in_dim` (I32, as stored).
    columns: crate::buf::Buf<i32>,
    /// FP32 weights, row-major `[out_dim][columns.len()]`.
    values: crate::buf::Buf<f32>,
    /// The CPU has the FMA instruction, detected when the columns were set
    /// so that [`Exceptions::accumulate`] does not check per call.
    #[cfg(target_arch = "x86_64")]
    fma: bool,
}

impl Exceptions {
    /// Columns and values as stored, with the instruction set detected once.
    fn new(columns: crate::buf::Buf<i32>, values: crate::buf::Buf<f32>) -> Self {
        Self {
            columns,
            values,
            #[cfg(target_arch = "x86_64")]
            fma: std::is_x86_feature_detected!("fma"),
        }
    }

    /// Output row `row`'s FP32 weights, in column order.
    fn row(&self, row: usize) -> &[f32] {
        let k = self.columns.len();
        &self.values[row * k..(row + 1) * k]
    }

    /// `acc` plus the products of `x` with output row `row`'s FP32 weights:
    /// one fused multiply-add per exception column, in ascending column
    /// order. A fused multiply-add rounds once, so the FMA instruction and
    /// libm's `fmaf` (used where the CPU has no FMA) give the same bits.
    #[inline(always)]
    fn accumulate(&self, x: &[f32], row: usize, acc: f32) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.fma {
            // SAFETY: FMA was detected when the columns were set.
            return unsafe { exception_fmas_fma(x, &self.columns, self.row(row), acc) };
        }
        exception_fmas(x, &self.columns, self.row(row), acc)
    }

    fn bytes(&self) -> usize {
        (self.columns.len() + self.values.len()) * 4
    }
}

/// `acc + Σ_i x[columns[i]] · values[i]`, one `mul_add` per term in order
/// (see [`Exceptions::accumulate`]).
#[inline(always)]
fn exception_fmas(x: &[f32], columns: &[i32], values: &[f32], mut acc: f32) -> f32 {
    for (&column, &value) in columns.iter().zip(values) {
        acc = x[column as usize].mul_add(value, acc);
    }
    acc
}

/// [`exception_fmas`] compiled with the FMA instruction (without it, x86-64
/// `mul_add` calls libm; same bits, slower).
///
/// # Safety
///
/// The CPU must support FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn exception_fmas_fma(x: &[f32], columns: &[i32], values: &[f32], acc: f32) -> f32 {
    exception_fmas(x, columns, values, acc)
}

/// A quantized linear layer with 8- or 16-bit codes and optional exception
/// columns.
#[derive(Debug)]
pub struct QuantLinear {
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    codes: Codes,
    scales: crate::buf::Buf<f32>,
    exceptions: Option<Exceptions>,
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
            exceptions: None,
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
    /// The exception columns kept unquantized in FP32 (empty without any).
    pub fn exception_columns(&self) -> &[i32] {
        self.exceptions.as_ref().map_or(&[], |e| &e.columns)
    }
    /// Bytes of the exception columns' indices and FP32 weights (zero
    /// without any); decode reads them on top of the codes.
    pub fn exception_bytes(&self) -> usize {
        self.exceptions.as_ref().map_or(0, Exceptions::bytes)
    }
    /// Tensor payload only (codes, scales and exception columns), excluding
    /// headers/alignment/allocator overhead.
    pub fn payload_bytes(&self) -> usize {
        let codes = match &self.codes {
            Codes::I8(codes) => codes.len(),
            Codes::I16(codes) => 2 * codes.len(),
        };
        codes + self.scales.len() * size_of::<f32>() + self.exception_bytes()
    }

    /// Row `row` of the reconstructed weights: `fl(code * scale)`, and the
    /// FP32 weight at every exception column.
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
        if let Some(exceptions) = &self.exceptions {
            for (&column, &value) in exceptions.columns.iter().zip(exceptions.row(row)) {
                output[column as usize] = value;
            }
        }
    }

    /// The reconstructed weight at element `index` (row-major), as
    /// [`QuantLinear::dequantize_row`] writes it.
    fn weight(&self, index: usize) -> f32 {
        let groups = self.in_dim.div_ceil(self.group_size);
        let (row, column) = (index / self.in_dim, index % self.in_dim);
        if let Some(exceptions) = &self.exceptions
            && let Some(i) = i32::try_from(column)
                .ok()
                .and_then(|c| exceptions.columns.binary_search(&c).ok())
        {
            return exceptions.row(row)[i];
        }
        let code = match &self.codes {
            Codes::I8(codes) => codes[index] as f32,
            Codes::I16(codes) => codes[index] as f32,
        };
        code * self.scales[row * groups + column / self.group_size]
    }

    /// Scalar reference: ascending-K FMA over the reconstructed weights
    /// ([`QuantLinear::dequantize_row`], so exception columns contribute
    /// their FP32 weights in place).
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
            exceptions: None,
        })
    }

    /// Keep `columns` of this matrix unquantized: `values` holds their FP32
    /// weights row-major (`[out_dim][columns.len()]`). The columns must be
    /// sorted, unique, below `in_dim` and at least one, and the codes there
    /// zero (the quantizer computed codes and scales with them zeroed).
    pub fn with_exceptions(mut self, columns: Vec<i32>, values: Vec<f32>) -> anyhow::Result<Self> {
        self.check_exception_columns(&columns)?;
        ensure!(
            self.out_dim.checked_mul(columns.len()) == Some(values.len()),
            "exception values shape"
        );
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite exception value");
        for &column in &columns {
            let column = column as usize;
            let zero = match &self.codes {
                Codes::I8(codes) => (0..self.out_dim).all(|r| codes[r * self.in_dim + column] == 0),
                Codes::I16(codes) => (0..self.out_dim).all(|r| codes[r * self.in_dim + column] == 0),
            };
            ensure!(zero, "nonzero code in exception column {column}");
        }
        self.exceptions = Some(Exceptions::new(
            crate::buf::Buf::Owned(columns),
            crate::buf::Buf::Owned(values),
        ));
        Ok(self)
    }

    /// [`QuantLinear::with_exceptions`] from `count` I32 columns and their
    /// FP32 values in byte ranges of a kernel-ready model file. The columns
    /// are checked and the values must be finite; the zero codes at the
    /// columns are trusted like the codes of [`QuantLinear::from_mapped`]
    /// (checking them would read every page of the codes; the file's tensor
    /// digest, checked on request, covers them).
    pub(crate) fn with_mapped_exceptions(
        mut self,
        map: &std::sync::Arc<memmap2::Mmap>,
        columns: std::ops::Range<usize>,
        values: std::ops::Range<usize>,
        count: usize,
    ) -> anyhow::Result<Self> {
        let columns = crate::buf::Buf::<i32>::mapped(map, columns, count)?;
        self.check_exception_columns(&columns)?;
        // `count <= in_dim` now, so the product fits like the codes' size.
        let values = crate::buf::Buf::<f32>::mapped(map, values, self.out_dim * count)?;
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite exception value");
        self.exceptions = Some(Exceptions::new(columns, values));
        Ok(self)
    }

    fn check_exception_columns(&self, columns: &[i32]) -> anyhow::Result<()> {
        ensure!(self.exceptions.is_none(), "exception columns already set");
        ensure!(!columns.is_empty(), "empty exception column set");
        ensure!(
            columns.windows(2).all(|pair| pair[0] < pair[1]),
            "exception columns must be sorted and unique"
        );
        ensure!(
            columns[0] >= 0 && (columns[columns.len() - 1] as usize) < self.in_dim,
            "exception column outside the {} inputs",
            self.in_dim
        );
        Ok(())
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
            exceptions: None,
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

    /// Raw exception column (I32) and value (FP32) bytes, for writing
    /// kernel-ready files; `None` without exception columns.
    pub(crate) fn raw_exceptions(&self) -> Option<(&[u8], &[u8])> {
        self.exceptions.as_ref().map(|e| (e.columns.bytes(), e.values.bytes()))
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
    ///
    /// Exception columns: 1..=8 rows add one fused multiply-add per exception
    /// column, in ascending column order, to the kernel's result over the codes
    /// (which are zero there), with the same scalar code for one row and for
    /// several, so each row stays bitwise its single-row product. That is
    /// bitwise `kernels::linear_with_simd` on the dequantized matrix with the
    /// exception columns zeroed, followed by those FMAs; large-M uses the exact
    /// weights in place ([`QuantLinear::dequantize_row`]), which differs from
    /// the decode order by rounding only.
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
    /// dot products (exception columns included) and gate expression as
    /// `linear` followed by `kernels::squared_relu_gate`, bit for bit.
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
    /// matrices use the BF16 panels where the CPU has AVX512-BF16, except a
    /// matrix with exception columns: BF16 panels hold only the codes, so it
    /// keeps the FP32 panels (its FP32 weights at its exception columns).
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
                && self.exceptions.is_none()
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
        if let Some(exceptions) = &self.exceptions {
            // The kernel is bitwise `row_dot`'s per row; so is this pass.
            for (row, value) in out[..rows].iter_mut().enumerate() {
                *value = exceptions.accumulate(&input[row * in_dim..(row + 1) * in_dim], channel, *value);
            }
        }
        true
    }

    /// `dot` of `x` with output row `channel`, then the exception columns.
    #[inline(always)]
    fn row_dot(&self, dot: Dot, x: &[f32], channel: usize) -> f32 {
        let (in_dim, groups) = (self.in_dim, self.in_dim.div_ceil(self.group_size));
        let scales = &self.scales[channel * groups..(channel + 1) * groups];
        let span = channel * in_dim..(channel + 1) * in_dim;
        let value = match (dot, &self.codes) {
            (Dot::I8(f), Codes::I8(codes)) => f(x, &codes[span], scales),
            (Dot::I16(f), Codes::I16(codes)) => f(x, &codes[span], scales),
            _ => unreachable!("dot kernel selected for another code type"),
        };
        match &self.exceptions {
            Some(exceptions) => exceptions.accumulate(x, channel, value),
            None => value,
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
