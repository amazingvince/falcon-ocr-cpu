//! INT16-activation prefill projections for 8-bit weights.
//!
//! Every row of `A` is quantized per 64-input group to INT16 codes with one
//! FP32 step, `absmax / 32767` (round to nearest, ties to even; the RMS-norm
//! row factor multiplies the step, not the values). The 8-bit weight codes
//! are sign-extended into INT16 panels, so a group's dot product is an exact
//! INT32 sum (`64 * 32767 * 127 < 2^31`). Per group and output that sum
//! converts to FP32 and adds `sum * (s_w * s_a)` with one FMA, in ascending
//! group order. The integer part is exact, so the AVX512-VNNI (`vpdpwssd`),
//! AVX2 (`vpmaddwd`) and scalar kernels agree bit for bit.
//!
//! Only activation rounding separates this from the FP32 panels. With a
//! normal FP32 step, the maximum error is about half a step, 1/65534 of the
//! group's absmax, against BF16's 2^-8 of every value. INT16 activations are
//! the default for 8-bit models (`Tuning::prefill_projections`).
//! FP32 exception columns are zeroed before activation quantization and add
//! one FMA per column, in ascending order, after the integer groups.
use rayon::prelude::*;

use super::{
    driver::{self, MR, PanelKernel},
    panel::Epilogue,
};

/// Output channels per panel in the product (two 16-lane vectors).
pub(crate) const NR: usize = 32;
/// Inputs per weight scale and per activation step.
const GROUP: usize = 64;
/// Input pairs per group (one `vpdpwssd` lane step each).
const PAIRS: usize = GROUP / 2;
/// Largest activation code.
const QMAX: f32 = 32767.0;
/// `1.5 * 2^23`: adding it to `|y| <= 2^22` rounds `y` to an integer (ties
/// to even) in the low mantissa bits.
const MAGIC: f32 = 12_582_912.0;

/// The integer dot-product kernel of a GEMM. All three give the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    /// `vpdpwssd` on 512-bit vectors (AVX-512F and AVX512-VNNI).
    Vnni,
    /// `vpmaddwd` and `vpaddd` on 256-bit vectors (AVX2 and FMA).
    Avx2,
    /// Portable scalar code: the bitwise oracle.
    Scalar,
}

impl Isa {
    /// Whether this CPU runs the kernel.
    pub(crate) fn available(self) -> bool {
        static FEATURES: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
        let (vnni, avx2) = *FEATURES.get_or_init(|| {
            (
                std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vnni"),
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            )
        });
        match self {
            Self::Vnni => vnni && avx2,
            Self::Avx2 => avx2,
            Self::Scalar => true,
        }
    }
}

/// Packed weights: codes as INT16 pairs `[panel][k / 2][nr][2]` and scales
/// `[panel][k / 64][nr]`, plus FP32 exception weights
/// `[panel][columns][nr]`.
#[derive(Default)]
pub(crate) struct Panels {
    nr: usize,
    codes: Vec<i16>,
    scales: Vec<f32>,
    columns: Vec<i32>,
    values: Vec<f32>,
}

/// Pack `n x k` 8-bit codes with one scale per 64 inputs (`scales` is
/// `[n][k / 64]`) into `NR`-column panels. `n` must be a multiple of `NR`
/// and `k` of 64.
pub(crate) fn pack<const NR: usize>(n: usize, k: usize, codes: &[i8], scales: &[f32], out: &mut Panels) {
    assert!(n.is_multiple_of(NR) && k.is_multiple_of(GROUP), "INT16 panel shape");
    assert_eq!(codes.len(), n * k);
    let groups = k / GROUP;
    assert_eq!(scales.len(), n * groups);
    out.nr = NR;
    out.columns.clear();
    out.values.clear();
    out.codes.resize(n * k, 0);
    out.scales.resize(n * groups, 0.0);
    out.codes
        .par_chunks_mut(NR * k)
        .zip(out.scales.par_chunks_mut(NR * groups))
        .enumerate()
        .for_each(|(panel, (dst, sdst))| {
            let base = &codes[panel * NR * k..(panel + 1) * NR * k];
            for (kp, pairs) in dst.chunks_exact_mut(2 * NR).enumerate() {
                for (col, pair) in pairs.chunks_exact_mut(2).enumerate() {
                    pair[0] = base[col * k + 2 * kp] as i16;
                    pair[1] = base[col * k + 2 * kp + 1] as i16;
                }
            }
            for col in 0..NR {
                for g in 0..groups {
                    sdst[g * NR + col] = scales[(panel * NR + col) * groups + g];
                }
            }
        });
}

/// Pack FP32 exception weights from `[n][columns]` into
/// `[panel][columns][NR]`, keeping their ascending input-column order.
pub(crate) fn pack_exceptions<const NR: usize>(n: usize, columns: &[i32], values: &[f32], out: &mut Panels) {
    assert!(n.is_multiple_of(NR), "INT16 exception panel shape");
    assert_eq!(values.len(), n * columns.len());
    out.columns.extend_from_slice(columns);
    out.values.resize(values.len(), 0.0);
    if columns.is_empty() {
        return;
    }
    for (panel, dst) in out.values.chunks_exact_mut(columns.len() * NR).enumerate() {
        for (column, weights) in dst.chunks_exact_mut(NR).enumerate() {
            for (row, weight) in weights.iter_mut().enumerate() {
                *weight = values[(panel * NR + row) * columns.len() + column];
            }
        }
    }
}

/// `C = A * W^T` for `a` (`m x k`) and 8-bit `panels` on kernel `isa`,
/// with row `r` of `A` scaled by `row_scale[r]` through its steps. Requires
/// `isa.available()`.
pub(crate) fn gemm<const NR: usize>(
    isa: Isa,
    a: &[f32],
    m: usize,
    k: usize,
    panels: &Panels,
    n: usize,
    row_scale: Option<&[f32]>,
    epilogue: Epilogue<'_>,
) {
    assert!(isa.available(), "INT16 panel kernel {isa:?} needs a CPU feature");
    assert!(n.is_multiple_of(NR) && k.is_multiple_of(GROUP), "INT16 panel shape");
    assert!(
        panels.nr == NR && panels.codes.len() == n * k,
        "INT16 panels packed for another shape"
    );
    driver::tiled_gemm(&I16Panels::<NR> { panels, isa }, a, m, k, n, row_scale, epilogue);
}

/// The packed panels and the kernel that multiplies them.
struct I16Panels<'a, const NR: usize> {
    panels: &'a Panels,
    isa: Isa,
}

impl<const NR: usize> PanelKernel<NR> for I16Panels<'_, NR> {
    type Packed = i32;
    /// Code pairs `[k / 2][MR]`, then the steps' bits `[k / 64][MR]`
    /// and the scaled FP32 exceptions' bits `[columns][MR]`.
    fn group_len(&self, k: usize) -> usize {
        MR * (k / 2 + k / GROUP + self.panels.columns.len())
    }
    fn pack_rows(&self, a: &[f32], rows: usize, padded: usize, k: usize, row_scale: Option<&[f32]>, out: &mut [i32]) {
        if self.isa != Isa::Scalar {
            // SAFETY: `gemm` checked that the kernel is available, and both
            // vector kernels need AVX2.
            unsafe { pack_rows_avx2(a, rows, padded, k, row_scale, &self.panels.columns, out) }
        } else {
            pack_rows(a, rows, padded, k, row_scale, &self.panels.columns, out)
        }
    }
    unsafe fn tile(&self, k: usize, a: &[i32], panel: usize, tile: &mut [[f32; NR]; MR]) {
        let codes = &self.panels.codes[panel * NR * k..(panel + 1) * NR * k];
        let groups = k / GROUP;
        let scales = &self.panels.scales[panel * NR * groups..(panel + 1) * NR * groups];
        debug_assert_eq!(a.len(), self.group_len(k));
        match self.isa {
            // SAFETY: the caller's contract (`gemm` checked the kernel); the
            // operands hold one panel and one packed row group.
            Isa::Vnni => unsafe { kernel_vnni(k, a.as_ptr(), codes.as_ptr(), scales.as_ptr(), tile) },
            // SAFETY: as above.
            Isa::Avx2 => unsafe { kernel_avx2(k, a.as_ptr(), codes.as_ptr(), scales.as_ptr(), tile) },
            Isa::Scalar => kernel_scalar(k, a, codes, scales, tile),
        }
        if !self.panels.columns.is_empty() {
            let count = self.panels.columns.len();
            let inputs = &a[MR * (k / 2 + groups)..];
            let values = &self.panels.values[panel * count * NR..(panel + 1) * count * NR];
            if self.isa == Isa::Scalar {
                exception_tile(inputs, values, tile);
            } else {
                // SAFETY: both vector ISAs require AVX2 and FMA, checked
                // by `gemm` before reaching this tile.
                unsafe { exception_tile_fma(inputs, values, tile) };
            }
        }
    }
}

/// Add ascending exception-column FMAs before the tile's epilogue.
#[inline(always)]
fn exception_tile<const NR: usize>(a: &[i32], values: &[f32], tile: &mut [[f32; NR]; MR]) {
    for (inputs, weights) in a.chunks_exact(MR).zip(values.chunks_exact(NR)) {
        for (r, out) in tile.iter_mut().enumerate() {
            let x = f32::from_bits(inputs[r] as u32);
            for (out, &weight) in out.iter_mut().zip(weights) {
                *out = x.mul_add(weight, *out);
            }
        }
    }
}

/// [`exception_tile`] compiled with native fused multiply-adds.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn exception_tile_fma<const NR: usize>(a: &[i32], values: &[f32], tile: &mut [[f32; NR]; MR]) {
    exception_tile(a, values, tile)
}

/// Quantize one 64-input group into `q` and return its step `absmax /
/// 32767`: 0 for an all-zero group, at least the smallest subnormal for a
/// nonzero group, NaN for a group with a non-finite input (the NaN reaches
/// every output of the row, as in the FP32 panels).
/// Tiny nonzero groups round the step up to avoid clipping their maximum.
/// This preserves the quantized group, not every resulting product: the
/// kernel multiplies the activation and weight scales in FP32 first, which
/// can lose subnormal precision or underflow before applying the integer sum.
#[inline(always)]
fn quantize_group(x: &[f32], q: &mut [i16; GROUP]) -> f32 {
    // Non-negative floats order like their bit patterns, and NaN and
    // infinity sort above every finite value.
    let top = x.iter().fold(0_u32, |m, v| m.max(v.to_bits() & 0x7fff_ffff));
    if top >= f32::INFINITY.to_bits() {
        *q = [0; GROUP];
        return f32::NAN;
    }
    let absmax = f32::from_bits(top);
    let inv = QMAX / absmax;
    if !inv.is_finite() {
        if absmax == 0.0 {
            *q = [0; GROUP];
            return 0.0;
        }
        // Keep the group representable when the FP32 reciprocal overflows.
        // Its later product with a weight scale can still underflow.
        let mut step = (absmax / QMAX).max(f32::from_bits(1));
        // Round a subnormal step up if its rounding would clip the max.
        if step as f64 * (QMAX as f64) < absmax as f64 {
            step = f32::from_bits(step.to_bits() + 1);
        }
        for (q, &v) in q.iter_mut().zip(x) {
            *q = (v as f64 / step as f64).round_ties_even().clamp(-32767.0, 32767.0) as i16;
        }
        return step;
    }
    for (q, &v) in q.iter_mut().zip(x) {
        // |v * inv| <= 32767 * (1 + 2^-23); the clamp only guards rounding.
        let y = (v * inv).clamp(-QMAX, QMAX);
        *q = ((y + MAGIC).to_bits() as i32 - MAGIC.to_bits() as i32) as i16;
    }
    absmax / QMAX
}

/// The activation layout of [`I16Panels`]: per group of `MR` rows, INT16
/// code pairs `[k / 2][MR]` (low half first) and then the steps (times the
/// row scale) as FP32 bits `[k / 64][MR]`, then scaled exception inputs
/// `[columns][MR]`; zero codes, steps and exceptions past `rows`. Every
/// operation is exact or one IEEE rounding, so any compiled form gives
/// the same bits.
#[inline(always)]
fn pack_rows(
    a: &[f32],
    rows: usize,
    padded: usize,
    k: usize,
    row_scale: Option<&[f32]>,
    columns: &[i32],
    out: &mut [i32],
) {
    let group_len = MR * (k / 2 + k / GROUP + columns.len());
    let pairs = k / 2;
    let mut q = [0_i16; GROUP];
    for (block, dst) in out[..padded / MR * group_len].chunks_exact_mut(group_len).enumerate() {
        let (codes, rest) = dst.split_at_mut(MR * pairs);
        let (steps, exceptions) = rest.split_at_mut(MR * k / GROUP);
        for r in 0..MR {
            let row = block * MR + r;
            if row >= rows {
                for kp in 0..pairs {
                    codes[kp * MR + r] = 0;
                }
                for g in 0..k / GROUP {
                    steps[g * MR + r] = 0;
                }
                for c in 0..columns.len() {
                    exceptions[c * MR + r] = 0;
                }
                continue;
            }
            let src = &a[row * k..(row + 1) * k];
            let scale = row_scale.map_or(1.0, |s| s[row]);
            let mut column = 0;
            for (g, x) in src.chunks_exact(GROUP).enumerate() {
                let step = if column < columns.len() && (columns[column] as usize) < (g + 1) * GROUP {
                    let mut zeroed = [0.0_f32; GROUP];
                    zeroed.copy_from_slice(x);
                    while column < columns.len() && (columns[column] as usize) < (g + 1) * GROUP {
                        zeroed[columns[column] as usize % GROUP] = 0.0;
                        column += 1;
                    }
                    quantize_group(&zeroed, &mut q)
                } else {
                    quantize_group(x, &mut q)
                };
                steps[g * MR + r] = (step * scale).to_bits() as i32;
                for (i, pair) in q.chunks_exact(2).enumerate() {
                    codes[(g * PAIRS + i) * MR + r] = (pair[0] as u16 as u32 | (pair[1] as u16 as u32) << 16) as i32;
                }
            }
            for (c, &column) in columns.iter().enumerate() {
                exceptions[c * MR + r] = (src[column as usize] * scale).to_bits() as i32;
            }
        }
    }
}

/// [`pack_rows`] compiled for AVX2 (same bits, vectorized).
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn pack_rows_avx2(
    a: &[f32],
    rows: usize,
    padded: usize,
    k: usize,
    row_scale: Option<&[f32]>,
    columns: &[i32],
    out: &mut [i32],
) {
    pack_rows(a, rows, padded, k, row_scale, columns, out)
}

/// The oracle tile: per output, ascending groups of an exact INT32 sum, then
/// `acc = fma(f32(sum), s_w * s_a, acc)` from zero.
fn kernel_scalar<const NR: usize>(k: usize, a: &[i32], b: &[i16], sw: &[f32], tile: &mut [[f32; NR]; MR]) {
    let pairs = k / 2;
    let sa = &a[MR * pairs..];
    for (r, out) in tile.iter_mut().enumerate() {
        for (j, out) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for g in 0..k / GROUP {
                let mut sum = 0_i32;
                for kp in g * PAIRS..(g + 1) * PAIRS {
                    let x = a[kp * MR + r];
                    let w = &b[(kp * NR + j) * 2..(kp * NR + j) * 2 + 2];
                    sum += (x as i16) as i32 * w[0] as i32 + (x >> 16) * w[1] as i32;
                }
                let s = sw[g * NR + j] * f32::from_bits(sa[g * MR + r] as u32);
                acc = (sum as f32).mul_add(s, acc);
            }
            *out = acc;
        }
    }
}

/// One `MR x NR` tile (`NR` 16 or 32) with `vpdpwssd`: per group, INT32
/// accumulators over the pairs, then the FP32 step per accumulator.
///
/// # Safety
/// The CPU must support AVX-512F and AVX512-VNNI; `a` holds one packed row
/// group, `b` and `sw` one panel's codes and scales.
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn kernel_vnni<const NR: usize>(
    k: usize,
    a: *const i32,
    b: *const i16,
    sw: *const f32,
    tile: &mut [[f32; NR]; MR],
) {
    use std::arch::x86_64::*;
    const { assert!(NR == 16 || NR == 32) };
    let wide = NR == 32;
    // SAFETY: the caller's contract bounds every offset below.
    unsafe {
        let sa = a.add(MR * k / 2).cast::<f32>();
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        for g in 0..k / GROUP {
            let mut part = [[_mm512_setzero_si512(); 2]; MR];
            for kp in g * PAIRS..(g + 1) * PAIRS {
                let w = b.add(kp * 2 * NR);
                let w0 = _mm512_loadu_si512(w.cast());
                let w1 = if wide { _mm512_loadu_si512(w.add(32).cast()) } else { w0 };
                let x = a.add(kp * MR);
                for (r, part) in part.iter_mut().enumerate() {
                    let xr = _mm512_set1_epi32(*x.add(r));
                    part[0] = _mm512_dpwssd_epi32(part[0], xr, w0);
                    if wide {
                        part[1] = _mm512_dpwssd_epi32(part[1], xr, w1);
                    }
                }
            }
            let s0 = _mm512_loadu_ps(sw.add(g * NR));
            let s1 = if wide { _mm512_loadu_ps(sw.add(g * NR + 16)) } else { s0 };
            for (r, (acc, part)) in acc.iter_mut().zip(&part).enumerate() {
                let s = _mm512_set1_ps(*sa.add(g * MR + r));
                acc[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(part[0]), _mm512_mul_ps(s0, s), acc[0]);
                if wide {
                    acc[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(part[1]), _mm512_mul_ps(s1, s), acc[1]);
                }
            }
        }
        for (row, acc) in tile.iter_mut().zip(&acc) {
            _mm512_storeu_ps(row.as_mut_ptr(), acc[0]);
            if wide {
                _mm512_storeu_ps(row.as_mut_ptr().add(16), acc[1]);
            }
        }
    }
}

/// One `MR x NR` tile with `vpmaddwd`: 16 columns at a time, the FP32
/// accumulators in `tile`.
///
/// # Safety
/// The CPU must support AVX2 and FMA; operands as in [`kernel_vnni`].
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2<const NR: usize>(
    k: usize,
    a: *const i32,
    b: *const i16,
    sw: *const f32,
    tile: &mut [[f32; NR]; MR],
) {
    use std::arch::x86_64::*;
    const { assert!(NR.is_multiple_of(16)) };
    for row in tile.iter_mut() {
        row.fill(0.0);
    }
    // SAFETY: the caller's contract bounds every offset below.
    unsafe {
        let sa = a.add(MR * k / 2).cast::<f32>();
        for g in 0..k / GROUP {
            for c in (0..NR).step_by(16) {
                let mut part = [[_mm256_setzero_si256(); 2]; MR];
                for kp in g * PAIRS..(g + 1) * PAIRS {
                    let w = b.add(kp * 2 * NR + 2 * c);
                    let w0 = _mm256_loadu_si256(w.cast());
                    let w1 = _mm256_loadu_si256(w.add(16).cast());
                    let x = a.add(kp * MR);
                    for (r, part) in part.iter_mut().enumerate() {
                        let xr = _mm256_set1_epi32(*x.add(r));
                        part[0] = _mm256_add_epi32(part[0], _mm256_madd_epi16(xr, w0));
                        part[1] = _mm256_add_epi32(part[1], _mm256_madd_epi16(xr, w1));
                    }
                }
                let s0 = _mm256_loadu_ps(sw.add(g * NR + c));
                let s1 = _mm256_loadu_ps(sw.add(g * NR + c + 8));
                for (r, part) in part.iter().enumerate() {
                    let s = _mm256_set1_ps(*sa.add(g * MR + r));
                    let dst = tile[r].as_mut_ptr().add(c);
                    let lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(part[0]), _mm256_mul_ps(s0, s), _mm256_loadu_ps(dst));
                    let hi = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(part[1]),
                        _mm256_mul_ps(s1, s),
                        _mm256_loadu_ps(dst.add(8)),
                    );
                    _mm256_storeu_ps(dst, lo);
                    _mm256_storeu_ps(dst.add(8), hi);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
