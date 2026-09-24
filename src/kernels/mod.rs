//! CPU operators for the FP32 Falcon-OCR execution graph.
//!
//! Shapes are checked at these safe entry points, before entering SIMD/GEMM code.
//! Every parallel operation uses the caller's current Rayon pool. A `Runner` can
//! therefore bound all of these operators with one `ThreadPool::install` call.
//! The local tests establish operator accuracy; they do not establish GPU parity.

use rayon::prelude::*;

mod attention;
// Bitwise equal to the platform expf on the softmax domain (exhaustive test).
pub(crate) mod exp;
mod panels;
#[cfg(test)]
mod tests;

pub(crate) use attention::{
    Bf16Kv, attention_prefill_bf16, attention_with, prefill_bf16_rows_available, store_prefill_bf16_row,
};
pub use attention::{CompactKv, Geometry, PrefillOptions, attention, attention_with_simd};
#[cfg(target_arch = "x86_64")]
pub(crate) use exp as vexp;
pub(crate) use panels::panel as panel_gemm;
#[cfg(target_arch = "x86_64")]
pub(crate) use panels::panel_bf16;

/// Prefill attention stage split (`Tuning::prefill_profile`); a no-op unless
/// `enabled`.
pub(crate) fn report_prefill_stage_cycles(enabled: bool) {
    attention::prefill64::report_stage_cycles(enabled);
}

/// Whether this CPU runs the BF16 prefill kernels (AVX2/FMA, AVX-512F and
/// AVX512-BF16).
pub(crate) fn bf16_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        avx2_available() && panel_bf16::available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the FP32 prefill attention runs the 16-lane AVX-512 tiles: the
/// `auto` backend on an AVX-512F CPU (bitwise identical to the 8-lane tiles;
/// an explicit `avx2` backend keeps 8 lanes everywhere).
pub(crate) fn wide_attention(simd: Simd) -> bool {
    cfg!(target_arch = "x86_64") && simd == Simd::Auto && avx512_available()
}

/// Prefill projection kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillProjection {
    /// FP32 GEMM (the `gemm` crate) over FP32 weights.
    GemmF32,
    /// Quantized codes dequantized into FP32 panels, AVX2/FMA tiles.
    PanelAvx2,
    /// Quantized codes into BF16 panels, AVX512-BF16 dot products.
    PanelBf16,
}

/// Prefill attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillAttention {
    Scalar,
    /// 8-lane FP32 tiles.
    Avx2,
    /// 16-lane FP32 tiles.
    Avx512Wide,
    /// BF16 keys and values, AVX512-BF16 dot products.
    Bf16,
    Neon,
}

/// The prefill kernels a body and CPU get.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefillPlan {
    pub projection: PrefillProjection,
    pub attention: PrefillAttention,
}

impl std::fmt::Display for PrefillProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::GemmF32 => "gemm-f32",
            Self::PanelAvx2 => "panel-avx2",
            Self::PanelBf16 => "panel-bf16",
        })
    }
}

impl std::fmt::Display for PrefillAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512Wide => "avx512-wide",
            Self::Bf16 => "bf16",
            Self::Neon => "neon",
        })
    }
}

/// The prefill kernels for a body of `body_bits` (`Model::body_bits`: `None`
/// for FP32) on this CPU with the `simd` backend: the single predicate set
/// behind `Model::forward_layers` and `auto::Resolved`. Only the `auto`
/// backend takes the AVX-512 paths (wide FP32 tiles, BF16 for 8-bit bodies);
/// `avx2` is 8-lane FP32 throughout. Traces, linear-input captures and
/// prefills of at most 8 rows take the FP32 path regardless.
pub fn prefill_plan(body_bits: Option<u32>, simd: Simd, prefill_bf16: crate::config::PrefillBf16) -> PrefillPlan {
    use crate::config::PrefillBf16;
    let panel = body_bits.is_some() && panel_gemm::available(simd);
    // 8-bit bodies (fast mode) also run prefill attention, and with
    // `prefill_bf16 = All` the projections, in BF16 where the CPU has
    // AVX512-BF16 (`attention_prefill_bf16`).
    let eight_bit = panel && body_bits == Some(8) && simd == Simd::Auto && bf16_available();
    let projection = if !panel {
        PrefillProjection::GemmF32
    } else if eight_bit && prefill_bf16 == PrefillBf16::All {
        PrefillProjection::PanelBf16
    } else {
        PrefillProjection::PanelAvx2
    };
    let attention = if eight_bit && prefill_bf16 != PrefillBf16::Off {
        PrefillAttention::Bf16
    } else if wide_attention(simd) {
        PrefillAttention::Avx512Wide
    } else {
        match simd.resolved() {
            Simd::Scalar => PrefillAttention::Scalar,
            Simd::Avx2 => PrefillAttention::Avx2,
            Simd::Neon => PrefillAttention::Neon,
            Simd::Auto => unreachable!("resolved"),
        }
    };
    PrefillPlan { projection, attention }
}

/// Vector implementation used for small-batch GEMV and attention. GEMM has its own dispatch.
///
/// Explicit unavailable variants return an error from [`Simd::validate`] and
/// panic if passed directly to a kernel. `Auto` resolves to AVX2/FMA on x86-64
/// (decode kernels are 8-lane; AVX-512F only widens the prefill attention
/// tiles, bitwise identically, and AVX512-BF16 serves the 8-bit prefill) and
/// to NEON on aarch64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Simd {
    Auto,
    Scalar,
    Avx2,
    /// aarch64 Advanced SIMD (always present on aarch64).
    Neon,
}

impl Simd {
    pub fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Auto | Self::Scalar => Ok(()),
            Self::Avx2 if avx2_available() => Ok(()),
            Self::Avx2 => Err("AVX2 and FMA are unavailable on this CPU"),
            Self::Neon if cfg!(target_arch = "aarch64") => Ok(()),
            Self::Neon => Err("NEON requires an aarch64 CPU"),
        }
    }

    /// Resolve automatic selection for reporting and one-time dispatch.
    pub fn resolved(self) -> Self {
        self.validate().expect("unsupported SIMD override");
        match self {
            Self::Auto if avx2_available() => Self::Avx2,
            Self::Auto if cfg!(target_arch = "aarch64") => Self::Neon,
            Self::Auto => Self::Scalar,
            explicit => explicit,
        }
    }
}

/// The instruction set the fixed-width decode kernels are compiled for:
/// AVX2/FMA on x86-64, NEON on aarch64 (never matched elsewhere).
fn native_vector() -> Simd {
    if cfg!(target_arch = "aarch64") {
        Simd::Neon
    } else if cfg!(target_arch = "x86_64") {
        Simd::Avx2
    } else {
        Simd::Auto
    }
}

fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn elements(rows: usize, cols: usize) -> usize {
    rows.checked_mul(cols).expect("tensor shape overflows usize")
}

/// Matrix multiplication with HF row-major `[out_dim, in_dim]` weights.
/// Input and output are row-major `[rows, in_dim]` and `[rows, out_dim]`.
pub fn linear(input: &[f32], rows: usize, in_dim: usize, weight: &[f32], out_dim: usize, out: &mut [f32]) {
    linear_with_simd(input, rows, in_dim, weight, out_dim, out, Simd::Auto);
}

/// [`linear`] with an explicit GEMV vector implementation.
///
/// `Scalar` also selects a scalar matrix multiplication for numerical debugging.
/// AVX2/AVX-512 selection controls batches of up to eight rows; larger calls
/// use `gemm`, which independently selects supported matrix instructions.
pub fn linear_with_simd(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    out: &mut [f32],
    simd: Simd,
) {
    assert_eq!(input.len(), elements(rows, in_dim), "linear input shape");
    assert_eq!(weight.len(), elements(out_dim, in_dim), "linear weight shape");
    assert_eq!(out.len(), elements(rows, out_dim), "linear output shape");
    let selected = simd.resolved();
    if out.is_empty() {
        return;
    }
    if in_dim == 0 {
        out.fill(0.0);
        return;
    }
    if rows <= 8 {
        let dot = dot_kernel(selected);
        // No packing/allocation on the decode and small-batch paths. This
        // avoids packing a full weight matrix to multiply only 1-8 vectors.
        // Tasks partition the flattened outputs, exposing channel parallelism
        // even when the number of request rows is smaller than the pool.
        let total = out.len();
        let block = crate::team::block_size(total, 8, 32);
        let shared = crate::team::SharedMut::new(out);
        crate::team::for_each(total.div_ceil(block), |b| {
            let start = b * block;
            let len = block.min(total - start);
            // SAFETY: blocks are disjoint ranges of `out`.
            let values = unsafe { shared.slice(start, len) };
            for (offset, value) in values.iter_mut().enumerate() {
                let index = start + offset;
                let row = index / out_dim;
                let channel = index % out_dim;
                *value = dot(
                    &input[row * in_dim..(row + 1) * in_dim],
                    &weight[channel * in_dim..(channel + 1) * in_dim],
                );
            }
        });
        return;
    }
    if selected == Simd::Scalar {
        out.par_chunks_mut(out_dim).enumerate().for_each(|(row, dst)| {
            let x = &input[row * in_dim..(row + 1) * in_dim];
            for (channel, y) in dst.iter_mut().enumerate() {
                *y = dot_scalar(x, &weight[channel * in_dim..(channel + 1) * in_dim]);
            }
        });
        return;
    }
    let in_stride = isize::try_from(in_dim).expect("linear input stride too large");
    let out_stride = isize::try_from(out_dim).expect("linear output stride too large");
    let threads = rayon::current_num_threads();
    let parallelism = if threads == 1 {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(threads)
    };
    // SAFETY: Exact slice lengths above bound every addressed element. Inputs
    // are immutable and Rust's exclusive output borrow disallows aliasing.
    // gemm's cs precedes rs, and it computes alpha*dst + beta*lhs*rhs.
    unsafe {
        gemm::gemm(
            rows,
            out_dim,
            in_dim,
            out.as_mut_ptr(),
            1,
            out_stride,
            false,
            input.as_ptr(),
            1,
            in_stride,
            weight.as_ptr(),
            in_stride,
            1,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            parallelism,
        );
    }
}

/// Normalize each contiguous `width`-element row in FP32.
/// Optional affine weights have one value per channel and no additive bias.
pub fn rms_norm(input: &[f32], out: &mut [f32], width: usize, eps: f32, weight: Option<&[f32]>) {
    assert!(width > 0, "RMSNorm width must be positive");
    assert_eq!(input.len() % width, 0, "RMSNorm input shape");
    assert_eq!(input.len(), out.len(), "RMSNorm output shape");
    assert!(
        eps.is_finite() && eps > 0.0,
        "RMSNorm epsilon must be finite and positive"
    );
    if let Some(w) = weight {
        assert_eq!(w.len(), width, "RMSNorm affine shape");
    }
    // Decode-sized inputs (one hidden row, or 16 query/key heads of up to eight
    // rows) cost less than waking the pool; each row's arithmetic is identical.
    if input.len() <= RMS_NORM_SERIAL_ELEMENTS {
        for (dst, src) in out.chunks_mut(width).zip(input.chunks(width)) {
            rms_norm_row(src, dst, width, eps, weight);
        }
        return;
    }
    out.par_chunks_mut(width)
        .zip(input.par_chunks(width))
        .for_each(|(dst, src)| rms_norm_row(src, dst, width, eps, weight));
}

const RMS_NORM_SERIAL_ELEMENTS: usize = 16_384;

#[inline]
pub(crate) fn rms_norm_row(src: &[f32], dst: &mut [f32], width: usize, eps: f32, weight: Option<&[f32]>) {
    // Pairwise FP32 reduction limits accumulated rounding error on
    // 768-channel rows without widening the reference's dtype or
    // fusing the square into an accumulation. CUDA reduction order
    // may still differ and is checked by independent operator traces.
    let scale = rms_scale(src, width, eps);
    match weight {
        Some(w) => {
            for ((y, x), affine) in dst.iter_mut().zip(src).zip(w) {
                *y = (*x * scale) * *affine;
            }
        }
        None => {
            for (y, x) in dst.iter_mut().zip(src) {
                *y = *x * scale;
            }
        }
    }
}

/// The factor `rms_norm_row` multiplies a row by.
#[inline]
pub(crate) fn rms_scale(src: &[f32], width: usize, eps: f32) -> f32 {
    (sum_squares_pairwise(src) / width as f32 + eps).sqrt().recip()
}

fn sum_squares_pairwise(input: &[f32]) -> f32 {
    const LANES: usize = 32;
    if input.len() > LANES {
        // Balance the tree while retaining full 32-value leaves except at the
        // tail. Recursion needs only logarithmic stack depth, no heap scratch.
        let middle = (input.len() / 2).next_multiple_of(LANES);
        return sum_squares_pairwise(&input[..middle]) + sum_squares_pairwise(&input[middle..]);
    }
    let mut partial = [0.0_f32; LANES];
    for (dst, value) in partial.iter_mut().zip(input) {
        *dst = *value * *value;
    }
    for width in [16, 8, 4, 2, 1] {
        for index in 0..width {
            partial[index] += partial[index + width];
        }
    }
    partial[0]
}

/// Interleaved `[gate_0, up_0, gate_1, up_1, ...]` squared-ReLU GLU.
pub fn squared_relu_gate(interleaved: &[f32], output: &mut [f32]) {
    assert_eq!(interleaved.len(), elements(output.len(), 2), "GLU shape");
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, y)| *y = squared_relu_glu(interleaved[i * 2], interleaved[i * 2 + 1]));
}

#[inline(always)]
pub(crate) fn squared_relu_glu(gate: f32, up: f32) -> f32 {
    // Match the pinned Triton kernel's tl.where(gate > 0, gate, 0),
    // including its treatment of NaN gates and signed zero.
    let relu = if gate > 0.0 { gate } else { 0.0 };
    (relu * relu) * up
}

/// `linear_with_simd` into interleaved `[gate, up]` channels followed by
/// `squared_relu_gate`, fused for 1..=8 rows. Each gated value uses the same
/// per-channel dot product and gate expression as the unfused pair, so the
/// result is bit-identical while one parallel region replaces two and the
/// `2 * ffn_dim` intermediate is never written. Returns `false` (and writes
/// nothing) for shapes the fused path does not cover.
pub fn linear_glu_with_simd(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    ffn_dim: usize,
    gated: &mut [f32],
    simd: Simd,
) -> bool {
    assert_eq!(input.len(), elements(rows, in_dim), "GLU input shape");
    assert_eq!(weight.len(), elements(elements(ffn_dim, 2), in_dim), "GLU weight shape");
    assert_eq!(gated.len(), elements(rows, ffn_dim), "GLU output shape");
    if rows == 0 || rows > 8 || in_dim == 0 {
        return false;
    }
    let dot = dot_kernel(simd.resolved());
    let total = gated.len();
    let block = crate::team::block_size(total, 4, 16);
    let shared = crate::team::SharedMut::new(gated);
    crate::team::for_each(total.div_ceil(block), |b| {
        let start = b * block;
        let len = block.min(total - start);
        // SAFETY: blocks are disjoint ranges of `gated`.
        let values = unsafe { shared.slice(start, len) };
        for (offset, value) in values.iter_mut().enumerate() {
            let index = start + offset;
            let row = index / ffn_dim;
            let channel = index % ffn_dim;
            let x = &input[row * in_dim..(row + 1) * in_dim];
            let gate = dot(x, &weight[2 * channel * in_dim..(2 * channel + 1) * in_dim]);
            let up = dot(x, &weight[(2 * channel + 1) * in_dim..(2 * channel + 2) * in_dim]);
            *value = squared_relu_glu(gate, up);
        }
    });
    true
}

pub(crate) type Dot = fn(&[f32], &[f32]) -> f32;
pub(crate) type Axpy = fn(f32, &[f32], &mut [f32]);

pub(crate) fn dot_kernel(simd: Simd) -> Dot {
    match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => |a, b| {
            // SAFETY: Entry point validated CPU features; shapes give equal
            // slice lengths. The implementation loads only full vector chunks.
            unsafe { x86::dot_avx2(a, b) }
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; equal slice lengths.
        Simd::Neon => |a, b| unsafe { crate::simd::dot::<crate::simd::Neon>(a, b) },
        Simd::Scalar => dot_scalar,
        _ => unreachable!("unresolved or unsupported vector implementation"),
    }
}

pub(crate) fn axpy_kernel(simd: Simd) -> Axpy {
    match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => |a, x, y| unsafe { x86::axpy_avx2(a, x, y) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; equal slice lengths.
        Simd::Neon => |a, x, y| unsafe { crate::simd::axpy::<crate::simd::Neon>(a, x, y) },
        Simd::Scalar => axpy_scalar,
        _ => unreachable!("unresolved or unsupported vector implementation"),
    }
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn axpy_scalar(a: f32, x: &[f32], y: &mut [f32]) {
    for (dst, src) in y.iter_mut().zip(x) {
        *dst += a * *src;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    /// `simd::dot::<Avx2>` behind the AVX2/FMA target feature: the
    /// function-pointer dot of the AVX2 backend.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe { crate::simd::dot::<crate::simd::Avx2>(a, b) }
    }

    /// `simd::axpy::<Avx2>` behind the AVX2/FMA target feature.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_avx2(a: f32, x: &[f32], y: &mut [f32]) {
        unsafe { crate::simd::axpy::<crate::simd::Avx2>(a, x, y) }
    }

    /// The original hand-written intrinsics kernels: the bitwise oracle of
    /// the generic ones (`attention::decode64::tests`).
    #[cfg(test)]
    pub(crate) mod reference {
        use std::arch::x86_64::*;

        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            // SAFETY: Caller checks features. Unaligned loads stay inside the two
            // equal-sized slices; the scalar tail covers fewer than eight elements.
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut i = 0;
                while i + 32 <= a.len() {
                    acc0 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.as_ptr().add(i)),
                        _mm256_loadu_ps(b.as_ptr().add(i)),
                        acc0,
                    );
                    acc1 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                        _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                        acc1,
                    );
                    acc2 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                        _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                        acc2,
                    );
                    acc3 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                        _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                        acc3,
                    );
                    i += 32;
                }
                let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                while i + 8 <= a.len() {
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.as_ptr().add(i)),
                        _mm256_loadu_ps(b.as_ptr().add(i)),
                        acc,
                    );
                    i += 8;
                }
                let halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
                let pairs = _mm_hadd_ps(halves, halves);
                let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
                while i < a.len() {
                    sum += a[i] * b[i];
                    i += 1;
                }
                sum
            }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn axpy_avx2(a: f32, x: &[f32], y: &mut [f32]) {
            debug_assert_eq!(x.len(), y.len());
            unsafe {
                let factor = _mm256_set1_ps(a);
                let mut i = 0;
                while i + 8 <= x.len() {
                    let value = _mm256_fmadd_ps(
                        factor,
                        _mm256_loadu_ps(x.as_ptr().add(i)),
                        _mm256_loadu_ps(y.as_ptr().add(i)),
                    );
                    _mm256_storeu_ps(y.as_mut_ptr().add(i), value);
                    i += 8;
                }
                while i < x.len() {
                    y[i] += a * x[i];
                    i += 1;
                }
            }
        }
    }
}
