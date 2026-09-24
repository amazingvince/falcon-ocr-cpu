//! Minimal portable SIMD layer: an 8-lane logical `f32` vector and the few
//! operations the decode kernels need, implemented once per instruction set.
//!
//! Kernels are written once as `#[inline(always)]` generic functions over
//! [`Simd`] and instantiated per ISA by small entry wrappers that carry the
//! matching `#[target_feature]` (x86) or nothing (NEON is baseline on
//! aarch64). Every implementation uses the same lane layout, fused
//! multiply-add, and the same fixed reduction tree, so an instantiation for
//! AVX2, AVX-512 or NEON is bitwise equal to the [`Portable`] one on the same
//! inputs; each machine can test its fast path against portable code.
//!
//! See `docs/CPU_PORTABILITY.md`.

/// 8-lane logical `f32` vector operations. All methods are `unsafe` because
/// pointer arguments are unchecked and x86 implementations require their
/// target features to be enabled in the calling (inlined-into) function.
pub(crate) trait Simd: Copy + Send + Sync + 'static {
    type V: Copy;
    unsafe fn zero() -> Self::V;
    unsafe fn splat(x: f32) -> Self::V;
    /// Eight consecutive `f32`, unaligned.
    unsafe fn load(p: *const f32) -> Self::V;
    unsafe fn store(p: *mut f32, v: Self::V);
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    /// `a * b + c` with a single rounding.
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// Eight consecutive `i8` codes converted exactly to `f32`.
    unsafe fn load_i8(p: *const i8) -> Self::V;
    /// Eight consecutive `i16` codes converted exactly to `f32`.
    unsafe fn load_i16(p: *const i16) -> Self::V;
    /// Eight consecutive BF16 bit patterns widened exactly to `f32`.
    unsafe fn load_bf16(p: *const u16) -> Self::V;
    /// Eight consecutive IEEE FP16 bit patterns widened exactly to `f32`
    /// (x86 callers must also enable `f16c`).
    unsafe fn load_f16(p: *const u16) -> Self::V;
    /// `((v0+v4) + (v1+v5)) + ((v2+v6) + (v3+v7))`: the x86 `dot_avx2` tree
    /// (128-bit halves added, then two horizontal pair additions).
    unsafe fn sum(v: Self::V) -> f32;
    /// Lane-wise `if a > b { a } else { b }`: x86 `maxps(a, b)` semantics
    /// (`b` when either is NaN or both are zeros), identical on every ISA.
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V;
    /// All-ones lanes where `a == b` (ordered), zero lanes elsewhere.
    unsafe fn eq(a: Self::V, b: Self::V) -> Self::V;
    /// Lanes of `a` where `mask` is all-ones, else lanes of `b`.
    unsafe fn select(mask: Self::V, a: Self::V, b: Self::V) -> Self::V;
    /// Bit `i` set when lane `i` of `mask` is all-ones.
    unsafe fn mask_bits(mask: Self::V) -> u32;
    /// [`exp_poly`] on every lane (same bits on every ISA).
    unsafe fn exp_fast(x: Self::V) -> Self::V;
    /// The preferred vector exp: [`Simd::exp_fast`], except on AVX2, where it
    /// is the platform-exact vector exp that keeps outputs bitwise equal to
    /// the scalar platform `expf` (and so to earlier results on this host).
    #[inline(always)]
    unsafe fn exp(x: Self::V) -> Self::V {
        unsafe { Self::exp_fast(x) }
    }
    /// [`Simd::exp`] plus a bit mask of lanes the caller must recompute with
    /// scalar `f32::exp` (only the platform-exact AVX2 exp reports lanes).
    #[inline(always)]
    unsafe fn exp_raw(x: Self::V) -> (Self::V, u32) {
        unsafe { (Self::exp(x), 0) }
    }
    /// `values[i] = exp(values[i] - shift)` with [`Simd::exp`] semantics
    /// (the tail uses the scalar form of the same function).
    #[inline(always)]
    unsafe fn exp_shifted(values: &mut [f32], shift: f32) {
        unsafe {
            let s = Self::splat(shift);
            let mut chunks = values.chunks_exact_mut(8);
            for chunk in &mut chunks {
                let x = Self::sub(Self::load(chunk.as_ptr()), s);
                Self::store(chunk.as_mut_ptr(), Self::exp(x));
            }
            for value in chunks.into_remainder() {
                *value = exp_poly(*value - shift);
            }
        }
    }
}

/// Portable single-precision exp (Cephes coefficients, about 1-2 ulp):
/// `exp(x) = 2^n * e^r`, `n = round(x * log2 e)`, `r = x - n ln 2` in two
/// FMA steps, `e^r = 1 + r + r^2 * P(r)`. Arguments below -87 return 0 (no
/// subnormal results, so no flush-to-zero dependence), above [`EXP_MAX`]
/// (88.376) return infinity, NaN returns NaN. The true exp stays finite up to
/// 88.72, but the `2^n` bit construction below overflows once `n` rounds to
/// 128, which happens from 88.3763 on; the ceiling stops short of that. Every
/// caller is a softmax with non-positive arguments. Every ISA's `exp_fast`
/// performs exactly these IEEE operations, so the results are bit-identical
/// across machines.
#[inline(always)]
pub(crate) fn exp_poly(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x < EXP_MIN {
        return 0.0;
    }
    if x > EXP_MAX {
        return f32::INFINITY;
    }
    let n = (x * EXP_LOG2E).round_ties_even();
    let r = n.mul_add(-EXP_LN2_HI, x);
    let r = n.mul_add(-EXP_LN2_LO, r);
    let mut p = EXP_P[0];
    for c in &EXP_P[1..] {
        p = p.mul_add(r, *c);
    }
    let y = p.mul_add(r * r, r) + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}
pub(crate) const EXP_MIN: f32 = -87.0;
pub(crate) const EXP_MAX: f32 = 88.376;
pub(crate) const EXP_LOG2E: f32 = std::f32::consts::LOG2_E;
pub(crate) const EXP_LN2_HI: f32 = 0.693_359_4;
pub(crate) const EXP_LN2_LO: f32 = -2.121_944_4e-4;
pub(crate) const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_65,
    0.5,
];

/// The reference implementation and the fallback for any CPU: plain arrays
/// with `f32::mul_add` (hardware FMA on aarch64 and x86 with FMA).
#[derive(Clone, Copy)]
#[cfg_attr(
    all(not(test), any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) struct Portable;

impl Simd for Portable {
    type V = [f32; 8];
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        [0.0; 8]
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        [x; 8]
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        unsafe { p.cast::<[f32; 8]>().read_unaligned() }
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        unsafe { p.cast::<[f32; 8]>().write_unaligned(v) }
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        std::array::from_fn(|i| a[i] + b[i])
    }
    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        std::array::from_fn(|i| a[i] - b[i])
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        std::array::from_fn(|i| a[i] * b[i])
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        std::array::from_fn(|i| a[i].mul_add(b[i], c[i]))
    }
    #[inline(always)]
    unsafe fn load_i8(p: *const i8) -> Self::V {
        let codes = unsafe { p.cast::<[i8; 8]>().read_unaligned() };
        codes.map(f32::from)
    }
    #[inline(always)]
    unsafe fn load_i16(p: *const i16) -> Self::V {
        let codes = unsafe { p.cast::<[i16; 8]>().read_unaligned() };
        codes.map(f32::from)
    }
    #[inline(always)]
    unsafe fn load_bf16(p: *const u16) -> Self::V {
        let bits = unsafe { p.cast::<[u16; 8]>().read_unaligned() };
        bits.map(|b| f32::from_bits(u32::from(b) << 16))
    }
    #[inline(always)]
    unsafe fn load_f16(p: *const u16) -> Self::V {
        let bits = unsafe { p.cast::<[u16; 8]>().read_unaligned() };
        bits.map(|b| half::f16::from_bits(b).to_f32())
    }
    #[inline(always)]
    unsafe fn sum(v: Self::V) -> f32 {
        ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]))
    }
    #[inline(always)]
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V {
        std::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }
    #[inline(always)]
    unsafe fn eq(a: Self::V, b: Self::V) -> Self::V {
        std::array::from_fn(|i| f32::from_bits(if a[i] == b[i] { u32::MAX } else { 0 }))
    }
    #[inline(always)]
    unsafe fn select(mask: Self::V, a: Self::V, b: Self::V) -> Self::V {
        std::array::from_fn(|i| if mask[i].to_bits() != 0 { a[i] } else { b[i] })
    }
    #[inline(always)]
    unsafe fn mask_bits(mask: Self::V) -> u32 {
        (0..8).fold(0, |bits, i| bits | ((mask[i].to_bits() >> 31) << i))
    }
    #[inline(always)]
    unsafe fn exp_fast(x: Self::V) -> Self::V {
        x.map(exp_poly)
    }
}

/// x86 AVX2 + FMA: one `__m256`. Also used for AVX-512 machines through an
/// entry wrapper that enables `avx512f,avx512vl` (32 vector registers).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx2 {
    type V = std::arch::x86_64::__m256;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        unsafe { std::arch::x86_64::_mm256_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_set1_ps(x) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        unsafe { std::arch::x86_64::_mm256_storeu_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_sub_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn load_i8(p: *const i8) -> Self::V {
        use std::arch::x86_64::*;
        unsafe { _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.cast()))) }
    }
    #[inline(always)]
    unsafe fn load_i16(p: *const i16) -> Self::V {
        use std::arch::x86_64::*;
        unsafe { _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(p.cast()))) }
    }
    #[inline(always)]
    unsafe fn load_bf16(p: *const u16) -> Self::V {
        use std::arch::x86_64::*;
        unsafe {
            let bits = _mm_loadu_si128(p.cast());
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(bits)))
        }
    }
    #[inline(always)]
    unsafe fn load_f16(p: *const u16) -> Self::V {
        use std::arch::x86_64::*;
        // F16C: exact widening; every AVX2 CPU has it (callers check).
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p.cast())) }
    }
    #[inline(always)]
    unsafe fn sum(v: Self::V) -> f32 {
        use std::arch::x86_64::*;
        unsafe {
            let halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let pairs = _mm_hadd_ps(halves, halves);
            _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
        }
    }
    #[inline(always)]
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_max_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn eq(a: Self::V, b: Self::V) -> Self::V {
        use std::arch::x86_64::*;
        unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b) }
    }
    #[inline(always)]
    unsafe fn select(mask: Self::V, a: Self::V, b: Self::V) -> Self::V {
        unsafe { std::arch::x86_64::_mm256_blendv_ps(b, a, mask) }
    }
    #[inline(always)]
    unsafe fn mask_bits(mask: Self::V) -> u32 {
        unsafe { std::arch::x86_64::_mm256_movemask_ps(mask) as u32 }
    }
    #[inline(always)]
    unsafe fn exp_fast(x: Self::V) -> Self::V {
        use std::arch::x86_64::*;
        unsafe {
            let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
                x,
                _mm256_set1_ps(EXP_LOG2E),
            ));
            let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-EXP_LN2_HI), x);
            let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-EXP_LN2_LO), r);
            let mut p = _mm256_set1_ps(EXP_P[0]);
            for c in &EXP_P[1..] {
                p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(*c));
            }
            let y = _mm256_add_ps(_mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r), _mm256_set1_ps(1.0));
            let bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)));
            let value = _mm256_mul_ps(y, _mm256_castsi256_ps(bits));
            let value = _mm256_blendv_ps(
                value,
                _mm256_setzero_ps(),
                _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(EXP_MIN)),
            );
            let value = _mm256_blendv_ps(
                value,
                _mm256_set1_ps(f32::INFINITY),
                _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(EXP_MAX)),
            );
            _mm256_blendv_ps(value, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
        }
    }
    #[inline(always)]
    unsafe fn exp(x: Self::V) -> Self::V {
        unsafe { crate::kernels::vexp::exp8(x) }
    }
    #[inline(always)]
    unsafe fn exp_raw(x: Self::V) -> (Self::V, u32) {
        unsafe {
            let (value, lanes) = crate::kernels::vexp::exp8_raw(x);
            (value, lanes as u32)
        }
    }
    #[inline(always)]
    unsafe fn exp_shifted(values: &mut [f32], shift: f32) {
        // Vector exp, bitwise equal to the platform expf (exhaustive test).
        unsafe { crate::kernels::vexp::exp_shifted_in_place(values, shift) }
    }
}

/// [`Avx2`] with the portable fast exp ([`exp_poly`]) in place of the
/// platform-exact one, so its kernels are bitwise equal to the NEON and
/// portable instantiations. Not bitwise equal to the scalar platform `expf`;
/// used where token agreement, not bit continuity, is the bar.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Avx2Fast;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx2Fast {
    type V = <Avx2 as Simd>::V;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        unsafe { Avx2::zero() }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        unsafe { Avx2::splat(x) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        unsafe { Avx2::load(p) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        unsafe { Avx2::store(p, v) }
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        unsafe { Avx2::add(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        unsafe { Avx2::sub(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        unsafe { Avx2::mul(a, b) }
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        unsafe { Avx2::fma(a, b, c) }
    }
    #[inline(always)]
    unsafe fn load_i8(p: *const i8) -> Self::V {
        unsafe { Avx2::load_i8(p) }
    }
    #[inline(always)]
    unsafe fn load_i16(p: *const i16) -> Self::V {
        unsafe { Avx2::load_i16(p) }
    }
    #[inline(always)]
    unsafe fn load_bf16(p: *const u16) -> Self::V {
        unsafe { Avx2::load_bf16(p) }
    }
    #[inline(always)]
    unsafe fn load_f16(p: *const u16) -> Self::V {
        unsafe { Avx2::load_f16(p) }
    }
    #[inline(always)]
    unsafe fn sum(v: Self::V) -> f32 {
        unsafe { Avx2::sum(v) }
    }
    #[inline(always)]
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V {
        unsafe { Avx2::max(a, b) }
    }
    #[inline(always)]
    unsafe fn eq(a: Self::V, b: Self::V) -> Self::V {
        unsafe { Avx2::eq(a, b) }
    }
    #[inline(always)]
    unsafe fn select(mask: Self::V, a: Self::V, b: Self::V) -> Self::V {
        unsafe { Avx2::select(mask, a, b) }
    }
    #[inline(always)]
    unsafe fn mask_bits(mask: Self::V) -> u32 {
        unsafe { Avx2::mask_bits(mask) }
    }
    #[inline(always)]
    unsafe fn exp_fast(x: Self::V) -> Self::V {
        unsafe { Avx2::exp_fast(x) }
    }
}

/// aarch64 NEON: two `float32x4_t` (lanes 0-3, 4-7). Baseline on aarch64.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) struct Neon;

#[cfg(target_arch = "aarch64")]
impl Simd for Neon {
    type V = (std::arch::aarch64::float32x4_t, std::arch::aarch64::float32x4_t);
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        use std::arch::aarch64::*;
        unsafe { (vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        use std::arch::aarch64::*;
        unsafe { (vdupq_n_f32(x), vdupq_n_f32(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        use std::arch::aarch64::*;
        unsafe { (vld1q_f32(p), vld1q_f32(p.add(4))) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        use std::arch::aarch64::*;
        unsafe {
            vst1q_f32(p, v.0);
            vst1q_f32(p.add(4), v.1);
        }
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        unsafe { (vaddq_f32(a.0, b.0), vaddq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        unsafe { (vsubq_f32(a.0, b.0), vsubq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        unsafe { (vmulq_f32(a.0, b.0), vmulq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        // vfmaq_f32(c, a, b) = c + a * b, fused.
        unsafe { (vfmaq_f32(c.0, a.0, b.0), vfmaq_f32(c.1, a.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn load_i16(p: *const i16) -> Self::V {
        use std::arch::aarch64::*;
        unsafe {
            let wide = vld1q_s16(p);
            (
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide))),
            )
        }
    }
    #[inline(always)]
    unsafe fn load_i8(p: *const i8) -> Self::V {
        use std::arch::aarch64::*;
        unsafe {
            let wide = vmovl_s8(vld1_s8(p));
            (
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide))),
            )
        }
    }
    #[inline(always)]
    unsafe fn load_bf16(p: *const u16) -> Self::V {
        use std::arch::aarch64::*;
        unsafe {
            let bits = vld1q_u16(p);
            (
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bits))),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bits))),
            )
        }
    }
    #[inline(always)]
    unsafe fn load_f16(p: *const u16) -> Self::V {
        // Exact but scalar; a native FCVTL path needs testing on aarch64.
        unsafe {
            let values = Portable::load_f16(p);
            (Self::load(values.as_ptr()).0, Self::load(values.as_ptr().add(4)).1)
        }
    }
    #[inline(always)]
    unsafe fn sum(v: Self::V) -> f32 {
        use std::arch::aarch64::*;
        unsafe {
            // [v0+v4, v1+v5, v2+v6, v3+v7] -> [h0+h1, h2+h3, ..] -> (h0+h1)+(h2+h3)
            let halves = vaddq_f32(v.0, v.1);
            let pairs = vpaddq_f32(halves, halves);
            vgetq_lane_f32::<0>(pairs) + vgetq_lane_f32::<1>(pairs)
        }
    }
    #[inline(always)]
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        // x86 maxps semantics: a where a > b (ordered), else b.
        unsafe {
            (
                vbslq_f32(vcgtq_f32(a.0, b.0), a.0, b.0),
                vbslq_f32(vcgtq_f32(a.1, b.1), a.1, b.1),
            )
        }
    }
    #[inline(always)]
    unsafe fn eq(a: Self::V, b: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        unsafe {
            (
                vreinterpretq_f32_u32(vceqq_f32(a.0, b.0)),
                vreinterpretq_f32_u32(vceqq_f32(a.1, b.1)),
            )
        }
    }
    #[inline(always)]
    unsafe fn select(mask: Self::V, a: Self::V, b: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        unsafe {
            (
                vbslq_f32(vreinterpretq_u32_f32(mask.0), a.0, b.0),
                vbslq_f32(vreinterpretq_u32_f32(mask.1), a.1, b.1),
            )
        }
    }
    #[inline(always)]
    unsafe fn mask_bits(mask: Self::V) -> u32 {
        use std::arch::aarch64::*;
        unsafe {
            let shifts = vld1q_s32([0, 1, 2, 3].as_ptr());
            let half = |m: float32x4_t| vaddvq_u32(vshlq_u32(vshrq_n_u32::<31>(vreinterpretq_u32_f32(m)), shifts));
            half(mask.0) | (half(mask.1) << 4)
        }
    }
    #[inline(always)]
    unsafe fn exp_fast(x: Self::V) -> Self::V {
        use std::arch::aarch64::*;
        unsafe {
            let lane = |x: float32x4_t| {
                let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(EXP_LOG2E)));
                // vfmaq_f32(c, a, b) = c + a * b.
                let r = vfmaq_f32(x, n, vdupq_n_f32(-EXP_LN2_HI));
                let r = vfmaq_f32(r, n, vdupq_n_f32(-EXP_LN2_LO));
                let mut p = vdupq_n_f32(EXP_P[0]);
                for c in &EXP_P[1..] {
                    p = vfmaq_f32(vdupq_n_f32(*c), p, r);
                }
                let y = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0));
                let bits = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
                let value = vmulq_f32(y, vreinterpretq_f32_s32(bits));
                let value = vbslq_f32(vcltq_f32(x, vdupq_n_f32(EXP_MIN)), vdupq_n_f32(0.0), value);
                let value = vbslq_f32(vcgtq_f32(x, vdupq_n_f32(EXP_MAX)), vdupq_n_f32(f32::INFINITY), value);
                // NaN lanes (x != x) return x.
                vbslq_f32(vceqq_f32(x, x), value, x)
            };
            (lane(x.0), lane(x.1))
        }
    }
}

/// FP32 dot product (`kernels::x86::dot_avx2` order).
#[inline(always)]
pub(crate) unsafe fn dot<S: Simd>(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = S::zero();
        let mut a1 = S::zero();
        let mut a2 = S::zero();
        let mut a3 = S::zero();
        let mut i = 0;
        while i + 32 <= a.len() {
            a0 = S::fma(S::load(ap.add(i)), S::load(bp.add(i)), a0);
            a1 = S::fma(S::load(ap.add(i + 8)), S::load(bp.add(i + 8)), a1);
            a2 = S::fma(S::load(ap.add(i + 16)), S::load(bp.add(i + 16)), a2);
            a3 = S::fma(S::load(ap.add(i + 24)), S::load(bp.add(i + 24)), a3);
            i += 32;
        }
        let mut acc = S::add(S::add(a0, a1), S::add(a2, a3));
        while i + 8 <= a.len() {
            acc = S::fma(S::load(ap.add(i)), S::load(bp.add(i)), acc);
            i += 8;
        }
        let mut total = S::sum(acc);
        while i < a.len() {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }
}

/// Integer weight codes that [`dot_q`] converts exactly to `f32`.
pub(crate) trait QCode: Copy + Send + Sync + 'static {
    /// Eight consecutive codes as `f32` lanes.
    unsafe fn load8<S: Simd>(p: *const Self) -> S::V;
    fn to_f32(self) -> f32;
}
impl QCode for i8 {
    #[inline(always)]
    unsafe fn load8<S: Simd>(p: *const Self) -> S::V {
        unsafe { S::load_i8(p) }
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}
impl QCode for i16 {
    #[inline(always)]
    unsafe fn load8<S: Simd>(p: *const Self) -> S::V {
        unsafe { S::load_i16(p) }
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}

/// Dot product with `fl(code * scale)` weights for any integer code type, one
/// scale per `G` inputs (`G` a multiple of 32, so each 32-element block uses
/// one scale, loaded and broadcast once). The same operation order as
/// [`dot`], so it equals [`dot`] on the dequantized row.
#[inline(always)]
pub(crate) unsafe fn dot_q<S: Simd, C: QCode, const G: usize>(x: &[f32], codes: &[C], scales: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), codes.len());
    debug_assert_eq!(G % 32, 0);
    unsafe {
        let (xp, cp) = (x.as_ptr(), codes.as_ptr());
        let w = |i: usize, s: S::V| S::mul(C::load8::<S>(cp.add(i)), s);
        let mut a0 = S::zero();
        let mut a1 = S::zero();
        let mut a2 = S::zero();
        let mut a3 = S::zero();
        let mut i = 0;
        while i + 32 <= x.len() {
            let s = S::splat(*scales.get_unchecked(i / G));
            a0 = S::fma(S::load(xp.add(i)), w(i, s), a0);
            a1 = S::fma(S::load(xp.add(i + 8)), w(i + 8, s), a1);
            a2 = S::fma(S::load(xp.add(i + 16)), w(i + 16, s), a2);
            a3 = S::fma(S::load(xp.add(i + 24)), w(i + 24, s), a3);
            i += 32;
        }
        let mut acc = S::add(S::add(a0, a1), S::add(a2, a3));
        while i + 8 <= x.len() {
            let s = S::splat(*scales.get_unchecked(i / G));
            acc = S::fma(S::load(xp.add(i)), w(i, s), acc);
            i += 8;
        }
        let mut total = S::sum(acc);
        while i < x.len() {
            total += x[i] * (codes[i].to_f32() * scales[i / G]);
            i += 1;
        }
        total
    }
}

/// [`dot_q`] of `R` input rows (`x[r * stride..][..codes.len()]`) with one
/// weight row: each 8-weight chunk is decoded once and feeds every row's
/// accumulators. Per row this is exactly `dot_q`'s operation sequence
/// (phase accumulators, tree, vector and scalar tails), so each result is
/// bitwise the single-row one.
#[inline(always)]
pub(crate) unsafe fn dot_q_rows<S: Simd, C: QCode, const G: usize, const R: usize>(
    x: &[f32],
    stride: usize,
    codes: &[C],
    scales: &[f32],
) -> [f32; R] {
    let len = codes.len();
    debug_assert!(R >= 1 && x.len() >= (R - 1) * stride + len);
    debug_assert_eq!(G % 32, 0);
    unsafe {
        let (xp, cp) = (x.as_ptr(), codes.as_ptr());
        let w = |i: usize, s: S::V| S::mul(C::load8::<S>(cp.add(i)), s);
        let mut a = [[S::zero(); 4]; R];
        let mut i = 0;
        while i + 32 <= len {
            let s = S::splat(*scales.get_unchecked(i / G));
            let weights = [w(i, s), w(i + 8, s), w(i + 16, s), w(i + 24, s)];
            for (r, a) in a.iter_mut().enumerate() {
                let xr = xp.add(r * stride + i);
                for (j, (a, weight)) in a.iter_mut().zip(weights).enumerate() {
                    *a = S::fma(S::load(xr.add(8 * j)), weight, *a);
                }
            }
            i += 32;
        }
        let mut acc = [S::zero(); R];
        for (acc, a) in acc.iter_mut().zip(&a) {
            *acc = S::add(S::add(a[0], a[1]), S::add(a[2], a[3]));
        }
        while i + 8 <= len {
            let s = S::splat(*scales.get_unchecked(i / G));
            let weight = w(i, s);
            for (r, acc) in acc.iter_mut().enumerate() {
                *acc = S::fma(S::load(xp.add(r * stride + i)), weight, *acc);
            }
            i += 8;
        }
        let mut total = [0.0_f32; R];
        for (total, acc) in total.iter_mut().zip(acc) {
            *total = S::sum(acc);
        }
        while i < len {
            let weight = codes[i].to_f32() * scales[i / G];
            for (r, total) in total.iter_mut().enumerate() {
                *total += x[r * stride + i] * weight;
            }
            i += 1;
        }
        total
    }
}

/// `y += a * x` with one FMA per element in eight-wide chunks and an unfused
/// scalar tail (`kernels::x86::axpy_avx2` order).
#[inline(always)]
pub(crate) unsafe fn axpy<S: Simd>(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    unsafe {
        let factor = S::splat(a);
        let (xp, yp) = (x.as_ptr(), y.as_mut_ptr());
        let mut i = 0;
        while i + 8 <= x.len() {
            S::store(yp.add(i), S::fma(factor, S::load(xp.add(i)), S::load(yp.add(i))));
            i += 8;
        }
        while i < x.len() {
            y[i] += a * x[i];
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = ((state >> 8) as f64 / 16_777_216.0 * 8.0 - 4.0) as f32;
                match i % 53 {
                    0 => -0.0,
                    1 => f32::from_bits(5),
                    _ => x,
                }
            })
            .collect()
    }

    #[test]
    fn fast_exp_is_identical_across_isas_and_accurate() {
        let mut inputs: Vec<f32> = (0..200_000).map(|i| -110.0 + 200.0 * (i as f32) / 200_000.0).collect();
        inputs.extend([
            0.0,
            -0.0,
            -87.0,
            -86.999,
            -87.001,
            88.375,
            88.377,
            88.72,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            -1e-30,
            1e-30,
        ]);
        let mut worst = 0.0_f64;
        for chunk in inputs.chunks(8) {
            let mut lanes = [0.0_f32; 8];
            lanes[..chunk.len()].copy_from_slice(chunk);
            let portable = unsafe { Portable::exp_fast(lanes) };
            for (x, y) in lanes.iter().zip(&portable) {
                let exact = f64::from(*x).exp();
                if x.is_finite() && *x >= EXP_MIN && *x <= EXP_MAX {
                    // Finite and close everywhere up to the ceiling itself.
                    assert!(y.is_finite(), "exp_poly({x}) = {y}");
                    let ulp = f64::from(f32::EPSILON) * exact.abs().max(f64::from(f32::MIN_POSITIVE));
                    worst = worst.max((f64::from(*y) - exact).abs() / ulp);
                } else if *x > EXP_MAX {
                    assert_eq!(*y, f32::INFINITY, "exp_poly({x})");
                }
            }
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn native(x: [f32; 8]) -> [f32; 8] {
                    let mut out = [0.0_f32; 8];
                    unsafe { Avx2::store(out.as_mut_ptr(), Avx2::exp_fast(Avx2::load(x.as_ptr()))) };
                    out
                }
                let avx2 = unsafe { native(lanes) };
                for (a, b) in avx2.iter().zip(&portable) {
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let mut out = [0.0_f32; 8];
                unsafe { Neon::store(out.as_mut_ptr(), Neon::exp_fast(Neon::load(lanes.as_ptr()))) };
                for (a, b) in out.iter().zip(&portable) {
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
                }
            }
        }
        assert!(worst < 3.0, "exp_poly error {worst} ulp");
        assert_eq!(exp_poly(f32::NEG_INFINITY), 0.0);
        assert!(exp_poly(f32::NAN).is_nan());
        // The ceiling sits below the first input whose 2^n construction
        // overflows (88.3763); the largest finite f32 exp argument is 88.72.
        assert!(exp_poly(EXP_MAX).is_finite());
        assert_eq!(exp_poly(EXP_MAX.next_up()), f32::INFINITY);
        assert!((f64::from(EXP_MAX) * std::f64::consts::LOG2_E).round() <= 127.0);
    }

    /// Every native implementation equals the portable one bit for bit.
    #[test]
    fn native_matches_portable_bitwise() {
        let codes: Vec<i8> = (0..2400).map(|i| (i * 37 % 255 - 127) as i8).collect();
        let scales = values(2400 / 64 + 1, 9)
            .iter()
            .map(|s| s.abs() * 0.01)
            .collect::<Vec<_>>();
        for len in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200, 768, 1024, 2304] {
            let a = values(len, 1 + len as u32);
            let b = values(len, 7 + len as u32);
            let portable = unsafe { dot::<Portable>(&a, &b) };
            let portable_q8 = unsafe { dot_q::<Portable, i8, 64>(&a, &codes[..len], &scales) };
            let mut portable_y = b.clone();
            unsafe { axpy::<Portable>(0.37, &a, &mut portable_y) };
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn native(a: &[f32], b: &[f32], c: &[i8], s: &[f32], y: &mut [f32]) -> (f32, f32) {
                    unsafe {
                        axpy::<Avx2>(0.37, a, y);
                        (dot::<Avx2>(a, b), dot_q::<Avx2, i8, 64>(a, c, s))
                    }
                }
                let mut y = b.clone();
                let (d, q) = unsafe { native(&a, &b, &codes[..len], &scales, &mut y) };
                assert_eq!(d.to_bits(), portable.to_bits(), "dot len {len}");
                assert_eq!(q.to_bits(), portable_q8.to_bits(), "dot_q8 len {len}");
                assert!(y.iter().zip(&portable_y).all(|(u, v)| u.to_bits() == v.to_bits()));
                // The generic AVX2 instantiation is the hand-written kernel.
                let hand = crate::kernels::dot_kernel(crate::kernels::Simd::Avx2)(&a, &b);
                assert_eq!(d.to_bits(), hand.to_bits(), "hand-written dot len {len}");
            }
            #[cfg(target_arch = "aarch64")]
            {
                let mut y = b.clone();
                let (d, q) = unsafe {
                    axpy::<Neon>(0.37, &a, &mut y);
                    (dot::<Neon>(&a, &b), dot_q::<Neon, i8, 64>(&a, &codes[..len], &scales))
                };
                assert_eq!(d.to_bits(), portable.to_bits(), "dot len {len}");
                assert_eq!(q.to_bits(), portable_q8.to_bits(), "dot_q8 len {len}");
                assert!(y.iter().zip(&portable_y).all(|(u, v)| u.to_bits() == v.to_bits()));
            }
        }
    }
}
