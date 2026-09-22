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
    #[allow(dead_code)] // used by the generic prefill kernels to come
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    /// `a * b + c` with a single rounding.
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// Eight consecutive `i8` codes converted exactly to `f32`.
    unsafe fn load_i8(p: *const i8) -> Self::V;
    /// Eight consecutive BF16 bit patterns widened exactly to `f32`.
    unsafe fn load_bf16(p: *const u16) -> Self::V;
    /// `((v0+v4) + (v1+v5)) + ((v2+v6) + (v3+v7))`: the x86 `dot_avx2` tree
    /// (128-bit halves added, then two horizontal pair additions).
    unsafe fn sum(v: Self::V) -> f32;
    /// `values[i] = (values[i] - shift).exp()`, bitwise the platform `expf`.
    #[inline(always)]
    unsafe fn exp_shifted(values: &mut [f32], shift: f32) {
        for value in values {
            *value = (*value - shift).exp();
        }
    }
}

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
    unsafe fn load_bf16(p: *const u16) -> Self::V {
        let bits = unsafe { p.cast::<[u16; 8]>().read_unaligned() };
        bits.map(|b| f32::from_bits(u32::from(b) << 16))
    }
    #[inline(always)]
    unsafe fn sum(v: Self::V) -> f32 {
        ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]))
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
    unsafe fn load_bf16(p: *const u16) -> Self::V {
        use std::arch::x86_64::*;
        unsafe {
            let bits = _mm_loadu_si128(p.cast());
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(bits)))
        }
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
    unsafe fn exp_shifted(values: &mut [f32], shift: f32) {
        // Vector exp, bitwise equal to the platform expf (exhaustive test).
        unsafe { crate::kernels::vexp::exp_shifted_in_place(values, shift) }
    }
}

/// aarch64 NEON: two `float32x4_t` (lanes 0-3, 4-7). Baseline on aarch64.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) struct Neon;

#[cfg(target_arch = "aarch64")]
impl Simd for Neon {
    type V = (
        std::arch::aarch64::float32x4_t,
        std::arch::aarch64::float32x4_t,
    );
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
    unsafe fn sum(v: Self::V) -> f32 {
        use std::arch::aarch64::*;
        unsafe {
            // [v0+v4, v1+v5, v2+v6, v3+v7] -> [h0+h1, h2+h3, ..] -> (h0+h1)+(h2+h3)
            let halves = vaddq_f32(v.0, v.1);
            let pairs = vpaddq_f32(halves, halves);
            vgetq_lane_f32::<0>(pairs) + vgetq_lane_f32::<1>(pairs)
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

/// Dot product with `fl(code * scale)` weights, one scale per `G` inputs
/// (`G` a multiple of 32, so each 32-element block uses one scale, loaded and
/// broadcast once). Same operation order as [`dot`].
#[inline(always)]
pub(crate) unsafe fn dot_q8<S: Simd, const G: usize>(
    x: &[f32],
    codes: &[i8],
    scales: &[f32],
) -> f32 {
    debug_assert_eq!(x.len(), codes.len());
    debug_assert_eq!(G % 32, 0);
    unsafe {
        let (xp, cp) = (x.as_ptr(), codes.as_ptr());
        let w = |i: usize, s: S::V| S::mul(S::load_i8(cp.add(i)), s);
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
            total += x[i] * (codes[i] as f32 * scales[i / G]);
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
            S::store(
                yp.add(i),
                S::fma(factor, S::load(xp.add(i)), S::load(yp.add(i))),
            );
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

    /// Every native implementation equals the portable one bit for bit.
    #[test]
    fn native_matches_portable_bitwise() {
        let codes: Vec<i8> = (0..2400)
            .map(|i| ((i * 37 % 255) as i32 - 127) as i8)
            .collect();
        let scales = values(2400 / 64 + 1, 9)
            .iter()
            .map(|s| s.abs() * 0.01)
            .collect::<Vec<_>>();
        for len in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200, 768, 1024, 2304] {
            let a = values(len, 1 + len as u32);
            let b = values(len, 7 + len as u32);
            let portable = unsafe { dot::<Portable>(&a, &b) };
            let portable_q8 = unsafe { dot_q8::<Portable, 64>(&a, &codes[..len], &scales) };
            let mut portable_y = b.clone();
            unsafe { axpy::<Portable>(0.37, &a, &mut portable_y) };
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn native(
                    a: &[f32],
                    b: &[f32],
                    c: &[i8],
                    s: &[f32],
                    y: &mut [f32],
                ) -> (f32, f32) {
                    unsafe {
                        axpy::<Avx2>(0.37, a, y);
                        (dot::<Avx2>(a, b), dot_q8::<Avx2, 64>(a, c, s))
                    }
                }
                let mut y = b.clone();
                let (d, q) = unsafe { native(&a, &b, &codes[..len], &scales, &mut y) };
                assert_eq!(d.to_bits(), portable.to_bits(), "dot len {len}");
                assert_eq!(q.to_bits(), portable_q8.to_bits(), "dot_q8 len {len}");
                assert!(
                    y.iter()
                        .zip(&portable_y)
                        .all(|(u, v)| u.to_bits() == v.to_bits())
                );
                // The generic AVX2 instantiation is the hand-written kernel.
                let hand = crate::kernels::dot_kernel(crate::kernels::Simd::Avx2)(&a, &b);
                assert_eq!(d.to_bits(), hand.to_bits(), "hand-written dot len {len}");
            }
            #[cfg(target_arch = "aarch64")]
            {
                let mut y = b.clone();
                let (d, q) = unsafe {
                    axpy::<Neon>(0.37, &a, &mut y);
                    (
                        dot::<Neon>(&a, &b),
                        dot_q8::<Neon, 64>(&a, &codes[..len], &scales),
                    )
                };
                assert_eq!(d.to_bits(), portable.to_bits(), "dot len {len}");
                assert_eq!(q.to_bits(), portable_q8.to_bits(), "dot_q8 len {len}");
                assert!(
                    y.iter()
                        .zip(&portable_y)
                        .all(|(u, v)| u.to_bits() == v.to_bits())
                );
            }
        }
    }
}
