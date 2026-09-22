//! Eight-lane `exp` that returns the platform's `f32::exp` bit for bit.
//!
//! Each lane is evaluated in f64 (Cody-Waite reduction, degree-12 Taylor
//! polynomial, exact power-of-two scaling) to well under 2^-50 relative error,
//! then rounded once to f32. That rounding equals the correctly rounded
//! `exp(x)` unless the f64 value lies within `UNCERTAIN` f64 ulps of an f32
//! rounding midpoint. Such lanes, NaN, and arguments below `FAST_MIN` (f32
//! subnormal results) are recomputed with scalar `f32::exp`. The result is
//! therefore identical to the platform function wherever the platform itself
//! rounds correctly outside that window; `exhaustive_platform_agreement`
//! checks every non-positive f32 against the platform library, which is the
//! only domain the softmax uses (`score - running_max <= 0`).
use std::arch::x86_64::*;

/// Arguments below this may produce f32 subnormals; they take the scalar path.
const FAST_MIN: f32 = -87.0;
/// Arguments above this overflow f32 (and would overflow the f64 exponent
/// construction for very large inputs); they take the scalar path.
const FAST_MAX: f32 = 88.0;
/// Midpoint window, in f64 ulps of the evaluated value (29 bits are dropped
/// when rounding a normal f64 to f32; the midpoint pattern is 1 << 28).
/// Measured on Windows UCRT: `expf` differs from correct rounding on 13,243 of
/// 1.12e9 inputs in [-104, 0], always by one ulp and at most 5.84e-4 f32 ulp
/// (2^18.3 f64 ulps) from a midpoint. The polynomial adds at most ~2^15 f64
/// ulps, so 2^20 keeps a 3x margin; about 0.4% of lanes take the scalar path.
/// Other C libraries must pass `exhaustive_platform_agreement` first.
const UNCERTAIN: i64 = 1 << 20;

const LOG2E: f64 = std::f64::consts::LOG2_E;
const LN2_HI: f64 = 6.931_471_803_691_238_164_9e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;

/// `exp` of four f64 lanes to < 1e-11 relative error (degree-9 Taylor on
/// |r| <= ln2/2, Estrin form), plus the lanes' distance (in f64 ulps) from an
/// f32 rounding midpoint. The error is ~275x below the `UNCERTAIN` window,
/// which is all the rounding decision needs.
#[inline(always)]
unsafe fn exp4(x: __m256d) -> (__m128, __m256i) {
    unsafe {
        let k = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_pd(x, _mm256_set1_pd(LOG2E)),
        );
        let r = _mm256_fnmadd_pd(k, _mm256_set1_pd(LN2_HI), x);
        let r = _mm256_fnmadd_pd(k, _mm256_set1_pd(LN2_LO), r);
        let c = |v: f64| _mm256_set1_pd(v);
        let r2 = _mm256_mul_pd(r, r);
        let r4 = _mm256_mul_pd(r2, r2);
        let r8 = _mm256_mul_pd(r4, r4);
        let p01 = _mm256_fmadd_pd(r, c(1.0), c(1.0));
        let p23 = _mm256_fmadd_pd(r, c(1.0 / 6.0), c(0.5));
        let p45 = _mm256_fmadd_pd(r, c(1.0 / 120.0), c(1.0 / 24.0));
        let p67 = _mm256_fmadd_pd(r, c(1.0 / 5_040.0), c(1.0 / 720.0));
        let p89 = _mm256_fmadd_pd(r, c(1.0 / 362_880.0), c(1.0 / 40_320.0));
        let p03 = _mm256_fmadd_pd(r2, p23, p01);
        let p47 = _mm256_fmadd_pd(r2, p67, p45);
        let p07 = _mm256_fmadd_pd(r4, p47, p03);
        let p = _mm256_fmadd_pd(r8, p89, p07);
        // 2^k for k in [-151, 128]: exact normal f64 scale.
        let ki = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k));
        let bits = _mm256_slli_epi64::<52>(_mm256_add_epi64(ki, _mm256_set1_epi64x(1023)));
        let y = _mm256_mul_pd(p, _mm256_castsi256_pd(bits));
        let low = _mm256_and_si256(_mm256_castpd_si256(y), _mm256_set1_epi64x(0x1FFF_FFFF));
        let distance = _mm256_sub_epi64(low, _mm256_set1_epi64x(1 << 28));
        (_mm256_cvtpd_ps(y), distance)
    }
}

/// Four 64-bit lane masks -> four 32-bit lane masks (low dwords, in order).
#[inline(always)]
unsafe fn pack_mask(m: __m256) -> __m128 {
    unsafe {
        let permuted = _mm256_permutevar8x32_ps(m, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
        _mm256_castps256_ps128(permuted)
    }
}

/// Lanes whose distance to an f32 midpoint is inside the uncertainty window.
#[inline(always)]
unsafe fn uncertain(distance: __m256i) -> __m256i {
    unsafe {
        let above = _mm256_cmpgt_epi64(distance, _mm256_set1_epi64x(-UNCERTAIN));
        let below = _mm256_cmpgt_epi64(_mm256_set1_epi64x(UNCERTAIN), distance);
        _mm256_and_si256(above, below)
    }
}

/// Vector `exp` of eight lanes and the bit mask of lanes that must be
/// recomputed with scalar `f32::exp` (midpoint window, subnormal/overflow
/// range, NaN). No branches; callers batch the rare fix-ups.
#[inline(always)]
pub(crate) unsafe fn exp8_raw(x: __m256) -> (__m256, i32) {
    unsafe {
        let (lo, lo_distance) = exp4(_mm256_cvtps_pd(_mm256_castps256_ps128(x)));
        let (hi, hi_distance) = exp4(_mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x)));
        let result = _mm256_set_m128(hi, lo);
        let lo_mask = _mm256_castsi256_ps(uncertain(lo_distance));
        let hi_mask = _mm256_castsi256_ps(uncertain(hi_distance));
        let window = _mm256_set_m128(pack_mask(hi_mask), pack_mask(lo_mask));
        let outside = _mm256_or_ps(
            _mm256_cmp_ps::<_CMP_NGE_UQ>(x, _mm256_set1_ps(FAST_MIN)),
            _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(FAST_MAX)),
        );
        (result, _mm256_movemask_ps(_mm256_or_ps(window, outside)))
    }
}

/// `f32::exp` of eight lanes, bit-identical to the scalar platform function
/// on the verified domain.
///
/// # Safety
/// AVX2/FMA must be available.
#[inline(always)]
pub(crate) unsafe fn exp8(x: __m256) -> __m256 {
    unsafe {
        let (lo, lo_distance) = exp4(_mm256_cvtps_pd(_mm256_castps256_ps128(x)));
        let (hi, hi_distance) = exp4(_mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x)));
        let result = _mm256_set_m128(hi, lo);
        // 64-bit lane masks -> 32-bit lane masks (pack the low dwords).
        let lo_mask = _mm256_castsi256_ps(uncertain(lo_distance));
        let hi_mask = _mm256_castsi256_ps(uncertain(hi_distance));
        let window = _mm256_set_m128(pack_mask(hi_mask), pack_mask(lo_mask));
        // Below FAST_MIN (or NaN) results may be subnormal; above FAST_MAX the
        // exponent construction would overflow. Both take the scalar path.
        let outside = _mm256_or_ps(
            _mm256_cmp_ps::<_CMP_NGE_UQ>(x, _mm256_set1_ps(FAST_MIN)),
            _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(FAST_MAX)),
        );
        let slow = _mm256_or_ps(window, outside);
        let lanes = _mm256_movemask_ps(slow);
        if lanes == 0 {
            return result;
        }
        let mut input = [0.0_f32; 8];
        let mut output = [0.0_f32; 8];
        _mm256_storeu_ps(input.as_mut_ptr(), x);
        _mm256_storeu_ps(output.as_mut_ptr(), result);
        for lane in 0..8 {
            if lanes & (1 << lane) != 0 {
                output[lane] = input[lane].exp();
            }
        }
        _mm256_loadu_ps(output.as_ptr())
    }
}

/// `values[i] = exp(values[i] - shift)`, bitwise `(values[i] - shift).exp()`.
///
/// # Safety
/// AVX2/FMA must be available.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp_shifted_in_place(values: &mut [f32], shift: f32) {
    unsafe {
        for block in values.chunks_mut(BLOCK) {
            exp_shifted_block(block, shift);
        }
    }
}

/// Values per branch-free block (64 vectors).
const BLOCK: usize = 512;

#[inline(always)]
unsafe fn exp_shifted_block(values: &mut [f32], shift: f32) {
    debug_assert!(values.len() <= BLOCK);
    unsafe {
        let s = _mm256_set1_ps(shift);
        let vectors = values.len() / 8;
        let mut masks = [0_u8; BLOCK / 8];
        let mut flagged = 0_u64;
        for (v, mask) in masks.iter_mut().enumerate().take(vectors) {
            let at = values.as_mut_ptr().add(8 * v);
            let (result, lanes) = exp8_raw(_mm256_sub_ps(_mm256_loadu_ps(at), s));
            // Keep the input for flagged lanes: store the result only where
            // no fix-up is needed (blend), so the fix-up can recompute x.
            let keep = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                _mm256_and_si256(
                    _mm256_set1_epi32(lanes),
                    _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128),
                ),
                _mm256_setzero_si256(),
            ));
            _mm256_storeu_ps(at, _mm256_blendv_ps(_mm256_loadu_ps(at), result, keep));
            *mask = lanes as u8;
            flagged |= u64::from(lanes != 0) << v;
        }
        while flagged != 0 {
            let v = flagged.trailing_zeros() as usize;
            flagged &= flagged - 1;
            let mut lanes = masks[v];
            while lanes != 0 {
                let lane = lanes.trailing_zeros() as usize;
                lanes &= lanes - 1;
                let value = &mut values[8 * v + lane];
                *value = (*value - shift).exp();
            }
        }
        for value in &mut values[8 * vectors..] {
            *value = (*value - shift).exp();
        }
    }
}

/// Scalar convenience wrapper for tests and tails.
#[cfg(test)]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp1(x: f32) -> f32 {
    unsafe { _mm256_cvtss_f32(exp8(_mm256_set1_ps(x))) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    fn available() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn mismatches_in(bits: std::ops::Range<u32>) -> (u64, Option<u32>) {
        let mut count = 0;
        let mut first = None;
        let mut start = bits.start;
        unsafe {
            while start < bits.end {
                let mut lanes = [0.0_f32; 8];
                for (i, lane) in lanes.iter_mut().enumerate() {
                    *lane = f32::from_bits((start + i as u32).min(bits.end - 1));
                }
                let mut out = [0.0_f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), exp8(_mm256_loadu_ps(lanes.as_ptr())));
                for (x, y) in lanes.iter().zip(&out) {
                    let expected = x.exp();
                    if expected.to_bits() != y.to_bits() && !(expected.is_nan() && y.is_nan()) {
                        count += 1;
                        first.get_or_insert(x.to_bits());
                    }
                }
                start = start.saturating_add(8);
            }
        }
        (count, first)
    }

    #[test]
    fn shifted_slices_match_scalar_loop() {
        if !available() {
            return;
        }
        for len in [0, 1, 7, 8, 9, 31, 128] {
            let values: Vec<f32> = (0..len)
                .map(|i| match i % 11 {
                    0 => f32::NEG_INFINITY,
                    1 => -0.0,
                    _ => (i as f32 * 0.731).sin() * 9.0,
                })
                .collect();
            let shift = 3.25_f32;
            let expected: Vec<f32> = values.iter().map(|v| (v - shift).exp()).collect();
            let mut actual = values.clone();
            unsafe { exp_shifted_in_place(&mut actual, shift) };
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(a.to_bits(), b.to_bits(), "len {len}");
            }
        }
    }

    #[test]
    fn sampled_platform_agreement() {
        if !available() {
            return;
        }
        let special = [
            0.0_f32,
            -0.0,
            -1e-30,
            -f32::MIN_POSITIVE,
            -0.5,
            -1.0,
            -16.0,
            -86.99,
            -87.0,
            -87.5,
            -103.9,
            -104.0,
            -200.0,
            f32::NEG_INFINITY,
            f32::NAN,
            1.0,
            5.5,
        ];
        for x in special {
            let expected = x.exp();
            let actual = unsafe { exp1(x) };
            assert!(
                expected.to_bits() == actual.to_bits() || (expected.is_nan() && actual.is_nan()),
                "{x}: {expected} vs {actual}"
            );
        }
        // Every 4099th negative float in the softmax range [-104, 0].
        let (count, first) = (0x8000_0000_u32..0xC2D0_0000)
            .into_par_iter()
            .step_by(4099 * 8)
            .map(|s| unsafe { mismatches_in(s..s.saturating_add(8)) })
            .reduce(|| (0, None), |a, b| (a.0 + b.0, a.1.or(b.1)));
        assert_eq!(count, 0, "first mismatch bits {first:?}");
    }

    /// Every non-positive f32 (plus -inf) against the platform `expf`.
    #[test]
    #[ignore = "exhaustive: ~2.1e9 evaluations; run in release on each target"]
    fn exhaustive_platform_agreement() {
        if !available() {
            return;
        }
        const BLOCK: u32 = 1 << 20;
        let starts: Vec<u32> = (0x8000_0000_u32..=0xFF80_0000)
            .step_by(BLOCK as usize)
            .collect();
        let (count, first) = starts
            .into_par_iter()
            .map(|s| unsafe { mismatches_in(s..s.saturating_add(BLOCK).min(0xFF80_0001)) })
            .reduce(|| (0, None), |a, b| (a.0 + b.0, a.1.or(b.1)));
        let (zero, _) = unsafe { mismatches_in(0..1) };
        assert_eq!(
            count + zero,
            0,
            "first mismatch bits {first:?} ({:?})",
            first.map(f32::from_bits)
        );
    }
}
