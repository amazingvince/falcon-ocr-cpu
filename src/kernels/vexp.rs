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
/// 1.12e9 inputs in [-104, 0], always by one ulp and always within 0.01 f32 ulp
/// (5.4e6 f64 ulps) of a midpoint. 2^23 f64 ulps (0.0156 f32 ulp) covers those
/// with margin; about 3% of lanes take the scalar path. Other C libraries must
/// pass `exhaustive_platform_agreement` before relying on bitwise agreement.
const UNCERTAIN: i64 = 1 << 23;

const LOG2E: f64 = std::f64::consts::LOG2_E;
const LN2_HI: f64 = 6.931_471_803_691_238_164_9e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;

/// `exp` of four f64 lanes to < 2^-50 relative error, plus the lanes'
/// distance (in f64 ulps) from an f32 rounding midpoint.
#[inline(always)]
unsafe fn exp4(x: __m256d) -> (__m128, __m256i) {
    unsafe {
        let k = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_pd(x, _mm256_set1_pd(LOG2E)),
        );
        let r = _mm256_fnmadd_pd(k, _mm256_set1_pd(LN2_HI), x);
        let r = _mm256_fnmadd_pd(k, _mm256_set1_pd(LN2_LO), r);
        // Horner, 1/12! down to 1/0!.
        let mut p = _mm256_set1_pd(1.0 / 479_001_600.0);
        for c in [
            1.0 / 39_916_800.0,
            1.0 / 3_628_800.0,
            1.0 / 362_880.0,
            1.0 / 40_320.0,
            1.0 / 5_040.0,
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
            1.0,
        ] {
            p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(c));
        }
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

/// `f32::exp` of eight lanes, bit-identical to the scalar platform function
/// on the verified domain.
///
/// # Safety
/// AVX2/FMA must be available.
#[inline(always)]
pub(super) unsafe fn exp8(x: __m256) -> __m256 {
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
