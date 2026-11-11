//! Static-shift softmax for the FP32 prefill tiles (when
//! `PrefillOptions::fp32_softmax` is static: `Model::forward_layers` takes
//! it from `prefill_plan`, which plans it for 16-bit bodies unless
//! `Tuning::prefill_softmax_fp32` asks for the online softmax, for 8-bit
//! ones only when it asks for this one, and never for FP32 bodies, so exact
//! mode never runs it).
//!
//! As in `bf16::static_shift`, the per-head RMS norms bound every score
//! `q.k / 8` by 8 in magnitude, so `exp(score)` lies within about
//! [e^-8, e^8] and needs no shift: the running maximum, the block-max pass,
//! the rescales of the output rows and the denominators, and the separate
//! denominator pass of the online softmax disappear, and key tiles add
//! without correction. The rest is the parent's FP32 arithmetic:
//! * QK and PV are the parent's lane kernels (8 lanes, or 16 with
//!   AVX-512F), for every tile shape: no reference order is kept, so the
//!   short tiles skip gemm too;
//! * `p = exp(score)` follows `crate::simd::exp_poly` without its range
//!   checks. Eight-lane tiles use `Simd::exp_unchecked`; the AVX-512 tiles
//!   use `exp16`, scaling by `vscalefps` instead of constructing exponent
//!   bits. Both are bitwise `exp_poly(score)` within the bound; `ExpMode`
//!   does not apply;
//! * masked lanes are +0, and each denominator sums, in key order, the
//!   probabilities PV multiplies.
//!
//! The 8- and 16-lane forms therefore agree bit for bit. The exp argument
//! is the score itself, where the online kernel rounds `score - max` first.
//! The sink enters at the end through `logsumexp = ln(denominator)`, as in
//! the online kernels. Scores must stay below about 80 in magnitude (beyond
//! that a page-long row's denominator overflows, and beyond 87 the unchecked
//! exp leaves `exp_poly`), which the norms guarantee ten times over.
#[cfg(any(test, target_arch = "x86_64"))]
use super::{CompactKv, Shape};
use super::{Isa, KEY_TILE, QUERY_TILE};
#[cfg(target_arch = "x86_64")]
use crate::simd::{EXP_LN2_HI, EXP_LN2_LO, EXP_LOG2E, EXP_P};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Static-softmax instantiation of the parent's tile loop.
///
/// # Safety
/// As the parent's `compact_prefill`, with a pure prefix and bounded scores.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn tile_head_narrow(
    q: &[f32],
    cache: &CompactKv<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    // SAFETY: caller supplies the features, shapes and bounded scores.
    unsafe {
        super::tile_head::<crate::simd::Avx2, false, true>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}

/// Sixteen-lane instantiation, bitwise [`tile_head_narrow`].
///
/// # Safety
/// As [`tile_head_narrow`], with AVX-512F also available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn tile_head_wide(
    q: &[f32],
    cache: &CompactKv<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    // SAFETY: caller supplies the features, shapes and bounded scores.
    unsafe {
        super::tile_head::<crate::simd::Avx2, true, true>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}

/// Portable instantiation, the bitwise oracle for every native ISA.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_portable(
    q: &[f32],
    cache: &CompactKv<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    // SAFETY: test inputs supply the shapes and bounded scores.
    unsafe {
        super::tile_head::<crate::simd::Portable, false, true>(
            q, cache, shape, tile, queries, head, kv_head, sinks, output,
        )
    }
}

/// NEON instantiation; it awaits token qualification before plan selection.
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_neon(
    q: &[f32],
    cache: &CompactKv<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    // SAFETY: NEON is baseline, and tests supply shapes and bounded scores.
    unsafe {
        super::tile_head::<crate::simd::Neon, false, true>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}

/// Probabilities on eight lanes, summed in key order. Masked -inf lanes
/// become +0; NaNs propagate. Unmasked scores obey the per-head norm bound.
///
/// # Safety
/// The caller must enable `S`'s features; queries and keys must fit the tiles.
#[inline(always)]
pub(super) unsafe fn exp_lanes<S: Isa>(
    queries: usize,
    keys: usize,
    all_image: bool,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
) {
    // SAFETY: lanes remain inside the fixed-size tiles, and callers enable
    // S's target features. Masked lanes are discarded before use.
    unsafe {
        let hidden = S::splat(f32::NEG_INFINITY);
        for lane0 in (0..queries).step_by(8) {
            let mut sum = S::load(denominators.as_ptr().add(lane0));
            for key in 0..keys {
                let at = st.as_mut_ptr().add(key * QUERY_TILE + lane0);
                let score = S::load(at);
                let mut p = S::exp_unchecked(score);
                if !all_image {
                    p = S::select(S::eq(score, hidden), S::zero(), p);
                }
                sum = S::add(sum, p);
                S::store(at, p);
            }
            S::store(denominators.as_mut_ptr().add(lane0), sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
/// Static probabilities and denominators on sixteen lanes (AVX-512F),
/// with the same key order and bounded-score results as [`exp_lanes`].
///
/// # Safety
/// AVX-512F/FMA must be available; queries and keys must fit the tiles.
#[inline(always)]
pub(super) unsafe fn exp_lanes16<const MASKED: bool>(
    queries: usize,
    keys: usize,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
) {
    // SAFETY: callers enable AVX-512F/FMA and bound the tile dimensions.
    // Every sixteen-lane load/store fits the fixed-size tile arrays;
    // masked -inf lanes are discarded after exp16, and NaNs propagate.
    unsafe {
        let hidden = _mm512_set1_ps(f32::NEG_INFINITY);
        for lane0 in (0..queries).step_by(16) {
            let mut sum = _mm512_loadu_ps(denominators.as_ptr().add(lane0));
            for key in 0..keys {
                let at = st.as_mut_ptr().add(key * QUERY_TILE + lane0);
                let score = _mm512_loadu_ps(at);
                let mut p = exp16(score);
                if MASKED {
                    p = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(score, hidden), p);
                }
                sum = _mm512_add_ps(sum, p);
                _mm512_storeu_ps(at, p);
            }
            _mm512_storeu_ps(denominators.as_mut_ptr().add(lane0), sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
/// The polynomial core on sixteen lanes, using `vscalefps` for `2^n`.
/// For finite inputs in [-87, 87] the normal power-of-two scaling is exact,
/// so the results equal `Simd::exp_unchecked` bit for bit. NaNs propagate;
/// other inputs have unspecified values, discarded for masked -inf lanes.
///
/// # Safety
/// AVX-512F must be available.
#[inline(always)]
unsafe fn exp16(x: __m512) -> __m512 {
    // SAFETY: the caller enables AVX-512F, including these FMA operations.
    unsafe {
        let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm512_mul_ps(
            x,
            _mm512_set1_ps(EXP_LOG2E),
        ));
        let r = _mm512_fmadd_ps(n, _mm512_set1_ps(-EXP_LN2_HI), x);
        let r = _mm512_fmadd_ps(n, _mm512_set1_ps(-EXP_LN2_LO), r);
        let mut p = _mm512_set1_ps(EXP_P[0]);
        for c in &EXP_P[1..] {
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(*c));
        }
        let y = _mm512_add_ps(_mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r), _mm512_set1_ps(1.0));
        _mm512_scalef_ps(y, n)
    }
}

#[cfg(test)]
mod tests;
