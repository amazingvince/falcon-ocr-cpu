//! Static-shift softmax for the FP32 prefill tiles (x86, when
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
//! * `p = exp(score)` is the operation sequence of `crate::simd::exp_poly`
//!   (the online kernels' fast exp) without its range checks, which never
//!   fire within the bound, so every lane is `exp_poly(score)` bit for bit;
//!   `ExpMode` does not apply;
//! * masked lanes are +0, and each denominator sums, in key order, the
//!   probabilities PV multiplies.
//!
//! The 8- and 16-lane forms therefore agree bit for bit. The exp argument
//! is the score itself, where the online kernel rounds `score - max` first.
//! The sink enters at the end through `logsumexp = ln(denominator)`, as in
//! the online kernels. Scores must stay below about 80 in magnitude (beyond
//! that a page-long row's denominator overflows, and beyond 87 the unchecked
//! exp leaves `exp_poly`), which the norms guarantee ten times over.
use super::{
    HEAD_DIM, KEY_TILE, QUERY_TILE, STAGE_CYCLES, Shape, cycles, mask_lanes, pv_lanes, qk_lanes, tile_span, wide,
};
use crate::kernels::CompactKv;
use crate::simd::{Avx2, EXP_LN2_HI, EXP_LN2_LO, EXP_LOG2E, EXP_P};
use std::arch::x86_64::*;

/// One (query tile, head) with the 8-lane products.
///
/// # Safety
/// AVX2 and FMA must be available; shapes as in the parent's
/// `compact_prefill`, with a pure prefix and every score within the bound.
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
    unsafe { tile_head::<false>(q, cache, shape, tile, queries, head, kv_head, sinks, output) }
}

/// One (query tile, head) with the 16-lane AVX-512F products, bitwise
/// [`tile_head_narrow`].
///
/// # Safety
/// As [`tile_head_narrow`], and AVX-512F must be available.
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
    unsafe { tile_head::<true>(q, cache, shape, tile, queries, head, kv_head, sinks, output) }
}

/// One (query tile, head) across all visible key tiles: QK, the static
/// exp with the denominators, PV, then normalization and the sink.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head<const WIDE: bool>(
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
    // `attention_with` sends only pure prefixes to the x86 tiles.
    assert_eq!(
        cache.prefix_len, cache.total_len,
        "the static-shift tiles take pure prefixes"
    );
    let first_query = tile * QUERY_TILE;
    let (first_absolute, visible_end, all_image) = tile_span(shape, first_query, queries);
    let mut qt = [0.0_f32; HEAD_DIM * QUERY_TILE];
    for r in 0..queries {
        let row = &q[(first_query + r) * shape.query_width + head * HEAD_DIM..][..HEAD_DIM];
        for (d, &value) in row.iter().enumerate() {
            qt[d * QUERY_TILE + r] = value;
        }
    }
    // SAFETY: row offsets stay inside `output` for this task's rows and head.
    let out = unsafe { output.add(first_query * shape.query_width + head * HEAD_DIM) };
    for r in 0..queries {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) }.fill(0.0);
    }
    let mut denominators = [0.0_f32; QUERY_TILE];
    let mut st = [0.0_f32; KEY_TILE * QUERY_TILE];
    for key_start in (0..visible_end).step_by(KEY_TILE) {
        let keys = (visible_end - key_start).min(KEY_TILE);
        let profile = shape.profile;
        let t0 = if profile { cycles() } else { 0 };
        // SAFETY: a pure prefix holds `visible_end` rows of every head's
        // keys and every KV head's values; `out` rows are this task's.
        unsafe {
            let k = cache
                .prefix_k
                .as_ptr()
                .add(key_start * shape.query_width + head * HEAD_DIM);
            if WIDE {
                wide::qk_lanes(&qt, queries, k, shape.query_width, keys, shape.scale, &mut st);
            } else {
                qk_lanes::<Avx2>(&qt, queries, k, shape.query_width, keys, shape.scale, &mut st);
            }
        }
        let t1 = if profile { cycles() } else { 0 };
        if !all_image {
            mask_lanes(shape, first_absolute, queries, key_start, keys, &mut st);
        }
        // SAFETY: as above.
        unsafe {
            match (WIDE, all_image) {
                (true, true) => exp_lanes16::<false>(queries, keys, &mut st, &mut denominators),
                (true, false) => exp_lanes16::<true>(queries, keys, &mut st, &mut denominators),
                (false, true) => exp_lanes8::<false>(queries, keys, &mut st, &mut denominators),
                (false, false) => exp_lanes8::<true>(queries, keys, &mut st, &mut denominators),
            }
        }
        let t2 = if profile { cycles() } else { 0 };
        // SAFETY: as above.
        unsafe {
            let v = cache.v.as_ptr().add(key_start * shape.kv_width + kv_head * HEAD_DIM);
            if WIDE {
                wide::pv_lanes(&st, queries, keys, v, shape.kv_width, out, shape.query_width);
            } else {
                pv_lanes::<Avx2>(&st, queries, keys, v, shape.kv_width, out, shape.query_width);
            }
        }
        if profile {
            use std::sync::atomic::Ordering::Relaxed;
            STAGE_CYCLES[0].fetch_add(t1 - t0, Relaxed);
            STAGE_CYCLES[1].fetch_add(t2 - t1, Relaxed);
            STAGE_CYCLES[2].fetch_add(cycles() - t2, Relaxed);
        }
    }
    for (r, &denominator) in denominators.iter().enumerate().take(queries) {
        // SAFETY: as above.
        let row = unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) };
        let logsumexp = denominator.ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in row {
            *value = (*value / denominator) * sink_scale;
        }
    }
}

/// `st[key][query] = exp(st[key][query])` over the tile's keys and queries,
/// eight lanes per vector, each probability added to its query's
/// denominator in key order. With `MASKED`, scores of -inf (`mask_lanes`)
/// give +0; a NaN score stays NaN (the unordered compare), as in the
/// all-image tiles, the online kernels and the scalar oracle.
#[inline(always)]
unsafe fn exp_lanes8<const MASKED: bool>(
    queries: usize,
    keys: usize,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
) {
    unsafe {
        let hidden = _mm256_set1_ps(f32::NEG_INFINITY);
        for lane0 in (0..queries).step_by(8) {
            let mut sum = _mm256_loadu_ps(denominators.as_ptr().add(lane0));
            for key in 0..keys {
                let at = st.as_mut_ptr().add(key * QUERY_TILE + lane0);
                let score = _mm256_loadu_ps(at);
                let mut p = exp8(score);
                if MASKED {
                    p = _mm256_and_ps(p, _mm256_cmp_ps::<_CMP_NEQ_UQ>(score, hidden));
                }
                sum = _mm256_add_ps(sum, p);
                _mm256_storeu_ps(at, p);
            }
            _mm256_storeu_ps(denominators.as_mut_ptr().add(lane0), sum);
        }
    }
}

/// [`exp_lanes8`] sixteen lanes per vector (AVX-512F).
#[inline(always)]
unsafe fn exp_lanes16<const MASKED: bool>(
    queries: usize,
    keys: usize,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
) {
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

/// `exp_poly` on eight lanes without its range checks: the same IEEE
/// operations, so the same bits wherever |x| <= 87. It is not
/// `Simd::exp_fast`, whose range checks (three compares and blends per
/// vector, on every tile) made the 8-lane tiles about 5% slower than this
/// exp with `exp_lanes8`'s compare and AND on masked tiles only.
#[inline(always)]
unsafe fn exp8(x: __m256) -> __m256 {
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
        _mm256_mul_ps(y, _mm256_castsi256_ps(bits))
    }
}

/// [`exp8`] on sixteen lanes, `2^n` by `vscalefps`: exact, as the bit
/// construction is, while the result is a normal number (|x| <= 87).
#[inline(always)]
unsafe fn exp16(x: __m512) -> __m512 {
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
