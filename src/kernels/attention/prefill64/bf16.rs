//! BF16 prefill attention for AVX512-BF16 CPUs (fast mode only).
//!
//! The same tiling, mask and FP32 online softmax as the parent kernel, but
//! the QK and PV products use `vdpbf16ps` (32 multiply-adds per instruction
//! against 16 for an FP32 FMA). Q, K and V round to BF16 once per layer, and
//! the probabilities round to BF16 before PV. Products accumulate in FP32,
//! and scores, maxima, denominators and outputs stay FP32. This is not
//! bitwise equal to the FP32 kernel; fidelity is measured against the FP32
//! anchor like any lossy fast-mode change.
use super::{
    HEAD_DIM, KEY_TILE, OutputPtr, QUERY_TILE, STAGE_CYCLES, Shape, cycles, mask_lanes, pv_uses_main_path,
    qk_uses_main_path, reference_key_tile, softmax_lanes,
};
use crate::config::ExpMode;
use crate::kernels::{Bf16Kv, CompactKv, Geometry, PrefillOptions};
use rayon::prelude::*;
use std::arch::x86_64::*;

/// Time spent converting K/V to BF16 (probe; `Tuning::prefill_profile`).
pub(in crate::kernels) static CONVERT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pairs of BF16 values per 64-wide row.
const PAIRS: usize = HEAD_DIM / 2;

/// Writes prefill row `row` of `total` in the BF16 layouts above: its keys
/// (`k`, `[heads][64]`) into `keys` and its half of each value pair (`v`,
/// `[kv_heads][64]`) into `values`, the same conversion as the kernel's own.
/// Rows may be written concurrently (each writes only its own 16-bit
/// halves). With an odd `total`, the caller zeroes the last pair's odd half.
///
/// # Safety
/// AVX-512F, AVX512-BF16 and AVX512-BW must be available; `keys` holds
/// `heads * total * 32` and `values` `kv_heads * total.div_ceil(2) * 64`
/// elements.
#[target_feature(enable = "avx512f,avx512bf16,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn store_row(
    k: &[f32],
    v: &[f32],
    row: usize,
    total: usize,
    heads: usize,
    kv_heads: usize,
    keys: *mut u32,
    values: *mut u32,
) {
    debug_assert!(k.len() == heads * HEAD_DIM && v.len() == kv_heads * HEAD_DIM && row < total);
    unsafe {
        for h in 0..heads {
            to_bf16(k.as_ptr().add(h * HEAD_DIM), keys.add((h * total + row) * PAIRS).cast());
        }
        let pairs = total.div_ceil(2);
        let odd = row % 2 == 1;
        let mask: __mmask32 = if odd { 0xAAAA_AAAA } else { 0x5555_5555 };
        for g in 0..kv_heads {
            let dst = values.add((g * pairs + row / 2) * HEAD_DIM);
            for c in 0..HEAD_DIM / 16 {
                let lanes = bf16_lanes(_mm512_loadu_ps(v.as_ptr().add(g * HEAD_DIM + 16 * c)));
                let lanes = if odd { _mm512_slli_epi32::<16>(lanes) } else { lanes };
                _mm512_mask_storeu_epi16(dst.add(16 * c).cast(), mask, lanes);
            }
        }
    }
}

/// Pure prefill of `cache` (`head_dim == 64`, `prefix_len == total_len`).
/// `kv` supplies the BF16 key/value copies or the scratch to convert them
/// into; `options` select the softmax exp and the stage counters.
///
/// # Safety
/// AVX2, FMA, AVX-512F and AVX512-BF16 must be available; shapes as in the
/// parent's `compact_prefill`.
pub(in crate::kernels) unsafe fn compact_prefill(
    q: &[f32],
    cache: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    kv: Bf16Kv<'_>,
    options: PrefillOptions,
) {
    let (k, v, n_heads, n_kv_heads) = (cache.prefix_k, cache.v, cache.n_heads, cache.n_kv_heads);
    let PrefillOptions { exp, profile } = options;
    let shape = Shape {
        query_width: n_heads * HEAD_DIM,
        kv_width: n_kv_heads * HEAD_DIM,
        query_offset: geometry.query_offset,
        image_start: geometry.image_start,
        image_end: geometry.image_end,
        scale: (HEAD_DIM as f32).sqrt().recip(),
        profile,
    };
    let total = k.len() / shape.query_width;
    let pairs = total.div_ceil(2);
    let convert_start = std::time::Instant::now();
    let (keys_bf16, value_pairs): (&[u32], &[u32]) = match kv {
        Bf16Kv::Converted(keys, values) => {
            assert!(keys.len() == n_heads * total * PAIRS && values.len() == n_kv_heads * pairs * HEAD_DIM);
            (keys, values)
        }
        Bf16Kv::Convert(keys_bf16, value_pairs) => {
            keys_bf16.resize(n_heads * total * PAIRS, 0);
            value_pairs.resize(n_kv_heads * pairs * HEAD_DIM, 0);
            keys_bf16
                .par_chunks_mut(total * PAIRS)
                .enumerate()
                .for_each(|(head, dst)| {
                    for (t, row) in dst.chunks_exact_mut(PAIRS).enumerate() {
                        let src = &k[t * shape.query_width + head * HEAD_DIM..][..HEAD_DIM];
                        // SAFETY: AVX512-BF16 checked by the caller.
                        unsafe { to_bf16(src.as_ptr(), row.as_mut_ptr().cast()) };
                    }
                });
            value_pairs
                .par_chunks_mut(pairs * HEAD_DIM)
                .enumerate()
                .for_each(|(kv_head, dst)| {
                    for (p, row) in dst.chunks_exact_mut(HEAD_DIM).enumerate() {
                        let even = v[2 * p * shape.kv_width + kv_head * HEAD_DIM..].as_ptr();
                        let odd = (2 * p + 1 < total)
                            .then(|| v[(2 * p + 1) * shape.kv_width + kv_head * HEAD_DIM..].as_ptr());
                        // SAFETY: as above; both rows hold 64 values.
                        unsafe { pair_rows::<4>(even, odd, row.as_mut_ptr()) };
                    }
                });
            (&keys_bf16[..], &value_pairs[..])
        }
    };
    if profile {
        CONVERT_NS.fetch_add(
            convert_start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let fast = exp == ExpMode::Fast;
    let repeat = n_heads / n_kv_heads;
    let tiles = query_len.div_ceil(QUERY_TILE);
    let out = OutputPtr(output.as_mut_ptr());
    for kv_head in 0..n_kv_heads {
        (0..tiles * repeat).into_par_iter().for_each(|task| {
            let (tile, pair) = (task / repeat, task % repeat);
            let head = kv_head * repeat + pair;
            let queries = (query_len - tile * QUERY_TILE).min(QUERY_TILE);
            let operands = Operands {
                q,
                k,
                v,
                keys: &keys_bf16[head * total * PAIRS..(head + 1) * total * PAIRS],
                values: &value_pairs[kv_head * pairs * HEAD_DIM..(kv_head + 1) * pairs * HEAD_DIM],
            };
            // SAFETY: features and shapes checked by the caller; each task
            // owns rows [tile * 32, tile * 32 + queries) of `head`.
            unsafe {
                if fast {
                    tile_head_fast(&operands, &shape, tile, queries, head, kv_head, sinks, out.get());
                } else {
                    tile_head_native(&operands, &shape, tile, queries, head, kv_head, sinks, out.get());
                }
            }
        });
    }
}

struct Operands<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    /// This head's keys, `[total][32]` BF16 pairs.
    keys: &'a [u32],
    /// This KV head's value pairs, `[total / 2][64]`.
    values: &'a [u32],
}

/// 64 FP32 values to BF16 (round to nearest even).
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn to_bf16(src: *const f32, dst: *mut u16) {
    unsafe {
        for c in (0..HEAD_DIM).step_by(32) {
            let lo = _mm512_loadu_ps(src.add(c));
            let hi = _mm512_loadu_ps(src.add(c + 16));
            let packed: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(hi, lo));
            _mm512_storeu_si512(dst.add(c).cast(), packed);
        }
    }
}

/// `dst[i] = bf16(even[i]) | bf16(odd[i]) << 16` for `16 * V` values; a
/// missing odd row is zero.
#[inline(always)]
unsafe fn pair_rows<const V: usize>(even: *const f32, odd: Option<*const f32>, dst: *mut u32) {
    unsafe {
        for c in 0..V {
            let lo = bf16_lanes(_mm512_loadu_ps(even.add(16 * c)));
            let hi = match odd {
                Some(odd) => _mm512_slli_epi32::<16>(bf16_lanes(_mm512_loadu_ps(odd.add(16 * c)))),
                None => _mm512_setzero_si512(),
            };
            _mm512_storeu_si512(dst.add(16 * c).cast(), _mm512_or_si512(lo, hi));
        }
    }
}

/// 16 FP32 lanes to BF16 bits, zero-extended to 32-bit lanes.
#[inline(always)]
unsafe fn bf16_lanes(x: __m512) -> __m512i {
    unsafe { _mm512_cvtepu16_epi32(std::mem::transmute::<__m256bh, __m256i>(_mm512_cvtneps_pbh(x))) }
}

#[inline(always)]
fn bh(x: __m512i) -> __m512bh {
    // SAFETY: both are 512-bit plain vectors.
    unsafe { std::mem::transmute(x) }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bf16")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_native(
    operands: &Operands<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    unsafe { tile_head::<crate::simd::Avx2, false>(operands, shape, tile, queries, head, kv_head, sinks, output) }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bf16")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_fast(
    operands: &Operands<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    // The fused softmax (accepted A/B); the separate-pass form stays as the
    // oracle of `fused_softmax_matches_separate_passes`.
    unsafe { tile_head::<crate::simd::Avx2Fast, true>(operands, shape, tile, queries, head, kv_head, sinks, output) }
}

/// One (query tile, head) across all visible key tiles: the parent's
/// `tile_head` with BF16 products on main-path tiles.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head<S: crate::simd::Simd, const FUSED: bool>(
    operands: &Operands<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    let first_query = tile * QUERY_TILE;
    let first_absolute = shape.query_offset + first_query;
    let last_absolute = first_absolute + queries - 1;
    let intersects_image = first_absolute < shape.image_end && last_absolute >= shape.image_start;
    let visible_end = if intersects_image {
        (last_absolute + 1).max(shape.image_end)
    } else {
        last_absolute + 1
    };
    let all_image = first_absolute >= shape.image_start && last_absolute < shape.image_end;
    // Query pairs, `[d / 2][32 queries]`, zero past `queries`.
    let mut query_pairs = [0_u32; PAIRS * QUERY_TILE];
    for r in 0..queries {
        let row = &operands.q[(first_query + r) * shape.query_width + head * HEAD_DIM..][..HEAD_DIM];
        for (p, pair) in row.chunks_exact(2).enumerate() {
            let even = half::bf16::from_f32(pair[0]).to_bits() as u32;
            let odd = half::bf16::from_f32(pair[1]).to_bits() as u32;
            query_pairs[p * QUERY_TILE + r] = even | odd << 16;
        }
    }
    let mut maxima = [f32::NEG_INFINITY; QUERY_TILE];
    let mut denominators = [0.0_f32; QUERY_TILE];
    // SAFETY: row offsets stay inside `output` for this task's rows and head.
    let out = unsafe { output.add(first_query * shape.query_width + head * HEAD_DIM) };
    for r in 0..queries {
        unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) }.fill(0.0);
    }
    let mut st = [0.0_f32; KEY_TILE * QUERY_TILE];
    let mut scores = [0.0_f32; QUERY_TILE * KEY_TILE];
    let mut probability_pairs = [0_u32; KEY_TILE / 2 * QUERY_TILE];
    for key_start in (0..visible_end).step_by(KEY_TILE) {
        let keys = (visible_end - key_start).min(KEY_TILE);
        if qk_uses_main_path(queries, keys) && pv_uses_main_path(queries, keys) {
            let profile = shape.profile;
            let t0 = if profile { cycles() } else { 0 };
            let (mut t1, mut t2) = (0, 0);
            unsafe {
                qk_lanes(
                    &query_pairs,
                    queries,
                    operands.keys.as_ptr().add(key_start * PAIRS),
                    keys,
                    shape.scale,
                    &mut st,
                );
                if profile {
                    t1 = cycles();
                }
                if !all_image {
                    mask_lanes(shape, first_absolute, queries, key_start, keys, &mut st);
                }
                let key_pairs = keys.div_ceil(2);
                if FUSED {
                    softmax_pairs(
                        queries,
                        keys,
                        &st,
                        &mut maxima,
                        &mut denominators,
                        out,
                        shape.query_width,
                        &mut probability_pairs,
                    );
                } else {
                    softmax_lanes::<S>(
                        queries,
                        keys,
                        &mut st,
                        &mut maxima,
                        &mut denominators,
                        out,
                        shape.query_width,
                    );
                    for kp in 0..key_pairs {
                        let odd = (2 * kp + 1 < keys).then(|| st.as_ptr().add((2 * kp + 1) * QUERY_TILE));
                        pair_rows::<2>(
                            st.as_ptr().add(2 * kp * QUERY_TILE),
                            odd,
                            probability_pairs.as_mut_ptr().add(kp * QUERY_TILE),
                        );
                    }
                }
                if profile {
                    t2 = cycles();
                }
                pv_lanes(
                    &probability_pairs,
                    queries,
                    key_pairs,
                    operands.values.as_ptr().add(key_start / 2 * HEAD_DIM),
                    out,
                    shape.query_width,
                );
            }
            if profile {
                use std::sync::atomic::Ordering::Relaxed;
                let t3 = cycles();
                STAGE_CYCLES[0].fetch_add(t1 - t0, Relaxed);
                STAGE_CYCLES[1].fetch_add(t2 - t1, Relaxed);
                STAGE_CYCLES[2].fetch_add(t3 - t2, Relaxed);
            }
        } else {
            // Short tiles: the parent's FP32 reference path.
            unsafe {
                reference_key_tile(
                    shape,
                    operands
                        .q
                        .as_ptr()
                        .add(first_query * shape.query_width + head * HEAD_DIM),
                    first_absolute,
                    queries,
                    key_start,
                    keys,
                    operands.k.as_ptr().add(key_start * shape.query_width + head * HEAD_DIM),
                    operands.v.as_ptr().add(key_start * shape.kv_width + kv_head * HEAD_DIM),
                    &mut scores,
                    &mut maxima,
                    &mut denominators,
                    out,
                );
            }
        }
    }
    for r in 0..queries {
        let row = unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) };
        let logsumexp = maxima[r] + denominators[r].ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in row {
            *value = (*value / denominators[r]) * sink_scale;
        }
    }
}

/// The parent's `softmax_lanes` with the fast exp, 16 queries per vector,
/// in one pass over the scores: each lane follows the same sequence (block
/// max, rescale, `p = exp(s - max)`, `denominator += p` in key order), and
/// the probabilities go straight to BF16 pairs instead of back to `st`.
/// Output rows whose factor is exactly 1 are not rewritten (`x * 1 == x`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn softmax_pairs(
    queries: usize,
    keys: usize,
    st: &[f32; KEY_TILE * QUERY_TILE],
    maxima: &mut [f32; QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
    out: *mut f32,
    out_stride: usize,
    pairs: &mut [u32; KEY_TILE / 2 * QUERY_TILE],
) {
    unsafe {
        let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);
        let zero = _mm512_setzero_ps();
        for half in 0..queries.div_ceil(16) {
            let lane0 = 16 * half;
            let at = |j: usize| st.as_ptr().add(j * QUERY_TILE + lane0);
            let mut block_max = neg_inf;
            for j in 0..keys {
                block_max = _mm512_max_ps(_mm512_loadu_ps(at(j)), block_max);
            }
            let old_max = _mm512_loadu_ps(maxima.as_ptr().add(lane0));
            let new_max = _mm512_max_ps(block_max, old_max);
            let skip = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(new_max, neg_inf);
            let first = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(old_max, neg_inf);
            let mut rescale = exp16(_mm512_sub_ps(old_max, new_max));
            rescale = _mm512_mask_blend_ps(first, rescale, zero);
            rescale = _mm512_mask_blend_ps(skip, rescale, _mm512_set1_ps(1.0));
            let mut factors = [0.0_f32; 16];
            _mm512_storeu_ps(factors.as_mut_ptr(), rescale);
            for (lane, &factor) in factors.iter().enumerate() {
                let r = lane0 + lane;
                if r >= queries || skip & (1 << lane) != 0 || factor == 1.0 {
                    continue;
                }
                let row = out.add(r * out_stride);
                let f = _mm512_set1_ps(factor);
                for c in (0..HEAD_DIM).step_by(16) {
                    _mm512_storeu_ps(row.add(c), _mm512_mul_ps(_mm512_loadu_ps(row.add(c)), f));
                }
            }
            let live = !skip;
            let mut denominator = _mm512_mul_ps(_mm512_loadu_ps(denominators.as_ptr().add(lane0)), rescale);
            for kp in 0..keys.div_ceil(2) {
                let even = _mm512_maskz_mov_ps(live, exp16(_mm512_sub_ps(_mm512_loadu_ps(at(2 * kp)), new_max)));
                denominator = _mm512_add_ps(denominator, even);
                let odd = if 2 * kp + 1 < keys {
                    let p = _mm512_maskz_mov_ps(live, exp16(_mm512_sub_ps(_mm512_loadu_ps(at(2 * kp + 1)), new_max)));
                    denominator = _mm512_add_ps(denominator, p);
                    _mm512_slli_epi32::<16>(bf16_lanes(p))
                } else {
                    _mm512_setzero_si512()
                };
                _mm512_storeu_si512(
                    pairs.as_mut_ptr().add(kp * QUERY_TILE + lane0).cast(),
                    _mm512_or_si512(bf16_lanes(even), odd),
                );
            }
            _mm512_storeu_ps(denominators.as_mut_ptr().add(lane0), denominator);
            _mm512_storeu_ps(
                maxima.as_mut_ptr().add(lane0),
                _mm512_mask_blend_ps(skip, new_max, old_max),
            );
        }
    }
}

/// `crate::simd::exp_poly` on 16 lanes: the same IEEE operations as the
/// AVX2 `exp_fast`, so every lane has the same bits.
#[inline(always)]
unsafe fn exp16(x: __m512) -> __m512 {
    use crate::simd::{EXP_LN2_HI, EXP_LN2_LO, EXP_LOG2E, EXP_MAX, EXP_MIN, EXP_P};
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
        let bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)));
        let value = _mm512_mul_ps(y, _mm512_castsi512_ps(bits));
        let value = _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(EXP_MIN)),
            value,
            _mm512_setzero_ps(),
        );
        let value = _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, _mm512_set1_ps(EXP_MAX)),
            value,
            _mm512_set1_ps(f32::INFINITY),
        );
        _mm512_mask_blend_ps(_mm512_cmp_ps_mask::<_CMP_UNORD_Q>(x, x), value, x)
    }
}

/// `st[key * 32 + query] = scale * q[query] . k[key]` with queries as lanes
/// (one or two vectors) and one key pair broadcast per `vdpbf16ps`.
#[inline(always)]
unsafe fn qk_lanes(
    query_pairs: &[u32; PAIRS * QUERY_TILE],
    queries: usize,
    keys_ptr: *const u32,
    keys: usize,
    scale: f32,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
) {
    unsafe {
        if queries > 16 {
            qk_blocks::<2>(query_pairs, keys_ptr, keys, scale, st);
        } else {
            qk_blocks::<1>(query_pairs, keys_ptr, keys, scale, st);
        }
    }
}

#[inline(always)]
unsafe fn qk_blocks<const V: usize>(
    query_pairs: &[u32; PAIRS * QUERY_TILE],
    keys_ptr: *const u32,
    keys: usize,
    scale: f32,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
) {
    unsafe {
        let scale = _mm512_set1_ps(scale);
        let mut key = 0;
        while key + 8 <= keys {
            qk_keys::<V, 8>(query_pairs, keys_ptr, key, scale, st);
            key += 8;
        }
        match keys - key {
            7 => qk_keys::<V, 7>(query_pairs, keys_ptr, key, scale, st),
            6 => qk_keys::<V, 6>(query_pairs, keys_ptr, key, scale, st),
            5 => qk_keys::<V, 5>(query_pairs, keys_ptr, key, scale, st),
            4 => qk_keys::<V, 4>(query_pairs, keys_ptr, key, scale, st),
            3 => qk_keys::<V, 3>(query_pairs, keys_ptr, key, scale, st),
            2 => qk_keys::<V, 2>(query_pairs, keys_ptr, key, scale, st),
            1 => qk_keys::<V, 1>(query_pairs, keys_ptr, key, scale, st),
            _ => {}
        }
    }
}

#[inline(always)]
unsafe fn qk_keys<const V: usize, const N: usize>(
    query_pairs: &[u32; PAIRS * QUERY_TILE],
    keys_ptr: *const u32,
    key: usize,
    scale: __m512,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
) {
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); V]; N];
        for p in 0..PAIRS {
            let mut q = [bh(_mm512_setzero_si512()); V];
            for (v, q) in q.iter_mut().enumerate() {
                *q = bh(_mm512_loadu_si512(
                    query_pairs.as_ptr().add(p * QUERY_TILE + 16 * v).cast(),
                ));
            }
            for (n, acc) in acc.iter_mut().enumerate() {
                let kv = bh(_mm512_set1_epi32(*keys_ptr.add((key + n) * PAIRS + p) as i32));
                for (a, q) in acc.iter_mut().zip(q) {
                    *a = _mm512_dpbf16_ps(*a, q, kv);
                }
            }
        }
        for (n, acc) in acc.iter().enumerate() {
            let dst = st.as_mut_ptr().add((key + n) * QUERY_TILE);
            for (v, a) in acc.iter().enumerate() {
                _mm512_storeu_ps(dst.add(16 * v), _mm512_mul_ps(scale, *a));
            }
        }
    }
}

/// `out[query][d] += sum_j p[j][query] * v[j][d]` over key pairs: 6 query
/// rows by the whole 64-wide row (24 accumulators).
#[inline(always)]
unsafe fn pv_lanes(
    probability_pairs: &[u32; KEY_TILE / 2 * QUERY_TILE],
    queries: usize,
    key_pairs: usize,
    values: *const u32,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut row = 0;
        while row < queries {
            let rows = (queries - row).min(6);
            match rows {
                6 => pv_rows::<6>(probability_pairs, row, key_pairs, values, out, out_stride),
                5 => pv_rows::<5>(probability_pairs, row, key_pairs, values, out, out_stride),
                4 => pv_rows::<4>(probability_pairs, row, key_pairs, values, out, out_stride),
                3 => pv_rows::<3>(probability_pairs, row, key_pairs, values, out, out_stride),
                2 => pv_rows::<2>(probability_pairs, row, key_pairs, values, out, out_stride),
                _ => pv_rows::<1>(probability_pairs, row, key_pairs, values, out, out_stride),
            }
            row += rows;
        }
    }
}

#[inline(always)]
unsafe fn pv_rows<const R: usize>(
    probability_pairs: &[u32; KEY_TILE / 2 * QUERY_TILE],
    row: usize,
    key_pairs: usize,
    values: *const u32,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); 4]; R];
        for kp in 0..key_pairs {
            let vr = values.add(kp * HEAD_DIM);
            let v = [
                bh(_mm512_loadu_si512(vr.cast())),
                bh(_mm512_loadu_si512(vr.add(16).cast())),
                bh(_mm512_loadu_si512(vr.add(32).cast())),
                bh(_mm512_loadu_si512(vr.add(48).cast())),
            ];
            let p = probability_pairs.as_ptr().add(kp * QUERY_TILE + row);
            for (r, acc) in acc.iter_mut().enumerate() {
                let pv = bh(_mm512_set1_epi32(*p.add(r) as i32));
                for (a, v) in acc.iter_mut().zip(v) {
                    *a = _mm512_dpbf16_ps(*a, v, pv);
                }
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            let dst = out.add((row + r) * out_stride);
            for (c, a) in acc.iter().enumerate() {
                let at = dst.add(16 * c);
                _mm512_storeu_ps(at, _mm512_add_ps(*a, _mm512_loadu_ps(at)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available() -> bool {
        crate::kernels::panel_bf16::available()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
    }

    fn values(n: usize, seed: u32, range: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // BF16-exact, so the short tiles' FP32 path sees the same operands.
                let x = ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32 * range;
                half::bf16::from_f32(x).to_f32()
            })
            .collect()
    }

    /// The fused softmax produces the same probabilities, denominators and
    /// maxima as the separate passes, so the same outputs bit for bit.
    #[test]
    fn fused_softmax_matches_separate_passes() {
        if !available() {
            return;
        }
        for (query_len, image_start, image_end) in [(70, 3, 65), (300, 5, 262), (95, 10, 90), (40, 0, 3)] {
            let (heads, kv_heads) = (16, 8);
            let q = values(query_len * heads * 64, 17 + query_len as u32, 3.0);
            let k = values(query_len * heads * 64, 19 + query_len as u32, 3.0);
            let v = values(query_len * kv_heads * 64, 23 + query_len as u32, 3.0);
            let sinks = values(heads, 29, 2.0);
            let shape = Shape {
                query_width: heads * HEAD_DIM,
                kv_width: kv_heads * HEAD_DIM,
                query_offset: 0,
                image_start,
                image_end,
                scale: 0.125,
                profile: false,
            };
            let total = query_len;
            let pairs = total.div_ceil(2);
            let mut keys_bf16 = vec![0_u32; heads * total * 32];
            let mut value_pairs = vec![0_u32; kv_heads * pairs * 64];
            for head in 0..heads {
                for t in 0..total {
                    unsafe {
                        to_bf16(
                            k[(t * heads + head) * 64..].as_ptr(),
                            keys_bf16[(head * total + t) * 32..].as_mut_ptr().cast(),
                        )
                    };
                }
            }
            for kv_head in 0..kv_heads {
                for p in 0..pairs {
                    let even = v[(2 * p * kv_heads + kv_head) * 64..].as_ptr();
                    let odd = (2 * p + 1 < total).then(|| v[((2 * p + 1) * kv_heads + kv_head) * 64..].as_ptr());
                    unsafe { pair_rows::<4>(even, odd, value_pairs[(kv_head * pairs + p) * 64..].as_mut_ptr()) };
                }
            }
            let mut fused = vec![f32::NAN; q.len()];
            let mut separate = vec![f32::NAN; q.len()];
            for head in 0..heads {
                let kv_head = head / 2;
                let operands = Operands {
                    q: &q,
                    k: &k,
                    v: &v,
                    keys: &keys_bf16[head * total * 32..(head + 1) * total * 32],
                    values: &value_pairs[kv_head * pairs * 64..(kv_head + 1) * pairs * 64],
                };
                for tile in 0..query_len.div_ceil(QUERY_TILE) {
                    let queries = (query_len - tile * QUERY_TILE).min(QUERY_TILE);
                    unsafe {
                        both(
                            &operands,
                            &shape,
                            tile,
                            queries,
                            head,
                            kv_head,
                            &sinks,
                            fused.as_mut_ptr(),
                            separate.as_mut_ptr(),
                        );
                    }
                }
            }
            for (i, (a, b)) in fused.iter().zip(&separate).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "query_len {query_len} index {i}");
            }
        }

        #[target_feature(enable = "avx2,fma,avx512f,avx512bf16")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn both(
            operands: &Operands<'_>,
            shape: &Shape,
            tile: usize,
            queries: usize,
            head: usize,
            kv_head: usize,
            sinks: &[f32],
            fused: *mut f32,
            separate: *mut f32,
        ) {
            unsafe {
                tile_head::<crate::simd::Avx2Fast, true>(operands, shape, tile, queries, head, kv_head, sinks, fused);
                tile_head::<crate::simd::Avx2Fast, false>(
                    operands, shape, tile, queries, head, kv_head, sinks, separate,
                );
            }
        }
    }

    /// `store_row` (rows in any order) writes exactly the layouts the kernel
    /// converts itself, so both paths give bitwise equal outputs.
    #[test]
    fn stored_rows_match_the_kernel_conversion() {
        if !available() || !std::is_x86_feature_detected!("avx512bw") {
            return;
        }
        for (query_len, image_start, image_end) in [(70, 3, 65), (95, 10, 90), (129, 1, 128)] {
            let (heads, kv_heads) = (16, 8);
            let q = values(query_len * heads * 64, 17 + query_len as u32, 3.0);
            let k = values(query_len * heads * 64, 19 + query_len as u32, 3.0);
            let v = values(query_len * kv_heads * 64, 23 + query_len as u32, 3.0);
            let sinks = values(heads, 29, 2.0);
            let pairs = query_len.div_ceil(2);
            let mut keys = vec![u32::MAX; heads * query_len * 32];
            let mut value_pairs = vec![u32::MAX; kv_heads * pairs * 64];
            if query_len % 2 == 1 {
                for g in 0..kv_heads {
                    value_pairs[(g * pairs + pairs - 1) * 64..(g * pairs + pairs) * 64].fill(0);
                }
            }
            for row in (0..query_len).rev() {
                unsafe {
                    store_row(
                        &k[row * heads * 64..(row + 1) * heads * 64],
                        &v[row * kv_heads * 64..(row + 1) * kv_heads * 64],
                        row,
                        query_len,
                        heads,
                        kv_heads,
                        keys.as_mut_ptr(),
                        value_pairs.as_mut_ptr(),
                    )
                };
            }
            let mut own = vec![f32::NAN; q.len()];
            let mut stored = vec![f32::NAN; q.len()];
            let (mut scratch_keys, mut scratch_values) = (Vec::new(), Vec::new());
            unsafe {
                compact_prefill(
                    &q,
                    &CompactKv::prefill(&k, &v, query_len, heads, kv_heads, 64),
                    query_len,
                    Geometry::new(0, image_start, image_end),
                    &sinks,
                    &mut own,
                    Bf16Kv::Convert(&mut scratch_keys, &mut scratch_values),
                    PrefillOptions::EXACT,
                );
                compact_prefill(
                    &q,
                    &CompactKv::prefill(&k, &v, query_len, heads, kv_heads, 64),
                    query_len,
                    Geometry::new(0, image_start, image_end),
                    &sinks,
                    &mut stored,
                    Bf16Kv::Converted(&keys, &value_pairs),
                    PrefillOptions::EXACT,
                );
            }
            for (i, (a, b)) in own.iter().zip(&stored).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "query_len {query_len} index {i}");
            }
        }
    }

    fn round(x: f32) -> f64 {
        half::bf16::from_f32(x).to_f64()
    }

    /// Within the probability-rounding bound (relative 2^-9 per probability)
    /// of an f64 reference, for every tile shape (short tiles, odd key
    /// counts, partial query tiles, text rows before and after the image).
    #[test]
    fn matches_bf16_operand_reference() {
        if !available() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for (query_len, image_start, image_end) in [
            (70, 3, 65),
            (300, 5, 262),
            (257, 0, 257),
            (129, 1, 128),
            (40, 0, 3),
            (95, 10, 90),
        ] {
            let (heads, kv_heads) = (16, 8);
            let q = values(query_len * heads * 64, 17 + query_len as u32, 3.0);
            let k = values(query_len * heads * 64, 19 + query_len as u32, 3.0);
            let v = values(query_len * kv_heads * 64, 23 + query_len as u32, 3.0);
            let sinks = values(heads, 29, 2.0);
            let mut out = vec![f32::NAN; q.len()];
            let (mut scratch_keys, mut scratch_values) = (Vec::new(), Vec::new());
            pool.install(|| unsafe {
                compact_prefill(
                    &q,
                    &CompactKv::prefill(&k, &v, query_len, heads, kv_heads, 64),
                    query_len,
                    Geometry::new(0, image_start, image_end),
                    &sinks,
                    &mut out,
                    Bf16Kv::Convert(&mut scratch_keys, &mut scratch_values),
                    PrefillOptions::EXACT,
                )
            });
            let mut worst = 0.0_f64;
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                for row in 0..query_len {
                    let image_query = row >= image_start && row < image_end;
                    let mut logits = Vec::new();
                    for key in 0..query_len {
                        if key <= row || (image_query && key >= image_start && key < image_end) {
                            let dot: f64 = (0..64)
                                .map(|d| {
                                    round(q[(row * heads + head) * 64 + d]) * round(k[(key * heads + head) * 64 + d])
                                })
                                .sum();
                            logits.push((key, dot / 8.0));
                        }
                    }
                    let max = logits.iter().map(|l| l.1).fold(f64::NEG_INFINITY, f64::max);
                    let denominator: f64 = logits.iter().map(|l| (l.1 - max).exp()).sum();
                    let sink = 1.0 / (1.0 + (sinks[head] as f64 - (max + denominator.ln())).exp());
                    for d in 0..64 {
                        let mut sum = 0.0;
                        for &(key, logit) in &logits {
                            sum += (logit - max).exp() * round(v[(key * kv_heads + kv_head) * 64 + d]);
                        }
                        let expected = sum / denominator * sink;
                        let got = out[(row * heads + head) * 64 + d] as f64;
                        worst = worst.max((got - expected).abs());
                    }
                }
            }
            // Probabilities round to BF16 (relative 2^-9) against |v| <= 3,
            // plus FP32 accumulation of logits up to ~70.
            assert!(
                worst <= 3.0 * 2f64.powi(-9) + 2e-3,
                "query_len {query_len}: worst {worst}"
            );
            eprintln!("query_len {query_len}: worst {worst:.2e}");
        }
    }
}
