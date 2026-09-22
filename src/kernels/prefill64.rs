//! Compact-cache prefill attention for `head_dim == 64`, generic over
//! `crate::simd::Simd` (AVX2/FMA on x86, NEON on aarch64). The AVX2
//! instantiation is bitwise equal to `attention_gemm_compact`; NEON and the
//! portable path use the portable fast exp and are bitwise equal to each other.
//!
//! The reference runs two single-threaded `gemm` calls per (32-query tile,
//! head, 128-key tile) with a scalar mask/softmax between them. For the shapes
//! routed to gemm's main microkernel, every QK and PV element is one FMA chain
//! over the depth from +0.0 followed by `scale * acc` (QK) or `acc + out`
//! (PV); see `qk_uses_main_path` / `pv_uses_main_path`. This module keeps
//! those per-element sequences but
//! * makes the tile's 32 queries the vector lanes: Q is transposed once per
//!   (tile, head) and each key row is read contiguously (no K transpose);
//! * keeps scores key-major so the online softmax runs eight queries per
//!   vector with the platform-exact vector exp, each lane in the reference's
//!   key order;
//! * schedules KV group by KV group, so one group's keys and values (about
//!   5 MB at a full page) stay in cache while every query tile reads them.
//!
//! Key tiles outside gemm's main path (the last, short tiles) run the
//! reference's own gemm calls and scalar softmax, so all tiles stay bitwise
//! equal. Tests compare the kernels with gemm for every tile shape and the
//! whole function with `attention_gemm_compact`.
use crate::simd::Simd as Isa;
use rayon::prelude::*;

pub(super) const QUERY_TILE: usize = 32;
pub(super) const KEY_TILE: usize = 128;
const HEAD_DIM: usize = 64;

/// gemm's main microkernel handles QK (m = keys, n = queries after its
/// transpose, contiguous depth on both sides) only above its horizontal limit.
pub(super) fn qk_uses_main_path(queries: usize, keys: usize) -> bool {
    queries * keys > 16 * 16
}

/// PV (m = 64, n = queries, depth = keys) avoids gevv (depth <= 2) and the
/// single-column gemv.
pub(super) fn pv_uses_main_path(queries: usize, keys: usize) -> bool {
    keys >= 3 && queries >= 2
}

/// Output base pointer shared by tasks that write disjoint (row, head) blocks.
#[derive(Clone, Copy)]
struct OutputPtr(*mut f32);
// SAFETY: every task writes only its own query rows of its own head.
unsafe impl Send for OutputPtr {}
unsafe impl Sync for OutputPtr {}
impl OutputPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

struct Shape {
    query_width: usize,
    kv_width: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    scale: f32,
}

/// Pure prefill (no generated keys): `k` is `[total][n_heads][64]`, `v` is
/// `[total][n_kv_heads][64]`, `q`/`output` are `[query_len][n_heads][64]`.
///
/// # Safety
/// AVX2/FMA must be available and every shape must satisfy the checks of
/// `attention_compact_with_simd` with `head_dim == 64` and `prefix_len ==
/// total_len`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn compact_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    query_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    unsafe {
        compact_prefill_with(
            tile_head_native,
            q,
            k,
            v,
            query_len,
            n_heads,
            n_kv_heads,
            query_offset,
            image_start,
            image_end,
            sinks,
            output,
        )
    }
}

/// Entry for one (query tile, head) with a given instruction set.
type TileFn =
    unsafe fn(&[f32], &[f32], &[f32], &Shape, usize, usize, usize, usize, &[f32], *mut f32);

#[allow(clippy::too_many_arguments)]
unsafe fn compact_prefill_with(
    tile_head: TileFn,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    query_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    let shape = Shape {
        query_width: n_heads * HEAD_DIM,
        kv_width: n_kv_heads * HEAD_DIM,
        query_offset,
        image_start,
        image_end,
        scale: (HEAD_DIM as f32).sqrt().recip(),
    };
    let repeat = n_heads / n_kv_heads;
    let tiles = query_len.div_ceil(QUERY_TILE);
    let out = OutputPtr(output.as_mut_ptr());
    for kv_head in 0..n_kv_heads {
        (0..tiles * repeat).into_par_iter().for_each(|task| {
            let (tile, pair) = (task / repeat, task % repeat);
            let head = kv_head * repeat + pair;
            let queries = (query_len - tile * QUERY_TILE).min(QUERY_TILE);
            // SAFETY: features and shapes were checked by the caller; each
            // task owns rows [tile * 32, tile * 32 + queries) of `head`.
            unsafe {
                tile_head(
                    q,
                    k,
                    v,
                    &shape,
                    tile,
                    queries,
                    head,
                    kv_head,
                    sinks,
                    out.get(),
                );
            }
        });
    }
}

/// One (query tile, head) across all visible key tiles, native ISA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_native(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    unsafe {
        tile_head::<crate::simd::Avx2>(q, k, v, shape, tile, queries, head, kv_head, sinks, output)
    }
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_native(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    unsafe {
        tile_head::<crate::simd::Neon>(q, k, v, shape, tile, queries, head, kv_head, sinks, output)
    }
}
/// The portable instantiation (tests; bitwise equal to NEON).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_portable(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    unsafe {
        tile_head::<crate::simd::Portable>(
            q, k, v, shape, tile, queries, head, kv_head, sinks, output,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head<S: Isa>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
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
    // Rows of image queries see every key below image_end; nothing to mask.
    let all_image = first_absolute >= shape.image_start && last_absolute < shape.image_end;
    let mut qt = [0.0_f32; HEAD_DIM * QUERY_TILE];
    for r in 0..queries {
        let row = &q[(first_query + r) * shape.query_width + head * HEAD_DIM..][..HEAD_DIM];
        for (d, &value) in row.iter().enumerate() {
            qt[d * QUERY_TILE + r] = value;
        }
    }
    let mut maxima = [f32::NEG_INFINITY; QUERY_TILE];
    let mut denominators = [0.0_f32; QUERY_TILE];
    // SAFETY: row offsets stay inside `output` for this task's rows and head.
    let out = unsafe { output.add(first_query * shape.query_width + head * HEAD_DIM) };
    for r in 0..queries {
        unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) }
            .fill(0.0);
    }
    let mut st = [0.0_f32; KEY_TILE * QUERY_TILE];
    let mut scores = [0.0_f32; QUERY_TILE * KEY_TILE];
    for key_start in (0..visible_end).step_by(KEY_TILE) {
        let keys = (visible_end - key_start).min(KEY_TILE);
        let key_rows = unsafe {
            k.as_ptr()
                .add(key_start * shape.query_width + head * HEAD_DIM)
        };
        let value_rows = unsafe {
            v.as_ptr()
                .add(key_start * shape.kv_width + kv_head * HEAD_DIM)
        };
        if qk_uses_main_path(queries, keys) && pv_uses_main_path(queries, keys) {
            unsafe {
                qk_lanes::<S>(
                    &qt,
                    queries,
                    key_rows,
                    shape.query_width,
                    keys,
                    shape.scale,
                    &mut st,
                );
                if !all_image {
                    mask_lanes(shape, first_absolute, queries, key_start, keys, &mut st);
                }
                softmax_lanes::<S>(
                    queries,
                    keys,
                    &mut st,
                    &mut maxima,
                    &mut denominators,
                    out,
                    shape.query_width,
                );
                pv_lanes::<S>(
                    &st,
                    queries,
                    keys,
                    value_rows,
                    shape.kv_width,
                    out,
                    shape.query_width,
                );
            }
        } else {
            unsafe {
                reference_key_tile(
                    shape,
                    q.as_ptr()
                        .add(first_query * shape.query_width + head * HEAD_DIM),
                    first_absolute,
                    queries,
                    key_start,
                    keys,
                    key_rows,
                    value_rows,
                    &mut scores,
                    &mut maxima,
                    &mut denominators,
                    out,
                );
            }
        }
    }
    for r in 0..queries {
        let row =
            unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) };
        let logsumexp = maxima[r] + denominators[r].ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in row {
            *value = (*value / denominators[r]) * sink_scale;
        }
    }
}

/// `st[key * 32 + query] = scale * sum_d q[query][d] * k[key][d]`, the depth
/// summed in ascending order by FMA from +0.0 (gemm's main-path order).
/// Register blocking is 16 queries x 6 keys (12 accumulators, 2 query
/// vectors, 1 broadcast), which fits the 16 AVX2 registers without spills.
#[inline(always)]
unsafe fn qk_lanes<S: Isa>(
    qt: &[f32; HEAD_DIM * QUERY_TILE],
    queries: usize,
    k: *const f32,
    k_stride: usize,
    keys: usize,
    scale: f32,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
) {
    unsafe {
        let scale = S::splat(scale);
        for half in 0..queries.div_ceil(16) {
            let lane0 = 16 * half;
            let mut key = 0;
            while key + 6 <= keys {
                qk_keys::<S, 6>(qt, lane0, k, k_stride, key, scale, st);
                key += 6;
            }
            match keys - key {
                5 => qk_keys::<S, 5>(qt, lane0, k, k_stride, key, scale, st),
                4 => qk_keys::<S, 4>(qt, lane0, k, k_stride, key, scale, st),
                3 => qk_keys::<S, 3>(qt, lane0, k, k_stride, key, scale, st),
                2 => qk_keys::<S, 2>(qt, lane0, k, k_stride, key, scale, st),
                1 => qk_keys::<S, 1>(qt, lane0, k, k_stride, key, scale, st),
                _ => {}
            }
        }
    }
}

#[inline(always)]
unsafe fn qk_keys<S: Isa, const N: usize>(
    qt: &[f32; HEAD_DIM * QUERY_TILE],
    lane0: usize,
    k: *const f32,
    k_stride: usize,
    key: usize,
    scale: S::V,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
) {
    unsafe {
        let mut acc = [[S::zero(); 2]; N];
        for d in 0..HEAD_DIM {
            let lanes = qt.as_ptr().add(d * QUERY_TILE + lane0);
            let q0 = S::load(lanes);
            let q1 = S::load(lanes.add(8));
            for (n, acc) in acc.iter_mut().enumerate() {
                let kv = S::splat(*k.add((key + n) * k_stride + d));
                acc[0] = S::fma(q0, kv, acc[0]);
                acc[1] = S::fma(q1, kv, acc[1]);
            }
        }
        for (n, acc) in acc.iter().enumerate() {
            let dst = st.as_mut_ptr().add((key + n) * QUERY_TILE + lane0);
            S::store(dst, S::mul(scale, acc[0]));
            S::store(dst.add(8), S::mul(scale, acc[1]));
        }
    }
}

/// The reference's causal/image visibility rule for tiles with text queries.
#[inline(always)]
fn mask_lanes(
    shape: &Shape,
    first_absolute: usize,
    queries: usize,
    key_start: usize,
    keys: usize,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
) {
    for r in 0..queries {
        let absolute = first_absolute + r;
        let image_query = absolute >= shape.image_start && absolute < shape.image_end;
        for j in 0..keys {
            let key = key_start + j;
            if key > absolute && !(image_query && key >= shape.image_start && key < shape.image_end)
            {
                st[j * QUERY_TILE + r] = f32::NEG_INFINITY;
            }
        }
    }
}

/// The reference's per-row online softmax step, eight rows per vector. Each
/// lane follows the scalar sequence: block max, new max, (skip if no visible
/// key yet), rescale output and denominator, then `p = exp(s - max)` and
/// `denominator += p` in key order. Probabilities overwrite `st`.
#[inline(always)]
unsafe fn softmax_lanes<S: Isa>(
    queries: usize,
    keys: usize,
    st: &mut [f32; KEY_TILE * QUERY_TILE],
    maxima: &mut [f32; QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let neg_inf = S::splat(f32::NEG_INFINITY);
        for group in 0..queries.div_ceil(8) {
            let lane0 = 8 * group;
            let mut block_max = neg_inf;
            for j in 0..keys {
                let s = S::load(st.as_ptr().add(j * QUERY_TILE + lane0));
                block_max = S::max(s, block_max);
            }
            let old_max = S::load(maxima.as_ptr().add(lane0));
            let new_max = S::max(block_max, old_max);
            let skip = S::eq(new_max, neg_inf);
            let first = S::eq(old_max, neg_inf);
            let mut rescale = S::exp(S::sub(old_max, new_max));
            rescale = S::select(first, S::zero(), rescale);
            rescale = S::select(skip, S::splat(1.0), rescale);
            let mut factors = [0.0_f32; 8];
            S::store(factors.as_mut_ptr(), rescale);
            let skip_bits = S::mask_bits(skip);
            for (lane, &factor) in factors.iter().enumerate() {
                let r = lane0 + lane;
                if r >= queries || skip_bits & (1 << lane) != 0 {
                    continue;
                }
                let row = std::slice::from_raw_parts_mut(out.add(r * out_stride), HEAD_DIM);
                for value in row {
                    *value *= factor;
                }
            }
            // Probabilities for every key first (branch-free), then the rare
            // scalar fix-ups, then the denominator in key order per lane.
            let mut probabilities = [0.0_f32; KEY_TILE * 8];
            let mut masks = [0_u8; KEY_TILE];
            let mut flagged = [0_u64; KEY_TILE / 64];
            for j in 0..keys {
                let at = st.as_ptr().add(j * QUERY_TILE + lane0);
                let (p, lanes) = S::exp_raw(S::sub(S::load(at), new_max));
                let p = S::select(skip, S::zero(), p);
                S::store(probabilities.as_mut_ptr().add(8 * j), p);
                let lanes = lanes & !skip_bits;
                masks[j] = lanes as u8;
                flagged[j / 64] |= u64::from(lanes != 0) << (j % 64);
            }
            let mut maxima_lanes = [0.0_f32; 8];
            S::store(maxima_lanes.as_mut_ptr(), new_max);
            for (word, mut bits) in flagged.into_iter().enumerate() {
                while bits != 0 {
                    let j = 64 * word + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let mut lanes = masks[j];
                    while lanes != 0 {
                        let lane = lanes.trailing_zeros() as usize;
                        lanes &= lanes - 1;
                        let score = st[j * QUERY_TILE + lane0 + lane];
                        probabilities[8 * j + lane] = (score - maxima_lanes[lane]).exp();
                    }
                }
            }
            let mut denominator = S::mul(S::load(denominators.as_ptr().add(lane0)), rescale);
            for j in 0..keys {
                let p = S::load(probabilities.as_ptr().add(8 * j));
                denominator = S::add(denominator, p);
                S::store(st.as_mut_ptr().add(j * QUERY_TILE + lane0), p);
            }
            S::store(denominators.as_mut_ptr().add(lane0), denominator);
            S::store(
                maxima.as_mut_ptr().add(lane0),
                S::select(skip, old_max, new_max),
            );
        }
    }
}

/// `out[query][d] += sum_j p[j][query] * v[j][d]`, the keys summed in
/// ascending order by FMA from +0.0 before the single add (gemm's order).
#[inline(always)]
unsafe fn pv_lanes<S: Isa>(
    st: &[f32; KEY_TILE * QUERY_TILE],
    queries: usize,
    keys: usize,
    v: *const f32,
    v_stride: usize,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut row = 0;
        while row < queries {
            let rows = (queries - row).min(6);
            for d in (0..HEAD_DIM).step_by(16) {
                match rows {
                    6 => pv_rows::<S, 6>(st, row, keys, v, v_stride, d, out, out_stride),
                    5 => pv_rows::<S, 5>(st, row, keys, v, v_stride, d, out, out_stride),
                    4 => pv_rows::<S, 4>(st, row, keys, v, v_stride, d, out, out_stride),
                    3 => pv_rows::<S, 3>(st, row, keys, v, v_stride, d, out, out_stride),
                    2 => pv_rows::<S, 2>(st, row, keys, v, v_stride, d, out, out_stride),
                    _ => pv_rows::<S, 1>(st, row, keys, v, v_stride, d, out, out_stride),
                }
            }
            row += rows;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_rows<S: Isa, const R: usize>(
    st: &[f32; KEY_TILE * QUERY_TILE],
    row: usize,
    keys: usize,
    v: *const f32,
    v_stride: usize,
    d: usize,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut acc = [[S::zero(); 2]; R];
        for j in 0..keys {
            let v0 = S::load(v.add(j * v_stride + d));
            let v1 = S::load(v.add(j * v_stride + d + 8));
            let p = st.as_ptr().add(j * QUERY_TILE + row);
            for (r, acc) in acc.iter_mut().enumerate() {
                let pv = S::splat(*p.add(r));
                acc[0] = S::fma(v0, pv, acc[0]);
                acc[1] = S::fma(v1, pv, acc[1]);
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            let dst = out.add((row + r) * out_stride + d);
            S::store(dst, S::add(acc[0], S::load(dst)));
            S::store(dst.add(8), S::add(acc[1], S::load(dst.add(8))));
        }
    }
}

/// A key tile outside gemm's main path: the reference code verbatim (gemm QK,
/// scalar mask/softmax, gemm PV) on row-major scores.
#[allow(clippy::too_many_arguments)]
unsafe fn reference_key_tile(
    shape: &Shape,
    q: *const f32,
    first_absolute: usize,
    queries: usize,
    key_start: usize,
    keys: usize,
    key_rows: *const f32,
    value_rows: *const f32,
    scores: &mut [f32; QUERY_TILE * KEY_TILE],
    maxima: &mut [f32; QUERY_TILE],
    denominators: &mut [f32; QUERY_TILE],
    out: *mut f32,
) {
    let query_stride = shape.query_width as isize;
    unsafe {
        gemm::gemm(
            queries,
            keys,
            HEAD_DIM,
            scores.as_mut_ptr(),
            1,
            KEY_TILE as isize,
            false,
            q,
            1,
            query_stride,
            key_rows,
            query_stride,
            1,
            0.0_f32,
            shape.scale,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
    for row in 0..queries {
        let absolute = first_absolute + row;
        let image_query = absolute >= shape.image_start && absolute < shape.image_end;
        let row_scores = &mut scores[row * KEY_TILE..row * KEY_TILE + keys];
        let mut block_max = f32::NEG_INFINITY;
        for (col, score) in row_scores.iter_mut().enumerate() {
            let key = key_start + col;
            if key > absolute && !(image_query && key >= shape.image_start && key < shape.image_end)
            {
                *score = f32::NEG_INFINITY;
            }
            block_max = block_max.max(*score);
        }
        let new_max = maxima[row].max(block_max);
        if new_max == f32::NEG_INFINITY {
            row_scores.fill(0.0);
            continue;
        }
        let rescale = if maxima[row] == f32::NEG_INFINITY {
            0.0
        } else {
            (maxima[row] - new_max).exp()
        };
        let out_row =
            unsafe { std::slice::from_raw_parts_mut(out.add(row * shape.query_width), HEAD_DIM) };
        for value in out_row {
            *value *= rescale;
        }
        denominators[row] *= rescale;
        for probability in row_scores.iter_mut() {
            *probability = (*probability - new_max).exp();
            denominators[row] += *probability;
        }
        maxima[row] = new_max;
    }
    unsafe {
        gemm::gemm(
            queries,
            HEAD_DIM,
            keys,
            out,
            1,
            query_stride,
            true,
            scores.as_ptr(),
            1,
            KEY_TILE as isize,
            value_rows,
            1,
            shape.kv_width as isize,
            1.0_f32,
            1.0_f32,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32, range: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32 * range;
                // Signed zeros and a subnormal exercise edge rounding.
                match i % 97 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::from_bits(3),
                    _ => x,
                }
            })
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    fn available() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    /// Random prefill operands: (q, k, v, sinks) for 16 query / 8 KV heads.
    fn operands(query_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            values(query_len * 16 * 64, 17 + query_len as u32, 3.0),
            values(query_len * 16 * 64, 19 + query_len as u32, 3.0),
            values(query_len * 8 * 64, 23 + query_len as u32, 3.0),
            values(16, 29, 2.0),
        )
    }

    const CASES: [(usize, usize, usize); 7] = [
        (70, 3, 65),
        (300, 5, 262),
        (257, 0, 257),
        (160, 16, 144),
        (129, 1, 128),
        (40, 0, 3),
        (95, 10, 90),
    ];

    fn run(tile: TileFn, query_len: usize, image: (usize, usize)) -> Vec<f32> {
        let (q, k, v, sinks) = operands(query_len);
        let mut out = vec![f32::NAN; q.len()];
        unsafe {
            compact_prefill_with(
                tile, &q, &k, &v, query_len, 16, 8, 0, image.0, image.1, &sinks, &mut out,
            );
        }
        out
    }

    /// The portable instantiation (fast exp) stays close to the reference.
    #[test]
    fn portable_prefill_is_close_to_reference() {
        for (query_len, image_start, image_end) in CASES {
            let (q, k, v, sinks) = operands(query_len);
            let mut expected = vec![f32::NAN; q.len()];
            super::super::attention_gemm_compact(
                &q,
                &k,
                &[],
                &v,
                query_len,
                16,
                8,
                64,
                0,
                image_start,
                image_end,
                &sinks,
                &mut expected,
            );
            let actual = run(tile_head_portable, query_len, (image_start, image_end));
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                    "query_len {query_len} index {i}: {a} vs {b}"
                );
            }
        }
    }

    /// NEON is bitwise equal to the portable instantiation.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_prefill_matches_portable_bitwise() {
        for (query_len, image_start, image_end) in CASES {
            let native = run(tile_head_native, query_len, (image_start, image_end));
            let portable = run(tile_head_portable, query_len, (image_start, image_end));
            for (i, (a, b)) in native.iter().zip(&portable).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "query_len {query_len} index {i}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn qk_via_lanes(
        q: &[f32],
        q_stride: usize,
        queries: usize,
        k: &[f32],
        k_stride: usize,
        keys: usize,
    ) -> [f32; KEY_TILE * QUERY_TILE] {
        let mut qt = [0.0_f32; HEAD_DIM * QUERY_TILE];
        for r in 0..queries {
            for d in 0..HEAD_DIM {
                qt[d * QUERY_TILE + r] = q[r * q_stride + d];
            }
        }
        let mut st = [f32::NAN; KEY_TILE * QUERY_TILE];
        unsafe {
            qk_lanes::<crate::simd::Avx2>(&qt, queries, k.as_ptr(), k_stride, keys, 0.125, &mut st)
        };
        st
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn qk_lanes_match_gemm_main_path_for_every_tile_shape() {
        if !available() {
            return;
        }
        let stride = 1024;
        let q = values(QUERY_TILE * stride, 3, 4.0);
        let k = values(KEY_TILE * stride, 5, 4.0);
        let mut checked = 0;
        for queries in 1..=QUERY_TILE {
            for keys in 1..=KEY_TILE {
                if !qk_uses_main_path(queries, keys) {
                    continue;
                }
                let mut expected = [f32::NAN; QUERY_TILE * KEY_TILE];
                unsafe {
                    gemm::gemm(
                        queries,
                        keys,
                        HEAD_DIM,
                        expected.as_mut_ptr(),
                        1,
                        KEY_TILE as isize,
                        false,
                        q.as_ptr(),
                        1,
                        stride as isize,
                        k.as_ptr(),
                        stride as isize,
                        1,
                        0.0_f32,
                        0.125,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                let st = unsafe { qk_via_lanes(&q, stride, queries, &k, stride, keys) };
                for row in 0..queries {
                    for key in 0..keys {
                        assert_eq!(
                            st[key * QUERY_TILE + row].to_bits(),
                            expected[row * KEY_TILE + key].to_bits(),
                            "queries {queries} keys {keys} row {row} key {key}"
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 3000);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pv_lanes_match_gemm_main_path_for_every_tile_shape() {
        if !available() {
            return;
        }
        let (v_stride, out_stride) = (512, 1024);
        let v = values(KEY_TILE * v_stride, 7, 4.0);
        let probabilities: Vec<f32> = values(QUERY_TILE * KEY_TILE, 11, 1.0)
            .into_iter()
            .map(|p| if p < -0.7 { 0.0 } else { p.abs() })
            .collect();
        let mut row_major = [0.0_f32; QUERY_TILE * KEY_TILE];
        let mut key_major = [0.0_f32; KEY_TILE * QUERY_TILE];
        for r in 0..QUERY_TILE {
            for j in 0..KEY_TILE {
                row_major[r * KEY_TILE + j] = probabilities[r * KEY_TILE + j];
                key_major[j * QUERY_TILE + r] = probabilities[r * KEY_TILE + j];
            }
        }
        let initial = values(QUERY_TILE * out_stride, 13, 4.0);
        let mut checked = 0;
        for queries in 1..=QUERY_TILE {
            for keys in 1..=KEY_TILE {
                if !pv_uses_main_path(queries, keys) {
                    continue;
                }
                let mut expected = initial.clone();
                unsafe {
                    gemm::gemm(
                        queries,
                        HEAD_DIM,
                        keys,
                        expected.as_mut_ptr(),
                        1,
                        out_stride as isize,
                        true,
                        row_major.as_ptr(),
                        1,
                        KEY_TILE as isize,
                        v.as_ptr(),
                        1,
                        v_stride as isize,
                        1.0_f32,
                        1.0_f32,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                let mut actual = initial.clone();
                unsafe {
                    pv_lanes::<crate::simd::Avx2>(
                        &key_major,
                        queries,
                        keys,
                        v.as_ptr(),
                        v_stride,
                        actual.as_mut_ptr(),
                        out_stride,
                    );
                }
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "queries {queries} keys {keys} {i}"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 3000);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn whole_prefill_matches_reference_bitwise() {
        if !available() {
            return;
        }
        let (n_heads, n_kv_heads) = (16, 8);
        // (query_len, image_start, image_end): tail tiles, text before and
        // after the image, image ends inside and on key-tile boundaries.
        for (query_len, image_start, image_end) in [
            (70, 3, 65),
            (300, 5, 262),
            (257, 0, 257),
            (160, 16, 144),
            (129, 1, 128),
            (40, 0, 3),
            (95, 10, 90),
        ] {
            let q = values(query_len * n_heads * 64, 17 + query_len as u32, 3.0);
            let k = values(query_len * n_heads * 64, 19 + query_len as u32, 3.0);
            let v = values(query_len * n_kv_heads * 64, 23 + query_len as u32, 3.0);
            let sinks = values(n_heads, 29, 2.0);
            let mut expected = vec![f32::NAN; q.len()];
            super::super::attention_gemm_compact(
                &q,
                &k,
                &[],
                &v,
                query_len,
                n_heads,
                n_kv_heads,
                64,
                0,
                image_start,
                image_end,
                &sinks,
                &mut expected,
            );
            let mut actual = vec![f32::NAN; q.len()];
            unsafe {
                compact_prefill(
                    &q,
                    &k,
                    &v,
                    query_len,
                    n_heads,
                    n_kv_heads,
                    0,
                    image_start,
                    image_end,
                    &sinks,
                    &mut actual,
                );
            }
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "query_len {query_len} image {image_start}..{image_end} index {i}"
                );
            }
        }
    }
}
