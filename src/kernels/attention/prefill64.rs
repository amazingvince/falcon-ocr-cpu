//! Compact-cache prefill attention for `head_dim == 64`, generic over
//! `crate::simd::Simd` (AVX2/FMA on x86, NEON on aarch64). The AVX2
//! online instantiation with the platform exp is bitwise equal to
//! `tiled::attention_gemm`; NEON and the portable path use the portable
//! fast exp and are bitwise equal to each other.
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
//! With online softmax, key tiles outside gemm's main path (the last,
//! short tiles) run its gemm calls and scalar softmax, keeping the AVX2
//! platform-exp results bitwise equal to `tiled::attention_gemm`. Static
//! softmax uses lane kernels for every tile shape and has its own bitwise
//! scalar oracle.
//!
//! With AVX-512F (and backend `auto`), the QK and PV products use
//! 16-lane `wide` kernels: every element keeps the same FMA chain and epilogue,
//! so results are bitwise identical to the AVX2 kernels, while 32 registers
//! let each broadcast feed all 32 queries of a tile (QK) or a whole 64-wide
//! row (PV). Key blocks stay at 8 rows: key rows are 4 KB apart, so more rows
//! per block would exceed the 8 ways of an L1 set.
//!
//! For online softmax, `ExpMode::Fast` (`RunnerConfig::exp`) switches x86
//! to the portable fast exp (`simd::Avx2Fast`): no scalar fix-ups, bitwise
//! equal to the NEON and portable instantiations, but no longer equal to
//! the platform `expf`.
//! `PrefillOptions::fp32_softmax` swaps in `static_shift`'s tiles on x86;
//! their unchecked polynomial ignores `ExpMode`.
use super::{CompactKv, Geometry, PrefillOptions};
#[cfg(target_arch = "x86_64")]
use crate::config::{ExpMode, PrefillSoftmax};
use crate::simd::Simd as Isa;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
/// BF16 prefill attention using AVX512-BF16 dot products.
pub(super) mod bf16;
/// The keys a query tile sees.
mod span;
/// The FP32 tiles with the static-shift softmax.
mod static_shift;
#[cfg(test)]
mod test_data;
use span::tile_span;

/// Maximum query rows in a prefill tile.
pub(super) const QUERY_TILE: usize = 32;
/// Maximum key rows in a prefill tile.
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

/// Cycle counters of the main-path stages (probe; `Tuning::prefill_profile`).
static STAGE_CYCLES: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
#[inline(always)]
fn cycles() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}
/// Print and reset the stage split (QK, mask+softmax, PV); a no-op unless
/// `enabled`.
pub(crate) fn report_stage_cycles(enabled: bool) {
    if !enabled {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let ns = bf16::CONVERT_NS.swap(0, std::sync::atomic::Ordering::Relaxed);
        if ns > 0 {
            crate::note!("prefill attention BF16 K/V conversion: {:.1} ms", ns as f64 / 1e6);
        }
    }
    let v: Vec<u64> = STAGE_CYCLES
        .iter()
        .map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed))
        .collect();
    let total = v.iter().sum::<u64>().max(1) as f64;
    crate::note!(
        "prefill attention stages (thread cycles): qk {:.1}% softmax {:.1}% pv {:.1}% total {:.2e}",
        100.0 * v[0] as f64 / total,
        100.0 * v[1] as f64 / total,
        100.0 * v[2] as f64 / total,
        total
    );
}

struct Shape {
    query_width: usize,
    kv_width: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    scale: f32,
    /// Accumulate the stage cycle counters (`Tuning::prefill_profile`).
    profile: bool,
}

/// Prefill of `cache` (`head_dim == 64`): a pure prefix, or one with generated
/// keys past the prefix (read from their own rows). On x86, `wide` selects
/// the 16-lane tiles, and static `options.fp32_softmax` selects the
/// static-shift tiles (pure prefixes only). NEON uses the online softmax
/// with the portable fast exp regardless of these options.
///
/// # Safety
/// AVX2/FMA (NEON on aarch64) must be available and the shapes must satisfy
/// `CompactKv::validate`. On x86, `wide` also requires AVX-512F.
pub(super) unsafe fn compact_prefill(
    q: &[f32],
    cache: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    wide: bool,
    options: PrefillOptions,
) {
    #[cfg(target_arch = "x86_64")]
    let tile: TileFn = match (options.fp32_softmax, options.exp == ExpMode::Fast, wide) {
        (PrefillSoftmax::Static, _, true) => static_shift::tile_head_wide,
        (PrefillSoftmax::Static, _, false) => static_shift::tile_head_narrow,
        (PrefillSoftmax::Online, true, true) => tile_head_fast_wide,
        (PrefillSoftmax::Online, true, false) => tile_head_fast,
        (PrefillSoftmax::Online, false, true) => tile_head_native_wide,
        (PrefillSoftmax::Online, false, false) => tile_head_native,
    };
    #[cfg(not(target_arch = "x86_64"))]
    let tile: TileFn = {
        let _ = (wide, options.exp, options.fp32_softmax);
        tile_head_native
    };
    unsafe { compact_prefill_with(tile, q, cache, query_len, geometry, sinks, output, options.profile) }
}

/// Entry for one (query tile, head) with a given instruction set.
type TileFn = unsafe fn(&[f32], &CompactKv<'_>, &Shape, usize, usize, usize, usize, &[f32], *mut f32);

unsafe fn compact_prefill_with(
    tile_head: TileFn,
    q: &[f32],
    cache: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    profile: bool,
) {
    let (n_heads, n_kv_heads) = (cache.n_heads, cache.n_kv_heads);
    let shape = Shape {
        query_width: n_heads * HEAD_DIM,
        kv_width: n_kv_heads * HEAD_DIM,
        query_offset: geometry.query_offset,
        image_start: geometry.image_start,
        image_end: geometry.image_end,
        scale: (HEAD_DIM as f32).sqrt().recip(),
        profile,
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
                tile_head(q, cache, &shape, tile, queries, head, kv_head, sinks, out.get());
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
    cache: &CompactKv<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    kv_head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    unsafe {
        tile_head::<crate::simd::Avx2, false, false>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}
/// AVX-512F QK/PV products with the platform-exact exp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_native_wide(
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
    unsafe { tile_head::<crate::simd::Avx2, true, false>(q, cache, shape, tile, queries, head, kv_head, sinks, output) }
}
/// AVX-512F QK/PV products with the portable fast exp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_fast_wide(
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
    unsafe {
        tile_head::<crate::simd::Avx2Fast, true, false>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}
/// AVX2 with the portable fast exp (`ExpMode::Fast`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_fast(
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
    unsafe {
        tile_head::<crate::simd::Avx2Fast, false, false>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head_native(
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
    unsafe {
        tile_head::<crate::simd::Neon, false, false>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}
/// The portable instantiation (tests; bitwise equal to NEON).
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
    unsafe {
        tile_head::<crate::simd::Portable, false, false>(q, cache, shape, tile, queries, head, kv_head, sinks, output)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_head<S: Isa, const WIDE: bool, const STATIC: bool>(
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
    if STATIC {
        assert_eq!(
            cache.prefix_len, cache.total_len,
            "the static-shift tiles take pure prefixes"
        );
    }
    let first_query = tile * QUERY_TILE;
    let (first_absolute, visible_end, all_image) = tile_span(shape, first_query, queries);
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
        unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) }.fill(0.0);
    }
    let mut st = [0.0_f32; KEY_TILE * QUERY_TILE];
    let mut scores = [0.0_f32; QUERY_TILE * KEY_TILE];
    // The keys of a tile: the expanded prefix rows, the generated rows of
    // this KV head, or (for the one tile crossing the prefix boundary) a
    // gathered copy, so the tile width and the reduction order are those
    // of the expanded layout, as in `tiled::attention_gemm`. Pure prefixes
    // never gather.
    let mut gathered = Vec::<f32>::new();
    for key_start in (0..visible_end).step_by(KEY_TILE) {
        let keys = (visible_end - key_start).min(KEY_TILE);
        let (key_rows, key_stride) = if key_start + keys <= cache.prefix_len {
            let at = key_start * shape.query_width + head * HEAD_DIM;
            (unsafe { cache.prefix_k.as_ptr().add(at) }, shape.query_width)
        } else if key_start >= cache.prefix_len {
            let at = (key_start - cache.prefix_len) * shape.kv_width + kv_head * HEAD_DIM;
            (unsafe { cache.generated_k.as_ptr().add(at) }, shape.kv_width)
        } else {
            gathered.resize(keys * HEAD_DIM, 0.0);
            for key in 0..keys {
                gathered[key * HEAD_DIM..(key + 1) * HEAD_DIM].copy_from_slice(cache.key(key_start + key, head));
            }
            (gathered.as_ptr(), HEAD_DIM)
        };
        let value_rows = unsafe { cache.v.as_ptr().add(key_start * shape.kv_width + kv_head * HEAD_DIM) };
        if STATIC || (qk_uses_main_path(queries, keys) && pv_uses_main_path(queries, keys)) {
            let profile = shape.profile;
            let t0 = if profile { cycles() } else { 0 };
            let mut t1 = 0;
            let mut t2 = 0;
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if WIDE {
                    wide::qk_lanes(&qt, queries, key_rows, key_stride, keys, shape.scale, &mut st);
                } else {
                    qk_lanes::<S>(&qt, queries, key_rows, key_stride, keys, shape.scale, &mut st);
                }
                #[cfg(not(target_arch = "x86_64"))]
                qk_lanes::<S>(&qt, queries, key_rows, key_stride, keys, shape.scale, &mut st);
                if profile {
                    t1 = cycles();
                }
                if !all_image {
                    mask_lanes(shape, first_absolute, queries, key_start, keys, &mut st);
                }
                if STATIC {
                    #[cfg(target_arch = "x86_64")]
                    if WIDE {
                        if all_image {
                            static_shift::exp_lanes16::<false>(queries, keys, &mut st, &mut denominators);
                        } else {
                            static_shift::exp_lanes16::<true>(queries, keys, &mut st, &mut denominators);
                        }
                    } else {
                        static_shift::exp_lanes::<S>(queries, keys, all_image, &mut st, &mut denominators);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    static_shift::exp_lanes::<S>(queries, keys, all_image, &mut st, &mut denominators);
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
                }
                if profile {
                    t2 = cycles();
                }
                #[cfg(target_arch = "x86_64")]
                if WIDE {
                    wide::pv_lanes(&st, queries, keys, value_rows, shape.kv_width, out, shape.query_width);
                } else {
                    pv_lanes::<S>(&st, queries, keys, value_rows, shape.kv_width, out, shape.query_width);
                }
                #[cfg(not(target_arch = "x86_64"))]
                pv_lanes::<S>(&st, queries, keys, value_rows, shape.kv_width, out, shape.query_width);
            }
            if profile {
                use std::sync::atomic::Ordering::Relaxed;
                let t3 = cycles();
                STAGE_CYCLES[0].fetch_add(t1 - t0, Relaxed);
                STAGE_CYCLES[1].fetch_add(t2 - t1, Relaxed);
                STAGE_CYCLES[2].fetch_add(t3 - t2, Relaxed);
            }
        } else {
            unsafe {
                reference_key_tile(
                    shape,
                    q.as_ptr().add(first_query * shape.query_width + head * HEAD_DIM),
                    first_absolute,
                    queries,
                    key_start,
                    keys,
                    key_rows,
                    key_stride,
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
        let row = unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) };
        let logsumexp = if STATIC {
            denominators[r].ln()
        } else {
            maxima[r] + denominators[r].ln()
        };
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
            if key > absolute && !(image_query && key >= shape.image_start && key < shape.image_end) {
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
                // A factor of exactly 1 leaves the row unchanged (`x * 1 == x`).
                if r >= queries || skip_bits & (1 << lane) != 0 || factor == 1.0 {
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
            S::store(maxima.as_mut_ptr().add(lane0), S::select(skip, old_max, new_max));
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

/// AVX-512F forms of [`qk_lanes`] and [`pv_lanes`]. Each element keeps their
/// exact sequence (an FMA chain from +0.0 in the same order, then `scale *
/// acc` or `acc + out`), so outputs are bitwise identical.
#[cfg(target_arch = "x86_64")]
mod wide {
    use super::{HEAD_DIM, KEY_TILE, QUERY_TILE};
    use std::arch::x86_64::*;

    /// `st[key * 32 + query] = scale * sum_d q[query][d] * k[key][d]`: all
    /// the tile's queries (one or two 16-lane vectors) by 8-key blocks.
    ///
    /// # Safety
    /// AVX-512F and FMA must be available. `queries` and `keys` must fit
    /// their tiles; `k` must address `keys` 64-element rows at `k_stride`.
    #[inline(always)]
    pub(super) unsafe fn qk_lanes(
        qt: &[f32; HEAD_DIM * QUERY_TILE],
        queries: usize,
        k: *const f32,
        k_stride: usize,
        keys: usize,
        scale: f32,
        st: &mut [f32; KEY_TILE * QUERY_TILE],
    ) {
        unsafe {
            if queries > 16 {
                qk_blocks::<2>(qt, k, k_stride, keys, scale, st);
            } else {
                qk_blocks::<1>(qt, k, k_stride, keys, scale, st);
            }
        }
    }

    #[inline(always)]
    unsafe fn qk_blocks<const V: usize>(
        qt: &[f32; HEAD_DIM * QUERY_TILE],
        k: *const f32,
        k_stride: usize,
        keys: usize,
        scale: f32,
        st: &mut [f32; KEY_TILE * QUERY_TILE],
    ) {
        unsafe {
            let scale = _mm512_set1_ps(scale);
            let mut key = 0;
            while key + 8 <= keys {
                qk_keys::<V, 8>(qt, k, k_stride, key, scale, st);
                key += 8;
            }
            match keys - key {
                7 => qk_keys::<V, 7>(qt, k, k_stride, key, scale, st),
                6 => qk_keys::<V, 6>(qt, k, k_stride, key, scale, st),
                5 => qk_keys::<V, 5>(qt, k, k_stride, key, scale, st),
                4 => qk_keys::<V, 4>(qt, k, k_stride, key, scale, st),
                3 => qk_keys::<V, 3>(qt, k, k_stride, key, scale, st),
                2 => qk_keys::<V, 2>(qt, k, k_stride, key, scale, st),
                1 => qk_keys::<V, 1>(qt, k, k_stride, key, scale, st),
                _ => {}
            }
        }
    }

    #[inline(always)]
    unsafe fn qk_keys<const V: usize, const N: usize>(
        qt: &[f32; HEAD_DIM * QUERY_TILE],
        k: *const f32,
        k_stride: usize,
        key: usize,
        scale: __m512,
        st: &mut [f32; KEY_TILE * QUERY_TILE],
    ) {
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); V]; N];
            for d in 0..HEAD_DIM {
                let mut q = [_mm512_setzero_ps(); V];
                for (v, q) in q.iter_mut().enumerate() {
                    *q = _mm512_loadu_ps(qt.as_ptr().add(d * QUERY_TILE + 16 * v));
                }
                for (n, acc) in acc.iter_mut().enumerate() {
                    let kv = _mm512_set1_ps(*k.add((key + n) * k_stride + d));
                    for (a, q) in acc.iter_mut().zip(q) {
                        *a = _mm512_fmadd_ps(q, kv, *a);
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

    /// `out[query][d] += sum_j p[j][query] * v[j][d]`: 6 query rows by the
    /// whole 64-wide row (24 accumulators), one broadcast per probability.
    ///
    /// # Safety
    /// AVX-512F and FMA must be available. Counts must fit their tiles; `v`
    /// and `out` must address the respective 64-element rows at their strides.
    /// Output rows must be writable and must not alias inputs.
    #[inline(always)]
    pub(super) unsafe fn pv_lanes(
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
                match rows {
                    6 => pv_rows::<6>(st, row, keys, v, v_stride, out, out_stride),
                    5 => pv_rows::<5>(st, row, keys, v, v_stride, out, out_stride),
                    4 => pv_rows::<4>(st, row, keys, v, v_stride, out, out_stride),
                    3 => pv_rows::<3>(st, row, keys, v, v_stride, out, out_stride),
                    2 => pv_rows::<2>(st, row, keys, v, v_stride, out, out_stride),
                    _ => pv_rows::<1>(st, row, keys, v, v_stride, out, out_stride),
                }
                row += rows;
            }
        }
    }

    #[inline(always)]
    unsafe fn pv_rows<const R: usize>(
        st: &[f32; KEY_TILE * QUERY_TILE],
        row: usize,
        keys: usize,
        v: *const f32,
        v_stride: usize,
        out: *mut f32,
        out_stride: usize,
    ) {
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); 4]; R];
            for j in 0..keys {
                let vr = v.add(j * v_stride);
                let values = [
                    _mm512_loadu_ps(vr),
                    _mm512_loadu_ps(vr.add(16)),
                    _mm512_loadu_ps(vr.add(32)),
                    _mm512_loadu_ps(vr.add(48)),
                ];
                let p = st.as_ptr().add(j * QUERY_TILE + row);
                for (r, acc) in acc.iter_mut().enumerate() {
                    let pv = _mm512_set1_ps(*p.add(r));
                    for (a, value) in acc.iter_mut().zip(values) {
                        *a = _mm512_fmadd_ps(value, pv, *a);
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
    key_stride: usize,
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
            key_stride as isize,
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
            if key > absolute && !(image_query && key >= shape.image_start && key < shape.image_end) {
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
        let out_row = unsafe { std::slice::from_raw_parts_mut(out.add(row * shape.query_width), HEAD_DIM) };
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
mod tests;
