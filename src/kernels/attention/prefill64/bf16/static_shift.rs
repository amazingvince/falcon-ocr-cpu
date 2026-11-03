//! Static-shift softmax for the BF16 prefill tiles (`PrefillSoftmax::Static`,
//! `--tune prefill-softmax=static`).
//!
//! Falcon-OCR RMS-normalizes every query and key head without a weight
//! before the rotary positions, which preserve norms, so a 64-wide head has
//! |q| = |k| = 8 and every score `q.k / 8` lies in [-8, 8] (BF16 operands
//! move it by under 1%; a journal page measures at most 7.993). exp(score)
//! then lies within [e^-8.1, e^8.1], so no shift is needed at all: the
//! running maximum, the block-max pass, and the rescales of the output and
//! the denominator of the online kernel disappear, key tiles add without
//! correction, and the exp moves into the QK epilogue, straight from the
//! accumulators:
//! * the score scale carries log2(e), so `p = 2^t` with `t = n + f`,
//!   `n = round(t)` and |f| <= 1/2;
//! * `2^f` is a degree-3 minimax polynomial (relative error 7.5e-5, against
//!   the up to 2^-8 of the BF16 rounding every probability gets before PV)
//!   and `2^n` comes from `vscalefps`, which also zeroes masked lanes;
//! * the probabilities go to BF16 pairs for the parent's `pv_lanes`, and the
//!   FP32 denominators sum those same rounded values in key order, so the
//!   weights PV applies sum to one. (The online kernel sums the unrounded
//!   values; there the largest probability is exactly 1 and loses nothing to
//!   rounding, here every one rounds, and normalizing by the rounded sum
//!   cancels the common part of that error.)
//!
//! The sink enters at the end through `logsumexp = ln(denominator)`, as in
//! the online kernels. Scores must stay below about 87 in magnitude (beyond
//! that, exp overflows or every term underflows), which the norms guarantee
//! ten times over. Every tile shape takes the BF16 products here, including
//! the short tiles the online kernel computes in FP32.
use super::{Operands, PAIRS, bh, pv_lanes, query_pairs, tile_span};
use crate::kernels::attention::prefill64::{HEAD_DIM, KEY_TILE, QUERY_TILE, STAGE_CYCLES, Shape, cycles};
use std::arch::x86_64::*;

/// Minimax coefficients of `2^f` on [-1/2, 1/2], constant term first
/// (maximum relative error 7.5e-5 in FP32 Horner form).
const EXP2: [f32; 4] = [0.999_928_06, 0.693_260_97, 0.242_611_13, 0.055_171_67];

/// `vpermw` indices that interleave `[even keys | odd keys]` (two 16-lane
/// halves of BF16 probabilities) into per-query pairs.
const INTERLEAVE: [u16; 32] = {
    let mut index = [0_u16; 32];
    let mut lane = 0;
    while lane < 16 {
        index[2 * lane] = lane as u16;
        index[2 * lane + 1] = 16 + lane as u16;
        lane += 1;
    }
    index
};

/// Which of a tile's 32 query lanes see a key (bit `r` for query row `r`):
/// the causal and image rule of the parent's `mask_lanes`, as bit masks.
struct Visibility {
    first_absolute: usize,
    /// Lanes whose query is an image token.
    image_lanes: u32,
    image_start: usize,
    image_end: usize,
    /// All of the tile's queries are image tokens: every key is visible.
    all: bool,
}

impl Visibility {
    fn new(shape: &Shape, first_absolute: usize, all: bool) -> Self {
        let image_lanes = (0..QUERY_TILE)
            .filter(|r| (shape.image_start..shape.image_end).contains(&(first_absolute + r)))
            .fold(0, |lanes, r| lanes | 1 << r);
        Self {
            first_absolute,
            image_lanes,
            image_start: shape.image_start,
            image_end: shape.image_end,
            all,
        }
    }

    /// The lanes that see absolute key position `key`.
    #[inline(always)]
    fn lanes(&self, key: usize) -> u32 {
        if self.all {
            return u32::MAX;
        }
        // Lane r sees every key up to its own position, first_absolute + r.
        let causal = match key.checked_sub(self.first_absolute) {
            None | Some(0) => u32::MAX,
            Some(late) if late >= 32 => 0,
            Some(late) => u32::MAX << late,
        };
        if (self.image_start..self.image_end).contains(&key) {
            causal | self.image_lanes
        } else {
            causal
        }
    }
}

/// One (query tile, head) across all visible key tiles: QK with the exp in
/// its epilogue, then the parent's BF16 PV, then normalization and the sink.
///
/// # Safety
/// AVX2, FMA, AVX-512F, AVX512-BF16 and AVX512BW must be available; shapes as
/// in `compact_prefill`, and every score within the module's bound.
#[target_feature(enable = "avx2,fma,avx512f,avx512bf16,avx512bw")]
pub(super) unsafe fn tile_head(
    operands: &Operands<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    let first_query = tile * QUERY_TILE;
    let (first_absolute, visible_end, all_image) = tile_span(shape, first_query, queries);
    let visibility = Visibility::new(shape, first_absolute, all_image);
    let query_pairs = query_pairs(operands.q, shape, first_query, queries, head);
    // SAFETY: row offsets stay inside `output` for this task's rows and head.
    let out = unsafe { output.add(first_query * shape.query_width + head * HEAD_DIM) };
    for r in 0..queries {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) }.fill(0.0);
    }
    let mut denominators = [0.0_f32; QUERY_TILE];
    let mut probability_pairs = [0_u32; KEY_TILE / 2 * QUERY_TILE];
    for key_start in (0..visible_end).step_by(KEY_TILE) {
        let keys = (visible_end - key_start).min(KEY_TILE);
        let profile = shape.profile;
        let t0 = if profile { cycles() } else { 0 };
        // SAFETY: the head's keys and values hold `visible_end` rows (pairs
        // rounded up); key tiles start at even rows.
        unsafe {
            qk_exp_lanes(
                &query_pairs,
                queries,
                operands.keys.as_ptr().add(key_start * PAIRS),
                key_start,
                keys,
                exponent_scale(shape),
                &visibility,
                &mut denominators,
                &mut probability_pairs,
            );
        }
        let t1 = if profile { cycles() } else { 0 };
        // SAFETY: as above; `out` rows are this task's.
        unsafe {
            pv_lanes(
                &probability_pairs,
                queries,
                keys.div_ceil(2),
                operands.values.as_ptr().add(key_start / 2 * HEAD_DIM),
                out,
                shape.query_width,
            );
        }
        if profile {
            // The softmax runs inside the QK stage here; its counter stays 0.
            use std::sync::atomic::Ordering::Relaxed;
            STAGE_CYCLES[0].fetch_add(t1 - t0, Relaxed);
            STAGE_CYCLES[2].fetch_add(cycles() - t1, Relaxed);
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

/// The score scale times log2(e): `scale * q.k` is the exponent of 2.
fn exponent_scale(shape: &Shape) -> f32 {
    std::f32::consts::LOG2_E * shape.scale
}

/// Keys `[key_start, key_start + keys)` against the tile's queries (one or
/// two 16-lane vectors): `p = 2^(scale * q.k)` for the lanes that see each
/// key and 0 for the rest, stored as BF16 pairs `pairs[key / 2][query]` for
/// `pv_lanes` and summed, as rounded, into `denominators` in key order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn qk_exp_lanes(
    query_pairs: &[u32; PAIRS * QUERY_TILE],
    queries: usize,
    keys_ptr: *const u32,
    key_start: usize,
    keys: usize,
    scale: f32,
    visibility: &Visibility,
    denominators: &mut [f32; QUERY_TILE],
    pairs: &mut [u32; KEY_TILE / 2 * QUERY_TILE],
) {
    unsafe {
        let block = Block {
            query_pairs,
            keys_ptr,
            key_start,
            scale: _mm512_set1_ps(scale),
            interleave: _mm512_loadu_si512(INTERLEAVE.as_ptr().cast()),
            visibility,
        };
        if queries > 16 {
            block.tile::<2>(keys, denominators, pairs);
        } else {
            block.tile::<1>(keys, denominators, pairs);
        }
    }
}

/// The loop invariants of one key tile.
struct Block<'a> {
    query_pairs: &'a [u32; PAIRS * QUERY_TILE],
    keys_ptr: *const u32,
    key_start: usize,
    scale: __m512,
    interleave: __m512i,
    visibility: &'a Visibility,
}

impl Block<'_> {
    /// All `keys` of the tile, eight at a time, for `V` query vectors.
    #[inline(always)]
    unsafe fn tile<const V: usize>(
        &self,
        keys: usize,
        denominators: &mut [f32; QUERY_TILE],
        pairs: &mut [u32; KEY_TILE / 2 * QUERY_TILE],
    ) {
        unsafe {
            let mut sums = [_mm512_setzero_ps(); V];
            for (v, sum) in sums.iter_mut().enumerate() {
                *sum = _mm512_loadu_ps(denominators.as_ptr().add(16 * v));
            }
            let mut key = 0;
            while key + 8 <= keys {
                self.keys::<V, 8>(key, &mut sums, pairs);
                key += 8;
            }
            match keys - key {
                7 => self.keys::<V, 7>(key, &mut sums, pairs),
                6 => self.keys::<V, 6>(key, &mut sums, pairs),
                5 => self.keys::<V, 5>(key, &mut sums, pairs),
                4 => self.keys::<V, 4>(key, &mut sums, pairs),
                3 => self.keys::<V, 3>(key, &mut sums, pairs),
                2 => self.keys::<V, 2>(key, &mut sums, pairs),
                1 => self.keys::<V, 1>(key, &mut sums, pairs),
                _ => {}
            }
            for (v, sum) in sums.iter().enumerate() {
                _mm512_storeu_ps(denominators.as_mut_ptr().add(16 * v), *sum);
            }
        }
    }

    /// `N` keys from tile row `key` (even): the products in the online
    /// kernel's `qk_keys` order, then the epilogue per key pair.
    #[inline(always)]
    unsafe fn keys<const V: usize, const N: usize>(
        &self,
        key: usize,
        sums: &mut [__m512; V],
        pairs: &mut [u32; KEY_TILE / 2 * QUERY_TILE],
    ) {
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); V]; N];
            for p in 0..PAIRS {
                let mut q = [bh(_mm512_setzero_si512()); V];
                for (v, q) in q.iter_mut().enumerate() {
                    *q = bh(_mm512_loadu_si512(
                        self.query_pairs.as_ptr().add(p * QUERY_TILE + 16 * v).cast(),
                    ));
                }
                for (n, acc) in acc.iter_mut().enumerate() {
                    let kv = bh(_mm512_set1_epi32(*self.keys_ptr.add((key + n) * PAIRS + p) as i32));
                    for (a, q) in acc.iter_mut().zip(q) {
                        *a = _mm512_dpbf16_ps(*a, q, kv);
                    }
                }
            }
            for n in (0..N).step_by(2) {
                let even_lanes = self.visibility.lanes(self.key_start + key + n);
                let odd_lanes = if n + 1 < N {
                    self.visibility.lanes(self.key_start + key + n + 1)
                } else {
                    0
                };
                for v in 0..V {
                    let even = exp2_lanes(_mm512_mul_ps(self.scale, acc[n][v]), (even_lanes >> (16 * v)) as u16);
                    let odd = if n + 1 < N {
                        exp2_lanes(_mm512_mul_ps(self.scale, acc[n + 1][v]), (odd_lanes >> (16 * v)) as u16)
                    } else {
                        _mm512_setzero_ps()
                    };
                    // `[bf16(even) x 16 | bf16(odd) x 16]`, then per-lane pairs.
                    let halves: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(odd, even));
                    let pair = _mm512_permutexvar_epi16(self.interleave, halves);
                    _mm512_storeu_si512(pairs.as_mut_ptr().add((key + n) / 2 * QUERY_TILE + 16 * v).cast(), pair);
                    // The denominators sum the rounded probabilities PV
                    // multiplies (the BF16 halves as FP32), even key first.
                    let even = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(pair));
                    let odd = _mm512_castsi512_ps(_mm512_and_si512(pair, _mm512_set1_epi32(0xFFFF_0000_u32 as i32)));
                    sums[v] = _mm512_add_ps(_mm512_add_ps(sums[v], even), odd);
                }
            }
        }
    }
}

/// `2^t` on 16 lanes, +0 where `visible` is clear: `n = round(t)` (ties to
/// even), `f = t - n` (exact), [`EXP2`] in `f` by FMA from the top, then
/// `vscalefps` (exact multiplication by `2^n` while |t| < 126).
#[inline(always)]
unsafe fn exp2_lanes(t: __m512, visible: __mmask16) -> __m512 {
    unsafe {
        let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(t);
        let f = _mm512_sub_ps(t, n);
        let mut p = _mm512_set1_ps(EXP2[3]);
        p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(EXP2[2]));
        p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(EXP2[1]));
        p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(EXP2[0]));
        _mm512_maskz_scalef_ps(visible, p, n)
    }
}

#[cfg(test)]
mod tests;
