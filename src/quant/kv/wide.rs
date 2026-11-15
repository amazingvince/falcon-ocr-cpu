//! AVX-512F kernels of verification rows ([`super::decode::pair_rows`] on
//! the `auto` backend). Each 512-bit vector holds two 8-lane vectors of the
//! AVX2 kernels, so every element sees the same FMA chain and the same
//! additions in the same operand order: the logits and outputs are bitwise
//! those of the 8-lane kernels (and so of single-row decode).
use std::arch::x86_64::*;

use super::{TILE, decode::KEY_ROW};

/// Floats per pair of decoded keys in the paired layout: chunk `c` (eight
/// values of the `KEY_ROW`-value key) of keys `2p` and `2p + 1` side by
/// side, so that one 512-bit load holds the chunk of both keys.
pub(super) const PAIR: usize = 2 * KEY_ROW;

/// Offset of chunk `c` of key `j` in the paired layout.
#[inline(always)]
pub(super) fn paired(j: usize, c: usize) -> usize {
    (j / 2) * PAIR + 16 * c + 8 * (j % 2)
}

/// One row's two query heads, each eight-value chunk in both 256-bit halves.
pub(super) struct Queries {
    q0: [__m512; 8],
    q1: [__m512; 8],
}

impl Queries {
    /// Loads the 64-value queries `q0` and `q1`.
    ///
    /// # Safety
    /// AVX-512F must be enabled, and both slices must hold 64 values.
    #[inline(always)]
    pub(super) unsafe fn new(q0: &[f32], q1: &[f32]) -> Self {
        debug_assert!(q0.len() >= 64 && q1.len() >= 64);
        // SAFETY: 8 chunks of 8 values lie inside each 64-value query.
        unsafe {
            let dup = |q: &[f32], c: usize| {
                let half = _mm256_castps_pd(_mm256_loadu_ps(q.as_ptr().add(8 * c)));
                _mm512_castpd_ps(_mm512_broadcast_f64x4(half))
            };
            let mut queries = Self {
                q0: [_mm512_setzero_ps(); 8],
                q1: [_mm512_setzero_ps(); 8],
            };
            for c in 0..8 {
                queries.q0[c] = dup(q0, c);
                queries.q1[c] = dup(q1, c);
            }
            queries
        }
    }
}

/// Lane `i` of the result: lane `IDX[i]` of the last reduction step, whose
/// 128-bit lane `k` holds head 0's keys `k` and `k + 4`, then head 1's.
const IDX: [i32; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// The two heads' unscaled logits of eight decoded keys in the paired layout
/// (`keys` at the first key, which is even): head 0's keys in the low 256
/// bits, head 1's in the high. Per key and head this is `qk2`'s FMA sequence,
/// `(a0+a1)+(a2+a3)` and `Simd::sum`'s tree: halves `lo + hi`, then two
/// horizontal pair additions (odd lane plus even lane, as `haddps` adds).
///
/// # Safety
/// AVX-512F must be enabled, and eight keys must lie in the buffer.
#[inline(always)]
pub(super) unsafe fn qk8(keys: *const f32, q: &Queries) -> __m512 {
    // SAFETY: four pairs of `PAIR` values follow `keys` (the caller's
    // contract); the rest is register arithmetic.
    unsafe {
        let zero = _mm512_setzero_ps();
        let mut t = [[zero; 4]; 2];
        let [t0, t1] = &mut t;
        for (p, (t0, t1)) in t0.iter_mut().zip(t1).enumerate() {
            let k = keys.add(p * PAIR);
            let mut a = [zero; 4];
            let mut b = [zero; 4];
            for j in 0..4 {
                let temporal = _mm512_loadu_ps(k.add(16 * j));
                a[j] = _mm512_fmadd_ps(q.q0[j], temporal, a[j]);
                b[j] = _mm512_fmadd_ps(q.q1[j], temporal, b[j]);
            }
            for j in 0..4 {
                a[j] = _mm512_fmadd_ps(q.q0[4 + j], _mm512_loadu_ps(k.add(16 * (4 + j))), a[j]);
                b[j] = _mm512_fmadd_ps(q.q1[4 + j], _mm512_loadu_ps(k.add(16 * (8 + j))), b[j]);
            }
            *t0 = _mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3]));
            *t1 = _mm512_add_ps(_mm512_add_ps(b[0], b[1]), _mm512_add_ps(b[2], b[3]));
        }
        // Halves: the low plus the high 128 bits of each key, keys 0-3 from
        // the first two pairs and 4-7 from the last two.
        let halves = |x: __m512, y: __m512| {
            _mm512_add_ps(_mm512_shuffle_f32x4::<0x88>(x, y), _mm512_shuffle_f32x4::<0xDD>(x, y))
        };
        // Horizontal pair additions within 128-bit lanes (`haddps`).
        let hadd =
            |x: __m512, y: __m512| _mm512_add_ps(_mm512_shuffle_ps::<0xDD>(x, y), _mm512_shuffle_ps::<0x88>(x, y));
        let pairs = |t: [__m512; 4]| hadd(halves(t[0], t[1]), halves(t[2], t[3]));
        let sums = hadd(pairs(t[0]), pairs(t[1]));
        _mm512_permutexvar_ps(_mm512_loadu_si512(IDX.as_ptr().cast()), sums)
    }
}

/// `decode::qk_row` on the paired layout: the logits of keys `sub..hi`
/// (`keys` holds the sub-tile from key `sub`), eight keys per [`qk8`]. A
/// last group past `hi` computes logits of stale buffered keys that only
/// land in `logits` past `hi`, outside the maximum's mask.
///
/// # Safety
/// AVX-512F must be enabled, `sub` must be a multiple of 8, `hi - sub` at
/// most the buffered keys rounded up to 8, and `hi` at most `TILE`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn qk_row(
    keys: *const f32,
    q0: &[f32],
    q1: &[f32],
    sub: usize,
    hi: usize,
    scale: f32,
    logits: &mut [[f32; TILE]; 2],
    block_max: &mut [f32; 2],
) {
    // SAFETY: every group of eight keys lies in the buffer, and `j + 8` is
    // at most `TILE` (`j` is a multiple of 8 below `hi`).
    unsafe {
        let q = Queries::new(q0, q1);
        let scale = _mm512_set1_ps(scale);
        let mut m = _mm512_set1_ps(f32::NEG_INFINITY);
        for j in (sub..hi).step_by(8) {
            let l = _mm512_mul_ps(qk8(keys.add(paired(j - sub, 0)), &q), scale);
            _mm256_storeu_ps(logits[0].as_mut_ptr().add(j), _mm512_castps512_ps256(l));
            _mm256_storeu_ps(
                logits[1].as_mut_ptr().add(j),
                _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(l))),
            );
            // `max(l, m)` keeps `m` where `l` is NaN, like `Simd::max`.
            let live = (1_u16 << (hi - j).min(8)) - 1;
            m = _mm512_mask_max_ps(m, live | (live << 8), l, m);
        }
        let mut lanes = [0.0_f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), m);
        for (h, lanes) in lanes.chunks_exact(8).enumerate() {
            block_max[h] = lanes.iter().copied().fold(block_max[h], f32::max);
        }
    }
}

/// One row's PV update of both heads over decoded values (`[keys][64]`)
/// with the probabilities `p0` and `p1`: the outputs stay in registers, and
/// each element sees `pv_decoded`'s FMA chain in key order; each denominator
/// adds its probabilities in the same order.
///
/// # Safety
/// AVX-512F must be enabled; `values` must hold `p0.len()` rows of 64 values,
/// and `p1` must be as long as `p0`.
#[inline(always)]
pub(super) unsafe fn pv(
    values: *const f32,
    [p0, p1]: [&[f32]; 2],
    denominator: &mut [f32; 2],
    [o0, o1]: &mut [[f32; 64]; 2],
) {
    debug_assert_eq!(p0.len(), p1.len());
    // SAFETY: the outputs hold 64 values, and `values` holds a row per key.
    unsafe {
        let mut a = [_mm512_setzero_ps(); 4];
        let mut b = [_mm512_setzero_ps(); 4];
        for c in 0..4 {
            a[c] = _mm512_loadu_ps(o0.as_ptr().add(16 * c));
            b[c] = _mm512_loadu_ps(o1.as_ptr().add(16 * c));
        }
        let [mut d0, mut d1] = *denominator;
        for (j, (&x0, &x1)) in p0.iter().zip(p1).enumerate() {
            d0 += x0;
            d1 += x1;
            let (f0, f1) = (_mm512_set1_ps(x0), _mm512_set1_ps(x1));
            let v = values.add(64 * j);
            for c in 0..4 {
                let value = _mm512_loadu_ps(v.add(16 * c));
                a[c] = _mm512_fmadd_ps(f0, value, a[c]);
                b[c] = _mm512_fmadd_ps(f1, value, b[c]);
            }
        }
        *denominator = [d0, d1];
        for c in 0..4 {
            _mm512_storeu_ps(o0.as_mut_ptr().add(16 * c), a[c]);
            _mm512_storeu_ps(o1.as_mut_ptr().add(16 * c), b[c]);
        }
    }
}
