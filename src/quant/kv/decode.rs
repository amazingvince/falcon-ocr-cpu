//! Decode attention over the split cache: record stores, the per-group
//! kernels and their `#[target_feature]` entries.
use super::{
    MAX_ROWS, SPATIAL, TEMPORAL, TILE, VALUE,
    codec::{bf16_f32, nibble},
};
/// A decode kernel over one prefix/tail store pair (see `SplitPrefix::dispatch`).
pub(super) trait PairKernel {
    unsafe fn run<R: RecordStore, T: RecordStore>(&mut self, store: &R, tail: &T);
}

/// [`pair_rows`] over several query rows of one (group, chunk).
pub(super) struct RowsKernel<'a, 'b> {
    pub(super) spans: &'a [Span<'b>],
    pub(super) parts: &'a mut [Partial],
    pub(super) fast_exp: bool,
}
impl PairKernel for RowsKernel<'_, '_> {
    #[inline(always)]
    unsafe fn run<R: RecordStore, T: RecordStore>(&mut self, store: &R, tail: &T) {
        unsafe {
            if self.fast_exp {
                pair_rows_native_fast(store, tail, self.spans, self.parts)
            } else {
                pair_rows_native(store, tail, self.spans, self.parts)
            }
        }
    }
}

/// Unnormalized online-softmax state of both heads of one group over a span.
#[derive(Clone, Copy)]
pub(super) struct Partial {
    pub(super) out: [[f32; 64]; 2],
    pub(super) max: [f32; 2],
    pub(super) denominator: [f32; 2],
}
impl Partial {
    pub(super) const EMPTY: Self = Self {
        out: [[0.0; 64]; 2],
        max: [f32::NEG_INFINITY; 2],
        denominator: [0.0; 2],
    };
}

/// One task of decode attention: the positions `start..end` of a group
/// (prefix records below `prefix_len`, tail records after) against the
/// queries of its two heads.
pub(super) struct Span<'a> {
    pub(super) prefix_len: usize,
    pub(super) group: usize,
    /// First tail record of the group.
    pub(super) tail_base: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) q0: &'a [f32],
    pub(super) q1: &'a [f32],
}

/// Offsets of the key halves and the value inside a record.
#[derive(Clone, Copy)]
struct Layout {
    pub(super) temporal: usize,
    pub(super) spatial: [usize; 2],
    pub(super) value: usize,
}
/// Prefix records: shared temporal half, one spatial half per head, value.
const PREFIX: Layout = Layout {
    temporal: TEMPORAL,
    spatial: SPATIAL,
    value: VALUE,
};
/// Tail records: one key for both heads (its halves at 0 and 32), value.
const TAIL: Layout = Layout {
    temporal: 0,
    spatial: [32, 32],
    value: 64,
};

use crate::simd::Simd as Isa;

/// Eight decoded FP32 values of one `W`-element record, at an offset that is
/// a multiple of 8, as a vector of instruction set `S`.
pub(super) trait RecordStore {
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V;
}
/// FP32 records of `W` elements.
pub(super) struct F32Rec<'a, const W: usize>(pub(super) &'a [f32]);
/// 8-bit records of `W` elements.
pub(super) struct Q8Rec<'a, const W: usize> {
    pub(super) codes: &'a [i8],
    /// One BF16 scale per 32 elements.
    pub(super) scales: &'a [u16],
}
impl<const W: usize> RecordStore for F32Rec<'_, W> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * W <= self.0.len());
        unsafe { S::load(self.0.as_ptr().add(record * W + offset)) }
    }
}
impl<const W: usize> RecordStore for Q8Rec<'_, W> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * W <= self.codes.len());
        // fl(code * scale), exactly the scalar dequantization.
        unsafe {
            S::mul(
                S::load_i8(self.codes.as_ptr().add(record * W + offset)),
                S::splat(bf16_f32(*self.scales.get_unchecked((record * W + offset) / 32))),
            )
        }
    }
}

/// 16-bit records of `W` elements.
pub(super) struct Q16Rec<'a, const W: usize> {
    pub(super) codes: &'a [i16],
    /// One BF16 scale per 32 elements.
    pub(super) scales: &'a [u16],
}
impl<const W: usize> RecordStore for Q16Rec<'_, W> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * W <= self.codes.len());
        // fl(code * scale), exactly the scalar dequantization.
        unsafe {
            S::mul(
                S::load_i16(self.codes.as_ptr().add(record * W + offset)),
                S::splat(bf16_f32(*self.scales.get_unchecked((record * W + offset) / 32))),
            )
        }
    }
}

/// 4-bit records of `W` elements.
pub(super) struct Q4Rec<'a, const W: usize> {
    /// Two 4-bit codes per byte, element `2i` in the low nibble.
    pub(super) codes: &'a [u8],
    /// One BF16 scale per 32 elements.
    pub(super) scales: &'a [u16],
}
impl<const W: usize> RecordStore for Q4Rec<'_, W> {
    /// Unpacks the eight nibbles into `i8` codes and decodes them like
    /// [`Q8Rec`], so every instruction set is correct without new vector
    /// code (a vector nibble unpack would be faster).
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        let at = record * W + offset;
        debug_assert!(at.is_multiple_of(8) && (record + 1) * W <= 2 * self.codes.len());
        // SAFETY: callers pass a stored record and an offset below `W` that is
        // a multiple of 8 (the trait's contract), so the four code bytes at
        // `at / 2` and the scale at `at / 32` are in bounds; the bytes are read
        // unaligned, the eight codes come from a local array, and the caller
        // runs with `S`'s instruction set enabled.
        unsafe {
            let bytes = self.codes.as_ptr().add(at / 2).cast::<[u8; 4]>().read_unaligned();
            let codes: [i8; 8] = std::array::from_fn(|i| nibble(bytes[i / 2], i % 2));
            // fl(code * scale), exactly the scalar dequantization.
            S::mul(
                S::load_i8(codes.as_ptr()),
                S::splat(bf16_f32(*self.scales.get_unchecked(at / 32))),
            )
        }
    }
}

/// `dot64(q, key)` for both heads of a record: the shared temporal half feeds
/// both accumulator sets first (dot64's i = 0 pass), then each spatial half
/// (its i = 32 pass). Per head this is dot64's exact FMA sequence and
/// reduction tree (`(a0+a1)+(a2+a3)`, then `Simd::sum`), which is also
/// `simd::dot` over 64 elements (the tail layout's single key).
#[inline(always)]
unsafe fn qk2<S: Isa, R: RecordStore>(store: &R, layout: Layout, record: usize, q0: &[f32], q1: &[f32]) -> (f32, f32) {
    unsafe {
        let mut a = [S::zero(); 4];
        let mut b = [S::zero(); 4];
        for j in 0..4 {
            let t = store.load8::<S>(record, layout.temporal + 8 * j);
            a[j] = S::fma(S::load(q0.as_ptr().add(8 * j)), t, a[j]);
            b[j] = S::fma(S::load(q1.as_ptr().add(8 * j)), t, b[j]);
        }
        for j in 0..4 {
            let s0 = store.load8::<S>(record, layout.spatial[0] + 8 * j);
            a[j] = S::fma(S::load(q0.as_ptr().add(32 + 8 * j)), s0, a[j]);
            let s1 = if layout.spatial[1] == layout.spatial[0] {
                s0
            } else {
                store.load8::<S>(record, layout.spatial[1] + 8 * j)
            };
            b[j] = S::fma(S::load(q1.as_ptr().add(32 + 8 * j)), s1, b[j]);
        }
        (
            S::sum(S::add(S::add(a[0], a[1]), S::add(a[2], a[3]))),
            S::sum(S::add(S::add(b[0], b[1]), S::add(b[2], b[3]))),
        )
    }
}

/// `axpy64` into both heads' outputs with one load of each value vector.
#[inline(always)]
unsafe fn pv2<S: Isa>(v: [S::V; 8], p0: f32, p1: f32, o0: &mut [f32; 64], o1: &mut [f32; 64]) {
    unsafe {
        let f0 = S::splat(p0);
        let f1 = S::splat(p1);
        for (chunk, value) in v.into_iter().enumerate() {
            let i = 8 * chunk;
            S::store(o0.as_mut_ptr().add(i), S::fma(f0, value, S::load(o0.as_ptr().add(i))));
            S::store(o1.as_mut_ptr().add(i), S::fma(f1, value, S::load(o1.as_ptr().add(i))));
        }
    }
}

#[inline(always)]
unsafe fn record_value<S: Isa, R: RecordStore>(store: &R, layout: Layout, record: usize) -> [S::V; 8] {
    let value = layout.value;
    unsafe {
        [
            store.load8::<S>(record, value),
            store.load8::<S>(record, value + 8),
            store.load8::<S>(record, value + 16),
            store.load8::<S>(record, value + 24),
            store.load8::<S>(record, value + 32),
            store.load8::<S>(record, value + 40),
            store.load8::<S>(record, value + 48),
            store.load8::<S>(record, value + 56),
        ]
    }
}

/// Both heads of one KV group over positions `[start, end)` (tile aligned),
/// prefix positions from `store` and generated ones from `tail`.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // key index j addresses both heads' logits and the record
unsafe fn pair<S: Isa, R: RecordStore, T: RecordStore>(store: &R, tail: &T, span: &Span<'_>, part: &mut Partial) {
    let scale = (64_f32).sqrt().recip();
    let [o0, o1] = &mut part.out;
    o0.fill(0.0);
    o1.fill(0.0);
    let mut max = [f32::NEG_INFINITY; 2];
    let mut denominator = [0.0_f32; 2];
    let mut logits = [[0.0_f32; TILE]; 2];
    let first_record = span.group * span.prefix_len;
    let tail_record = |key: usize| span.tail_base + (key - span.prefix_len);
    let mut start = span.start;
    // SAFETY: records and tail offsets are within the shape-checked stores.
    unsafe {
        while start < span.end {
            let len = (span.end - start).min(TILE);
            let mut block_max = [f32::NEG_INFINITY; 2];
            for j in 0..len {
                let key = start + j;
                let (s0, s1) = if key < span.prefix_len {
                    qk2::<S, R>(store, PREFIX, first_record + key, span.q0, span.q1)
                } else {
                    qk2::<S, T>(tail, TAIL, tail_record(key), span.q0, span.q1)
                };
                logits[0][j] = s0 * scale;
                block_max[0] = block_max[0].max(logits[0][j]);
                logits[1][j] = s1 * scale;
                block_max[1] = block_max[1].max(logits[1][j]);
            }
            let mut new_max = [0.0_f32; 2];
            for (h, out) in [&mut *o0, &mut *o1].into_iter().enumerate() {
                new_max[h] = max[h].max(block_max[h]);
                let rescale = if max[h] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (max[h] - new_max[h]).exp()
                };
                for value in out.iter_mut() {
                    *value *= rescale;
                }
                denominator[h] *= rescale;
            }
            S::exp_shifted(&mut logits[0][..len], new_max[0]);
            S::exp_shifted(&mut logits[1][..len], new_max[1]);
            for j in 0..len {
                let key = start + j;
                let p0 = logits[0][j];
                denominator[0] += p0;
                let p1 = logits[1][j];
                denominator[1] += p1;
                let value = if key < span.prefix_len {
                    record_value::<S, R>(store, PREFIX, first_record + key)
                } else {
                    record_value::<S, T>(tail, TAIL, tail_record(key))
                };
                pv2::<S>(value, p0, p1, o0, o1);
            }
            max = new_max;
            start += len;
        }
    }
    part.max = max;
    part.denominator = denominator;
}

/// Per-ISA entry for [`pair`] (the target features enclose the whole loop).
/// Each entry holds one instantiation: a runtime choice inside one
/// `#[target_feature]` function made the exact kernel ~50% slower.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pair_native<R: RecordStore, T: RecordStore>(store: &R, tail: &T, span: &Span<'_>, part: &mut Partial) {
    unsafe { pair::<crate::simd::Avx2, R, T>(store, tail, span, part) }
}
/// [`pair_native`] with the portable polynomial exp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pair_native_fast<R: RecordStore, T: RecordStore>(store: &R, tail: &T, span: &Span<'_>, part: &mut Partial) {
    unsafe { pair::<crate::simd::Avx2Fast, R, T>(store, tail, span, part) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn pair_native<R: RecordStore, T: RecordStore>(store: &R, tail: &T, span: &Span<'_>, part: &mut Partial) {
    unsafe { pair::<crate::simd::Neon, R, T>(store, tail, span, part) }
}
/// NEON's exp is already the polynomial one.
#[cfg(target_arch = "aarch64")]
unsafe fn pair_native_fast<R: RecordStore, T: RecordStore>(store: &R, tail: &T, span: &Span<'_>, part: &mut Partial) {
    unsafe { pair_native(store, tail, span, part) }
}
/// [`pair_native`] or [`pair_native_fast`] (`SplitPrefix::with_fast_exp`).
#[inline(always)]
pub(super) unsafe fn pair_entry<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    span: &Span<'_>,
    part: &mut Partial,
    fast_exp: bool,
) {
    unsafe {
        if fast_exp {
            pair_native_fast(store, tail, span, part)
        } else {
            pair_native(store, tail, span, part)
        }
    }
}

/// Keys per sub-tile that [`pair_rows`] decodes once for all rows.
const SUB: usize = 32;
/// Decoded key layout in the sub-tile buffer: temporal half, then the two
/// heads' spatial halves.
const KEY_ROW: usize = 96;

/// One record's decoded key halves into `dst` (`KEY_ROW` values).
#[inline(always)]
unsafe fn decode_key<S: Isa, R: RecordStore>(store: &R, layout: Layout, record: usize, dst: *mut f32) {
    unsafe {
        for j in 0..4 {
            S::store(dst.add(8 * j), store.load8::<S>(record, layout.temporal + 8 * j));
            S::store(dst.add(32 + 8 * j), store.load8::<S>(record, layout.spatial[0] + 8 * j));
            S::store(dst.add(64 + 8 * j), store.load8::<S>(record, layout.spatial[1] + 8 * j));
        }
    }
}

/// [`qk2`] on a decoded key (`decode_key`): the same FMA sequence and
/// reduction per head.
#[inline(always)]
unsafe fn qk2_decoded<S: Isa>(key: *const f32, q0: &[f32], q1: &[f32]) -> (f32, f32) {
    unsafe {
        let mut a = [S::zero(); 4];
        let mut b = [S::zero(); 4];
        for j in 0..4 {
            let t = S::load(key.add(8 * j));
            a[j] = S::fma(S::load(q0.as_ptr().add(8 * j)), t, a[j]);
            b[j] = S::fma(S::load(q1.as_ptr().add(8 * j)), t, b[j]);
        }
        for j in 0..4 {
            a[j] = S::fma(S::load(q0.as_ptr().add(32 + 8 * j)), S::load(key.add(32 + 8 * j)), a[j]);
            b[j] = S::fma(S::load(q1.as_ptr().add(32 + 8 * j)), S::load(key.add(64 + 8 * j)), b[j]);
        }
        (
            S::sum(S::add(S::add(a[0], a[1]), S::add(a[2], a[3]))),
            S::sum(S::add(S::add(b[0], b[1]), S::add(b[2], b[3]))),
        )
    }
}

/// One head's `pv2` updates over decoded values (`[keys][64]`) with the
/// output held in registers, and the denominator in the same key order.
#[inline(always)]
unsafe fn pv_decoded<S: Isa>(values: *const f32, probabilities: &[f32], denominator: &mut f32, out: &mut [f32; 64]) {
    unsafe {
        let mut o = [S::zero(); 8];
        for (c, o) in o.iter_mut().enumerate() {
            *o = S::load(out.as_ptr().add(8 * c));
        }
        for (j, &p) in probabilities.iter().enumerate() {
            *denominator += p;
            let f = S::splat(p);
            let v = values.add(j * 64);
            for (c, o) in o.iter_mut().enumerate() {
                *o = S::fma(f, S::load(v.add(8 * c)), *o);
            }
        }
        for (c, o) in o.iter().enumerate() {
            S::store(out.as_mut_ptr().add(8 * c), *o);
        }
    }
}

/// [`pair`] for several query rows of one group over spans that start at the
/// same position and end at their own lengths (empty spans are skipped). Each
/// row performs exactly [`pair`]'s operation sequence (per element: the same
/// FMA chains in key order, reductions, softmax and denominators). The rows
/// share the record decoding: each 32-key sub-tile's keys and values are
/// decoded once into a small buffer, and each row then runs over it with its
/// outputs in registers.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // key index j addresses every row's logits and the record
unsafe fn pair_rows<S: Isa, R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    spans: &[Span<'_>],
    parts: &mut [Partial],
) {
    let n = spans.len();
    debug_assert!(n <= MAX_ROWS && parts.len() == n);
    let scale = (64_f32).sqrt().recip();
    let mut max = [[f32::NEG_INFINITY; 2]; MAX_ROWS];
    let mut denominator = [[0.0_f32; 2]; MAX_ROWS];
    let mut logits = [[[0.0_f32; TILE]; 2]; MAX_ROWS];
    for part in parts.iter_mut() {
        part.out[0].fill(0.0);
        part.out[1].fill(0.0);
    }
    let first = &spans[0];
    let (prefix_len, tail_base) = (first.prefix_len, first.tail_base);
    let first_record = first.group * prefix_len;
    let tail_record = |key: usize| tail_base + (key - prefix_len);
    let live = |span: &Span<'_>| span.start < span.end;
    let Some(mut start) = spans.iter().filter(|s| live(s)).map(|s| s.start).min() else {
        return;
    };
    debug_assert!(spans.iter().filter(|s| live(s)).all(|s| s.start == start));
    let end = spans.iter().map(|s| s.end).max().unwrap_or(start);
    let mut keys = [0.0_f32; SUB * KEY_ROW];
    let mut values = [0.0_f32; SUB * 64];
    // SAFETY: records and tail offsets are within the shape-checked stores.
    unsafe {
        while start < end {
            let mut lens = [0_usize; MAX_ROWS];
            for (r, span) in spans.iter().enumerate() {
                if live(span) && span.end > start {
                    lens[r] = (span.end - start).min(TILE);
                }
            }
            let len = lens[..n].iter().copied().max().unwrap_or(0);
            let mut block_max = [[f32::NEG_INFINITY; 2]; MAX_ROWS];
            for sub in (0..len).step_by(SUB) {
                let width = SUB.min(len - sub);
                for j in 0..width {
                    let key = start + sub + j;
                    let dst = keys.as_mut_ptr().add(j * KEY_ROW);
                    if key < prefix_len {
                        decode_key::<S, R>(store, PREFIX, first_record + key, dst);
                    } else {
                        decode_key::<S, T>(tail, TAIL, tail_record(key), dst);
                    }
                }
                for r in 0..n {
                    let span = &spans[r];
                    for j in sub..lens[r].min(sub + width) {
                        let (s0, s1) = qk2_decoded::<S>(keys.as_ptr().add((j - sub) * KEY_ROW), span.q0, span.q1);
                        logits[r][0][j] = s0 * scale;
                        block_max[r][0] = block_max[r][0].max(logits[r][0][j]);
                        logits[r][1][j] = s1 * scale;
                        block_max[r][1] = block_max[r][1].max(logits[r][1][j]);
                    }
                }
            }
            let mut new_max = [[0.0_f32; 2]; MAX_ROWS];
            for r in 0..n {
                if lens[r] == 0 {
                    continue;
                }
                let [o0, o1] = &mut parts[r].out;
                for (h, out) in [&mut *o0, &mut *o1].into_iter().enumerate() {
                    new_max[r][h] = max[r][h].max(block_max[r][h]);
                    let rescale = if max[r][h] == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (max[r][h] - new_max[r][h]).exp()
                    };
                    for value in out.iter_mut() {
                        *value *= rescale;
                    }
                    denominator[r][h] *= rescale;
                }
                let [l0, l1] = &mut logits[r];
                S::exp_shifted(&mut l0[..lens[r]], new_max[r][0]);
                S::exp_shifted(&mut l1[..lens[r]], new_max[r][1]);
            }
            for sub in (0..len).step_by(SUB) {
                let width = SUB.min(len - sub);
                for j in 0..width {
                    let key = start + sub + j;
                    let value = if key < prefix_len {
                        record_value::<S, R>(store, PREFIX, first_record + key)
                    } else {
                        record_value::<S, T>(tail, TAIL, tail_record(key))
                    };
                    for (c, v) in value.into_iter().enumerate() {
                        S::store(values.as_mut_ptr().add(j * 64 + 8 * c), v);
                    }
                }
                for r in 0..n {
                    let hi = lens[r].min(sub + width);
                    if hi <= sub {
                        continue;
                    }
                    for h in 0..2 {
                        pv_decoded::<S>(
                            values.as_ptr(),
                            &logits[r][h][sub..hi],
                            &mut denominator[r][h],
                            &mut parts[r].out[h],
                        );
                    }
                }
            }
            for r in 0..n {
                if lens[r] > 0 {
                    max[r] = new_max[r];
                }
            }
            start += len;
        }
    }
    for (r, part) in parts.iter_mut().enumerate() {
        part.max = max[r];
        part.denominator = denominator[r];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pair_rows_native<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    spans: &[Span<'_>],
    parts: &mut [Partial],
) {
    unsafe { pair_rows::<crate::simd::Avx2, R, T>(store, tail, spans, parts) }
}
/// [`pair_rows_native`] with the portable polynomial exp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pair_rows_native_fast<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    spans: &[Span<'_>],
    parts: &mut [Partial],
) {
    unsafe { pair_rows::<crate::simd::Avx2Fast, R, T>(store, tail, spans, parts) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn pair_rows_native<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    spans: &[Span<'_>],
    parts: &mut [Partial],
) {
    unsafe { pair_rows::<crate::simd::Neon, R, T>(store, tail, spans, parts) }
}
/// NEON's exp is already the polynomial one.
#[cfg(target_arch = "aarch64")]
unsafe fn pair_rows_native_fast<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    spans: &[Span<'_>],
    parts: &mut [Partial],
) {
    unsafe { pair_rows_native(store, tail, spans, parts) }
}
