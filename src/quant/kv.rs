//! Immutable split-prefix storage for the opt-in attempt. The reference prefill
//! finishes in FP32 arithmetic with the selected weights; sealing then
//! compresses its caches for subsequent decode.
//! This is DEFERRED cache compression, not a quantized-prefill-attention claim.
//!
//! Layout is group-major: for KV group `g` (query heads `2g`, `2g+1`) and prefix
//! position `t`, one 160-element record holds the temporal key half shared by
//! both heads, each head's spatial key half, and the value:
//! `[T 0..32 | S(2g) 32..64 | S(2g+1) 64..96 | V 96..160]`. One decode task per
//! group streams its records contiguously, loads T and V once for both heads,
//! and keeps `attention64::compact_head`'s operation order per head (global
//! 128-key tiles, rescale before PV, serial denominator, the same dot/AXPY FMA
//! trees, sink once at the end). With one position chunk the FP32 layout is
//! therefore bit-identical to the compact kernel, and Q16/Q8 records are
//! bit-identical to the compact kernel over their dequantized values. More
//! chunks split positions on tile boundaries and merge partial softmaxes, which
//! changes rounding only.
//!
//! Generated positions are stored in the same format as the prefix, in
//! group-major 128-element tail records `[K 0..64 | V 64..128]` (text keys are
//! identical within a GQA pair, so one key serves both heads). With FP32
//! storage the tail keeps the exact values; Q16/Q8 storage compresses each
//! generated position as it is appended, so long outputs do not stream an
//! FP32 tail that grows past the compressed prefix.
//!
//! Rotated storage (`Kv::Q8Rot`, `Kv::Q4Rot`) keeps every 32-element block of
//! a record, prefix and tail, as `H D x` (`super::rotation`: the temporal and
//! spatial key halves and the two value halves each with their own signs `D`)
//! and quantizes that. Decode rotates the queries instead of the records and
//! un-rotates each head's output once, so the kernels below run unchanged:
//! the output is bitwise the compact kernel over the dequantized rotated
//! values with the rotated query, followed by the inverse rotation, and it
//! differs from unrotated storage by rounding and quantization error only.
//! Q4 records (4-bit codes, two per byte) decode through the 8-bit loads.
use super::{
    Kv,
    rotation::{self, BLOCK, Block},
};
use crate::{
    config::ModelConfig,
    kernels::{self, Simd},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

mod codec;
mod decode;

use codec::{bf16_f32, bf16_up, each_block, nibble, q4_block, q8_code, q8_scale, q16_code, q16_scale, zeroed};
use decode::{F32Rec, PairKernel, Partial, Q4Rec, Q8Rec, Q16Rec, RowsKernel, Span, pair_entry};

/// Elements per (group, position) record and their offsets.
const RECORD: usize = 160;
const TEMPORAL: usize = 0;
const SPATIAL: [usize; 2] = [32, 64];
const VALUE: usize = 96;
/// Elements per (group, generated position) tail record: `[K 64 | V 64]`.
const TAIL_RECORD: usize = 128;
/// Q8 scales per record: one per 32-element chunk.
const Q8_SCALES: usize = RECORD / 32;
/// Online-softmax tile, identical to the compact decode kernel.
const TILE: usize = 128;
const MAX_GROUPS: usize = 8;
const MAX_CHUNKS: usize = 4;
/// Query rows of one verification step (the last token plus drafts).
pub(crate) const MAX_ROWS: usize = 8;
/// Rotation of the 32-element blocks of a prefix record, of a tail record
/// (both `[T | S.. | V_lo | V_hi]`), of a key head and of a value or output
/// head.
const RECORD_BLOCKS: [Block; RECORD / BLOCK] = [
    Block::Temporal,
    Block::Spatial,
    Block::Spatial,
    Block::ValueLow,
    Block::ValueHigh,
];
const TAIL_BLOCKS: [Block; TAIL_RECORD / BLOCK] = [Block::Temporal, Block::Spatial, Block::ValueLow, Block::ValueHigh];
const KEY_HALVES: [Block; 2] = [Block::Temporal, Block::Spatial];
const VALUE_HALVES: [Block; 2] = [Block::ValueLow, Block::ValueHigh];

#[derive(Debug)]
enum Records {
    F32(Vec<f32>),
    /// 8-bit codes, one BF16 scale per 32 elements (the codes are computed
    /// against the stored scale, so it is exact).
    Q8 {
        codes: Vec<i8>,
        scales: Vec<u16>,
    },
    /// 16-bit codes, one BF16 scale per 32 elements (prefix and tail).
    Q16 {
        codes: Vec<i16>,
        scales: Vec<u16>,
    },
    /// 4-bit codes in -7..=7, two per byte (element `2i` in the low nibble),
    /// one BF16 scale per 32 elements.
    Q4 {
        codes: Vec<u8>,
        scales: Vec<u16>,
    },
}

/// Element format of split records ([`Records`]' variants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    F32,
    Q16,
    Q8,
    Q4,
}

/// How a split cache stores its records: the element format, and whether
/// every 32-element block is rotated first (`rotation`). Profiles use the
/// storages [`Storage::of`] names; the tests also seal the others (rotated
/// FP32 records, unrotated Q4) to isolate what rotation changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Storage {
    format: Format,
    rotated: bool,
}

impl Storage {
    /// The storage of a sealed `mode` (`None`: the compact cache is never
    /// sealed).
    fn of(mode: Kv) -> Option<Self> {
        let (format, rotated) = match mode {
            Kv::Compact => return None,
            Kv::F32Split => (Format::F32, false),
            Kv::Q16 => (Format::Q16, false),
            Kv::Q8 => (Format::Q8, false),
            Kv::Q8Rot => (Format::Q8, true),
            Kv::Q4Rot => (Format::Q4, true),
        };
        Some(Self { format, rotated })
    }
}

pub(crate) struct SplitPrefix {
    records: Records,
    prefix_len: usize,
    heads: usize,
    kv_heads: usize,
    /// Position chunks per group; 1 keeps the compact kernel's exact rounding.
    chunks: usize,
    /// Generated positions, `[group][tail_capacity]` records of
    /// [`TAIL_RECORD`] elements in the prefix's storage format.
    tail: Records,
    tail_capacity: usize,
    tail_len: usize,
    /// x86: the softmax uses the portable polynomial exp (the NEON kernels'
    /// exp) instead of the platform-exact one (see [`default_fast_exp`]).
    fast_exp: bool,
    /// Records hold rotated blocks: decode rotates queries and un-rotates
    /// outputs ([`SplitPrefix::in_storage_basis`]).
    rotated: bool,
}

/// Default decode exp: the polynomial one for 8-bit caches (and the rotated
/// 8- and 4-bit research caches) when the run allows fast exps
/// (`ExpMode::Fast`, i.e. fast mode: token agreement with FP32 and the
/// English gate unchanged, verification attention 3-10% faster), the
/// platform-exact one otherwise; `requested` (`Tuning::decode_fast_exp`)
/// overrides it.
pub(crate) fn default_fast_exp(mode: Kv, fast_exps: bool, requested: Option<bool>) -> bool {
    requested.unwrap_or(fast_exps && matches!(mode, Kv::Q8 | Kv::Q8Rot | Kv::Q4Rot))
}

/// Default position chunks: exact for FP32 records, split (rounding-level)
/// for coded ones; `requested` (`Tuning::split_chunks`, 1..=4) overrides it.
fn default_chunks(format: Format, requested: Option<usize>) -> usize {
    // Four chunks (32 tasks) measured ~13% faster attention than two on a
    // 16-thread 7950X at a 6.5k prefix; FP32 keeps the exact single scan.
    requested
        .filter(|n| (1..=MAX_CHUNKS).contains(n))
        .unwrap_or(match format {
            Format::F32 => 1,
            _ => 4,
        })
}

impl Records {
    fn bytes(&self) -> usize {
        match self {
            Records::F32(d) => d.capacity() * 4,
            Records::Q8 { codes, scales } => codes.capacity() + scales.capacity() * 2,
            Records::Q16 { codes, scales } => codes.capacity() * 2 + scales.capacity() * 2,
            Records::Q4 { codes, scales } => codes.capacity() + scales.capacity() * 2,
        }
    }

    /// Element `offset` of `record` (`width` elements per record), decoded to
    /// the FP32 value decode uses.
    fn value(&self, width: usize, record: usize, offset: usize) -> f32 {
        let at = record * width + offset;
        match self {
            Records::F32(d) => d[at],
            Records::Q8 { codes, scales } => codes[at] as f32 * bf16_f32(scales[at / 32]),
            Records::Q16 { codes, scales } => codes[at] as f32 * bf16_f32(scales[at / 32]),
            Records::Q4 { codes, scales } => nibble(codes[at / 2], at % 2) as f32 * bf16_f32(scales[at / 32]),
        }
    }
}

impl SplitPrefix {
    /// Seals a compact prefix into `mode`'s record format; `chunks` overrides
    /// the position chunking of decode attention (`Tuning::split_chunks`).
    pub fn from_compact(
        k: &[f32],
        v: &[f32],
        prefix_len: usize,
        capacity: usize,
        c: &ModelConfig,
        mode: Kv,
        chunks: Option<usize>,
    ) -> Result<Self> {
        let storage = Storage::of(mode).context("the compact cache is never sealed")?;
        Self::seal(k, v, prefix_len, capacity, c, storage, chunks)
    }

    /// [`SplitPrefix::from_compact`] into any [`Storage`].
    fn seal(
        k: &[f32],
        v: &[f32],
        prefix_len: usize,
        capacity: usize,
        c: &ModelConfig,
        storage: Storage,
        chunks: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            c.head_dim == 64 && c.n_heads == 2 * c.n_kv_heads && c.n_kv_heads <= MAX_GROUPS,
            "split cache requires Falcon 64-wide paired GQA heads"
        );
        ensure!(
            prefix_len > 0 && capacity >= prefix_len,
            "invalid split-prefix capacity"
        );
        ensure!(
            prefix_len.checked_mul(c.query_dim()) == Some(k.len()),
            "split prefix key shape"
        );
        ensure!(
            prefix_len.checked_mul(c.kv_dim()) == Some(v.len()),
            "split prefix value shape"
        );
        let (heads, groups) = (c.n_heads, c.n_kv_heads);
        // Validate every pair before building anything.
        (0..prefix_len).into_par_iter().try_for_each(|t| {
            for g in 0..groups {
                let a = (t * heads + 2 * g) * 64;
                ensure!(
                    k[a..a + 32]
                        .iter()
                        .zip(&k[a + 64..a + 96])
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "temporal keys differ inside GQA pair; refusing illegal sharing"
                );
            }
            Ok(())
        })?;
        let gather = |record: usize, dst: &mut [f32; RECORD]| -> Result<()> {
            let (g, t) = (record / prefix_len, record % prefix_len);
            let a = (t * heads + 2 * g) * 64;
            let b = a + 64;
            let value = (t * groups + g) * 64;
            dst[TEMPORAL..TEMPORAL + 32].copy_from_slice(&k[a..a + 32]);
            dst[SPATIAL[0]..SPATIAL[0] + 32].copy_from_slice(&k[a + 32..a + 64]);
            dst[SPATIAL[1]..SPATIAL[1] + 32].copy_from_slice(&k[b + 32..b + 64]);
            dst[VALUE..VALUE + 64].copy_from_slice(&v[value..value + 64]);
            ensure!(dst.iter().all(|x| x.is_finite()), "nonfinite or incomplete KV group");
            // The pair check above ran on the unrotated keys.
            if storage.rotated {
                each_block(dst, &RECORD_BLOCKS, rotation::rotate);
                ensure!(dst.iter().all(|x| x.is_finite()), "rotated KV group overflows");
            }
            Ok(())
        };
        let count = groups * prefix_len;
        let records = match storage.format {
            Format::F32 => {
                let mut data = vec![0.0_f32; count * RECORD];
                data.par_chunks_mut(RECORD).enumerate().try_for_each(|(record, dst)| {
                    let mut values = [0.0; RECORD];
                    gather(record, &mut values)?;
                    dst.copy_from_slice(&values);
                    Ok::<_, anyhow::Error>(())
                })?;
                Records::F32(data)
            }
            Format::Q8 => {
                let mut codes = vec![0_i8; count * RECORD];
                let mut scales = vec![0_u16; count * Q8_SCALES];
                codes
                    .par_chunks_mut(RECORD)
                    .zip(scales.par_chunks_mut(Q8_SCALES))
                    .enumerate()
                    .try_for_each(|(record, (dst, record_scales))| {
                        let mut values = [0.0; RECORD];
                        gather(record, &mut values)?;
                        for ((codes, values), scale_out) in dst
                            .chunks_exact_mut(32)
                            .zip(values.chunks_exact(32))
                            .zip(record_scales.iter_mut())
                        {
                            let stored = bf16_up(q8_scale(values));
                            *scale_out = stored;
                            let scale = bf16_f32(stored);
                            for (code, &value) in codes.iter_mut().zip(values) {
                                *code = q8_code(value, scale);
                                ensure!((*code as f32 * scale).is_finite(), "Q8 KV conversion overflow");
                            }
                        }
                        Ok(())
                    })?;
                Records::Q8 { codes, scales }
            }
            Format::Q16 => {
                let mut codes = vec![0_i16; count * RECORD];
                let mut scales = vec![0_u16; count * Q8_SCALES];
                codes
                    .par_chunks_mut(RECORD)
                    .zip(scales.par_chunks_mut(Q8_SCALES))
                    .enumerate()
                    .try_for_each(|(record, (dst, record_scales))| {
                        let mut values = [0.0; RECORD];
                        gather(record, &mut values)?;
                        for ((codes, values), scale_out) in dst
                            .chunks_exact_mut(32)
                            .zip(values.chunks_exact(32))
                            .zip(record_scales.iter_mut())
                        {
                            let stored = bf16_up(q16_scale(values));
                            *scale_out = stored;
                            let scale = bf16_f32(stored);
                            for (code, &value) in codes.iter_mut().zip(values) {
                                *code = q16_code(value, scale);
                                ensure!((*code as f32 * scale).is_finite(), "Q16 KV conversion overflow");
                            }
                        }
                        Ok(())
                    })?;
                Records::Q16 { codes, scales }
            }
            Format::Q4 => {
                let mut codes = vec![0_u8; count * RECORD / 2];
                let mut scales = vec![0_u16; count * Q8_SCALES];
                codes
                    .par_chunks_mut(RECORD / 2)
                    .zip(scales.par_chunks_mut(Q8_SCALES))
                    .enumerate()
                    .try_for_each(|(record, (dst, record_scales))| {
                        let mut values = [0.0; RECORD];
                        gather(record, &mut values)?;
                        for ((codes, values), scale_out) in dst
                            .chunks_exact_mut(16)
                            .zip(values.chunks_exact(32))
                            .zip(record_scales.iter_mut())
                        {
                            *scale_out = q4_block(values, codes);
                            // Codes are at most 7 in magnitude.
                            ensure!((7.0 * bf16_f32(*scale_out)).is_finite(), "Q4 KV conversion overflow");
                        }
                        Ok(())
                    })?;
                Records::Q4 { codes, scales }
            }
        };
        let tail_capacity = capacity - prefix_len;
        let tail_elements = (groups * tail_capacity)
            .checked_mul(TAIL_RECORD)
            .ok_or_else(|| anyhow::anyhow!("tail capacity overflow"))?;
        let tail = match &records {
            Records::F32(_) => Records::F32(zeroed(tail_elements)?),
            Records::Q8 { .. } => Records::Q8 {
                codes: zeroed(tail_elements)?,
                scales: zeroed(tail_elements / 32)?,
            },
            Records::Q16 { .. } => Records::Q16 {
                codes: zeroed(tail_elements)?,
                scales: zeroed(tail_elements / 32)?,
            },
            Records::Q4 { .. } => Records::Q4 {
                codes: zeroed(tail_elements / 2)?,
                scales: zeroed(tail_elements / 32)?,
            },
        };
        Ok(Self {
            records,
            prefix_len,
            heads,
            kv_heads: groups,
            chunks: default_chunks(storage.format, chunks),
            tail,
            tail_capacity,
            tail_len: 0,
            fast_exp: false,
            rotated: storage.rotated,
        })
    }

    /// Use the portable polynomial exp in decode attention (x86; NEON always
    /// does). Single-row and verification steps switch together, so rows stay
    /// bitwise the single-row steps.
    pub(crate) fn with_fast_exp(mut self, fast: bool) -> Self {
        self.fast_exp = fast;
        self
    }

    /// Appends generated positions. `k` and `v` are expanded
    /// `[rows][heads][64]`, identical within each GQA pair (text positions).
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let width = self.heads * 64;
        assert!(k.len() == v.len() && k.len().is_multiple_of(width), "tail row shape");
        let mut key = [0.0_f32; MAX_GROUPS * 64];
        let mut value = [0.0_f32; MAX_GROUPS * 64];
        let unique = self.kv_heads * 64;
        for (k, v) in k.chunks_exact(width).zip(v.chunks_exact(width)) {
            for g in 0..self.kv_heads {
                let pair = 2 * g * 64;
                debug_assert!(
                    (0..64).all(|d| k[pair + d].to_bits() == k[pair + 64 + d].to_bits()
                        && v[pair + d].to_bits() == v[pair + 64 + d].to_bits()),
                    "split tail requires identical duplicated heads"
                );
                key[g * 64..(g + 1) * 64].copy_from_slice(&k[pair..pair + 64]);
                value[g * 64..(g + 1) * 64].copy_from_slice(&v[pair..pair + 64]);
            }
            self.push_unique(&key[..unique], &value[..unique]);
        }
    }

    /// Appends one generated position given one key and one value per group
    /// (`[groups][64]` each), encoded in the prefix's storage format (and
    /// rotated like it). A non-finite input keeps a non-finite decoded value
    /// (NaN for any Q16, Q8 or Q4 block that holds one; rotation first spreads
    /// it over its block), so it still reaches the logits.
    fn push_unique(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_heads * 64);
        assert_eq!(v.len(), k.len());
        assert!(self.tail_len < self.tail_capacity, "split tail capacity");
        for g in 0..self.kv_heads {
            let mut values = [0.0_f32; TAIL_RECORD];
            values[..64].copy_from_slice(&k[g * 64..(g + 1) * 64]);
            values[64..].copy_from_slice(&v[g * 64..(g + 1) * 64]);
            if self.rotated {
                each_block(&mut values, &TAIL_BLOCKS, rotation::rotate);
            }
            let record = g * self.tail_capacity + self.tail_len;
            let at = record * TAIL_RECORD;
            match &mut self.tail {
                Records::F32(d) => d[at..at + TAIL_RECORD].copy_from_slice(&values),
                Records::Q16 { codes, scales } => {
                    let blocks = TAIL_RECORD / 32;
                    for (block, values) in values.chunks_exact(32).enumerate() {
                        let stored = bf16_up(if values.iter().all(|x| x.is_finite()) {
                            q16_scale(values)
                        } else {
                            f32::NAN
                        });
                        scales[record * blocks + block] = stored;
                        let scale = bf16_f32(stored);
                        let codes = &mut codes[at + 32 * block..at + 32 * (block + 1)];
                        for (code, &value) in codes.iter_mut().zip(values) {
                            *code = q16_code(value, scale);
                        }
                    }
                }
                Records::Q8 { codes, scales } => {
                    let blocks = TAIL_RECORD / 32;
                    for (block, values) in values.chunks_exact(32).enumerate() {
                        let stored = bf16_up(if values.iter().all(|x| x.is_finite()) {
                            q8_scale(values)
                        } else {
                            f32::NAN
                        });
                        scales[record * blocks + block] = stored;
                        let scale = bf16_f32(stored);
                        let codes = &mut codes[at + 32 * block..at + 32 * (block + 1)];
                        for (code, &value) in codes.iter_mut().zip(values) {
                            *code = q8_code(value, scale);
                        }
                    }
                }
                Records::Q4 { codes, scales } => {
                    let blocks = TAIL_RECORD / 32;
                    for (block, values) in values.chunks_exact(32).enumerate() {
                        let codes = &mut codes[at / 2 + 16 * block..at / 2 + 16 * (block + 1)];
                        scales[record * blocks + block] = q4_block(values, codes);
                    }
                }
            }
        }
        self.tail_len += 1;
    }

    #[cfg(test)]
    fn with_chunks(mut self, chunks: usize) -> Self {
        assert!((1..=MAX_CHUNKS).contains(&chunks));
        self.chunks = chunks;
        self
    }

    pub fn bytes(&self) -> usize {
        self.records.bytes() + self.tail.bytes()
    }

    /// Record element `offset` of `record`, as the FP32 value decode uses.
    fn value(&self, record: usize, offset: usize) -> f32 {
        self.records.value(RECORD, record, offset)
    }

    /// Element `offset` of the tail record of `group` at generated position
    /// `index`.
    fn tail_value(&self, group: usize, index: usize, offset: usize) -> f32 {
        self.tail.value(TAIL_RECORD, group * self.tail_capacity + index, offset)
    }

    /// The compact `[t][heads][64]` keys and `[t][groups][64]` values that the
    /// stored records decode to (the exactness oracle for lossy storage). For
    /// rotated storage they are in the rotated basis, which decode pairs with
    /// rotated queries.
    #[cfg(test)]
    fn dequantized_compact(&self) -> (Vec<f32>, Vec<f32>) {
        let (p, groups) = (self.prefix_len, self.kv_heads);
        let mut k = vec![0.0; p * self.heads * 64];
        let mut v = vec![0.0; p * groups * 64];
        for g in 0..groups {
            for t in 0..p {
                let record = g * p + t;
                for (pair, spatial) in SPATIAL.iter().enumerate() {
                    let base = (t * self.heads + 2 * g + pair) * 64;
                    for d in 0..32 {
                        k[base + d] = self.value(record, TEMPORAL + d);
                        k[base + 32 + d] = self.value(record, spatial + d);
                    }
                }
                for d in 0..64 {
                    v[(t * groups + g) * 64 + d] = self.value(record, VALUE + d);
                }
            }
        }
        (k, v)
    }

    pub fn attention_decode(&self, q: &[f32], total_len: usize, sinks: &[f32], output: &mut [f32], simd: Simd) {
        assert_eq!(q.len(), self.heads * 64);
        assert_eq!(output.len(), q.len());
        assert_eq!(sinks.len(), self.heads);
        assert_eq!(total_len, self.prefix_len + self.tail_len);
        self.in_storage_basis::<{ MAX_GROUPS * 128 }>(q, output, |q, output| {
            self.decode_stored(q, total_len, sinks, output, simd)
        });
    }

    /// Runs `decode` on the queries in the basis the records are stored in
    /// and leaves its output in the model's: directly for unrotated storage;
    /// for rotated storage every head's query halves are rotated first
    /// (`rotation::rotate_query`, so their dots with the stored keys are the
    /// original ones) and its output halves un-rotated after
    /// (`rotation::unrotate`). The rotated queries live in an `N`-element
    /// stack buffer, so decode steps still do not allocate.
    fn in_storage_basis<const N: usize>(&self, q: &[f32], output: &mut [f32], decode: impl FnOnce(&[f32], &mut [f32])) {
        if !self.rotated {
            return decode(q, output);
        }
        let mut buffer = [0.0_f32; N];
        let rotated = &mut buffer[..q.len()];
        rotated.copy_from_slice(q);
        each_block(rotated, &KEY_HALVES, rotation::rotate_query);
        decode(rotated, output);
        each_block(output, &VALUE_HALVES, rotation::unrotate);
    }

    /// [`SplitPrefix::attention_decode`] in the records' basis.
    fn decode_stored(&self, q: &[f32], total_len: usize, sinks: &[f32], output: &mut [f32], simd: Simd) {
        let selected = simd.resolved();
        #[cfg(target_arch = "x86_64")]
        if selected != Simd::Scalar
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            self.attention_pairs(q, total_len, sinks, output);
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if selected == Simd::Neon {
            self.attention_pairs(q, total_len, sinks, output);
            return;
        }
        self.attention_generic(q, total_len, sinks, output, selected);
    }

    /// Portable path: one task per query head through the backend's dot/AXPY,
    /// in the compact kernel's order (bit-identical to it for FP32 records).
    fn attention_generic(&self, q: &[f32], total_len: usize, sinks: &[f32], output: &mut [f32], selected: Simd) {
        let dot = kernels::dot_kernel(selected);
        let axpy = kernels::axpy_kernel(selected);
        output.par_chunks_mut(64).enumerate().for_each(|(h, out)| {
            let group = h / 2;
            let query = &q[h * 64..(h + 1) * 64];
            let mut max = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            let mut logits = [0.0_f32; TILE];
            let mut key = [0.0_f32; 64];
            let mut value = [0.0_f32; 64];
            out.fill(0.0);
            for start in (0..total_len).step_by(TILE) {
                let len = (total_len - start).min(TILE);
                let mut block_max = f32::NEG_INFINITY;
                for (j, logit) in logits[..len].iter_mut().enumerate() {
                    let t = start + j;
                    if t < self.prefix_len {
                        let record = group * self.prefix_len + t;
                        for d in 0..32 {
                            key[d] = self.value(record, TEMPORAL + d);
                            key[32 + d] = self.value(record, SPATIAL[h % 2] + d);
                        }
                    } else {
                        for (d, x) in key.iter_mut().enumerate() {
                            *x = self.tail_value(group, t - self.prefix_len, d);
                        }
                    }
                    *logit = dot(query, &key) * 0.125;
                    block_max = block_max.max(*logit);
                }
                let new_max = max.max(block_max);
                let scale = if max == f32::NEG_INFINITY {
                    0.0
                } else {
                    (max - new_max).exp()
                };
                for x in out.iter_mut() {
                    *x *= scale;
                }
                denominator *= scale;
                for (j, logit) in logits[..len].iter().enumerate() {
                    let probability = (*logit - new_max).exp();
                    denominator += probability;
                    let t = start + j;
                    if t < self.prefix_len {
                        let record = group * self.prefix_len + t;
                        for (d, x) in value.iter_mut().enumerate() {
                            *x = self.value(record, VALUE + d);
                        }
                    } else {
                        for (d, x) in value.iter_mut().enumerate() {
                            *x = self.tail_value(group, t - self.prefix_len, 64 + d);
                        }
                    }
                    axpy(probability, &value, out);
                }
                max = new_max;
            }
            // Same tile order and sink policy as the reference. The sink occurs
            // once after the combined image+tail scan, not once per segment.
            let lse = max + denominator.ln();
            let sink = 1.0 / (1.0 + (sinks[h] - lse).exp());
            for y in out {
                *y = (*y / denominator) * sink;
            }
        });
    }

    /// Vector path (AVX2 or NEON): one task per (KV group, position chunk)
    /// serving both heads.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn attention_pairs(&self, q: &[f32], total_len: usize, sinks: &[f32], output: &mut [f32]) {
        let groups = self.kv_heads;
        let tiles = total_len.div_ceil(TILE);
        let chunks = self.chunks.clamp(1, MAX_CHUNKS).min(tiles);
        let tiles_per_chunk = tiles.div_ceil(chunks);
        let mut parts = [Partial::EMPTY; MAX_GROUPS * MAX_CHUNKS];
        let shared = crate::team::SharedMut::new(&mut parts[..groups * chunks]);
        crate::team::for_each(groups * chunks, |index| {
            // SAFETY: each task owns one disjoint element of `parts`.
            let part = unsafe { &mut shared.slice(index, 1)[0] };
            let (g, chunk) = (index / chunks, index % chunks);
            let start = (chunk * tiles_per_chunk * TILE).min(total_len);
            let end = ((chunk + 1) * tiles_per_chunk * TILE).min(total_len);
            let span = Span {
                prefix_len: self.prefix_len,
                group: g,
                tail_base: g * self.tail_capacity,
                start,
                end,
                q0: &q[2 * g * 64..(2 * g + 1) * 64],
                q1: &q[(2 * g + 1) * 64..(2 * g + 2) * 64],
            };
            // SAFETY: the native vector ISA was checked by the caller; `Span`
            // and the record stores were shape-checked at construction and entry.
            unsafe {
                match (&self.records, &self.tail) {
                    (Records::F32(d), Records::F32(t)) => pair_entry(
                        &F32Rec::<RECORD>(d),
                        &F32Rec::<TAIL_RECORD>(t),
                        &span,
                        part,
                        self.fast_exp,
                    ),
                    (
                        Records::Q8 { codes, scales },
                        Records::Q8 {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                    ) => pair_entry(
                        &Q8Rec::<RECORD> { codes, scales },
                        &Q8Rec::<TAIL_RECORD> {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                        &span,
                        part,
                        self.fast_exp,
                    ),
                    (
                        Records::Q16 { codes, scales },
                        Records::Q16 {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                    ) => pair_entry(
                        &Q16Rec::<RECORD> { codes, scales },
                        &Q16Rec::<TAIL_RECORD> {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                        &span,
                        part,
                        self.fast_exp,
                    ),
                    (
                        Records::Q4 { codes, scales },
                        Records::Q4 {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                    ) => pair_entry(
                        &Q4Rec::<RECORD> { codes, scales },
                        &Q4Rec::<TAIL_RECORD> {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                        &span,
                        part,
                        self.fast_exp,
                    ),
                    _ => unreachable!("tail storage matches the prefix"),
                }
            }
        });
        self.merge(&parts[..groups * chunks], chunks, sinks, output);
    }

    /// Final softmax normalization of every head from its group's position
    /// chunks (`parts[g * chunks + chunk]`), then the sink.
    fn merge(&self, parts: &[Partial], chunks: usize, sinks: &[f32], output: &mut [f32]) {
        let groups = self.kv_heads;
        for g in 0..groups {
            let group_parts = &parts[g * chunks..(g + 1) * chunks];
            for pair in 0..2 {
                let h = 2 * g + pair;
                let out = &mut output[h * 64..(h + 1) * 64];
                let (max, denominator) = if chunks == 1 {
                    // Exactly the single-scan state; no merge arithmetic.
                    let part = &group_parts[0];
                    out.copy_from_slice(&part.out[pair]);
                    (part.max[pair], part.denominator[pair])
                } else {
                    let max = group_parts
                        .iter()
                        .map(|p| p.max[pair])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut denominator = 0.0_f32;
                    out.fill(0.0);
                    for part in group_parts {
                        if part.max[pair] == f32::NEG_INFINITY {
                            continue;
                        }
                        let factor = (part.max[pair] - max).exp();
                        denominator = part.denominator[pair].mul_add(factor, denominator);
                        for (y, x) in out.iter_mut().zip(&part.out[pair]) {
                            *y = x.mul_add(factor, *y);
                        }
                    }
                    (max, denominator)
                };
                let lse = max + denominator.ln();
                let sink = 1.0 / (1.0 + (sinks[h] - lse).exp());
                for y in out {
                    *y = (*y / denominator) * sink;
                }
            }
        }
    }

    /// Record stores of this cache, passed to `kernel` with their concrete types.
    ///
    /// # Safety
    /// The kernel's own requirements (native vector ISA, span bounds).
    unsafe fn dispatch(&self, kernel: &mut impl PairKernel) {
        unsafe {
            match (&self.records, &self.tail) {
                (Records::F32(d), Records::F32(t)) => kernel.run(&F32Rec::<RECORD>(d), &F32Rec::<TAIL_RECORD>(t)),
                (Records::Q8 { codes, scales }, Records::Q8 { codes: tc, scales: ts }) => kernel.run(
                    &Q8Rec::<RECORD> { codes, scales },
                    &Q8Rec::<TAIL_RECORD> { codes: tc, scales: ts },
                ),
                (Records::Q16 { codes, scales }, Records::Q16 { codes: tc, scales: ts }) => kernel.run(
                    &Q16Rec::<RECORD> { codes, scales },
                    &Q16Rec::<TAIL_RECORD> { codes: tc, scales: ts },
                ),
                (Records::Q4 { codes, scales }, Records::Q4 { codes: tc, scales: ts }) => kernel.run(
                    &Q4Rec::<RECORD> { codes, scales },
                    &Q4Rec::<TAIL_RECORD> { codes: tc, scales: ts },
                ),
                _ => unreachable!("tail storage matches the prefix"),
            }
        }
    }

    /// Generated positions currently stored.
    pub fn tail_len(&self) -> usize {
        self.tail_len
    }

    /// Forget generated positions past `tail_len` (rejected draft tokens).
    pub fn truncate_tail(&mut self, tail_len: usize) {
        assert!(tail_len <= self.tail_len, "cannot grow the tail by truncation");
        self.tail_len = tail_len;
    }

    /// [`SplitPrefix::attention_decode`] with the last `rows` cached
    /// positions as queries: row `r` sees the first `total - rows + 1 + r`
    /// positions (causal verification of drafted tokens in one step). Every
    /// row's output is bitwise its single-row result at that length; rows
    /// that share a position partition scan each record once.
    pub fn attention_decode_rows(&self, q: &[f32], rows: usize, sinks: &[f32], output: &mut [f32], simd: Simd) {
        let width = self.heads * 64;
        assert!((1..=MAX_ROWS).contains(&rows) && rows <= self.tail_len.max(1));
        assert_eq!(q.len(), rows * width);
        assert_eq!(output.len(), q.len());
        assert_eq!(sinks.len(), self.heads);
        self.in_storage_basis::<{ MAX_ROWS * MAX_GROUPS * 128 }>(q, output, |q, output| {
            self.rows_stored(q, rows, sinks, output, simd)
        });
    }

    /// [`SplitPrefix::attention_decode_rows`] in the records' basis.
    fn rows_stored(&self, q: &[f32], rows: usize, sinks: &[f32], output: &mut [f32], simd: Simd) {
        let width = self.heads * 64;
        let last = self.prefix_len + self.tail_len;
        if rows == 1 {
            self.decode_stored(q, last, sinks, output, simd);
            return;
        }
        let selected = simd.resolved();
        #[cfg(target_arch = "x86_64")]
        if selected != Simd::Scalar
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            self.attention_pairs_rows(q, rows, last, sinks, output);
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if selected == Simd::Neon {
            self.attention_pairs_rows(q, rows, last, sinks, output);
            return;
        }
        for r in 0..rows {
            self.attention_generic(
                &q[r * width..(r + 1) * width],
                last + 1 + r - rows,
                sinks,
                &mut output[r * width..(r + 1) * width],
                selected,
            );
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[allow(clippy::needless_range_loop)] // rows are grouped by index into fixed-size arrays
    fn attention_pairs_rows(&self, q: &[f32], rows: usize, last: usize, sinks: &[f32], output: &mut [f32]) {
        let width = self.heads * 64;
        let groups = self.kv_heads;
        // The position partition `attention_pairs` uses at each length.
        let partition = |total: usize| {
            let tiles = total.div_ceil(TILE);
            let chunks = self.chunks.clamp(1, MAX_CHUNKS).min(tiles);
            (chunks, tiles.div_ceil(chunks))
        };
        let total = |r: usize| last + 1 + r - rows;
        let mut assigned = [false; MAX_ROWS];
        for first in 0..rows {
            if assigned[first] {
                continue;
            }
            let key = partition(total(first));
            let (chunks, per_chunk) = key;
            let mut members = [0_usize; MAX_ROWS];
            let mut n = 0;
            for r in first..rows {
                if !assigned[r] && partition(total(r)) == key {
                    assigned[r] = true;
                    members[n] = r;
                    n += 1;
                }
            }
            let mut parts = vec![Partial::EMPTY; n * groups * chunks];
            let shared = crate::team::SharedMut::new(&mut parts);
            crate::team::for_each(groups * chunks, |index| {
                let (g, chunk) = (index / chunks, index % chunks);
                let spans: [Span<'_>; MAX_ROWS] = std::array::from_fn(|i| {
                    let r = members[i.min(n - 1)];
                    let t = total(r);
                    let base = r * width;
                    Span {
                        prefix_len: self.prefix_len,
                        group: g,
                        tail_base: g * self.tail_capacity,
                        start: (chunk * per_chunk * TILE).min(t),
                        end: ((chunk + 1) * per_chunk * TILE).min(t),
                        q0: &q[base + 2 * g * 64..base + (2 * g + 1) * 64],
                        q1: &q[base + (2 * g + 1) * 64..base + (2 * g + 2) * 64],
                    }
                });
                let mut local = [Partial::EMPTY; MAX_ROWS];
                // SAFETY: the native vector ISA was checked by the caller; spans
                // lie inside the shape-checked stores.
                unsafe {
                    self.dispatch(&mut RowsKernel {
                        spans: &spans[..n],
                        parts: &mut local[..n],
                        fast_exp: self.fast_exp,
                    })
                };
                for (i, part) in local[..n].iter().enumerate() {
                    // SAFETY: element (i, g, chunk) belongs to this task alone.
                    unsafe { shared.slice((i * groups + g) * chunks + chunk, 1)[0] = *part };
                }
            });
            for (i, &r) in members[..n].iter().enumerate() {
                self.merge(
                    &parts[i * groups * chunks..(i + 1) * groups * chunks],
                    chunks,
                    sinks,
                    &mut output[r * width..(r + 1) * width],
                );
            }
        }
    }
}

#[cfg(test)]
mod probe;
#[cfg(test)]
mod record_bits;
#[cfg(test)]
mod tests;
