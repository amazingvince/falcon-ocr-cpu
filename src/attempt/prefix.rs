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
//! therefore bit-identical to the compact kernel, and BF16/Q8 records are
//! bit-identical to the compact kernel over their dequantized values. More
//! chunks split positions on tile boundaries and merge partial softmaxes, which
//! changes rounding only.
//!
//! Generated positions are stored in the same format as the prefix, in
//! group-major 128-element tail records `[K 0..64 | V 64..128]` (text keys are
//! identical within a GQA pair, so one key serves both heads). With FP32
//! storage the tail keeps the exact values; BF16/Q8 storage compresses each
//! generated position as it is appended, so long outputs do not stream an
//! FP32 tail that grows past the compressed prefix.
use super::PrefixMode;
use crate::{
    config::ModelConfig,
    kernels::{self, Simd},
};
use anyhow::{Result, ensure};
use half::bf16;
use rayon::prelude::*;

/// Elements per (group, position) record and their offsets.
const RECORD: usize = 160;
const TEMPORAL: usize = 0;
const SPATIAL: [usize; 2] = [32, 64];
const VALUE: usize = 96;
/// Elements per (group, generated position) tail record: `[K 64 | V 64]`.
const TAIL_RECORD: usize = 128;
/// Records per key-scale tile of `SplitQ8Kc` (a power of two, so the tile of
/// a record is a shift).
const KC_TILE: usize = 128;
/// Q8 scales per record: one per 32-element chunk.
const Q8_SCALES: usize = RECORD / 32;
/// Online-softmax tile, identical to the compact decode kernel.
const TILE: usize = 128;
const MAX_GROUPS: usize = 8;
const MAX_CHUNKS: usize = 4;

#[derive(Debug)]
enum Records {
    F32(Vec<f32>),
    /// BF16 bit patterns.
    Bf16(Vec<u16>),
    Q8 {
        codes: Vec<i8>,
        scales: Vec<f32>,
    },
    /// 16-bit codes, one scale per 32 elements (prefix and tail).
    Q16 {
        codes: Vec<i16>,
        scales: Vec<f32>,
    },
    /// Prefix only: key channel `c` of records `[t * KC_TILE, (t + 1) *
    /// KC_TILE)` shares `k_scales[t * VALUE + c]`; each record's two value
    /// halves have `v_scales[2 * record + h]`.
    Q8Kc {
        codes: Vec<i8>,
        k_scales: Vec<f32>,
        v_scales: Vec<f32>,
    },
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
}

/// Default position chunks: exact for FP32, split (rounding-level) for lossy storage.
/// `FALCON_OCR_SPLIT_CHUNKS=1|2|4` overrides it once, at sealing, for experiments.
fn default_chunks(mode: PrefixMode) -> usize {
    let requested = std::env::var("FALCON_OCR_SPLIT_CHUNKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| [1, 2, 4].contains(n));
    // Four chunks (32 tasks) measured ~13% faster attention than two on a
    // 16-thread 7950X at a 6.5k prefix; FP32 keeps the exact single scan.
    requested.unwrap_or(match mode {
        PrefixMode::SplitF32 => 1,
        _ => 4,
    })
}

impl Records {
    fn bytes(&self) -> usize {
        match self {
            Records::F32(d) => d.capacity() * 4,
            Records::Bf16(d) => d.capacity() * 2,
            Records::Q8 { codes, scales } => codes.capacity() + scales.capacity() * 4,
            Records::Q16 { codes, scales } => codes.capacity() * 2 + scales.capacity() * 4,
            Records::Q8Kc {
                codes,
                k_scales,
                v_scales,
            } => codes.capacity() + (k_scales.capacity() + v_scales.capacity()) * 4,
        }
    }

    /// Element `offset` of `record` (`width` elements per record), decoded to
    /// the FP32 value decode uses.
    fn value(&self, width: usize, record: usize, offset: usize) -> f32 {
        let at = record * width + offset;
        match self {
            Records::F32(d) => d[at],
            Records::Bf16(d) => bf16::from_bits(d[at]).to_f32(),
            Records::Q8 { codes, scales } => codes[at] as f32 * scales[at / 32],
            Records::Q16 { codes, scales } => codes[at] as f32 * scales[at / 32],
            Records::Q8Kc {
                codes,
                k_scales,
                v_scales,
            } => {
                debug_assert_eq!(width, RECORD);
                let scale = if offset < VALUE {
                    k_scales[record / KC_TILE * VALUE + offset]
                } else {
                    v_scales[2 * record + (offset - VALUE) / 32]
                };
                codes[at] as f32 * scale
            }
        }
    }
}

/// A zero-filled buffer, failing instead of aborting when memory is short.
fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)?;
    data.resize(len, T::default());
    Ok(data)
}

fn q8_scale(values: &[f32]) -> f32 {
    q8_scale_of_max(values.iter().fold(0.0_f32, |a, &b| a.max(b.abs())))
}

fn q8_scale_of_max(max: f32) -> f32 {
    if max == 0.0 {
        0.0
    } else {
        ((max as f64 / 127.0) as f32).max(f32::from_bits(1))
    }
}

fn q16_scale(values: &[f32]) -> f32 {
    let max = values.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    if max == 0.0 {
        0.0
    } else {
        ((max as f64 / 32767.0) as f32).max(f32::from_bits(1))
    }
}

fn q16_code(value: f32, scale: f32) -> i16 {
    if scale == 0.0 {
        0
    } else {
        (value as f64 / scale as f64)
            .round_ties_even()
            .clamp(-32767.0, 32767.0) as i16
    }
}

fn q8_code(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        0
    } else {
        (value as f64 / scale as f64)
            .round_ties_even()
            .clamp(-127.0, 127.0) as i8
    }
}

impl SplitPrefix {
    pub fn from_compact(
        k: &[f32],
        v: &[f32],
        prefix_len: usize,
        capacity: usize,
        c: &ModelConfig,
        mode: PrefixMode,
    ) -> Result<Self> {
        ensure!(
            mode != PrefixMode::Reference
                && c.head_dim == 64
                && c.n_heads == 2 * c.n_kv_heads
                && c.n_kv_heads <= MAX_GROUPS,
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
            ensure!(
                dst.iter().all(|x| x.is_finite()),
                "nonfinite or incomplete KV group"
            );
            Ok(())
        };
        let count = groups * prefix_len;
        let records = match mode {
            PrefixMode::SplitF32 => {
                let mut data = vec![0.0_f32; count * RECORD];
                data.par_chunks_mut(RECORD)
                    .enumerate()
                    .try_for_each(|(record, dst)| {
                        let mut values = [0.0; RECORD];
                        gather(record, &mut values)?;
                        dst.copy_from_slice(&values);
                        Ok::<_, anyhow::Error>(())
                    })?;
                Records::F32(data)
            }
            PrefixMode::SplitBf16 => {
                let mut data = vec![0_u16; count * RECORD];
                data.par_chunks_mut(RECORD)
                    .enumerate()
                    .try_for_each(|(record, dst)| {
                        let mut values = [0.0; RECORD];
                        gather(record, &mut values)?;
                        for (bits, &value) in dst.iter_mut().zip(&values) {
                            let encoded = bf16::from_f32(value);
                            ensure!(encoded.is_finite(), "BF16 KV conversion overflow");
                            *bits = encoded.to_bits();
                        }
                        Ok(())
                    })?;
                Records::Bf16(data)
            }
            PrefixMode::SplitQ8 => {
                let mut codes = vec![0_i8; count * RECORD];
                let mut scales = vec![0.0_f32; count * Q8_SCALES];
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
                            let scale = q8_scale(values);
                            *scale_out = scale;
                            for (code, &value) in codes.iter_mut().zip(values) {
                                *code = q8_code(value, scale);
                                ensure!(
                                    (*code as f32 * scale).is_finite(),
                                    "Q8 KV conversion overflow"
                                );
                            }
                        }
                        Ok(())
                    })?;
                Records::Q8 { codes, scales }
            }
            PrefixMode::SplitQ16 => {
                let mut codes = vec![0_i16; count * RECORD];
                let mut scales = vec![0.0_f32; count * Q8_SCALES];
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
                            let scale = q16_scale(values);
                            *scale_out = scale;
                            for (code, &value) in codes.iter_mut().zip(values) {
                                *code = q16_code(value, scale);
                                ensure!(
                                    (*code as f32 * scale).is_finite(),
                                    "Q16 KV conversion overflow"
                                );
                            }
                        }
                        Ok(())
                    })?;
                Records::Q16 { codes, scales }
            }
            PrefixMode::SplitQ8Kc => {
                let tiles = count.div_ceil(KC_TILE);
                let mut codes = vec![0_i8; count * RECORD];
                let mut k_scales = vec![0.0_f32; tiles * VALUE];
                let mut v_scales = vec![0.0_f32; count * 2];
                codes
                    .par_chunks_mut(KC_TILE * RECORD)
                    .zip(k_scales.par_chunks_mut(VALUE))
                    .zip(v_scales.par_chunks_mut(KC_TILE * 2))
                    .enumerate()
                    .try_for_each(|(tile, ((codes, k_scales), v_scales))| {
                        let first = tile * KC_TILE;
                        let n = codes.len() / RECORD;
                        let mut values = vec![[0.0_f32; RECORD]; n];
                        for (r, dst) in values.iter_mut().enumerate() {
                            gather(first + r, dst)?;
                        }
                        for (c, scale) in k_scales.iter_mut().enumerate() {
                            *scale = q8_scale_of_max(
                                values.iter().fold(0.0_f32, |m, v| m.max(v[c].abs())),
                            );
                        }
                        for (r, row) in values.iter().enumerate() {
                            let out = &mut codes[r * RECORD..(r + 1) * RECORD];
                            for c in 0..VALUE {
                                out[c] = q8_code(row[c], k_scales[c]);
                                ensure!(
                                    (out[c] as f32 * k_scales[c]).is_finite(),
                                    "Q8 KV conversion overflow"
                                );
                            }
                            for h in 0..2 {
                                let block = &row[VALUE + 32 * h..VALUE + 32 * (h + 1)];
                                let scale = q8_scale(block);
                                v_scales[2 * r + h] = scale;
                                for (i, &x) in block.iter().enumerate() {
                                    let code = q8_code(x, scale);
                                    ensure!(
                                        (code as f32 * scale).is_finite(),
                                        "Q8 KV conversion overflow"
                                    );
                                    out[VALUE + 32 * h + i] = code;
                                }
                            }
                        }
                        Ok::<_, anyhow::Error>(())
                    })?;
                Records::Q8Kc {
                    codes,
                    k_scales,
                    v_scales,
                }
            }
            PrefixMode::Reference => unreachable!("checked above"),
        };
        let tail_capacity = capacity - prefix_len;
        let tail_elements = (groups * tail_capacity)
            .checked_mul(TAIL_RECORD)
            .ok_or_else(|| anyhow::anyhow!("tail capacity overflow"))?;
        let tail = match &records {
            Records::F32(_) => Records::F32(zeroed(tail_elements)?),
            Records::Bf16(_) => Records::Bf16(zeroed(tail_elements)?),
            // Generated keys arrive one position at a time, so the tail keeps
            // per-record scales in both Q8 layouts.
            Records::Q8 { .. } | Records::Q8Kc { .. } => Records::Q8 {
                codes: zeroed(tail_elements)?,
                scales: zeroed(tail_elements / 32)?,
            },
            Records::Q16 { .. } => Records::Q16 {
                codes: zeroed(tail_elements)?,
                scales: zeroed(tail_elements / 32)?,
            },
        };
        Ok(Self {
            records,
            prefix_len,
            heads,
            kv_heads: groups,
            chunks: default_chunks(mode),
            tail,
            tail_capacity,
            tail_len: 0,
        })
    }

    /// Appends generated positions. `k` and `v` are expanded
    /// `[rows][heads][64]`, identical within each GQA pair (text positions).
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let width = self.heads * 64;
        assert!(k.len() == v.len() && k.len() % width == 0, "tail row shape");
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
    /// (`[groups][64]` each), encoded in the prefix's storage format. A
    /// non-finite input keeps a non-finite decoded value (NaN for BF16 NaN and
    /// for any Q8 block that holds one), so it still reaches the logits.
    fn push_unique(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_heads * 64);
        assert_eq!(v.len(), k.len());
        assert!(self.tail_len < self.tail_capacity, "split tail capacity");
        for g in 0..self.kv_heads {
            let mut values = [0.0_f32; TAIL_RECORD];
            values[..64].copy_from_slice(&k[g * 64..(g + 1) * 64]);
            values[64..].copy_from_slice(&v[g * 64..(g + 1) * 64]);
            let record = g * self.tail_capacity + self.tail_len;
            let at = record * TAIL_RECORD;
            match &mut self.tail {
                Records::F32(d) => d[at..at + TAIL_RECORD].copy_from_slice(&values),
                Records::Bf16(d) => {
                    for (bits, &value) in d[at..at + TAIL_RECORD].iter_mut().zip(&values) {
                        *bits = bf16::from_f32(value).to_bits();
                    }
                }
                Records::Q8Kc { .. } => unreachable!("the tail never uses per-channel scales"),
                Records::Q16 { codes, scales } => {
                    let blocks = TAIL_RECORD / 32;
                    for (block, values) in values.chunks_exact(32).enumerate() {
                        let scale = if values.iter().all(|x| x.is_finite()) {
                            q16_scale(values)
                        } else {
                            f32::NAN
                        };
                        scales[record * blocks + block] = scale;
                        let codes = &mut codes[at + 32 * block..at + 32 * (block + 1)];
                        for (code, &value) in codes.iter_mut().zip(values) {
                            *code = q16_code(value, scale);
                        }
                    }
                }
                Records::Q8 { codes, scales } => {
                    let blocks = TAIL_RECORD / 32;
                    for (block, values) in values.chunks_exact(32).enumerate() {
                        let scale = if values.iter().all(|x| x.is_finite()) {
                            q8_scale(values)
                        } else {
                            f32::NAN
                        };
                        scales[record * blocks + block] = scale;
                        let codes = &mut codes[at + 32 * block..at + 32 * (block + 1)];
                        for (code, &value) in codes.iter_mut().zip(values) {
                            *code = q8_code(value, scale);
                        }
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
        self.tail
            .value(TAIL_RECORD, group * self.tail_capacity + index, offset)
    }

    /// The compact `[t][heads][64]` keys and `[t][groups][64]` values that the
    /// stored records decode to (the exactness oracle for lossy storage).
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

    pub fn attention_decode(
        &self,
        q: &[f32],
        total_len: usize,
        sinks: &[f32],
        output: &mut [f32],
        simd: Simd,
    ) {
        assert_eq!(q.len(), self.heads * 64);
        assert_eq!(output.len(), q.len());
        assert_eq!(sinks.len(), self.heads);
        assert_eq!(total_len, self.prefix_len + self.tail_len);
        let selected = simd.resolved();
        #[cfg(target_arch = "x86_64")]
        if selected != Simd::Scalar
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
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
    fn attention_generic(
        &self,
        q: &[f32],
        total_len: usize,
        sinks: &[f32],
        output: &mut [f32],
        selected: Simd,
    ) {
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
                    (Records::F32(d), Records::F32(t)) => {
                        pair_native(&F32Rec::<RECORD>(d), &F32Rec::<TAIL_RECORD>(t), &span, part)
                    }
                    (Records::Bf16(d), Records::Bf16(t)) => pair_native(
                        &Bf16Rec::<RECORD>(d),
                        &Bf16Rec::<TAIL_RECORD>(t),
                        &span,
                        part,
                    ),
                    (
                        Records::Q8 { codes, scales },
                        Records::Q8 {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                    ) => pair_native(
                        &Q8Rec::<RECORD> { codes, scales },
                        &Q8Rec::<TAIL_RECORD> {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                        &span,
                        part,
                    ),
                    (
                        Records::Q16 { codes, scales },
                        Records::Q16 {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                    ) => pair_native(
                        &Q16Rec::<RECORD> { codes, scales },
                        &Q16Rec::<TAIL_RECORD> {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                        &span,
                        part,
                    ),
                    (
                        Records::Q8Kc {
                            codes,
                            k_scales,
                            v_scales,
                        },
                        Records::Q8 {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                    ) => pair_native(
                        &Q8KcRec {
                            codes,
                            k_scales,
                            v_scales,
                        },
                        &Q8Rec::<TAIL_RECORD> {
                            codes: tail_codes,
                            scales: tail_scales,
                        },
                        &span,
                        part,
                    ),
                    _ => unreachable!("tail storage matches the prefix"),
                }
            }
        });
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
}

/// Unnormalized online-softmax state of both heads of one group over a span.
#[derive(Clone, Copy)]
struct Partial {
    out: [[f32; 64]; 2],
    max: [f32; 2],
    denominator: [f32; 2],
}
impl Partial {
    const EMPTY: Self = Self {
        out: [[0.0; 64]; 2],
        max: [f32::NEG_INFINITY; 2],
        denominator: [0.0; 2],
    };
}

struct Span<'a> {
    prefix_len: usize,
    group: usize,
    /// First tail record of the group.
    tail_base: usize,
    start: usize,
    end: usize,
    q0: &'a [f32],
    q1: &'a [f32],
}

/// Offsets of the key halves and the value inside a record.
#[derive(Clone, Copy)]
struct Layout {
    temporal: usize,
    spatial: [usize; 2],
    value: usize,
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
trait RecordStore {
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V;
}
struct F32Rec<'a, const W: usize>(&'a [f32]);
struct Bf16Rec<'a, const W: usize>(&'a [u16]);
struct Q8Rec<'a, const W: usize> {
    codes: &'a [i8],
    /// One scale per 32 elements.
    scales: &'a [f32],
}
impl<const W: usize> RecordStore for F32Rec<'_, W> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * W <= self.0.len());
        unsafe { S::load(self.0.as_ptr().add(record * W + offset)) }
    }
}
impl<const W: usize> RecordStore for Bf16Rec<'_, W> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * W <= self.0.len());
        // BF16 -> FP32 is exact: the bits move to the high half.
        unsafe { S::load_bf16(self.0.as_ptr().add(record * W + offset)) }
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
                S::splat(*self.scales.get_unchecked((record * W + offset) / 32)),
            )
        }
    }
}

struct Q16Rec<'a, const W: usize> {
    codes: &'a [i16],
    /// One scale per 32 elements.
    scales: &'a [f32],
}
impl<const W: usize> RecordStore for Q16Rec<'_, W> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * W <= self.codes.len());
        // fl(code * scale), exactly the scalar dequantization.
        unsafe {
            S::mul(
                S::load_i16(self.codes.as_ptr().add(record * W + offset)),
                S::splat(*self.scales.get_unchecked((record * W + offset) / 32)),
            )
        }
    }
}

/// Prefix records of `Records::Q8Kc` (width `RECORD`).
struct Q8KcRec<'a> {
    codes: &'a [i8],
    k_scales: &'a [f32],
    v_scales: &'a [f32],
}
impl RecordStore for Q8KcRec<'_> {
    #[inline(always)]
    unsafe fn load8<S: Isa>(&self, record: usize, offset: usize) -> S::V {
        debug_assert!((record + 1) * RECORD <= self.codes.len());
        // fl(code * scale) per lane, exactly the scalar dequantization. The
        // offset is a constant after inlining, so the branch folds away.
        unsafe {
            let codes = S::load_i8(self.codes.as_ptr().add(record * RECORD + offset));
            if offset < VALUE {
                let at = record / KC_TILE * VALUE + offset;
                S::mul(codes, S::load(self.k_scales.as_ptr().add(at)))
            } else {
                let at = 2 * record + (offset - VALUE) / 32;
                S::mul(codes, S::splat(*self.v_scales.get_unchecked(at)))
            }
        }
    }
}

/// `dot64(q, key)` for both heads of a record: the shared temporal half feeds
/// both accumulator sets first (dot64's i = 0 pass), then each spatial half
/// (its i = 32 pass). Per head this is dot64's exact FMA sequence and
/// reduction tree (`(a0+a1)+(a2+a3)`, then `Simd::sum`), which is also
/// `simd::dot` over 64 elements (the tail layout's single key).
#[inline(always)]
unsafe fn qk2<S: Isa, R: RecordStore>(
    store: &R,
    layout: Layout,
    record: usize,
    q0: &[f32],
    q1: &[f32],
) -> (f32, f32) {
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
            S::store(
                o0.as_mut_ptr().add(i),
                S::fma(f0, value, S::load(o0.as_ptr().add(i))),
            );
            S::store(
                o1.as_mut_ptr().add(i),
                S::fma(f1, value, S::load(o1.as_ptr().add(i))),
            );
        }
    }
}

#[inline(always)]
unsafe fn record_value<S: Isa, R: RecordStore>(
    store: &R,
    layout: Layout,
    record: usize,
) -> [S::V; 8] {
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
unsafe fn pair<S: Isa, R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    span: &Span<'_>,
    part: &mut Partial,
) {
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_native<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    span: &Span<'_>,
    part: &mut Partial,
) {
    unsafe { pair::<crate::simd::Avx2, R, T>(store, tail, span, part) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn pair_native<R: RecordStore, T: RecordStore>(
    store: &R,
    tail: &T,
    span: &Span<'_>,
    part: &mut Partial,
) {
    unsafe { pair::<crate::simd::Neon, R, T>(store, tail, span, part) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(c: &ModelConfig, p: usize, seed: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut k = vec![0.0; p * c.query_dim()];
        for t in 0..p {
            for h in 0..c.n_heads {
                for d in 0..64 {
                    let identity = if d < 32 { h / 2 } else { h };
                    k[(t * c.n_heads + h) * 64 + d] =
                        ((t * 31 + identity * 17 + d * 7 + seed) % 101) as f32 / 151.0 - 0.3;
                }
            }
        }
        let v: Vec<_> = (0..p * c.kv_dim())
            .map(|i| ((i + seed) % 73) as f32 / 97.0 - 0.2)
            .collect();
        let q: Vec<_> = (0..c.query_dim())
            .map(|i| ((i * 7 + seed) % 61) as f32 / 21.0 - 1.4)
            .collect();
        (k, v, q)
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_reference(
        c: &ModelConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tail_k: &[f32],
        tail_v: &[f32],
        p: usize,
        sinks: &[f32],
        backend: Simd,
    ) -> Vec<f32> {
        let generated = tail_k.len() / c.kv_dim();
        let mut all_v = v.to_vec();
        all_v.extend_from_slice(tail_v);
        let mut expected = vec![0.0; c.query_dim()];
        kernels::attention_compact_with_simd(
            q,
            k,
            tail_k,
            &all_v,
            1,
            p,
            p + generated,
            c.n_heads,
            c.n_kv_heads,
            64,
            p + generated - 1,
            0,
            p,
            sinks,
            &mut expected,
            backend,
        );
        expected
    }

    /// Appends `[generated][groups][64]` keys and values.
    fn push_rows(cache: &mut SplitPrefix, c: &ModelConfig, k: &[f32], v: &[f32]) {
        for (k, v) in k.chunks_exact(c.kv_dim()).zip(v.chunks_exact(c.kv_dim())) {
            cache.push_unique(k, v);
        }
    }

    /// The generated keys and values the tail records decode to.
    fn dequantized_tail(cache: &SplitPrefix) -> (Vec<f32>, Vec<f32>) {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for t in 0..cache.tail_len {
            for g in 0..cache.kv_heads {
                k.extend((0..64).map(|d| cache.tail_value(g, t, d)));
                v.extend((0..64).map(|d| cache.tail_value(g, t, 64 + d)));
            }
        }
        (k, v)
    }

    fn backends() -> Vec<Simd> {
        [Simd::Scalar, Simd::Auto]
            .into_iter()
            .filter(|s| s.validate().is_ok())
            .collect()
    }

    #[test]
    fn records_decode_bitwise_like_compact_over_their_values() {
        // FP32 records must equal the compact kernel on the original cache;
        // BF16/Q8 records must equal it on their own dequantized values.
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        for (p, generated) in [(1, 1), (3, 2), (127, 1), (128, 5), (129, 17), (300, 130)] {
            let (k, v, q) = fixture(&c, p, p);
            let tail_k: Vec<_> = (0..generated * c.kv_dim())
                .map(|i| ((i * 13) % 37) as f32 / 53.0 - 0.3)
                .collect();
            let tail_v: Vec<_> = (0..generated * c.kv_dim())
                .map(|i| ((i * 11) % 41) as f32 / 67.0 - 0.3)
                .collect();
            let sinks: Vec<_> = (0..c.n_heads).map(|h| h as f32 * 0.25 - 1.0).collect();
            for mode in [
                PrefixMode::SplitF32,
                PrefixMode::SplitBf16,
                PrefixMode::SplitQ8,
                PrefixMode::SplitQ8Kc,
            PrefixMode::SplitQ16,
                PrefixMode::SplitQ16,
            ] {
                let mut cache = SplitPrefix::from_compact(&k, &v, p, p + generated, &c, mode)
                    .unwrap()
                    .with_chunks(1);
                push_rows(&mut cache, &c, &tail_k, &tail_v);
                let (dk, dv) = cache.dequantized_compact();
                let (tk, tv) = dequantized_tail(&cache);
                if mode == PrefixMode::SplitF32 {
                    assert_eq!(dk, k);
                    assert_eq!(dv, v);
                    assert_eq!(tk, tail_k);
                    assert_eq!(tv, tail_v);
                }
                for backend in backends() {
                    let expected =
                        compact_reference(&c, &q, &dk, &dv, &tk, &tv, p, &sinks, backend);
                    let mut output = vec![f32::NAN; c.query_dim()];
                    cache.attention_decode(&q, p + generated, &sinks, &mut output, backend);
                    for (a, b) in output.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{mode:?} {backend:?} p{p}");
                    }
                }
            }
        }
    }

    #[test]
    fn position_chunks_change_rounding_only() {
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let (p, generated) = (1000, 40);
        let (k, v, q) = fixture(&c, p, 3);
        let tail: Vec<_> = (0..generated * c.kv_dim())
            .map(|i| ((i * 13) % 37) as f32 / 53.0 - 0.3)
            .collect();
        let sinks = vec![0.5; c.n_heads];
        for mode in [
            PrefixMode::SplitF32,
            PrefixMode::SplitBf16,
            PrefixMode::SplitQ8,
            PrefixMode::SplitQ8Kc,
            PrefixMode::SplitQ16,
        ] {
            let mut reference = vec![0.0; c.query_dim()];
            for chunks in [1, 2, 4] {
                let mut cache = SplitPrefix::from_compact(&k, &v, p, p + generated, &c, mode)
                    .unwrap()
                    .with_chunks(chunks);
                push_rows(&mut cache, &c, &tail, &tail);
                let mut output = vec![0.0; c.query_dim()];
                cache.attention_decode(&q, p + generated, &sinks, &mut output, Simd::Auto);
                if chunks == 1 {
                    reference = output;
                } else {
                    for (a, b) in output.iter().zip(&reference) {
                        assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{mode:?} {chunks}");
                    }
                }
            }
        }
    }

    #[test]
    fn low_precision_stays_close_to_fp32() {
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let p = 257;
        let (k, v, q) = fixture(&c, p, 9);
        let tail = vec![0.02; c.kv_dim()];
        let sinks = vec![0.5; c.n_heads];
        let expected = compact_reference(&c, &q, &k, &v, &tail, &tail, p, &sinks, Simd::Auto);
        for mode in [
            PrefixMode::SplitBf16,
            PrefixMode::SplitQ8,
            PrefixMode::SplitQ8Kc,
            PrefixMode::SplitQ16,
        ] {
            let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 1, &c, mode).unwrap();
            push_rows(&mut cache, &c, &tail, &tail);
            let mut output = vec![0.0; c.query_dim()];
            cache.attention_decode(&q, p + 1, &sinks, &mut output, Simd::Auto);
            assert!(output.iter().all(|x| x.is_finite()));
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 0.02)
            );
        }
    }

    #[test]
    fn nonfinite_tail_values_reach_the_output() {
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let p = 130;
        let (k, v, q) = fixture(&c, p, 5);
        let sinks = vec![0.5; c.n_heads];
        for mode in [
            PrefixMode::SplitF32,
            PrefixMode::SplitBf16,
            PrefixMode::SplitQ8,
            PrefixMode::SplitQ8Kc,
            PrefixMode::SplitQ16,
        ] {
            let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 2, &c, mode).unwrap();
            let mut tail = vec![0.01; c.kv_dim()];
            push_rows(&mut cache, &c, &tail, &tail);
            tail[3] = f32::NAN;
            push_rows(&mut cache, &c, &tail, &tail);
            for backend in backends() {
                let mut output = vec![0.0; c.query_dim()];
                cache.attention_decode(&q, p + 2, &sinks, &mut output, backend);
                assert!(
                    output[..64].iter().all(|x| x.is_nan()),
                    "{mode:?} {backend:?}"
                );
            }
        }
    }

    #[test]
    fn refuses_nonidentical_temporal_keys() {
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let mut k = vec![0.0; c.query_dim()];
        k[64] = 1.0;
        assert!(
            SplitPrefix::from_compact(&k, &vec![0.0; c.kv_dim()], 1, 2, &c, PrefixMode::SplitF32)
                .is_err()
        );
    }
}
