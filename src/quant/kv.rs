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
use super::Kv;
use crate::{
    config::ModelConfig,
    kernels::{self, Simd},
};
use anyhow::{Result, ensure};
use rayon::prelude::*;

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
}

/// Default decode exp: the polynomial one for 8-bit caches when the run
/// allows fast exps (`ExpMode::Fast`, i.e. fast mode: token agreement with
/// FP32 and the English gate unchanged, verification attention 3-10%
/// faster), the platform-exact one otherwise; `requested`
/// (`Tuning::decode_fast_exp`) overrides it.
pub(crate) fn default_fast_exp(mode: Kv, fast_exps: bool, requested: Option<bool>) -> bool {
    requested.unwrap_or(fast_exps && mode == Kv::Q8)
}

/// Default position chunks: exact for FP32, split (rounding-level) for lossy
/// storage; `requested` (`Tuning::split_chunks`, 1..=4) overrides it.
fn default_chunks(mode: Kv, requested: Option<usize>) -> usize {
    // Four chunks (32 tasks) measured ~13% faster attention than two on a
    // 16-thread 7950X at a 6.5k prefix; FP32 keeps the exact single scan.
    requested
        .filter(|n| (1..=MAX_CHUNKS).contains(n))
        .unwrap_or(match mode {
            Kv::F32Split => 1,
            _ => 4,
        })
}

impl Records {
    fn bytes(&self) -> usize {
        match self {
            Records::F32(d) => d.capacity() * 4,
            Records::Q8 { codes, scales } => codes.capacity() + scales.capacity() * 2,
            Records::Q16 { codes, scales } => codes.capacity() * 2 + scales.capacity() * 2,
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

/// The smallest BF16 value at or above `scale` (a positive finite scale, 0,
/// or NaN), as BF16 bits. Codes quantized against it never exceed the range,
/// and the step grows by at most 2^-8 relative.
fn bf16_up(scale: f32) -> u16 {
    if scale.is_nan() {
        return 0x7FC0;
    }
    let bits = scale.to_bits();
    let high = (bits >> 16) as u16;
    if bits & 0xFFFF != 0 { high + 1 } else { high }
}

/// BF16 bits widened exactly to FP32.
#[inline(always)]
fn bf16_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn q16_code(value: f32, scale: f32) -> i16 {
    if scale == 0.0 {
        0
    } else {
        (value as f64 / scale as f64).round_ties_even().clamp(-32767.0, 32767.0) as i16
    }
}

fn q8_code(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        0
    } else {
        (value as f64 / scale as f64).round_ties_even().clamp(-127.0, 127.0) as i8
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
        ensure!(
            mode != Kv::Compact && c.head_dim == 64 && c.n_heads == 2 * c.n_kv_heads && c.n_kv_heads <= MAX_GROUPS,
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
            Ok(())
        };
        let count = groups * prefix_len;
        let records = match mode {
            Kv::F32Split => {
                let mut data = vec![0.0_f32; count * RECORD];
                data.par_chunks_mut(RECORD).enumerate().try_for_each(|(record, dst)| {
                    let mut values = [0.0; RECORD];
                    gather(record, &mut values)?;
                    dst.copy_from_slice(&values);
                    Ok::<_, anyhow::Error>(())
                })?;
                Records::F32(data)
            }
            Kv::Q8 => {
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
            Kv::Q16 => {
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
            Kv::Compact => unreachable!("checked above"),
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
        };
        Ok(Self {
            records,
            prefix_len,
            heads,
            kv_heads: groups,
            chunks: default_chunks(mode, chunks),
            tail,
            tail_capacity,
            tail_len: 0,
            fast_exp: false,
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

    pub fn attention_decode(&self, q: &[f32], total_len: usize, sinks: &[f32], output: &mut [f32], simd: Simd) {
        assert_eq!(q.len(), self.heads * 64);
        assert_eq!(output.len(), q.len());
        assert_eq!(sinks.len(), self.heads);
        assert_eq!(total_len, self.prefix_len + self.tail_len);
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
        let last = self.prefix_len + self.tail_len;
        if rows == 1 {
            self.attention_decode(q, last, sinks, output, simd);
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

/// A decode kernel over one prefix/tail store pair (see `SplitPrefix::dispatch`).
trait PairKernel {
    unsafe fn run<R: RecordStore, T: RecordStore>(&mut self, store: &R, tail: &T);
}

/// [`pair_rows`] over several query rows of one (group, chunk).
struct RowsKernel<'a, 'b> {
    spans: &'a [Span<'b>],
    parts: &'a mut [Partial],
    fast_exp: bool,
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
struct Q8Rec<'a, const W: usize> {
    codes: &'a [i8],
    /// One BF16 scale per 32 elements.
    scales: &'a [u16],
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

struct Q16Rec<'a, const W: usize> {
    codes: &'a [i16],
    /// One BF16 scale per 32 elements.
    scales: &'a [u16],
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
unsafe fn pair_entry<R: RecordStore, T: RecordStore>(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_decode_exp_defaults_to_8_bit_caches_under_fast_exps() {
        assert!(default_fast_exp(Kv::Q8, true, None));
        assert!(
            !default_fast_exp(Kv::Q8, false, None),
            "the reference configuration stays exact"
        );
        assert!(!default_fast_exp(Kv::Q16, true, None), "near-exact stays exact");
        assert!(!default_fast_exp(Kv::F32Split, true, None));
        assert!(!default_fast_exp(Kv::Q8, true, Some(false)));
        assert!(default_fast_exp(Kv::Q16, false, Some(true)));
    }

    #[test]
    fn verification_rows_are_bitwise_single_rows() {
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let width = c.query_dim();
        for (p, generated, rows) in [(300, 5, 5), (1000, 9, 8), (130, 3, 2), (7, 4, 4), (254, 6, 6)] {
            let (k, v, _) = fixture(&c, p, p + 3);
            let tail_k: Vec<_> = (0..generated * c.kv_dim())
                .map(|i| ((i * 13) % 37) as f32 / 53.0 - 0.3)
                .collect();
            let tail_v: Vec<_> = (0..generated * c.kv_dim())
                .map(|i| ((i * 11) % 41) as f32 / 67.0 - 0.3)
                .collect();
            let q: Vec<_> = (0..rows * width)
                .map(|i| ((i * 7 + 3) % 61) as f32 / 21.0 - 1.4)
                .collect();
            let sinks: Vec<_> = (0..c.n_heads).map(|h| h as f32 * 0.25 - 1.0).collect();
            for (mode, fast_exp) in [Kv::F32Split, Kv::Q8, Kv::Q16]
                .into_iter()
                .flat_map(|m| [(m, false), (m, true)])
            {
                for chunks in [1, 2, 4] {
                    for backend in backends() {
                        let mut cache = SplitPrefix::from_compact(&k, &v, p, p + generated, &c, mode, None)
                            .unwrap()
                            .with_chunks(chunks)
                            .with_fast_exp(fast_exp);
                        push_rows(&mut cache, &c, &tail_k, &tail_v);
                        let mut joint = vec![f32::NAN; rows * width];
                        cache.attention_decode_rows(&q, rows, &sinks, &mut joint, backend);
                        for r in (0..rows).rev() {
                            cache.truncate_tail(generated - (rows - 1 - r));
                            let mut single = vec![f32::NAN; width];
                            let total = p + cache.tail_len;
                            cache.attention_decode(&q[r * width..(r + 1) * width], total, &sinks, &mut single, backend);
                            for (a, b) in joint[r * width..(r + 1) * width].iter().zip(&single) {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "{mode:?} fast exp {fast_exp} chunks {chunks} {backend:?} p{p} row {r}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

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
        kernels::attention_with_simd(
            q,
            &kernels::CompactKv {
                prefix_k: k,
                generated_k: tail_k,
                v: &all_v,
                prefix_len: p,
                total_len: p + generated,
                n_heads: c.n_heads,
                n_kv_heads: c.n_kv_heads,
                head_dim: 64,
            },
            1,
            kernels::Geometry::new(p + generated - 1, 0, p),
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
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        for (p, generated) in [(1, 1), (3, 2), (127, 1), (128, 5), (129, 17), (300, 130)] {
            let (k, v, q) = fixture(&c, p, p);
            let tail_k: Vec<_> = (0..generated * c.kv_dim())
                .map(|i| ((i * 13) % 37) as f32 / 53.0 - 0.3)
                .collect();
            let tail_v: Vec<_> = (0..generated * c.kv_dim())
                .map(|i| ((i * 11) % 41) as f32 / 67.0 - 0.3)
                .collect();
            let sinks: Vec<_> = (0..c.n_heads).map(|h| h as f32 * 0.25 - 1.0).collect();
            for mode in [Kv::F32Split, Kv::Q8, Kv::Q16] {
                let mut cache = SplitPrefix::from_compact(&k, &v, p, p + generated, &c, mode, None)
                    .unwrap()
                    .with_chunks(1);
                push_rows(&mut cache, &c, &tail_k, &tail_v);
                let (dk, dv) = cache.dequantized_compact();
                let (tk, tv) = dequantized_tail(&cache);
                if mode == Kv::F32Split {
                    assert_eq!(dk, k);
                    assert_eq!(dv, v);
                    assert_eq!(tk, tail_k);
                    assert_eq!(tv, tail_v);
                }
                for backend in backends() {
                    let expected = compact_reference(&c, &q, &dk, &dv, &tk, &tv, p, &sinks, backend);
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
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let (p, generated) = (1000, 40);
        let (k, v, q) = fixture(&c, p, 3);
        let tail: Vec<_> = (0..generated * c.kv_dim())
            .map(|i| ((i * 13) % 37) as f32 / 53.0 - 0.3)
            .collect();
        let sinks = vec![0.5; c.n_heads];
        for mode in [Kv::F32Split, Kv::Q8, Kv::Q16] {
            let mut reference = vec![0.0; c.query_dim()];
            for chunks in [1, 2, 4] {
                let mut cache = SplitPrefix::from_compact(&k, &v, p, p + generated, &c, mode, None)
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
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let p = 257;
        let (k, v, q) = fixture(&c, p, 9);
        let tail = vec![0.02; c.kv_dim()];
        let sinks = vec![0.5; c.n_heads];
        let expected = compact_reference(&c, &q, &k, &v, &tail, &tail, p, &sinks, Simd::Auto);
        for mode in [Kv::Q8, Kv::Q16] {
            let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 1, &c, mode, None).unwrap();
            push_rows(&mut cache, &c, &tail, &tail);
            let mut output = vec![0.0; c.query_dim()];
            cache.attention_decode(&q, p + 1, &sinks, &mut output, Simd::Auto);
            assert!(output.iter().all(|x| x.is_finite()));
            assert!(output.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 0.02));
        }
    }

    #[test]
    fn nonfinite_tail_values_reach_the_output() {
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let p = 130;
        let (k, v, q) = fixture(&c, p, 5);
        let sinks = vec![0.5; c.n_heads];
        for mode in [Kv::F32Split, Kv::Q8, Kv::Q16] {
            let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 2, &c, mode, None).unwrap();
            let mut tail = vec![0.01; c.kv_dim()];
            push_rows(&mut cache, &c, &tail, &tail);
            tail[3] = f32::NAN;
            push_rows(&mut cache, &c, &tail, &tail);
            for backend in backends() {
                let mut output = vec![0.0; c.query_dim()];
                cache.attention_decode(&q, p + 2, &sinks, &mut output, backend);
                assert!(output[..64].iter().all(|x| x.is_nan()), "{mode:?} {backend:?}");
            }
        }
    }

    #[test]
    fn refuses_nonidentical_temporal_keys() {
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let mut k = vec![0.0; c.query_dim()];
        k[64] = 1.0;
        assert!(SplitPrefix::from_compact(&k, &vec![0.0; c.kv_dim()], 1, 2, &c, Kv::F32Split, None).is_err());
    }
}

#[cfg(test)]
mod probe {
    use super::*;

    /// Verification attention time per step by row count (22 layers of Q8
    /// caches, 6,544 prefix positions, 12 threads): the cost of each extra
    /// drafted row.
    #[test]
    #[ignore = "timing probe; run in release with --nocapture"]
    fn verify_rows_probe() {
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let (p, layers, generated) = (6544, 22, 8);
        let k: Vec<f32> = (0..p * c.query_dim())
            .map(|i| {
                let (t, rest) = (i / c.query_dim(), i % c.query_dim());
                let (h, d) = (rest / 64, rest % 64);
                let identity = if d < 32 { h / 2 } else { h };
                ((t * 31 + identity * 17 + d * 7) % 101) as f32 / 151.0 - 0.3
            })
            .collect();
        let v: Vec<f32> = (0..p * c.kv_dim()).map(|i| (i % 73) as f32 / 97.0 - 0.2).collect();
        let tail: Vec<f32> = (0..c.kv_dim()).map(|i| (i % 29) as f32 / 41.0 - 0.3).collect();
        let sinks = vec![0.0_f32; c.n_heads];
        let mode = std::env::var("PROBE_MODE").map_or(Kv::Q8, |m| match m.as_str() {
            "q16" => Kv::Q16,
            "f32" => Kv::F32Split,
            _ => Kv::Q8,
        });
        let caches: Vec<SplitPrefix> = (0..layers)
            .map(|_| {
                let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 16, &c, mode, None).unwrap();
                for _ in 0..generated {
                    cache.push_unique(&tail, &tail);
                }
                cache
            })
            .collect();
        let threads = std::env::var("PROBE_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(12);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        println!("{threads} threads");
        pool.install(|| {
            let mut base = 0.0;
            for rows in [1, 2, 3, 5, 8] {
                let q: Vec<f32> = (0..rows * c.query_dim())
                    .map(|i| (i % 61) as f32 / 21.0 - 1.4)
                    .collect();
                let mut out = vec![0.0; q.len()];
                for cache in &caches {
                    cache.attention_decode_rows(&q, rows, &sinks, &mut out, Simd::Auto);
                }
                let rounds = 10;
                let t = std::time::Instant::now();
                for _ in 0..rounds {
                    for cache in &caches {
                        cache.attention_decode_rows(&q, rows, &sinks, &mut out, Simd::Auto);
                    }
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
                if rows == 1 {
                    base = ms;
                }
                let extra = if rows > 1 { (ms - base) / (rows - 1) as f64 } else { 0.0 };
                println!("{mode:?} rows {rows}: {ms:.2} ms/step, {extra:.2} ms per extra row");
            }
        });
    }

    /// Decode attention time per token over 22 layers' worth of caches (so the
    /// records stream from memory) by record format and thread count.
    #[test]
    #[ignore = "timing probe; run in release with --nocapture"]
    fn decode_attention_bandwidth_probe() {
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let (p, layers) = (6544, 22);
        let k: Vec<f32> = (0..p * c.query_dim())
            .map(|i| {
                let (t, rest) = (i / c.query_dim(), i % c.query_dim());
                let (h, d) = (rest / 64, rest % 64);
                let identity = if d < 32 { h / 2 } else { h };
                ((t * 31 + identity * 17 + d * 7) % 101) as f32 / 151.0 - 0.3
            })
            .collect();
        let v: Vec<f32> = (0..p * c.kv_dim()).map(|i| (i % 73) as f32 / 97.0 - 0.2).collect();
        let q: Vec<f32> = (0..c.query_dim()).map(|i| (i % 61) as f32 / 21.0 - 1.4).collect();
        let sinks = vec![0.0_f32; c.n_heads];
        for mode in [Kv::Q8, Kv::Q16, Kv::F32Split] {
            let caches: Vec<SplitPrefix> = (0..layers)
                .map(|_| {
                    SplitPrefix::from_compact(&k, &v, p, p + 8, &c, mode, None)
                        .unwrap()
                        .with_chunks(4)
                })
                .collect();
            let bytes: usize = caches.iter().map(|x| x.records.bytes()).sum();
            for threads in [4, 8, 12, 16] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let mut out = vec![0.0; c.query_dim()];
                pool.install(|| {
                    for cache in &caches {
                        cache.attention_decode(&q, p, &sinks, &mut out, Simd::Auto);
                    }
                    let rounds = 10;
                    let t = std::time::Instant::now();
                    for _ in 0..rounds {
                        for cache in &caches {
                            cache.attention_decode(&q, p, &sinks, &mut out, Simd::Auto);
                        }
                    }
                    let ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
                    println!(
                        "{mode:?} {threads:2} threads: {ms:.2} ms/token, {:.0} MB, {:.1} GB/s",
                        bytes as f64 / 1e6,
                        bytes as f64 / (ms * 1e6)
                    );
                });
            }
        }
    }
}
