//! Experimental single-request BF16 attention following the observed Flex
//! arithmetic: 64-key tiles, local BF16 probability casts, FP32 accumulation,
//! sixteen decode splits, BF16 raw output, and a separate FP32 sink sigmoid.
//! Diagnostic copy of the production attention schedule; only QK dot changes.
//! Baseline equivalence is checked over every entire fixture before comparisons.

use super::qk_candidates::Candidate;
use falcon_ocr::bf16_kernels::{self, Backend};
use half::bf16;
use rayon::prelude::*;

pub use falcon_ocr::bf16_attention::Parameters;

struct Schedule {
    partial: [usize; 256],
    full: [usize; 256],
    partial_len: usize,
    full_len: usize,
}

fn visible_limit(query: usize, p: Parameters) -> usize {
    if (p.image_start..p.image_end).contains(&query) {
        p.image_end - 1
    } else {
        query
    }
}

fn schedule(query_absolute: usize, p: Parameters) -> Schedule {
    let start = query_absolute / 128 * 128;
    let minimum = visible_limit(start, p);
    let maximum = visible_limit(start + 127, p);
    let mut result = Schedule {
        partial: [0; 256],
        full: [0; 256],
        partial_len: 0,
        full_len: 0,
    };
    // Classification is on nominal cache blocks, NOT the truncated live query
    // or key dimensions. Decode slices the existing query block's schedule.
    for key in (0..p.capacity).step_by(128) {
        if key > maximum {
            continue;
        }
        let (indices, len) = if key + 127 <= minimum {
            (&mut result.full, &mut result.full_len)
        } else {
            (&mut result.partial, &mut result.partial_len)
        };
        indices[*len] = key;
        indices[*len + 1] = key + 64;
        *len += 2;
    }
    // Match the oracle's per-list cap; last nominal tiles may still lie beyond
    // the live key dimension and are explicitly masked by update().
    let tiles = p.kv_len.div_ceil(64);
    result.partial_len = result.partial_len.min(tiles);
    result.full_len = result.full_len.min(tiles);
    result
}

struct Partial {
    maximum: f32,
    denominator: f32,
    value: [f32; 64],
}
impl Partial {
    fn new() -> Self {
        Self {
            maximum: f32::NEG_INFINITY,
            denominator: 0.,
            value: [0.; 64],
        }
    }
}

fn sum<const N: usize>(mut values: [f32; N]) -> f32 {
    debug_assert!(N.is_power_of_two());
    let mut width = N / 2;
    while width > 0 {
        for i in 0..width {
            values[i] += values[i + width];
        }
        width /= 2;
    }
    values[0]
}

#[allow(clippy::too_many_arguments)]
fn update(
    state: &mut Partial,
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    head: usize,
    query: usize,
    key_start: usize,
    full: bool,
    p: Parameters,
    backend: Backend,
    candidate: Candidate,
) {
    if key_start >= p.kv_len {
        return;
    }
    let mut scores = [f32::NEG_INFINITY; 64];
    let count = 64.min(p.kv_len - key_start);
    for (lane, score) in scores[..count].iter_mut().enumerate() {
        let key = key_start + lane;
        if full
            || key <= query
            || ((p.image_start..p.image_end).contains(&query)
                && (p.image_start..p.image_end).contains(&key))
        {
            let offset = (key * p.heads + head) * 64;
            *score = (candidate.dot(q, &k[offset..offset + 64]) * 0.125) * 1.442_695_04_f32;
        }
    }
    let maximum = scores.iter().copied().fold(state.maximum, f32::max);
    let safe_maximum = if maximum == f32::NEG_INFINITY {
        0.
    } else {
        maximum
    };
    let alpha = (state.maximum - safe_maximum).exp2();
    let mut probabilities = [bf16::ZERO; 64];
    let mut unrounded = [0.; 64];
    for i in 0..64 {
        unrounded[i] = (scores[i] - safe_maximum).exp2();
        probabilities[i] = bf16::from_f32(unrounded[i]);
    }
    state.denominator = state.denominator * alpha + sum(unrounded);
    let mut column = [bf16::ZERO; 64];
    for dim in 0..64 {
        for lane in 0..count {
            column[lane] = v[((key_start + lane) * p.heads + head) * 64 + dim];
        }
        let product = bf16_kernels::dot(&probabilities, &column, backend);
        state.value[dim] = state.value[dim] * alpha + product;
    }
    state.maximum = maximum;
}

/// Q/K/V and outputs are token-major `[tokens,heads,64]`. LSE is token-major
/// `[query_len,heads]`. The raw and sink-scaled BF16 outputs are both exposed
/// so local operator checks can inspect the otherwise hidden cast boundary.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    sinks: &[bf16],
    p: Parameters,
    output: &mut [bf16],
    raw: &mut [bf16],
    lse: &mut [f32],
    backend: Backend,
    candidate: Candidate,
) {
    assert!(p.query_len > 0 && p.heads > 0 && p.kv_len > 0);
    assert!(p.capacity >= p.kv_len && p.capacity <= 16384 && p.capacity.is_multiple_of(128));
    assert!(p.image_start < p.image_end && p.image_end <= p.kv_len);
    assert!(p.query_offset + p.query_len <= p.kv_len);
    assert_eq!(q.len(), p.query_len * p.heads * 64);
    assert_eq!(k.len(), p.kv_len * p.heads * 64);
    assert_eq!(v.len(), k.len());
    assert_eq!(output.len(), q.len());
    assert_eq!(raw.len(), q.len());
    assert_eq!(lse.len(), p.query_len * p.heads);
    assert_eq!(sinks.len(), p.heads);
    let backend = backend.resolved();
    output
        .par_chunks_mut(64)
        .zip(raw.par_chunks_mut(64))
        .zip(lse.par_iter_mut())
        .enumerate()
        .for_each(|(index, ((out, raw), lse))| {
            let head = index % p.heads;
            let query = p.query_offset + index / p.heads;
            let q = &q[index * 64..(index + 1) * 64];
            let schedule = schedule(query, p);
            let (value, denominator, maximum) = if p.query_len == 1 {
                let mut splits: [Partial; 16] = std::array::from_fn(|_| Partial::new());
                let tiles_per_split = p.kv_len.div_ceil(16).div_ceil(64);
                for (split, state) in splits.iter_mut().enumerate() {
                    let lo = split * tiles_per_split;
                    let hi = ((split + 1) * tiles_per_split).min(schedule.partial_len);
                    for tile in lo..hi {
                        update(
                            state,
                            q,
                            k,
                            v,
                            head,
                            query,
                            schedule.partial[tile],
                            false,
                            p,
                            backend,
                            candidate,
                        );
                    }
                    let lo = (15 - split) * tiles_per_split;
                    let hi = ((16 - split) * tiles_per_split).min(schedule.full_len);
                    for tile in lo..hi {
                        update(
                            state,
                            q,
                            k,
                            v,
                            head,
                            query,
                            schedule.full[tile],
                            true,
                            p,
                            backend,
                            candidate,
                        );
                    }
                }
                let maximum = splits
                    .iter()
                    .map(|s| s.maximum)
                    .fold(f32::NEG_INFINITY, f32::max);
                let scales: [f32; 16] =
                    std::array::from_fn(|i| (splits[i].maximum - maximum).exp2());
                let denominator = sum(std::array::from_fn::<_, 16, _>(|i| {
                    splits[i].denominator * scales[i]
                }));
                let value = std::array::from_fn(|dim| {
                    sum(std::array::from_fn::<_, 16, _>(|i| {
                        splits[i].value[dim] * scales[i]
                    }))
                });
                (value, denominator, maximum)
            } else {
                let mut state = Partial::new();
                for &tile in &schedule.partial[..schedule.partial_len] {
                    update(
                        &mut state, q, k, v, head, query, tile, false, p, backend, candidate,
                    );
                }
                for &tile in &schedule.full[..schedule.full_len] {
                    update(
                        &mut state, q, k, v, head, query, tile, true, p, backend, candidate,
                    );
                }
                (
                    state.value,
                    if state.denominator == 0. {
                        1.
                    } else {
                        state.denominator
                    },
                    state.maximum,
                )
            };
            *lse = (maximum + denominator.log2()) * std::f32::consts::LN_2;
            let scale = (1. + (sinks[head].to_f32() - *lse).exp()).recip();
            for dim in 0..64 {
                raw[dim] = bf16::from_f32(value[dim] / denominator);
                out[dim] = bf16::from_f32(raw[dim].to_f32() * scale);
            }
        });
}
