//! Flash-style tiled GEMM prefill attention: each Rayon task owns one
//! contiguous query tile and all of its heads. Temporary logits and
//! probabilities are reused on the stack. GEMM is deliberately
//! single-threaded here because Rayon already schedules the outer tiles in
//! the runner's bounded pool.
use rayon::prelude::*;

use super::{CompactKv, Geometry};
#[cfg(target_arch = "x86_64")]
use crate::kernels::{avx2_available, vexp};

/// Several queries over `kv`. A key tile crossing the prefix boundary is
/// gathered before GEMM, so its tile width and softmax reduction order are
/// those of the expanded layout; an expanded cache never gathers.
pub(super) fn attention_gemm(q: &[f32], kv: &CompactKv<'_>, geometry: Geometry, sinks: &[f32], output: &mut [f32]) {
    const QUERY_TILE: usize = 32;
    const KEY_TILE: usize = 128;
    let (head_dim, prefix_len) = (kv.head_dim, kv.prefix_len);
    let (query_width, kv_width) = (kv.query_width(), kv.kv_width());
    let query_stride = isize::try_from(query_width).expect("attention query stride too large");
    let kv_stride = isize::try_from(kv_width).expect("attention KV stride too large");
    let scale = (head_dim as f32).sqrt().recip();
    let repeat = kv.repeat();
    let (prefix_k, generated_k, v) = (kv.prefix_k, kv.generated_k, kv.v);
    output
        .par_chunks_mut(QUERY_TILE * query_width)
        .enumerate()
        .for_each(|(tile, out)| {
            let first_query = tile * QUERY_TILE;
            let queries = out.len() / query_width;
            let first_absolute = geometry.query_offset + first_query;
            let last_absolute = first_absolute + queries - 1;
            // A tile can cross BOS/image/text boundaries. Every row is still
            // individually masked below; this bound only avoids invisible tiles.
            let intersects_image = first_absolute < geometry.image_end && last_absolute >= geometry.image_start;
            let visible_end = if intersects_image {
                (last_absolute + 1).max(geometry.image_end)
            } else {
                last_absolute + 1
            };
            let mut scores = [0.0_f32; QUERY_TILE * KEY_TILE];
            let mut maxima = [0.0_f32; QUERY_TILE];
            let mut denominators = [0.0_f32; QUERY_TILE];
            // Pure prefill has no segmented K boundary and leaves this empty.
            // For multiquery continuation only, at most one K tile per query tile
            // needs this bounded KEY_TILE*head_dim buffer; reuse it across heads.
            let mut boundary_keys = Vec::<f32>::new();
            for head in 0..kv.n_heads {
                let kv_head = head / repeat;
                maxima[..queries].fill(f32::NEG_INFINITY);
                denominators[..queries].fill(0.0);
                for row in 0..queries {
                    out[row * query_width + head * head_dim..row * query_width + (head + 1) * head_dim].fill(0.0);
                }
                for key_start in (0..visible_end).step_by(KEY_TILE) {
                    let keys = (visible_end - key_start).min(KEY_TILE);
                    let q_start = first_query * query_width + head * head_dim;
                    let (key_data, key_start_offset, key_stride) = if key_start + keys <= prefix_len {
                        (prefix_k, key_start * query_width + head * head_dim, query_stride)
                    } else if key_start >= prefix_len {
                        (
                            generated_k,
                            (key_start - prefix_len) * kv_width + kv_head * head_dim,
                            kv_stride,
                        )
                    } else {
                        boundary_keys.resize(keys * head_dim, 0.0);
                        for key in 0..keys {
                            boundary_keys[key * head_dim..(key + 1) * head_dim]
                                .copy_from_slice(kv.key(key_start + key, head));
                        }
                        (boundary_keys.as_slice(), 0, head_dim as isize)
                    };
                    // SAFETY: The entry point validated the shapes. The K
                    // interval is entirely in one buffer or gathered contiguously;
                    // no matrix straddles the separately allocated cache segments.
                    unsafe {
                        gemm::gemm(
                            queries,
                            keys,
                            head_dim,
                            scores.as_mut_ptr(),
                            1,
                            KEY_TILE as isize,
                            false,
                            q.as_ptr().add(q_start),
                            1,
                            query_stride,
                            key_data.as_ptr().add(key_start_offset),
                            key_stride,
                            1,
                            0.0_f32,
                            scale,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                    for row in 0..queries {
                        let absolute = first_absolute + row;
                        let row_scores = &mut scores[row * KEY_TILE..row * KEY_TILE + keys];
                        let mut block_max = f32::NEG_INFINITY;
                        for (col, score) in row_scores.iter_mut().enumerate() {
                            if geometry.masked(absolute, key_start + col) {
                                *score = f32::NEG_INFINITY;
                            }
                            block_max = block_max.max(*score);
                        }
                        let new_max = maxima[row].max(block_max);
                        // Avoid (-inf)-(-inf) when an entire block is masked.
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
                            &mut out[row * query_width + head * head_dim..row * query_width + (head + 1) * head_dim];
                        for value in out_row {
                            *value *= rescale;
                        }
                        denominators[row] *= rescale;
                        exp_shifted(row_scores, new_max);
                        for probability in row_scores.iter() {
                            denominators[row] += *probability;
                        }
                        maxima[row] = new_max;
                    }
                    let value_start = key_start * kv_width + kv_head * head_dim;
                    // Numerators += probabilities @ values. The output's row
                    // stride skips other heads; each Rayon task owns all rows of
                    // its output tile, so no writes overlap between tasks.
                    unsafe {
                        gemm::gemm(
                            queries,
                            head_dim,
                            keys,
                            out.as_mut_ptr().add(head * head_dim),
                            1,
                            query_stride,
                            true,
                            scores.as_ptr(),
                            1,
                            KEY_TILE as isize,
                            v.as_ptr().add(value_start),
                            1,
                            kv_stride,
                            1.0_f32,
                            1.0_f32,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                }
                for row in 0..queries {
                    let out_row =
                        &mut out[row * query_width + head * head_dim..row * query_width + (head + 1) * head_dim];
                    let logsumexp = maxima[row] + denominators[row].ln();
                    let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
                    for value in out_row {
                        *value = (*value / denominators[row]) * sink_scale;
                    }
                }
            }
        });
}

/// `values[i] = (values[i] - shift).exp()`, vectorized where available with a
/// bitwise-identical exp (see `vexp`).
fn exp_shifted(values: &mut [f32], shift: f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: AVX2/FMA availability checked.
        unsafe { vexp::exp_shifted_in_place(values, shift) };
        return;
    }
    for value in values {
        *value = (*value - shift).exp();
    }
}
