//! Fixed-width (`head_dim == 64`) AVX2/FMA single-query attention for decode.
//!
//! The per-head bodies are the generic online-softmax loops from `kernels.rs`
//! with their function-pointer dot/AXPY calls replaced by 64-wide helpers that
//! are inlined into a `#[target_feature]` head function. The helpers keep the
//! same four-accumulator FMA order and horizontal reduction as `x86::dot_avx2`
//! and `x86::axpy_avx2`, so every output is bit-identical to the generic path;
//! the tests below assert that. GQA repeat, tiles, image mask, softmax, sink,
//! division and multiplication order are unchanged.
//!
//! Promoted from `experiments/attention64` (expanded cache, measured 6.15–9.43%
//! lower full-page latency) and `experiments/attention64_compact` (compact
//! cache, a further 6.15–6.40%); see `docs/PERFORMANCE.md`.
use rayon::prelude::*;
use std::arch::x86_64::*;

/// Expanded-cache attention for one query row over `n_heads` heads of width 64.
///
/// # Safety
/// The caller must have validated AVX2/FMA support and every slice shape
/// exactly as `attention_with_simd` does for `query_len == 1`, `head_dim == 64`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    let scale = (64_f32).sqrt().recip();
    output.par_chunks_mut(64).enumerate().for_each(|(qh, out)| {
        // SAFETY: The parent checked AVX2/FMA and all shapes before entering.
        // The target-feature function encloses the complete per-head loop;
        // Rayon closures do not need to inherit the caller's target features.
        unsafe {
            head(
                qh,
                q,
                k,
                v,
                n_heads,
                query_offset,
                image_start,
                image_end,
                sinks,
                scale,
                out,
            );
        }
    });
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn head(
    qh: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    let head_dim = 64;
    let token_width = n_heads * head_dim;
    // SAFETY: Shapes and features were checked by the safe entry point. This
    // body is the generic loop verbatim except its dot and AXPY call targets.
    unsafe {
        let query = qh / n_heads;
        let head = qh % n_heads;
        let absolute_query = query_offset + query;
        let query_in_image = absolute_query >= image_start && absolute_query < image_end;
        let visible_end = if query_in_image {
            image_end
        } else {
            absolute_query + 1
        };
        let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
        out.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        const TILE: usize = 128;
        let mut logits = [0.0_f32; TILE];
        for start in (0..visible_end).step_by(TILE) {
            let len = (visible_end - start).min(TILE);
            let mut block_max = f32::NEG_INFINITY;
            for (j, logit) in logits[..len].iter_mut().enumerate() {
                let key = start + j;
                let begin = key * token_width + head * head_dim;
                *logit = dot64(qvec, &k[begin..begin + head_dim]) * scale;
                block_max = block_max.max(*logit);
            }
            let new_max = running_max.max(block_max);
            let rescale = if running_max == f32::NEG_INFINITY {
                0.0
            } else {
                (running_max - new_max).exp()
            };
            for value in out.iter_mut() {
                *value *= rescale;
            }
            denominator *= rescale;
            for (j, logit) in logits[..len].iter().enumerate() {
                let probability = (*logit - new_max).exp();
                denominator += probability;
                let begin = (start + j) * token_width + head * head_dim;
                axpy64(probability, &v[begin..begin + head_dim], out);
            }
            running_max = new_max;
        }
        let logsumexp = running_max + denominator.ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in out {
            *value = (*value / denominator) * sink_scale;
        }
    }
}

/// Compact-cache attention for one query row: prefix keys keep every query
/// head, generated keys and all values keep only the KV heads.
///
/// # Safety
/// The caller must have validated AVX2/FMA support and every slice shape
/// exactly as `attention_compact_with_simd` does for `query_len == 1`,
/// `head_dim == 64`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn compact(
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    prefix_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    let scale = (64_f32).sqrt().recip();
    output.par_chunks_mut(64).enumerate().for_each(|(qh, out)| {
        // SAFETY: The parent validated features, slices and shape relationships.
        unsafe {
            compact_head(
                qh,
                q,
                prefix_k,
                generated_k,
                v,
                prefix_len,
                n_heads,
                n_kv_heads,
                query_offset,
                image_start,
                image_end,
                sinks,
                scale,
                out,
            );
        }
    });
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn compact_head(
    qh: usize,
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    prefix_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    let head_dim = 64;
    let query_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let repeat = n_heads / n_kv_heads;
    // SAFETY: The safe entry point checked shapes/features. This body is the
    // generic compact loop verbatim except its dot and AXPY call targets.
    unsafe {
        let query = qh / n_heads;
        let head = qh % n_heads;
        let kv_head = head / repeat;
        let absolute_query = query_offset + query;
        let query_in_image = absolute_query >= image_start && absolute_query < image_end;
        let visible_end = if query_in_image {
            image_end
        } else {
            absolute_query + 1
        };
        let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
        out.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        const TILE: usize = 128;
        let mut logits = [0.0_f32; TILE];
        for start in (0..visible_end).step_by(TILE) {
            let len = (visible_end - start).min(TILE);
            let mut block_max = f32::NEG_INFINITY;
            for (j, logit) in logits[..len].iter_mut().enumerate() {
                let key = start + j;
                let kvec = if key < prefix_len {
                    let begin = key * query_width + head * head_dim;
                    &prefix_k[begin..begin + head_dim]
                } else {
                    let begin = (key - prefix_len) * kv_width + kv_head * head_dim;
                    &generated_k[begin..begin + head_dim]
                };
                *logit = dot64(qvec, kvec) * scale;
                block_max = block_max.max(*logit);
            }
            let new_max = running_max.max(block_max);
            let rescale = if running_max == f32::NEG_INFINITY {
                0.0
            } else {
                (running_max - new_max).exp()
            };
            for value in out.iter_mut() {
                *value *= rescale;
            }
            denominator *= rescale;
            for (j, logit) in logits[..len].iter().enumerate() {
                let probability = (*logit - new_max).exp();
                denominator += probability;
                let begin = (start + j) * kv_width + kv_head * head_dim;
                axpy64(probability, &v[begin..begin + head_dim], out);
            }
            running_max = new_max;
        }
        let logsumexp = running_max + denominator.ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in out {
            *value = (*value / denominator) * sink_scale;
        }
    }
}

// These helpers are deliberately inline(always), with the feature-qualified
// head as their caller. No function-pointer call remains inside the key loops.
#[inline(always)]
unsafe fn dot64(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 64);
    debug_assert_eq!(b.len(), 64);
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for i in [0, 32] {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                acc3,
            );
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let pairs = _mm_hadd_ps(halves, halves);
        _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
    }
}

#[inline(always)]
unsafe fn axpy64(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), 64);
    debug_assert_eq!(y.len(), 64);
    unsafe {
        let factor = _mm256_set1_ps(a);
        for i in [0, 8, 16, 24, 32, 40, 48, 56] {
            let value = _mm256_fmadd_ps(
                factor,
                _mm256_loadu_ps(x.as_ptr().add(i)),
                _mm256_loadu_ps(y.as_ptr().add(i)),
            );
            _mm256_storeu_ps(y.as_mut_ptr().add(i), value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        Simd, attention_compact_online_softmax, attention_compact_with_simd, attention_gemm,
        attention_gemm_compact, attention_online_softmax, attention_with_simd, x86,
    };
    use super::*;

    fn supported() {
        assert!(
            Simd::Avx2.validate().is_ok(),
            "these tests require AVX2/FMA; no silent skip"
        );
    }

    fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if i % 97 == 0 {
                    -0.0
                } else {
                    (((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32) * scale
                }
            })
            .collect()
    }

    fn exact(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "bit difference at {i}: {x:?} vs {y:?}"
            );
        }
    }

    /// The public expanded entry point without the fixed-width dispatch.
    #[allow(clippy::too_many_arguments)]
    fn expanded_oracle(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        query_len: usize,
        n_heads: usize,
        head_dim: usize,
        query_offset: usize,
        image_start: usize,
        image_end: usize,
        sinks: &[f32],
        output: &mut [f32],
        simd: Simd,
    ) {
        let selected = simd.resolved();
        if query_len >= 4 && selected != Simd::Scalar {
            attention_gemm(
                q,
                k,
                v,
                n_heads,
                head_dim,
                query_offset,
                image_start,
                image_end,
                sinks,
                output,
            );
            return;
        }
        attention_online_softmax(
            q,
            k,
            v,
            n_heads,
            head_dim,
            query_offset,
            image_start,
            image_end,
            sinks,
            output,
            selected,
        );
    }

    /// The public compact entry point without the fixed-width dispatch.
    #[allow(clippy::too_many_arguments)]
    fn compact_oracle(
        q: &[f32],
        prefix_k: &[f32],
        generated_k: &[f32],
        v: &[f32],
        query_len: usize,
        prefix_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        query_offset: usize,
        image_start: usize,
        image_end: usize,
        sinks: &[f32],
        output: &mut [f32],
        simd: Simd,
    ) {
        let selected = simd.resolved();
        if query_len >= 4 && selected != Simd::Scalar {
            attention_gemm_compact(
                q,
                prefix_k,
                generated_k,
                v,
                prefix_len,
                n_heads,
                n_kv_heads,
                head_dim,
                query_offset,
                image_start,
                image_end,
                sinks,
                output,
            );
            return;
        }
        attention_compact_online_softmax(
            q,
            prefix_k,
            generated_k,
            v,
            prefix_len,
            n_heads,
            n_kv_heads,
            head_dim,
            query_offset,
            image_start,
            image_end,
            sinks,
            output,
            selected,
        );
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn compare_vectors(a: &[f32], b: &[f32], factor: f32) {
        unsafe {
            assert_eq!(dot64(a, b).to_bits(), x86::dot_avx2(a, b).to_bits());
            let mut actual = a.to_vec();
            let mut expected = a.to_vec();
            axpy64(factor, b, &mut actual);
            x86::axpy_avx2(factor, b, &mut expected);
            exact(&actual, &expected);
        }
    }

    #[test]
    fn vectors_preserve_four_accumulators_and_fma() {
        supported();
        for seed in 1..=128 {
            let a = data(64, seed, if seed % 2 == 0 { 1e10 } else { 0.03125 });
            let b = data(64, seed + 997, if seed % 3 == 0 { 1e-10 } else { 8.0 });
            for factor in [-4.0, -0.0, 0.0, 0.125, 3.25] {
                unsafe { compare_vectors(&a, &b, factor) };
            }
        }
        let a: Vec<_> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        unsafe { compare_vectors(&a, &[1.0; 64], -0.0) };
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_case(
        kv: usize,
        offset: usize,
        image_start: usize,
        image_end: usize,
        query_len: usize,
        width: usize,
        simd: Simd,
        extreme: bool,
        threads: usize,
    ) {
        let heads = 16;
        let q = data(
            query_len * heads * width,
            11,
            if extreme { 100.0 } else { 0.75 },
        );
        let k = data(kv * heads * width, 23, if extreme { 100.0 } else { 0.75 });
        let v = data(kv * heads * width, 37, if extreme { 1e8 } else { 1.0 });
        let sinks: Vec<_> = (0..heads)
            .map(|h| match h % 4 {
                0 => -1000.0,
                1 => 1000.0,
                2 => 0.0,
                _ => 2.25,
            })
            .collect();
        let mut actual = vec![f32::NAN; q.len()];
        let mut expected = actual.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            expanded_oracle(
                &q,
                &k,
                &v,
                query_len,
                heads,
                width,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut expected,
                simd,
            );
            attention_with_simd(
                &q,
                &k,
                &v,
                query_len,
                kv,
                heads,
                width,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut actual,
                simd,
            );
        });
        assert!(actual.iter().all(|x| x.is_finite()));
        exact(&actual, &expected);
    }

    #[test]
    fn causal_tile_tails_are_bit_exact() {
        supported();
        for kv in [
            1, 2, 17, 63, 64, 127, 128, 129, 143, 144, 161, 255, 256, 257, 1025,
        ] {
            compare_case(kv, kv - 1, 0, 0, 1, 64, Simd::Avx2, false, 4);
        }
    }

    #[test]
    fn image_and_causal_boundaries_are_bit_exact() {
        supported();
        for (kv, offset, start, end) in [
            (17, 0, 1, 16),
            (17, 1, 1, 16),
            (17, 15, 1, 16),
            (17, 16, 1, 16),
            (161, 0, 1, 129),
            (161, 1, 1, 129),
            (161, 128, 1, 129),
            (161, 129, 1, 129),
            (161, 160, 1, 129),
        ] {
            compare_case(kv, offset, start, end, 1, 64, Simd::Avx2, false, 4);
        }
    }

    #[test]
    fn extreme_logits_sinks_and_long_context_are_bit_exact() {
        supported();
        compare_case(257, 256, 1, 144, 1, 64, Simd::Avx2, true, 4);
        // Real full-page prefix and the full context endpoint; no timing assertion.
        for kv in [6544, 16384] {
            compare_case(kv, kv - 1, 1, 6540, 1, 64, Simd::Avx2, false, 4);
        }
    }

    #[test]
    fn auto_and_one_thread_are_bit_exact() {
        supported();
        compare_case(161, 160, 1, 129, 1, 64, Simd::Auto, false, 1);
    }

    #[test]
    fn other_shapes_backends_and_prefill_stay_on_generic_path() {
        supported();
        for simd in [Simd::Scalar, Simd::Avx512] {
            if simd.validate().is_ok() {
                compare_case(17, 16, 1, 16, 1, 64, simd, false, 4);
            }
        }
        for width in [31, 63, 65, 80] {
            compare_case(17, 16, 1, 16, 1, width, Simd::Avx2, false, 4);
        }
        for rows in [2, 3, 4, 17] {
            compare_case(17, 17 - rows, 1, 16, rows, 64, Simd::Avx2, false, 4);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_case(
        total: usize,
        prefix: usize,
        offset: usize,
        image_start: usize,
        image_end: usize,
        rows: usize,
        heads: usize,
        kvheads: usize,
        width: usize,
        simd: Simd,
        extreme: bool,
    ) {
        let qw = heads * width;
        let kw = kvheads * width;
        let repeat = heads / kvheads;
        let q = data(rows * qw, 41, if extreme { 100.0 } else { 0.75 });
        let pk = data(prefix * qw, 53, if extreme { 100.0 } else { 0.75 });
        let gk = data(
            (total - prefix) * kw,
            67,
            if extreme { 100.0 } else { 0.75 },
        );
        let v = data(total * kw, 79, if extreme { 1e8 } else { 1.0 });
        let sinks: Vec<_> = (0..heads)
            .map(|h| match h % 4 {
                0 => -1000.0,
                1 => 1000.0,
                2 => 0.0,
                _ => 2.25,
            })
            .collect();
        let mut expected = vec![f32::NAN; q.len()];
        let mut actual = expected.clone();
        let mut expanded_out = expected.clone();
        let mut expanded_k = vec![0.0; total * qw];
        let mut expanded_v = expanded_k.clone();
        for token in 0..total {
            for head in 0..heads {
                let target = token * qw + head * width;
                let key = if token < prefix {
                    &pk[target..target + width]
                } else {
                    let source = (token - prefix) * kw + (head / repeat) * width;
                    &gk[source..source + width]
                };
                expanded_k[target..target + width].copy_from_slice(key);
                let source = token * kw + (head / repeat) * width;
                expanded_v[target..target + width].copy_from_slice(&v[source..source + width]);
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            compact_oracle(
                &q,
                &pk,
                &gk,
                &v,
                rows,
                prefix,
                heads,
                kvheads,
                width,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut expected,
                simd,
            );
            attention_compact_with_simd(
                &q,
                &pk,
                &gk,
                &v,
                rows,
                prefix,
                total,
                heads,
                kvheads,
                width,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut actual,
                simd,
            );
            expanded_oracle(
                &q,
                &expanded_k,
                &expanded_v,
                rows,
                heads,
                width,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut expanded_out,
                simd,
            );
        });
        assert!(actual.iter().all(|x| x.is_finite()));
        exact(&actual, &expected);
        exact(&actual, &expanded_out);
    }

    #[test]
    fn compact_zero_full_and_tile_crossovers_are_bit_exact() {
        supported();
        for total in [1, 17, 127, 128, 129, 257] {
            for prefix in [0, total / 2, total] {
                compact_case(
                    total,
                    prefix,
                    total - 1,
                    0,
                    0,
                    1,
                    16,
                    8,
                    64,
                    Simd::Avx2,
                    false,
                );
            }
        }
        for prefix in [127, 128, 129] {
            compact_case(257, prefix, 256, 1, prefix, 1, 16, 8, 64, Simd::Avx2, false);
        }
    }

    #[test]
    fn compact_image_and_generated_boundaries_are_bit_exact() {
        supported();
        for offset in [0, 1, 126, 127, 128, 129, 160] {
            compact_case(161, 129, offset, 1, 128, 1, 16, 8, 64, Simd::Avx2, false);
        }
    }

    #[test]
    fn compact_real_prefix_full_context_and_extremes_are_bit_exact() {
        supported();
        for (total, prefix) in [(6544, 6544), (6545, 6544), (16384, 6544)] {
            compact_case(
                total,
                prefix,
                total - 1,
                1,
                6540,
                1,
                16,
                8,
                64,
                Simd::Avx2,
                false,
            );
        }
        compact_case(257, 129, 256, 1, 129, 1, 16, 8, 64, Simd::Avx2, true);
    }

    #[test]
    fn compact_gqa_repeats_and_auto_are_bit_exact() {
        supported();
        for kvheads in [4, 8, 16] {
            compact_case(257, 129, 256, 1, 128, 1, 16, kvheads, 64, Simd::Auto, false);
        }
    }

    #[test]
    fn compact_other_shapes_and_prefill_stay_on_generic_path() {
        supported();
        for width in [31, 63, 65, 80] {
            compact_case(17, 9, 16, 1, 8, 1, 16, 8, width, Simd::Avx2, false);
        }
        for rows in [2, 3, 4, 17] {
            compact_case(17, 9, 17 - rows, 1, 8, rows, 16, 8, 64, Simd::Avx2, false);
        }
        for simd in [Simd::Scalar, Simd::Avx512] {
            if simd.validate().is_ok() {
                compact_case(17, 9, 16, 1, 8, 1, 16, 8, 64, simd, false);
            }
        }
    }
}
