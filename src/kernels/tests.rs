use super::*;

// A fixed, deterministic generator keeps tests independent of rand versions.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn implementations() -> Vec<Simd> {
    [Simd::Scalar, Simd::Auto, Simd::Avx2]
        .into_iter()
        .filter(|simd| simd.validate().is_ok())
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *b).abs() <= absolute + relative * b.abs(),
            "index {i}: actual={a}, expected={b}, delta={}",
            (*a - *b).abs()
        );
    }
}

fn linear_f64(input: &[f32], rows: usize, in_dim: usize, weights: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * out_dim];
    for row in 0..rows {
        for channel in 0..out_dim {
            let mut sum = 0.0_f64;
            for inner in 0..in_dim {
                sum += input[row * in_dim + inner] as f64 * weights[channel * in_dim + inner] as f64;
            }
            out[row * out_dim + channel] = sum as f32;
        }
    }
    out
}

#[test]
fn fused_glu_is_bitwise_linear_then_gate() {
    // Real FFN widths plus odd widths; gates include NaN, signed zeros and
    // negatives so every branch of the squared-ReLU expression is exercised.
    for (in_dim, ffn_dim) in [(768, 2304), (65, 19), (1, 1)] {
        let mut weights = values(2 * ffn_dim * in_dim, 7);
        for (i, w) in weights.iter_mut().enumerate().step_by(97) {
            *w = [f32::NAN, 0.0, -0.0, -3.0][i % 4];
        }
        for simd in implementations() {
            for rows in 1..=8 {
                let input = values(rows * in_dim, 11 + rows as u32);
                let mut packed = vec![0.0; rows * 2 * ffn_dim];
                linear_with_simd(&input, rows, in_dim, &weights, 2 * ffn_dim, &mut packed, simd);
                let mut expected = vec![0.0; rows * ffn_dim];
                squared_relu_gate(&packed, &mut expected);
                let mut fused = vec![f32::INFINITY; rows * ffn_dim];
                assert!(linear_glu_with_simd(
                    &input, rows, in_dim, &weights, ffn_dim, &mut fused, simd
                ));
                for (a, b) in fused.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{simd:?} rows {rows} in {in_dim}");
                }
            }
        }
    }
    let mut unused = vec![0.0; 9];
    assert!(!linear_glu_with_simd(
        &values(9 * 4, 1),
        9,
        4,
        &values(2 * 4, 2),
        1,
        &mut unused,
        Simd::Scalar
    ));
}

#[test]
fn serial_rms_norm_rows_match_parallel_rows() {
    // Small inputs run serially; the same rows inside a large (parallel)
    // input must normalize to identical bits.
    for width in [64, 768] {
        let small_rows = RMS_NORM_SERIAL_ELEMENTS / width;
        let large_rows = small_rows * 3;
        let large = values(large_rows * width, 5);
        let affine = values(width, 9);
        for weight in [None, Some(affine.as_slice())] {
            let mut parallel = vec![0.0; large.len()];
            rms_norm(&large, &mut parallel, width, 1e-5, weight);
            let small = &large[..small_rows * width];
            let mut serial = vec![0.0; small.len()];
            rms_norm(small, &mut serial, width, 1e-5, weight);
            for (a, b) in serial.iter().zip(&parallel) {
                assert_eq!(a.to_bits(), b.to_bits(), "width {width}");
            }
        }
    }
}

#[test]
fn linear_layout_and_overwrite() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = [1.0, 10.0, 100.0, -1.0, -2.0, -3.0];
    for simd in implementations() {
        let mut output = [f32::NAN; 4];
        linear_with_simd(&input, 2, 3, &weights, 2, &mut output, simd);
        assert_eq!(output, [321.0, -14.0, 654.0, -32.0]);
    }
}

#[test]
fn linear_real_model_dimensions_and_tails() {
    // Width 768, attention width 1024, projector input 16*16*3, and
    // nonmultiples exercise both actual model and vector-tail dimensions.
    for (rows, in_dim, out_dim) in [
        (1, 768, 1024),
        (1, 1024, 768),
        (2, 768, 128),
        (4, 768, 128),
        (8, 768, 128),
        (3, 768, 97),
        (17, 67, 129),
        (1, 79, 53),
        (1, 7, 3),
    ] {
        let input = values(rows * in_dim, 61);
        let weights = values(out_dim * in_dim, 19);
        let expected = linear_f64(&input, rows, in_dim, &weights, out_dim);
        for simd in implementations() {
            let mut output = vec![0.0; rows * out_dim];
            linear_with_simd(&input, rows, in_dim, &weights, out_dim, &mut output, simd);
            // FP32 sequential reductions are intentionally included;
            // these are numerical unit bounds, not frozen GPU tolerances.
            assert_close(&output, &expected, 6e-5, 3e-6);
        }
    }
}

#[test]
fn linear_empty_dimensions() {
    linear(&[], 0, 17, &[0.0; 51], 3, &mut []);
    let mut output = [f32::NAN; 12];
    linear(&[], 4, 0, &[], 3, &mut output);
    assert_eq!(output, [0.0; 12]);
    linear(&[1.0; 6], 2, 3, &[], 0, &mut []);
}

#[test]
#[should_panic(expected = "linear weight shape")]
fn linear_rejects_mismatched_weight_shape() {
    linear(&[1.0, 2.0], 1, 2, &[3.0], 1, &mut [0.0]);
}

#[test]
fn norm_matches_independent_f64_reference() {
    for width in [1, 7, 64, 768] {
        let input = values(3 * width, 12);
        let weights = values(width, 102);
        for affine in [None, Some(weights.as_slice())] {
            let mut output = vec![0.0; input.len()];
            rms_norm(&input, &mut output, width, f32::EPSILON, affine);
            let mut reference = vec![0.0; input.len()];
            for row in 0..3 {
                let norm = (input[row * width..(row + 1) * width]
                    .iter()
                    .map(|v| (*v as f64).powi(2))
                    .sum::<f64>()
                    / width as f64
                    + f32::EPSILON as f64)
                    .sqrt();
                for col in 0..width {
                    let w = affine.map_or(1.0, |w| w[col]) as f64;
                    reference[row * width + col] = (input[row * width + col] as f64 / norm * w) as f32;
                }
            }
            assert_close(&output, &reference, 2e-6, 2e-6);
        }
    }
    let mut zeros = [1.0; 64];
    rms_norm(&[0.0; 64], &mut zeros, 64, f32::EPSILON, None);
    assert_eq!(zeros, [0.0; 64]);
}

#[test]
fn glu_is_interleaved_and_relu_is_squared() {
    let mut output = [0.0; 4];
    squared_relu_gate(&[2.0, 3.0, -4.0, 5.0, 0.5, -8.0, 0.0, 12.0], &mut output);
    assert_eq!(output, [12.0, 0.0, -2.0, 0.0]);
    squared_relu_gate(&[f32::NAN, 1.0, 1.0, f32::NAN, -1.0, 2.0, 2.0, 0.0], &mut output);
    assert_eq!(output[0], 0.0);
    assert!(output[1].is_nan());
    assert_eq!(&output[2..], &[0.0, 0.0]);
}

#[allow(clippy::too_many_arguments)]
fn attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qlen: usize,
    kvlen: usize,
    heads: usize,
    dim: usize,
    offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; qlen * heads * dim];
    for query in 0..qlen {
        let absolute = offset + query;
        for head in 0..heads {
            // Dense logits and explicit mask are intentionally independent
            // from the production contiguous-visible-range/online algorithm.
            let mut scores = vec![f64::NEG_INFINITY; kvlen];
            for key in 0..kvlen {
                let allowed = key <= absolute
                    || (absolute >= image_start && absolute < image_end && key >= image_start && key < image_end);
                if !allowed {
                    continue;
                }
                let mut score = 0.0_f64;
                for d in 0..dim {
                    score += q[(query * heads + head) * dim + d] as f64 * k[(key * heads + head) * dim + d] as f64;
                }
                scores[key] = score / (dim as f64).sqrt();
            }
            let max = scores.iter().copied().fold(sinks[head] as f64, f64::max);
            let denom = (sinks[head] as f64 - max).exp() + scores.iter().map(|s| (s - max).exp()).sum::<f64>();
            for d in 0..dim {
                out[(query * heads + head) * dim + d] = (scores
                    .iter()
                    .enumerate()
                    .map(|(key, s)| (s - max).exp() / denom * v[(key * heads + head) * dim + d] as f64)
                    .sum::<f64>()) as f32;
            }
        }
    }
    out
}

#[test]
fn attention_matches_dense_for_hybrid_prefill_cached_decode_and_tails() {
    for (qlen, kvlen, heads, dim, offset, start, end) in [
        (7, 7, 2, 7, 0, 1, 5),
        (5, 5, 3, 17, 0, 0, 0),
        (1, 267, 16, 64, 266, 1, 201),
        (3, 260, 2, 33, 257, 2, 193),
        (33, 267, 2, 64, 234, 2, 252),
        (133, 133, 2, 64, 0, 16, 98),
        (129, 129, 2, 8, 0, 0, 129),
        (1, 1, 1, 1, 0, 0, 0),
    ] {
        let q = values(qlen * heads * dim, 71);
        let k = values(kvlen * heads * dim, 83);
        let v = values(kvlen * heads * dim, 117);
        let sinks = values(heads, 24);
        let expected = attention_f64(&q, &k, &v, qlen, kvlen, heads, dim, offset, start, end, &sinks);
        for simd in implementations() {
            let mut out = vec![f32::NAN; q.len()];
            attention_with_simd(
                &q, &k, &v, qlen, kvlen, heads, dim, offset, start, end, &sinks, &mut out, simd,
            );
            assert_close(&out, &expected, 2e-6, 4e-6);
        }
    }
}

#[test]
fn attention_image_boundaries_and_sink_value_are_correct() {
    // All real logits and the sink logit are zero. Image positions 1 and 2
    // see three real keys; BOS cannot look forward; img_end is causal.
    let qk = [0.0; 5];
    let v = [1.0, 10.0, 100.0, 1000.0, 10000.0];
    let mut out = [0.0; 5];
    attention(&qk, &qk, &v, 5, 5, 1, 1, 0, 1, 3, &[0.0], &mut out);
    assert_close(&out, &[0.5, 27.75, 27.75, 222.2, 1851.8334], 1e-6, 1e-7);
    let mut no_sink = [0.0];
    attention(
        &[0.0],
        &[0.0],
        &[13.0],
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        &[f32::NEG_INFINITY],
        &mut no_sink,
    );
    assert_eq!(no_sink, [13.0]);
}

#[test]
fn attention_stable_with_large_positive_and_negative_logits_and_sinks() {
    let keys: Vec<f32> = (0..257).map(|i| if i == 140 { 1010.0 } else { -1000.0 }).collect();
    let values: Vec<f32> = (0..257).map(|i| i as f32 * 0.1 - 3.0).collect();
    for sink in [-10000.0, 10000.0, 1009.0, f32::NEG_INFINITY] {
        let expected = attention_f64(&[1.0], &keys, &values, 1, 257, 1, 1, 256, 0, 0, &[sink]);
        for simd in implementations() {
            let mut out = [f32::NAN];
            attention_with_simd(&[1.0], &keys, &values, 1, 257, 1, 1, 256, 0, 0, &[sink], &mut out, simd);
            assert_close(&out, &expected, 1e-6, 1e-6);
            assert!(out[0].is_finite());
        }
    }
    // Exercise rescaling through the separate multiquery GEMM prefill.
    let q = [1.0; 4];
    let expected = attention_f64(&q, &keys, &values, 4, 257, 1, 1, 253, 0, 0, &[-10000.0]);
    let mut out = [f32::NAN; 4];
    attention(&q, &keys, &values, 4, 257, 1, 1, 253, 0, 0, &[-10000.0], &mut out);
    assert_close(&out, &expected, 1e-6, 1e-6);
}

#[test]
fn parallelism_uses_callers_pool_without_changing_results() {
    let input = values(17 * 768, 191);
    let weights = values(128 * 768, 122);
    let expected = linear_f64(&input, 17, 768, &weights, 128);
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut out = vec![0.0; expected.len()];
        pool.install(|| {
            assert_eq!(rayon::current_num_threads(), threads);
            linear(&input, 17, 768, &weights, 128, &mut out);
        });
        assert_close(&out, &expected, 6e-5, 3e-6);
    }
}

#[allow(clippy::too_many_arguments)]
fn expand_compact_cache(
    prefix: &[f32],
    generated: &[f32],
    values: &[f32],
    prefix_len: usize,
    total_len: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut keys = vec![0.0; total_len * heads * dim];
    let mut expanded_values = vec![0.0; keys.len()];
    keys[..prefix.len()].copy_from_slice(prefix);
    for token in 0..total_len {
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let dst = (token * heads + head) * dim;
            let value_src = (token * kv_heads + kv_head) * dim;
            expanded_values[dst..dst + dim].copy_from_slice(&values[value_src..value_src + dim]);
            if token >= prefix_len {
                let src = ((token - prefix_len) * kv_heads + kv_head) * dim;
                keys[dst..dst + dim].copy_from_slice(&generated[src..src + dim]);
            }
        }
    }
    (keys, expanded_values)
}

#[test]
fn compact_cache_is_bit_identical_to_expanded_for_every_vector_backend() {
    for (queries, prefix_len, total_len, heads, kv_heads, dim, offset, image_start, image_end) in [
        (7, 7, 7, 2, 1, 7, 0, 1, 5),
        (144, 144, 144, 16, 8, 64, 0, 0, 133),
        (137, 137, 137, 16, 8, 64, 0, 2, 129),
        (1, 144, 145, 16, 8, 64, 144, 0, 133),
        (4, 137, 141, 16, 8, 64, 137, 1, 129),
        (33, 129, 267, 4, 2, 33, 234, 1, 126),
        (9, 0, 9, 4, 2, 17, 0, 0, 0),
        (2, 257, 260, 4, 1, 7, 258, 0, 249),
        (1, 128, 4099, 16, 8, 64, 4098, 0, 120),
        (1, 127, 16384, 16, 8, 64, 16383, 1, 119),
    ] {
        let q = values(queries * heads * dim, 71);
        // Prefix heads are intentionally independent. Collapsing paired
        // spatial keys would visibly change the expected output.
        let prefix = values(prefix_len * heads * dim, 83);
        let generated = values((total_len - prefix_len) * kv_heads * dim, 37);
        let v = values(total_len * kv_heads * dim, 117);
        let sinks = values(heads, 24);
        let (expanded_k, expanded_v) =
            expand_compact_cache(&prefix, &generated, &v, prefix_len, total_len, heads, kv_heads, dim);
        for simd in implementations() {
            let mut expected = vec![f32::NAN; q.len()];
            let mut actual = vec![f32::NAN; q.len()];
            attention_with_simd(
                &q,
                &expanded_k,
                &expanded_v,
                queries,
                total_len,
                heads,
                dim,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut expected,
                simd,
            );
            attention_compact_with_simd(
                &q,
                &prefix,
                &generated,
                &v,
                queries,
                prefix_len,
                total_len,
                heads,
                kv_heads,
                dim,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut actual,
                simd,
            );
            for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "compact mismatch at {index}: {a} versus {e}; {simd:?}, Q={queries}, prefix={prefix_len}, total={total_len}, heads={heads}, dim={dim}"
                );
            }
        }
    }
}

#[test]
fn compact_cache_extreme_sinks_preserve_expanded_rounding() {
    let prefix_len = 3;
    let total_len = 133;
    let q = [1.0; 4 * 2];
    let prefix = vec![1000.0; prefix_len * 2];
    let mut generated = vec![-1000.0; total_len - prefix_len];
    generated[127] = 1010.0;
    let v = values(total_len, 131);
    let (kfull, vfull) = expand_compact_cache(&prefix, &generated, &v, prefix_len, total_len, 2, 1, 1);
    for simd in implementations() {
        for sinks in [[1009.0, -10000.0], [f32::NEG_INFINITY, 10000.0]] {
            let mut actual = [f32::NAN; 8];
            let mut expected = [f32::NAN; 8];
            attention_with_simd(
                &q,
                &kfull,
                &vfull,
                4,
                total_len,
                2,
                1,
                129,
                0,
                2,
                &sinks,
                &mut expected,
                simd,
            );
            attention_compact_with_simd(
                &q,
                &prefix,
                &generated,
                &v,
                4,
                prefix_len,
                total_len,
                2,
                1,
                1,
                129,
                0,
                2,
                &sinks,
                &mut actual,
                simd,
            );
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}

#[test]
#[should_panic(expected = "compact attention image must be inside prefix")]
fn compact_cache_rejects_spatial_keys_in_generated_region() {
    attention_compact(
        &[0.0; 2],
        &[0.0; 2],
        &[0.0],
        &[0.0; 2],
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        0,
        2,
        &[0.0; 2],
        &mut [0.0; 2],
    );
}
