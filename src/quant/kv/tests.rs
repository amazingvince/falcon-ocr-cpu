//! Bitwise tests of the split cache against the compact kernel over its
//! decoded values, and of its codes, chunking and verification rows.
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
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
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
                k[(t * c.n_heads + h) * 64 + d] = ((t * 31 + identity * 17 + d * 7 + seed) % 101) as f32 / 151.0 - 0.3;
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
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
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
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
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
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
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
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
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
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let mut k = vec![0.0; c.query_dim()];
    k[64] = 1.0;
    assert!(SplitPrefix::from_compact(&k, &vec![0.0; c.kv_dim()], 1, 2, &c, Kv::F32Split, None).is_err());
}
