//! Included only in the copied model module; no weights or inference required.
use super::*;
use crate::temporal_candidate::tests::{backends, bits_equal, operands};

#[test]
fn model_cache_dispatch_prefill_decode_matches_compact() {
    let c: ModelConfig =
        serde_json::from_str(include_str!("../tests/fixtures/model-config.json")).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        for simd in backends() {
            for prefix in [17, 129] {
                let (k, v) = operands(prefix, 3);
                let mut compact = Session::new(
                    &c,
                    prefix + 3,
                    prefix,
                    1,
                    prefix - 1,
                    simd,
                    CacheLayout::Compact,
                )
                .unwrap();
                let mut candidate = Session::new(
                    &c,
                    prefix + 3,
                    prefix,
                    1,
                    prefix - 1,
                    simd,
                    CacheLayout::TemporalCandidate,
                )
                .unwrap();
                let q: Vec<_> = (0..prefix * 1024)
                    .map(|i| ((i * 11 % 97) as i32 - 48) as f32 / 43.)
                    .collect();
                let mut a = vec![0.; q.len()];
                let mut b = a.clone();
                for (offset, rows) in [(0, prefix), (prefix, 1), (prefix + 1, 1), (prefix + 2, 1)] {
                    let range = offset * 1024..(offset + rows) * 1024;
                    for cache in [&mut compact.layers[0], &mut candidate.layers[0]] {
                        cache
                            .append(&k[range.clone()], &v[range.clone()], offset, &c)
                            .unwrap();
                    }
                    compact.layers[0]
                        .attention(
                            &q[..rows * 1024],
                            &k[range.clone()],
                            rows,
                            offset + rows,
                            &c,
                            offset,
                            1,
                            prefix - 1,
                            &[0.; 16],
                            &mut a[..rows * 1024],
                            simd,
                        )
                        .unwrap();
                    candidate.layers[0]
                        .attention(
                            &q[..rows * 1024],
                            &k[range.clone()],
                            rows,
                            offset + rows,
                            &c,
                            offset,
                            1,
                            prefix - 1,
                            &[0.; 16],
                            &mut b[..rows * 1024],
                            simd,
                        )
                        .unwrap();
                    bits_equal(&a[..rows * 1024], &b[..rows * 1024]);
                }
            }
        }
    });
}

#[test]
fn model_candidate_is_explicit_default_unchanged() {
    assert_eq!(CacheLayout::default(), CacheLayout::Expanded);
    assert_eq!(
        serde_json::to_string(&CacheLayout::TemporalCandidate).unwrap(),
        "\"temporal_candidate\""
    );
    let mut c: ModelConfig =
        serde_json::from_str(include_str!("../tests/fixtures/model-config.json")).unwrap();
    c.head_dim = 32;
    assert!(
        Session::new(
            &c,
            20,
            17,
            1,
            16,
            kernels::Simd::Scalar,
            CacheLayout::TemporalCandidate
        )
        .is_err()
    );
}
