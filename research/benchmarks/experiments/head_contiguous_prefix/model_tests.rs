//! Private copied-model cache dispatch checks; no weights or model generation.
use super::*;
use crate::head_contiguous_prefix::{pack_prefill_values, tests::{backends, bits_equal, operands}};

#[test]
fn model_head_prefix_prefill_then_decode_matches_unchanged_compact() {
    let c: ModelConfig = serde_json::from_str(include_str!("../tests/fixtures/model-config.json")).unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
        for prefix in [17, 129] {
            let (k, v) = operands(prefix, 3);
            for simd in backends() {
                let mut compact = Session::new(&c, prefix + 3, prefix, 1, prefix - 1,
                    simd, CacheLayout::Compact).unwrap();
                let mut candidate = Session::new(&c, prefix + 3, prefix, 1, prefix - 1,
                    simd, CacheLayout::HeadContiguousPrefix).unwrap();
                assert!(matches!(&compact.layers[0], LayerCache::Compact { .. }));
                assert!(matches!(&candidate.layers[0], LayerCache::HeadContiguousPrefix(_)));
                let q: Vec<_> = (0..prefix * 1024)
                    .map(|i| ((i * 11 % 97) as i32 - 48) as f32 / 43.).collect();
                let mut expected = vec![0.; q.len()]; let mut actual = expected.clone();
                for (offset, rows) in [(0, prefix), (prefix, 1), (prefix + 1, 1), (prefix + 2, 1)] {
                    let range = offset * 1024..(offset + rows) * 1024;
                    compact.layers[0].append(&k[range.clone()], &v[range.clone()], offset, &c).unwrap();
                    candidate.layers[0].append(&k[range.clone()], &v[range.clone()], offset, &c).unwrap();
                    // This local Vec is dropped after this call; generated decoding
                    // cannot rely on any persistent token-major prefix mirror.
                    let mut scratch = Vec::with_capacity(if offset == 0 {prefix * 512} else {0});
                    if offset == 0 { pack_prefill_values(&v[range.clone()], &mut scratch).unwrap(); }
                    compact.layers[0].attention(&q[..rows * 1024], &k[range.clone()], &[],
                        rows, offset + rows, &c, offset, 1, prefix - 1, &[0.; 16],
                        &mut expected[..rows * 1024], simd).unwrap();
                    candidate.layers[0].attention(&q[..rows * 1024], &k[range.clone()], &scratch,
                        rows, offset + rows, &c, offset, 1, prefix - 1, &[0.; 16],
                        &mut actual[..rows * 1024], simd).unwrap();
                    bits_equal(&actual[..rows * 1024], &expected[..rows * 1024]);
                }
            }
        }
    });
}

#[test]
fn model_head_prefix_enum_is_explicit_and_invalid_sessions_reject() {
    assert_eq!(CacheLayout::default(), CacheLayout::Expanded);
    assert_eq!(serde_json::to_string(&CacheLayout::Compact).unwrap(), "\"compact\"");
    assert_eq!(serde_json::to_string(&CacheLayout::HeadContiguousPrefix).unwrap(), "\"head_contiguous_prefix\"");
    let c: ModelConfig = serde_json::from_str(include_str!("../tests/fixtures/model-config.json")).unwrap();
    for (prefix, capacity) in [(0, 4), (17, 16)] {
        assert!(Session::new(&c, capacity, prefix, 0, 0, kernels::Simd::Scalar,
                            CacheLayout::HeadContiguousPrefix).is_err());
    }
    let mut invalid = c;
    invalid.head_dim = 32;
    assert!(Session::new(&invalid, 20, 17, 1, 16, kernels::Simd::Scalar,
                        CacheLayout::HeadContiguousPrefix).is_err());
}
