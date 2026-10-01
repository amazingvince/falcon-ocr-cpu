//! Bitwise tests of the split cache against the compact kernel over its
//! decoded values, and of its codes, chunking and verification rows.
use super::{decode::RecordStore, *};

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
    // The rotated research caches follow the 8-bit one.
    assert!(default_fast_exp(Kv::Q8Rot, true, None) && default_fast_exp(Kv::Q4Rot, true, None));
    assert!(!default_fast_exp(Kv::Q8Rot, false, None) && !default_fast_exp(Kv::Q4Rot, false, None));
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
        for (storage, fast_exp) in storages().into_iter().flat_map(|s| [(s, false), (s, true)]) {
            for chunks in [1, 2, 4] {
                for backend in backends() {
                    let mut cache = SplitPrefix::seal(&k, &v, p, p + generated, &c, storage, None)
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
                                "{storage:?} fast exp {fast_exp} chunks {chunks} {backend:?} p{p} row {r}"
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

/// Every element format, unrotated and rotated: the profiles' storages
/// (`Storage::of`) and the others, which isolate what rotation changes.
fn storages() -> Vec<Storage> {
    [Format::F32, Format::Q16, Format::Q8, Format::Q4]
        .into_iter()
        .flat_map(|format| [false, true].map(|rotated| Storage { format, rotated }))
        .collect()
}

const F32_PLAIN: Storage = Storage {
    format: Format::F32,
    rotated: false,
};

/// `q` in the basis of `storage`'s records (`in_storage_basis`).
fn stored_queries(storage: Storage, q: &[f32]) -> Vec<f32> {
    let mut q = q.to_vec();
    if storage.rotated {
        each_block(&mut q, &KEY_HALVES, rotation::rotate_query);
    }
    q
}

/// Decode output of `storage` over a prefix and generated positions.
#[allow(clippy::too_many_arguments)]
fn decode(
    c: &ModelConfig,
    storage: Storage,
    k: &[f32],
    v: &[f32],
    tail_k: &[f32],
    tail_v: &[f32],
    q: &[f32],
    sinks: &[f32],
) -> Vec<f32> {
    let (p, generated) = (k.len() / c.query_dim(), tail_k.len() / c.kv_dim());
    let mut cache = SplitPrefix::seal(k, v, p, p + generated, c, storage, None).unwrap();
    push_rows(&mut cache, c, tail_k, tail_v);
    let mut output = vec![f32::NAN; c.query_dim()];
    cache.attention_decode(q, p + generated, sinks, &mut output, Simd::Auto);
    output
}

#[test]
fn records_decode_bitwise_like_compact_over_their_values() {
    // FP32 records must equal the compact kernel on the original cache;
    // coded records must equal it on their own dequantized values. For
    // rotated records rotation is only a pre/post transform: the compact
    // kernel runs on the dequantized rotated values with the rotated
    // query, and its output is un-rotated.
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
        for storage in storages() {
            let mut cache = SplitPrefix::seal(&k, &v, p, p + generated, &c, storage, None)
                .unwrap()
                .with_chunks(1);
            push_rows(&mut cache, &c, &tail_k, &tail_v);
            let (dk, dv) = cache.dequantized_compact();
            let (tk, tv) = dequantized_tail(&cache);
            if storage.format == Format::F32 {
                // Exactly the (rotated) values: every block with its kind.
                let rotated = |values: &[f32], kinds: &[Block]| {
                    let mut values = values.to_vec();
                    if storage.rotated {
                        each_block(&mut values, kinds, rotation::rotate);
                    }
                    values
                };
                assert_eq!(dk, rotated(&k, &KEY_HALVES));
                assert_eq!(dv, rotated(&v, &VALUE_HALVES));
                assert_eq!(tk, rotated(&tail_k, &KEY_HALVES));
                assert_eq!(tv, rotated(&tail_v, &VALUE_HALVES));
            }
            for backend in backends() {
                let rq = stored_queries(storage, &q);
                let mut expected = compact_reference(&c, &rq, &dk, &dv, &tk, &tv, p, &sinks, backend);
                if storage.rotated {
                    each_block(&mut expected, &VALUE_HALVES, rotation::unrotate);
                }
                let mut output = vec![f32::NAN; c.query_dim()];
                cache.attention_decode(&q, p + generated, &sinks, &mut output, backend);
                for (a, b) in output.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{storage:?} {backend:?} p{p}");
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
    for storage in storages() {
        let mut reference = vec![0.0; c.query_dim()];
        for chunks in 1..=MAX_CHUNKS {
            let mut cache = SplitPrefix::seal(&k, &v, p, p + generated, &c, storage, None)
                .unwrap()
                .with_chunks(chunks);
            push_rows(&mut cache, &c, &tail, &tail);
            let mut output = vec![0.0; c.query_dim()];
            cache.attention_decode(&q, p + generated, &sinks, &mut output, Simd::Auto);
            if chunks == 1 {
                reference = output;
            } else {
                for (a, b) in output.iter().zip(&reference) {
                    assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{storage:?} {chunks}");
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
    // Worst output errors measured: Q8 2.4e-4, Q16 1.0e-6, Q8Rot 5.7e-4
    // (this fixture's blocks are flatter than Gaussian, which rotation
    // does not help) and Q4Rot 2.0e-2 (a step 18 times the 8-bit one).
    for (mode, bound) in [(Kv::Q8, 0.02), (Kv::Q16, 0.02), (Kv::Q8Rot, 0.02), (Kv::Q4Rot, 0.05)] {
        let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 1, &c, mode, None).unwrap();
        push_rows(&mut cache, &c, &tail, &tail);
        let mut output = vec![0.0; c.query_dim()];
        cache.attention_decode(&q, p + 1, &sinks, &mut output, Simd::Auto);
        assert!(output.iter().all(|x| x.is_finite()));
        assert!(
            output.iter().zip(&expected).all(|(a, b)| (a - b).abs() < bound),
            "{mode:?}"
        );
    }
}

#[test]
fn nonfinite_tail_values_reach_the_output() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let p = 130;
    let (k, v, q) = fixture(&c, p, 5);
    let sinks = vec![0.5; c.n_heads];
    for storage in storages() {
        let mut cache = SplitPrefix::seal(&k, &v, p, p + 2, &c, storage, None).unwrap();
        let mut tail = vec![0.01; c.kv_dim()];
        push_rows(&mut cache, &c, &tail, &tail);
        tail[3] = f32::NAN;
        push_rows(&mut cache, &c, &tail, &tail);
        for backend in backends() {
            let mut output = vec![0.0; c.query_dim()];
            cache.attention_decode(&q, p + 2, &sinks, &mut output, backend);
            assert!(output[..64].iter().all(|x| x.is_nan()), "{storage:?} {backend:?}");
        }
    }
}

#[test]
fn truncated_tails_take_new_positions_like_fresh_ones() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let (p, generated, kept) = (37, 6, 2);
    let (k, v, q) = fixture(&c, p, p + 5);
    let row = |seed: usize| -> Vec<f32> {
        (0..generated * c.kv_dim())
            .map(|i| ((i * seed) % 43) as f32 / 59.0 - 0.35)
            .collect()
    };
    let (old_k, old_v, new_k, new_v) = (row(13), row(17), row(19), row(23));
    let split = kept * c.kv_dim();
    let fresh_k = [&old_k[..split], &new_k[split..]].concat();
    let fresh_v = [&old_v[..split], &new_v[split..]].concat();
    let sinks: Vec<_> = (0..c.n_heads).map(|h| h as f32 * 0.2 - 0.9).collect();
    for storage in storages() {
        // Rejected drafts truncate the tail; the next positions overwrite
        // every code and scale of the records they reuse.
        let mut reused = SplitPrefix::seal(&k, &v, p, p + generated, &c, storage, None).unwrap();
        push_rows(&mut reused, &c, &old_k, &old_v);
        reused.truncate_tail(kept);
        push_rows(&mut reused, &c, &new_k[split..], &new_v[split..]);
        let mut output = vec![f32::NAN; c.query_dim()];
        reused.attention_decode(&q, p + generated, &sinks, &mut output, Simd::Auto);
        let expected = decode(&c, storage, &k, &v, &fresh_k, &fresh_v, &q, &sinks);
        assert_eq!(
            output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{storage:?}"
        );
    }
}

#[test]
fn refuses_nonidentical_temporal_keys() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let mut k = vec![0.0; c.query_dim()];
    let v = vec![0.0; c.kv_dim()];
    k[64] = 1.0;
    assert!(SplitPrefix::from_compact(&k, &v, 1, 2, &c, Kv::F32Split, None).is_err());
    // Rotated storage checks the pair on the unrotated keys.
    assert!(SplitPrefix::from_compact(&k, &v, 1, 2, &c, Kv::Q8Rot, None).is_err());
    k[64] = 0.0;
    assert!(SplitPrefix::from_compact(&k, &v, 1, 2, &c, Kv::Q8Rot, None).is_ok());
    assert!(SplitPrefix::from_compact(&k, &v, 1, 2, &c, Kv::Compact, None).is_err());
}

#[test]
fn q4_codes_pack_two_per_byte_and_decode_like_q8() {
    // Every code survives the nibble round trip, sign-extended.
    for code in -8_i8..=7 {
        let nibble_bits = (code as u8) & 0x0F;
        assert_eq!(nibble(nibble_bits, 0), code);
        assert_eq!(nibble((nibble_bits << 4) | 0x0F, 1), code);
    }
    // A block of exact multiples of its scale: codes -7..=7 and back.
    let values: Vec<f32> = (0..32).map(|i| ((i % 15) as f32 - 7.0) * 0.5).collect();
    let mut codes = [0_u8; 16];
    let scale = bf16_f32(q4_block(&values, &mut codes));
    assert_eq!(scale, 0.5);
    let decoded: Vec<f32> = (0..32).map(|i| nibble(codes[i / 2], i % 2) as f32 * scale).collect();
    assert_eq!(decoded, values);
    // A non-finite value poisons the block; an all-zero block is zero.
    let mut poisoned = values.clone();
    poisoned[9] = f32::INFINITY;
    assert!(bf16_f32(q4_block(&poisoned, &mut codes)).is_nan());
    assert_eq!(q4_block(&[0.0; 32], &mut codes), 0);
    assert_eq!(codes, [0; 16]);
    // The vector load decodes like the scalar one on every instruction set.
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let (k, v, _) = fixture(&c, 3, 11);
    let storage = Storage {
        format: Format::Q4,
        rotated: false,
    };
    let cache = SplitPrefix::seal(&k, &v, 3, 3, &c, storage, None).unwrap();
    let Records::Q4 { codes, scales } = &cache.records else {
        unreachable!()
    };
    let store = Q4Rec::<RECORD> { codes, scales };
    let records = 3 * c.n_kv_heads;
    /// Every 8-lane load of the `records` records of `store` through `S`
    /// (inlined, so the vector instantiations compile with the caller's
    /// instruction set as the production kernels do).
    #[inline(always)]
    unsafe fn loads<S: crate::simd::Simd>(store: &Q4Rec<'_, RECORD>, records: usize) -> Vec<u32> {
        let mut lanes = vec![0.0_f32; records * RECORD];
        for record in 0..records {
            for offset in (0..RECORD).step_by(8) {
                // SAFETY: the offset lies inside a stored record, and the
                // caller runs with `S`'s instruction set enabled.
                unsafe {
                    S::store(
                        lanes.as_mut_ptr().add(record * RECORD + offset),
                        store.load8::<S>(record, offset),
                    )
                };
            }
        }
        lanes.iter().map(|x| x.to_bits()).collect()
    }
    let scalar: Vec<u32> = (0..records * RECORD)
        .map(|i| cache.value(i / RECORD, i % RECORD).to_bits())
        .collect();
    // SAFETY: `Portable` needs no CPU features.
    assert_eq!(unsafe { loads::<crate::simd::Portable>(&store, records) }, scalar);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn native(store: &Q4Rec<'_, RECORD>, records: usize) -> [Vec<u32>; 2] {
            // SAFETY: the caller enables AVX2 and FMA, all `Avx2` and
            // `Avx2Fast` need.
            unsafe {
                [
                    loads::<crate::simd::Avx2>(store, records),
                    loads::<crate::simd::Avx2Fast>(store, records),
                ]
            }
        }
        // SAFETY: AVX2 and FMA were detected above.
        for lanes in unsafe { native(&store, records) } {
            assert_eq!(lanes, scalar);
        }
    }
    // SAFETY: NEON is baseline on aarch64.
    #[cfg(target_arch = "aarch64")]
    assert_eq!(unsafe { loads::<crate::simd::Neon>(&store, records) }, scalar);
    // Half a byte per element plus one 16-bit scale per 32 elements.
    assert_eq!(cache.records.bytes(), 3 * c.n_kv_heads * (RECORD / 2 + 2 * RECORD / 32));
}

#[test]
fn q4_scales_round_up_to_bf16() {
    // Absmax 1: 1/7 is no BF16 value, so the stored scale is the smallest
    // BF16 above it (never below, which would push codes past 7).
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 15.5).collect();
    let mut codes = [0_u8; 16];
    let bits = q4_block(&values, &mut codes);
    let (scale, below) = (f64::from(bf16_f32(bits)), f64::from(bf16_f32(bits - 1)));
    assert!(
        scale >= 1.0 / 7.0 && below < 1.0 / 7.0,
        "scale {scale}, next below {below}"
    );
    for (i, &value) in values.iter().enumerate() {
        let code = nibble(codes[i / 2], i % 2);
        assert!((-7..=7).contains(&code), "code {code}");
        assert!((f64::from(code) * scale - f64::from(value)).abs() <= scale / 2.0);
    }
}

#[test]
fn rotated_fp32_records_change_rounding_only() {
    // FP32 records hold the rotated values unquantized, so the decode
    // differs from unrotated records by the rotations' rounding and the
    // dots over rotated values only.
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    let rotated = Storage {
        format: Format::F32,
        rotated: true,
    };
    let mut worst = 0.0_f32;
    for (p, generated) in [(1, 1), (129, 17), (1000, 40)] {
        let (k, v, q) = fixture(&c, p, 7);
        let tail_k: Vec<_> = (0..generated * c.kv_dim())
            .map(|i| ((i * 13) % 37) as f32 / 53.0 - 0.3)
            .collect();
        let tail_v: Vec<_> = (0..generated * c.kv_dim())
            .map(|i| ((i * 11) % 41) as f32 / 67.0 - 0.3)
            .collect();
        let sinks: Vec<_> = (0..c.n_heads).map(|h| h as f32 * 0.25 - 1.0).collect();
        let plain = decode(&c, F32_PLAIN, &k, &v, &tail_k, &tail_v, &q, &sinks);
        let turned = decode(&c, rotated, &k, &v, &tail_k, &tail_v, &q, &sinks);
        for (a, b) in turned.iter().zip(&plain) {
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
    }
    // A few ulps of outputs below 1 (3.2e-7 measured on AVX2).
    assert!(worst <= 1e-6, "rotation moved an output by {worst:e}");
}

/// Deterministic standard normal samples (Box-Muller over SplitMix64).
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut uniform = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, w) = (uniform(), uniform());
            ((-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * w).cos()) as f32
        })
        .collect()
}

/// Synthetic `p` prefix and `generated` tail positions: unit Gaussian
/// keys (temporal halves shared per pair) and values, with `outliers`
/// channels like the model's: value channels 3, 37 and 60 at 10, 25 and
/// 50 times the RMS, and two key dimension pairs (temporal 4-5, spatial
/// 44-45) at 15 and 30 times, rotating with the position like RoPE
/// pairs.
#[allow(clippy::type_complexity)]
fn outlier_cache(
    c: &ModelConfig,
    p: usize,
    generated: usize,
    outliers: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (heads, groups) = (c.n_heads, c.n_kv_heads);
    let n = p + generated;
    let noise_k = gaussian(1, n * heads * 64);
    let noise_v = gaussian(2, n * groups * 64);
    let mut keys = vec![0.0; n * heads * 64];
    let mut values = vec![0.0; n * groups * 64];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..64 {
                // The temporal half comes from the pair's first head.
                let source = if d < 32 { h / 2 * 2 } else { h };
                let mut x = noise_k[(t * heads + source) * 64 + d];
                if outliers && (d / 2 == 2 || d / 2 == 22) {
                    let (amplitude, rate) = if d < 32 { (15.0, 0.3) } else { (30.0, 0.05) };
                    let angle = rate * t as f32 + source as f32;
                    x = amplitude * if d % 2 == 0 { angle.cos() } else { angle.sin() };
                }
                keys[(t * heads + h) * 64 + d] = x;
            }
        }
        for g in 0..groups {
            for d in 0..64 {
                let at = (t * groups + g) * 64 + d;
                let factor = match d {
                    3 => 10.0,
                    37 => 25.0,
                    60 => 50.0,
                    _ => 0.0,
                };
                values[at] = if outliers && factor > 0.0 {
                    factor * (1.0 + 0.25 * noise_v[at])
                } else {
                    noise_v[at]
                };
            }
        }
    }
    let tail_k = keys.split_off(p * heads * 64);
    // The tail stores one key per group: the pair's first head.
    let tail_k = tail_k.chunks_exact(128).flat_map(|pair| pair[..64].to_vec()).collect();
    let tail_v = values.split_off(p * groups * 64);
    (keys, values, tail_k, tail_v)
}

/// RMS error of `storage`'s output against unrotated FP32 records.
fn output_error(c: &ModelConfig, storage: Storage, outliers: bool) -> f32 {
    let (k, v, tail_k, tail_v) = outlier_cache(c, 600, 8, outliers);
    // Queries scaled so the logits spread over a few units.
    let q: Vec<f32> = gaussian(3, c.query_dim()).iter().map(|x| 0.4 * x).collect();
    let sinks = vec![0.0; c.n_heads];
    let exact = decode(c, F32_PLAIN, &k, &v, &tail_k, &tail_v, &q, &sinks);
    let coded = decode(c, storage, &k, &v, &tail_k, &tail_v, &q, &sinks);
    let sum: f64 = coded.iter().zip(&exact).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    (sum / exact.len() as f64).sqrt() as f32
}

#[test]
fn rotation_cuts_the_error_of_outlier_channels() {
    let c: ModelConfig = serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap();
    // (format, required error ratio with outliers): measured 0.26 for Q8
    // and 0.52 for Q4, whose 15 levels cannot resolve blocks whose energy
    // two channels dominate, rotated or not.
    for (format, margin) in [(Format::Q8, 0.5), (Format::Q4, 0.75)] {
        let error = |rotated, outliers| output_error(&c, Storage { format, rotated }, outliers);
        // Outlier channels set the absmax step of their whole block;
        // rotation spreads them over it, so the other values keep a
        // finer step.
        let (plain, rotated) = (error(false, true), error(true, true));
        eprintln!("{format:?} outliers: plain {plain:.3e} rotated {rotated:.3e}");
        assert!(
            rotated < margin * plain,
            "{format:?} with outliers: {rotated:e} vs {plain:e}"
        );
        // Isotropic Gaussian blocks gain nothing, and lose nothing beyond
        // noise.
        let (plain, rotated) = (error(false, false), error(true, false));
        eprintln!("{format:?} isotropic: plain {plain:.3e} rotated {rotated:.3e}");
        assert!(rotated < 1.2 * plain, "{format:?} isotropic: {rotated:e} vs {plain:e}");
    }
}
