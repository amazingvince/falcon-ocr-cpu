//! Independent focused tests injected as the storage module's test child.
use super::*;

pub(crate) fn bits_equal(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "different bits at element {i}");
    }
}

pub(crate) fn backends() -> Vec<Simd> {
    [Simd::Scalar, Simd::Auto, Simd::Avx2, Simd::Avx512]
        .into_iter().filter(|s| s.validate().is_ok()).collect()
}

// Prefix K intentionally differs in every channel between adjacent GQA heads.
// Only generated K and all V duplicate pairs, exactly as the compact contract.
pub(crate) fn operands(prefix: usize, generated: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.; (prefix + generated) * 1024];
    let mut v = k.clone();
    for token in 0..prefix + generated {
        for head in 0..16 {
            for dim in 0..64 {
                let kh = if token < prefix { head } else { head / 2 };
                let index = (token * 16 + head) * 64 + dim;
                k[index] = (((token * 41 + kh * 13 + dim * 7) % 211) as i32 - 105) as f32 / 37.;
                v[index] = (((token * 29 + (head / 2) * 17 + dim * 3) % 199) as i32 - 99) as f32 / 19.;
            }
        }
    }
    (k, v)
}

pub(crate) fn compact_rows(expanded: &[f32]) -> Vec<f32> {
    assert_eq!(expanded.len() % 1024, 0);
    expanded.chunks_exact(128).flat_map(|pair| pair[..64].iter().copied()).collect()
}

fn fill(cache: &mut HeadContiguousPrefixCache, k: &[f32], v: &[f32], prefix: usize) {
    cache.append(&k[..prefix * 1024], &v[..prefix * 1024], 0).unwrap();
    for token in prefix..k.len() / 1024 {
        cache.append(&k[token * 1024..(token + 1) * 1024],
                     &v[token * 1024..(token + 1) * 1024], token).unwrap();
    }
}

type Snapshot = (usize, usize, usize, [usize; 4], [usize; 4], [Vec<u32>; 4]);
fn snapshot(c: &HeadContiguousPrefixCache) -> Snapshot {
    let buffers = [&c.prefix_k, &c.prefix_v, &c.generated_k, &c.generated_v];
    (c.prefix, c.capacity, c.len,
     buffers.map(|v| v.capacity()), buffers.map(|v| v.as_ptr() as usize),
     buffers.map(|v| v.iter().map(|x| x.to_bits()).collect()))
}

#[test]
fn storage_full_head_order_and_special_bits() {
    let (prefix, generated) = (3, 2);
    let (mut k, mut v) = operands(prefix, generated);
    let patterns = [0x80000000, 0x00000000, 0x7fc01234, 0x7fa04321, 0xffc05678, 1];
    for (d, bits) in patterns.into_iter().enumerate() {
        k[d] = f32::from_bits(bits);
        // The adjacent prefix head has distinct payloads, including NaNs and zero.
        k[64 + d] = f32::from_bits(bits ^ 0x80000000);
        for token in 0..prefix + generated {
            v[token * 1024 + d] = f32::from_bits(bits);
            v[token * 1024 + 64 + d] = f32::from_bits(bits);
        }
        for token in prefix..prefix + generated {
            k[token * 1024 + d] = f32::from_bits(bits);
            k[token * 1024 + 64 + d] = f32::from_bits(bits);
        }
    }
    let mut c = HeadContiguousPrefixCache::new(prefix, prefix + generated, 16, 8, 64).unwrap();
    fill(&mut c, &k, &v, prefix);
    assert_eq!((c.len, c.prefix_k.len(), c.prefix_v.len(), c.generated_k.len(), c.generated_v.len()),
               (5, prefix * 1024, prefix * 512, generated * 512, generated * 512));
    for token in 0..prefix {
        for head in 0..16 {
            let src = (token * 16 + head) * 64;
            let dst = (head * prefix + token) * 64;
            bits_equal(&c.prefix_k[dst..dst + 64], &k[src..src + 64]);
            let vd = ((head / 2) * prefix + token) * 64;
            bits_equal(&c.prefix_v[vd..vd + 64], &v[src..src + 64]);
        }
    }
    bits_equal(&c.generated_k, &compact_rows(&k[prefix * 1024..]));
    bits_equal(&c.generated_v, &compact_rows(&v[prefix * 1024..]));
    assert_ne!(c.prefix_k[0].to_bits(), c.prefix_k[prefix * 64].to_bits());
    assert_ne!(k[32].to_bits(), k[96].to_bits());
}

#[test]
fn duplicate_bit_failures_leave_every_buffer_unchanged() {
    for generated in [false, true] {
        for key in [false, true] {
            if key && !generated { continue; } // Prefix K must remain distinct.
            for pair in [(0x00000000, 0x80000000), (0x7fc01234, 0x7fc01235)] {
                let (k, v) = operands(3, 1);
                let mut c = HeadContiguousPrefixCache::new(3, 4, 16, 8, 64).unwrap();
                let offset = if generated {
                    c.append(&k[..3072], &v[..3072], 0).unwrap();
                    3
                } else { 0 };
                let mut ka = if generated { k[3072..].to_vec() } else { k[..3072].to_vec() };
                let mut va = if generated { v[3072..].to_vec() } else { v[..3072].to_vec() };
                let target = if key { &mut ka } else { &mut va };
                let last_pair = target.len() - 128;
                target[last_pair + 63] = f32::from_bits(pair.0);
                target[last_pair + 127] = f32::from_bits(pair.1);
                let before = snapshot(&c);
                assert!(c.append(&ka, &va, offset).is_err());
                assert_eq!(snapshot(&c), before, "late duplicate failure mutated cache");
            }
        }
    }
}

#[test]
fn invalid_sizes_offsets_and_overflow_reject_before_mutation() {
    for (p, capacity, h, kv, d) in [(0, 4, 16, 8, 64), (3, 2, 16, 8, 64),
        (3, 4, 8, 8, 64), (3, 4, 16, 4, 64), (3, 4, 16, 8, 32),
        (usize::MAX, usize::MAX, 16, 8, 64), (1, usize::MAX, 16, 8, 64)] {
        assert!(HeadContiguousPrefixCache::new(p, capacity, h, kv, d).is_err());
    }
    let (k, v) = operands(3, 2);
    let mut c = HeadContiguousPrefixCache::new(3, 4, 16, 8, 64).unwrap();
    for (kl, vl, off) in [(0, 0, 0), (1024, 1024, 0), (3071, 3071, 0),
                           (3072, 3071, 0), (3072, 3072, 1)] {
        let before = snapshot(&c);
        assert!(c.append(&k[..kl], &v[..vl], off).is_err());
        assert_eq!(snapshot(&c), before);
    }
    c.append(&k[..3072], &v[..3072], 0).unwrap();
    for (start, rows, off) in [(0, 3, 0), (3, 2, 3), (3, 1, 2), (3, 1, 4),
                               (3, 1, usize::MAX), (3, 0, 3)] {
        let before = snapshot(&c);
        assert!(c.append(&k[start * 1024..(start + rows) * 1024],
                         &v[start * 1024..(start + rows) * 1024], off).is_err());
        assert_eq!(snapshot(&c), before);
    }
    c.append(&k[3072..4096], &v[3072..4096], 3).unwrap();
    let before = snapshot(&c);
    assert!(c.append(&k[4096..], &v[4096..], 4).is_err());
    assert_eq!(snapshot(&c), before);
}

#[test]
fn reserved_capacity_and_addresses_stay_fixed_through_appends() {
    let (prefix, generated) = (144, 17);
    let mut c = HeadContiguousPrefixCache::new(prefix, prefix + generated, 16, 8, 64).unwrap();
    let before = snapshot(&c);
    let requested = [prefix * 1024, prefix * 512, generated * 512, generated * 512];
    for (actual, need) in before.3.into_iter().zip(requested) { assert!(actual >= need); }
    let (k, v) = operands(prefix, generated);
    fill(&mut c, &k, &v, prefix);
    let after = snapshot(&c);
    assert_eq!(before.3, after.3);
    assert_eq!(before.4, after.4);
    assert_eq!(after.5.each_ref().map(|v| v.len()), requested);
    let logical_bytes = (1536 * prefix + 1024 * generated) * 4;
    assert_eq!(logical_bytes, 954_368);
    assert_eq!(22usize * 4 * (1536 * 6544 + 1024 * 4096), 1_253_638_144);
    assert_eq!(6544usize * 512 * 4, 13_402_112);
    eprintln!("HEAD_PREFIX_CAPACITY {{\"prefix\":{prefix},\"generated_reserved\":{generated},\"logical_bytes\":{logical_bytes},\"actual_vec_capacity_bytes\":{},\"capacities\":{:?},\"same_addresses\":true,\"scope\":\"four persistent Vec capacities only; no scratch/RSS/timing\"}}", before.3.iter().sum::<usize>() * 4, before.3);
}

#[test]
fn reusable_prefill_scratch_has_compact_order_and_transactional_errors() {
    let (_, mut v) = operands(3, 0);
    v[0] = f32::from_bits(0x7fc01234); v[64] = v[0];
    v[1] = -0.; v[65] = -0.;
    let mut scratch = Vec::with_capacity(3 * 512);
    let pointer = scratch.as_ptr(); let capacity = scratch.capacity();
    for _ in 0..2 {
        pack_prefill_values(&v, &mut scratch).unwrap();
        bits_equal(&scratch, &compact_rows(&v));
        assert_eq!(scratch.as_ptr(), pointer); assert_eq!(scratch.capacity(), capacity);
    }
    let saved: Vec<_> = scratch.iter().map(|v| v.to_bits()).collect();
    let mut bad = v.clone(); bad[3 * 1024 - 1] = f32::from_bits(bad[3 * 1024 - 65].to_bits() ^ 1);
    for invalid in [&bad[..], &v[..v.len() - 1]] {
        assert!(pack_prefill_values(invalid, &mut scratch).is_err());
        assert_eq!(scratch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), saved);
        assert_eq!(scratch.as_ptr(), pointer); assert_eq!(scratch.capacity(), capacity);
    }
    let mut small = vec![123.]; let old_ptr = small.as_ptr(); let old_capacity = small.capacity();
    assert!(pack_prefill_values(&v, &mut small).is_err());
    assert_eq!(small, [123.]); assert_eq!(small.as_ptr(), old_ptr); assert_eq!(small.capacity(), old_capacity);
}

#[test]
fn cache_attention_rejects_wrong_intervals_and_retained_scratch() {
    let (k, v) = operands(3, 1);
    let mut c = HeadContiguousPrefixCache::new(3, 4, 16, 8, 64).unwrap();
    c.append(&k[..3072], &v[..3072], 0).unwrap();
    let cv = compact_rows(&v[..3072]); let q = [0.; 3072]; let mut out = [123.; 3072];
    for (rows, total, offset, kl, vl) in [(3, 3, 0, 3071, 1536), (3, 3, 0, 3072, 1535),
                                        (2, 2, 0, 3072, 1536), (2, 3, 1, 3072, 1536)] {
        let before = snapshot(&c);
        assert!(c.attention(&q[..rows * 1024], &k[..kl], &cv[..vl], rows, total,
            offset, 1, 2, &[0.; 16], &mut out[..rows * 1024], Simd::Scalar).is_err());
        assert_eq!(snapshot(&c), before);
        assert!(out.iter().all(|v| v.to_bits() == 123f32.to_bits()));
    }
    c.append(&k[3072..], &v[3072..], 3).unwrap();
    let before = snapshot(&c);
    assert!(c.attention(&q[..1024], &[], &cv, 1, 4, 3,
        1, 2, &[0.; 16], &mut out[..1024], Simd::Scalar).is_err());
    assert_eq!(snapshot(&c), before);
    assert!(out.iter().all(|v| v.to_bits() == 123f32.to_bits()));
}

fn query() -> Vec<f32> {
    (0..1024).map(|i| ((i * 17 % 101) as i32 - 50) as f32 / 31.).collect()
}

fn operator_case(prefix: usize, total: usize, offsets: &[usize], extreme: bool) {
    let (k, v) = operands(prefix, total - prefix);
    let cv = compact_rows(&v); let gk = compact_rows(&k[prefix * 1024..]);
    // Independently pack head-major prefix data, avoiding the storage implementation
    // as the kernel test oracle. Generated buffers stay token-major.
    let pk: Vec<_> = (0..16).flat_map(|h| (0..prefix).flat_map(move |t| (0..64).map(move |d| (t * 16 + h) * 64 + d)))
        .map(|i| k[i]).collect();
    let pv: Vec<_> = (0..8).flat_map(|h| (0..prefix).flat_map(move |t| (0..64).map(move |d| (t * 16 + 2 * h) * 64 + d)))
        .map(|i| v[i]).collect();
    let gv = &cv[prefix * 512..];
    let q: Vec<_> = query().into_iter().map(|v| if extreme {v * 128.} else {v}).collect();
    let sinks: Vec<_> = (0..16).map(|h| [-1000., -2., 0., 1000.][h % 4]).collect();
    let (image_start, image_end) = if prefix > 2 { (1, prefix - 1) } else { (0, 0) };
    for simd in backends() {
        for &offset in offsets {
            let mut expected = [0.; 1024]; let mut actual = [0.; 1024];
            kernels::attention_compact_with_simd(&q, &k[..prefix * 1024], &gk, &cv,
                1, prefix, total, 16, 8, 64, offset, image_start, image_end, &sinks, &mut expected, simd);
            kernels::attention_head_contiguous_prefix_with_simd(&q, &pk, &pv, &gk, gv,
                1, prefix, total, 16, 8, 64, offset, image_start, image_end, &sinks, &mut actual, simd);
            assert!(actual.iter().all(|v| v.is_finite()), "nonfinite {simd:?} P{prefix}/T{total}");
            bits_equal(&actual, &expected);
        }
    }
}

#[test]
fn operator_short_prefix_zero_full_and_tile_crossings_all_supported_backends() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
        for (prefix, total) in [(0, 1), (0, 129), (1, 1), (17, 17), (127, 130),
                               (128, 257), (129, 259), (257, 257)] {
            let offsets = [0, total / 2, total - 1];
            operator_case(prefix, total, &offsets, false);
        }
        operator_case(129, 257, &[0, 1, 126, 127, 128, 129, 256], true);
    });
}

#[test]
fn operator_fullpage_crossing_tile_and_16384_context_all_supported_backends() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
        for total in [6544, 6545, 6655, 6656, 6657, 16383, 16384] {
            operator_case(6544, total, &[total - 1], false);
        }
    });
}

#[test]
fn unchanged_prefill_and_storage_decode_dispatch_match_compact() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
        for prefix in [3, 17, 129] {
            let (k, v) = operands(prefix, 1); let cv = compact_rows(&v);
            for simd in backends() {
                let mut c = HeadContiguousPrefixCache::new(prefix, prefix + 1, 16, 8, 64).unwrap();
                c.append(&k[..prefix * 1024], &v[..prefix * 1024], 0).unwrap();
                let mut scratch = Vec::with_capacity(prefix * 512);
                pack_prefill_values(&v[..prefix * 1024], &mut scratch).unwrap();
                let q: Vec<_> = query().into_iter().cycle().take(prefix * 1024).collect();
                let mut expected = vec![0.; q.len()]; let mut actual = expected.clone();
                kernels::attention_compact_with_simd(&q, &k[..prefix * 1024], &[], &cv[..prefix * 512],
                    prefix, prefix, prefix, 16, 8, 64, 0, 1, prefix - 1, &[0.; 16], &mut expected, simd);
                c.attention(&q, &k[..prefix * 1024], &scratch, prefix, prefix, 0,
                            1, prefix - 1, &[0.; 16], &mut actual, simd).unwrap();
                bits_equal(&actual, &expected);
                drop(scratch); // Decode must work without a retained prefix scratch.
                c.append(&k[prefix * 1024..], &v[prefix * 1024..], prefix).unwrap();
                kernels::attention_compact_with_simd(&q[..1024], &k[..prefix * 1024],
                    &compact_rows(&k[prefix * 1024..]), &cv, 1, prefix, prefix + 1, 16, 8, 64,
                    prefix, 1, prefix - 1, &[0.; 16], &mut expected[..1024], simd);
                c.attention(&q[..1024], &[], &[], 1, prefix + 1, prefix,
                            1, prefix - 1, &[0.; 16], &mut actual[..1024], simd).unwrap();
                bits_equal(&actual[..1024], &expected[..1024]);
            }
        }
    });
}

#[test]
fn operator_invalid_shapes_preserve_output_before_dispatch() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    for kind in 0..14 {
        let q = [0.; 1024]; let pk = [0.; 1024]; let pv = [0.; 512];
        let gk = [0.; 512]; let gv = [0.; 512]; let sinks = [0.; 16];
        let mut output = [123.; 1024];
        let result = catch_unwind(AssertUnwindSafe(|| {
            kernels::attention_head_contiguous_prefix_with_simd(
                &q[..if kind == 0 {1023} else {1024}],
                &pk[..if kind == 1 {1023} else {1024}],
                &pv[..if kind == 2 {511} else {512}],
                &gk[..if kind == 3 {511} else {512}],
                &gv[..if kind == 4 {511} else {512}],
                if kind == 5 {2} else {1}, 1, 2,
                if kind == 6 {8} else {16}, if kind == 7 {4} else {8},
                if kind == 8 {32} else {64}, if kind == 9 {2} else {1},
                if kind == 10 {2} else {0}, if kind == 11 {2} else {0},
                &sinks[..if kind == 12 {15} else {16}],
                &mut output[..if kind == 13 {1023} else {1024}], Simd::Scalar);
        }));
        assert!(result.is_err(), "invalid operator case {kind} accepted");
        assert!(output.iter().all(|v| v.to_bits() == 123f32.to_bits()));
    }
}
