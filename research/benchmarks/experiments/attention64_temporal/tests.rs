use super::*;
use crate::kernels::{self, Simd};
use crate::temporal_candidate::tests::{bits_equal, compact_v, operands};

fn supported() {
    assert!(Simd::Avx2.validate().is_ok(), "Selected direct-kernel tests require AVX2/FMA; no silent skip");
}

fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|i| {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        if i % 97 == 0 { -0.0 } else {
            (((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32) * scale
        }
    }).collect()
}

#[target_feature(enable="avx2,fma")]
unsafe fn compare_dot(q: &[f32], temporal: &[f32], spatial: &[f32]) {
    let mut contiguous = [0.; 64];
    contiguous[..32].copy_from_slice(temporal);
    contiguous[32..].copy_from_slice(spatial);
    unsafe {
        let expected = kernels::x86::dot_avx2(q, &contiguous);
        assert_eq!(dot64_split(q, temporal, spatial).to_bits(), expected.to_bits());
        assert_eq!(dot64(q, &contiguous).to_bits(), expected.to_bits());
    }
}

#[test]
fn split_dot_preserves_four_accumulators_unaligned_halves_and_cancellation() {
    supported();
    for seed in 1..=96 {
        let q = data(72, seed, if seed % 2 == 0 {1e10} else {0.03125});
        let temporal = data(40, seed + 997, if seed % 3 == 0 {1e-10} else {8.0});
        let spatial = data(40, seed + 1559, if seed % 5 == 0 {1e-10} else {8.0});
        for (qo, to, so) in [(0, 0, 0), (1, 3, 5), (7, 1, 2)] {
            unsafe { compare_dot(&q[qo..qo+64], &temporal[to..to+32], &spatial[so..so+32]); }
        }
    }
    for factor in [-0.0, 0.0, 1.0, 8192.0, 1.0 / 8192.0] {
        let q: Vec<_> = (0..64).map(|i| if i % 2 == 0 {factor} else {-factor}).collect();
        let temporal: Vec<_> = (0..32).map(|i| if i % 3 == 0 {1.0000001} else {1.0}).collect();
        let spatial: Vec<_> = temporal.iter().map(|x| -*x).collect();
        unsafe { compare_dot(&q, &temporal, &spatial); }
    }
}

fn case(total: usize, prefix: usize, offset: usize, image_start: usize,
        image_end: usize, simd: Simd, extreme: bool) {
    let (k, v) = operands(prefix, total - prefix);
    let mut temporal = Vec::with_capacity(prefix * 256);
    let mut spatial = Vec::with_capacity(prefix * 512);
    for group in k[..prefix * 1024].chunks_exact(128) {
        temporal.extend_from_slice(&group[..32]);
        spatial.extend_from_slice(&group[32..64]);
        spatial.extend_from_slice(&group[96..128]);
    }
    let generated = compact_v(&k[prefix * 1024..]);
    let values = compact_v(&v);
    let q = data(1024, 41, if extreme {100.0} else {0.75});
    let sinks: Vec<_> = (0..16).map(|h| [-1000., -2., 0., 1000.][h % 4]).collect();
    let mut compact = [0.; 1024];
    let mut expanded = [0.; 1024];
    let mut generic = [0.; 1024];
    let mut direct = [0.; 1024];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| {
        kernels::attention_compact_with_simd(&q, &k[..prefix * 1024], &generated,
            &values, 1, prefix, total, 16, 8, 64, offset, image_start, image_end,
            &sinks, &mut compact, simd);
        kernels::attention_with_simd(&q, &k, &v, 1, total, 16, 64, offset,
            image_start, image_end, &sinks, &mut expanded, simd);
        kernels::attention_temporal_generic_for_test(&q, &temporal, &spatial,
            &generated, &values, 1, prefix, total, 16, 8, 64, offset,
            image_start, image_end, &sinks, &mut generic, simd);
        kernels::attention_temporal_candidate_with_simd(&q, &temporal, &spatial,
            &generated, &values, 1, prefix, total, 16, 8, 64, offset,
            image_start, image_end, &sinks, &mut direct, simd);
    });
    assert!(direct.iter().all(|x| x.is_finite()));
    bits_equal(&direct, &compact);
    bits_equal(&direct, &expanded);
    bits_equal(&direct, &generic);
}

#[test]
fn split_attention_zero_full_prefix_and_tile_crossovers_match() {
    supported();
    for total in [1, 17, 127, 128, 129, 257] {
        for prefix in [0, total / 2, total] {
            case(total, prefix, total - 1, 0, 0, Simd::Avx2, false);
        }
    }
    for prefix in [127, 128, 129] {
        case(257, prefix, 256, 1, prefix, Simd::Auto, false);
    }
}

#[test]
fn split_attention_image_boundaries_and_extreme_sinks_match() {
    supported();
    for offset in [0, 1, 126, 127, 128, 129, 160] {
        case(161, 129, offset, 1, 128, Simd::Avx2, false);
    }
    case(257, 129, 256, 1, 129, Simd::Avx2, true);
}

#[test]
fn split_attention_real_prefix_and_full_context_match() {
    supported();
    for (total, prefix) in [(6544, 6544), (6545, 6544), (16384, 6544)] {
        case(total, prefix, total - 1, 1, 6540, Simd::Avx2, false);
    }
    assert_eq!(6544usize * 256 * 22 * 4, 147_423_232);
}

#[test]
fn split_attention_rejects_bad_shapes_before_dispatch() {
    supported();
    use std::panic::{catch_unwind, AssertUnwindSafe};
    // Every supplied buffer is valid for one prefix row except the one
    // deliberately shortened or unsupported field selected in each case.
    for kind in 0..8 {
        let q = [0.; 1024]; let temporal = [0.; 256]; let spatial = [0.; 512];
        let values = [0.; 512]; let sinks = [0.; 16]; let mut output = [0.; 1024];
        let result = catch_unwind(AssertUnwindSafe(|| {
            kernels::attention_temporal_candidate_with_simd(
                &q[..if kind == 0 {1023} else {1024}],
                &temporal[..if kind == 1 {255} else {256}],
                &spatial[..if kind == 2 {511} else {512}], &[],
                &values[..if kind == 3 {511} else {512}],
                if kind == 4 {2} else {1}, 1, 1,
                if kind == 5 {8} else {16}, 8, 64,
                if kind == 6 {1} else {0}, 0, 0,
                &sinks[..if kind == 7 {15} else {16}], &mut output, Simd::Avx2);
        }));
        assert!(result.is_err(), "invalid case {kind} was accepted");
    }
}
