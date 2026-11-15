//! Oracles for the FP32 prefill tiles and their ISA instantiations.
use super::*;

fn values(n: usize, seed: u32, range: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32 * range;
            // Signed zeros and a subnormal exercise edge rounding.
            match i % 97 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(3),
                _ => x,
            }
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn available() -> bool {
    crate::kernels::avx2_available()
}

/// Random prefill operands: (q, k, v, sinks) for 16 query / 8 KV heads.
fn operands(query_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        values(query_len * 16 * 64, 17 + query_len as u32, 3.0),
        values(query_len * 16 * 64, 19 + query_len as u32, 3.0),
        values(query_len * 8 * 64, 23 + query_len as u32, 3.0),
        values(16, 29, 2.0),
    )
}

const CASES: [(usize, usize, usize); 7] = [
    (70, 3, 65),
    (300, 5, 262),
    (257, 0, 257),
    (160, 16, 144),
    (129, 1, 128),
    (40, 0, 3),
    (95, 10, 90),
];

fn run(tile: TileFn, query_len: usize, image: (usize, usize)) -> Vec<f32> {
    let (q, k, v, sinks) = operands(query_len);
    let mut out = vec![f32::NAN; q.len()];
    unsafe {
        compact_prefill_with(
            tile,
            &q,
            &CompactKv::prefill(&k, &v, query_len, 16, 8, 64),
            query_len,
            Geometry::new(0, image.0, image.1),
            &sinks,
            &mut out,
            false,
        );
    }
    out
}

/// The portable instantiation (fast exp) stays close to the reference.
#[test]
fn portable_prefill_is_close_to_reference() {
    for (query_len, image_start, image_end) in CASES {
        let (q, k, v, sinks) = operands(query_len);
        let mut expected = vec![f32::NAN; q.len()];
        super::super::tiled::attention_gemm(
            &q,
            &CompactKv::prefill(&k, &v, query_len, 16, 8, 64),
            Geometry::new(0, image_start, image_end),
            &sinks,
            &mut expected,
        );
        let actual = run(tile_head_portable, query_len, (image_start, image_end));
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "query_len {query_len} index {i}: {a} vs {b}"
            );
        }
    }
}

/// With generated keys the tiles gather the boundary tile like the tiled
/// GEMM path and stay bitwise equal to it (on x86 the two share their
/// arithmetic order); this is the path aarch64 takes for both layouts.
#[cfg(target_arch = "x86_64")]
#[test]
fn mixed_cache_tiles_match_tiled_gemm_bitwise() {
    if !available() {
        return;
    }
    // (queries, prefix, total, query offset, image start, image end)
    for (query_len, prefix_len, total_len, offset, image_start, image_end) in [
        (4, 137, 141, 137, 1, 129),
        (9, 128, 137, 128, 1, 129),
        (33, 100, 133, 100, 1, 90),
        (40, 3, 43, 3, 1, 40),
        (5, 250, 300, 295, 0, 0),
    ] {
        let q = values(query_len * 16 * 64, 71, 3.0);
        let prefix = values(prefix_len * 16 * 64, 83, 3.0);
        let generated = values((total_len - prefix_len) * 8 * 64, 37, 3.0);
        let v = values(total_len * 8 * 64, 117, 3.0);
        let sinks = values(16, 29, 2.0);
        let cache = CompactKv {
            prefix_k: &prefix,
            generated_k: &generated,
            v: &v,
            prefix_len,
            total_len,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 64,
        };
        let geometry = Geometry::new(offset, image_start, image_end);
        let mut expected = vec![f32::NAN; q.len()];
        super::super::tiled::attention_gemm(&q, &cache, geometry, &sinks, &mut expected);
        for wide in [false, true] {
            if wide && !wide_available() {
                continue;
            }
            let mut actual = vec![f32::NAN; q.len()];
            unsafe {
                compact_prefill(
                    &q,
                    &cache,
                    query_len,
                    geometry,
                    &sinks,
                    &mut actual,
                    wide,
                    PrefillOptions::EXACT,
                );
            }
            for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "Q={query_len} prefix={prefix_len} total={total_len} wide={wide} index {i}: {a} vs {e}"
                );
            }
        }
    }
}

/// NEON is bitwise equal to the portable instantiation.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_prefill_matches_portable_bitwise() {
    for (query_len, image_start, image_end) in CASES {
        let native = run(tile_head_native, query_len, (image_start, image_end));
        let portable = run(tile_head_portable, query_len, (image_start, image_end));
        for (i, (a, b)) in native.iter().zip(&portable).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "query_len {query_len} index {i}");
        }
    }
}

/// AVX2 with the fast exp is bitwise equal to the portable instantiation.
#[cfg(target_arch = "x86_64")]
#[test]
fn fast_exp_prefill_matches_portable_bitwise() {
    if !available() {
        return;
    }
    for (query_len, image_start, image_end) in CASES {
        let fast = run(tile_head_fast, query_len, (image_start, image_end));
        let portable = run(tile_head_portable, query_len, (image_start, image_end));
        for (i, (a, b)) in fast.iter().zip(&portable).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "query_len {query_len} index {i}");
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn wide_available() -> bool {
    available() && crate::kernels::avx512_available()
}

/// The AVX-512 QK and PV kernels equal the AVX2 ones bit for bit on every
/// tile shape (both keep one FMA chain per element in the same order).
#[cfg(target_arch = "x86_64")]
#[test]
fn wide_qk_and_pv_match_avx2_for_every_tile_shape() {
    if !wide_available() {
        return;
    }
    #[target_feature(enable = "avx2,fma,avx512f")]
    unsafe fn both(
        qt: &[f32; HEAD_DIM * QUERY_TILE],
        queries: usize,
        keys: usize,
        k: &[f32],
        v: &[f32],
        probabilities: &[f32; KEY_TILE * QUERY_TILE],
        initial: &[f32],
    ) -> [Vec<f32>; 4] {
        let mut st_narrow = [f32::NAN; KEY_TILE * QUERY_TILE];
        let mut st_wide = [f32::NAN; KEY_TILE * QUERY_TILE];
        let mut out_narrow = initial.to_vec();
        let mut out_wide = initial.to_vec();
        unsafe {
            qk_lanes::<crate::simd::Avx2>(qt, queries, k.as_ptr(), 1024, keys, 0.125, &mut st_narrow);
            wide::qk_lanes(qt, queries, k.as_ptr(), 1024, keys, 0.125, &mut st_wide);
            pv_lanes::<crate::simd::Avx2>(
                probabilities,
                queries,
                keys,
                v.as_ptr(),
                512,
                out_narrow.as_mut_ptr(),
                1024,
            );
            wide::pv_lanes(
                probabilities,
                queries,
                keys,
                v.as_ptr(),
                512,
                out_wide.as_mut_ptr(),
                1024,
            );
        }
        let keep = |st: &[f32; KEY_TILE * QUERY_TILE]| {
            (0..keys)
                .flat_map(|j| (0..queries).map(move |r| (j, r)))
                .map(|(j, r)| st[j * QUERY_TILE + r])
                .collect::<Vec<_>>()
        };
        [keep(&st_narrow), keep(&st_wide), out_narrow, out_wide]
    }
    let k = values(KEY_TILE * 1024, 5, 4.0);
    let v = values(KEY_TILE * 512, 7, 4.0);
    let q = values(QUERY_TILE * HEAD_DIM, 3, 4.0);
    let mut qt = [0.0_f32; HEAD_DIM * QUERY_TILE];
    let mut probabilities = [0.0_f32; KEY_TILE * QUERY_TILE];
    for (i, p) in values(KEY_TILE * QUERY_TILE, 11, 1.0).into_iter().enumerate() {
        probabilities[i] = if p < -0.7 { 0.0 } else { p.abs() };
    }
    let initial = values(QUERY_TILE * 1024, 13, 4.0);
    for queries in 1..=QUERY_TILE {
        qt.fill(0.0);
        for r in 0..queries {
            for d in 0..HEAD_DIM {
                qt[d * QUERY_TILE + r] = q[r * HEAD_DIM + d];
            }
        }
        for keys in 1..=KEY_TILE {
            let [a, b, c, d] = unsafe { both(&qt, queries, keys, &k, &v, &probabilities, &initial) };
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "qk q{queries} k{keys} {i}");
            }
            for (i, (x, y)) in c.iter().zip(&d).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "pv q{queries} k{keys} {i}");
            }
        }
    }
}

/// Whole prefill: the wide entries equal the AVX2 entries bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
fn wide_prefill_matches_avx2_bitwise() {
    if !wide_available() {
        return;
    }
    for (query_len, image_start, image_end) in CASES {
        for (narrow, wide) in [
            (tile_head_native as TileFn, tile_head_native_wide as TileFn),
            (tile_head_fast as TileFn, tile_head_fast_wide as TileFn),
        ] {
            let a = run(narrow, query_len, (image_start, image_end));
            let b = run(wide, query_len, (image_start, image_end));
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "query_len {query_len} index {i}");
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn qk_via_lanes(
    q: &[f32],
    q_stride: usize,
    queries: usize,
    k: &[f32],
    k_stride: usize,
    keys: usize,
) -> [f32; KEY_TILE * QUERY_TILE] {
    let mut qt = [0.0_f32; HEAD_DIM * QUERY_TILE];
    for r in 0..queries {
        for d in 0..HEAD_DIM {
            qt[d * QUERY_TILE + r] = q[r * q_stride + d];
        }
    }
    let mut st = [f32::NAN; KEY_TILE * QUERY_TILE];
    unsafe { qk_lanes::<crate::simd::Avx2>(&qt, queries, k.as_ptr(), k_stride, keys, 0.125, &mut st) };
    st
}

#[cfg(target_arch = "x86_64")]
#[test]
#[cfg_attr(feature = "gemm-avx512", ignore = "gemm dispatches to its AVX-512 kernels")]
fn qk_lanes_match_gemm_main_path_for_every_tile_shape() {
    if !available() {
        return;
    }
    let stride = 1024;
    let q = values(QUERY_TILE * stride, 3, 4.0);
    let k = values(KEY_TILE * stride, 5, 4.0);
    let mut checked = 0;
    for queries in 1..=QUERY_TILE {
        for keys in 1..=KEY_TILE {
            if !qk_uses_main_path(queries, keys) {
                continue;
            }
            let mut expected = [f32::NAN; QUERY_TILE * KEY_TILE];
            unsafe {
                gemm::gemm(
                    queries,
                    keys,
                    HEAD_DIM,
                    expected.as_mut_ptr(),
                    1,
                    KEY_TILE as isize,
                    false,
                    q.as_ptr(),
                    1,
                    stride as isize,
                    k.as_ptr(),
                    stride as isize,
                    1,
                    0.0_f32,
                    0.125,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            let st = unsafe { qk_via_lanes(&q, stride, queries, &k, stride, keys) };
            for row in 0..queries {
                for key in 0..keys {
                    assert_eq!(
                        st[key * QUERY_TILE + row].to_bits(),
                        expected[row * KEY_TILE + key].to_bits(),
                        "queries {queries} keys {keys} row {row} key {key}"
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 3000);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[cfg_attr(feature = "gemm-avx512", ignore = "gemm dispatches to its AVX-512 kernels")]
fn pv_lanes_match_gemm_main_path_for_every_tile_shape() {
    if !available() {
        return;
    }
    let (v_stride, out_stride) = (512, 1024);
    let v = values(KEY_TILE * v_stride, 7, 4.0);
    let probabilities: Vec<f32> = values(QUERY_TILE * KEY_TILE, 11, 1.0)
        .into_iter()
        .map(|p| if p < -0.7 { 0.0 } else { p.abs() })
        .collect();
    let mut row_major = [0.0_f32; QUERY_TILE * KEY_TILE];
    let mut key_major = [0.0_f32; KEY_TILE * QUERY_TILE];
    for r in 0..QUERY_TILE {
        for j in 0..KEY_TILE {
            row_major[r * KEY_TILE + j] = probabilities[r * KEY_TILE + j];
            key_major[j * QUERY_TILE + r] = probabilities[r * KEY_TILE + j];
        }
    }
    let initial = values(QUERY_TILE * out_stride, 13, 4.0);
    let mut checked = 0;
    for queries in 1..=QUERY_TILE {
        for keys in 1..=KEY_TILE {
            if !pv_uses_main_path(queries, keys) {
                continue;
            }
            let mut expected = initial.clone();
            unsafe {
                gemm::gemm(
                    queries,
                    HEAD_DIM,
                    keys,
                    expected.as_mut_ptr(),
                    1,
                    out_stride as isize,
                    true,
                    row_major.as_ptr(),
                    1,
                    KEY_TILE as isize,
                    v.as_ptr(),
                    1,
                    v_stride as isize,
                    1.0_f32,
                    1.0_f32,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            let mut actual = initial.clone();
            unsafe {
                pv_lanes::<crate::simd::Avx2>(
                    &key_major,
                    queries,
                    keys,
                    v.as_ptr(),
                    v_stride,
                    actual.as_mut_ptr(),
                    out_stride,
                );
            }
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "queries {queries} keys {keys} {i}");
            }
            checked += 1;
        }
    }
    assert!(checked > 3000);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[cfg_attr(feature = "gemm-avx512", ignore = "gemm dispatches to its AVX-512 kernels")]
fn whole_prefill_matches_reference_bitwise() {
    if !available() {
        return;
    }
    let (n_heads, n_kv_heads) = (16, 8);
    // Tail tiles, text before and after the image, image ends inside and
    // on key-tile boundaries.
    for (query_len, image_start, image_end) in CASES {
        let (q, k, v, sinks) = operands(query_len);
        let mut expected = vec![f32::NAN; q.len()];
        super::super::tiled::attention_gemm(
            &q,
            &CompactKv::prefill(&k, &v, query_len, n_heads, n_kv_heads, 64),
            Geometry::new(0, image_start, image_end),
            &sinks,
            &mut expected,
        );
        let mut actual = vec![f32::NAN; q.len()];
        unsafe {
            compact_prefill(
                &q,
                &CompactKv::prefill(&k, &v, query_len, n_heads, n_kv_heads, 64),
                query_len,
                Geometry::new(0, image_start, image_end),
                &sinks,
                &mut actual,
                false,
                PrefillOptions::EXACT,
            );
        }
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "query_len {query_len} image {image_start}..{image_end} index {i}"
            );
        }
    }
}
