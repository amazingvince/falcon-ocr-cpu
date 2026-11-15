//! Tests of the FP32 static-shift tiles: the unchecked exps against
//! `exp_poly`, the 8- and 16-lane tiles against a scalar oracle (bitwise),
//! and the kernel against an f64 reference; plus a timing probe.
use super::*;
use crate::config::{ExpMode, PrefillSoftmax};
#[cfg(target_arch = "x86_64")]
use crate::kernels::attention::prefill64::compact_prefill;
#[cfg(target_arch = "x86_64")]
use crate::kernels::attention::prefill64::test_data::{normalized, values};
use crate::kernels::attention::prefill64::{
    HEAD_DIM, TileFn, compact_prefill_with,
    test_data::{operands, reference_errors},
    tile_span,
};
use crate::kernels::{Geometry, PrefillOptions};
use crate::simd::exp_poly;

fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

fn wide_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        available() && std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The static tiles with the given exp (which they ignore).
fn options(exp: ExpMode) -> PrefillOptions {
    PrefillOptions {
        exp,
        fp32_softmax: PrefillSoftmax::Static,
        ..PrefillOptions::EXACT
    }
}

/// (queries, total, query offset, image start, image end): partial query
/// tiles, short and odd key tiles, text before and after the image, image
/// ends inside and on key-tile boundaries, all-image tiles, and pure
/// prefixes whose queries start past zero (chunked prefill).
const CASES: [(usize, usize, usize, usize, usize); 10] = [
    (70, 70, 0, 3, 65),
    (300, 300, 0, 5, 262),
    (257, 257, 0, 0, 257),
    (160, 160, 0, 16, 144),
    (129, 129, 0, 1, 128),
    (40, 40, 0, 0, 3),
    (95, 95, 0, 10, 90),
    (5, 5, 0, 1, 3),
    (37, 300, 263, 5, 262),
    (100, 230, 130, 1, 129),
];

fn run(case: (usize, usize, usize, usize, usize), wide: bool, options: PrefillOptions) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        let (queries, total, offset, image_start, image_end) = case;
        let (q, k, v, sinks) = operands(queries, total);
        let mut out = vec![f32::NAN; q.len()];
        // SAFETY: callers check AVX2/FMA and AVX-512F before selecting wide;
        // normalized heads bound the scores, and these are pure prefixes.
        unsafe {
            compact_prefill(
                &q,
                &CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM),
                queries,
                Geometry::new(offset, image_start, image_end),
                &sinks,
                &mut out,
                wide,
                options,
            );
        }
        out
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = wide;
        let tile: TileFn = if options.fp32_softmax == PrefillSoftmax::Static {
            tile_head_neon
        } else {
            super::super::tile_head_native
        };
        run_tile(case, tile)
    }
}

fn run_tile(case: (usize, usize, usize, usize, usize), tile: TileFn) -> Vec<f32> {
    let (queries, total, offset, image_start, image_end) = case;
    let (q, k, v, sinks) = operands(queries, total);
    let mut out = vec![f32::NAN; q.len()];
    // SAFETY: native callers check features; shapes are consistent.
    unsafe {
        compact_prefill_with(
            tile,
            &q,
            &CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM),
            queries,
            Geometry::new(offset, image_start, image_end),
            &sinks,
            &mut out,
            false,
        );
    }
    out
}

/// Bit for bit `exp_poly` over every 1021st FP32 value in [-87, 87], on 8
/// and 16 lanes.
#[cfg(target_arch = "x86_64")]
#[test]
fn unchecked_exps_match_exp_poly() {
    if !available() {
        return;
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn narrow(x: &[f32; 16]) -> [f32; 16] {
        let mut out = [0.0_f32; 16];
        // SAFETY: AVX2/FMA checked by the caller; both arrays hold 16 lanes.
        unsafe {
            for half in 0..2 {
                let lanes = crate::simd::Avx2::exp_unchecked(_mm256_loadu_ps(x.as_ptr().add(8 * half)));
                _mm256_storeu_ps(out.as_mut_ptr().add(8 * half), lanes);
            }
        }
        out
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn wide(x: &[f32; 16]) -> [f32; 16] {
        let mut out = [0.0_f32; 16];
        // SAFETY: AVX-512F checked by the caller; both arrays hold 16 lanes.
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), exp16(_mm512_loadu_ps(x.as_ptr()))) };
        out
    }
    let positive = (0..=87.0_f32.to_bits()).step_by(1021).map(f32::from_bits);
    let inputs: Vec<f32> = positive.flat_map(|x| [x, -x]).chain([87.0, -87.0]).collect();
    let mut checked = 0;
    for chunk in inputs.chunks_exact(16) {
        let x: [f32; 16] = chunk.try_into().unwrap();
        let expected = x.map(exp_poly);
        // SAFETY: features checked above.
        let mut forms = vec![unsafe { narrow(&x) }];
        if wide_available() {
            forms.push(unsafe { wide(&x) });
        }
        for got in forms {
            for ((x, e), g) in x.iter().zip(&expected).zip(&got) {
                assert_eq!(g.to_bits(), e.to_bits(), "exp({x})");
            }
        }
        checked += 16;
    }
    assert!(checked > 2_000_000);
}

/// The static tiles in scalar code: each score an FMA chain over the depth
/// from +0 times the scale, `exp_poly` of it (+0 where masked), the
/// denominators in key order, and per key tile a PV chain over its keys
/// from +0 added to the output; then normalization and the sink.
fn oracle(case: (usize, usize, usize, usize, usize)) -> Vec<f32> {
    let (queries, total, offset, image_start, image_end) = case;
    let (q, k, v, sinks) = operands(queries, total);
    let cache = CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM);
    let geometry = Geometry::new(offset, image_start, image_end);
    let shape = Shape {
        query_width: cache.query_width(),
        kv_width: cache.kv_width(),
        query_offset: offset,
        image_start,
        image_end,
        scale: 0.125,
        profile: false,
    };
    let mut out = vec![f32::NAN; q.len()];
    for head in 0..16 {
        for first in (0..queries).step_by(QUERY_TILE) {
            let rows = (queries - first).min(QUERY_TILE);
            let (_, visible_end, _) = tile_span(&shape, first, rows);
            for row in first..first + rows {
                let query = &q[row * shape.query_width + head * HEAD_DIM..][..HEAD_DIM];
                let (mut denominator, mut sum) = (0.0_f32, [0.0_f32; HEAD_DIM]);
                for key_start in (0..visible_end).step_by(KEY_TILE) {
                    let mut products = [0.0_f32; HEAD_DIM];
                    for key in key_start..(key_start + KEY_TILE).min(visible_end) {
                        let dot = query
                            .iter()
                            .zip(cache.key(key, head))
                            .fold(0.0, |a, (x, y)| x.mul_add(*y, a));
                        let p = if geometry.masked(offset + row, key) {
                            0.0
                        } else {
                            exp_poly(0.125 * dot)
                        };
                        denominator += p;
                        for (a, value) in products.iter_mut().zip(cache.value(key, head / 2)) {
                            *a = value.mul_add(p, *a);
                        }
                    }
                    // The kernels' `acc + out` (addition commutes bitwise).
                    for (s, a) in sum.iter_mut().zip(products) {
                        *s += a;
                    }
                }
                let sink_scale = 1.0 / (1.0 + (sinks[head] - denominator.ln()).exp());
                let dst = &mut out[row * shape.query_width + head * HEAD_DIM..][..HEAD_DIM];
                for (o, s) in dst.iter_mut().zip(sum) {
                    *o = (s / denominator) * sink_scale;
                }
            }
        }
    }
    out
}

/// The 8- and 16-lane tiles give the scalar oracle's outputs bit for bit on
/// every case, under either `ExpMode`.
#[test]
fn static_tiles_match_the_scalar_oracle_bitwise() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for case in CASES {
        let expected = oracle(case);
        let portable = pool.install(|| run_tile(case, tile_head_portable));
        assert_eq!(
            portable.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "portable {case:?}"
        );
        if !available() {
            continue;
        }
        for wide in [false, true] {
            if wide && !wide_available() {
                continue;
            }
            for exp in [ExpMode::Exact, ExpMode::Fast] {
                let got = pool.install(|| run(case, wide, options(exp)));
                for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "{case:?} wide={wide} {exp:?} index {i}: {g} vs {e}"
                    );
                }
            }
        }
    }
}

/// Options for the pure-prefix tiles do not affect generated-key caches
/// or calls too short for those tiles. These go through the safe dispatcher.
#[test]
fn static_options_leave_fallbacks_unchanged() {
    use crate::kernels::{Simd, attention_with};
    for queries in [1, 2, 3, 4, 5] {
        for generated in [false, true] {
            if !generated && queries >= 4 {
                continue; // Pure prefixes of this length use the static tiles.
            }
            let prefix = if generated { 37 } else { queries };
            let total = if generated { prefix + queries } else { prefix };
            let (q, k, v, sinks) = operands(queries, total);
            let generated_k = &k[prefix * 16 * HEAD_DIM..][..(total - prefix) * 8 * HEAD_DIM];
            let cache = CompactKv {
                prefix_k: &k[..prefix * 16 * HEAD_DIM],
                generated_k,
                v: &v,
                prefix_len: prefix,
                total_len: total,
                n_heads: 16,
                n_kv_heads: 8,
                head_dim: HEAD_DIM,
            };
            let geometry = Geometry::new(if generated { prefix } else { 0 }, 0, prefix);
            for simd in [Simd::Auto, Simd::Scalar] {
                let mut expected = vec![f32::NAN; q.len()];
                attention_with(
                    &q,
                    &cache,
                    queries,
                    geometry,
                    &sinks,
                    &mut expected,
                    simd,
                    PrefillOptions::EXACT,
                );
                for exp in [ExpMode::Exact, ExpMode::Fast] {
                    let mut got = vec![f32::NAN; q.len()];
                    attention_with(&q, &cache, queries, geometry, &sinks, &mut got, simd, options(exp));
                    assert_eq!(
                        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "queries={queries} generated={generated} {simd:?} {exp:?}"
                    );
                }
            }
        }
    }
}

/// A NaN score stays NaN in the masked tiles (those not all image), as in
/// the all-image tiles, the online kernels and the scalar oracle: the -inf
/// mask's compare is unordered, so it zeroes only the masked lanes.
#[test]
fn nan_keys_propagate_in_masked_tiles() {
    if !available() {
        return;
    }
    let (queries, total, offset, image_start, image_end) = (70, 70, 0, 3, 65);
    let (q, mut k, v, sinks) = operands(queries, total);
    // Head 0 of key 66, text after the image: rows 66 to 69 see it, in the
    // query tile of rows 64 to 69, where rows 64 and 65 mask it.
    k[66 * 16 * HEAD_DIM] = f32::NAN;
    for wide in [false, true] {
        if wide && !wide_available() {
            continue;
        }
        let mut out = vec![0.0_f32; q.len()];
        #[cfg(target_arch = "aarch64")]
        let tile: TileFn = tile_head_neon;
        // SAFETY: features checked above (AVX-512F before `wide`); shapes
        // consistent.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            compact_prefill(
                &q,
                &CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM),
                queries,
                Geometry::new(offset, image_start, image_end),
                &sinks,
                &mut out,
                wide,
                options(ExpMode::Exact),
            );
            #[cfg(target_arch = "aarch64")]
            compact_prefill_with(
                tile,
                &q,
                &CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM),
                queries,
                Geometry::new(offset, image_start, image_end),
                &sinks,
                &mut out,
                false,
            );
        }
        for row in 0..queries {
            for head in 0..16 {
                let poisoned = head == 0 && row >= 66;
                let values = &out[(row * 16 + head) * HEAD_DIM..][..HEAD_DIM];
                assert!(
                    values.iter().all(|x| x.is_nan() == poisoned),
                    "wide={wide} row {row} head {head}"
                );
            }
        }
    }
}

/// Against an f64 reference with the exact exp: the static tiles are as
/// close as the online ones (printed for comparison), well inside 1e-5.
#[test]
fn matches_f64_reference() {
    if !available() {
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for case in CASES {
        let (queries, total, offset, image_start, image_end) = case;
        let (q, k, v, sinks) = operands(queries, total);
        let cache = CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM);
        let geometry = Geometry::new(offset, image_start, image_end);
        let got = pool.install(|| run(case, false, options(ExpMode::Fast)));
        let fast = PrefillOptions {
            exp: ExpMode::Fast,
            ..PrefillOptions::EXACT
        };
        let online = pool.install(|| run(case, false, fast));
        let (worst, worst_online) = reference_errors(&q, &cache, geometry, &sinks, &got, &online, 8.0 + 1e-4);
        assert!(worst <= 1e-5, "{case:?}: worst {worst}");
        eprintln!("{case:?}: worst {worst:.2e} (online softmax {worst_online:.2e})");
    }
}

/// Wall time of one layer's prefill attention at a full page's shape (6,544
/// rows, 6,534 image tokens, 16 heads): online (fast exp) against static
/// softmax, interleaved, 8- and 16-lane tiles, on the caller's threads.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "timing probe; run in release with --nocapture"]
fn static_softmax_timing_probe() {
    if !available() {
        return;
    }
    let (queries, image_end) = (6544, 6534);
    let q = normalized(queries * 16 * HEAD_DIM, 3);
    let k = normalized(queries * 16 * HEAD_DIM, 5);
    let v = values(queries * 8 * HEAD_DIM, 7, 3.0);
    let sinks = values(16, 11, 2.0);
    let mut out = vec![0.0_f32; q.len()];
    let fast = PrefillOptions {
        exp: ExpMode::Fast,
        ..PrefillOptions::EXACT
    };
    let mut time = |wide: bool, options: PrefillOptions| {
        let start = std::time::Instant::now();
        // SAFETY: features checked above (AVX-512F before `wide`); shapes
        // consistent.
        unsafe {
            compact_prefill(
                &q,
                &CompactKv::prefill(&k, &v, queries, 16, 8, HEAD_DIM),
                queries,
                Geometry::new(0, 0, image_end),
                &sinks,
                &mut out,
                wide,
                options,
            )
        };
        start.elapsed().as_secs_f64() * 1e3
    };
    let median = |x: &mut Vec<f64>| {
        x.sort_by(f64::total_cmp);
        x[x.len() / 2]
    };
    for wide in [false, true] {
        if wide && !wide_available() {
            continue;
        }
        let (mut online, mut fixed) = (Vec::new(), Vec::new());
        for _ in 0..8 {
            online.push(time(wide, fast));
            fixed.push(time(wide, options(ExpMode::Fast)));
        }
        eprintln!(
            "one layer, wide={wide}: online {:.1} ms, static {:.1} ms (medians of 8; {} threads)",
            median(&mut online),
            median(&mut fixed),
            rayon::current_num_threads()
        );
    }
}
