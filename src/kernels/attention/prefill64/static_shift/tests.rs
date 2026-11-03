//! Tests of the FP32 static-shift tiles: the unchecked exps against
//! `exp_poly`, the 8- and 16-lane tiles against a scalar oracle (bitwise),
//! and the kernel against an f64 reference; plus a timing probe.
use super::*;
use crate::config::{ExpMode, PrefillSoftmax};
use crate::kernels::attention::prefill64::compact_prefill;
use crate::kernels::{Geometry, PrefillOptions};
use crate::simd::exp_poly;

fn available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn wide_available() -> bool {
    available() && std::is_x86_feature_detected!("avx512f")
}

/// The static tiles with the given exp (which they ignore).
fn options(exp: ExpMode) -> PrefillOptions {
    PrefillOptions {
        exp,
        fp32_softmax: PrefillSoftmax::Static,
        ..PrefillOptions::EXACT
    }
}

/// A fixed generator in [-range, range).
fn values(n: usize, seed: u32, range: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32 * range
        })
        .collect()
}

/// Random 64-wide heads scaled to norm 8, as the model's per-head RMS norms
/// leave queries and keys.
fn normalized(n: usize, seed: u32) -> Vec<f32> {
    let mut x = values(n, seed, 1.0);
    for head in x.chunks_exact_mut(HEAD_DIM) {
        let norm = head.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in head {
            *v *= 8.0 / norm;
        }
    }
    x
}

/// Prefill operands (q, k, v, sinks) for `queries` queries over `total`
/// cached rows, 16 query and 8 KV heads, normalized queries and keys.
fn operands(queries: usize, total: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let seed = 17 + total as u32;
    (
        normalized(queries * 16 * HEAD_DIM, seed),
        normalized(total * 16 * HEAD_DIM, seed + 2),
        values(total * 8 * HEAD_DIM, seed + 6, 3.0),
        values(16, 29, 2.0),
    )
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
    let (queries, total, offset, image_start, image_end) = case;
    let (q, k, v, sinks) = operands(queries, total);
    let mut out = vec![f32::NAN; q.len()];
    // SAFETY: the callers check the features; shapes are consistent.
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

/// Bit for bit `exp_poly` over every 1021st FP32 value in [-87, 87], on 8
/// and 16 lanes.
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
                let lanes = exp8(_mm256_loadu_ps(x.as_ptr().add(8 * half)));
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
    if !available() {
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for case in CASES {
        let expected = oracle(case);
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
        // SAFETY: features checked above (AVX-512F before `wide`); shapes
        // consistent.
        unsafe {
            compact_prefill(
                &q,
                &CompactKv::prefill(&k, &v, total, 16, 8, HEAD_DIM),
                queries,
                Geometry::new(offset, image_start, image_end),
                &sinks,
                &mut out,
                wide,
                options(ExpMode::Fast),
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
        let (mut worst, mut worst_online) = (0.0_f64, 0.0_f64);
        for head in 0..16 {
            for row in 0..queries {
                let absolute = offset + row;
                let logits: Vec<(usize, f64)> = (0..geometry.visible_end(absolute))
                    .filter(|&key| !geometry.masked(absolute, key))
                    .map(|key| {
                        let query = &q[(row * 16 + head) * HEAD_DIM..][..HEAD_DIM];
                        let dot: f64 = query
                            .iter()
                            .zip(cache.key(key, head))
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum();
                        (key, dot / 8.0)
                    })
                    .collect();
                assert!(logits.iter().all(|l| l.1.abs() <= 8.0 + 1e-4));
                let denominator: f64 = logits.iter().map(|l| l.1.exp()).sum();
                let sink = 1.0 / (1.0 + (sinks[head] as f64 - denominator.ln()).exp());
                for d in 0..HEAD_DIM {
                    let sum: f64 = logits
                        .iter()
                        .map(|&(key, logit)| logit.exp() * cache.value(key, head / 2)[d] as f64)
                        .sum();
                    let expected = sum / denominator * sink;
                    let at = (row * 16 + head) * HEAD_DIM + d;
                    worst = worst.max((got[at] as f64 - expected).abs());
                    worst_online = worst_online.max((online[at] as f64 - expected).abs());
                }
            }
        }
        assert!(worst <= 1e-5, "{case:?}: worst {worst}");
        eprintln!("{case:?}: worst {worst:.2e} (online softmax {worst_online:.2e})");
    }
}

/// Wall time of one layer's prefill attention at a full page's shape (6,544
/// rows, 6,534 image tokens, 16 heads): online (fast exp) against static
/// softmax, interleaved, 8- and 16-lane tiles, on the caller's threads.
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
