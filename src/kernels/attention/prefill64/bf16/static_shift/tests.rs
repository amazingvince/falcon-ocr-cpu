//! Tests of the static-shift softmax: its exp against a scalar oracle, the
//! fused epilogue against separate passes (bitwise), and the whole kernel
//! against an f64 reference; plus a timing probe.
use super::*;
use crate::config::PrefillSoftmax;
use crate::kernels::attention::prefill64::bf16::{compact_prefill, pair_rows, qk_lanes, to_bf16};
use crate::kernels::attention::prefill64::mask_lanes;
use crate::kernels::{Bf16Kv, CompactKv, Geometry, PrefillOptions};

fn available() -> bool {
    crate::kernels::panel_bf16::available()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("avx512bw")
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

/// [`exp2_lanes`] on one value: the same IEEE operations in the same order.
fn exp2_scalar(t: f32) -> f32 {
    let n = t.round_ties_even();
    let f = t - n;
    let p = EXP2[3].mul_add(f, EXP2[2]).mul_add(f, EXP2[1]).mul_add(f, EXP2[0]);
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Bit for bit the scalar sequence, +0 in masked lanes, and within 7.6e-5
/// (relative) of `2^t` over every 509th FP32 value in [-120, 120].
#[test]
fn exp2_lanes_match_the_scalar_oracle() {
    if !available() {
        return;
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn lanes(t: &[f32; 16], visible: u16) -> [f32; 16] {
        let mut out = [0.0_f32; 16];
        // SAFETY: AVX-512F checked by the caller; both arrays hold 16 lanes.
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), exp2_lanes(_mm512_loadu_ps(t.as_ptr()), visible)) };
        out
    }
    let positive = (0..=120.0_f32.to_bits()).step_by(509).map(f32::from_bits);
    let inputs: Vec<f32> = positive.flat_map(|t| [t, -t]).collect();
    let mut worst = 0.0_f64;
    for (i, chunk) in inputs.chunks_exact(16).enumerate() {
        let t: [f32; 16] = chunk.try_into().unwrap();
        let visible = (i as u16).wrapping_mul(40_503) | 1;
        // SAFETY: AVX-512F checked above.
        let got = unsafe { lanes(&t, visible) };
        for (lane, (&x, &p)) in t.iter().zip(&got).enumerate() {
            if visible & (1 << lane) == 0 {
                assert_eq!(p.to_bits(), 0, "masked lane at {x}");
                continue;
            }
            assert_eq!(p.to_bits(), exp2_scalar(x).to_bits(), "2^{x}");
            worst = worst.max((p as f64 / (x as f64).exp2() - 1.0).abs());
        }
    }
    assert!(worst < 7.6e-5, "worst relative error {worst:e}");
    eprintln!("exp2 polynomial: worst relative error {worst:.3e}");
}

/// Random prefill operands (q, k, v, sinks, key copies, value pairs) for 16
/// query and 8 KV heads, with normalized queries and keys.
#[allow(clippy::type_complexity)]
fn operands(query_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
    let (heads, kv_heads) = (16, 8);
    let q = normalized(query_len * heads * HEAD_DIM, 17 + query_len as u32);
    let k = normalized(query_len * heads * HEAD_DIM, 19 + query_len as u32);
    let v = values(query_len * kv_heads * HEAD_DIM, 23 + query_len as u32, 3.0);
    let sinks = values(heads, 29, 2.0);
    let pairs = query_len.div_ceil(2);
    let mut keys = vec![0_u32; heads * query_len * PAIRS];
    let mut value_pairs = vec![0_u32; kv_heads * pairs * HEAD_DIM];
    for head in 0..heads {
        for t in 0..query_len {
            let src = k[(t * heads + head) * HEAD_DIM..].as_ptr();
            // SAFETY: AVX512-BF16 checked by the callers; 64 values each way.
            unsafe { to_bf16(src, keys[(head * query_len + t) * PAIRS..].as_mut_ptr().cast()) };
        }
    }
    for kv_head in 0..kv_heads {
        for p in 0..pairs {
            let even = v[(2 * p * kv_heads + kv_head) * HEAD_DIM..].as_ptr();
            let odd = (2 * p + 1 < query_len).then(|| v[((2 * p + 1) * kv_heads + kv_head) * HEAD_DIM..].as_ptr());
            let dst = value_pairs[(kv_head * pairs + p) * HEAD_DIM..].as_mut_ptr();
            // SAFETY: as above.
            unsafe { pair_rows::<4>(even, odd, dst) };
        }
    }
    (q, k, v, sinks, keys, value_pairs)
}

const CASES: [(usize, usize, usize); 6] = [
    (70, 3, 65),
    (300, 5, 262),
    (257, 0, 257),
    (129, 1, 128),
    (40, 0, 3),
    (95, 10, 90),
];

/// The same tile in separate passes: the online kernel's `qk_lanes` with the
/// same exponent scale, the parent's `mask_lanes`, [`exp2_scalar`] per score
/// in key order, `pair_rows`, then the same PV and normalization.
#[target_feature(enable = "avx2,fma,avx512f,avx512bf16,avx512bw")]
unsafe fn separate_tile_head(
    operands: &Operands<'_>,
    shape: &Shape,
    tile: usize,
    queries: usize,
    head: usize,
    sinks: &[f32],
    output: *mut f32,
) {
    let first_query = tile * QUERY_TILE;
    let (first_absolute, visible_end, all_image) = tile_span(shape, first_query, queries);
    let query_pairs = query_pairs(operands.q, shape, first_query, queries, head);
    // SAFETY: the caller owns this tile's rows of `head`.
    let out = unsafe { output.add(first_query * shape.query_width + head * HEAD_DIM) };
    for r in 0..queries {
        unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) }.fill(0.0);
    }
    let mut denominators = [0.0_f32; QUERY_TILE];
    let mut st = [0.0_f32; KEY_TILE * QUERY_TILE];
    let mut pairs = [0_u32; KEY_TILE / 2 * QUERY_TILE];
    for key_start in (0..visible_end).step_by(KEY_TILE) {
        let keys = (visible_end - key_start).min(KEY_TILE);
        // SAFETY: as in `tile_head`.
        unsafe {
            qk_lanes(
                &query_pairs,
                queries,
                operands.keys.as_ptr().add(key_start * PAIRS),
                keys,
                exponent_scale(shape),
                &mut st,
            );
        }
        if !all_image {
            mask_lanes(shape, first_absolute, queries, key_start, keys, &mut st);
        }
        for j in 0..keys {
            for (r, denominator) in denominators.iter_mut().enumerate() {
                let s = &mut st[j * QUERY_TILE + r];
                *s = if *s == f32::NEG_INFINITY { 0.0 } else { exp2_scalar(*s) };
                *denominator += *s;
            }
        }
        for kp in 0..keys.div_ceil(2) {
            // SAFETY: rows 2 kp and 2 kp + 1 of `st` hold 32 lanes.
            unsafe {
                let odd = (2 * kp + 1 < keys).then(|| st.as_ptr().add((2 * kp + 1) * QUERY_TILE));
                pair_rows::<2>(
                    st.as_ptr().add(2 * kp * QUERY_TILE),
                    odd,
                    pairs.as_mut_ptr().add(kp * QUERY_TILE),
                );
            }
        }
        // SAFETY: as in `tile_head`.
        unsafe {
            pv_lanes(
                &pairs,
                queries,
                keys.div_ceil(2),
                operands.values.as_ptr().add(key_start / 2 * HEAD_DIM),
                out,
                shape.query_width,
            );
        }
    }
    for (r, &denominator) in denominators.iter().enumerate().take(queries) {
        let row = unsafe { std::slice::from_raw_parts_mut(out.add(r * shape.query_width), HEAD_DIM) };
        let logsumexp = denominator.ln();
        let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
        for value in row {
            *value = (*value / denominator) * sink_scale;
        }
    }
}

/// The fused epilogue (exp in registers, bit-mask visibility, `vpermw`
/// pairs) gives the separate passes' outputs bit for bit, for every tile
/// shape: short and odd key tiles, partial query tiles, text rows before and
/// after the image.
#[test]
fn fused_epilogue_matches_separate_passes() {
    if !available() {
        return;
    }
    let (heads, kv_heads) = (16, 8);
    for (query_len, image_start, image_end) in CASES {
        let (q, k, v, sinks, keys, value_pairs) = operands(query_len);
        let shape = Shape {
            query_width: heads * HEAD_DIM,
            kv_width: kv_heads * HEAD_DIM,
            query_offset: 0,
            image_start,
            image_end,
            scale: 0.125,
            profile: false,
        };
        let pairs = query_len.div_ceil(2);
        let mut fused = vec![f32::NAN; q.len()];
        let mut separate = vec![f32::NAN; q.len()];
        for head in 0..heads {
            let kv_head = head / 2;
            let operands = Operands {
                q: &q,
                k: &k,
                v: &v,
                keys: &keys[head * query_len * PAIRS..(head + 1) * query_len * PAIRS],
                values: &value_pairs[kv_head * pairs * HEAD_DIM..(kv_head + 1) * pairs * HEAD_DIM],
            };
            for tile in 0..query_len.div_ceil(QUERY_TILE) {
                let queries = (query_len - tile * QUERY_TILE).min(QUERY_TILE);
                // SAFETY: features checked above; each call owns its tile.
                unsafe {
                    tile_head(&operands, &shape, tile, queries, head, &sinks, fused.as_mut_ptr());
                    separate_tile_head(&operands, &shape, tile, queries, head, &sinks, separate.as_mut_ptr());
                }
            }
        }
        for (i, (a, b)) in fused.iter().zip(&separate).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "query_len {query_len} index {i}");
        }
    }
}

fn round(x: f32) -> f64 {
    half::bf16::from_f32(x).to_f64()
}

/// The whole kernel (through `compact_prefill`) against an f64 reference on
/// the same BF16 operands with the exact exp: within the BF16 rounding of
/// the probabilities (relative 2^-8) plus twice the polynomial's 7.5e-5,
/// times max |v| = 3, plus FP32 accumulation slack. Also prints the online
/// kernel's error on the same operands.
#[test]
fn matches_f64_reference() {
    if !available() {
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let (heads, kv_heads) = (16, 8);
    let bf16_exact = |x: Vec<f32>| -> Vec<f32> { x.iter().map(|&x| round(x) as f32).collect() };
    for (query_len, image_start, image_end) in CASES {
        let (q, k, v, sinks, _, _) = operands(query_len);
        // BF16-exact, so the online kernel's FP32 short tiles see the same
        // operands.
        let (q, k, v) = (bf16_exact(q), bf16_exact(k), bf16_exact(v));
        let run = |softmax| {
            let mut out = vec![f32::NAN; q.len()];
            let (mut scratch_keys, mut scratch_values) = (Vec::new(), Vec::new());
            let options = PrefillOptions {
                exp: crate::config::ExpMode::Fast,
                softmax,
                ..PrefillOptions::EXACT
            };
            // SAFETY: features checked above; shapes consistent.
            pool.install(|| unsafe {
                compact_prefill(
                    &q,
                    &CompactKv::prefill(&k, &v, query_len, heads, kv_heads, HEAD_DIM),
                    query_len,
                    Geometry::new(0, image_start, image_end),
                    &sinks,
                    &mut out,
                    Bf16Kv::Convert(&mut scratch_keys, &mut scratch_values),
                    options,
                )
            });
            out
        };
        let (out, online) = (run(PrefillSoftmax::Static), run(PrefillSoftmax::Online));
        let (mut worst, mut worst_online) = (0.0_f64, 0.0_f64);
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            for row in 0..query_len {
                let geometry = Geometry::new(0, image_start, image_end);
                let logits: Vec<(usize, f64)> = (0..query_len)
                    .filter(|&key| !geometry.masked(row, key))
                    .map(|key| {
                        let dot: f64 = (0..HEAD_DIM)
                            .map(|d| {
                                round(q[(row * heads + head) * HEAD_DIM + d])
                                    * round(k[(key * heads + head) * HEAD_DIM + d])
                            })
                            .sum();
                        (key, dot / 8.0)
                    })
                    .collect();
                assert!(logits.iter().all(|l| l.1.abs() <= 8.1));
                let denominator: f64 = logits.iter().map(|l| l.1.exp()).sum();
                let sink = 1.0 / (1.0 + (sinks[head] as f64 - denominator.ln()).exp());
                for d in 0..HEAD_DIM {
                    let sum: f64 = logits
                        .iter()
                        .map(|&(key, logit)| logit.exp() * round(v[(key * kv_heads + kv_head) * HEAD_DIM + d]))
                        .sum();
                    let expected = sum / denominator * sink;
                    let at = (row * heads + head) * HEAD_DIM + d;
                    worst = worst.max((out[at] as f64 - expected).abs());
                    worst_online = worst_online.max((online[at] as f64 - expected).abs());
                }
            }
        }
        assert!(
            worst <= 3.0 * (2f64.powi(-8) + 1.5e-4) + 1e-4,
            "query_len {query_len}: worst {worst}"
        );
        eprintln!("query_len {query_len}: worst {worst:.2e} (online softmax {worst_online:.2e})");
    }
}

/// Wall time of one layer's prefill attention at a full page's shape (6,544
/// rows, 6,534 image tokens, 16 heads), online against static softmax,
/// interleaved, on the caller's threads.
#[test]
#[ignore = "timing probe; run in release with --nocapture"]
fn static_softmax_timing_probe() {
    if !available() {
        return;
    }
    let (query_len, image_end, heads, kv_heads) = (6544, 6534, 16, 8);
    let q = normalized(query_len * heads * HEAD_DIM, 3);
    let k = normalized(query_len * heads * HEAD_DIM, 5);
    let v = values(query_len * kv_heads * HEAD_DIM, 7, 3.0);
    let sinks = values(heads, 11, 2.0);
    let mut out = vec![0.0_f32; q.len()];
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    let mut run = |softmax, convert: bool| {
        let options = PrefillOptions {
            exp: crate::config::ExpMode::Fast,
            softmax,
            ..PrefillOptions::EXACT
        };
        let start = std::time::Instant::now();
        // The page converts K/V in the fused QKV pass; time only the first
        // call's conversion.
        let kv = if convert {
            Bf16Kv::Convert(&mut keys, &mut values)
        } else {
            Bf16Kv::Converted(&keys, &values)
        };
        // SAFETY: features checked above; shapes consistent.
        unsafe {
            compact_prefill(
                &q,
                &CompactKv::prefill(&k, &v, query_len, heads, kv_heads, HEAD_DIM),
                query_len,
                Geometry::new(0, 0, image_end),
                &sinks,
                &mut out,
                kv,
                options,
            )
        };
        start.elapsed().as_secs_f64() * 1e3
    };
    eprintln!(
        "first call with K/V conversion: {:.1} ms",
        run(PrefillSoftmax::Online, true)
    );
    let (mut online, mut fixed) = (Vec::new(), Vec::new());
    for _ in 0..12 {
        online.push(run(PrefillSoftmax::Online, false));
        fixed.push(run(PrefillSoftmax::Static, false));
    }
    let median = |x: &mut Vec<f64>| {
        x.sort_by(f64::total_cmp);
        x[x.len() / 2]
    };
    eprintln!(
        "one layer: online {:.1} ms, static {:.1} ms (medians of 12; {} threads)",
        median(&mut online),
        median(&mut fixed),
        rayon::current_num_threads()
    );
}
