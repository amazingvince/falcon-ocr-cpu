//! The quantized layer against the FP32 product over its dequantized weights:
//! shapes, multi-row dots, the small-M path, the fused GLU, artifact checks
//! and a throughput probe.
use super::*;
use crate::kernels::Simd;
#[test]
fn integrated_shapes_and_phase_values() {
    for k in [31, 64, 65, 768, 1024, 2304] {
        let n = 9;
        let w: Vec<_> = (0..n * k).map(|i| ((i * 137 % 193) as f32 - 96.0) / 113.0).collect();
        let q = QuantLinear::quantize(&w, n, k, 64).unwrap();
        for rows in [1, 2, 4, 8, 9, 17] {
            let x: Vec<_> = (0..rows * k).map(|i| ((i * 31 % 151) as f32 - 75.0) / 97.0).collect();
            let mut dense = vec![0.0; n * k];
            for r in 0..n {
                q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
            }
            for backend in [Simd::Scalar, Simd::Auto] {
                let mut out = vec![0.0; rows * n];
                let mut scratch = Scratch::default();
                q.linear(&x, rows, &mut out, &mut scratch, backend).unwrap();
                for r in 0..rows {
                    for c in 0..n {
                        let mut expected = 0.0_f64;
                        let mut magnitude = 0.0_f64;
                        for j in 0..k {
                            let p = x[r * k + j] as f64 * dense[c * k + j] as f64;
                            expected += p;
                            magnitude += p.abs();
                        }
                        assert!(
                            (out[r * n + c] as f64 - expected).abs()
                                <= 4.0 * k as f64 * f32::EPSILON as f64 * magnitude + 1e-6
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn dot_rows_are_bitwise_single_dots() {
    for (k, group) in [
        (768, 64),
        (1024, 64),
        (2304, 64),
        (65, 64),
        (31, 32),
        (100, 32),
        (768, 128),
    ] {
        let x: Vec<f32> = (0..8 * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
        for bits in [8, 16] {
            let w: Vec<f32> = (0..3 * k).map(|i| ((i * 13 % 89) as f32 - 44.0) / 23.0).collect();
            let q = QuantLinear::quantize_bits(&w, 3, k, group, bits).unwrap();
            for simd in [crate::kernels::Simd::Auto, crate::kernels::Simd::Scalar] {
                let (dot, dot_rows) = (q.dot_fn(simd), q.dot_rows_fn(simd));
                for rows in 1..=8 {
                    for channel in 0..3 {
                        let mut values = [f32::NAN; 8];
                        if !q.rows_dot(dot_rows, &x[..rows * k], rows, channel, &mut values) {
                            continue;
                        }
                        for row in 0..rows {
                            let single = q.row_dot(dot, &x[row * k..(row + 1) * k], channel);
                            assert_eq!(values[row].to_bits(), single.to_bits(), "k {k} bits {bits} rows {rows}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn small_m_is_bitwise_fp32_on_dequantized_weights() {
    // The oracle for every W8 decode result: the FP32 GEMV on fl(code*scale).
    for (n, k, group) in [
        (37, 768, 64),
        (5, 1024, 64),
        (3, 2304, 64),
        (4, 65, 64),
        (2, 31, 64),
        (37, 768, 32),
        (3, 2304, 32),
        (4, 65, 32),
    ] {
        let w: Vec<_> = (0..n * k)
            .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 977.0)
            .collect();
        let q = QuantLinear::quantize(&w, n, k, group).unwrap();
        let mut dense = vec![0.0; n * k];
        for r in 0..n {
            q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
        }
        for backend in [Simd::Scalar, Simd::Auto, Simd::Avx2]
            .into_iter()
            .filter(|s| s.validate().is_ok())
        {
            for rows in 1..=8 {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 104729 % 1009) as f32 - 504.0) / 311.0)
                    .collect();
                let mut expected = vec![0.0; rows * n];
                crate::kernels::linear_with_simd(&x, rows, k, &dense, n, &mut expected, backend);
                let mut out = vec![f32::NAN; rows * n];
                q.linear(&x, rows, &mut out, &mut Scratch::default(), backend).unwrap();
                for (a, b) in out.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{backend:?} n{n} k{k} g{group} rows{rows}");
                }
            }
        }
    }
}
#[test]
fn w16_small_m_is_bitwise_fp32_on_dequantized_weights() {
    for (n, k) in [(37, 768), (5, 1024), (3, 2304), (4, 65)] {
        let w: Vec<_> = (0..n * k)
            .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 977.0)
            .collect();
        let q = QuantLinear::quantize_bits(&w, n, k, 64, 16).unwrap();
        assert_eq!(q.bits(), 16);
        assert_eq!(q.payload_bytes(), 2 * n * k + 4 * n * k.div_ceil(64));
        let mut dense = vec![0.0; n * k];
        for r in 0..n {
            q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
        }
        // 16-bit codes: every weight within half a step of 1/32767 of its group's absmax.
        for (a, b) in dense.iter().zip(&w) {
            assert!((a - b).abs() <= 1.0 / 32767.0 * 1.03 + 1e-7, "{a} vs {b}");
        }
        for backend in [Simd::Scalar, Simd::Auto, Simd::Avx2]
            .into_iter()
            .filter(|s| s.validate().is_ok())
        {
            for rows in 1..=8 {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 104729 % 1009) as f32 - 504.0) / 311.0)
                    .collect();
                let mut expected = vec![0.0; rows * n];
                crate::kernels::linear_with_simd(&x, rows, k, &dense, n, &mut expected, backend);
                let mut out = vec![f32::NAN; rows * n];
                q.linear(&x, rows, &mut out, &mut Scratch::default(), backend).unwrap();
                for (a, b) in out.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{backend:?} n{n} k{k} rows{rows}");
                }
            }
        }
    }
}
#[test]
fn fused_glu_is_bitwise_linear_then_gate() {
    for (ffn, k) in [(19, 768), (5, 65), (3, 2304)] {
        let n = 2 * ffn;
        let w: Vec<_> = (0..n * k).map(|i| ((i * 6007 % 1999) as f32 - 999.0) / 613.0).collect();
        let q = QuantLinear::quantize(&w, n, k, 64).unwrap();
        for backend in [Simd::Scalar, Simd::Auto] {
            for rows in 1..=8 {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 7727 % 997) as f32 - 498.0) / 211.0)
                    .collect();
                let mut packed = vec![0.0; rows * n];
                q.linear(&x, rows, &mut packed, &mut Scratch::default(), backend)
                    .unwrap();
                let mut expected = vec![0.0; rows * ffn];
                crate::kernels::squared_relu_gate(&packed, &mut expected);
                let mut fused = vec![f32::NAN; rows * ffn];
                assert!(
                    q.linear_glu(&x, rows, &mut fused, &mut Scratch::default(), backend)
                        .unwrap()
                );
                for (a, b) in fused.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{backend:?} ffn{ffn} rows{rows}");
                }
            }
        }
    }
}
/// Decode-shaped GEMV throughput, cold in cache (working set > L3).
#[test]
#[ignore = "timing probe; run in release with --nocapture"]
fn decode_gemv_throughput_probe() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
    for (name, n, k) in [
        ("wo", 768, 1024),
        ("qkv", 2048, 768),
        ("w13", 4608, 768),
        ("w2", 768, 2304),
    ] {
        let copies = (96 << 20) / (n * k) + 1;
        let mats: Vec<QuantLinear> = (0..copies)
            .map(|c| {
                let w: Vec<f32> = (0..n * k)
                    .map(|i| (((i + c) * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
                    .collect();
                QuantLinear::quantize(&w, n, k, 64).unwrap()
            })
            .collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 / 13.0).collect();
        let mut out = vec![0.0; n];
        pool.install(|| {
            for m in &mats {
                m.linear(&x, 1, &mut out, &mut Scratch::default(), Simd::Auto).unwrap();
            }
            let rounds = 5;
            let t = std::time::Instant::now();
            for _ in 0..rounds {
                for m in &mats {
                    m.linear(&x, 1, &mut out, &mut Scratch::default(), Simd::Auto).unwrap();
                }
            }
            let calls = (rounds * mats.len()) as f64;
            let us = t.elapsed().as_secs_f64() * 1e6 / calls;
            let bytes = (n * k + n * k.div_ceil(64) * 4) as f64;
            println!("{name}: {us:.1} us/call, {:.1} GB/s", bytes / (us * 1e3));
        });
    }
}
#[test]
fn artifact_validation() {
    assert!(QuantLinear::from_parts(1, 1, 64, vec![-128], vec![1.0]).is_err());
    assert!(QuantLinear::from_parts(1, 1, 64, vec![1], vec![0.0]).is_err());
    assert!(QuantLinear::from_parts(1, 1, 64, vec![0], vec![f32::NAN]).is_err());
}
