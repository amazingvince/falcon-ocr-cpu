//! The quantized layer against the FP32 product over its dequantized weights:
//! shapes, multi-row dots, the small-M path, the fused GLU, artifact checks
//! and a throughput probe.
use super::*;
use crate::kernels::Simd;

/// `w` (`[n][k]`) quantized with `columns` zeroed, as the overlay tools
/// do, and those columns kept exact; also the dequantized codes alone
/// (zero at the exception columns) and the exact values, row-major.
fn with_exact_columns(
    w: &[f32],
    n: usize,
    k: usize,
    group: usize,
    bits: u32,
    columns: &[i32],
) -> (QuantLinear, Vec<f32>, Vec<f32>) {
    let mut zeroed = w.to_vec();
    let mut values = Vec::new();
    for row in 0..n {
        for &column in columns {
            values.push(w[row * k + column as usize]);
            zeroed[row * k + column as usize] = 0.0;
        }
    }
    let q = QuantLinear::quantize_bits(&zeroed, n, k, group, bits).unwrap();
    let mut dense = vec![0.0; n * k];
    for r in 0..n {
        q.dequantize_row(r, &mut dense[r * k..(r + 1) * k]);
    }
    let q = q.with_exceptions(columns.to_vec(), values.clone()).unwrap();
    (q, dense, values)
}

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
            // Without and with exception columns (first, middle, last).
            let exceptions = [0, k as i32 / 3, k as i32 - 1];
            for q in [
                QuantLinear::quantize_bits(&w, 3, k, group, bits).unwrap(),
                with_exact_columns(&w, 3, k, group, bits, &exceptions).0,
            ] {
                let e = q.exception_columns().len();
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
                                assert_eq!(
                                    values[row].to_bits(),
                                    single.to_bits(),
                                    "k {k} bits {bits} rows {rows} exceptions {e}"
                                );
                            }
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
/// The oracle for decode with exception columns: the FP32 GEMV on the
/// dequantized codes (zero at the exceptions), then one `mul_add` per
/// exception column in ascending order.
#[test]
fn exception_decode_is_bitwise_the_documented_order() {
    for (n, k, group, columns) in [
        (37, 768, 64, vec![0]),
        (5, 1024, 64, vec![3, 511, 1023]),
        (3, 2304, 64, vec![17, 18, 1500, 2303]),
        (4, 65, 64, vec![64]),
        (2, 31, 32, vec![0, 30]),
        (37, 768, 32, vec![100, 700]),
    ] {
        // Large weights at the exception columns, as on the massive channels.
        let w: Vec<_> = (0..n * k)
            .map(|i| {
                ((i * 7919 % 2003) as f32 - 1001.0) / 977.0
                    * if columns.contains(&((i % k) as i32)) {
                        300.0
                    } else {
                        1.0
                    }
            })
            .collect();
        for bits in [8, 16] {
            let (q, dense, values) = with_exact_columns(&w, n, k, group, bits, &columns);
            let m = columns.len();
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
                    for r in 0..rows {
                        for c in 0..n {
                            let mut acc = expected[r * n + c];
                            for (i, &column) in columns.iter().enumerate() {
                                acc = x[r * k + column as usize].mul_add(values[c * m + i], acc);
                            }
                            expected[r * n + c] = acc;
                        }
                    }
                    let mut out = vec![f32::NAN; rows * n];
                    q.linear(&x, rows, &mut out, &mut Scratch::default(), backend).unwrap();
                    for (a, b) in out.iter().zip(&expected) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{backend:?} n{n} k{k} g{group} b{bits} rows{rows}"
                        );
                    }
                }
            }
        }
    }
}

/// Large-M products (prefill, and the dense fallback) use the exact
/// weights in place; with BF16 projections requested, a matrix with
/// exception columns keeps the FP32 panels.
#[test]
fn exception_large_m_and_prefill_use_the_exact_weights() {
    use crate::kernels::panel_gemm::Epilogue;
    for (n, k, columns) in [
        (32, 768, vec![5, 700]),
        (48, 2304, vec![0, 1234, 2303]),
        (16, 130, vec![129]),
    ] {
        let w: Vec<_> = (0..n * k)
            .map(|i| {
                ((i * 6007 % 1999) as f32 - 999.0) / 613.0
                    * if columns.contains(&((i % k) as i32)) {
                        500.0
                    } else {
                        1.0
                    }
            })
            .collect();
        // The reconstructed matrix: the codes' weights with the exact
        // values placed at the exception columns.
        let (q, mut exact, values) = with_exact_columns(&w, n, k, 64, 8, &columns);
        for r in 0..n {
            for (i, &column) in columns.iter().enumerate() {
                exact[r * k + column as usize] = values[r * columns.len() + i];
            }
        }
        for rows in [9, 17, 40] {
            let x: Vec<_> = (0..rows * k).map(|i| ((i * 31 % 151) as f32 - 75.0) / 97.0).collect();
            // The dense fallback is the FP32 product of the reconstructed matrix.
            let mut dense = vec![f32::NAN; rows * n];
            q.linear(&x, rows, &mut dense, &mut Scratch::default(), Simd::Scalar)
                .unwrap();
            let mut expected = vec![0.0; rows * n];
            crate::kernels::linear_with_simd(&x, rows, k, &exact, n, &mut expected, Simd::Scalar);
            assert!(dense.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
            if !q.panel_gemm(Simd::Auto) {
                continue;
            }
            let mut panel = vec![f32::NAN; rows * n];
            q.linear(&x, rows, &mut panel, &mut Scratch::default(), Simd::Auto)
                .unwrap();
            let (mut fp32, mut bf16) = (vec![f32::NAN; rows * n], vec![f32::NAN; rows * n]);
            q.prefill(
                &x,
                rows,
                None,
                Epilogue::Store(&mut fp32),
                &mut Scratch::default(),
                false,
            );
            q.prefill(
                &x,
                rows,
                None,
                Epilogue::Store(&mut bf16),
                &mut Scratch::default(),
                true,
            );
            assert!(bf16.iter().zip(&fp32).all(|(a, b)| a.to_bits() == b.to_bits()));
            for (r, c) in (0..rows).flat_map(|r| (0..n).map(move |c| (r, c))) {
                let products: Vec<f64> = (0..k).map(|j| x[r * k + j] as f64 * exact[c * k + j] as f64).collect();
                let sum: f64 = products.iter().sum();
                let bound = 4.0 * k as f64 * f32::EPSILON as f64 * products.iter().map(|p| p.abs()).sum::<f64>();
                for value in [panel[r * n + c], fp32[r * n + c]] {
                    assert!((value as f64 - sum).abs() <= bound + 1e-6, "n{n} k{k} rows{rows}");
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
        for q in [
            QuantLinear::quantize(&w, n, k, 64).unwrap(),
            with_exact_columns(&w, n, k, 64, 8, &[1, k as i32 / 2, k as i32 - 1]).0,
        ] {
            let e = q.exception_columns().len();
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
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{backend:?} ffn{ffn} rows{rows} exceptions {e}"
                        );
                    }
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
