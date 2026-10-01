//! Unit tests of the quantized layer: codes and scales, error bounds, and
//! finite checks.
use super::*;

#[test]
fn codes_ties_zero_and_signed_endpoints() {
    let mut weights = vec![0.; 128];
    weights[..8].copy_from_slice(&[254., -254., 1., 3., 5., -1., -3., -5.]);
    let q = QuantLinear::quantize(&weights, 1, 128, 64).unwrap();
    assert_eq!(q.scales(), &[2., 0.]);
    assert_eq!(&q.codes()[..8], &[127, -127, 0, 2, 2, 0, -2, -2]);
    let mut restored = vec![0.; 128];
    q.dequantize_row(0, &mut restored);
    assert_eq!(&restored[..8], &[254., -254., 0., 4., 4., 0., -4., -4.]);
    assert!(restored[8..].iter().all(|&x| x == 0.));
}

#[test]
fn odd_rows_partial_groups_and_error_bound() {
    for k in [1usize, 31, 65, 127, 129, 257] {
        for g in [64, 128] {
            let weights: Vec<_> = (0..3 * k).map(|i| ((i * 137 % 127) as f32 - 63.) / 19.).collect();
            let q = QuantLinear::quantize(&weights, 3, k, g).unwrap();
            assert_eq!(q.payload_bytes(), 3 * k + 12 * k.div_ceil(g));
            for row in 0..3 {
                let mut restored = vec![0.; k];
                q.dequantize_row(row, &mut restored);
                for (column, &w) in restored.iter().enumerate() {
                    let scale = q.scales()[row * k.div_ceil(g) + column / g] as f64;
                    let original = weights[row * k + column] as f64;
                    assert!((w as f64 - original).abs() <= scale / 2. + 2. * f32::EPSILON as f64 * original.abs());
                }
            }
        }
    }
}

#[test]
fn real_reduction_widths_batches_and_f64_bound() {
    for k in [768usize, 1024, 2304] {
        for g in [64, 128] {
            let weights: Vec<_> = (0..7 * k).map(|i| ((i * 67 % 199) as f32 - 99.) / 31.).collect();
            let q = QuantLinear::quantize(&weights, 7, k, g).unwrap();
            for rows in [1, 2, 4, 8] {
                let x: Vec<_> = (0..rows * k).map(|i| ((i * 101 % 257) as f32 - 128.) / 37.).collect();
                let mut y = vec![0.; rows * 7];
                q.linear_f32(&x, rows, &mut y).unwrap();
                for channel in 0..7 {
                    let mut restored = vec![0.; k];
                    q.dequantize_row(channel, &mut restored);
                    for row in 0..rows {
                        let products: Vec<_> = x[row * k..(row + 1) * k]
                            .iter()
                            .zip(&restored)
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .collect();
                        let sum: f64 = products.iter().sum();
                        let absolute_sum: f64 = products.iter().map(|x| x.abs()).sum();
                        let ku = k as f64 * f32::EPSILON as f64 / 2.;
                        assert!((y[row * 7 + channel] as f64 - sum).abs() <= ku / (1. - ku) * absolute_sum);
                    }
                }
            }
        }
    }
}

#[test]
fn finite_checks_subnormals_and_overflow() {
    let tiny = f32::from_bits(1);
    let q = QuantLinear::quantize(&[tiny, -tiny], 1, 2, 64).unwrap();
    let mut restored = [0.; 2];
    q.dequantize_row(0, &mut restored);
    assert_eq!(restored, [tiny, -tiny]);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(QuantLinear::quantize(&[value], 1, 1, 64).is_err());
    }
    assert!(QuantLinear::quantize(&[], usize::MAX, 2, 64).is_err());
    assert!(QuantLinear::quantize(&[], 0, 0, 64).is_err());
    assert!(QuantLinear::quantize(&[1.], 1, 1, 16).is_err());
    let q = QuantLinear::quantize(&[2., 2.], 1, 2, 64).unwrap();
    assert!(q.linear_f32(&[f32::NAN, 1.], 1, &mut [0.]).is_err());
    assert!(q.linear_f32(&[f32::MAX, f32::MAX], 1, &mut [0.]).is_err());
    assert!(q.linear_f32(&[], usize::MAX, &mut []).is_err());
}
