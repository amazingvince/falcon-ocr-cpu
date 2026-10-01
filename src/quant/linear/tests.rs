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

#[test]
fn exception_columns_are_validated_counted_and_restored_exactly() {
    // Two rows of 70 inputs; columns 3 and 66 are exceptions (codes zero there).
    let (n, k) = (2, 70);
    let mut codes: Vec<i8> = (0..n * k).map(|i| (i % 11) as i8 - 5).collect();
    for row in 0..n {
        codes[row * k + 3] = 0;
        codes[row * k + 66] = 0;
    }
    let scales = vec![0.5; n * 2];
    let make = || QuantLinear::from_parts(n, k, 64, codes.clone(), scales.clone()).unwrap();
    let values = vec![1.0e3, -2.5, 7.0, 1.0e-3];
    for (columns, values, error) in [
        (vec![], vec![], "empty"),
        (vec![66, 3], values.clone(), "sorted"),
        (vec![3, 3], values.clone(), "sorted"),
        (vec![-1, 3], values.clone(), "outside"),
        (vec![3, 70], values.clone(), "outside"),
        (vec![3, 66], vec![1.0; 3], "shape"),
        (vec![3, 66], vec![1.0, f32::NAN, 1.0, 1.0], "nonfinite"),
        (vec![3, 5], values.clone(), "nonzero code"),
    ] {
        let message = make().with_exceptions(columns, values).unwrap_err().to_string();
        assert!(message.contains(error), "{message}");
    }
    let plain = make().payload_bytes();
    let q = make().with_exceptions(vec![3, 66], values.clone()).unwrap();
    assert_eq!(q.exception_columns(), &[3, 66]);
    assert_eq!(q.exception_bytes(), 4 * (2 + n * 2));
    assert_eq!(q.payload_bytes(), plain + q.exception_bytes());
    assert!(q.with_exceptions(vec![4], vec![1.0; n]).is_err(), "set twice");
    let q = make().with_exceptions(vec![3, 66], values.clone()).unwrap();
    let mut restored = vec![0.0; k];
    for row in 0..n {
        q.dequantize_row(row, &mut restored);
        for column in 0..k {
            let expected = match column {
                3 => values[2 * row],
                66 => values[2 * row + 1],
                _ => codes[row * k + column] as f32 * 0.5,
            };
            assert_eq!(restored[column].to_bits(), expected.to_bits());
        }
    }
    // The scalar reference sums the reconstructed weights in place.
    let x: Vec<f32> = (0..k).map(|i| (i as f32 - 30.0) / 17.0).collect();
    let mut y = [0.0; 2];
    q.linear_f32(&x, 1, &mut y).unwrap();
    for (row, &value) in y.iter().enumerate() {
        q.dequantize_row(row, &mut restored);
        let expected = x.iter().zip(&restored).fold(0.0f32, |sum, (a, &w)| a.mul_add(w, sum));
        assert_eq!(value.to_bits(), expected.to_bits());
    }
}
