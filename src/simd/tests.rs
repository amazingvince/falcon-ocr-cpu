//! Bitwise checks of the portable and native SIMD operations.
use super::*;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = ((state >> 8) as f64 / 16_777_216.0 * 8.0 - 4.0) as f32;
            match i % 53 {
                0 => -0.0,
                1 => f32::from_bits(5),
                _ => x,
            }
        })
        .collect()
}

#[test]
fn pair_and_tree_operations_agree_across_isas() {
    fn check<S: Simd>() {
        let a = values(8, 5);
        let b = values(8, 9);
        unsafe {
            let (va, vb) = (S::load(a.as_ptr()), S::load(b.as_ptr()));
            let mut out = [0.0_f32; 8];
            let expect = |name: &str, actual: [f32; 8], expected: [f32; 8]| {
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits), "{name}");
            };
            S::store(out.as_mut_ptr(), S::dup_even(va));
            expect("dup_even", out, std::array::from_fn(|i| a[i & !1]));
            S::store(out.as_mut_ptr(), S::dup_odd(va));
            expect("dup_odd", out, std::array::from_fn(|i| a[i | 1]));
            S::store(out.as_mut_ptr(), S::swap_pairs(va));
            expect("swap_pairs", out, std::array::from_fn(|i| a[i ^ 1]));
            S::store(out.as_mut_ptr(), S::addsub(va, vb));
            expect(
                "addsub",
                out,
                std::array::from_fn(|i| if i % 2 == 0 { a[i] - b[i] } else { a[i] + b[i] }),
            );
            assert_eq!(
                S::sum_tree(va).to_bits(),
                (((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]))).to_bits()
            );
        }
    }
    check::<Portable>();
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn native() {
            check::<Avx2>();
            check::<Avx2Fast>();
        }
        unsafe { native() };
    }
    #[cfg(target_arch = "aarch64")]
    check::<Neon>();
}

#[test]
fn fast_exp_is_identical_across_isas_and_accurate() {
    let mut inputs: Vec<f32> = (0..200_000).map(|i| -110.0 + 200.0 * (i as f32) / 200_000.0).collect();
    inputs.extend([
        0.0,
        -0.0,
        -87.0,
        -86.999,
        -87.001,
        88.375,
        88.377,
        88.72,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        -1e-30,
        1e-30,
    ]);
    let mut worst = 0.0_f64;
    for chunk in inputs.chunks(8) {
        let mut lanes = [0.0_f32; 8];
        lanes[..chunk.len()].copy_from_slice(chunk);
        let portable = unsafe { Portable::exp_fast(lanes) };
        let bounded = lanes.map(|x| if x.is_nan() { 0.0 } else { x.clamp(-87.0, 87.0) });
        let unchecked = unsafe { Portable::exp_unchecked(bounded) };
        assert_eq!(unchecked.map(f32::to_bits), bounded.map(exp_poly).map(f32::to_bits));
        for (x, y) in lanes.iter().zip(&portable) {
            let exact = f64::from(*x).exp();
            if x.is_finite() && *x >= EXP_MIN && *x <= EXP_MAX {
                // Finite and close everywhere up to the ceiling itself.
                assert!(y.is_finite(), "exp_poly({x}) = {y}");
                let ulp = f64::from(f32::EPSILON) * exact.abs().max(f64::from(f32::MIN_POSITIVE));
                worst = worst.max((f64::from(*y) - exact).abs() / ulp);
            } else if *x > EXP_MAX {
                assert_eq!(*y, f32::INFINITY, "exp_poly({x})");
            }
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn native(x: [f32; 8]) -> [f32; 8] {
                let mut out = [0.0_f32; 8];
                unsafe { Avx2::store(out.as_mut_ptr(), Avx2::exp_fast(Avx2::load(x.as_ptr()))) };
                out
            }
            #[target_feature(enable = "avx2,fma")]
            unsafe fn unchecked_native(x: [f32; 8]) -> [f32; 8] {
                let mut out = [0.0_f32; 8];
                // SAFETY: caller enables AVX2/FMA and both arrays hold 8 lanes.
                unsafe { Avx2::store(out.as_mut_ptr(), Avx2::exp_unchecked(Avx2::load(x.as_ptr()))) };
                out
            }
            assert_eq!(
                unsafe { unchecked_native(bounded) }.map(f32::to_bits),
                unchecked.map(f32::to_bits)
            );
            let avx2 = unsafe { native(lanes) };
            for (a, b) in avx2.iter().zip(&portable) {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let mut out = [0.0_f32; 8];
            // SAFETY: NEON is baseline and both arrays hold eight lanes.
            unsafe { Neon::store(out.as_mut_ptr(), Neon::exp_unchecked(Neon::load(bounded.as_ptr()))) };
            assert_eq!(out.map(f32::to_bits), unchecked.map(f32::to_bits));
            unsafe { Neon::store(out.as_mut_ptr(), Neon::exp_fast(Neon::load(lanes.as_ptr()))) };
            for (a, b) in out.iter().zip(&portable) {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
            }
        }
    }
    assert!(worst < 3.0, "exp_poly error {worst} ulp");
    assert_eq!(exp_poly(f32::NEG_INFINITY), 0.0);
    assert!(exp_poly(f32::NAN).is_nan());
    // The ceiling sits below the first input whose 2^n construction
    // overflows (88.3763); the largest finite f32 exp argument is 88.72.
    assert!(exp_poly(EXP_MAX).is_finite());
    assert_eq!(exp_poly(EXP_MAX.next_up()), f32::INFINITY);
    assert!((f64::from(EXP_MAX) * std::f64::consts::LOG2_E).round() <= 127.0);
}

/// Every native implementation equals the portable one bit for bit.
#[test]
fn native_matches_portable_bitwise() {
    let codes: Vec<i8> = (0..2400).map(|i| (i * 37 % 255 - 127) as i8).collect();
    let scales = values(2400 / 64 + 1, 9)
        .iter()
        .map(|s| s.abs() * 0.01)
        .collect::<Vec<_>>();
    for len in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200, 768, 1024, 2304] {
        let a = values(len, 1 + len as u32);
        let b = values(len, 7 + len as u32);
        let portable = unsafe { dot::<Portable>(&a, &b) };
        let portable_q8 = unsafe { dot_q::<Portable, i8, 64>(&a, &codes[..len], &scales) };
        let mut portable_y = b.clone();
        unsafe { axpy::<Portable>(0.37, &a, &mut portable_y) };
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn native(a: &[f32], b: &[f32], c: &[i8], s: &[f32], y: &mut [f32]) -> (f32, f32) {
                unsafe {
                    axpy::<Avx2>(0.37, a, y);
                    (dot::<Avx2>(a, b), dot_q::<Avx2, i8, 64>(a, c, s))
                }
            }
            let mut y = b.clone();
            let (d, q) = unsafe { native(&a, &b, &codes[..len], &scales, &mut y) };
            assert_eq!(d.to_bits(), portable.to_bits(), "dot len {len}");
            assert_eq!(q.to_bits(), portable_q8.to_bits(), "dot_q8 len {len}");
            assert!(y.iter().zip(&portable_y).all(|(u, v)| u.to_bits() == v.to_bits()));
            // The generic AVX2 instantiation is the hand-written kernel.
            let hand = crate::kernels::dot_kernel(crate::kernels::Simd::Avx2)(&a, &b);
            assert_eq!(d.to_bits(), hand.to_bits(), "hand-written dot len {len}");
        }
        #[cfg(target_arch = "aarch64")]
        {
            let mut y = b.clone();
            let (d, q) = unsafe {
                axpy::<Neon>(0.37, &a, &mut y);
                (dot::<Neon>(&a, &b), dot_q::<Neon, i8, 64>(&a, &codes[..len], &scales))
            };
            assert_eq!(d.to_bits(), portable.to_bits(), "dot len {len}");
            assert_eq!(q.to_bits(), portable_q8.to_bits(), "dot_q8 len {len}");
            assert!(y.iter().zip(&portable_y).all(|(u, v)| u.to_bits() == v.to_bits()));
        }
    }
}
