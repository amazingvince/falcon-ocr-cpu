//! INT16 panel tests: the VNNI and AVX2 kernels against the scalar oracle,
//! the error bound against exact products and the FP32 panels, edge rows,
//! the quantizer, and an ignored throughput probe.
use super::*;
use crate::kernels::{Simd, panel_gemm};

/// Deterministic noise in [-0.5, 0.5).
fn noise(i: usize, seed: u64) -> f32 {
    let h = (i as u64 ^ (seed << 40)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (h >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
}

/// Activations with a 700x outlier channel (as in the model's W2 inputs)
/// and an all-zero first group in row 2.
fn activations(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let v = noise(i, 1) * 4.0;
            if r == 2 && c < GROUP {
                0.0
            } else if c == 37 {
                v * 700.0
            } else {
                v
            }
        })
        .collect()
}

/// 8-bit codes in [-127, 127] and positive group scales.
fn weights(n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let codes = (0..n * k)
        .map(|i| (noise(i, 2) * 254.0).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let scales = (0..n * k / GROUP).map(|i| 0.001 + noise(i, 3).abs() * 0.01).collect();
    (codes, scales)
}

fn isas() -> impl Iterator<Item = Isa> {
    [Isa::Vnni, Isa::Avx2, Isa::Scalar]
        .into_iter()
        .filter(|isa| isa.available())
}

/// Store (with `row_scale`), gate (without) and residual-add outputs.
fn outputs<const NR: usize>(
    isa: Isa,
    a: &[f32],
    m: usize,
    k: usize,
    panels: &Panels,
    n: usize,
    row_scale: &[f32],
    pool: &rayon::ThreadPool,
) -> [Vec<f32>; 3] {
    let mut store = vec![f32::NAN; m * n];
    let mut gated = vec![f32::NAN; m * n / 2];
    let mut added: Vec<f32> = (0..m * n).map(|i| (i % 13) as f32 - 6.0).collect();
    pool.install(|| {
        gemm::<NR>(isa, a, m, k, panels, n, Some(row_scale), Epilogue::Store(&mut store));
        gemm::<NR>(isa, a, m, k, panels, n, None, Epilogue::Glu(&mut gated));
        gemm::<NR>(isa, a, m, k, panels, n, Some(row_scale), Epilogue::Add(&mut added));
    });
    [store, gated, added]
}

fn bits(outputs: &[Vec<f32>; 3]) -> Vec<Vec<u32>> {
    outputs
        .iter()
        .map(|o| o.iter().map(|v| v.to_bits()).collect())
        .collect()
}

fn agree<const NR: usize>(a: &[f32], m: usize, k: usize, codes: &[i8], scales: &[f32], n: usize, row_scale: &[f32]) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(5).build().unwrap();
    let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let mut panels = Panels::default();
    pack::<NR>(n, k, codes, scales, &mut panels);
    let oracle = outputs::<NR>(Isa::Scalar, a, m, k, &panels, n, row_scale, &pool);
    let [store, gated, added] = &oracle;
    // The epilogues are the FP32 panels' (gate of the stored pair, residual
    // plus the product).
    let unscaled = outputs::<NR>(Isa::Scalar, a, m, k, &panels, n, &vec![1.0; m], &pool);
    for r in 0..m {
        for i in 0..n / 2 {
            let (g, u) = (unscaled[0][r * n + 2 * i], unscaled[0][r * n + 2 * i + 1]);
            assert_eq!(
                gated[r * (n / 2) + i].to_bits(),
                crate::kernels::squared_relu_glu(g, u).to_bits()
            );
        }
    }
    for (i, (x, c)) in added.iter().zip(store).enumerate() {
        assert_eq!(x.to_bits(), ((i % 13) as f32 - 6.0 + c).to_bits());
    }
    for isa in isas() {
        for pool in [&pool, &single] {
            let got = outputs::<NR>(isa, a, m, k, &panels, n, row_scale, pool);
            assert!(bits(&got) == bits(&oracle), "{isa:?} NR {NR} m{m} k{k} n{n}");
        }
    }
}

/// The VNNI and AVX2 kernels (and the AVX2-compiled activation packing) give
/// the scalar oracle's bits for every epilogue, panel width and thread count.
#[test]
fn vector_kernels_are_bitwise_the_scalar_oracle() {
    for (m, k, n) in [(1, 64, 32), (13, 768, 64), (97, 1024, 32), (50, 2304, 64), (7, 128, 96)] {
        let a = activations(m, k);
        let (codes, scales) = weights(n, k);
        let row_scale: Vec<f32> = (0..m).map(|r| 0.25 + r as f32 / 9.0).collect();
        agree::<16>(&a, m, k, &codes, &scales, n, &row_scale);
        agree::<32>(&a, m, k, &codes, &scales, n, &row_scale);
    }
}

fn exception_agree<const NR: usize>(m: usize, k: usize, n: usize) {
    // Two pairs share a group, as they do in real overlays.
    let columns = [0, 1, (k / 2) as i32, (k / 2 + 1) as i32, (k - 1) as i32];
    let values: Vec<f32> = (0..n * columns.len()).map(|i| noise(i, 4) * 0.1).collect();
    let mut a = activations(m, k);
    let (mut codes, scales) = weights(n, k);
    for row in codes.chunks_exact_mut(k) {
        for &column in &columns {
            row[column as usize] = 0;
        }
    }
    for row in a.chunks_exact_mut(k) {
        for &column in &columns {
            row[column as usize] = 100_000.0 * (1.0 + noise(column as usize, 5));
        }
    }
    let mut zeroed = a.clone();
    for row in zeroed.chunks_exact_mut(k) {
        for &column in &columns {
            row[column as usize] = 0.0;
        }
    }
    let row_scale: Vec<f32> = (0..m).map(|r| 0.25 + r as f32 / 9.0).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(5).build().unwrap();
    let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let mut plain = Panels::default();
    pack::<NR>(n, k, &codes, &scales, &mut plain);
    let mut expected = outputs::<NR>(Isa::Scalar, &zeroed, m, k, &plain, n, &row_scale, &pool);
    let mut unscaled = vec![0.0_f32; m * n];
    pool.install(|| {
        gemm::<NR>(
            Isa::Scalar,
            &zeroed,
            m,
            k,
            &plain,
            n,
            None,
            Epilogue::Store(&mut unscaled),
        );
    });
    for r in 0..m {
        for c in 0..n {
            let index = r * n + c;
            for (i, &column) in columns.iter().enumerate() {
                let x = a[r * k + column as usize];
                let value = values[c * columns.len() + i];
                expected[0][index] = (x * row_scale[r]).mul_add(value, expected[0][index]);
                unscaled[index] = x.mul_add(value, unscaled[index]);
            }
            expected[2][index] = (index % 13) as f32 - 6.0 + expected[0][index];
        }
        for c in 0..n / 2 {
            expected[1][r * n / 2 + c] =
                crate::kernels::squared_relu_glu(unscaled[r * n + 2 * c], unscaled[r * n + 2 * c + 1]);
        }
    }
    let mut panels = Panels::default();
    pack::<NR>(n, k, &codes, &scales, &mut panels);
    pack_exceptions::<NR>(n, &columns, &values, &mut panels);
    let oracle = outputs::<NR>(Isa::Scalar, &a, m, k, &panels, n, &row_scale, &pool);
    assert_eq!(
        bits(&oracle),
        bits(&expected),
        "exception FMA order NR{NR} m{m} k{k} n{n}"
    );
    for isa in isas() {
        for pool in [&pool, &single] {
            let got = outputs::<NR>(isa, &a, m, k, &panels, n, &row_scale, pool);
            assert_eq!(bits(&got), bits(&oracle), "{isa:?} exceptions NR{NR} m{m} k{k} n{n}");
        }
    }
    let mut nonfinite = a.clone();
    nonfinite[columns[0] as usize] = f32::NAN;
    nonfinite[k + columns[2] as usize] = f32::INFINITY;
    for isa in isas() {
        let got = outputs::<NR>(isa, &nonfinite, m, k, &panels, n, &row_scale, &pool);
        for output in got {
            let width = output.len() / m;
            assert!(
                output[..2 * width].iter().all(|v| !v.is_finite()),
                "{isa:?} exception nonfinite"
            );
        }
    }
    // Scratch reuses the panels for a matrix without exception columns.
    pack::<NR>(n, k, &codes, &scales, &mut panels);
    assert!(panels.columns.is_empty() && panels.values.is_empty());
    let reused = outputs::<NR>(Isa::Scalar, &a, m, k, &panels, n, &row_scale, &pool);
    let fresh = outputs::<NR>(Isa::Scalar, &a, m, k, &plain, n, &row_scale, &pool);
    assert_eq!(bits(&reused), bits(&fresh));
}

/// Exception inputs do not coarsen their groups, then contribute ascending
/// FP32 FMAs before every epilogue, bitwise on every ISA and pool size.
#[test]
fn exception_columns_are_bitwise_the_scalar_oracle() {
    for (m, k, n) in [(13, 768, 64), (50, 2304, 64), (97, 1024, 32)] {
        exception_agree::<16>(m, k, n);
        exception_agree::<32>(m, k, n);
    }
}

/// Against exact products of the unquantized activations: the error is at
/// most the activation rounding, `0.51 * absmax_g * row_scale / 32767` per
/// unit of `sum_{k in g} |w_k|` (half a step, plus the FP32 rounding of the
/// reciprocal, the scaled value and the step), plus FP32 rounding of the
/// per-group epilogue, `4 (groups + 1) 2^-24` of `sum |x w|`. The FP32 panels
/// on the same weights stay within `k 2^-23` of that sum, so the two kernels
/// differ by at most the sum of both bounds.
#[test]
fn error_against_exact_and_fp32_panels_is_within_the_quantization_bound() {
    let isa = isas().next().unwrap();
    for (m, k, n) in [(13, 768, 64), (31, 2304, 32), (6, 1024, 64)] {
        let a = activations(m, k);
        let (codes, scales) = weights(n, k);
        let row_scale: Vec<f32> = (0..m).map(|r| 0.5 + r as f32 / 7.0).collect();
        let groups = k / GROUP;
        let mut panels = Panels::default();
        pack::<NR>(n, k, &codes, &scales, &mut panels);
        let mut got = vec![f32::NAN; m * n];
        gemm::<NR>(isa, &a, m, k, &panels, n, Some(&row_scale), Epilogue::Store(&mut got));
        let mut fp32 = vec![f32::NAN; m * n];
        if panel_gemm::available(Simd::Auto) {
            let mut dense = Vec::new();
            panel_gemm::pack_panels(
                n,
                k,
                |r, dst| {
                    for (i, d) in dst.iter_mut().enumerate() {
                        *d = codes[r * k + i] as f32 * scales[r * groups + i / GROUP];
                    }
                },
                &mut dense,
            );
            panel_gemm::gemm(
                &a,
                m,
                k,
                &dense,
                n,
                Some(&row_scale),
                panel_gemm::Epilogue::Store(&mut fp32),
            );
        }
        let (mut worst, mut err2, mut ref2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for r in 0..m {
            let x = &a[r * k..(r + 1) * k];
            let rs = row_scale[r] as f64;
            for c in 0..n {
                let (mut exact, mut mag, mut quant) = (0.0_f64, 0.0_f64, 0.0_f64);
                for g in 0..groups {
                    let absmax = x[g * GROUP..(g + 1) * GROUP]
                        .iter()
                        .fold(0.0_f64, |m, v| m.max(v.abs() as f64));
                    let mut wsum = 0.0_f64;
                    for kk in g * GROUP..(g + 1) * GROUP {
                        let w = codes[c * k + kk] as f64 * scales[c * groups + g] as f64;
                        exact += x[kk] as f64 * rs * w;
                        mag += (x[kk] as f64 * rs * w).abs();
                        wsum += w.abs();
                    }
                    quant += wsum * 0.51 * absmax * rs / 32767.0;
                }
                let bound = quant + 4.0 * (groups + 1) as f64 * 2.0_f64.powi(-24) * mag + 1e-30;
                let v = got[r * n + c] as f64;
                assert!((v - exact).abs() <= bound, "({r},{c}) {v} vs {exact}, bound {bound}");
                worst = worst.max((v - exact).abs() / bound);
                if panel_gemm::available(Simd::Auto) {
                    let f = fp32[r * n + c] as f64;
                    let fp32_bound = k as f64 * f32::EPSILON as f64 * mag;
                    assert!((v - f).abs() <= bound + fp32_bound, "({r},{c}) {v} vs FP32 panels {f}");
                    err2 += (v - f) * (v - f);
                    ref2 += f * f;
                }
            }
        }
        eprintln!(
            "m{m} k{k} n{n}: worst error / bound {worst:.3}, RMS difference from FP32 panels {:.2e} of RMS",
            (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt()
        );
    }
}

/// Non-finite rows give NaN and zero rows give zeros on every kernel.
/// Tiny groups agree bitwise with the scalar oracle even when their scale
/// products lose subnormal precision or underflow.
#[test]
fn non_finite_rows_are_nan_and_zero_rows_are_zero() {
    let (m, k, n) = (8, 128, 32);
    let mut a = activations(m, k);
    a[5] = f32::NAN;
    a[k + 70] = f32::INFINITY;
    a[3 * k..4 * k].fill(0.0);
    for (row, magnitude) in [1e-35, 1e-37, 1e-38, f32::from_bits(1)].into_iter().enumerate() {
        a[(row + 4) * k..(row + 5) * k].fill(magnitude);
    }
    let (codes, scales) = weights(n, k);
    let mut panels = Panels::default();
    pack::<NR>(n, k, &codes, &scales, &mut panels);
    let mut oracle = vec![0.0_f32; m * n];
    gemm::<NR>(Isa::Scalar, &a, m, k, &panels, n, None, Epilogue::Store(&mut oracle));
    for isa in isas() {
        let mut out = vec![0.0_f32; m * n];
        gemm::<NR>(isa, &a, m, k, &panels, n, None, Epilogue::Store(&mut out));
        for (r, row) in out.chunks_exact(n).enumerate() {
            match r {
                0 | 1 => assert!(row.iter().all(|v| v.is_nan()), "{isa:?} row {r}"),
                3 => assert!(row.iter().all(|v| v.to_bits() == 0), "{isa:?} row {r}"),
                4..=7 => {
                    assert!(row.iter().all(|v| v.is_finite()), "{isa:?} tiny row {r}");
                    if r == 4 {
                        assert!(row.iter().any(|&v| v != 0.0), "{isa:?} tiny row retains this magnitude");
                    } else if r == 7 {
                        assert!(row.iter().all(|&v| v == 0.0), "{isa:?} scale product underflows");
                    }
                    assert_eq!(
                        row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        oracle[r * n..(r + 1) * n]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "{isa:?} tiny row {r}"
                    );
                }
                _ => assert!(row.iter().all(|v| v.is_finite()), "{isa:?} row {r}"),
            }
        }
    }
    // Folded RMS row scales, both panel widths, every epilogue and pool size
    // must preserve the scalar arithmetic, including underflow.
    let tiny = &a[4 * k..];
    let row_scale = [1.0, 2.0, 1024.0, 1e6];
    agree::<16>(tiny, 4, k, &codes, &scales, n, &row_scale);
    agree::<32>(tiny, 4, k, &codes, &scales, n, &row_scale);
}

/// The quantizer's codes and step against an f64 rounding of `x / step`.
#[test]
fn quantizer_rounds_to_nearest_even_within_the_step() {
    let mut q = [0_i16; GROUP];
    for seed in 0..64 {
        let x: Vec<f32> = (0..GROUP).map(|i| noise(i, 10 + seed) * (seed as f32 + 0.5)).collect();
        let step = quantize_group(&x, &mut q);
        let absmax = x.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        assert_eq!(step, absmax / QMAX);
        assert!(q.iter().any(|&c| c.abs() == 32767));
        for (&c, &v) in q.iter().zip(&x) {
            let y = (v * (QMAX / absmax)) as f64;
            assert_eq!(c as f64, y.round_ties_even(), "{v}");
            assert!((c as f64 * step as f64 - v as f64).abs() <= 0.51 * step as f64);
        }
    }
    // Ties go to even.
    let mut x = [0.0_f32; GROUP];
    x[0] = 32767.0;
    (x[1], x[2], x[3]) = (2.5, 3.5, -2.5);
    assert_eq!(quantize_group(&x, &mut q), 1.0);
    assert_eq!(q[..4], [32767, 2, 4, -2]);
    // Tiny groups retain codes and a nonzero step when the reciprocal
    // overflows. Kernel scale products can still underflow independently.
    for absmax in [
        1e-35,
        1e-38,
        f32::from_bits(8192),
        f32::from_bits(32767),
        f32::from_bits(32768),
        f32::from_bits(1),
    ] {
        let x: Vec<f32> = (0..GROUP)
            .map(|i| absmax * (2.0 * i as f32 / (GROUP - 1) as f32 - 1.0))
            .collect();
        assert!(!(QMAX / absmax).is_finite());
        let step = quantize_group(&x, &mut q);
        assert!(step > 0.0 && step.is_finite());
        // The smallest covering step, including the minimum subnormal's
        // full-scale boundary; f64 keeps these products exact.
        assert!(step as f64 * 32767.0 >= absmax as f64, "{absmax:e}, step {step:e}");
        assert!(
            step.next_down() as f64 * 32767.0 < absmax as f64,
            "{absmax:e}, step {step:e} is too large"
        );
        assert!(q.iter().any(|&c| c != 0));
        for (&c, &v) in q.iter().zip(&x) {
            assert_eq!(
                c as f64,
                (v as f64 / step as f64).round_ties_even().clamp(-32767.0, 32767.0)
            );
            assert!((c as f64 * step as f64 - v as f64).abs() <= 0.51 * step as f64);
        }
    }
}

/// Median seconds of `reps` runs of `f` (after one warm-up) and the clock
/// in MHz while they ran, from Linux `/proc/cpuinfo` (0 when unreadable):
/// the mean over all CPUs, and the mean of each sample's fastest CPU (the
/// clock of a single busy thread).
fn timed(reps: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    f();
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let sampler = scope.spawn(|| {
            let (mut sum, mut count, mut top, mut samples) = (0.0, 0, 0.0, 0);
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
                    let clocks: Vec<f64> = info
                        .lines()
                        .filter_map(|l| l.strip_prefix("cpu MHz"))
                        .filter_map(|v| v.trim_start_matches([' ', '\t', ':']).trim().parse().ok())
                        .collect();
                    if !clocks.is_empty() {
                        sum += clocks.iter().sum::<f64>();
                        count += clocks.len();
                        top += clocks.iter().fold(0.0_f64, |m, &v| m.max(v));
                        samples += 1;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(15));
            }
            let mean = |total: f64, n: usize| if n == 0 { 0.0 } else { total / n as f64 };
            (mean(sum, count), mean(top, samples))
        });
        let mut times: Vec<f64> = (0..reps)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .collect();
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        times.sort_by(f64::total_cmp);
        let (mean, top) = sampler.join().unwrap();
        (times[reps / 2], mean, top)
    })
}

/// Prefill-shaped throughput of the INT16 kernels (both panel widths)
/// against the FP32 panels, in multiply-adds per cycle per physical core at
/// the sampled clock.
#[test]
#[ignore = "timing probe; run in release with --nocapture"]
fn i16_prefill_throughput_probe() {
    let m = 6544;
    let cores = crate::cpu::physical_cores() as f64;
    // Enough runs per arm (about 0.1-0.7 s) for the clock samples to cover
    // the steady state.
    let reps = 15;
    for (name, k, n) in [
        ("qkv", 768, 2048),
        ("wo", 1024, 768),
        ("w13", 768, 4608),
        ("w2", 2304, 768),
    ] {
        let a = activations(m, k);
        let (codes, scales) = weights(n, k);
        let groups = k / GROUP;
        let mut out = vec![0.0_f32; m * n];
        let macs = (m * k * n) as f64;
        let report = |label: &str, (s, mhz, _): (f64, f64, f64)| {
            println!(
                "{name} {label}: {:.2} ms, {:.0} GMAC/s, {:.1} MACs/cycle/core at {:.0} MHz",
                s * 1e3,
                macs / s / 1e9,
                macs / s / (cores * mhz * 1e6),
                mhz
            );
        };
        let dequantize = |r: usize, dst: &mut [f32]| {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = codes[r * k + i] as f32 * scales[r * groups + i / GROUP];
            }
        };
        if panel_gemm::available(Simd::Auto) {
            let mut dense = Vec::new();
            let t = timed(reps, || panel_gemm::pack_panels(n, k, dequantize, &mut dense));
            println!("{name} f32 pack: {:.2} ms", t.0 * 1e3);
            let t = timed(reps, || {
                panel_gemm::gemm(&a, m, k, &dense, n, None, panel_gemm::Epilogue::Store(&mut out))
            });
            report("f32 panels", t);
        }
        for isa in [Isa::Vnni, Isa::Avx2].into_iter().filter(|isa| isa.available()) {
            let mut panels = Panels::default();
            let t = timed(reps, || pack::<32>(n, k, &codes, &scales, &mut panels));
            println!("{name} i16 pack: {:.2} ms", t.0 * 1e3);
            let t = timed(reps, || {
                gemm::<32>(isa, &a, m, k, &panels, n, None, Epilogue::Store(&mut out))
            });
            report(&format!("i16 {isa:?} 6x32"), t);
            pack::<16>(n, k, &codes, &scales, &mut panels);
            let t = timed(reps, || {
                gemm::<16>(isa, &a, m, k, &panels, n, None, Epilogue::Store(&mut out))
            });
            report(&format!("i16 {isa:?} 6x16"), t);
        }
    }
    // One thread on one core (no SMT sibling): the kernels' own throughput.
    let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let (m, k, n) = (960, 768, 1024);
    let a = activations(m, k);
    let (codes, scales) = weights(n, k);
    let mut out = vec![0.0_f32; m * n];
    let macs = (m * k * n) as f64;
    let report = |label: &str, (s, _, mhz): (f64, f64, f64)| {
        println!(
            "single thread {label}: {:.2} ms, {:.1} MACs/cycle at {mhz:.0} MHz",
            s * 1e3,
            macs / s / (mhz * 1e6),
        );
    };
    if panel_gemm::available(Simd::Auto) {
        let mut dense = Vec::new();
        panel_gemm::pack_panels(
            n,
            k,
            |r, dst| {
                dst.iter_mut()
                    .enumerate()
                    .for_each(|(i, d)| *d = codes[r * k + i] as f32)
            },
            &mut dense,
        );
        let t = timed(reps, || {
            single.install(|| panel_gemm::gemm(&a, m, k, &dense, n, None, panel_gemm::Epilogue::Store(&mut out)))
        });
        report("f32 panels", t);
    }
    for isa in [Isa::Vnni, Isa::Avx2].into_iter().filter(|isa| isa.available()) {
        let mut panels = Panels::default();
        pack::<32>(n, k, &codes, &scales, &mut panels);
        let t = timed(reps, || {
            single.install(|| gemm::<32>(isa, &a, m, k, &panels, n, None, Epilogue::Store(&mut out)))
        });
        report(&format!("i16 {isa:?} 6x32"), t);
        pack::<16>(n, k, &codes, &scales, &mut panels);
        let t = timed(reps, || {
            single.install(|| gemm::<16>(isa, &a, m, k, &panels, n, None, Epilogue::Store(&mut out)))
        });
        report(&format!("i16 {isa:?} 6x16"), t);
    }
}
