//! Prefill projections `C[m][n] = sum_k A[m][k] * W[n][k]` for quantized
//! weights, replacing "dequantize the whole matrix, then a general GEMM".
//!
//! The weights are dequantized once per call into 16-column FP32 panels
//! (`[n/16][k][16]`), which stay cache-resident while every row block of `A`
//! streams past them. The AVX2/FMA micro-kernel keeps a 6x16 output tile in
//! twelve registers over the whole reduction (`k` is at most a few thousand
//! here), so partial sums never leave registers, and the epilogue writes the
//! tile directly in its final form: plain, or through the squared-ReLU gate
//! of interleaved `[gate, up]` channels (the W13 intermediate then never
//! reaches memory).
//!
//! Every output is one FMA chain over ascending `k` starting from zero:
//! deterministic and independent of the thread count, but not the summation
//! order of the `gemm` crate, so results differ from it at rounding level.
use rayon::prelude::*;

/// Output channels per panel.
pub(crate) const NR: usize = 16;
/// Rows per micro-kernel call.
const MR: usize = 6;

/// Where a finished 6x16 tile goes.
pub(crate) enum Epilogue<'a> {
    /// `out[m][n] = C[m][n]`.
    Store(&'a mut [f32]),
    /// `out[m][i] = squared_relu_glu(C[m][2i], C[m][2i + 1])` (`n / 2` columns).
    Glu(&'a mut [f32]),
    /// `out[m][n] += C[m][n]` (a residual add fused into the product).
    Add(&'a mut [f32]),
}

/// Whether [`gemm`] runs natively here for this backend (otherwise callers
/// keep their existing path). `FALCON_OCR_PANEL_GEMM=0` disables it (A/B).
pub(crate) fn available(simd: crate::kernels::Simd) -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var("FALCON_OCR_PANEL_GEMM").is_ok_and(|v| v == "0")) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        simd.resolved() != crate::kernels::Simd::Scalar
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = simd;
        false
    }
}

/// Dequantize an `n x k` matrix into panels: `row(r, dst)` writes output
/// channel `r` (`k` values). `n` must be a multiple of [`NR`].
pub(crate) fn pack_panels(
    n: usize,
    k: usize,
    row: impl Fn(usize, &mut [f32]) + Sync,
    panels: &mut Vec<f32>,
) {
    assert_eq!(n % NR, 0, "panel GEMM needs whole panels");
    panels.resize(n * k, 0.0);
    panels
        .par_chunks_mut(k * NR)
        .enumerate()
        .for_each_init(
            || vec![0.0_f32; k],
            |values, (panel, dst)| {
                for j in 0..NR {
                    row(panel * NR + j, values);
                    for (kk, &v) in values.iter().enumerate() {
                        dst[kk * NR + j] = v;
                    }
                }
            },
        );
}

/// `C = A * W^T` for `a` (`m x k`, row-major) and `panels` from
/// [`pack_panels`], written through `epilogue`. With `row_scale`, row `r` of
/// `A` is used as `a[r][k] * row_scale[r]` (an RMS norm folded into packing:
/// the same product `rms_norm_row` stores). Requires [`available`].
pub(crate) fn gemm(
    a: &[f32],
    m: usize,
    k: usize,
    panels: &[f32],
    n: usize,
    row_scale: Option<&[f32]>,
    epilogue: Epilogue<'_>,
) {
    assert_eq!(a.len(), m * k, "panel GEMM input shape");
    assert_eq!(panels.len(), n * k, "panel GEMM weight shape");
    assert_eq!(n % NR, 0, "panel GEMM needs whole panels");
    if let Some(scale) = row_scale {
        assert_eq!(scale.len(), m, "panel GEMM row scales");
    }
    let (out, glu, add) = match epilogue {
        Epilogue::Store(out) => {
            assert_eq!(out.len(), m * n, "panel GEMM output shape");
            (out, false, false)
        }
        Epilogue::Glu(out) => {
            assert_eq!(out.len(), m * (n / 2), "panel GEMM gated shape");
            (out, true, false)
        }
        Epilogue::Add(out) => {
            assert_eq!(out.len(), m * n, "panel GEMM output shape");
            (out, false, true)
        }
    };
    if m == 0 || n == 0 {
        return;
    }
    let out_width = if glu { n / 2 } else { n };
    // Row blocks sized so the packed block stays in L2 (about 1 MB on Zen 4).
    let block_rows = if k <= 1024 { 96 } else { 48 };
    let row_blocks = m.div_ceil(block_rows);
    let panels_total = n / NR;
    let threads = rayon::current_num_threads().max(1);
    let splits = (2 * threads).div_ceil(row_blocks).clamp(1, panels_total);
    let panels_per_split = panels_total.div_ceil(splits);
    let shared = crate::team::SharedMut::new(out);
    (0..row_blocks * splits).into_par_iter().for_each_init(
        || vec![0.0_f32; block_rows.next_multiple_of(MR) * k],
        |packed, unit| {
            let (block, split) = (unit / splits, unit % splits);
            let row0 = block * block_rows;
            let rows = block_rows.min(m - row0);
            let groups = rows.div_ceil(MR);
            // Pack the row block as [group][k][6], zero-padding the last group.
            for g in 0..groups {
                let dst = &mut packed[g * k * MR..(g + 1) * k * MR];
                for r in 0..MR {
                    let src_row = row0 + g * MR + r;
                    if src_row < row0 + rows {
                        let src = &a[src_row * k..(src_row + 1) * k];
                        match row_scale {
                            Some(scale) => {
                                let s = scale[src_row];
                                for (kk, &v) in src.iter().enumerate() {
                                    dst[kk * MR + r] = v * s;
                                }
                            }
                            None => {
                                for (kk, &v) in src.iter().enumerate() {
                                    dst[kk * MR + r] = v;
                                }
                            }
                        }
                    } else {
                        for kk in 0..k {
                            dst[kk * MR + r] = 0.0;
                        }
                    }
                }
            }
            let first_panel = split * panels_per_split;
            let last_panel = (first_panel + panels_per_split).min(panels_total);
            let mut tile = [[0.0_f32; NR]; MR];
            for panel in first_panel..last_panel {
                let b = &panels[panel * k * NR..(panel + 1) * k * NR];
                for g in 0..groups {
                    let a_group = &packed[g * k * MR..(g + 1) * k * MR];
                    // SAFETY: `available` was checked by the caller; both
                    // packed operands hold exactly `k` steps.
                    unsafe { kernel(k, a_group, b, &mut tile) };
                    let valid = MR.min(rows - g * MR);
                    for (r, values) in tile[..valid].iter().enumerate() {
                        let row = row0 + g * MR + r;
                        if glu {
                            let base = row * out_width + panel * (NR / 2);
                            // SAFETY: (row, panel) tiles are disjoint across units.
                            let dst = unsafe { shared.slice(base, NR / 2) };
                            for (i, y) in dst.iter_mut().enumerate() {
                                *y = crate::kernels::squared_relu_glu(values[2 * i], values[2 * i + 1]);
                            }
                        } else {
                            let base = row * out_width + panel * NR;
                            // SAFETY: (row, panel) tiles are disjoint across units.
                            let dst = unsafe { shared.slice(base, NR) };
                            if add {
                                for (y, v) in dst.iter_mut().zip(values) {
                                    *y += v;
                                }
                            } else {
                                dst.copy_from_slice(values);
                            }
                        }
                    }
                }
            }
        },
    );
}

/// One 6x16 tile over the whole reduction.
#[inline(always)]
unsafe fn kernel(k: usize, a: &[f32], b: &[f32], tile: &mut [[f32; NR]; MR]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        kernel_avx2(k, a.as_ptr(), b.as_ptr(), tile)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (k, a, b, tile);
        unreachable!("panel GEMM is x86-only for now")
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2(k: usize, a: *const f32, b: *const f32, tile: &mut [[f32; NR]; MR]) {
    use std::arch::x86_64::*;
    unsafe {
        let mut c00 = _mm256_setzero_ps();
        let mut c01 = _mm256_setzero_ps();
        let mut c10 = _mm256_setzero_ps();
        let mut c11 = _mm256_setzero_ps();
        let mut c20 = _mm256_setzero_ps();
        let mut c21 = _mm256_setzero_ps();
        let mut c30 = _mm256_setzero_ps();
        let mut c31 = _mm256_setzero_ps();
        let mut c40 = _mm256_setzero_ps();
        let mut c41 = _mm256_setzero_ps();
        let mut c50 = _mm256_setzero_ps();
        let mut c51 = _mm256_setzero_ps();
        for kk in 0..k {
            let b0 = _mm256_loadu_ps(b.add(kk * NR));
            let b1 = _mm256_loadu_ps(b.add(kk * NR + 8));
            let ap = a.add(kk * MR);
            let x = _mm256_broadcast_ss(&*ap);
            c00 = _mm256_fmadd_ps(x, b0, c00);
            c01 = _mm256_fmadd_ps(x, b1, c01);
            let x = _mm256_broadcast_ss(&*ap.add(1));
            c10 = _mm256_fmadd_ps(x, b0, c10);
            c11 = _mm256_fmadd_ps(x, b1, c11);
            let x = _mm256_broadcast_ss(&*ap.add(2));
            c20 = _mm256_fmadd_ps(x, b0, c20);
            c21 = _mm256_fmadd_ps(x, b1, c21);
            let x = _mm256_broadcast_ss(&*ap.add(3));
            c30 = _mm256_fmadd_ps(x, b0, c30);
            c31 = _mm256_fmadd_ps(x, b1, c31);
            let x = _mm256_broadcast_ss(&*ap.add(4));
            c40 = _mm256_fmadd_ps(x, b0, c40);
            c41 = _mm256_fmadd_ps(x, b1, c41);
            let x = _mm256_broadcast_ss(&*ap.add(5));
            c50 = _mm256_fmadd_ps(x, b0, c50);
            c51 = _mm256_fmadd_ps(x, b1, c51);
        }
        for (row, (lo, hi)) in [
            (c00, c01),
            (c10, c11),
            (c20, c21),
            (c30, c31),
            (c40, c41),
            (c50, c51),
        ]
        .into_iter()
        .enumerate()
        {
            _mm256_storeu_ps(tile[row].as_mut_ptr(), lo);
            _mm256_storeu_ps(tile[row].as_mut_ptr().add(8), hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Simd;

    fn matrix(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 2654435761 + seed * 97) % 2003) as f32 - 1001.0) / 977.0)
            .collect()
    }

    #[test]
    fn matches_f64_reference_and_glu_is_gated_store() {
        if !available(Simd::Auto) {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(7).build().unwrap();
        for (m, k, n) in [(1, 64, 16), (13, 768, 48), (97, 1024, 32), (200, 2304, 64), (9, 7, 16)] {
            let a = matrix(m * k, 1);
            let w = matrix(n * k, 2);
            let mut panels = Vec::new();
            pool.install(|| {
                pack_panels(n, k, |r, dst| dst.copy_from_slice(&w[r * k..(r + 1) * k]), &mut panels)
            });
            let mut out = vec![f32::NAN; m * n];
            pool.install(|| gemm(&a, m, k, &panels, n, None, Epilogue::Store(&mut out)));
            for row in 0..m {
                for col in 0..n {
                    let (mut exact, mut magnitude) = (0.0_f64, 0.0_f64);
                    for kk in 0..k {
                        let p = a[row * k + kk] as f64 * w[col * k + kk] as f64;
                        exact += p;
                        magnitude += p.abs();
                    }
                    let got = out[row * n + col] as f64;
                    assert!(
                        (got - exact).abs() <= k as f64 * f32::EPSILON as f64 * magnitude + 1e-6,
                        "m{m} k{k} n{n} ({row},{col}): {got} vs {exact}"
                    );
                }
            }
            let mut gated = vec![f32::NAN; m * n / 2];
            pool.install(|| gemm(&a, m, k, &panels, n, None, Epilogue::Glu(&mut gated)));
            // Row scales equal pre-scaled rows; Add equals Store then +=.
            let scale: Vec<f32> = (0..m).map(|r| 0.5 + r as f32 / 7.0).collect();
            let scaled: Vec<f32> = a.iter().enumerate().map(|(i, v)| v * scale[i / k]).collect();
            let mut direct = vec![f32::NAN; m * n];
            pool.install(|| gemm(&scaled, m, k, &panels, n, None, Epilogue::Store(&mut direct)));
            let mut folded = vec![f32::NAN; m * n];
            pool.install(|| gemm(&a, m, k, &panels, n, Some(&scale), Epilogue::Store(&mut folded)));
            assert!(direct.iter().zip(&folded).all(|(x, y)| x.to_bits() == y.to_bits()));
            let base: Vec<f32> = (0..m * n).map(|i| (i % 17) as f32 - 8.0).collect();
            let mut added = base.clone();
            pool.install(|| gemm(&a, m, k, &panels, n, None, Epilogue::Add(&mut added)));
            for ((x, b), c) in added.iter().zip(&base).zip(&out) {
                assert_eq!(x.to_bits(), (b + c).to_bits());
            }
            for row in 0..m {
                for i in 0..n / 2 {
                    let expected = crate::kernels::squared_relu_glu(out[row * n + 2 * i], out[row * n + 2 * i + 1]);
                    assert_eq!(gated[row * (n / 2) + i].to_bits(), expected.to_bits());
                }
            }
            // Thread count never changes the result.
            let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
            let mut again = vec![f32::NAN; m * n];
            single.install(|| gemm(&a, m, k, &panels, n, None, Epilogue::Store(&mut again)));
            assert!(out.iter().zip(&again).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    /// Prefill-shaped throughput against the `gemm` crate path.
    #[test]
    #[ignore = "timing probe; run in release with --nocapture"]
    fn prefill_throughput_probe() {
        let m = 6544;
        for (name, k, n) in [("qkv", 768, 2048), ("wo", 1024, 768), ("w13", 768, 4608), ("w2", 2304, 768)] {
            let a = matrix(m * k, 3);
            let w = matrix(n * k, 4);
            let mut out = vec![0.0_f32; m * n];
            let flops = 2.0 * (m * k * n) as f64;
            let t = std::time::Instant::now();
            for _ in 0..3 {
                crate::kernels::linear_with_simd(&a, m, k, &w, n, &mut out, Simd::Auto);
            }
            let crate_s = t.elapsed().as_secs_f64() / 3.0;
            let mut panels = Vec::new();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                pack_panels(n, k, |r, dst| dst.copy_from_slice(&w[r * k..(r + 1) * k]), &mut panels);
                gemm(&a, m, k, &panels, n, None, Epilogue::Store(&mut out));
            }
            let panel_s = t.elapsed().as_secs_f64() / 3.0;
            println!(
                "{name}: gemm crate {:.1} ms ({:.2} TFLOP/s), panels {:.1} ms ({:.2} TFLOP/s)",
                crate_s * 1e3,
                flops / crate_s / 1e12,
                panel_s * 1e3,
                flops / panel_s / 1e12
            );
        }
    }
}
