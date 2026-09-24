//! BF16 prefill projections for 8-bit weights on CPUs with AVX512-BF16.
//!
//! `vdpbf16ps` multiplies BF16 pairs and accumulates in FP32: 32 multiply-adds
//! per instruction against 16 for an FP32 FMA. The 8-bit weight codes are
//! exact in BF16, so the weights stay exact: each 64-input group accumulates
//! code products separately and its FP32 scale is applied with one FMA per
//! column. Only the activations round to BF16 (ties to even; the RMS-norm
//! factor is applied first). Opt-in for 8-bit models (see [`projections`]):
//! it changes results at the level of BF16 activation rounding.
use rayon::prelude::*;

use super::panel_gemm::Epilogue;

/// Output channels per panel (two 16-lane vectors).
pub(crate) const NR: usize = 32;
/// Rows per micro-kernel call.
const MR: usize = 6;
/// Inputs per weight scale.
const GROUP: usize = 64;

/// Whether this CPU runs the BF16 kernels (AVX-512F and AVX512-BF16).
pub(crate) fn available() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    })
}

/// Which 8-bit prefill stages use BF16 on a capable CPU. By default only
/// attention does: measured against the FP32 anchor it is fidelity-neutral,
/// while BF16 projections add about 17% KL. `FALCON_OCR_PREFILL_BF16` =
/// `1` (projections too), `proj`, `attn` or `0` (neither) overrides it.
fn stages() -> (bool, bool) {
    static STAGES: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *STAGES.get_or_init(|| match std::env::var("FALCON_OCR_PREFILL_BF16").as_deref() {
        Ok("0") => (false, false),
        Ok("1") => (true, true),
        Ok("proj") => (true, false),
        _ => (false, true),
    })
}

/// BF16 prefill projections for 8-bit weights here.
pub(crate) fn projections() -> bool {
    available() && stages().0
}

/// BF16 prefill attention for 8-bit models here.
pub(crate) fn attention() -> bool {
    available() && stages().1
}

/// Packed weights: codes as BF16 pairs `[panel][k / 2][32][2]` and scales
/// `[panel][k / 64][32]`.
#[derive(Default)]
pub(crate) struct Panels {
    codes: Vec<u16>,
    scales: Vec<f32>,
}

/// Pack `n x k` 8-bit codes with one scale per 64 inputs (`scales` is
/// `[n][k / 64]`). `n` must be a multiple of [`NR`] and `k` of 64.
pub(crate) fn pack(n: usize, k: usize, codes: &[i8], scales: &[f32], out: &mut Panels) {
    assert!(n % NR == 0 && k % GROUP == 0, "BF16 panel shape");
    assert_eq!(codes.len(), n * k);
    let groups = k / GROUP;
    assert_eq!(scales.len(), n * groups);
    out.codes.resize(n * k, 0);
    out.scales.resize(n * groups, 0.0);
    out.codes
        .par_chunks_mut(NR * k)
        .zip(out.scales.par_chunks_mut(NR * groups))
        .enumerate()
        .for_each(|(panel, (dst, sdst))| {
            // Integers up to 127 are exact in BF16.
            let lut: [u16; 256] = std::array::from_fn(|b| half::bf16::from_f32(b as u8 as i8 as f32).to_bits());
            let base = &codes[panel * NR * k..(panel + 1) * NR * k];
            for (kp, pairs) in dst.chunks_exact_mut(2 * NR).enumerate() {
                for (col, pair) in pairs.chunks_exact_mut(2).enumerate() {
                    let src = &base[col * k + 2 * kp..col * k + 2 * kp + 2];
                    pair[0] = lut[src[0] as u8 as usize];
                    pair[1] = lut[src[1] as u8 as usize];
                }
            }
            for col in 0..NR {
                let row = panel * NR + col;
                for g in 0..groups {
                    sdst[g * NR + col] = scales[row * groups + g];
                }
            }
        });
}

/// `C = A * W^T` for `a` (`m x k`) and 8-bit `panels`, optionally scaling
/// row `r` of `A` by `row_scale[r]` before rounding it to BF16. Requires
/// [`available`].
pub(crate) fn gemm(
    a: &[f32],
    m: usize,
    k: usize,
    panels: &Panels,
    n: usize,
    row_scale: Option<&[f32]>,
    epilogue: Epilogue<'_>,
) {
    assert!(available(), "AVX512-BF16 required");
    assert_eq!(a.len(), m * k);
    assert!(n % NR == 0 && k % GROUP == 0);
    let (out, mode) = match epilogue {
        Epilogue::Store(out) => (out, 0),
        Epilogue::Glu(out) => (out, 1),
        Epilogue::Add(out) => (out, 2),
    };
    let out_width = if mode == 1 { n / 2 } else { n };
    assert_eq!(out.len(), m * out_width, "BF16 panel output shape");
    if let Some(scale) = row_scale {
        assert_eq!(scale.len(), m);
    }
    if m == 0 {
        return;
    }
    let block_rows = if k <= 1024 { 96 } else { 48 };
    let row_blocks = m.div_ceil(block_rows);
    let panels_total = n / NR;
    let threads = rayon::current_num_threads().max(1);
    let splits = (2 * threads).div_ceil(row_blocks).clamp(1, panels_total);
    let per_split = panels_total.div_ceil(splits);
    let shared = crate::team::SharedMut::new(out);
    (0..row_blocks * splits).into_par_iter().for_each_init(
        || vec![0_u16; block_rows.next_multiple_of(MR) * k],
        |packed, unit| {
            let (block, split) = (unit / splits, unit % splits);
            let row0 = block * block_rows;
            let rows = block_rows.min(m - row0);
            let padded = rows.next_multiple_of(MR);
            // SAFETY: AVX512-BF16 checked by `available`; shapes checked above.
            unsafe { pack_rows(&a[row0 * k..(row0 + rows) * k], rows, padded, k, row_scale.map(|s| &s[row0..row0 + rows]), packed) };
            let first = split * per_split;
            let last = (first + per_split).min(panels_total);
            let mut tile = [[0.0_f32; NR]; MR];
            for panel in first..last {
                let codes = &panels.codes[panel * NR * k..(panel + 1) * NR * k];
                let scales = &panels.scales[panel * NR * (k / GROUP)..(panel + 1) * NR * (k / GROUP)];
                for g in 0..padded / MR {
                    // SAFETY: as above; the packed rows hold `padded x k` values.
                    unsafe { kernel(k, &packed[g * MR * k..(g + 1) * MR * k], codes, scales, &mut tile) };
                    let valid = MR.min(rows.saturating_sub(g * MR));
                    for (r, values) in tile[..valid].iter().enumerate() {
                        let row = row0 + g * MR + r;
                        match mode {
                            1 => {
                                // SAFETY: (row, panel) tiles are disjoint across units.
                                let dst = unsafe { shared.slice(row * out_width + panel * (NR / 2), NR / 2) };
                                for (i, y) in dst.iter_mut().enumerate() {
                                    *y = crate::kernels::squared_relu_glu(values[2 * i], values[2 * i + 1]);
                                }
                            }
                            2 => {
                                // SAFETY: as above.
                                let dst = unsafe { shared.slice(row * out_width + panel * NR, NR) };
                                for (y, v) in dst.iter_mut().zip(values) {
                                    *y += v;
                                }
                            }
                            _ => {
                                // SAFETY: as above.
                                unsafe { shared.slice(row * out_width + panel * NR, NR) }.copy_from_slice(values);
                            }
                        }
                    }
                }
            }
        },
    );
}

/// Rows of `a` (times their row scale) as BF16, row-major, zero rows padding
/// to `padded`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn pack_rows(a: &[f32], rows: usize, padded: usize, k: usize, row_scale: Option<&[f32]>, out: &mut [u16]) {
    use std::arch::x86_64::*;
    unsafe {
        for r in 0..padded {
            let dst = out.as_mut_ptr().add(r * k);
            if r >= rows {
                std::ptr::write_bytes(dst, 0, k);
                continue;
            }
            let src = a.as_ptr().add(r * k);
            let scale = _mm512_set1_ps(row_scale.map_or(1.0, |s| s[r]));
            let scaled = row_scale.is_some();
            for c in (0..k).step_by(32) {
                let mut lo = _mm512_loadu_ps(src.add(c));
                let mut hi = _mm512_loadu_ps(src.add(c + 16));
                if scaled {
                    lo = _mm512_mul_ps(lo, scale);
                    hi = _mm512_mul_ps(hi, scale);
                }
                let packed: __m512bh = _mm512_cvtne2ps_pbh(hi, lo);
                _mm512_storeu_si512(dst.add(c).cast(), std::mem::transmute::<__m512bh, __m512i>(packed));
            }
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn pack_rows(_: &[f32], _: usize, _: usize, _: usize, _: Option<&[f32]>, _: &mut [u16]) {
    unreachable!("BF16 panels are x86-only")
}

/// One 6x32 tile: per 64-input group, BF16 pair dot products into group
/// accumulators, then `acc += group * scale` per column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn kernel(k: usize, a: &[u16], codes: &[u16], scales: &[f32], tile: &mut [[f32; NR]; MR]) {
    use std::arch::x86_64::*;
    unsafe {
        let zero = _mm512_setzero_ps();
        let mut acc = [[zero; 2]; MR];
        let a32 = a.as_ptr().cast::<u32>();
        let b = codes.as_ptr().cast::<__m512i>();
        let pairs = k / 2;
        for g in 0..k / GROUP {
            let mut part = [[zero; 2]; MR];
            for kp in g * (GROUP / 2)..(g + 1) * (GROUP / 2) {
                let b0: __m512bh = std::mem::transmute(_mm512_loadu_si512(b.add(2 * kp)));
                let b1: __m512bh = std::mem::transmute(_mm512_loadu_si512(b.add(2 * kp + 1)));
                for (r, part) in part.iter_mut().enumerate() {
                    let x: __m512bh = std::mem::transmute(_mm512_set1_epi32(*a32.add(r * pairs + kp) as i32));
                    part[0] = _mm512_dpbf16_ps(part[0], x, b0);
                    part[1] = _mm512_dpbf16_ps(part[1], x, b1);
                }
            }
            let s0 = _mm512_loadu_ps(scales.as_ptr().add(g * NR));
            let s1 = _mm512_loadu_ps(scales.as_ptr().add(g * NR + 16));
            for (acc, part) in acc.iter_mut().zip(&part) {
                acc[0] = _mm512_fmadd_ps(part[0], s0, acc[0]);
                acc[1] = _mm512_fmadd_ps(part[1], s1, acc[1]);
            }
        }
        for (row, acc) in acc.iter().enumerate() {
            _mm512_storeu_ps(tile[row].as_mut_ptr(), acc[0]);
            _mm512_storeu_ps(tile[row].as_mut_ptr().add(16), acc[1]);
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn kernel(_: usize, _: &[u16], _: &[u16], _: &[f32], _: &mut [[f32; NR]; MR]) {
    unreachable!("BF16 panels are x86-only")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_bf16_rounded_reference() {
        if !available() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(5).build().unwrap();
        for (m, k, n) in [(7, 64, 32), (61, 768, 64), (100, 2304, 96), (13, 1024, 32)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i * 2654435761 % 2003) as f32 - 1001.0) / 377.0).collect();
            let codes: Vec<i8> = (0..n * k).map(|i| ((i * 7919 % 255) as i32 - 127) as i8).collect();
            let scales: Vec<f32> = (0..n * k / 64).map(|i| 0.001 + (i % 17) as f32 * 0.0007).collect();
            let row_scale: Vec<f32> = (0..m).map(|r| 0.5 + r as f32 / 13.0).collect();
            let mut panels = Panels::default();
            pack(n, k, &codes, &scales, &mut panels);
            let mut out = vec![f32::NAN; m * n];
            pool.install(|| gemm(&a, m, k, &panels, n, Some(&row_scale), Epilogue::Store(&mut out)));
            for r in 0..m {
                for c in 0..n {
                    let (mut exact, mut mag) = (0.0_f64, 0.0_f64);
                    for kk in 0..k {
                        let x = half::bf16::from_f32(a[r * k + kk] * row_scale[r]).to_f32() as f64;
                        let w = codes[c * k + kk] as f64 * scales[c * (k / 64) + kk / 64] as f64;
                        exact += x * w;
                        mag += (x * w).abs();
                    }
                    let got = out[r * n + c] as f64;
                    assert!((got - exact).abs() <= 1e-5 * mag + 1e-6, "m{m} k{k} n{n} ({r},{c}) {got} vs {exact}");
                }
            }
            let mut gated = vec![f32::NAN; m * n / 2];
            pool.install(|| gemm(&a, m, k, &panels, n, Some(&row_scale), Epilogue::Glu(&mut gated)));
            for r in 0..m {
                for i in 0..n / 2 {
                    let e = crate::kernels::squared_relu_glu(out[r * n + 2 * i], out[r * n + 2 * i + 1]);
                    assert_eq!(gated[r * (n / 2) + i].to_bits(), e.to_bits());
                }
            }
        }
    }

    #[test]
    #[ignore = "timing probe; run in release with --nocapture"]
    fn bf16_prefill_throughput_probe() {
        if !available() {
            return;
        }
        let m = 6544;
        for (name, k, n) in [("qkv", 768, 2048), ("wo", 1024, 768), ("w13", 768, 4608), ("w2", 2304, 768)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
            let codes: Vec<i8> = (0..n * k).map(|i| ((i * 31 % 255) as i32 - 127) as i8).collect();
            let scales = vec![0.01_f32; n * k / 64];
            let mut out = vec![0.0; m * n];
            let mut panels = Panels::default();
            pack(n, k, &codes, &scales, &mut panels);
            let t = std::time::Instant::now();
            for _ in 0..3 {
                pack(n, k, &codes, &scales, &mut panels);
            }
            let pack_s = t.elapsed().as_secs_f64() / 3.0;
            let t = std::time::Instant::now();
            for _ in 0..3 {
                gemm(&a, m, k, &panels, n, None, Epilogue::Store(&mut out));
            }
            let s = t.elapsed().as_secs_f64() / 3.0;
            println!(
                "{name}: bf16 pack {:.2} ms, gemm {:.1} ms ({:.2} TFLOP/s)",
                pack_s * 1e3,
                s * 1e3,
                2.0 * (m * k * n) as f64 / s / 1e12
            );
        }
    }
}
