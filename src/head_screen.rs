//! Exact greedy selection over the vocabulary head through an INT8 screen.
//!
//! The FP32 head stays authoritative. An INT8 copy (64-channel groups, absmax
//! /127 FP32 scales, ties-to-even codes) gives each row an approximate logit
//! `q_v` together with a rigorous bound `B_v` such that `|L_v - q_v| <= B_v`,
//! where `L_v` is the FP32 logit that *any* summation order of the reference
//! dot product produces. With `tau = max_u (q_u - B_u)`, every row whose
//! reference logit can reach the maximum satisfies `q_v + B_v >= tau`. Those
//! candidate rows are recomputed with the reference dot kernel, so the chosen
//! token, including torch's first-index tie break, equals the argmax over the
//! complete FP32 logits. Anything outside the proof's assumptions (nonfinite
//! or huge activations, too many candidates, no AVX2/FMA) returns `None` and
//! the caller evaluates the full FP32 head, reproducing its exact behaviour.
//!
//! Bound terms, for one row with groups `g`, scales `s_g`, `n_g = ||x_g||_1`:
//! * quantization: `|w - c*s| <= s/2 * (1 + 2^-20)`, verified in f64 at build;
//! * reference FP32 rounding: `gamma_n * sum|x_i w_i|`, with `|w| <= 127.5 s`;
//! * screen FP32 rounding: `gamma_(n+groups+8) * sum|x_i c_i s|`.
//!
//! All three are proportional to `S_v = sum_g s_g n_g`; `kappa` bounds their
//! sum with slack for the FP32 evaluation of `S_v` and `q_v +/- B_v`.

use crate::kernels::Dot;
use anyhow::{Result, ensure};
use rayon::prelude::*;

/// Input channels per quantization group.
pub(crate) const GROUP: usize = 64;
/// Beyond this many candidates the full FP32 head is cheaper to evaluate.
pub(crate) const CANDIDATE_LIMIT: usize = 1024;
/// Vocabulary rows per parallel screening task.
const BLOCK: usize = 512;
/// Absolute slack covering subnormal rounding in the screen.
const ETA: f32 = 7.888_609e-31; // 2^-100
/// Inputs per [`ScreenedHead::select_rows`] call.
pub(crate) const MAX_ROWS: usize = 8;

pub(crate) struct ScreenedHead {
    vocab: usize,
    dim: usize,
    groups: usize,
    codes: crate::buf::Buf<i8>,
    scales: crate::buf::Buf<f32>,
    weight_abs_max: f32,
    kappa: f32,
}

/// Per-session scratch, allocated before decode so warm steps never allocate.
#[derive(Default)]
pub(crate) struct HeadScratch {
    upper: Vec<f32>,
    lower_max: Vec<f32>,
    candidates: Vec<u32>,
    norms: Vec<f32>,
}

impl HeadScratch {
    pub(crate) fn reserve(&mut self, head: &ScreenedHead) {
        self.upper.resize(head.vocab, 0.0);
        self.lower_max
            .resize(head.vocab.div_ceil(BLOCK), f32::NEG_INFINITY);
        self.norms.resize(head.groups, 0.0);
        self.candidates
            .reserve(CANDIDATE_LIMIT + 1 - self.candidates.len().min(CANDIDATE_LIMIT + 1));
    }
}

/// Outcome of one screened selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Screened {
    /// Exact argmax over the full FP32 logits, and the number of rows recomputed.
    Token { token: u32, candidates: usize },
    /// Outside the proof's assumptions; evaluate the full FP32 head instead.
    Fallback { candidates: usize },
}

fn gamma(n: usize) -> f64 {
    let u = f64::from(f32::EPSILON) / 2.0;
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

impl ScreenedHead {
    /// Quantize and verify `weights` (`[vocab][dim]`, row-major FP32).
    pub(crate) fn build(weights: &[f32], vocab: usize, dim: usize) -> Result<Self> {
        ensure!(
            vocab > 0 && dim > 0 && dim.is_multiple_of(GROUP),
            "screened head needs a nonempty head with 64-channel groups"
        );
        ensure!(
            vocab.checked_mul(dim) == Some(weights.len()) && vocab <= u32::MAX as usize,
            "screened head shape"
        );
        let groups = dim / GROUP;
        let mut codes = vec![0_i8; weights.len()];
        let mut scales = vec![0.0_f32; vocab * groups];
        let weight_abs_max = codes
            .par_chunks_mut(dim)
            .zip(scales.par_chunks_mut(groups))
            .zip(weights.par_chunks(dim))
            .map(|((codes, scales), row)| quantize_row(row, codes, scales))
            .try_reduce(|| 0.0_f32, |a, b| Ok(a.max(b)))?;
        ensure!(
            weight_abs_max.is_finite(),
            "screened head weights must be finite"
        );
        let n = dim;
        let kappa = (0.5 * (1.0 + 2f64.powi(-20))
            + gamma(n) * 127.5 * (1.0 + 2f64.powi(-20))
            + gamma(n + groups + 8) * 127.0)
            * (1.0 + 1e-3)
            + 2e-3;
        ensure!(kappa < 0.53, "screened head bound is unexpectedly loose");
        let kappa_f32 = (kappa as f32).next_up();
        Ok(Self {
            vocab,
            dim,
            groups,
            codes: crate::buf::Buf::Owned(codes),
            scales: crate::buf::Buf::Owned(scales),
            weight_abs_max,
            kappa: kappa_f32,
        })
    }

    /// A screen stored in a kernel-ready model file (written by
    /// `Model::write_packed` from `build`), used in place.
    pub(crate) fn from_mapped(
        vocab: usize,
        dim: usize,
        map: &std::sync::Arc<memmap2::Mmap>,
        codes: std::ops::Range<usize>,
        scales: std::ops::Range<usize>,
        weight_abs_max: f32,
        kappa: f32,
    ) -> Result<Self> {
        ensure!(vocab > 0 && dim > 0 && dim.is_multiple_of(GROUP), "mapped screen shape");
        ensure!(
            weight_abs_max.is_finite() && kappa.is_finite() && kappa > 0.0 && kappa < 0.53,
            "mapped screen constants"
        );
        let groups = dim / GROUP;
        Ok(Self {
            vocab,
            dim,
            groups,
            codes: crate::buf::Buf::mapped(map, codes, vocab * dim)?,
            scales: crate::buf::Buf::mapped(map, scales, vocab * groups)?,
            weight_abs_max,
            kappa,
        })
    }

    /// Code bytes, scale bytes, and the bound constants (`weight_abs_max`, `kappa`).
    pub(crate) fn raw_parts(&self) -> (&[u8], &[u8], f32, f32) {
        (self.codes.bytes(), self.scales.bytes(), self.weight_abs_max, self.kappa)
    }

    pub(crate) fn payload_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * size_of::<f32>()
    }

    /// Select the FP32 argmax for one normalized hidden row `x`.
    ///
    /// `fp32` is the authoritative `[vocab][dim]` head and `dot` the exact
    /// reference kernel used for full-head logits (`kernels::dot_kernel`).
    pub(crate) fn select(
        &self,
        x: &[f32],
        fp32: &[f32],
        dot: Dot,
        scratch: &mut HeadScratch,
    ) -> Screened {
        assert_eq!(x.len(), self.dim, "screened head input width");
        assert_eq!(
            fp32.len(),
            self.vocab * self.dim,
            "screened head FP32 shape"
        );
        let fallback = Screened::Fallback { candidates: 0 };
        if !self.prepare(x, scratch) {
            return fallback;
        }
        let norms = &scratch.norms;
        let (dim, groups, kappa, vocab) = (self.dim, self.groups, self.kappa, self.vocab);
        let upper_all = crate::team::SharedMut::new(&mut scratch.upper);
        let lower_all = crate::team::SharedMut::new(&mut scratch.lower_max);
        crate::team::for_each(vocab.div_ceil(BLOCK), |block| {
            let start = block * BLOCK;
            // SAFETY: each task owns one disjoint block of rows.
            let upper = unsafe { upper_all.slice(start, BLOCK.min(vocab - start)) };
            let rows = start..start + upper.len();
            let codes = &self.codes[rows.start * dim..rows.end * dim];
            let scales = &self.scales[rows.start * groups..rows.end * groups];
            // SAFETY: the native vector ISA was detected above; slices are
            // whole rows of this block.
            let lower_max = unsafe { screen_block_native(x, codes, scales, norms, kappa, upper) };
            // SAFETY: one maximum slot per block.
            unsafe { lower_all.slice(block, 1)[0] = lower_max };
        });
        self.finish(x, fp32, dot, scratch)
    }

    /// Input checks and per-group L1 norms (rounded up) into `scratch`;
    /// `false` when the proof's assumptions do not hold for `x`.
    fn prepare(&self, x: &[f32], scratch: &mut HeadScratch) -> bool {
        #[cfg(target_arch = "x86_64")]
        let vector = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(target_arch = "aarch64")]
        let vector = true;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let vector = false;
        if !vector || !x.iter().all(|v| v.is_finite()) {
            return false;
        }
        // Every FP32 logit is provably finite below this; beyond it, let the full
        // head decide (and report) exactly as the reference does.
        let l1: f64 = x.iter().map(|v| f64::from(v.abs())).sum();
        if l1 * f64::from(self.weight_abs_max) * 1.01 >= f64::from(f32::MAX) / 2.0 {
            return false;
        }
        scratch.reserve(self);
        for (norm, group) in scratch.norms.iter_mut().zip(x.chunks_exact(GROUP)) {
            let exact: f64 = group.iter().map(|v| f64::from(v.abs())).sum();
            let rounded = exact as f32;
            // Round up: the bound must never shrink.
            *norm = if f64::from(rounded) < exact {
                rounded.next_up()
            } else {
                rounded
            };
        }
        true
    }

    /// The exact token from the screened bounds in `scratch`.
    fn finish(&self, x: &[f32], fp32: &[f32], dot: Dot, scratch: &mut HeadScratch) -> Screened {
        let fallback = Screened::Fallback { candidates: 0 };
        let dim = self.dim;
        let tau = scratch
            .lower_max
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !tau.is_finite() {
            return fallback;
        }
        scratch.candidates.clear();
        for (row, &hi) in scratch.upper.iter().enumerate() {
            if hi >= tau {
                if scratch.candidates.len() == CANDIDATE_LIMIT {
                    return Screened::Fallback {
                        candidates: CANDIDATE_LIMIT + 1,
                    };
                }
                scratch.candidates.push(row as u32);
            }
        }
        let mut best: Option<(u32, f32)> = None;
        for &row in &scratch.candidates {
            let start = row as usize * dim;
            let logit = dot(x, &fp32[start..start + dim]);
            if !logit.is_finite() {
                return fallback;
            }
            // Ascending candidates plus strict `>` keep the first-index tie break.
            if best.is_none_or(|(_, value)| logit > value) {
                best = Some((row, logit));
            }
        }
        match best {
            Some((token, _)) => Screened::Token {
                token,
                candidates: scratch.candidates.len(),
            },
            None => fallback,
        }
    }

    /// [`ScreenedHead::select`] for `rows` inputs (`xs` is `rows x dim`),
    /// reading each screen row once for all of them; `out[r]` is exactly what
    /// `select` returns for input `r` (the token is the exact FP32 argmax,
    /// whatever the screen's evaluation order).
    pub(crate) fn select_rows(
        &self,
        xs: &[f32],
        rows: usize,
        fp32: &[f32],
        dot: Dot,
        scratch: &mut [HeadScratch],
        out: &mut [Screened],
    ) {
        let dim = self.dim;
        assert!(rows <= MAX_ROWS && rows <= scratch.len() && rows <= out.len());
        assert_eq!(xs.len(), rows * dim, "screened head input shape");
        assert_eq!(fp32.len(), self.vocab * dim, "screened head FP32 shape");
        let mut live = [0_usize; MAX_ROWS];
        let mut n = 0;
        for r in 0..rows {
            out[r] = Screened::Fallback { candidates: 0 };
            if self.prepare(&xs[r * dim..(r + 1) * dim], &mut scratch[r]) {
                live[n] = r;
                n += 1;
            }
        }
        if n == 0 {
            return;
        }
        let (groups, kappa, vocab) = (self.groups, self.kappa, self.vocab);
        let uppers: [crate::team::SharedMut<f32>; MAX_ROWS] = std::array::from_fn(|i| {
            crate::team::SharedMut::new(&mut scratch[live[i.min(n - 1)]].upper)
        });
        let lowers: [crate::team::SharedMut<f32>; MAX_ROWS] = std::array::from_fn(|i| {
            crate::team::SharedMut::new(&mut scratch[live[i.min(n - 1)]].lower_max)
        });
        {
            let inputs: [&[f32]; MAX_ROWS] =
                std::array::from_fn(|i| &xs[live[i.min(n - 1)] * dim..(live[i.min(n - 1)] + 1) * dim]);
            let norms: [&[f32]; MAX_ROWS] =
                std::array::from_fn(|i| scratch[live[i.min(n - 1)]].norms.as_slice());
            crate::team::for_each(vocab.div_ceil(BLOCK), |block| {
                let start = block * BLOCK;
                let len = BLOCK.min(vocab - start);
                let codes = &self.codes[start * dim..(start + len) * dim];
                let scales = &self.scales[start * groups..(start + len) * groups];
                // SAFETY: each task owns one disjoint block of every input's rows.
                let mut upper: [&mut [f32]; MAX_ROWS] =
                    std::array::from_fn(|i| unsafe { uppers[i].slice(start, if i < n { len } else { 0 }) });
                // SAFETY: the native vector ISA was checked by `prepare`; slices
                // are whole rows of this block.
                let lower = unsafe {
                    screen_block_rows_native(&inputs[..n], &norms[..n], codes, scales, kappa, &mut upper[..n])
                };
                for (i, value) in lower[..n].iter().enumerate() {
                    // SAFETY: one maximum slot per (input, block).
                    unsafe { lowers[i].slice(block, 1)[0] = *value };
                }
            });
        }
        for &r in &live[..n] {
            out[r] = self.finish(&xs[r * dim..(r + 1) * dim], fp32, dot, &mut scratch[r]);
        }
    }
}

/// Quantize one row and verify the quantization-error premise of the bound.
/// Returns the row's maximum absolute weight.
fn quantize_row(row: &[f32], codes: &mut [i8], scales: &mut [f32]) -> Result<f32> {
    let mut row_max = 0.0_f32;
    for ((values, codes), scale_out) in row
        .chunks_exact(GROUP)
        .zip(codes.chunks_exact_mut(GROUP))
        .zip(scales.iter_mut())
    {
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "screened head weights must be finite"
        );
        let maximum = values.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        row_max = row_max.max(maximum);
        let scale = if maximum == 0.0 {
            0.0
        } else {
            ((f64::from(maximum) / 127.0) as f32).max(f32::from_bits(1))
        };
        *scale_out = scale;
        let tolerance = f64::from(scale) * 0.5 * (1.0 + 2f64.powi(-20));
        for (code, &value) in codes.iter_mut().zip(values) {
            *code = if scale == 0.0 {
                0
            } else {
                (f64::from(value) / f64::from(scale))
                    .round_ties_even()
                    .clamp(-127.0, 127.0) as i8
            };
            let reconstructed = f64::from(*code) * f64::from(scale);
            ensure!(
                (f64::from(value) - reconstructed).abs() <= tolerance
                    && f64::from(value.abs()) <= 127.5 * f64::from(scale) * (1.0 + 2f64.powi(-20)),
                "screened head quantization premise violated"
            );
        }
    }
    Ok(row_max)
}

/// Approximate logit `q` and `S = sum_g s_g n_g` for one INT8 row. The bound
/// holds for any summation order, so the reduction tree is not significant.
#[inline(always)]
unsafe fn screen_row<S: crate::simd::Simd>(
    x: &[f32],
    codes: &[i8],
    scales: &[f32],
    norms: &[f32],
) -> (f32, f32) {
    debug_assert_eq!(x.len(), codes.len());
    debug_assert_eq!(scales.len() * GROUP, codes.len());
    let mut row = unsafe { S::zero() };
    let mut total = 0.0_f32;
    for (group, (&scale, &norm)) in scales.iter().zip(norms).enumerate() {
        let base = group * GROUP;
        // SAFETY: base + 64 <= len for both slices; unaligned loads are used.
        unsafe {
            let mut a0 = S::zero();
            let mut a1 = S::zero();
            for j in (0..GROUP).step_by(16) {
                let low = S::load_i8(codes.as_ptr().add(base + j));
                let high = S::load_i8(codes.as_ptr().add(base + j + 8));
                a0 = S::fma(S::load(x.as_ptr().add(base + j)), low, a0);
                a1 = S::fma(S::load(x.as_ptr().add(base + j + 8)), high, a1);
            }
            row = S::fma(S::add(a0, a1), S::splat(scale), row);
        }
        total = scale.mul_add(norm, total);
    }
    (unsafe { S::sum(row) }, total)
}

/// Screen one block of rows: write `q + B` per row, return the block's
/// largest `q - B`.
#[inline(always)]
unsafe fn screen_block<S: crate::simd::Simd>(
    x: &[f32],
    codes: &[i8],
    scales: &[f32],
    norms: &[f32],
    kappa: f32,
    upper: &mut [f32],
) -> f32 {
    let dim = x.len();
    let groups = dim / GROUP;
    let mut lower_max = f32::NEG_INFINITY;
    for (i, hi) in upper.iter_mut().enumerate() {
        let (q, s) = unsafe {
            screen_row::<S>(
                x,
                &codes[i * dim..(i + 1) * dim],
                &scales[i * groups..(i + 1) * groups],
                norms,
            )
        };
        let bound = kappa.mul_add(s, ETA);
        *hi = q + bound;
        lower_max = lower_max.max(q - bound);
    }
    lower_max
}

/// [`screen_block`] for several inputs: each screen row is loaded once and
/// scored against every input.
#[inline(always)]
unsafe fn screen_block_rows<S: crate::simd::Simd>(
    inputs: &[&[f32]],
    norms: &[&[f32]],
    codes: &[i8],
    scales: &[f32],
    kappa: f32,
    upper: &mut [&mut [f32]],
) -> [f32; MAX_ROWS] {
    let dim = inputs[0].len();
    let groups = dim / GROUP;
    let mut lower_max = [f32::NEG_INFINITY; MAX_ROWS];
    for i in 0..upper[0].len() {
        let row_codes = &codes[i * dim..(i + 1) * dim];
        let row_scales = &scales[i * groups..(i + 1) * groups];
        for (r, x) in inputs.iter().enumerate() {
            let (q, s) = unsafe { screen_row::<S>(x, row_codes, row_scales, norms[r]) };
            let bound = kappa.mul_add(s, ETA);
            upper[r][i] = q + bound;
            lower_max[r] = lower_max[r].max(q - bound);
        }
    }
    lower_max
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn screen_block_rows_native(
    inputs: &[&[f32]],
    norms: &[&[f32]],
    codes: &[i8],
    scales: &[f32],
    kappa: f32,
    upper: &mut [&mut [f32]],
) -> [f32; MAX_ROWS] {
    unsafe { screen_block_rows::<crate::simd::Avx2>(inputs, norms, codes, scales, kappa, upper) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn screen_block_rows_native(
    inputs: &[&[f32]],
    norms: &[&[f32]],
    codes: &[i8],
    scales: &[f32],
    kappa: f32,
    upper: &mut [&mut [f32]],
) -> [f32; MAX_ROWS] {
    unsafe { screen_block_rows::<crate::simd::Neon>(inputs, norms, codes, scales, kappa, upper) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn screen_block_rows_native(
    inputs: &[&[f32]],
    norms: &[&[f32]],
    codes: &[i8],
    scales: &[f32],
    kappa: f32,
    upper: &mut [&mut [f32]],
) -> [f32; MAX_ROWS] {
    unsafe { screen_block_rows::<crate::simd::Portable>(inputs, norms, codes, scales, kappa, upper) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn screen_block_native(
    x: &[f32],
    codes: &[i8],
    scales: &[f32],
    norms: &[f32],
    kappa: f32,
    upper: &mut [f32],
) -> f32 {
    unsafe { screen_block::<crate::simd::Avx2>(x, codes, scales, norms, kappa, upper) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn screen_block_native(
    x: &[f32],
    codes: &[i8],
    scales: &[f32],
    norms: &[f32],
    kappa: f32,
    upper: &mut [f32],
) -> f32 {
    unsafe { screen_block::<crate::simd::Neon>(x, codes, scales, norms, kappa, upper) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn screen_block_native(
    x: &[f32],
    codes: &[i8],
    scales: &[f32],
    norms: &[f32],
    kappa: f32,
    upper: &mut [f32],
) -> f32 {
    unsafe { screen_block::<crate::simd::Portable>(x, codes, scales, norms, kappa, upper) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{Simd, dot_kernel};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn full_argmax(x: &[f32], head: &[f32], dim: usize, dot: Dot) -> u32 {
        let mut best = 0;
        let mut best_value = dot(x, &head[..dim]);
        for row in 1..head.len() / dim {
            let value = dot(x, &head[row * dim..(row + 1) * dim]);
            if value > best_value {
                best = row;
                best_value = value;
            }
        }
        best as u32
    }

    fn backends() -> Vec<Simd> {
        let mut result = vec![Simd::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                result.push(Simd::Avx2);
            }
            if std::is_x86_feature_detected!("avx512f") {
                result.push(Simd::Avx512);
            }
        }
        result
    }

    #[test]
    fn screened_argmax_equals_full_argmax_including_ties() {
        let (vocab, dim) = (3000, 768);
        let mut rng = Lcg(11);
        let mut head: Vec<f32> = (0..vocab * dim).map(|_| rng.next() * 0.05).collect();
        // Planted exact duplicates: identical rows produce identical logits, so
        // the first index must win whichever copy the screen ranks higher.
        for (dst, src) in [(2000, 17), (2500, 17), (40, 1999)] {
            let row: Vec<f32> = head[src * dim..(src + 1) * dim].to_vec();
            head[dst * dim..(dst + 1) * dim].copy_from_slice(&row);
        }
        let screened = ScreenedHead::build(&head, vocab, dim).unwrap();
        let mut scratch = HeadScratch::default();
        for simd in backends() {
            let dot = dot_kernel(simd);
            for trial in 0..60 {
                let mut x: Vec<f32> = (0..dim).map(|_| rng.next() * 3.0).collect();
                if trial % 3 == 0 {
                    // Aim at a duplicated row so ties decide the result.
                    let target = if trial % 2 == 0 { 17 } else { 1999 };
                    for (xi, w) in x.iter_mut().zip(&head[target * dim..(target + 1) * dim]) {
                        *xi = w * 40.0;
                    }
                }
                let expected = full_argmax(&x, &head, dim, dot);
                match screened.select(&x, &head, dot, &mut scratch) {
                    Screened::Token { token, candidates } => {
                        assert_eq!(token, expected, "{simd:?} trial {trial}");
                        assert!(candidates >= 1);
                    }
                    Screened::Fallback { .. } => {
                        #[cfg(target_arch = "x86_64")]
                        assert!(
                            !std::is_x86_feature_detected!("avx2"),
                            "unexpected fallback"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn select_rows_equals_select_per_row() {
        let (vocab, dim) = (2100, 768);
        let mut rng = Lcg(29);
        let mut head: Vec<f32> = (0..vocab * dim).map(|_| rng.next() * 0.05).collect();
        for (dst, src) in [(1500, 3), (900, 1200)] {
            let row: Vec<f32> = head[src * dim..(src + 1) * dim].to_vec();
            head[dst * dim..(dst + 1) * dim].copy_from_slice(&row);
        }
        let screened = ScreenedHead::build(&head, vocab, dim).unwrap();
        for simd in backends() {
            let dot = dot_kernel(simd);
            for rows in 1..=MAX_ROWS {
                let mut xs: Vec<f32> = (0..rows * dim).map(|_| rng.next() * 3.0).collect();
                // Aim one input at a duplicated row (a tie), make one nonfinite.
                for (xi, w) in xs[..dim].iter_mut().zip(&head[3 * dim..4 * dim]) {
                    *xi = w * 40.0;
                }
                if rows > 2 {
                    xs[2 * dim + 5] = f32::NAN;
                }
                let mut scratch: Vec<HeadScratch> = (0..MAX_ROWS).map(|_| HeadScratch::default()).collect();
                let mut joint = [Screened::Fallback { candidates: 0 }; MAX_ROWS];
                screened.select_rows(&xs, rows, &head, dot, &mut scratch, &mut joint);
                let mut single_scratch = HeadScratch::default();
                for r in 0..rows {
                    let single = screened.select(&xs[r * dim..(r + 1) * dim], &head, dot, &mut single_scratch);
                    assert_eq!(joint[r], single, "{simd:?} rows {rows} row {r}");
                }
            }
        }
    }

    #[test]
    fn bound_covers_adversarial_rounding_direction() {
        // x_i = sign(w_i - w_hat_i) makes every quantization error add up.
        let (vocab, dim) = (64, 768);
        let mut rng = Lcg(5);
        let head: Vec<f32> = (0..vocab * dim).map(|_| rng.next()).collect();
        let screened = ScreenedHead::build(&head, vocab, dim).unwrap();
        for row in 0..vocab {
            let codes = &screened.codes[row * dim..(row + 1) * dim];
            let scales = &screened.scales[row * screened.groups..(row + 1) * screened.groups];
            let x: Vec<f32> = (0..dim)
                .map(|i| {
                    let w_hat = f32::from(codes[i]) * scales[i / GROUP];
                    if head[row * dim + i] >= w_hat {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            let exact: f64 = x
                .iter()
                .zip(&head[row * dim..(row + 1) * dim])
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norms: Vec<f32> = x
                .chunks(GROUP)
                .map(|g| g.iter().map(|v| v.abs()).sum())
                .collect();
            {
                let (q, s) =
                    unsafe { screen_row::<crate::simd::Portable>(&x, codes, scales, &norms) };
                let bound = f64::from(screened.kappa) * f64::from(s);
                assert!((exact - f64::from(q)).abs() <= bound, "row {row}");
            }
        }
    }

    #[test]
    fn nonfinite_or_degenerate_inputs_fall_back() {
        let (vocab, dim) = (2048, 64);
        let mut rng = Lcg(3);
        let head: Vec<f32> = (0..vocab * dim).map(|_| rng.next()).collect();
        let screened = ScreenedHead::build(&head, vocab, dim).unwrap();
        let mut scratch = HeadScratch::default();
        let dot = dot_kernel(Simd::Scalar);
        let mut x = vec![0.5_f32; dim];
        x[7] = f32::NAN;
        assert!(matches!(
            screened.select(&x, &head, dot, &mut scratch),
            Screened::Fallback { .. }
        ));
        // All-zero input ties every row; the candidate limit forces fallback.
        let zero = vec![0.0_f32; dim];
        assert!(matches!(
            screened.select(&zero, &head, dot, &mut scratch),
            Screened::Fallback { .. }
        ));
        let huge = vec![f32::MAX / 4.0; dim];
        assert!(matches!(
            screened.select(&huge, &head, dot, &mut scratch),
            Screened::Fallback { .. }
        ));
    }

    #[test]
    fn rejects_nonfinite_weights() {
        let mut head = vec![0.25_f32; 4 * 64];
        head[70] = f32::INFINITY;
        assert!(ScreenedHead::build(&head, 4, 64).is_err());
    }
}
