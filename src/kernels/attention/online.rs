//! The online-softmax attention head: one body for every cache layout and
//! instruction set. [`Arith`] supplies the dot product, AXPY and exp, so the
//! fixed-width decode kernels (`decode64`, inlined `crate::simd` kernels) and
//! the generic function-pointer path share the operation order: 128-key
//! tiles, rescale before PV, the serial denominator, the sink applied once at
//! the end. The tests in `decode64` assert the two are bit-identical.
use rayon::prelude::*;

use super::{CompactKv, Geometry};
use crate::kernels::{Axpy, Dot, Simd, axpy_kernel, dot_kernel};

/// The arithmetic of one head: dot, AXPY and `exp(x - shift)`.
pub(super) trait Arith {
    /// # Safety
    /// The implementation's instruction set is available.
    unsafe fn dot(&self, a: &[f32], b: &[f32]) -> f32;
    /// # Safety
    /// As [`Arith::dot`]; `x` and `y` have equal lengths.
    unsafe fn axpy(&self, a: f32, x: &[f32], y: &mut [f32]);
    /// `values[i] = exp(values[i] - shift)`.
    ///
    /// # Safety
    /// As [`Arith::dot`].
    unsafe fn exp_shifted(&self, values: &mut [f32], shift: f32);
}

/// Function-pointer kernels and the platform `expf`: any head width, any
/// backend (the generic path and the bit-exact oracle of the fixed-width one).
pub(super) struct FnPtr {
    dot: Dot,
    axpy: Axpy,
}

impl FnPtr {
    /// Kernels of the resolved `selected` backend.
    pub(super) fn new(selected: Simd) -> Self {
        Self {
            dot: dot_kernel(selected),
            axpy: axpy_kernel(selected),
        }
    }
}

impl Arith for FnPtr {
    #[inline(always)]
    unsafe fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        (self.dot)(a, b)
    }
    #[inline(always)]
    unsafe fn axpy(&self, a: f32, x: &[f32], y: &mut [f32]) {
        (self.axpy)(a, x, y)
    }
    #[inline(always)]
    unsafe fn exp_shifted(&self, values: &mut [f32], shift: f32) {
        for value in values {
            *value = (*value - shift).exp();
        }
    }
}

/// The 64-wide `crate::simd` kernels of instruction set `S`, inlined into a
/// `#[target_feature]` caller: the same four-accumulator FMA order and
/// reduction as the function-pointer kernels, and an exp bitwise equal to
/// the platform `expf` on the softmax domain.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) struct Isa<S>(std::marker::PhantomData<S>);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<S: crate::simd::Simd> Isa<S> {
    pub(super) const NEW: Self = Self(std::marker::PhantomData);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<S: crate::simd::Simd> Arith for Isa<S> {
    #[inline(always)]
    unsafe fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { crate::simd::dot::<S>(a, b) }
    }
    #[inline(always)]
    unsafe fn axpy(&self, a: f32, x: &[f32], y: &mut [f32]) {
        unsafe { crate::simd::axpy::<S>(a, x, y) }
    }
    #[inline(always)]
    unsafe fn exp_shifted(&self, values: &mut [f32], shift: f32) {
        unsafe { S::exp_shifted(values, shift) }
    }
}

/// Query head `qh` (query `qh / n_heads`, head `qh % n_heads`) of one call,
/// written to `out` (`head_dim` values).
///
/// # Safety
/// `arith`'s instruction set is available; shapes were validated by the
/// entry point (`CompactKv::validate`).
#[inline(always)]
pub(super) unsafe fn head<A: Arith>(
    arith: &A,
    qh: usize,
    q: &[f32],
    kv: &CompactKv<'_>,
    geometry: &Geometry,
    sinks: &[f32],
    out: &mut [f32],
) {
    let head_dim = kv.head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let query = qh / kv.n_heads;
    let head = qh % kv.n_heads;
    let kv_head = head / kv.repeat();
    let absolute_query = geometry.query_offset + query;
    let visible_end = geometry.visible_end(absolute_query);
    let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
    out.fill(0.0);
    let mut running_max = f32::NEG_INFINITY;
    let mut denominator = 0.0_f32;
    const TILE: usize = 128;
    let mut logits = [0.0_f32; TILE];
    for start in (0..visible_end).step_by(TILE) {
        let len = (visible_end - start).min(TILE);
        let mut block_max = f32::NEG_INFINITY;
        for (j, logit) in logits[..len].iter_mut().enumerate() {
            // SAFETY: the caller's contract.
            *logit = unsafe { arith.dot(qvec, kv.key(start + j, head)) } * scale;
            block_max = block_max.max(*logit);
        }
        let new_max = running_max.max(block_max);
        let rescale = if running_max == f32::NEG_INFINITY {
            0.0
        } else {
            (running_max - new_max).exp()
        };
        for value in out.iter_mut() {
            *value *= rescale;
        }
        denominator *= rescale;
        // SAFETY: the caller's contract.
        unsafe { arith.exp_shifted(&mut logits[..len], new_max) };
        for (j, &probability) in logits[..len].iter().enumerate() {
            denominator += probability;
            // SAFETY: the caller's contract; both slices hold `head_dim` values.
            unsafe { arith.axpy(probability, kv.value(start + j, kv_head), out) };
        }
        running_max = new_max;
    }
    let logsumexp = running_max + denominator.ln();
    let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
    for value in out {
        *value = (*value / denominator) * sink_scale;
    }
}

/// Every query head on the caller's Rayon pool with the function-pointer
/// kernels of `selected` (already resolved and validated): the generic path
/// for every shape or backend the fixed-width kernels do not cover.
pub(super) fn attention(
    q: &[f32],
    kv: &CompactKv<'_>,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    selected: Simd,
) {
    let arith = FnPtr::new(selected);
    output.par_chunks_mut(kv.head_dim).enumerate().for_each(|(qh, out)| {
        // SAFETY: function-pointer kernels of a validated backend; shapes
        // validated by the entry point.
        unsafe { head(&arith, qh, q, kv, &geometry, sinks, out) }
    });
}
