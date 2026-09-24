//! Fused prefill QKV pass: split, per-head RMS norm, RoPE and the compact
//! cache append in one row kernel.
use rayon::prelude::*;

use crate::{config::ModelConfig, kernels};

use super::cache::LayerCache;

/// Prefill rows of `qkv` straight into `q` and a compact cache: per row, the
/// query heads and the GQA-expanded key heads are copied, RMS-normalized per
/// head and rotated, keys go to the cache's prefix rows and the unique value
/// heads to its values. Every element sees exactly the operations of the
/// separate split, `rms_norm`, rotation and `LayerCache::append` passes (the
/// AVX2 row keeps `rms_norm_row`'s reduction tree and the rotation's
/// products, so both row kernels are bitwise equal).
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_prefix_rows(
    c: &ModelConfig,
    rows: usize,
    qkv: &[f32],
    rope: &[[f32; 2]],
    q: &mut [f32],
    cache: &mut LayerCache,
    bf16: Option<(&mut Vec<u32>, &mut Vec<u32>)>,
    simd: kernels::Simd,
) {
    let LayerCache::Compact {
        prefix_k, v: values, ..
    } = cache
    else {
        unreachable!("fused prefill needs the compact cache");
    };
    let (qdim, kdim) = (c.query_dim(), c.kv_dim());
    let qkv_width = qdim + 2 * kdim;
    let rope_width = c.n_heads * (c.head_dim / 2);
    prefix_k.reserve_exact(rows * qdim);
    values.reserve_exact(rows * kdim);
    let (k_start, v_start) = (prefix_k.len(), values.len());
    let k_out = crate::team::SharedMut::new(prefix_k.spare_capacity_mut());
    let v_out = crate::team::SharedMut::new(values.spare_capacity_mut());
    let vector = c.head_dim == 64 && simd.resolved() != kernels::Simd::Scalar && native_row_available();
    // BF16 attention copies (`kernels::store_prefill_bf16_row`): keys
    // `[head][rows][32]`, value pairs `[kv_head][rows / 2][64]`.
    let pairs = rows.div_ceil(2);
    let bf16 = bf16.map(|(keys, values)| {
        keys.resize(c.n_heads * rows * (c.head_dim / 2), 0);
        values.resize(c.n_kv_heads * pairs * c.head_dim, 0);
        if rows % 2 == 1 {
            // The last pair's odd half has no row.
            for g in 0..c.n_kv_heads {
                values[(g * pairs + pairs - 1) * c.head_dim..(g * pairs + pairs) * c.head_dim].fill(0);
            }
        }
        (
            crate::team::SharedMut::new(&mut keys[..]),
            crate::team::SharedMut::new(&mut values[..]),
        )
    });
    q[..rows * qdim].par_chunks_mut(qdim).enumerate().for_each(|(row, q)| {
        let src = &qkv[row * qkv_width..(row + 1) * qkv_width];
        let rope = &rope[row * rope_width..(row + 1) * rope_width];
        // SAFETY: rows write disjoint, reserved (uninitialized) slots,
        // each fully overwritten before the lengths are set below.
        let (k, v) = unsafe {
            let k: &mut [std::mem::MaybeUninit<f32>] = k_out.slice(row * qdim, qdim);
            let v: &mut [std::mem::MaybeUninit<f32>] = v_out.slice(row * kdim, kdim);
            (
                std::slice::from_raw_parts_mut(k.as_mut_ptr().cast::<f32>(), qdim),
                std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), kdim),
            )
        };
        if vector {
            // SAFETY: the native vector ISA was detected above; head_dim is 64.
            unsafe { fused_row_native(c, src, rope, q, k, v) };
        } else {
            fused_row(c, src, rope, q, k, v);
        }
        if let Some((keys, values)) = &bf16 {
            // SAFETY: the caller checked `prefill_bf16_rows_available`;
            // buffers sized above; each row writes only its own slots.
            unsafe {
                kernels::store_prefill_bf16_row(k, v, row, rows, c.n_heads, c.n_kv_heads, keys.ptr(), values.ptr())
            };
        }
    });
    // SAFETY: every reserved slot of the new rows was written above.
    unsafe {
        prefix_k.set_len(k_start + rows * qdim);
        values.set_len(v_start + rows * kdim);
    }
}

/// One row of `fused_prefix_rows` (portable).
fn fused_row(c: &ModelConfig, src: &[f32], rope: &[[f32; 2]], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    let (qdim, kdim, hd) = (c.query_dim(), c.kv_dim(), c.head_dim);
    let group = c.n_heads / c.n_kv_heads;
    for head in 0..c.n_heads {
        let dst = head * hd..(head + 1) * hd;
        kernels::rms_norm_row(&src[dst.clone()], &mut q[dst.clone()], hd, f32::EPSILON, None);
        let key = qdim + (head / group) * hd;
        kernels::rms_norm_row(&src[key..key + hd], &mut k[dst], hd, f32::EPSILON, None);
    }
    for x in [&mut *q, &mut *k] {
        for head in 0..c.n_heads {
            for pair in 0..hd / 2 {
                let [cos, sin] = rope[head * (hd / 2) + pair];
                let p = head * hd + 2 * pair;
                let (a, b) = (x[p], x[p + 1]);
                x[p] = a * cos - b * sin;
                x[p + 1] = a * sin + b * cos;
            }
        }
    }
    v.copy_from_slice(&src[qdim + kdim..qdim + 2 * kdim]);
}

/// Whether `fused_row_native` runs on this CPU.
fn native_row_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// `fused_row_vector` on the native instruction set.
///
/// # Safety
/// [`native_row_available`] returned true; `head_dim == 64`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_row_native(
    c: &ModelConfig,
    src: &[f32],
    rope: &[[f32; 2]],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    unsafe { fused_row_vector::<crate::simd::Avx2>(c, src, rope, q, k, v) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn fused_row_native(
    c: &ModelConfig,
    src: &[f32],
    rope: &[[f32; 2]],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    unsafe { fused_row_vector::<crate::simd::Neon>(c, src, rope, q, k, v) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn fused_row_native(_: &ModelConfig, _: &[f32], _: &[[f32; 2]], _: &mut [f32], _: &mut [f32], _: &mut [f32]) {
    unreachable!("no native vector row on this architecture")
}

/// `fused_row` for 64-wide heads on instruction set `S`: the same reduction
/// tree and products as the portable row, so bitwise equal to it.
///
/// # Safety
/// `S`'s instruction set is enabled in the caller; `head_dim == 64`.
#[inline(always)]
unsafe fn fused_row_vector<S: crate::simd::Simd>(
    c: &ModelConfig,
    src: &[f32],
    rope: &[[f32; 2]],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    let qdim = c.query_dim();
    let group = c.n_heads / c.n_kv_heads;
    let rope = rope.as_ptr().cast::<f32>();
    // SAFETY: every head is 64 in-bounds floats; rope holds 32 pairs per head.
    unsafe {
        for head in 0..c.n_heads {
            let factors = rope.add(head * 64);
            norm_rope_head::<S>(src.as_ptr().add(head * 64), factors, q.as_mut_ptr().add(head * 64));
            let key = qdim + (head / group) * 64;
            norm_rope_head::<S>(src.as_ptr().add(key), factors, k.as_mut_ptr().add(head * 64));
        }
    }
    v.copy_from_slice(&src[qdim + c.kv_dim()..qdim + 2 * c.kv_dim()]);
}

/// `rms_norm_row` (width 64, no weight) then the pairwise rotation of one head.
#[inline(always)]
unsafe fn norm_rope_head<S: crate::simd::Simd>(x: *const f32, rope: *const f32, out: *mut f32) {
    unsafe {
        // `sum_squares_pairwise` on 32 values: squares, then lanes i += i + 16,
        // i += i + 8, i += i + 4, i += i + 2, i += i + 1.
        let leaf = |x: *const f32| {
            let square = |i: usize| {
                let v = S::load(x.add(i));
                S::mul(v, v)
            };
            let a = S::add(square(0), square(16));
            let b = S::add(square(8), square(24));
            S::sum_tree(S::add(a, b))
        };
        let sum = leaf(x) + leaf(x.add(32));
        let scale = S::splat((sum / 64.0 + f32::EPSILON).sqrt().recip());
        for j in 0..8 {
            let v = S::mul(S::load(x.add(8 * j)), scale);
            // Four [cos, sin] pairs: duplicate cos and sin into both lanes of a pair.
            let factors = S::load(rope.add(8 * j));
            let cos = S::dup_even(factors);
            let sin = S::dup_odd(factors);
            let swapped = S::swap_pairs(v);
            // Even lanes a*cos - b*sin, odd lanes b*cos + a*sin.
            let rotated = S::addsub(S::mul(v, cos), S::mul(swapped, sin));
            S::store(out.add(8 * j), rotated);
        }
    }
}

#[cfg(test)]
mod fused_row_tests {
    use super::*;

    /// The native and portable instantiations of the vector row against the
    /// scalar row.
    #[test]
    fn vector_rows_are_bitwise_the_portable_row() {
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let width = c.query_dim() + 2 * c.kv_dim();
        let mut state = 12345_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
        };
        for trial in 0..50 {
            let mut src: Vec<f32> = (0..width).map(|_| next()).collect();
            if trial % 5 == 0 {
                src[3] = 0.0;
                src[70] = 1e-30;
                src[200] = 3e4;
            }
            let rope: Vec<[f32; 2]> = (0..c.n_heads * c.head_dim / 2)
                .map(|_| {
                    let t = next();
                    [t.cos(), t.sin()]
                })
                .collect();
            let (mut q1, mut k1, mut v1) = (
                vec![0.0; c.query_dim()],
                vec![0.0; c.query_dim()],
                vec![0.0; c.kv_dim()],
            );
            let (mut q2, mut k2, mut v2) = (q1.clone(), k1.clone(), v1.clone());
            fused_row(&c, &src, &rope, &mut q1, &mut k1, &mut v1);
            unsafe { fused_row_vector::<crate::simd::Portable>(&c, &src, &rope, &mut q2, &mut k2, &mut v2) };
            for (a, b) in q1.iter().chain(&k1).chain(&v1).zip(q2.iter().chain(&k2).chain(&v2)) {
                assert_eq!(a.to_bits(), b.to_bits(), "portable trial {trial}");
            }
            if native_row_available() {
                unsafe { fused_row_native(&c, &src, &rope, &mut q2, &mut k2, &mut v2) };
                for (a, b) in q1.iter().chain(&k1).chain(&v1).zip(q2.iter().chain(&k2).chain(&v2)) {
                    assert_eq!(a.to_bits(), b.to_bits(), "native trial {trial}");
                }
            }
        }
    }
}
