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
    #[cfg(target_arch = "x86_64")]
    let vector = c.head_dim == 64
        && simd.resolved() != kernels::Simd::Scalar
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let vector = {
        let _ = simd;
        false
    };
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
        #[cfg(target_arch = "x86_64")]
        let done = vector && {
            // SAFETY: AVX2/FMA detected above; head_dim is 64.
            unsafe { fused_row_avx2(c, src, rope, q, k, v) };
            true
        };
        #[cfg(not(target_arch = "x86_64"))]
        let done = false;
        if !done {
            let _ = vector;
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

/// `fused_row` for 64-wide heads with AVX2: the same reduction tree and
/// products, so bitwise equal.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_row_avx2(c: &ModelConfig, src: &[f32], rope: &[[f32; 2]], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    let qdim = c.query_dim();
    let group = c.n_heads / c.n_kv_heads;
    let rope = rope.as_ptr().cast::<f32>();
    // SAFETY: every head is 64 in-bounds floats; rope holds 32 pairs per head.
    unsafe {
        for head in 0..c.n_heads {
            let factors = rope.add(head * 64);
            norm_rope_head_avx2(src.as_ptr().add(head * 64), factors, q.as_mut_ptr().add(head * 64));
            let key = qdim + (head / group) * 64;
            norm_rope_head_avx2(src.as_ptr().add(key), factors, k.as_mut_ptr().add(head * 64));
        }
    }
    v.copy_from_slice(&src[qdim + c.kv_dim()..qdim + 2 * c.kv_dim()]);
}

/// `rms_norm_row` (width 64, no weight) then the pairwise rotation of one head.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_rope_head_avx2(x: *const f32, rope: *const f32, out: *mut f32) {
    use std::arch::x86_64::*;
    // `sum_squares_pairwise` on 32 values: squares, then lanes i += i + 16,
    // i += i + 8, i += i + 4, i += i + 2, i += i + 1.
    unsafe fn leaf(x: *const f32) -> f32 {
        unsafe {
            let square = |i: usize| {
                let v = _mm256_loadu_ps(x.add(i));
                _mm256_mul_ps(v, v)
            };
            let a = _mm256_add_ps(square(0), square(16));
            let b = _mm256_add_ps(square(8), square(24));
            let w = _mm256_add_ps(a, b);
            let h = _mm_add_ps(_mm256_castps256_ps128(w), _mm256_extractf128_ps(w, 1));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)))
        }
    }
    unsafe {
        let sum = leaf(x) + leaf(x.add(32));
        let scale = _mm256_set1_ps((sum / 64.0 + f32::EPSILON).sqrt().recip());
        for j in 0..8 {
            let v = _mm256_mul_ps(_mm256_loadu_ps(x.add(8 * j)), scale);
            // Four [cos, sin] pairs: duplicate cos and sin into both lanes of a pair.
            let factors = _mm256_loadu_ps(rope.add(8 * j));
            let cos = _mm256_moveldup_ps(factors);
            let sin = _mm256_movehdup_ps(factors);
            let swapped = _mm256_permute_ps(v, 0b1011_0001);
            // Even lanes a*cos - b*sin, odd lanes b*cos + a*sin.
            let rotated = _mm256_addsub_ps(_mm256_mul_ps(v, cos), _mm256_mul_ps(swapped, sin));
            _mm256_storeu_ps(out.add(8 * j), rotated);
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod fused_row_tests {
    use super::*;

    #[test]
    fn avx2_row_is_bitwise_the_portable_row() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
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
            unsafe { fused_row_avx2(&c, &src, &rope, &mut q2, &mut k2, &mut v2) };
            for (a, b) in q1.iter().chain(&k1).chain(&v1).zip(q2.iter().chain(&k2).chain(&v2)) {
                assert_eq!(a.to_bits(), b.to_bits(), "trial {trial}");
            }
        }
    }
}
