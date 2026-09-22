//! Allocation-free AVX2/FMA replacements for the two per-tile `gemm` calls of
//! compact prefill attention (`head_dim == 64`), bit-identical to them.
//!
//! For the shapes routed here, gemm 0.19 transposes the destination and runs
//! its main microkernel: every output element is one FMA chain over the depth
//! starting from +0.0, followed by `beta * acc` (QK, `read_dst == false`) or
//! `acc + dst` (PV, `alpha == beta == 1`). The helpers below keep exactly that
//! per-element sequence while vectorizing across independent elements, and
//! avoid gemm's per-call packing and heap allocation (about 7.5 million calls
//! per full page). Shapes for which gemm takes a different path (horizontal
//! dot products when `keys * queries <= 256`, `gevv` for `keys <= 2`, `gemv`
//! for a single query) stay on gemm; see [`qk_uses_main_path`] and
//! [`pv_uses_main_path`]. Tests compare every tile shape against gemm itself.
use std::arch::x86_64::*;

pub(super) const QUERY_TILE: usize = 32;
pub(super) const KEY_TILE: usize = 128;
const HEAD_DIM: usize = 64;

/// gemm's main microkernel handles QK (m = keys, n = queries after its
/// transpose, contiguous depth on both sides) only above its horizontal limit.
pub(super) fn qk_uses_main_path(queries: usize, keys: usize) -> bool {
    queries * keys > 16 * 16
}

/// PV (m = 64, n = queries, depth = keys) avoids gevv (depth <= 2) and the
/// single-column gemv.
pub(super) fn pv_uses_main_path(queries: usize, keys: usize) -> bool {
    keys >= 3 && queries >= 2
}

/// Transposed K tile, `[d][KEY_TILE]`, reused by every row group of one tile.
pub(super) struct KeyTile(pub [f32; HEAD_DIM * KEY_TILE]);

/// `scores[row * KEY_TILE + key] = scale * sum_d q[row][d] * k[key][d]`, with
/// the depth summed in ascending order by FMA from +0.0 for each element.
///
/// # Safety
/// AVX2/FMA must be available; `q` addresses `queries` rows of 64 floats at
/// `q_stride`, `k` addresses `keys` rows of 64 floats at `k_stride`, and
/// `queries <= QUERY_TILE`, `keys <= KEY_TILE`.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn qk(
    q: *const f32,
    q_stride: usize,
    queries: usize,
    k: *const f32,
    k_stride: usize,
    keys: usize,
    scale: f32,
    scores: &mut [f32; QUERY_TILE * KEY_TILE],
    transposed: &mut KeyTile,
) {
    debug_assert!(queries <= QUERY_TILE && keys <= KEY_TILE);
    let padded = keys.next_multiple_of(8);
    let kt = &mut transposed.0;
    unsafe {
        for d in 0..HEAD_DIM {
            let row = &mut kt[d * KEY_TILE..d * KEY_TILE + padded];
            for (key, value) in row.iter_mut().enumerate() {
                *value = if key < keys {
                    *k.add(key * k_stride + d)
                } else {
                    0.0
                };
            }
        }
        let scale = _mm256_set1_ps(scale);
        let mut row = 0;
        while row < queries {
            let rows = (queries - row).min(6);
            let mut key = 0;
            while key < padded {
                match rows {
                    6 => qk_block::<6>(q, q_stride, row, kt, key, scale, scores),
                    5 => qk_block::<5>(q, q_stride, row, kt, key, scale, scores),
                    4 => qk_block::<4>(q, q_stride, row, kt, key, scale, scores),
                    3 => qk_block::<3>(q, q_stride, row, kt, key, scale, scores),
                    2 => qk_block::<2>(q, q_stride, row, kt, key, scale, scores),
                    _ => qk_block::<1>(q, q_stride, row, kt, key, scale, scores),
                }
                key += 8;
            }
            row += rows;
        }
    }
}

#[inline(always)]
unsafe fn qk_block<const R: usize>(
    q: *const f32,
    q_stride: usize,
    row: usize,
    kt: &[f32; HEAD_DIM * KEY_TILE],
    key: usize,
    scale: __m256,
    scores: &mut [f32; QUERY_TILE * KEY_TILE],
) {
    unsafe {
        let mut acc = [_mm256_setzero_ps(); R];
        for d in 0..HEAD_DIM {
            let kv = _mm256_loadu_ps(kt.as_ptr().add(d * KEY_TILE + key));
            for (r, acc) in acc.iter_mut().enumerate() {
                let qv = _mm256_set1_ps(*q.add((row + r) * q_stride + d));
                *acc = _mm256_fmadd_ps(kv, qv, *acc);
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            _mm256_storeu_ps(
                scores.as_mut_ptr().add((row + r) * KEY_TILE + key),
                _mm256_mul_ps(scale, *acc),
            );
        }
    }
}

/// `out[row][d] = out[row][d] + sum_j p[row][j] * v[j][d]`, with the keys
/// summed in ascending order by FMA from +0.0 for each element.
///
/// # Safety
/// AVX2/FMA must be available; `p` is the row-major probability tile; `v`
/// addresses `keys` rows of 64 floats at `v_stride`; `out` addresses
/// `queries` rows of 64 floats at `out_stride`; `queries <= QUERY_TILE`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn pv(
    p: &[f32; QUERY_TILE * KEY_TILE],
    queries: usize,
    keys: usize,
    v: *const f32,
    v_stride: usize,
    out: *mut f32,
    out_stride: usize,
) {
    debug_assert!(queries <= QUERY_TILE && keys <= KEY_TILE);
    unsafe {
        let mut row = 0;
        while row < queries {
            let rows = (queries - row).min(6);
            for d in (0..HEAD_DIM).step_by(16) {
                match rows {
                    6 => pv_block::<6>(p, row, keys, v, v_stride, d, out, out_stride),
                    5 => pv_block::<5>(p, row, keys, v, v_stride, d, out, out_stride),
                    4 => pv_block::<4>(p, row, keys, v, v_stride, d, out, out_stride),
                    3 => pv_block::<3>(p, row, keys, v, v_stride, d, out, out_stride),
                    2 => pv_block::<2>(p, row, keys, v, v_stride, d, out, out_stride),
                    _ => pv_block::<1>(p, row, keys, v, v_stride, d, out, out_stride),
                }
            }
            row += rows;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_block<const R: usize>(
    p: &[f32; QUERY_TILE * KEY_TILE],
    row: usize,
    keys: usize,
    v: *const f32,
    v_stride: usize,
    d: usize,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); 2]; R];
        for j in 0..keys {
            let v0 = _mm256_loadu_ps(v.add(j * v_stride + d));
            let v1 = _mm256_loadu_ps(v.add(j * v_stride + d + 8));
            for (r, acc) in acc.iter_mut().enumerate() {
                let pv = _mm256_set1_ps(*p.get_unchecked((row + r) * KEY_TILE + j));
                acc[0] = _mm256_fmadd_ps(v0, pv, acc[0]);
                acc[1] = _mm256_fmadd_ps(v1, pv, acc[1]);
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            let dst = out.add((row + r) * out_stride + d);
            _mm256_storeu_ps(dst, _mm256_add_ps(acc[0], _mm256_loadu_ps(dst)));
            _mm256_storeu_ps(
                dst.add(8),
                _mm256_add_ps(acc[1], _mm256_loadu_ps(dst.add(8))),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = ((state >> 8) as f64 / 16_777_216.0 * 8.0 - 4.0) as f32;
                // Signed zeros and a subnormal exercise edge rounding.
                match i % 97 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::from_bits(3),
                    _ => x,
                }
            })
            .collect()
    }

    fn available() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[test]
    fn qk_matches_gemm_main_path_for_every_tile_shape() {
        if !available() {
            return;
        }
        let (q_stride, k_stride) = (1024, 1024);
        let q = values(QUERY_TILE * q_stride, 3);
        let k = values(KEY_TILE * k_stride, 5);
        let mut transposed = KeyTile([0.0; HEAD_DIM * KEY_TILE]);
        let mut checked = 0;
        for queries in 1..=QUERY_TILE {
            for keys in 1..=KEY_TILE {
                if !qk_uses_main_path(queries, keys) {
                    continue;
                }
                let mut expected = [f32::NAN; QUERY_TILE * KEY_TILE];
                unsafe {
                    gemm::gemm(
                        queries,
                        keys,
                        HEAD_DIM,
                        expected.as_mut_ptr(),
                        1,
                        KEY_TILE as isize,
                        false,
                        q.as_ptr(),
                        1,
                        q_stride as isize,
                        k.as_ptr(),
                        k_stride as isize,
                        1,
                        0.0_f32,
                        0.125,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                let mut actual = [f32::NAN; QUERY_TILE * KEY_TILE];
                unsafe {
                    qk(
                        q.as_ptr(),
                        q_stride,
                        queries,
                        k.as_ptr(),
                        k_stride,
                        keys,
                        0.125,
                        &mut actual,
                        &mut transposed,
                    );
                }
                for row in 0..queries {
                    for key in 0..keys {
                        let at = row * KEY_TILE + key;
                        assert_eq!(
                            actual[at].to_bits(),
                            expected[at].to_bits(),
                            "queries {queries} keys {keys} row {row} key {key}"
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 3000);
    }

    #[test]
    fn pv_matches_gemm_main_path_for_every_tile_shape() {
        if !available() {
            return;
        }
        let (v_stride, out_stride) = (512, 1024);
        let v = values(KEY_TILE * v_stride, 7);
        let mut p = [0.0_f32; QUERY_TILE * KEY_TILE];
        for (x, y) in p.iter_mut().zip(values(QUERY_TILE * KEY_TILE, 11)) {
            // Probabilities in [0, 1], with exact zeros from masking.
            *x = if y < -3.0 {
                0.0
            } else {
                (y.abs() / 4.0).min(1.0)
            };
        }
        let initial = values(QUERY_TILE * out_stride, 13);
        let mut checked = 0;
        for queries in 1..=QUERY_TILE {
            for keys in 1..=KEY_TILE {
                if !pv_uses_main_path(queries, keys) {
                    continue;
                }
                let mut expected = initial.clone();
                unsafe {
                    gemm::gemm(
                        queries,
                        HEAD_DIM,
                        keys,
                        expected.as_mut_ptr(),
                        1,
                        out_stride as isize,
                        true,
                        p.as_ptr(),
                        1,
                        KEY_TILE as isize,
                        v.as_ptr(),
                        1,
                        v_stride as isize,
                        1.0_f32,
                        1.0_f32,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                let mut actual = initial.clone();
                unsafe {
                    pv(
                        &p,
                        queries,
                        keys,
                        v.as_ptr(),
                        v_stride,
                        actual.as_mut_ptr(),
                        out_stride,
                    );
                }
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "queries {queries} keys {keys} index {i}"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 3000);
    }
}
