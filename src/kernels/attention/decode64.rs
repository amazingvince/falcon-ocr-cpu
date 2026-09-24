//! Fixed-width (`head_dim == 64`) single-query attention for decode: the
//! online-softmax head (`online::head`) with the `crate::simd` kernels of the
//! native instruction set (AVX2/FMA on x86, NEON on aarch64) inlined into a
//! `#[target_feature]` function, one decode-team task per query head. The
//! helpers keep the same four-accumulator FMA order and horizontal reduction
//! as the function-pointer kernels, so every output is bit-identical to the
//! generic path; the tests below assert that.
//!
//! Promoted from `experiments/attention64` (expanded cache, measured
//! 6.15-9.43% lower full-page latency) and `experiments/attention64_compact`
//! (compact cache, a further 6.15-6.40%).
use super::{
    CompactKv, Geometry,
    online::{Isa, head},
};

/// One query row over every head of width 64 on the decode spin team.
///
/// # Safety
/// The native vector ISA must be available and the shapes validated as
/// `attention_with` does for `query_len == 1`, `head_dim == 64`.
pub(super) unsafe fn attention(q: &[f32], kv: &CompactKv<'_>, geometry: Geometry, sinks: &[f32], output: &mut [f32]) {
    let heads = output.len() / 64;
    let shared = crate::team::SharedMut::new(output);
    crate::team::for_each(heads, |qh| {
        // SAFETY: each task owns one disjoint 64-wide head of `output`.
        let out = unsafe { shared.slice(qh * 64, 64) };
        // SAFETY: the parent checked the ISA and all shapes before entering.
        // The target-feature function encloses the complete per-head loop;
        // team closures do not need to inherit the caller's target features.
        unsafe { native_head(qh, q, kv, &geometry, sinks, out) }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn native_head(qh: usize, q: &[f32], kv: &CompactKv<'_>, geometry: &Geometry, sinks: &[f32], out: &mut [f32]) {
    unsafe { head(&Isa::<crate::simd::Avx2>::NEW, qh, q, kv, geometry, sinks, out) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn native_head(qh: usize, q: &[f32], kv: &CompactKv<'_>, geometry: &Geometry, sinks: &[f32], out: &mut [f32]) {
    unsafe { head(&Isa::<crate::simd::Neon>::NEW, qh, q, kv, geometry, sinks, out) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::super::{CompactKv, Geometry, Simd, attention_with_simd, online, tiled};
    use crate::kernels::x86;

    fn supported() {
        assert!(
            Simd::Avx2.validate().is_ok(),
            "these tests require AVX2/FMA; no silent skip"
        );
    }

    fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if i % 97 == 0 {
                    -0.0
                } else {
                    (((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32) * scale
                }
            })
            .collect()
    }

    fn exact(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "bit difference at {i}: {x:?} vs {y:?}");
        }
    }

    /// The public entry point without the fixed-width dispatch: the tiled
    /// GEMM for several queries, else the function-pointer head.
    fn oracle(
        q: &[f32],
        kv: &CompactKv<'_>,
        query_len: usize,
        geometry: Geometry,
        sinks: &[f32],
        output: &mut [f32],
        simd: Simd,
    ) {
        let selected = simd.resolved();
        if query_len >= 4 && selected != Simd::Scalar {
            tiled::attention_gemm(q, kv, geometry, sinks, output);
        } else {
            online::attention(q, kv, geometry, sinks, output, selected);
        }
    }

    fn sinks(heads: usize) -> Vec<f32> {
        (0..heads)
            .map(|h| match h % 4 {
                0 => -1000.0,
                1 => 1000.0,
                2 => 0.0,
                _ => 2.25,
            })
            .collect()
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn compare_vectors(a: &[f32], b: &[f32], factor: f32) {
        unsafe {
            assert_eq!(
                crate::simd::dot::<crate::simd::Avx2>(a, b).to_bits(),
                x86::dot_avx2(a, b).to_bits()
            );
            let mut actual = a.to_vec();
            let mut expected = a.to_vec();
            crate::simd::axpy::<crate::simd::Avx2>(factor, b, &mut actual);
            x86::axpy_avx2(factor, b, &mut expected);
            exact(&actual, &expected);
        }
    }

    #[test]
    fn vectors_preserve_four_accumulators_and_fma() {
        supported();
        for seed in 1..=128 {
            let a = data(64, seed, if seed % 2 == 0 { 1e10 } else { 0.03125 });
            let b = data(64, seed + 997, if seed % 3 == 0 { 1e-10 } else { 8.0 });
            for factor in [-4.0, -0.0, 0.0, 0.125, 3.25] {
                unsafe { compare_vectors(&a, &b, factor) };
            }
        }
        let a: Vec<_> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        unsafe { compare_vectors(&a, &[1.0; 64], -0.0) };
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_case(
        kv_len: usize,
        offset: usize,
        image_start: usize,
        image_end: usize,
        query_len: usize,
        width: usize,
        simd: Simd,
        extreme: bool,
        threads: usize,
    ) {
        let heads = 16;
        let q = data(query_len * heads * width, 11, if extreme { 100.0 } else { 0.75 });
        let k = data(kv_len * heads * width, 23, if extreme { 100.0 } else { 0.75 });
        let v = data(kv_len * heads * width, 37, if extreme { 1e8 } else { 1.0 });
        let sinks = sinks(heads);
        let kv = CompactKv::expanded(&k, &v, kv_len, heads, width);
        let geometry = Geometry::new(offset, image_start, image_end);
        let mut actual = vec![f32::NAN; q.len()];
        let mut expected = actual.clone();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            oracle(&q, &kv, query_len, geometry, &sinks, &mut expected, simd);
            attention_with_simd(&q, &kv, query_len, geometry, &sinks, &mut actual, simd);
        });
        assert!(actual.iter().all(|x| x.is_finite()));
        exact(&actual, &expected);
    }

    #[test]
    fn causal_tile_tails_are_bit_exact() {
        supported();
        for kv in [1, 2, 17, 63, 64, 127, 128, 129, 143, 144, 161, 255, 256, 257, 1025] {
            compare_case(kv, kv - 1, 0, 0, 1, 64, Simd::Avx2, false, 4);
        }
    }

    #[test]
    fn image_and_causal_boundaries_are_bit_exact() {
        supported();
        for (kv, offset, start, end) in [
            (17, 0, 1, 16),
            (17, 1, 1, 16),
            (17, 15, 1, 16),
            (17, 16, 1, 16),
            (161, 0, 1, 129),
            (161, 1, 1, 129),
            (161, 128, 1, 129),
            (161, 129, 1, 129),
            (161, 160, 1, 129),
        ] {
            compare_case(kv, offset, start, end, 1, 64, Simd::Avx2, false, 4);
        }
    }

    #[test]
    fn extreme_logits_sinks_and_long_context_are_bit_exact() {
        supported();
        compare_case(257, 256, 1, 144, 1, 64, Simd::Avx2, true, 4);
        // Real full-page prefix and the full context endpoint; no timing assertion.
        for kv in [6544, 16384] {
            compare_case(kv, kv - 1, 1, 6540, 1, 64, Simd::Avx2, false, 4);
        }
    }

    #[test]
    fn auto_and_one_thread_are_bit_exact() {
        supported();
        compare_case(161, 160, 1, 129, 1, 64, Simd::Auto, false, 1);
    }

    #[test]
    fn other_shapes_backends_and_prefill_stay_on_generic_path() {
        supported();
        compare_case(17, 16, 1, 16, 1, 64, Simd::Scalar, false, 4);
        for width in [31, 63, 65, 80] {
            compare_case(17, 16, 1, 16, 1, width, Simd::Avx2, false, 4);
        }
        for rows in [2, 3, 4, 17] {
            compare_case(17, 17 - rows, 1, 16, rows, 64, Simd::Avx2, false, 4);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_case(
        total: usize,
        prefix: usize,
        offset: usize,
        image_start: usize,
        image_end: usize,
        rows: usize,
        heads: usize,
        kvheads: usize,
        width: usize,
        simd: Simd,
        extreme: bool,
    ) {
        let qw = heads * width;
        let kw = kvheads * width;
        let repeat = heads / kvheads;
        let q = data(rows * qw, 41, if extreme { 100.0 } else { 0.75 });
        let pk = data(prefix * qw, 53, if extreme { 100.0 } else { 0.75 });
        let gk = data((total - prefix) * kw, 67, if extreme { 100.0 } else { 0.75 });
        let v = data(total * kw, 79, if extreme { 1e8 } else { 1.0 });
        let sinks = sinks(heads);
        let kv = CompactKv {
            prefix_k: &pk,
            generated_k: &gk,
            v: &v,
            prefix_len: prefix,
            total_len: total,
            n_heads: heads,
            n_kv_heads: kvheads,
            head_dim: width,
        };
        let geometry = Geometry::new(offset, image_start, image_end);
        let mut expected = vec![f32::NAN; q.len()];
        let mut actual = expected.clone();
        let mut expanded_out = expected.clone();
        let mut expanded_k = vec![0.0; total * qw];
        let mut expanded_v = expanded_k.clone();
        for token in 0..total {
            for head in 0..heads {
                let target = token * qw + head * width;
                expanded_k[target..target + width].copy_from_slice(kv.key(token, head));
                expanded_v[target..target + width].copy_from_slice(kv.value(token, head / repeat));
            }
        }
        let expanded = CompactKv::expanded(&expanded_k, &expanded_v, total, heads, width);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            oracle(&q, &kv, rows, geometry, &sinks, &mut expected, simd);
            attention_with_simd(&q, &kv, rows, geometry, &sinks, &mut actual, simd);
            oracle(&q, &expanded, rows, geometry, &sinks, &mut expanded_out, simd);
        });
        assert!(actual.iter().all(|x| x.is_finite()));
        exact(&actual, &expected);
        exact(&actual, &expanded_out);
    }

    #[test]
    fn compact_zero_full_and_tile_crossovers_are_bit_exact() {
        supported();
        for total in [1, 17, 127, 128, 129, 257] {
            for prefix in [0, total / 2, total] {
                compact_case(total, prefix, total - 1, 0, 0, 1, 16, 8, 64, Simd::Avx2, false);
            }
        }
        for prefix in [127, 128, 129] {
            compact_case(257, prefix, 256, 1, prefix, 1, 16, 8, 64, Simd::Avx2, false);
        }
    }

    #[test]
    fn compact_image_and_generated_boundaries_are_bit_exact() {
        supported();
        for offset in [0, 1, 126, 127, 128, 129, 160] {
            compact_case(161, 129, offset, 1, 128, 1, 16, 8, 64, Simd::Avx2, false);
        }
    }

    #[test]
    fn compact_real_prefix_full_context_and_extremes_are_bit_exact() {
        supported();
        for (total, prefix) in [(6544, 6544), (6545, 6544), (16384, 6544)] {
            compact_case(total, prefix, total - 1, 1, 6540, 1, 16, 8, 64, Simd::Avx2, false);
        }
        compact_case(257, 129, 256, 1, 129, 1, 16, 8, 64, Simd::Avx2, true);
    }

    #[test]
    fn compact_gqa_repeats_and_auto_are_bit_exact() {
        supported();
        for kvheads in [4, 8, 16] {
            compact_case(257, 129, 256, 1, 128, 1, 16, kvheads, 64, Simd::Auto, false);
        }
    }

    #[test]
    fn compact_other_shapes_and_prefill_stay_on_generic_path() {
        supported();
        for width in [31, 63, 65, 80] {
            compact_case(17, 9, 16, 1, 8, 1, 16, 8, width, Simd::Avx2, false);
        }
        for rows in [2, 3, 4, 17] {
            compact_case(17, 9, 17 - rows, 1, 8, rows, 16, 8, 64, Simd::Avx2, false);
        }
        compact_case(17, 9, 16, 1, 8, 1, 16, 8, 64, Simd::Scalar, false);
    }
}
