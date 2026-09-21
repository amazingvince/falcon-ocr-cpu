//! Diagnostic-only copied-project cache. Never compiled into the live runner.
use crate::kernels::{self, Simd};
use anyhow::{Context, Result, ensure};

pub(crate) struct TemporalCache {
    temporal: Vec<f32>,
    spatial: Vec<f32>,
    generated: Vec<f32>,
    values: Vec<f32>,
    prefix: usize,
    capacity: usize,
    len: usize,
}

impl TemporalCache {
    pub(crate) fn new(
        prefix: usize,
        capacity: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
    ) -> Result<Self> {
        ensure!(
            (heads, kv_heads, dim) == (16, 8, 64),
            "temporal candidate requires 16Q/8KV/64 channels"
        );
        ensure!(
            prefix > 0 && prefix <= capacity,
            "invalid temporal candidate prefix/capacity"
        );
        fn reserve(rows: usize, width: usize) -> Result<Vec<f32>> {
            let mut result = Vec::new();
            result
                .try_reserve_exact(rows.checked_mul(width).context("cache size overflow")?)
                .context("allocate temporal candidate cache")?;
            Ok(result)
        }
        Ok(Self {
            temporal: reserve(prefix, 256)?,
            spatial: reserve(prefix, 512)?,
            generated: reserve(capacity - prefix, 512)?,
            values: reserve(capacity, 512)?,
            prefix,
            capacity,
            len: 0,
        })
    }

    pub(crate) fn append(&mut self, k: &[f32], v: &[f32], offset: usize) -> Result<()> {
        ensure!(
            k.len() == v.len() && k.len().is_multiple_of(1024),
            "invalid expanded K/V shape"
        );
        let rows = k.len() / 1024;
        ensure!(offset == self.len, "cache append is not contiguous");
        ensure!(
            offset
                .checked_add(rows)
                .is_some_and(|end| end <= self.capacity),
            "cache capacity exceeded"
        );
        let prefill = offset == 0;
        ensure!(
            if prefill {
                rows == self.prefix
            } else {
                offset >= self.prefix && rows == 1
            },
            "only one complete prefix and single-token continuation are supported"
        );
        // Validate all duplicate bits before mutating any buffer. Signed zero and
        // NaN payloads are storage bits too; float equality would be insufficient.
        for (kg, vg) in k.chunks_exact(128).zip(v.chunks_exact(128)) {
            let count = if prefill { 32 } else { 64 };
            ensure!(
                kg[..count]
                    .iter()
                    .zip(&kg[64..64 + count])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "temporal candidate duplicated K bits differ"
            );
            ensure!(
                vg[..64]
                    .iter()
                    .zip(&vg[64..])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "temporal candidate duplicated V bits differ"
            );
        }
        for (kg, vg) in k.chunks_exact(128).zip(v.chunks_exact(128)) {
            if prefill {
                self.temporal.extend_from_slice(&kg[..32]);
                self.spatial.extend_from_slice(&kg[32..64]);
                self.spatial.extend_from_slice(&kg[96..128]);
            } else {
                self.generated.extend_from_slice(&kg[..64]);
            }
            self.values.extend_from_slice(&vg[..64]);
        }
        self.len += rows;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &self,
        q: &[f32],
        current_expanded_k: &[f32],
        rows: usize,
        total_len: usize,
        offset: usize,
        image_start: usize,
        image_end: usize,
        sinks: &[f32],
        output: &mut [f32],
        simd: Simd,
    ) -> Result<()> {
        ensure!(
            self.len == total_len && offset.checked_add(rows) == Some(total_len),
            "attention/cache interval differs"
        );
        if offset == 0 {
            ensure!(
                rows == self.prefix && self.generated.is_empty(),
                "partial prefill unsupported"
            );
            ensure!(
                current_expanded_k.len() == rows * 1024,
                "prefill must borrow expanded workspace K"
            );
            // Preserve the original expanded workspace and existing GEMM strides.
            kernels::attention_compact_with_simd(
                q,
                current_expanded_k,
                &[],
                &self.values,
                rows,
                self.prefix,
                total_len,
                16,
                8,
                64,
                offset,
                image_start,
                image_end,
                sinks,
                output,
                simd,
            );
        } else {
            ensure!(
                offset >= self.prefix && rows == 1,
                "multiquery continuation unsupported"
            );
            kernels::attention_temporal_candidate_with_simd(
                q,
                &self.temporal,
                &self.spatial,
                &self.generated,
                &self.values,
                rows,
                self.prefix,
                total_len,
                16,
                8,
                64,
                offset,
                image_start,
                image_end,
                sinks,
                output,
                simd,
            );
        }
        Ok(())
    }

    #[cfg(test)]
    fn owned_capacity_bytes(&self) -> usize {
        [
            self.temporal.capacity(),
            self.spatial.capacity(),
            self.generated.capacity(),
            self.values.capacity(),
        ]
        .iter()
        .sum::<usize>()
            * std::mem::size_of::<f32>()
    }
    #[cfg(test)]
    fn reconstruct(&self, token: usize, head: usize) -> [f32; 64] {
        assert!(token < self.len && head < 16);
        let mut key = [0.; 64];
        if token < self.prefix {
            let t = (token * 8 + head / 2) * 32;
            let s = (token * 16 + head) * 32;
            key[..32].copy_from_slice(&self.temporal[t..t + 32]);
            key[32..].copy_from_slice(&self.spatial[s..s + 32]);
        } else {
            let begin = ((token - self.prefix) * 8 + head / 2) * 64;
            key.copy_from_slice(&self.generated[begin..begin + 64]);
        }
        key
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn operands(prefix: usize, generated: usize) -> (Vec<f32>, Vec<f32>) {
        let mut k = vec![0.; (prefix + generated) * 1024];
        let mut v = k.clone();
        for token in 0..prefix + generated {
            for head in 0..16 {
                for dim in 0..64 {
                    let shared = head / 2;
                    let kh = if token < prefix && dim >= 32 {
                        head
                    } else {
                        shared
                    };
                    let index = (token * 16 + head) * 64 + dim;
                    k[index] = (((token * 41 + kh * 13 + dim * 7) % 211) as i32 - 105) as f32 / 37.;
                    v[index] =
                        (((token * 29 + shared * 17 + dim * 3) % 199) as i32 - 99) as f32 / 19.;
                }
            }
        }
        (k, v)
    }
    pub(crate) fn compact_v(v: &[f32]) -> Vec<f32> {
        v.chunks_exact(128)
            .flat_map(|g| g[..64].iter().copied())
            .collect()
    }
    pub(crate) fn backends() -> Vec<Simd> {
        [Simd::Scalar, Simd::Auto, Simd::Avx2, Simd::Avx512]
            .into_iter()
            .filter(|x| x.validate().is_ok())
            .collect()
    }
    pub(crate) fn bits_equal(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "bit difference"
        );
    }
    fn fill(cache: &mut TemporalCache, k: &[f32], v: &[f32], prefix: usize) {
        cache
            .append(&k[..prefix * 1024], &v[..prefix * 1024], 0)
            .unwrap();
        for token in prefix..k.len() / 1024 {
            cache
                .append(
                    &k[token * 1024..(token + 1) * 1024],
                    &v[token * 1024..(token + 1) * 1024],
                    token,
                )
                .unwrap();
        }
    }

    #[test]
    fn storage_bits_including_signed_zero_and_nan_payload() {
        let (mut k, mut v) = operands(3, 2);
        for &(i, bits) in &[(0, 0x80000000), (1, 0x7fc01234), (2, 0x00000001)] {
            k[i] = f32::from_bits(bits);
            k[i + 64] = k[i];
            v[i] = k[i];
            v[i + 64] = k[i];
        }
        let mut cache = TemporalCache::new(3, 5, 16, 8, 64).unwrap();
        fill(&mut cache, &k, &v, 3);
        for token in 0..5 {
            for head in 0..16 {
                bits_equal(
                    &cache.reconstruct(token, head),
                    &k[(token * 16 + head) * 64..(token * 16 + head + 1) * 64],
                );
            }
        }
        bits_equal(&cache.values, &compact_v(&v));
        assert_ne!(k[32].to_bits(), k[96].to_bits());
    }

    #[test]
    fn invalid_duplicate_rejected_before_mutation() {
        for kind in 0..3 {
            let (mut k, mut v) = operands(3, 1);
            let mut cache = TemporalCache::new(3, 4, 16, 8, 64).unwrap();
            if kind < 2 {
                let values = if kind == 0 { &mut k } else { &mut v };
                values[3 * 1024 - 64] = f32::from_bits(values[3 * 1024 - 128].to_bits() ^ 1);
                assert!(cache.append(&k[..3 * 1024], &v[..3 * 1024], 0).is_err());
                assert_eq!(cache.len, 0);
                assert!(
                    cache.temporal.is_empty()
                        && cache.spatial.is_empty()
                        && cache.values.is_empty()
                );
            } else {
                cache.append(&k[..3 * 1024], &v[..3 * 1024], 0).unwrap();
                k[3 * 1024 + 96] = f32::from_bits(k[3 * 1024 + 32].to_bits() ^ 1);
                assert!(cache.append(&k[3 * 1024..], &v[3 * 1024..], 3).is_err());
                assert_eq!(cache.len, 3);
                assert!(cache.generated.is_empty());
                assert_eq!(cache.values.len(), 3 * 512);
            }
        }
    }

    #[test]
    fn unsupported_dimensions_partial_prefix_continuation_and_capacity() {
        assert!(TemporalCache::new(3, 4, 8, 8, 64).is_err());
        assert!(TemporalCache::new(3, 4, 16, 4, 64).is_err());
        assert!(TemporalCache::new(3, 4, 16, 8, 32).is_err());
        assert!(TemporalCache::new(0, 4, 16, 8, 64).is_err());
        assert!(TemporalCache::new(3, 2, 16, 8, 64).is_err());
        assert!(TemporalCache::new(usize::MAX, usize::MAX, 16, 8, 64).is_err());
        let (k, v) = operands(3, 2);
        let mut c = TemporalCache::new(3, 4, 16, 8, 64).unwrap();
        assert!(c.append(&k[..1024], &v[..1024], 0).is_err());
        c.append(&k[..3072], &v[..3072], 0).unwrap();
        assert!(c.append(&k[3072..], &v[3072..], 3).is_err());
        assert!(c.append(&k[3072..4096], &v[3072..4096], 2).is_err());
        c.append(&k[3072..4096], &v[3072..4096], 3).unwrap();
        assert!(c.append(&k[4096..], &v[4096..], 4).is_err());
        assert_eq!(c.len, 4);
        assert!(
            c.attention(
                &[0.; 2048],
                &[],
                2,
                4,
                2,
                1,
                2,
                &[0.; 16],
                &mut [0.; 2048],
                Simd::Scalar
            )
            .is_err()
        );
    }

    #[test]
    fn unchanged_prefill_and_single_decode_all_supported_backends() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            for simd in backends() {
                for prefix in [3, 17, 127, 128, 129] {
                    let (k, v) = operands(prefix, 1);
                    let cv = compact_v(&v);
                    let mut cache = TemporalCache::new(prefix, prefix + 1, 16, 8, 64).unwrap();
                    cache
                        .append(&k[..prefix * 1024], &v[..prefix * 1024], 0)
                        .unwrap();
                    let q: Vec<_> = (0..prefix * 1024)
                        .map(|i| ((i * 17 % 101) as i32 - 50) as f32 / 31.)
                        .collect();
                    let sinks: Vec<_> = (0..16)
                        .map(|h| if h % 3 == 0 { 40. } else { -2. })
                        .collect();
                    let mut expected = vec![0.; q.len()];
                    let mut actual = expected.clone();
                    kernels::attention_compact_with_simd(
                        &q,
                        &k[..prefix * 1024],
                        &[],
                        &cv[..prefix * 512],
                        prefix,
                        prefix,
                        prefix,
                        16,
                        8,
                        64,
                        0,
                        1,
                        prefix - 1,
                        &sinks,
                        &mut expected,
                        simd,
                    );
                    cache
                        .attention(
                            &q,
                            &k[..prefix * 1024],
                            prefix,
                            prefix,
                            0,
                            1,
                            prefix - 1,
                            &sinks,
                            &mut actual,
                            simd,
                        )
                        .unwrap();
                    bits_equal(&actual, &expected);
                    cache
                        .append(&k[prefix * 1024..], &v[prefix * 1024..], prefix)
                        .unwrap();
                    let gk = compact_v(&k[prefix * 1024..]);
                    kernels::attention_compact_with_simd(
                        &q[..1024],
                        &k[..prefix * 1024],
                        &gk,
                        &cv,
                        1,
                        prefix,
                        prefix + 1,
                        16,
                        8,
                        64,
                        prefix,
                        1,
                        prefix - 1,
                        &sinks,
                        &mut expected[..1024],
                        simd,
                    );
                    cache
                        .attention(
                            &q[..1024],
                            &k[prefix * 1024..],
                            1,
                            prefix + 1,
                            prefix,
                            1,
                            prefix - 1,
                            &sinks,
                            &mut actual[..1024],
                            simd,
                        )
                        .unwrap();
                    bits_equal(&actual[..1024], &expected[..1024]);
                }
            }
        });
    }

    #[test]
    fn decode_masks_key_tiles_long_tails_and_extreme_sinks() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            for simd in backends() {
                for (prefix, generated) in [(127, 3), (128, 129), (129, 130), (1025, 3)] {
                    let (k, v) = operands(prefix, generated);
                    let mut cache =
                        TemporalCache::new(prefix, prefix + generated, 16, 8, 64).unwrap();
                    fill(&mut cache, &k, &v, prefix);
                    let gk = compact_v(&k[prefix * 1024..]);
                    let cv = compact_v(&v);
                    let q: Vec<_> = (0..1024)
                        .map(|i| if i % 2 == 0 { 13.25 } else { -13.25 })
                        .collect();
                    let sinks: Vec<_> = (0..16).map(|h| [-1000., -2., 0., 1000.][h % 4]).collect();
                    // Low-level operator comparison includes causal/image boundaries;
                    // the cache API separately restricts real continuation to one row.
                    for offset in [0, 1, prefix - 2, prefix - 1, prefix, prefix + generated - 1] {
                        let mut expected = [0.; 1024];
                        let mut actual = [0.; 1024];
                        kernels::attention_compact_with_simd(
                            &q,
                            &k[..prefix * 1024],
                            &gk,
                            &cv,
                            1,
                            prefix,
                            prefix + generated,
                            16,
                            8,
                            64,
                            offset,
                            1,
                            prefix - 1,
                            &sinks,
                            &mut expected,
                            simd,
                        );
                        kernels::attention_temporal_candidate_with_simd(
                            &q,
                            &cache.temporal,
                            &cache.spatial,
                            &cache.generated,
                            &cache.values,
                            1,
                            prefix,
                            prefix + generated,
                            16,
                            8,
                            64,
                            offset,
                            1,
                            prefix - 1,
                            &sinks,
                            &mut actual,
                            simd,
                        );
                        bits_equal(&actual, &expected);
                        assert!(actual.iter().all(|x| x.is_finite()));
                    }
                }
            }
        });
    }

    #[test]
    fn reserved_owned_payload_and_append_capacity_stability() {
        let (prefix, generated) = (144, 17);
        let mut cache = TemporalCache::new(prefix, prefix + generated, 16, 8, 64).unwrap();
        let new_payload = (1280 * prefix + 1024 * generated) * 4;
        let old_payload = (1536 * prefix + 1024 * generated) * 4;
        let mut compact_buffers: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for (buffer, elements) in compact_buffers.iter_mut().zip([
            prefix * 1024,
            generated * 512,
            (prefix + generated) * 512,
        ]) {
            buffer.try_reserve_exact(elements).unwrap();
        }
        let measured_compact = compact_buffers
            .iter()
            .map(|v| v.capacity() * 4)
            .sum::<usize>();
        assert_eq!(measured_compact, old_payload);
        assert_eq!(cache.owned_capacity_bytes(), new_payload);
        let pointers = [
            cache.temporal.as_ptr(),
            cache.spatial.as_ptr(),
            cache.generated.as_ptr(),
            cache.values.as_ptr(),
        ];
        let (k, v) = operands(prefix, generated);
        fill(&mut cache, &k, &v, prefix);
        assert_eq!(
            pointers,
            [
                cache.temporal.as_ptr(),
                cache.spatial.as_ptr(),
                cache.generated.as_ptr(),
                cache.values.as_ptr()
            ]
        );
        assert_eq!(cache.owned_capacity_bytes(), new_payload);
        assert_eq!(old_payload - new_payload, prefix * 256 * 4);
        let b4_old = 22usize * 4 * (1536 * 17184 + 1024 * 4 * 4096);
        let b4_new = 22usize * 4 * (1280 * 17184 + 1024 * 4 * 4096);
        assert_eq!(
            (b4_old, b4_new, b4_old - b4_new),
            (3_799_121_920, 3_412_000_768, 387_121_152)
        );
        let report = serde_json::json!({"schema":1,"scope":"owned Vec capacities only; excludes allocator metadata, transient workspace, weights and RSS",
            "prefix":prefix,"generated_reserved":generated,"measured_candidate_buffer_capacity_bytes":cache.owned_capacity_bytes(),
            "candidate_buffer_lengths": [cache.temporal.len(),cache.spatial.len(),cache.generated.len(),cache.values.len()],
            "candidate_buffer_capacities": [cache.temporal.capacity(),cache.spatial.capacity(),cache.generated.capacity(),cache.values.capacity()],
            "compact_analytic_payload_bytes":old_payload,"candidate_analytic_payload_bytes":new_payload,
            "measured_compact_buffer_capacity_bytes":measured_compact,
            "compact_buffer_capacities":compact_buffers.iter().map(|v|v.capacity()).collect::<Vec<_>>(),
            "candidate_struct_bytes":std::mem::size_of::<TemporalCache>(),"buffer_addresses_unchanged_through_full_append":true,
            "frozen_b4_22layer_analytic_compact_bytes":b4_old,"frozen_b4_22layer_analytic_candidate_bytes":b4_new,"frozen_b4_analytic_saving_bytes":b4_old-b4_new,
            "supported_test_backends":backends().iter().map(|x|format!("{x:?}")).collect::<Vec<_>>()});
        if let Ok(path) = std::env::var("FOCR_TEMPORAL_MEMORY_REPORT") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap();
            writeln!(file, "{}", serde_json::to_string_pretty(&report).unwrap()).unwrap();
        }
    }
}
