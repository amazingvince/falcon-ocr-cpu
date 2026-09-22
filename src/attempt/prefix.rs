//! Immutable split-prefix storage for the opt-in attempt. The reference prefill
//! finishes in FP32 arithmetic with the selected weights; sealing then
//! compresses its caches for subsequent decode.
//! This is DEFERRED cache compression, not a quantized-prefill-attention claim.
use super::PrefixMode;
use crate::{
    config::ModelConfig,
    kernels::{self, Simd},
};
use anyhow::{Result, ensure};
use half::bf16;
use rayon::prelude::*;

#[derive(Debug)]
enum Packed {
    F32(Vec<f32>),
    Bf16(Vec<bf16>),
    Q8 { codes: Vec<i8>, scales: Vec<f32> },
}
impl Packed {
    fn new(mode: PrefixMode, len: usize) -> Result<Self> {
        match mode {
            PrefixMode::SplitF32 => {
                let mut data = Vec::new();
                data.try_reserve_exact(len)?;
                Ok(Self::F32(data))
            }
            PrefixMode::SplitBf16 => {
                let mut data = Vec::new();
                data.try_reserve_exact(len)?;
                Ok(Self::Bf16(data))
            }
            PrefixMode::SplitQ8 => {
                let mut codes = Vec::new();
                codes.try_reserve_exact(len)?;
                let mut scales = Vec::new();
                scales.try_reserve_exact(len / 32)?;
                Ok(Self::Q8 { codes, scales })
            }
            PrefixMode::Reference => anyhow::bail!("reference cache never uses split storage"),
        }
    }
    fn append32(&mut self, values: &[f32]) -> Result<()> {
        ensure!(
            values.len() == 32 && values.iter().all(|v| v.is_finite()),
            "nonfinite or incomplete KV group"
        );
        match self {
            Self::F32(dst) => dst.extend_from_slice(values),
            Self::Bf16(dst) => {
                for &v in values {
                    let q = bf16::from_f32(v);
                    ensure!(q.is_finite(), "BF16 KV conversion overflow");
                    dst.push(q);
                }
            }
            Self::Q8 { codes, scales } => {
                let max = values.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
                let s = if max == 0.0 {
                    0.0
                } else {
                    ((max as f64 / 127.0) as f32).max(f32::from_bits(1))
                };
                scales.push(s);
                for &v in values {
                    let q = if s == 0.0 {
                        0
                    } else {
                        (v as f64 / s as f64).round_ties_even().clamp(-127.0, 127.0) as i8
                    };
                    ensure!((q as f32 * s).is_finite(), "Q8 KV conversion overflow");
                    codes.push(q);
                }
            }
        }
        Ok(())
    }
    fn bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.capacity() * 4,
            Self::Bf16(v) => v.capacity() * 2,
            Self::Q8 { codes, scales } => codes.capacity() + scales.capacity() * 4,
        }
    }
    #[inline]
    fn read32(&self, start: usize, output: &mut [f32], avx2: bool) {
        assert_eq!(output.len(), 32);
        debug_assert_eq!(start % 32, 0);
        match self {
            Self::F32(v) => output.copy_from_slice(&v[start..start + 32]),
            Self::Bf16(v) => {
                let source = &v[start..start + 32];
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    unsafe {
                        decode_bf16_32(source, output);
                    }
                    return;
                }
                let _ = avx2;
                for (a, b) in output.iter_mut().zip(source) {
                    *a = b.to_f32();
                }
            }
            Self::Q8 { codes, scales } => {
                let source = &codes[start..start + 32];
                let scale = scales[start / 32];
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    unsafe {
                        decode_q8_32(source, scale, output);
                    }
                    return;
                }
                let _ = avx2;
                for (a, &b) in output.iter_mut().zip(source) {
                    *a = b as f32 * scale;
                }
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_bf16_32(source: &[bf16], output: &mut [f32]) {
    use std::arch::x86_64::*;
    // Caller verifies AVX2 and 32-element slice bounds. Each load is eight u16s.
    unsafe {
        for i in (0..32).step_by(8) {
            let packed = _mm_loadu_si128(source.as_ptr().add(i).cast());
            let bits = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(packed));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_castsi256_ps(bits));
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_q8_32(source: &[i8], scale: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;
    // Caller verifies AVX2 and 32-element slice bounds. Each load is eight i8s.
    unsafe {
        for i in (0..32).step_by(8) {
            let packed = _mm_loadl_epi64(source.as_ptr().add(i).cast());
            let floats = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(i),
                _mm256_mul_ps(floats, _mm256_set1_ps(scale)),
            );
        }
    }
}

pub(crate) struct SplitPrefix {
    temporal: Packed,
    spatial: Packed,
    values: Packed,
    prefix_len: usize,
    heads: usize,
    kv_heads: usize,
    pub generated_k: Vec<f32>,
    pub generated_v: Vec<f32>,
}
impl SplitPrefix {
    pub fn from_compact(
        k: &[f32],
        v: &[f32],
        prefix_len: usize,
        capacity: usize,
        c: &ModelConfig,
        mode: PrefixMode,
    ) -> Result<Self> {
        ensure!(
            mode != PrefixMode::Reference && c.head_dim == 64 && c.n_heads == 2 * c.n_kv_heads,
            "split cache requires Falcon 64-wide paired GQA heads"
        );
        ensure!(
            prefix_len > 0 && capacity >= prefix_len,
            "invalid split-prefix capacity"
        );
        ensure!(
            prefix_len.checked_mul(c.query_dim()) == Some(k.len()),
            "split prefix key shape"
        );
        ensure!(
            prefix_len.checked_mul(c.kv_dim()) == Some(v.len()),
            "split prefix value shape"
        );
        let mut temporal = Packed::new(mode, prefix_len * c.n_kv_heads * 32)?;
        let mut spatial = Packed::new(mode, prefix_len * c.n_heads * 32)?;
        let mut values = Packed::new(mode, prefix_len * c.kv_dim())?;
        for t in 0..prefix_len {
            for g in 0..c.n_kv_heads {
                let a = (t * c.n_heads + 2 * g) * 64;
                ensure!(
                    k[a..a + 32]
                        .iter()
                        .zip(&k[a + 64..a + 96])
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "temporal keys differ inside GQA pair; refusing illegal sharing"
                );
                temporal.append32(&k[a..a + 32])?;
            }
            for h in 0..c.n_heads {
                let a = (t * c.n_heads + h) * 64 + 32;
                spatial.append32(&k[a..a + 32])?;
            }
            for g in 0..c.n_kv_heads {
                let a = (t * c.n_kv_heads + g) * 64;
                values.append32(&v[a..a + 32])?;
                values.append32(&v[a + 32..a + 64])?;
            }
        }
        let count = (capacity - prefix_len)
            .checked_mul(c.kv_dim())
            .ok_or_else(|| anyhow::anyhow!("tail capacity overflow"))?;
        let mut generated_k = Vec::new();
        let mut generated_v = Vec::new();
        generated_k.try_reserve_exact(count)?;
        generated_v.try_reserve_exact(count)?;
        Ok(Self {
            temporal,
            spatial,
            values,
            prefix_len,
            heads: c.n_heads,
            kv_heads: c.n_kv_heads,
            generated_k,
            generated_v,
        })
    }
    pub fn bytes(&self) -> usize {
        self.temporal.bytes()
            + self.spatial.bytes()
            + self.values.bytes()
            + 4 * (self.generated_k.capacity() + self.generated_v.capacity())
    }

    pub fn attention_decode(
        &self,
        q: &[f32],
        total_len: usize,
        sinks: &[f32],
        output: &mut [f32],
        simd: Simd,
    ) {
        assert_eq!(q.len(), self.heads * 64);
        assert_eq!(output.len(), q.len());
        assert_eq!(sinks.len(), self.heads);
        assert_eq!(self.generated_k.len(), self.generated_v.len());
        assert_eq!(
            total_len,
            self.prefix_len + self.generated_k.len() / (self.kv_heads * 64)
        );
        let selected = simd.resolved();
        let dot = kernels::dot_kernel(selected);
        let axpy = kernels::axpy_kernel(selected);
        #[cfg(target_arch = "x86_64")]
        let avx2 = selected != Simd::Scalar && std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        output.par_chunks_mut(64).enumerate().for_each(|(h, out)| {
            let group = h / (self.heads / self.kv_heads);
            let query = &q[h * 64..(h + 1) * 64];
            let mut max = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            let mut logits = [0.0_f32; 128];
            let mut key = [0.0_f32; 64];
            let mut value = [0.0_f32; 64];
            out.fill(0.0);
            for start in (0..total_len).step_by(128) {
                let len = (total_len - start).min(128);
                let mut block_max = f32::NEG_INFINITY;
                for (j, logit) in logits[..len].iter_mut().enumerate() {
                    let t = start + j;
                    if t < self.prefix_len {
                        self.temporal.read32(
                            (t * self.kv_heads + group) * 32,
                            &mut key[..32],
                            avx2,
                        );
                        self.spatial
                            .read32((t * self.heads + h) * 32, &mut key[32..], avx2);
                    } else {
                        let a = ((t - self.prefix_len) * self.kv_heads + group) * 64;
                        key.copy_from_slice(&self.generated_k[a..a + 64]);
                    }
                    *logit = dot(query, &key) * 0.125;
                    block_max = block_max.max(*logit);
                }
                let new_max = max.max(block_max);
                let scale = if max == f32::NEG_INFINITY {
                    0.0
                } else {
                    (max - new_max).exp()
                };
                for x in out.iter_mut() {
                    *x *= scale;
                }
                denominator *= scale;
                for (j, logit) in logits[..len].iter().enumerate() {
                    let probability = (*logit - new_max).exp();
                    denominator += probability;
                    let t = start + j;
                    if t < self.prefix_len {
                        let a = (t * self.kv_heads + group) * 64;
                        self.values.read32(a, &mut value[..32], avx2);
                        self.values.read32(a + 32, &mut value[32..], avx2);
                    } else {
                        let a = ((t - self.prefix_len) * self.kv_heads + group) * 64;
                        value.copy_from_slice(&self.generated_v[a..a + 64]);
                    }
                    axpy(probability, &value, out);
                }
                max = new_max;
            }
            // Same tile order and sink policy as the reference. The sink occurs
            // once after the combined image+tail scan, not once per segment.
            let lse = max + denominator.ln();
            let sink = 1.0 / (1.0 + (sinks[h] - lse).exp());
            for y in out {
                *y = (*y / denominator) * sink;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_f32_matches_compact_and_low_precision_is_finite() {
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        for p in [1, 3, 127, 128, 129] {
            let mut k = vec![0.0; p * c.query_dim()];
            for t in 0..p {
                for h in 0..c.n_heads {
                    for d in 0..64 {
                        let identity = if d < 32 { h / 2 } else { h };
                        k[(t * c.n_heads + h) * 64 + d] =
                            ((t * 31 + identity * 17 + d * 7) % 101) as f32 / 151.0 - 0.3;
                    }
                }
            }
            let v: Vec<_> = (0..p * c.kv_dim())
                .map(|i| (i % 73) as f32 / 97.0 - 0.2)
                .collect();
            let q: Vec<_> = (0..c.query_dim())
                .map(|i| (i % 61) as f32 / 81.0 - 0.2)
                .collect();
            let tail = vec![0.02; c.kv_dim()];
            let sinks = vec![0.5; c.n_heads];
            let mut allv = v.clone();
            allv.extend_from_slice(&tail);
            for backend in [Simd::Scalar, Simd::Auto] {
                let mut expected = vec![0.0; c.query_dim()];
                kernels::attention_compact_with_simd(
                    &q,
                    &k,
                    &tail,
                    &allv,
                    1,
                    p,
                    p + 1,
                    c.n_heads,
                    c.n_kv_heads,
                    64,
                    p,
                    0,
                    p,
                    &sinks,
                    &mut expected,
                    backend,
                );
                for mode in [
                    PrefixMode::SplitF32,
                    PrefixMode::SplitBf16,
                    PrefixMode::SplitQ8,
                ] {
                    let mut cache = SplitPrefix::from_compact(&k, &v, p, p + 2, &c, mode).unwrap();
                    cache.generated_k.extend_from_slice(&tail);
                    cache.generated_v.extend_from_slice(&tail);
                    let mut output = vec![0.0; c.query_dim()];
                    cache.attention_decode(&q, p + 1, &sinks, &mut output, backend);
                    assert!(output.iter().all(|x| x.is_finite()));
                    if mode == PrefixMode::SplitF32 {
                        assert!(
                            output
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "split FP32 changed arithmetic"
                        );
                    } else {
                        assert!(
                            output
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| (a - b).abs() < 0.02)
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn refuses_nonidentical_temporal_keys() {
        let c: ModelConfig =
            serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        let mut k = vec![0.0; c.query_dim()];
        k[64] = 1.0;
        assert!(
            SplitPrefix::from_compact(&k, &vec![0.0; c.kv_dim()], 1, 2, &c, PrefixMode::SplitF32)
                .is_err()
        );
    }
}
