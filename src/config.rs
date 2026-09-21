use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const MODEL_REVISION: &str = "fe757d59ecd79d4d68760162306a70a015761ad9";
pub const WEIGHTS_SHA256: &str = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16";
pub const CONFIG_SHA256: &str = "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    Fp32,
    /// Experimental single-request BF16 graph; GPU qualification is incomplete.
    Bf16,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CacheLayout {
    #[default]
    Expanded,
    Compact,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum WeightLayout {
    #[default]
    Unpacked,
    /// Experimental shared phase packing for AVX2 batch decode with 2–8 live rows.
    PhasePacked,
}
impl Backend {
    pub(crate) fn simd(self) -> crate::kernels::Simd {
        match self {
            Self::Auto => crate::kernels::Simd::Auto,
            Self::Scalar => crate::kernels::Simd::Scalar,
            Self::Avx2 => crate::kernels::Simd::Avx2,
            Self::Avx512 => crate::kernels::Simd::Avx512,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub ffn_dim: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub channel_size: usize,
    pub spatial_patch_size: usize,
    pub temporal_patch_size: usize,
    pub eos_id: u32,
    pub img_id: u32,
    pub image_cls_token_id: u32,
    pub image_reg_1_token_id: u32,
    pub image_reg_2_token_id: u32,
    pub image_reg_3_token_id: u32,
    pub image_reg_4_token_id: u32,
    pub img_end_id: u32,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dim == 768
                && self.n_layers == 22
                && self.n_heads == 16
                && self.head_dim == 64
                && self.n_kv_heads == 8
                && self.ffn_dim == 2304
                && self.vocab_size == 65536,
            "unsupported architecture: expected Falcon-OCR v1.5 dimensions"
        );
        ensure!(
            self.channel_size == 3
                && self.spatial_patch_size == 16
                && self.temporal_patch_size == 1,
            "unsupported patch configuration"
        );
        ensure!(
            self.max_seq_len == 16384 && self.rope_theta == 10000.0 && self.norm_eps == 1e-5,
            "unsupported positional/normalization configuration"
        );
        ensure!(
            [
                self.eos_id,
                self.img_id,
                self.image_cls_token_id,
                self.image_reg_1_token_id,
                self.image_reg_2_token_id,
                self.image_reg_3_token_id,
                self.image_reg_4_token_id,
                self.img_end_id
            ] == [11, 227, 244, 245, 246, 247, 248, 230],
            "unsupported special token configuration"
        );
        Ok(())
    }

    pub fn query_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    pub fn patch_dim(&self) -> usize {
        self.channel_size * self.spatial_patch_size.pow(2)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub min_dimension: u32,
    pub max_dimension: u32,
    pub max_new_tokens: usize,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            min_dimension: 64,
            max_dimension: 1536,
            max_new_tokens: 8192,
        }
    }
}
impl GenerationOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.min_dimension > 0 && self.min_dimension <= self.max_dimension,
            "require 0 < min_dimension <= max_dimension"
        );
        ensure!(
            self.max_dimension >= 16 && self.max_dimension.is_multiple_of(16),
            "max_dimension must be a positive multiple of 16"
        );
        ensure!(self.max_new_tokens > 0, "max_new_tokens must be positive");
        Ok(())
    }
    pub fn check_budget(&self, input_tokens: usize, context: usize) -> Result<()> {
        ensure!(
            input_tokens
                .checked_add(self.max_new_tokens)
                .is_some_and(|n| n <= context),
            "context budget conflict: {input_tokens} input tokens + {} requested output tokens exceeds {context}; explicitly lower max_dimension or max_new_tokens",
            self.max_new_tokens
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Maximum threads in the owned compute pool. Zero selects logical CPUs.
    pub threads: usize,
    /// Maximum active requests. Prefill is independent; decoding shares projections.
    pub batch_size: usize,
    /// Controls GEMV and attention vectors. Large GEMMs dispatch independently.
    #[serde(default)]
    pub backend: Backend,
    /// Compact keeps sixteen prefix K heads and eight generated K/all V heads.
    #[serde(default)]
    pub cache_layout: CacheLayout,
    /// Extra immutable packed weights; single rows and prefill stay unpacked.
    #[serde(default)]
    pub weight_layout: WeightLayout,
}
impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            threads: 16,
            batch_size: 1,
            backend: Backend::Auto,
            cache_layout: CacheLayout::Expanded,
            weight_layout: WeightLayout::Unpacked,
        }
    }
}
impl RunnerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be positive");
        self.backend.simd().validate().map_err(anyhow::Error::msg)?;
        if self.weight_layout == WeightLayout::PhasePacked {
            ensure!(
                self.backend.simd().resolved() == crate::kernels::Simd::Avx2,
                "phase-packed weights require AVX2/FMA: use backend avx2 or auto on a compatible CPU"
            );
            ensure!(
                self.batch_size <= 8,
                "phase-packed weights support configured batch sizes 1–8; packing is used only with 2–8 live decode rows"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_budget_is_exact_and_checked() {
        let o = GenerationOptions::default();
        assert!(o.check_budget(8192, 16384).is_ok());
        assert!(o.check_budget(8193, 16384).is_err());
        assert!(o.check_budget(usize::MAX, 16384).is_err());
    }
    #[test]
    fn phase_packing_is_opt_in_and_rejects_incompatible_dispatch() {
        let legacy: RunnerConfig = serde_json::from_str(r#"{"threads":2,"batch_size":4}"#).unwrap();
        assert_eq!(legacy.weight_layout, WeightLayout::Unpacked);
        for backend in [Backend::Scalar, Backend::Avx512] {
            assert!(
                RunnerConfig {
                    backend,
                    weight_layout: WeightLayout::PhasePacked,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        if crate::kernels::Simd::Avx2.validate().is_ok() {
            for backend in [Backend::Auto, Backend::Avx2] {
                for batch_size in [1, 2, 4, 8] {
                    assert!(
                        RunnerConfig {
                            backend,
                            batch_size,
                            weight_layout: WeightLayout::PhasePacked,
                            ..Default::default()
                        }
                        .validate()
                        .is_ok()
                    );
                }
            }
            assert!(
                RunnerConfig {
                    batch_size: 9,
                    weight_layout: WeightLayout::PhasePacked,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }
}
