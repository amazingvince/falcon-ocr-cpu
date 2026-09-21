//! Experimental BF16 single-request runner. Output metadata explicitly records
//! precision and reference provenance; no GPU qualification is implied.

use crate::{
    bf16_kernels::Backend as Bf16Backend,
    bf16_model::{Bf16Model, Session},
    config::{Backend, CacheLayout, GenerationOptions, MODEL_REVISION, RunnerConfig, WeightLayout},
    model::{image_range, positions},
    preprocess::{PreparedImage, prepare_file_timed, prepare_rgb},
    runner::{FinishReason, OcrResult, Timings},
    tokenizer::OcrTokenizer,
    trace::{NoTrace, Trace},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bf16Result {
    #[serde(flatten)]
    pub result: OcrResult,
    pub experimental: bool,
    pub gpu_parity_qualified: bool,
    pub model_revision: String,
    pub weights_sha256: String,
    pub weight_tensor_bytes: usize,
    pub fixture_sha256: Option<String>,
    /// Entire canonical teacher sequence, even when a shorter trace is requested.
    pub teacher_tokens: Vec<u32>,
    pub implementation_sha256: BTreeMap<String, String>,
}

pub struct Bf16Runner {
    model: Arc<Bf16Model>,
    tokenizer: OcrTokenizer,
    pool: rayon::ThreadPool,
    config: RunnerConfig,
    backend: Bf16Backend,
}

impl Bf16Runner {
    /// Validate experimental limitations before loading the checkpoint.
    pub fn validate_config(config: &RunnerConfig) -> Result<Bf16Backend> {
        ensure!(
            config.batch_size == 1,
            "experimental BF16 supports batch_size=1 only"
        );
        ensure!(
            config.cache_layout == CacheLayout::Expanded,
            "experimental BF16 currently requires expanded KV storage"
        );
        ensure!(
            config.weight_layout == WeightLayout::Unpacked,
            "experimental BF16 requires unpacked weights"
        );
        let backend = match config.backend {
            Backend::Auto => Bf16Backend::Auto,
            Backend::Scalar => Bf16Backend::Scalar,
            Backend::Avx512 => Bf16Backend::Avx512Bf16,
            Backend::Avx2 => anyhow::bail!(
                "experimental BF16 supports auto, scalar, or avx512 (AVX-512F + AVX-512BF16); AVX2 is an FP32 backend"
            ),
        };
        backend.validate().map_err(anyhow::Error::msg)?;
        Ok(backend.resolved())
    }

    pub fn new(
        model: Arc<Bf16Model>,
        model_dir: impl AsRef<Path>,
        config: RunnerConfig,
    ) -> Result<Self> {
        let backend = Self::validate_config(&config)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()?;
        let tokenizer = OcrTokenizer::load(model_dir.as_ref())?;
        Ok(Self {
            model,
            tokenizer,
            pool,
            config,
            backend,
        })
    }

    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    pub fn recognize_file(
        &self,
        path: impl AsRef<Path>,
        options: &GenerationOptions,
    ) -> Result<Bf16Result> {
        options.validate()?;
        let start = Instant::now();
        let (prepared, decode_ms) =
            prepare_file_timed(path.as_ref(), options.min_dimension, options.max_dimension)?;
        validate_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000. - decode_ms;
        let mut output = self.run_scoped(prepared, tokens, options, prep_ms, &mut NoTrace, &[])?;
        output.result.timings.image_decode_ms = decode_ms;
        output.result.timings.total_ms = start.elapsed().as_secs_f64() * 1000.;
        output.result.timings.time_to_first_token_ms += decode_ms;
        Ok(output)
    }

    /// Multiple files execute sequentially; BF16 joint batching is unsupported.
    pub fn recognize_files<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
    ) -> Result<Vec<Bf16Result>> {
        options.validate()?;
        paths
            .iter()
            .map(|path| self.recognize_file(path, options))
            .collect()
    }

    pub fn recognize(&self, image: &RgbImage, options: &GenerationOptions) -> Result<Bf16Result> {
        self.recognize_with_trace(image, options, &mut NoTrace)
    }

    pub fn recognize_with_trace(
        &self,
        image: &RgbImage,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<Bf16Result> {
        options.validate()?;
        let start = Instant::now();
        let prepared = prepare_rgb(image, options.min_dimension, options.max_dimension)?;
        validate_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut output = self.run_scoped(prepared, tokens, options, prep_ms, trace, &[])?;
        output.result.timings.total_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_scoped(
        &self,
        prepared: PreparedImage,
        tokens: Vec<u32>,
        options: &GenerationOptions,
        prep_ms: f64,
        trace: &mut dyn Trace,
        teacher_tokens: &[u32],
    ) -> Result<Bf16Result> {
        self.pool
            .install(|| self.run(prepared, tokens, options, prep_ms, trace, teacher_tokens))
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        prepared: PreparedImage,
        tokens: Vec<u32>,
        options: &GenerationOptions,
        prep_ms: f64,
        trace: &mut dyn Trace,
        teacher_tokens: &[u32],
    ) -> Result<Bf16Result> {
        let start = Instant::now();
        let c = self.model.config();
        options.validate()?;
        options.check_budget(tokens.len(), c.max_seq_len)?;
        let (image_start, image_end) = image_range(&tokens, c)?;
        let (pos_t, pos_hw) = positions(&tokens, &prepared.positions_hw, c)?;
        let mut session = Session::new(
            c,
            tokens.len() + options.max_new_tokens,
            image_start,
            image_end,
            self.backend,
        );
        let mut hidden = Vec::new();
        self.model
            .embed(&tokens, Some(&prepared.patches), &mut hidden, self.backend)?;
        let logits =
            self.model
                .forward(&mut hidden, &pos_t, &pos_hw, &mut session, trace, "prefill")?;
        let mut next_token = teacher_tokens
            .first()
            .copied()
            .map(Ok)
            .unwrap_or_else(|| argmax(logits))?;
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.;
        let decode_start = Instant::now();
        let stops = self.tokenizer.stop_ids();
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut reason = FinishReason::Length;
        trace.decode_start();
        for step in 0..options.max_new_tokens {
            let token = next_token;
            generated.push(token);
            if stops.contains(&token) && teacher_tokens.is_empty() {
                reason = FinishReason::Eos;
                break;
            }
            if step + 1 == options.max_new_tokens {
                break;
            }
            self.model
                .embed(&[token], None, &mut hidden, self.backend)?;
            let phase = if trace.enabled() {
                format!("decode.{step}")
            } else {
                String::new()
            };
            let logits = self.model.forward(
                &mut hidden,
                &[session.next_position],
                &[[f32::NAN; 2]],
                &mut session,
                trace,
                &phase,
            )?;
            next_token = teacher_tokens
                .get(step + 1)
                .copied()
                .map(Ok)
                .unwrap_or_else(|| argmax(logits))?;
        }
        trace.decode_end();
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.;
        let text = self.tokenizer.decode(&generated)?;
        Ok(Bf16Result {
            result: OcrResult {
                text,
                output_tokens: generated.len(),
                token_ids: generated,
                finish_reason: reason,
                width: prepared.width,
                height: prepared.height,
                input_tokens: tokens.len(),
                precision: "bf16".into(),
                backend: format!("rust-bf16/{:?}", self.backend).to_lowercase(),
                cache_layout: CacheLayout::Expanded,
                weight_layout: WeightLayout::Unpacked,
                packed_weight_bytes: 0,
                weight_packing_ms: 0.,
                teacher_forced: !teacher_tokens.is_empty(),
                timings: Timings {
                    image_decode_ms: 0.,
                    preprocessing_ms: prep_ms,
                    image_projection_ms: None,
                    transformer_prefill_ms: None,
                    prefill_ms,
                    decode_ms,
                    total_ms: prep_ms + start.elapsed().as_secs_f64() * 1000.,
                    time_to_first_token_ms: prep_ms + prefill_ms,
                },
            },
            experimental: true,
            gpu_parity_qualified: false,
            model_revision: MODEL_REVISION.into(),
            weights_sha256: self.model.weights_sha256().into(),
            weight_tensor_bytes: self.model.weight_tensor_bytes(),
            fixture_sha256: None,
            teacher_tokens: teacher_tokens.to_vec(),
            implementation_sha256: [
                ("bf16_model.rs", include_bytes!("bf16_model.rs").as_slice()),
                (
                    "bf16_attention.rs",
                    include_bytes!("bf16_attention.rs").as_slice(),
                ),
                (
                    "bf16_kernels.rs",
                    include_bytes!("bf16_kernels.rs").as_slice(),
                ),
                ("bf16_ops.rs", include_bytes!("bf16_ops.rs").as_slice()),
                (
                    "bf16_runner.rs",
                    include_bytes!("bf16_runner.rs").as_slice(),
                ),
                ("model.rs", include_bytes!("model.rs").as_slice()),
                ("kernels.rs", include_bytes!("kernels.rs").as_slice()),
                ("config.rs", include_bytes!("config.rs").as_slice()),
                ("Cargo.lock", include_bytes!("../Cargo.lock").as_slice()),
            ]
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), format!("{:x}", Sha256::digest(bytes))))
            .collect(),
        })
    }

    /// Canonical same-prefix execution; BF16 tensor values are losslessly
    /// promoted to FP32 in TensorTrace for the existing safetensors writer.
    pub fn trace_reference(
        &self,
        fixture_path: impl AsRef<Path>,
        max_new_tokens: usize,
        trace: &mut dyn Trace,
    ) -> Result<Bf16Result> {
        use safetensors::{Dtype, SafeTensors};
        let bytes = std::fs::read(fixture_path)?;
        let fixture_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let tensors = SafeTensors::deserialize(&bytes)?;
        let read_ids = |name: &str| -> Result<Vec<u32>> {
            let tensor = tensors.tensor(name)?;
            match tensor.dtype() {
                Dtype::I64 => tensor
                    .data()
                    .chunks_exact(8)
                    .map(|v| {
                        u32::try_from(i64::from_le_bytes(v.try_into().unwrap()))
                            .context("invalid token/position id")
                    })
                    .collect(),
                Dtype::U32 => Ok(tensor
                    .data()
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect()),
                dtype => anyhow::bail!("unsupported {name} dtype {dtype:?}"),
            }
        };
        let tokens_tensor = tensors.tensor("tokens")?;
        ensure!(tokens_tensor.shape().len() <= 2, "invalid token tensor");
        let tokens = read_ids("tokens")?;
        let patches = tensors.tensor("patches")?;
        ensure!(
            patches.dtype() == Dtype::F32
                && patches.shape().len() == 2
                && patches.shape()[1] == self.model.config().patch_dim(),
            "expected F32 patches [patches,768]"
        );
        let patches = patches
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let reference_spatial = tensors.tensor("pos_hw")?;
        ensure!(
            reference_spatial.dtype() == Dtype::F32
                && reference_spatial.shape() == [tokens.len(), 2],
            "expected F32 spatial positions [S,2]"
        );
        let all_positions: Vec<[f32; 2]> = reference_spatial
            .data()
            .chunks_exact(8)
            .map(|v| {
                [
                    f32::from_le_bytes(v[..4].try_into().unwrap()),
                    f32::from_le_bytes(v[4..].try_into().unwrap()),
                ]
            })
            .collect();
        let positions_hw = tokens
            .iter()
            .zip(all_positions)
            .filter_map(|(&token, pos)| (token == self.model.config().img_id).then_some(pos))
            .collect();
        let teacher_tokens = if tensors.names().contains(&"teacher_tokens") {
            read_ids("teacher_tokens")?
        } else {
            Vec::new()
        };
        ensure!(
            teacher_tokens.is_empty() || max_new_tokens <= teacher_tokens.len(),
            "same-prefix trace requested {max_new_tokens} outputs but fixture has only {} teacher tokens",
            teacher_tokens.len()
        );
        ensure!(
            teacher_tokens
                .iter()
                .all(|&id| (id as usize) < self.model.config().vocab_size),
            "teacher token outside vocabulary"
        );
        let prepared = PreparedImage {
            width: 0,
            height: 0,
            patches,
            positions_hw,
        };
        let (computed_temporal, computed_spatial) =
            positions(&tokens, &prepared.positions_hw, self.model.config())?;
        ensure!(
            computed_temporal
                .iter()
                .map(|&v| v as u32)
                .eq(read_ids("pos_t")?),
            "temporal position parity failure"
        );
        for (actual, raw) in computed_spatial
            .iter()
            .flatten()
            .zip(reference_spatial.data().chunks_exact(4))
        {
            let expected = f32::from_le_bytes(raw.try_into().unwrap());
            ensure!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
                "canonical spatial position mismatch"
            );
        }
        let options = GenerationOptions {
            max_new_tokens,
            ..Default::default()
        };
        let mut output = self.run_scoped(prepared, tokens, &options, 0., trace, &teacher_tokens)?;
        output.fixture_sha256 = Some(fixture_sha256);
        Ok(output)
    }
}

fn validate_bounds(prepared: &PreparedImage, options: &GenerationOptions) -> Result<()> {
    ensure!(
        prepared.width <= options.max_dimension as usize
            && prepared.height <= options.max_dimension as usize,
        "minimum-area alignment conflicts with requested bounds: prepared {}x{} exceeds max_dimension={}; explicitly increase max_dimension",
        prepared.width,
        prepared.height,
        options.max_dimension
    );
    Ok(())
}

fn argmax(logits: &[bf16]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "nonfinite or empty BF16 logits"
    );
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_modes_fail_before_loading_weights() {
        let base = RunnerConfig {
            backend: Backend::Scalar,
            ..Default::default()
        };
        assert_eq!(
            Bf16Runner::validate_config(&base).unwrap(),
            Bf16Backend::Scalar
        );
        for invalid in [
            RunnerConfig {
                batch_size: 2,
                ..base.clone()
            },
            RunnerConfig {
                cache_layout: CacheLayout::Compact,
                ..base.clone()
            },
            RunnerConfig {
                weight_layout: WeightLayout::PhasePacked,
                ..base.clone()
            },
            RunnerConfig {
                backend: Backend::Avx2,
                ..base.clone()
            },
        ] {
            assert!(Bf16Runner::validate_config(&invalid).is_err());
        }
        let avx512 = RunnerConfig {
            backend: Backend::Avx512,
            ..base
        };
        assert_eq!(
            Bf16Runner::validate_config(&avx512).is_ok(),
            crate::bf16_kernels::avx512_bf16_available()
        );
    }
    #[test]
    fn bf16_argmax_uses_first_tie_and_rejects_nonfinite() {
        assert_eq!(
            argmax(&[bf16::from_f32(1.), bf16::from_f32(2.), bf16::from_f32(2.)]).unwrap(),
            1
        );
        assert!(argmax(&[bf16::NAN]).is_err());
        assert!(argmax(&[]).is_err());
    }
}
