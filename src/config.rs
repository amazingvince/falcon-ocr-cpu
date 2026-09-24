use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

pub const MODEL_REVISION: &str = "fe757d59ecd79d4d68760162306a70a015761ad9";
pub const WEIGHTS_SHA256: &str = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16";
pub const CONFIG_SHA256: &str = "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf";

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Avx512,
    /// aarch64 Advanced SIMD; `auto` selects it on aarch64.
    Neon,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CacheLayout {
    Expanded,
    #[default]
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
/// How greedy decoding evaluates the FP32 vocabulary head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum HeadMode {
    /// Every FP32 logit, then argmax.
    #[default]
    Full,
    /// INT8 screen with a proven error bound, then exact FP32 logits for the
    /// few rows that can still win. Selects the same token as `Full`.
    Screened,
}
/// Vector exp in the prefill attention tiles on x86.
///
/// `Exact` reproduces the platform `expf` bit for bit (`kernels::exp`), so
/// hidden states match the recorded references on this host. `Fast` is the
/// portable polynomial that NEON and the portable path use: at most a few ulp
/// from `Exact` in rare lanes, about 1 s faster per full page, and
/// token-identical to `Exact` on all 67 calibration pages (93,249 tokens).
/// NEON always uses the fast exp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ExpMode {
    #[default]
    Exact,
    Fast,
}

/// Which prefill stages of an 8-bit model use BF16 products on an
/// AVX512-BF16 CPU. Attention alone is fidelity-neutral against the FP32
/// anchor; the projections add about 17% KL, so they are opt-in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PrefillBf16 {
    Off,
    #[default]
    Attention,
    All,
}

/// Experiment and diagnostic knobs. The defaults are the accepted
/// configuration; every field is scheduling or instrumentation only, except
/// `prefill_bf16`, which changes fast-mode prefill rounding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// BF16 prefill stages of 8-bit models on AVX512-BF16 CPUs.
    pub prefill_bf16: PrefillBf16,
    /// Position chunks per KV group in split-cache decode attention (1..=4).
    /// `None` keeps the mode default: one exact scan for FP32 caches, four
    /// chunks (merged partial softmaxes, rounding-level) for quantized ones.
    pub split_chunks: Option<usize>,
    /// Print a wall-clock phase split of every forward pass to stderr.
    pub phases: bool,
    /// Accumulate cycle counters of the prefill attention stages (stderr).
    pub prefill_profile: bool,
}
impl Tuning {
    /// Set one knob from a `key=value` pair (`prefill-bf16=off|attention|all`,
    /// `split-chunks=1..4`, `phases=true|false`, `prefill-profile=true|false`).
    pub fn set(&mut self, pair: &str) -> Result<()> {
        let (key, value) = pair
            .split_once('=')
            .with_context(|| format!("tuning knob `{pair}` is not key=value"))?;
        let flag = |value: &str| -> Result<bool> {
            match value {
                "1" | "true" | "on" => Ok(true),
                "0" | "false" | "off" => Ok(false),
                _ => bail!("tuning knob {key}: expected true or false, got `{value}`"),
            }
        };
        match key {
            "prefill-bf16" => {
                self.prefill_bf16 = match value {
                    "off" | "0" => PrefillBf16::Off,
                    "attention" | "attn" => PrefillBf16::Attention,
                    "all" | "1" => PrefillBf16::All,
                    _ => bail!("tuning knob prefill-bf16: expected off, attention or all, got `{value}`"),
                }
            }
            "split-chunks" => {
                let chunks: usize = value
                    .parse()
                    .with_context(|| format!("tuning knob split-chunks: `{value}`"))?;
                ensure!(
                    (1..=4).contains(&chunks),
                    "tuning knob split-chunks: expected 1..=4, got {chunks}"
                );
                self.split_chunks = Some(chunks);
            }
            "phases" => self.phases = flag(value)?,
            "prefill-profile" => self.prefill_profile = flag(value)?,
            _ => bail!("unknown tuning knob `{key}` (prefill-bf16, split-chunks, phases, prefill-profile)"),
        }
        Ok(())
    }
    /// The default knobs with every `key=value` pair of `pairs` applied.
    pub fn from_pairs<I, S>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut tuning = Self::default();
        for pair in pairs {
            tuning.set(pair.as_ref())?;
        }
        Ok(tuning)
    }
}

impl Backend {
    pub(crate) fn simd(self) -> crate::kernels::Simd {
        match self {
            Self::Auto => crate::kernels::Simd::Auto,
            Self::Scalar => crate::kernels::Simd::Scalar,
            Self::Avx2 => crate::kernels::Simd::Avx2,
            Self::Avx512 => crate::kernels::Simd::Avx512,
            Self::Neon => crate::kernels::Simd::Neon,
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
            self.channel_size == 3 && self.spatial_patch_size == 16 && self.temporal_patch_size == 1,
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

/// The decode spin team: a fixed size, the prefill pool's size, or `Auto`,
/// where the first decode steps time a few team sizes (`tune`) and keep the
/// smallest within 2% of the fastest. Tokens never depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeThreads {
    Auto,
    Fixed(usize),
    /// As many threads as the prefill pool.
    Pool,
}
impl std::str::FromStr for DecodeThreads {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if value.eq_ignore_ascii_case("pool") {
            return Ok(Self::Pool);
        }
        match value.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Self::Fixed(n)),
            _ => Err(format!(
                "expected `auto`, `pool` or a positive thread count, got `{value}`"
            )),
        }
    }
}
impl std::fmt::Display for DecodeThreads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Pool => f.write_str("pool"),
            Self::Fixed(n) => write!(f, "{n}"),
        }
    }
}

/// Speculative decoding: up to `max_draft` tokens (1..=7) drafted from the
/// output so far by an n-gram match of at least `min_match` tokens are
/// verified in one step. Every accepted token is the model's own greedy
/// choice, so outputs never change; drafting pauses itself while it does not
/// pay. Single pages with a split KV cache only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Speculation {
    pub max_draft: usize,
    pub min_match: usize,
}
impl Default for Speculation {
    fn default() -> Self {
        Self {
            max_draft: 4,
            min_match: 2,
        }
    }
}

/// Everything a [`crate::Runner`] decides about; `Default` is the automatic
/// configuration and [`RunnerConfig::reference`] the bit-exact one.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RunnerConfig {
    /// Maximum threads in the owned compute pool. Zero selects logical CPUs.
    pub threads: usize,
    /// Maximum active requests. Prefill is independent; decoding shares projections.
    pub batch_size: usize,
    /// Controls GEMV and attention vectors. Large GEMMs dispatch independently.
    pub backend: Backend,
    /// Compact (the default) keeps sixteen prefix K heads and eight generated
    /// K/all V heads; expanded duplicates every KV head and remains selectable.
    pub cache_layout: CacheLayout,
    /// Extra immutable packed weights; single rows and prefill stay unpacked.
    pub weight_layout: WeightLayout,
    /// Vector exp of the prefill attention tiles (`Exact` for traces).
    pub exp: ExpMode,
    /// Experiment and diagnostic knobs.
    pub tuning: Tuning,
    /// How greedy decoding evaluates the vocabulary head; both select the
    /// same tokens.
    pub head: HeadMode,
    /// Speculative decoding, or `None`.
    pub speculation: Option<Speculation>,
    /// Let drafts continue full n-gram matches from earlier pages of one
    /// `recognize_files` call (outputs unchanged).
    pub document_drafts: bool,
    /// Stop a page once it repeats a cycle of at most 128 tokens for at least
    /// `max(256, 4 * cycle)` tokens (`FinishReason::Repetition`). Never
    /// applied to teacher-forced runs.
    pub repetition_stop: bool,
    /// The decode spin team.
    pub decode_threads: DecodeThreads,
}
impl Default for RunnerConfig {
    /// The automatic configuration: every logical CPU for prefill, a tuned
    /// decode team, the screened head, speculation with document drafts, the
    /// repetition stop and the fast exp.
    fn default() -> Self {
        Self {
            threads: 0,
            batch_size: 1,
            backend: Backend::Auto,
            cache_layout: CacheLayout::Compact,
            weight_layout: WeightLayout::Unpacked,
            exp: ExpMode::Fast,
            tuning: Tuning::default(),
            head: HeadMode::Screened,
            speculation: Some(Speculation::default()),
            document_drafts: true,
            repetition_stop: true,
            decode_threads: DecodeThreads::Auto,
        }
    }
}
impl RunnerConfig {
    /// The bit-exact reference configuration used by traces and the numerical
    /// gates: platform-exact exp, full head, no speculation or repetition
    /// stop, one decode team the size of a 16-thread pool.
    pub fn reference() -> Self {
        Self {
            threads: 16,
            exp: ExpMode::Exact,
            head: HeadMode::Full,
            speculation: None,
            document_drafts: false,
            repetition_stop: false,
            decode_threads: DecodeThreads::Pool,
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be positive");
        self.backend.simd().validate().map_err(anyhow::Error::msg)?;
        ensure!(
            self.tuning.split_chunks.is_none_or(|chunks| (1..=4).contains(&chunks)),
            "tuning.split_chunks must be 1..=4"
        );
        if let Some(speculation) = self.speculation {
            ensure!(
                (1..=7).contains(&speculation.max_draft) && speculation.min_match >= 1,
                "speculation needs 1..=7 drafts and a minimum match of at least 1"
            );
        }
        ensure!(
            !matches!(self.decode_threads, DecodeThreads::Fixed(0)),
            "decode_threads must be at least 1"
        );
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
    fn tuning_knobs_parse_and_reject_unknown_keys() {
        let tuning = Tuning::from_pairs(["prefill-bf16=all", "split-chunks=3", "phases=1"]).unwrap();
        assert_eq!(tuning.prefill_bf16, PrefillBf16::All);
        assert_eq!(tuning.split_chunks, Some(3));
        assert!(tuning.phases && !tuning.prefill_profile);
        assert_eq!(Tuning::from_pairs(Vec::<&str>::new()).unwrap(), Tuning::default());
        for bad in [
            "split-chunks=5",
            "split-chunks=x",
            "phases=maybe",
            "unknown=1",
            "phases",
        ] {
            assert!(Tuning::from_pairs([bad]).is_err(), "{bad}");
        }
        let config: RunnerConfig = serde_json::from_str(r#"{"threads":2,"batch_size":1}"#).unwrap();
        assert_eq!(config.exp, ExpMode::Fast);
        assert_eq!(config.tuning, Tuning::default());
    }
    #[test]
    fn default_is_automatic_and_reference_is_bit_exact() {
        let auto = RunnerConfig::default();
        assert_eq!(auto.threads, 0);
        assert_eq!(auto.head, HeadMode::Screened);
        assert_eq!(auto.speculation, Some(Speculation::default()));
        assert!(auto.document_drafts && auto.repetition_stop);
        assert_eq!(auto.decode_threads, DecodeThreads::Auto);
        assert_eq!(auto.exp, ExpMode::Fast);
        auto.validate().unwrap();
        let reference = RunnerConfig::reference();
        assert_eq!(reference.threads, 16);
        assert_eq!(reference.head, HeadMode::Full);
        assert_eq!(reference.speculation, None);
        assert!(!reference.document_drafts && !reference.repetition_stop);
        assert_eq!(reference.decode_threads, DecodeThreads::Pool);
        assert_eq!(reference.exp, ExpMode::Exact);
        reference.validate().unwrap();
        for bad in [
            RunnerConfig {
                speculation: Some(Speculation {
                    max_draft: 8,
                    min_match: 2,
                }),
                ..Default::default()
            },
            RunnerConfig {
                speculation: Some(Speculation {
                    max_draft: 4,
                    min_match: 0,
                }),
                ..Default::default()
            },
            RunnerConfig {
                decode_threads: DecodeThreads::Fixed(0),
                ..Default::default()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert_eq!("auto".parse::<DecodeThreads>().unwrap(), DecodeThreads::Auto);
        assert_eq!("pool".parse::<DecodeThreads>().unwrap(), DecodeThreads::Pool);
        assert_eq!("12".parse::<DecodeThreads>().unwrap(), DecodeThreads::Fixed(12));
        assert!("0".parse::<DecodeThreads>().is_err());
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
