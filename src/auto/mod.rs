//! Automatic configuration: what this host can run, where the weights come
//! from, and the plan a [`crate::Runner`] executes. The CLI, the library and
//! the eval binary all resolve through here, so defaults cannot drift, and
//! `doctor` and every result report the same [`Resolved`] plan.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    config::{DecodeThreads, Drafter, ExpMode, HeadMode, ModelConfig, RunnerConfig, Speculation, Tuning},
    kernels::PrefillPlan,
    model::{Model, WeightsSource},
    quant::{Kv, Profile, Weights},
};

/// What the runner optimizes for. `near-exact` is the default: 16-bit body
/// weights and a 16-bit KV cache changed 1 token in 24,262 against FP32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// FP32 weights and caches: the FP32 reference's tokens (bitwise under
    /// the reference configuration).
    Exact,
    /// 16-bit body weights and 16-bit KV cache (absmax scale per 64 weights
    /// or 32 cache values): about 1.8x faster than exact, 1 changed token in
    /// 24,262 teacher-forced steps (KL 9e-8).
    NearExact,
    /// 8-bit GPTQ body weights and an 8-bit KV cache, BF16 prefill attention
    /// on AVX512-BF16 CPUs: about 3x faster than exact, 63 changed tokens in
    /// 24,262; failed its held-out budget on handwriting loops and degraded
    /// scans.
    Fast,
}

/// The GPTQ overlay `Mode::Fast` reads from the model directory by default.
pub const DEFAULT_OVERLAY: &str = "w8-gptq.safetensors";
/// The published draft head, used automatically when it sits next to the
/// model files ([`with_default_draft_head`]).
pub const DRAFT_HEAD_FILE: &str = "falcon-ocr-v1.5-draft-head.safetensors";

/// The folder that holds the model files of `request`: the model file's
/// folder, else the model directory.
pub fn model_files_dir(request: &ModelRequest<'_>) -> PathBuf {
    request
        .model_file
        .and_then(Path::parent)
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(request.model_dir)
        .to_path_buf()
}

/// `config` drafting with the published draft head when `dir` holds
/// [`DRAFT_HEAD_FILE`], speculation is on and the caller chose no drafter
/// (`explicit`: `--drafter` or `--draft-head` was given). Drafts never
/// change tokens; the head only makes decoding faster.
pub fn with_default_draft_head(config: RunnerConfig, dir: &Path, explicit: bool) -> RunnerConfig {
    let path = dir.join(DRAFT_HEAD_FILE);
    if explicit || config.speculation.is_none() || config.draft_head.is_some() || !path.is_file() {
        return config;
    }
    RunnerConfig {
        drafter: Drafter::Head,
        draft_head: Some(path),
        ..config
    }
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::NearExact => "near-exact",
            Self::Fast => "fast",
        }
    }
    /// The weight/cache profile: `split_exact` picks the split FP32 cache for
    /// exact single-page runs (bitwise equal to the compact reference cache,
    /// about 7% faster decode).
    pub fn profile(self, split_exact: bool) -> Profile {
        match self {
            Self::Exact if split_exact => Profile::SPLIT_F32,
            Self::Exact => Profile::REFERENCE,
            Self::NearExact => Profile::W16_BODY_KV_Q16,
            Self::Fast => Profile::W8_BODY_KV_Q8,
        }
    }
    /// The mode a profile belongs to, if it is one of the shipped ones.
    pub fn of_profile(profile: Profile) -> Option<Self> {
        match (profile.weights, profile.kv) {
            (Weights::F32, Kv::Compact | Kv::F32Split) => Some(Self::Exact),
            (Weights::Int16, Kv::Q16) => Some(Self::NearExact),
            (Weights::Int8, Kv::Q8) => Some(Self::Fast),
            _ => None,
        }
    }
    /// File name of the published kernel-ready model file for this mode.
    pub fn packed_file_name(self) -> Option<&'static str> {
        match self {
            Self::Exact => None,
            Self::NearExact => Some("falcon-ocr-v1.5-near-exact.safetensors"),
            Self::Fast => Some("falcon-ocr-v1.5-fast.safetensors"),
        }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// CPU instruction sets the kernels dispatch on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512bf16: bool,
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub bf16: bool,
}

impl CpuFeatures {
    /// Names of the features present, for reports.
    pub fn names(&self) -> Vec<&'static str> {
        let all = [
            ("avx2", self.avx2),
            ("fma", self.fma),
            ("f16c", self.f16c),
            ("avx512f", self.avx512f),
            ("avx512bw", self.avx512bw),
            ("avx512bf16", self.avx512bf16),
            ("neon", self.neon),
            ("dotprod", self.dotprod),
            ("i8mm", self.i8mm),
            ("bf16", self.bf16),
        ];
        all.into_iter().filter(|&(_, on)| on).map(|(name, _)| name).collect()
    }
    /// The running CPU's features.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                fma: std::is_x86_feature_detected!("fma"),
                f16c: std::is_x86_feature_detected!("f16c"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512bw: std::is_x86_feature_detected!("avx512bw"),
                avx512bf16: std::is_x86_feature_detected!("avx512bf16"),
                ..Self::default()
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon: true,
                dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
                i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
                bf16: std::arch::is_aarch64_feature_detected!("bf16"),
                ..Self::default()
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }
}

/// The host as the runner sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostInfo {
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub physical_cores: usize,
    /// More logical CPUs than physical cores (SMT).
    pub smt: bool,
    /// Performance cores on a hybrid CPU (`None` when all cores are alike or
    /// the OS does not say).
    pub performance_cores: Option<usize>,
    pub features: CpuFeatures,
}

impl HostInfo {
    /// The running host, detected once.
    pub fn detect() -> &'static HostInfo {
        static HOST: std::sync::OnceLock<HostInfo> = std::sync::OnceLock::new();
        HOST.get_or_init(|| {
            let logical_cpus = crate::cpu::logical_cpus();
            let physical_cores = crate::cpu::physical_cores();
            HostInfo {
                os: std::env::consts::OS.to_owned(),
                arch: std::env::consts::ARCH.to_owned(),
                logical_cpus,
                physical_cores,
                smt: logical_cpus > physical_cores,
                performance_cores: crate::cpu::performance_cores(),
                features: CpuFeatures::detect(),
            }
        })
    }
}

/// Which weights a run loads: the source, and the mode and profile it yields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightsPlan {
    pub source: WeightsSource,
    /// `None` for a research profile (`ModelRequest::profile`, or a
    /// `ModelRequest::kv_cache` other than the mode's).
    pub mode: Option<Mode>,
    pub profile: Profile,
}

/// What the caller asked for; everything else is looked up.
#[derive(Clone, Debug)]
pub struct ModelRequest<'a> {
    /// Directory holding the checkpoint (`model.safetensors`, `config.json`,
    /// tokenizer files) and, optionally, the published packed files.
    pub model_dir: &'a Path,
    /// An explicit kernel-ready file; it decides the mode.
    pub model_file: Option<&'a Path>,
    /// `None` means near-exact.
    pub mode: Option<Mode>,
    /// An explicit W8 overlay for fast mode, applied to the checkpoint (a
    /// packed fast file in `model_dir` is then not used).
    pub w8_artifact: Option<&'a Path>,
    /// Let fast mode quantize round-to-nearest at load when no overlay exists
    /// (about three times the changed tokens of GPTQ).
    pub allow_rtn: bool,
    /// Exact mode may use the split FP32 cache (single pages, compact
    /// layout, unpacked weights).
    pub split_exact: bool,
    /// Skip the packed files and read the checkpoint (`pack` writes them).
    pub from_checkpoint: bool,
    /// Research: load this profile from the checkpoint whatever the mode
    /// (8-bit profiles take the overlay when present, else round-to-nearest).
    /// With `model_file` it must name the file's weights (`kv_cache` picks the
    /// cache).
    pub profile: Option<Profile>,
    /// Research: accept a kernel-ready file of any profile.
    pub allow_research: bool,
    /// Research: replace the KV half of the profile the rest of the request
    /// resolves to (`falcon-ocr --kv-cache`). The cache is built at run time,
    /// so this works with kernel-ready files too; the plan (and so `doctor`
    /// and every result) then names the profile that runs, a research
    /// profile unless it is a mode's own.
    pub kv_cache: Option<Kv>,
}

impl Default for ModelRequest<'_> {
    /// Near-exact from `artifacts/model`, the reference cache for exact.
    fn default() -> Self {
        Self {
            model_dir: Path::new("artifacts/model"),
            model_file: None,
            mode: None,
            w8_artifact: None,
            allow_rtn: false,
            split_exact: false,
            from_checkpoint: false,
            profile: None,
            allow_research: false,
            kv_cache: None,
        }
    }
}

/// Decide where the weights come from. Without `model_file`: the published
/// packed file for the mode in `model_dir`, then the FP32 checkpoint (fast
/// mode also needs the GPTQ overlay unless `allow_rtn`). An explicit
/// `w8_artifact` always reads the checkpoint, so the overlay asked for is
/// the one used. `kv_cache` then replaces the KV half of the profile.
pub fn resolve_weights(request: &ModelRequest<'_>) -> Result<WeightsPlan> {
    let plan = recorded_plan(request)?;
    Ok(match request.kv_cache {
        Some(kv) => {
            let profile = Profile::new(plan.profile.weights, kv);
            WeightsPlan {
                mode: Mode::of_profile(profile),
                profile,
                ..plan
            }
        }
        None => plan,
    })
}

/// [`resolve_weights`] without the KV override: the profile the model file
/// records, or the one the mode or research profile names.
fn recorded_plan(request: &ModelRequest<'_>) -> Result<WeightsPlan> {
    let dir = request.model_dir;
    if let Some(path) = request.model_file {
        let profile = Model::packed_profile(path)?;
        if let Some(requested) = request.profile {
            ensure!(
                requested.weights == profile.weights,
                "{} holds {} weights, not those of {}",
                path.display(),
                profile.label(),
                requested.label()
            );
        }
        let mode = Mode::of_profile(profile);
        ensure!(
            mode.is_some() || request.allow_research,
            "{} holds the research profile {}",
            path.display(),
            profile.label()
        );
        if let Some(requested) = request.mode {
            ensure!(
                mode == Some(requested),
                "{} is a {} model file, but --mode {requested} was requested",
                path.display(),
                mode.map_or("research", Mode::label),
            );
        }
        ensure!(
            request.w8_artifact.is_none(),
            "--w8-artifact applies to the checkpoint loader, not to a kernel-ready model file"
        );
        return Ok(WeightsPlan {
            source: WeightsSource::Packed {
                path: path.to_path_buf(),
            },
            mode,
            profile,
        });
    }
    if let Some(profile) = request.profile {
        // Research: a named profile from the checkpoint.
        if let Some(requested) = request.mode {
            ensure!(
                Mode::of_profile(profile) == Some(requested),
                "profile {} is not the {requested} mode's",
                profile.label()
            );
        }
        require_checkpoint(dir, None)?;
        let eight_bit = profile.quantizes_body() && profile.weight_bits() == 8;
        let overlay = if eight_bit {
            request
                .w8_artifact
                .map(Path::to_path_buf)
                .or_else(|| Some(dir.join(DEFAULT_OVERLAY)).filter(|p| p.is_file()))
        } else {
            ensure!(request.w8_artifact.is_none(), "--w8-artifact applies to 8-bit profiles");
            None
        };
        return Ok(WeightsPlan {
            source: WeightsSource::Checkpoint {
                dir: dir.to_path_buf(),
                rtn: eight_bit && overlay.is_none(),
                overlay,
            },
            mode: Mode::of_profile(profile),
            profile,
        });
    }
    let mode = request.mode.unwrap_or(Mode::NearExact);
    ensure!(
        request.w8_artifact.is_none() || mode == Mode::Fast,
        "--w8-artifact applies to --mode fast"
    );
    if let Some(name) = mode.packed_file_name()
        && !request.from_checkpoint
        && request.w8_artifact.is_none()
    {
        let packed = dir.join(name);
        if packed.is_file() {
            let profile = Model::packed_profile(&packed)?;
            ensure!(
                Mode::of_profile(profile) == Some(mode),
                "{} holds the {} profile, not a {mode} model",
                packed.display(),
                profile.label(),
            );
            return Ok(WeightsPlan {
                source: WeightsSource::Packed { path: packed },
                mode: Some(mode),
                profile,
            });
        }
    }
    if let Some(overlay) = request.w8_artifact
        && !dir.join("model.safetensors").is_file()
    {
        bail!(
            "--w8-artifact {} applies to the FP32 checkpoint, which {} lacks: download it with \
             python scripts/fetch_reference.py --output {}, or drop --w8-artifact to use a packed fast file",
            overlay.display(),
            dir.display(),
            dir.display()
        );
    }
    require_checkpoint(dir, Some(mode))?;
    let profile = mode.profile(request.split_exact);
    let mut overlay = None;
    if mode == Mode::Fast {
        overlay = request
            .w8_artifact
            .map(Path::to_path_buf)
            .or_else(|| Some(dir.join(DEFAULT_OVERLAY)).filter(|p| p.is_file()));
        ensure!(
            overlay.is_some() || request.allow_rtn,
            "fast mode needs the GPTQ overlay {} (round-to-nearest quantization triples the changed tokens): \
             download falcon-ocr-v1.5-fast.safetensors from amazingvince/falcon-ocr-v1.5-cpu into the model \
             directory, build the overlay with tools/make_gptq_overlay.sh, or pass --allow-rtn",
            dir.join(DEFAULT_OVERLAY).display()
        );
    }
    Ok(WeightsPlan {
        source: WeightsSource::Checkpoint {
            dir: dir.to_path_buf(),
            rtn: mode == Mode::Fast && overlay.is_none(),
            overlay,
        },
        mode: Some(mode),
        profile,
    })
}

/// Fail with the download remedies when `dir` holds no FP32 checkpoint.
fn require_checkpoint(dir: &Path, mode: Option<Mode>) -> Result<()> {
    if dir.join("model.safetensors").is_file() {
        return Ok(());
    }
    let mut remedies = vec![format!(
        "download the FP32 checkpoint: python scripts/fetch_reference.py --output {}",
        dir.display()
    )];
    if let Some(name) = mode.and_then(Mode::packed_file_name) {
        remedies.insert(
            0,
            format!(
                "download the packed file: huggingface-cli download amazingvince/falcon-ocr-v1.5-cpu {name} \
                 --local-dir {} (or pass --model-file)",
                dir.display()
            ),
        );
    }
    bail!(
        "no {} model in {}: {}",
        mode.map_or("checkpoint".to_owned(), |m| m.label().to_owned()),
        dir.display(),
        remedies.join("; or ")
    )
}

/// Load the weights a plan names. `verify` checks every tensor digest of a
/// kernel-ready file.
pub fn load_model(plan: &WeightsPlan, verify: bool) -> Result<Model> {
    match &plan.source {
        WeightsSource::Packed { path } => {
            // The file records its mode's profile; the plan may name another
            // KV cache for the same weights (`ModelRequest::kv_cache`).
            let model = Model::load_packed(path, verify)?;
            ensure!(
                model.profile().weights == plan.profile.weights,
                "{} holds {} weights, not those of the planned {}",
                path.display(),
                model.profile().label(),
                plan.profile.label()
            );
            Ok(model.with_kv_cache(plan.profile.kv))
        }
        WeightsSource::Checkpoint { dir, overlay, .. } => {
            if plan.profile == Profile::REFERENCE {
                Model::load(dir)
            } else {
                Model::load_profile(dir, plan.profile, overlay.as_deref())
            }
        }
    }
}

impl Model {
    /// Load `mode` from `dir`: the published packed file when present, else
    /// the FP32 checkpoint (fast mode then needs the GPTQ overlay). Exact mode
    /// uses the split FP32 cache, as `falcon-ocr run` does.
    pub fn load_mode(dir: impl AsRef<Path>, mode: Mode) -> Result<Self> {
        let plan = resolve_weights(&ModelRequest {
            model_dir: dir.as_ref(),
            mode: Some(mode),
            split_exact: true,
            ..ModelRequest::default()
        })?;
        load_model(&plan, false)
    }
}

/// What [`Resolved`] needs to know about a model, from the loaded model or
/// from a plan alone (headers only).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFacts {
    pub source: WeightsSource,
    pub profile: Profile,
    pub config: ModelConfig,
    /// `Model::body_bits`: the profile's width for a plan.
    pub body_bits: Option<u32>,
    pub overlay_sha256: Option<String>,
}

impl ModelFacts {
    /// From a plan without reading tensors: the checkpoint's `config.json`
    /// or the packed file's header (which also names its overlay digest).
    pub fn from_plan(plan: &WeightsPlan) -> Result<Self> {
        let (config, overlay_sha256) = match &plan.source {
            WeightsSource::Packed { path } => crate::model::packed_facts(path)?,
            WeightsSource::Checkpoint { dir, .. } => {
                let path = dir.join("config.json");
                let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
                let config: ModelConfig = serde_json::from_slice(&bytes)?;
                config.validate()?;
                (config, None)
            }
        };
        Ok(Self {
            source: plan.source.clone(),
            profile: plan.profile,
            config,
            body_bits: plan.profile.quantizes_body().then(|| plan.profile.weight_bits()),
            overlay_sha256,
        })
    }
}

impl Model {
    /// The facts `Resolved` reports about this model.
    pub fn facts(&self) -> ModelFacts {
        ModelFacts {
            source: self.source().clone(),
            profile: self.profile(),
            config: self.config().clone(),
            body_bits: self.body_bits(),
            overlay_sha256: self.overlay_sha256().map(str::to_owned),
        }
    }
}

/// Thread plan: the prefill pool and the decode team.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadPlan {
    pub prefill: usize,
    pub decode: DecodePlan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodePlan {
    Fixed(usize),
    /// Candidate team sizes the tuner times, and the one it starts on.
    Auto {
        candidates: Vec<usize>,
        start: usize,
    },
}

/// Bytes one decode step streams (what memory bandwidth must deliver).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteBudget {
    /// Body weights (codes and scales) read per token.
    pub weights_per_token: usize,
    /// Vocabulary head read per token (the INT8 screen, or the FP32 head).
    pub head_per_token: usize,
    /// KV cache bytes per cached position, summed over layers.
    pub kv_per_position: usize,
}

/// The plan a runner executes for a host, config and model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolved {
    /// `None` for a research profile (the eval binary's, or one
    /// `falcon-ocr --kv-cache` makes).
    pub mode: Option<Mode>,
    pub profile: Profile,
    pub weights: WeightsSource,
    pub overlay_sha256: Option<String>,
    pub tokenizer_dir: PathBuf,
    /// Instruction set of the decode GEMV and attention kernels.
    pub decode_isa: String,
    pub prefill: PrefillPlan,
    pub exp: ExpMode,
    pub head: HeadMode,
    pub speculation: Option<Speculation>,
    /// The draft source, and the draft head file when it drafts.
    #[serde(default)]
    pub drafter: Drafter,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_head: Option<PathBuf>,
    pub document_drafts: bool,
    pub repetition_stop: bool,
    pub threads: ThreadPlan,
    /// KV cache record format after the prefix is sealed.
    pub kv_cache: String,
    pub bytes: ByteBudget,
    pub image_decoder: String,
    pub tuning: Tuning,
}

impl Resolved {
    /// The plan for `config` on `host` for a model with `facts`. `config`
    /// must be valid ([`RunnerConfig::validate`]).
    pub fn new(host: &HostInfo, config: &RunnerConfig, facts: &ModelFacts, tokenizer_dir: &Path) -> Self {
        let profile = facts.profile;
        let simd = config.backend.simd();
        let threads = if config.threads == 0 {
            host.logical_cpus
        } else {
            config.threads
        };
        let decode = match config.decode_threads {
            DecodeThreads::Fixed(n) => DecodePlan::Fixed(n),
            DecodeThreads::Pool => DecodePlan::Fixed(threads),
            DecodeThreads::Auto => {
                let (candidates, start) = crate::tune::auto_candidates(host, threads, config.speculation.is_some());
                DecodePlan::Auto { candidates, start }
            }
        };
        Self {
            mode: Mode::of_profile(profile),
            profile,
            weights: facts.source.clone(),
            overlay_sha256: facts.overlay_sha256.clone(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            decode_isa: format!("{:?}", simd.resolved()).to_lowercase(),
            prefill: crate::kernels::prefill_plan(facts.body_bits, simd, config.tuning.prefill_bf16),
            exp: config.exp,
            head: config.head,
            speculation: config.speculation,
            drafter: config.drafter,
            draft_head: config.draft_head.clone().filter(|_| config.drafter != Drafter::Ngram),
            document_drafts: config.document_drafts,
            repetition_stop: config.repetition_stop,
            threads: ThreadPlan {
                prefill: threads,
                decode,
            },
            kv_cache: kv_cache_label(profile).to_owned(),
            bytes: byte_budget(&facts.config, profile, config.head),
            image_decoder: if cfg!(feature = "turbojpeg") {
                "libjpeg-turbo (Pillow-exact)"
            } else {
                "image-rs (not Pillow-exact)"
            }
            .to_owned(),
            tuning: config.tuning,
        }
    }
}

fn kv_cache_label(profile: Profile) -> &'static str {
    match profile.kv {
        Kv::Compact => "f32-compact",
        Kv::F32Split => "f32-split",
        Kv::Q16 => "q16",
        Kv::Q8 => "q8",
    }
}

/// Bytes a decode step streams for this model and profile.
fn byte_budget(c: &ModelConfig, profile: Profile, head: HeadMode) -> ByteBudget {
    let (qdim, kdim) = (c.query_dim(), c.kv_dim());
    let body_params = c.n_layers * (c.dim * (qdim + 2 * kdim) + qdim * c.dim + 3 * c.ffn_dim * c.dim);
    let weights_per_token = if profile.quantizes_body() {
        // One FP32 scale per 64 codes.
        body_params * profile.weight_bits() as usize / 8 + body_params / 64 * 4
    } else {
        body_params * 4
    };
    let head_params = c.vocab_size * c.dim;
    let head_per_token = match head {
        HeadMode::Screened => head_params + head_params / 64 * 4,
        HeadMode::Full => head_params * 4,
    };
    // Split records hold 160 elements per KV group and position (the shared
    // temporal key half, both heads' spatial halves and the value); the coded
    // formats add one 16-bit scale per 32 elements.
    let record = match profile.kv {
        Kv::Compact => (qdim + kdim) * 4 / c.n_kv_heads,
        Kv::F32Split => 160 * 4,
        Kv::Q16 => 160 * 2 + 5 * 2,
        Kv::Q8 => 160 + 5 * 2,
    };
    ByteBudget {
        weights_per_token,
        head_per_token,
        kv_per_position: c.n_layers * c.n_kv_heads * record,
    }
}

impl std::fmt::Display for Resolved {
    /// One line for the terminal (the JSON form is `serde`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let decode = match &self.threads.decode {
            DecodePlan::Fixed(n) => n.to_string(),
            DecodePlan::Auto { candidates, .. } => format!(
                "auto {{{}}}",
                candidates.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
            ),
        };
        let speculation = match self.speculation {
            Some(s) => format!(
                "{} drafts (min match {}){}",
                s.max_draft,
                s.min_match,
                match (&self.draft_head, self.drafter) {
                    (Some(path), Drafter::Head) => format!(", draft head {}", path.display()),
                    (Some(path), _) => format!(", n-gram then draft head {}", path.display()),
                    (None, _) => ", n-gram".to_owned(),
                }
            ),
            None => "off".to_owned(),
        };
        let weights = match &self.weights {
            WeightsSource::Packed { path } => format!("packed {}", path.display()),
            WeightsSource::Checkpoint { dir, overlay, rtn } => format!(
                "checkpoint {}{}",
                dir.display(),
                match (overlay, rtn) {
                    (Some(o), _) => format!(" + overlay {}", o.display()),
                    (None, true) => " (round-to-nearest)".to_owned(),
                    (None, false) => String::new(),
                }
            ),
        };
        write!(
            f,
            "mode {} ({}) | weights {weights} | prefill {} threads, projections {}, attention {} | decode {decode} \
             threads, {} kernels, kv {} | head {:?} | speculation {speculation} | repetition stop {} | exp {:?}",
            self.mode.map_or("research", Mode::label),
            self.profile.label(),
            self.threads.prefill,
            self.prefill.projection,
            self.prefill.attention,
            self.decode_isa,
            self.kv_cache,
            self.head,
            if self.repetition_stop { "on" } else { "off" },
            self.exp,
        )
    }
}

mod doctor;
#[cfg(test)]
mod tests;

pub use doctor::{Doctor, FileStatus, Files, LoadReport, ProbeReport, doctor};
