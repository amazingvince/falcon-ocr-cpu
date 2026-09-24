//! Automatic configuration: what this host can run, where the weights come
//! from, and the plan a [`crate::Runner`] executes. The CLI, the library and
//! the eval binary all resolve through here, so defaults cannot drift, and
//! `doctor` and every result report the same [`Resolved`] plan.
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    config::{DecodeThreads, ExpMode, HeadMode, ModelConfig, RunnerConfig, Speculation, Tuning},
    kernels::PrefillPlan,
    model::{Model, WeightsSource},
    quant::{Kv, Profile, Weights},
};

/// What the runner optimizes for. `near-exact` is the default: 16-bit body
/// weights and a 16-bit KV cache changed 1 token in 24,262 against FP32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// FP32 weights and caches: bit-identical to the FP32 reference.
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
    /// `None` for a research profile (`ModelRequest::profile`).
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
    /// An explicit W8 overlay for fast mode.
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
    pub profile: Option<Profile>,
    /// Research: accept a kernel-ready file of any profile.
    pub allow_research: bool,
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
        }
    }
}

/// Decide where the weights come from. Without `model_file`: the published
/// packed file for the mode in `model_dir`, then the FP32 checkpoint (fast
/// mode also needs the GPTQ overlay unless `allow_rtn`).
pub fn resolve_weights(request: &ModelRequest<'_>) -> Result<WeightsPlan> {
    let dir = request.model_dir;
    if let Some(path) = request.model_file {
        let profile = Model::packed_profile(path)?;
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
             directory, build the overlay with attempt3/make_gptq_overlay.sh, or pass --allow-rtn",
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
        WeightsSource::Packed { path } => Model::load_packed(path, verify),
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
    /// `None` for a research profile of the eval binary.
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
            Some(s) => format!("{} drafts (min match {})", s.max_draft, s.min_match),
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

/// One file `doctor` looked for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStatus {
    pub path: PathBuf,
    pub present: bool,
    pub bytes: Option<u64>,
}

impl FileStatus {
    fn of(path: PathBuf) -> Self {
        let bytes = std::fs::metadata(&path).ok().filter(|m| m.is_file()).map(|m| m.len());
        Self {
            present: bytes.is_some(),
            bytes,
            path,
        }
    }
}

/// The files `doctor` looked for in the model directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Files {
    pub model_file: Option<FileStatus>,
    pub packed_near_exact: FileStatus,
    pub packed_fast: FileStatus,
    pub checkpoint: FileStatus,
    pub overlay: FileStatus,
    pub tokenizer: FileStatus,
}

/// `doctor --load`: the model was loaded and its screened head built.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    pub load_ms: f64,
    pub screened_head_bytes: usize,
}

/// `doctor --probe`: memory read bandwidth and the decode floor it implies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub bytes: usize,
    pub threads: usize,
    /// GB/s of each pass.
    pub gb_per_s: Vec<f64>,
    pub median_gb_per_s: f64,
    /// Milliseconds the plan's per-token weight and head bytes take at the
    /// median bandwidth: no decode step can be faster.
    pub decode_floor_ms: Option<f64>,
}

/// What `doctor` reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Doctor {
    pub host: HostInfo,
    pub files: Files,
    /// The plan for the request, when its weights were found.
    pub plan: Option<Resolved>,
    /// Why there is no plan.
    pub error: Option<String>,
    pub load: Option<LoadReport>,
    pub probe: Option<ProbeReport>,
}

/// What `auto` would do for `request` and `config` on this host, without
/// reading tensors unless `load`; `probe` measures memory bandwidth (512 MB,
/// five passes on the physical cores).
pub fn doctor(request: &ModelRequest<'_>, config: &RunnerConfig, load: bool, probe: bool) -> Result<Doctor> {
    config.validate()?;
    let host = HostInfo::detect().clone();
    let dir = request.model_dir;
    let files = Files {
        model_file: request.model_file.map(|p| FileStatus::of(p.to_path_buf())),
        packed_near_exact: FileStatus::of(dir.join(Mode::NearExact.packed_file_name().unwrap_or_default())),
        packed_fast: FileStatus::of(dir.join(Mode::Fast.packed_file_name().unwrap_or_default())),
        checkpoint: FileStatus::of(dir.join("model.safetensors")),
        overlay: FileStatus::of(
            request
                .w8_artifact
                .map_or_else(|| dir.join(DEFAULT_OVERLAY), Path::to_path_buf),
        ),
        tokenizer: FileStatus::of(dir.join("tokenizer.json")),
    };
    let (mut plan, mut error, mut load_report) = (None, None, None);
    match resolve_weights(request) {
        Err(e) => error = Some(format!("{e:#}")),
        Ok(weights) => {
            let tokenizer_dir = weights.source.tokenizer_dir(dir);
            plan = Some(Resolved::new(
                &host,
                config,
                &ModelFacts::from_plan(&weights)?,
                &tokenizer_dir,
            ));
            if load {
                let started = Instant::now();
                let model = load_model(&weights, false)?;
                if config.head == HeadMode::Screened {
                    model.prepare_screened_head()?;
                }
                load_report = Some(LoadReport {
                    load_ms: started.elapsed().as_secs_f64() * 1000.0,
                    screened_head_bytes: model.screened_head_bytes(),
                });
                plan = Some(Resolved::new(&host, config, &model.facts(), &tokenizer_dir));
            }
        }
    }
    let probe = probe.then(|| {
        let bytes = 512 << 20;
        let threads = host.physical_cores;
        let gb_per_s = crate::cpu::read_bandwidth_gb_s(bytes, threads, 5);
        let median_gb_per_s = crate::tune::median(&gb_per_s);
        ProbeReport {
            bytes,
            threads,
            decode_floor_ms: plan
                .as_ref()
                .map(|p| (p.bytes.weights_per_token + p.bytes.head_per_token) as f64 / (median_gb_per_s * 1e9) * 1e3),
            gb_per_s,
            median_gb_per_s,
        }
    });
    Ok(Doctor {
        host,
        files,
        plan,
        error,
        load: load_report,
        probe,
    })
}

impl std::fmt::Display for Doctor {
    /// The text form (`doctor --text`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let h = &self.host;
        writeln!(
            f,
            "host: {} {}, {} logical / {} physical cores{}{}; {}",
            h.os,
            h.arch,
            h.logical_cpus,
            h.physical_cores,
            if h.smt { " (SMT)" } else { "" },
            h.performance_cores
                .map_or(String::new(), |p| format!(", {p} performance cores")),
            h.features.names().join(" ")
        )?;
        let files = [
            ("model file", self.files.model_file.as_ref()),
            ("packed near-exact", Some(&self.files.packed_near_exact)),
            ("packed fast", Some(&self.files.packed_fast)),
            ("checkpoint", Some(&self.files.checkpoint)),
            ("overlay", Some(&self.files.overlay)),
            ("tokenizer", Some(&self.files.tokenizer)),
        ];
        for (name, file) in files.into_iter().filter_map(|(n, f)| f.map(|f| (n, f))) {
            let size = match file.bytes {
                Some(b) => format!("({:.0} MB)", b as f64 / 1e6),
                None => "(missing)".to_owned(),
            };
            writeln!(f, "file: {name:18} {} {size}", file.path.display())?;
        }
        match (&self.plan, &self.error) {
            (Some(plan), _) => writeln!(f, "plan: {plan}")?,
            (None, Some(error)) => writeln!(f, "plan: none ({error})")?,
            (None, None) => {}
        }
        if let Some(load) = &self.load {
            writeln!(
                f,
                "load: {:.0} ms, screened head {:.0} MB",
                load.load_ms,
                load.screened_head_bytes as f64 / 1e6
            )?;
        }
        if let Some(probe) = &self.probe {
            writeln!(
                f,
                "probe: {} MB x{} on {} threads: {:.1} GB/s median{}",
                probe.bytes >> 20,
                probe.gb_per_s.len(),
                probe.threads,
                probe.median_gb_per_s,
                probe
                    .decode_floor_ms
                    .map_or(String::new(), |ms| format!("; decode floor {ms:.2} ms/token"))
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fabricate_packed(path: &Path, profile: &str) {
        let data = [0u8; 4];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data).unwrap();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("format".to_owned(), crate::model::PACKED_FORMAT.to_owned());
        metadata.insert("profile".to_owned(), profile.to_owned());
        safetensors::serialize_to_file(vec![("t", view)], Some(metadata), path).unwrap();
    }

    #[test]
    fn mode_labels_profiles_and_packed_names_round_trip() {
        for mode in [Mode::Exact, Mode::NearExact, Mode::Fast] {
            assert_eq!(Mode::of_profile(mode.profile(true)), Some(mode));
            assert_eq!(Mode::of_profile(mode.profile(false)), Some(mode));
            assert_eq!(serde_json::to_value(mode).unwrap(), serde_json::json!(mode.label()));
        }
        assert_eq!(Mode::Exact.profile(true), Profile::SPLIT_F32);
        assert_eq!(Mode::Exact.profile(false), Profile::REFERENCE);
        assert_eq!(Mode::of_profile(Profile::W8_BODY), None);
        assert_eq!(Mode::Exact.packed_file_name(), None);
        assert!(Mode::Fast.packed_file_name().unwrap().contains("fast"));
    }

    #[test]
    fn lookup_prefers_packed_files_then_the_checkpoint_and_names_remedies() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Nothing there: every mode fails with a remedy.
        let request = ModelRequest {
            model_dir: root,
            ..ModelRequest::default()
        };
        let error = resolve_weights(&request).unwrap_err().to_string();
        assert!(error.contains("no near-exact model"), "{error}");
        assert!(
            error.contains("huggingface-cli") && error.contains("fetch_reference"),
            "{error}"
        );
        // A checkpoint alone: near-exact quantizes at load, fast needs the overlay.
        std::fs::write(root.join("model.safetensors"), b"stub").unwrap();
        let plan = resolve_weights(&request).unwrap();
        assert_eq!(
            (plan.mode, plan.profile),
            (Some(Mode::NearExact), Profile::W16_BODY_KV_Q16)
        );
        assert!(matches!(
            plan.source,
            WeightsSource::Checkpoint {
                overlay: None,
                rtn: false,
                ..
            }
        ));
        let fast = ModelRequest {
            model_dir: root,
            mode: Some(Mode::Fast),
            ..ModelRequest::default()
        };
        let error = resolve_weights(&fast).unwrap_err().to_string();
        assert!(
            error.contains("--allow-rtn") && error.contains("make_gptq_overlay"),
            "{error}"
        );
        let rtn = ModelRequest {
            allow_rtn: true,
            ..fast.clone()
        };
        assert!(matches!(
            resolve_weights(&rtn).unwrap().source,
            WeightsSource::Checkpoint {
                rtn: true,
                overlay: None,
                ..
            }
        ));
        std::fs::write(root.join(DEFAULT_OVERLAY), b"stub").unwrap();
        assert!(matches!(
            resolve_weights(&fast).unwrap().source,
            WeightsSource::Checkpoint {
                rtn: false,
                overlay: Some(_),
                ..
            }
        ));
        assert!(
            resolve_weights(&ModelRequest {
                w8_artifact: Some(Path::new("x")),
                ..request.clone()
            })
            .is_err()
        );
        // Exact: the split cache for single pages, the reference cache otherwise.
        let exact = ModelRequest {
            model_dir: root,
            mode: Some(Mode::Exact),
            split_exact: true,
            ..ModelRequest::default()
        };
        assert_eq!(resolve_weights(&exact).unwrap().profile, Profile::SPLIT_F32);
        let exact = ModelRequest {
            split_exact: false,
            ..exact
        };
        assert_eq!(resolve_weights(&exact).unwrap().profile, Profile::REFERENCE);
        // A packed file next to the checkpoint wins for its mode, unless the
        // caller wants the checkpoint (pack).
        let packed = root.join(Mode::NearExact.packed_file_name().unwrap());
        fabricate_packed(&packed, "w16-body-kv-q16");
        let plan = resolve_weights(&request).unwrap();
        assert_eq!(plan.source, WeightsSource::Packed { path: packed.clone() });
        assert_eq!(
            (plan.mode, plan.profile),
            (Some(Mode::NearExact), Profile::W16_BODY_KV_Q16)
        );
        assert!(matches!(
            resolve_weights(&ModelRequest {
                from_checkpoint: true,
                ..request.clone()
            })
            .unwrap()
            .source,
            WeightsSource::Checkpoint { .. }
        ));
        // A mislabelled packed file is refused.
        fabricate_packed(&packed, "w8-body-kv-q8");
        let error = resolve_weights(&request).unwrap_err().to_string();
        assert!(error.contains("not a near-exact"), "{error}");
        // An explicit file decides the mode; a disagreeing --mode is an error.
        let file = root.join("custom.safetensors");
        fabricate_packed(&file, "w8-body-kv-q8");
        let explicit = ModelRequest {
            model_dir: root,
            model_file: Some(&file),
            ..ModelRequest::default()
        };
        let plan = resolve_weights(&explicit).unwrap();
        assert_eq!(plan.mode, Some(Mode::Fast));
        let conflict = ModelRequest {
            mode: Some(Mode::NearExact),
            ..explicit.clone()
        };
        let error = resolve_weights(&conflict).unwrap_err().to_string();
        assert!(error.contains("--mode near-exact"), "{error}");
        fabricate_packed(&file, "w8-body");
        let error = resolve_weights(&explicit).unwrap_err().to_string();
        assert!(error.contains("research profile"), "{error}");
        let research = resolve_weights(&ModelRequest {
            allow_research: true,
            ..explicit.clone()
        })
        .unwrap();
        assert_eq!((research.mode, research.profile), (None, Profile::W8_BODY));
        // A named research profile reads the checkpoint (8-bit: overlay or RTN).
        let named = ModelRequest {
            model_dir: root,
            profile: Some(Profile::KV_Q8),
            ..ModelRequest::default()
        };
        let named_plan = resolve_weights(&named).unwrap();
        assert_eq!((named_plan.mode, named_plan.profile), (None, Profile::KV_Q8));
        assert!(matches!(
            named_plan.source,
            WeightsSource::Checkpoint {
                rtn: false,
                overlay: None,
                ..
            }
        ));
        let named = ModelRequest {
            profile: Some(Profile::W8_BODY_KV_Q8),
            ..named
        };
        let named_plan = resolve_weights(&named).unwrap();
        assert_eq!(named_plan.mode, Some(Mode::Fast));
        assert!(matches!(
            named_plan.source,
            WeightsSource::Checkpoint { overlay: Some(_), .. }
        ));
        assert!(
            resolve_weights(&ModelRequest {
                mode: Some(Mode::Exact),
                ..named
            })
            .is_err()
        );
        // The tokenizer comes from the file's folder only when it holds one.
        assert_eq!(
            plan.source.tokenizer_dir(Path::new("elsewhere")),
            PathBuf::from("elsewhere")
        );
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(plan.source.tokenizer_dir(Path::new("elsewhere")), root.to_path_buf());
    }

    #[test]
    fn byte_budget_matches_the_documented_figures() {
        let c: ModelConfig = serde_json::from_str(include_str!("../tests/fixtures/model-config.json")).unwrap();
        let fast = byte_budget(&c, Profile::W8_BODY_KV_Q8, HeadMode::Screened);
        // 168.7M body parameters as INT8 plus FP32 scales per 64; the INT8
        // screen of the 50.3M-parameter head plus its scales.
        assert_eq!(fast.weights_per_token, 179_232_768);
        assert_eq!(fast.head_per_token, 53_477_376);
        // 22 layers x 8 groups x 170 bytes: 196 MB at the 6,544-token journal page.
        assert_eq!(fast.kv_per_position, 29_920);
        let near = byte_budget(&c, Profile::W16_BODY_KV_Q16, HeadMode::Screened);
        assert_eq!(near.weights_per_token, 2 * 168_689_664 + 10_543_104);
        assert_eq!(near.kv_per_position, 22 * 8 * 330);
        let exact = byte_budget(&c, Profile::SPLIT_F32, HeadMode::Full);
        assert_eq!(exact.weights_per_token, 4 * 168_689_664);
        assert_eq!(exact.head_per_token, 4 * 65536 * 768);
        assert_eq!(exact.kv_per_position, 22 * 8 * 640);
        assert_eq!(
            byte_budget(&c, Profile::REFERENCE, HeadMode::Full).kv_per_position,
            22 * 1536 * 4
        );
    }

    #[test]
    fn doctor_reports_files_and_a_plan_without_reading_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let config = RunnerConfig::default();
        let request = ModelRequest {
            model_dir: root,
            ..ModelRequest::default()
        };
        let report = doctor(&request, &config, false, false).unwrap();
        assert!(report.plan.is_none() && report.error.as_deref().unwrap().contains("no near-exact model"));
        assert!(!report.files.checkpoint.present && report.files.model_file.is_none());
        assert!(report.to_string().contains("plan: none"));
        // A checkpoint directory: the plan comes from config.json.
        std::fs::write(root.join("model.safetensors"), b"stub").unwrap();
        std::fs::write(
            root.join("config.json"),
            include_str!("../tests/fixtures/model-config.json"),
        )
        .unwrap();
        let report = doctor(&request, &config, false, false).unwrap();
        let plan = report.plan.as_ref().unwrap();
        assert_eq!(plan.mode, Some(Mode::NearExact));
        assert_eq!(plan.kv_cache, "q16");
        assert_eq!(plan.bytes.weights_per_token, 2 * 168_689_664 + 10_543_104);
        assert!(report.files.checkpoint.present && report.error.is_none());
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["plan"]["profile"], "w16-body-kv-q16");
        assert!(report.to_string().starts_with("host: "));
    }

    #[test]
    fn host_info_is_consistent() {
        let host = HostInfo::detect();
        assert!(host.physical_cores >= 1 && host.physical_cores <= host.logical_cpus);
        assert_eq!(host.smt, host.logical_cpus > host.physical_cores);
        if let Some(p) = host.performance_cores {
            assert!(p >= 1 && p < host.physical_cores);
        }
        #[cfg(target_arch = "x86_64")]
        assert_eq!(host.features.avx2, std::is_x86_feature_detected!("avx2"));
        assert!(serde_json::to_string(host).unwrap().contains("\"logical_cpus\""));
    }
}
