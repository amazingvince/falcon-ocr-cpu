use crate::{
    config::{CacheLayout, GenerationOptions, HeadMode, RunnerConfig, WeightLayout},
    model::{Model, Next, Session, image_range, positions},
    preprocess::{PreparedImage, decode_file, first_resize_rgb, prepare_file_timed, prepare_first, prepare_rgb},
    router::{self, Route, RoutedAttempt},
    tokenizer::OcrTokenizer,
    trace::{NoTrace, Trace},
};
use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc, time::Instant};

mod cohort;
mod generate;
mod speculate;

use generate::{DecodeLoop, Generation, Page};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Eos,
    Length,
    /// Stopped by the repetition stop (`RunnerConfig::repetition_stop`).
    Repetition,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Timings {
    /// File decode time; zero for an already decoded RGB buffer.
    pub image_decode_ms: f64,
    pub preprocessing_ms: f64,
    /// Image-projector linear call only; contained in prefill_ms. Excludes
    /// feature allocation and copying features/token embeddings into the prefix.
    /// None means the historical/backend result did not measure this stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_projection_ms: Option<f64>,
    /// Complete transformer prefill forward, including final vocabulary logits;
    /// contained in prefill_ms. Excludes embedding, setup and argmax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transformer_prefill_ms: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub time_to_first_token_ms: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OcrResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub width: usize,
    pub height: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// `fp32` for the exact profiles, else the weight/cache profile label.
    pub precision: String,
    /// Instruction set of the decode kernels (`avx2`, `neon`, `scalar`).
    pub backend: String,
    /// The mode the page ran in (`null` for a research profile).
    #[serde(default)]
    pub mode: Option<crate::auto::Mode>,
    /// The decode team this page used: the fixed size, or the size the tuner
    /// chose (`null` while it was still measuring).
    #[serde(default)]
    pub decode_threads: Option<usize>,
    /// Everything `auto` decided for the runner (`Runner::resolved`).
    #[serde(default)]
    pub plan: Option<crate::auto::Resolved>,
    /// The decode-thread tuner's measurements, on the page during which it
    /// chose (`Runner::decode_tuning` has it afterwards).
    #[serde(default)]
    pub decode_tuning: Option<crate::tune::TuneReport>,
    /// Records written before compact caches existed were expanded.
    #[serde(default = "legacy_cache_layout")]
    pub cache_layout: CacheLayout,
    #[serde(default)]
    pub weight_layout: WeightLayout,
    /// Shared packed tensor payload on Model, separate from per-request KV.
    #[serde(default)]
    pub packed_weight_bytes: usize,
    #[serde(default)]
    pub weight_packing_ms: f64,
    pub teacher_forced: bool,
    /// `max_new_tokens` was lowered to fit the context
    /// (`GenerationOptions::fit_budget`).
    #[serde(default)]
    pub budget_clamped: bool,
    /// The resolution router's choice (`GenerationOptions::route`); `width`
    /// and `height` are those of the input that produced `text`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<Route>,
    pub timings: Timings,
}

fn legacy_cache_layout() -> CacheLayout {
    CacheLayout::Expanded
}

pub use crate::config::DecodeThreads;

pub struct Runner {
    model: Arc<Model>,
    tokenizer: OcrTokenizer,
    pool: rayon::ThreadPool,
    config: RunnerConfig,
    /// What this runner will do, for reports (`Runner::resolved`).
    resolved: crate::auto::Resolved,
    head: HeadMode,
    /// Stop for degenerate repetition loops (`crate::repetition`).
    repetition_stop: bool,
    /// Spin-waiting workers for decode steps (prefill keeps using `pool`): one
    /// fixed team, or one team per candidate size with the tuner choosing
    /// between them (`DecodeThreads::Auto`).
    decode: Decode,
    /// Speculative decoding (`RunnerConfig::speculation`): at most this many
    /// drafted tokens per step and the minimum n-gram match; `None` is off.
    speculation: Option<(usize, usize)>,
    /// Drafts from earlier pages of the current `recognize_files` call
    /// (`RunnerConfig::document_drafts`).
    document: Option<std::sync::Mutex<crate::draft::DocumentHistory>>,
    /// The trained draft head, when the drafter uses one.
    draft_head: Option<Arc<crate::draft_head::DraftHead>>,
}

enum Decode {
    Fixed(crate::team::Team),
    Auto(AutoThreads),
}

struct AutoThreads {
    teams: Vec<crate::team::Team>,
    tuner: std::sync::Mutex<crate::tune::Tuner>,
}

/// One spin team per candidate decode size for a `pool_threads`-thread prefill
/// pool on `host`, and the tuner that picks between them (`crate::tune`).
fn auto_threads(host: &crate::auto::HostInfo, pool_threads: usize, speculating: bool) -> std::io::Result<AutoThreads> {
    let (sizes, start) = crate::tune::auto_candidates(host, pool_threads, speculating);
    let teams = sizes
        .iter()
        .map(|&size| crate::team::Team::new(size))
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(AutoThreads {
        teams,
        tuner: std::sync::Mutex::new(crate::tune::Tuner::new(sizes, start)),
    })
}

/// The decode team for the coming step: the fixed team, or the candidate the
/// automatic tuner wants measured or has chosen.
struct DecodeTeam<'a> {
    runner: &'a Runner,
    index: usize,
    entered: Option<crate::team::Entered<'a>>,
}
impl<'a> DecodeTeam<'a> {
    fn new(runner: &'a Runner) -> Self {
        let mut team = Self {
            runner,
            index: usize::MAX,
            entered: None,
        };
        team.select();
        team
    }
    /// Enter the team for the next step (a no-op unless the tuner switches).
    fn select(&mut self) {
        let (index, team) = match &self.runner.decode {
            Decode::Auto(auto) => {
                let index = auto.tuner.lock().unwrap().current();
                (index, &auto.teams[index])
            }
            Decode::Fixed(team) => (0, team),
        };
        if index != self.index {
            self.entered = None;
            self.entered = team.enter();
            self.index = index;
        }
    }
    /// Report the single-row step just run on the selected team.
    fn record(&self, ms: f64) {
        if let Decode::Auto(auto) = &self.runner.decode {
            auto.tuner.lock().unwrap().record(self.index, ms);
        }
    }
    /// Report a draft-verification step (several rows) on the selected team.
    fn record_verify(&self, ms: f64) {
        if let Decode::Auto(auto) = &self.runner.decode {
            auto.tuner.lock().unwrap().record_verify(self.index, ms);
        }
    }
}

impl Runner {
    pub fn new(model: Arc<Model>, model_dir: impl AsRef<Path>, config: RunnerConfig) -> Result<Self> {
        config.validate()?;
        if model.profile() != crate::quant::Profile::REFERENCE {
            ensure!(
                config.cache_layout == CacheLayout::Compact,
                "quantized profiles require compact source caches"
            );
            ensure!(config.batch_size <= 8, "quantized profiles support 1..=8 active rows");
            ensure!(
                config.weight_layout == WeightLayout::Unpacked,
                "quantized profiles do not combine with phase-packed FP32 copies"
            );
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(config.threads).build()?;
        let tokenizer = OcrTokenizer::load(model_dir.as_ref())?;
        if config.weight_layout == WeightLayout::PhasePacked {
            model.prepare_phase_packed();
        }
        let pool_threads = pool.current_num_threads();
        let host = crate::auto::HostInfo::detect();
        let resolved = crate::auto::Resolved::new(host, &config, &model.facts(), model_dir.as_ref());
        let speculation = config
            .speculation
            .map(|s| (s.max_draft.min(crate::head_screen::MAX_ROWS - 1), s.min_match.max(1)));
        let decode = match config.decode_threads {
            DecodeThreads::Auto => Decode::Auto(auto_threads(host, pool_threads, speculation.is_some())?),
            DecodeThreads::Fixed(threads) => {
                ensure!(
                    (1..=pool_threads).contains(&threads),
                    "decode threads must be between 1 and the runner's thread count"
                );
                Decode::Fixed(crate::team::Team::new(threads)?)
            }
            DecodeThreads::Pool => Decode::Fixed(crate::team::Team::new(pool_threads)?),
        };
        let head = config.head;
        let draft_head = match (&config.draft_head, config.drafter) {
            (Some(path), crate::config::Drafter::Head | crate::config::Drafter::Both) => {
                Some(Arc::new(crate::draft_head::DraftHead::load(path)?))
            }
            _ => None,
        };
        let mut runner = Self {
            model,
            tokenizer,
            pool,
            head: HeadMode::Full,
            repetition_stop: config.repetition_stop,
            decode,
            speculation,
            document: config.document_drafts.then(Default::default),
            draft_head,
            config,
            resolved,
        };
        runner.set_head_mode(head)?;
        Ok(runner)
    }
    /// Whether generation stops on repetition loops (`RunnerConfig::repetition_stop`).
    pub fn repetition_stop(&self) -> bool {
        self.repetition_stop
    }
    fn document_history(&self) -> Option<std::sync::MutexGuard<'_, crate::draft::DocumentHistory>> {
        self.document
            .as_ref()
            .map(|d| d.lock().unwrap_or_else(|e| e.into_inner()))
    }
    /// The automatically chosen decode thread count, once tuning finished.
    pub fn decode_threads_chosen(&self) -> Option<usize> {
        match &self.decode {
            Decode::Auto(auto) => auto.tuner.lock().unwrap().chosen(),
            Decode::Fixed(_) => None,
        }
    }
    /// What the automatic decode-thread tuner measured and chose, once it has.
    pub fn decode_tuning(&self) -> Option<crate::tune::TuneReport> {
        match &self.decode {
            Decode::Auto(auto) => auto.tuner.lock().unwrap().report().cloned(),
            Decode::Fixed(_) => None,
        }
    }
    /// The tuner's report, the first time after it chose (one page carries it).
    fn take_decode_tuning(&self) -> Option<crate::tune::TuneReport> {
        match &self.decode {
            Decode::Auto(auto) => auto.tuner.lock().unwrap().take_report(),
            Decode::Fixed(_) => None,
        }
    }
    /// The decode team in use: the fixed size, or the tuner's choice once made.
    fn decode_threads_used(&self) -> Option<usize> {
        match &self.decode {
            Decode::Fixed(team) => Some(team.size()),
            Decode::Auto(_) => self.decode_threads_chosen(),
        }
    }
    /// `fp32` for the exact profiles, else the profile label.
    fn precision(&self) -> String {
        let profile = self.model.profile();
        if profile.is_exact() { "fp32" } else { profile.label() }.to_owned()
    }
    /// Choose how greedy decoding evaluates the vocabulary head. `Screened`
    /// builds and verifies an INT8 copy of the FP32 head once per `Model`
    /// (about 53 MB) and selects exactly the same tokens as `Full`.
    pub fn set_head_mode(&mut self, mode: HeadMode) -> Result<()> {
        if mode == HeadMode::Screened {
            let model = &self.model;
            self.pool.install(|| model.prepare_screened_head())?;
        }
        self.head = mode;
        self.resolved.head = mode;
        Ok(())
    }
    pub fn head_mode(&self) -> HeadMode {
        self.head
    }
    /// Execution settings are immutable because the owned thread pool and
    /// validated CPU feature selection are established when constructing Runner.
    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }
    /// What this runner does on this host: mode, weights, kernels, threads.
    pub fn resolved(&self) -> &crate::auto::Resolved {
        &self.resolved
    }
    pub fn recognize_file(&self, path: impl AsRef<Path>, options: &GenerationOptions) -> Result<OcrResult> {
        self.recognize_file_with_trace(path, options, &mut NoTrace)
    }
    pub fn recognize_file_with_trace(
        &self,
        path: impl AsRef<Path>,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        options.validate()?;
        let start = Instant::now();
        if options.route {
            let (source, decode_ms) = decode_file(path.as_ref())?;
            let first_at = |max| source.first_resize(options.min_dimension, max);
            return self.recognize_routed(&first_at, decode_ms, options, trace, start);
        }
        let (prepared, decode_ms) = prepare_file_timed(path.as_ref(), options.min_dimension, options.max_dimension)?;
        validate_prepared_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000. - decode_ms;
        let mut result = self.run_prepared_scoped(prepared, tokens, options, prep_ms, trace, &[])?;
        result.timings.image_decode_ms = decode_ms;
        result.timings.total_ms = start.elapsed().as_secs_f64() * 1000.;
        result.timings.time_to_first_token_ms += decode_ms;
        Ok(result)
    }
    /// Teacher-forced run over `teacher` (for example a reference run's
    /// token IDs): every step feeds the forced token, and a trace whose
    /// `scores_teacher` is true receives the greedy choice at each step.
    /// `options.max_new_tokens` must equal `teacher.len()`.
    pub fn score_teacher_file(
        &self,
        path: impl AsRef<Path>,
        teacher: &[u32],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        options.validate()?;
        ensure!(
            !teacher.is_empty() && options.max_new_tokens == teacher.len(),
            "teacher scoring needs max_new_tokens equal to the teacher length"
        );
        ensure!(!options.route, "teacher scoring needs a fixed max_dimension");
        let (prepared, _) = prepare_file_timed(path.as_ref(), options.min_dimension, options.max_dimension)?;
        validate_prepared_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        self.run_prepared_scoped(prepared, tokens, options, 0.0, trace, teacher)
    }
    pub fn recognize(&self, image: &RgbImage, options: &GenerationOptions) -> Result<OcrResult> {
        self.recognize_with_trace(image, options, &mut NoTrace)
    }
    pub fn recognize_with_trace(
        &self,
        image: &RgbImage,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        options.validate()?;
        let start = Instant::now();
        if options.route {
            let first_at = |max| first_resize_rgb(image, options.min_dimension, max);
            return self.recognize_routed(&first_at, 0., options, trace, start);
        }
        let prepared = prepare_rgb(image, options.min_dimension, options.max_dimension)?;
        validate_prepared_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut result = self.run_prepared_scoped(prepared, tokens, options, prep_ms, trace, &[])?;
        result.timings.total_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(result)
    }

    /// `--max-dimension auto`: route the page from its first resize at the
    /// cap and run it at the chosen size (exactly the input a fixed run of
    /// that size sees); the safety net reruns it at the cap if it loops or
    /// hits the length limit. `first_at(max)` is the first resize at `max`.
    fn recognize_routed(
        &self,
        first_at: &dyn Fn(u32) -> Result<RgbImage>,
        decode_ms: f64,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
        start: Instant,
    ) -> Result<OcrResult> {
        let (first, mut route, capped) = self.plan_route(first_at)?;
        let routed = self.recognize_first(&first, decode_ms, &options.at(route.max_dimension), trace, start)?;
        let mut result = self.safety_net(routed, &mut route, capped, options, trace)?;
        result.route = Some(route);
        Ok(result)
    }

    /// The router's choice for a page, the first resize at the chosen size,
    /// and (when that is below the cap) the capped page for the safety net.
    fn plan_route(&self, first_at: &dyn Fn(u32) -> Result<RgbImage>) -> Result<(RgbImage, Route, Option<RgbImage>)> {
        let capped = first_at(router::CAP)?;
        let route = self.pool.install(|| router::route(&capped));
        if route.max_dimension == router::CAP {
            return Ok((capped, route, None));
        }
        Ok((first_at(route.max_dimension)?, route, Some(capped)))
    }

    /// Rerun a routed page at the cap when it stopped by repetition or
    /// length; the rerun's `total_ms` includes the routed attempt.
    fn safety_net(
        &self,
        routed: OcrResult,
        route: &mut Route,
        capped: Option<RgbImage>,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        let Some(capped) = capped.filter(|_| router::needs_rerun(route, routed.finish_reason)) else {
            return Ok(routed);
        };
        route.safety_net = Some(RoutedAttempt {
            max_dimension: route.max_dimension,
            finish_reason: routed.finish_reason,
            output_tokens: routed.output_tokens,
            total_ms: routed.timings.total_ms,
        });
        let mut rerun = self.recognize_first(&capped, 0., &options.at(router::CAP), trace, Instant::now())?;
        rerun.timings.image_decode_ms = routed.timings.image_decode_ms;
        rerun.timings.total_ms += routed.timings.total_ms;
        Ok(rerun)
    }

    /// Prepare and run a first-resized page; timings count from `start`,
    /// which `decode_ms` of file decoding preceded.
    fn recognize_first(
        &self,
        first: &RgbImage,
        decode_ms: f64,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
        start: Instant,
    ) -> Result<OcrResult> {
        let prepared = prepare_first(first)?;
        validate_prepared_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000. - decode_ms;
        let mut result = self.run_prepared_scoped(prepared, tokens, options, prep_ms, trace, &[])?;
        result.timings.image_decode_ms = decode_ms;
        result.timings.total_ms = start.elapsed().as_secs_f64() * 1000.;
        result.timings.time_to_first_token_ms += decode_ms;
        Ok(result)
    }

    fn seal_session(&self, session: &mut Session, trace: &mut dyn Trace) -> Result<()> {
        let mode = self.model.profile().kv;
        if mode != crate::quant::Kv::Compact {
            let before = session.cache_bytes();
            let t = Instant::now();
            session.seal_prefix(&self.model.config, mode, self.config.exp)?;
            trace.prefix_sealed(before, session.cache_bytes(), t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(())
    }

    fn run_prepared_scoped(
        &self,
        prepared: PreparedImage,
        tokens: Vec<u32>,
        options: &GenerationOptions,
        prep_ms: f64,
        trace: &mut dyn Trace,
        teacher_tokens: &[u32],
    ) -> Result<OcrResult> {
        self.pool
            .install(|| self.run_prepared(prepared, tokens, options, prep_ms, trace, teacher_tokens))
    }
    fn run_prepared(
        &self,
        prepared: PreparedImage,
        tokens: Vec<u32>,
        options: &GenerationOptions,
        prep_ms: f64,
        trace: &mut dyn Trace,
        teacher_tokens: &[u32],
    ) -> Result<OcrResult> {
        let start = Instant::now();
        let c = &self.model.config;
        let budget = options.budget(tokens.len(), c.max_seq_len)?;
        let budget_clamped = budget != options.max_new_tokens;
        if budget_clamped {
            eprintln!(
                "max_new_tokens lowered to {budget}: {} input tokens leave no more room in the {}-token context",
                tokens.len(),
                c.max_seq_len
            );
        }
        let options = &GenerationOptions {
            max_new_tokens: budget,
            ..options.clone()
        };
        let (image_start, image_end) = image_range(&tokens, c)?;
        let (pos_t, pos_hw) = positions(&tokens, &prepared.positions_hw, c)?;
        let simd = self.config.backend.simd();
        let mut session = Session::new(
            c,
            tokens.len() + options.max_new_tokens,
            tokens.len(),
            image_start,
            image_end,
            simd,
            self.config.cache_layout,
            self.config.exp,
            self.config.tuning,
        )?;
        if let Some(head) = &self.draft_head
            && self.speculation.is_some()
            && teacher_tokens.is_empty()
            && !trace.enabled()
        {
            session.capture_features(&head.layers());
        }
        let mut hidden = Vec::new();
        let image_projection_ms = self.model.embed(&tokens, Some(&prepared.patches), simd, &mut hidden)?;
        let screen = self.head == HeadMode::Screened && teacher_tokens.is_empty();
        if screen {
            session.reserve_screened_head(&self.model);
        }
        let transformer_started = Instant::now();
        let next = self
            .model
            .forward_next(&mut hidden, &pos_t, &pos_hw, &mut session, trace, "prefill", screen)?;
        let transformer_prefill_ms = transformer_started.elapsed().as_secs_f64() * 1000.0;
        let score = trace.scores_teacher();
        let next_token = if let Some(&forced) = teacher_tokens.first() {
            if score {
                trace.teacher_step(0, forced, select(&next, 0, c.vocab_size)?);
                if let Next::Logits(logits) = &next {
                    trace.teacher_logits(0, &logits[..c.vocab_size]);
                }
            }
            forced
        } else {
            select(&next, 0, c.vocab_size)?
        };
        if (teacher_tokens.len() > 1 || !self.tokenizer.stop_ids().contains(&next_token)) && options.max_new_tokens > 1
        {
            self.seal_session(&mut session, trace)?;
        }
        if self.model.profile().memory_hygiene() {
            session.prepare_small_decode(c);
            hidden = Vec::with_capacity(c.dim);
            drop(prepared.patches);
            drop(prepared.positions_hw);
            drop(pos_t);
            drop(pos_hw);
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.;
        let decode_start = Instant::now();
        let stops = self.tokenizer.stop_ids();
        let stop_loops = self.repetition_stop && teacher_tokens.is_empty();
        let mut generation = Generation::new(options.max_new_tokens, stop_loops);
        let mut team = DecodeTeam::new(self);
        trace.decode_start();
        let speculate = self
            .speculation
            .filter(|_| teacher_tokens.is_empty() && !trace.enabled() && session.is_split());
        // Teacher forcing runs through its whole script; EOS never stops it.
        let eos: &[u32] = if teacher_tokens.is_empty() { &stops } else { &[] };
        let mut decode = DecodeLoop {
            session: &mut session,
            hidden: &mut hidden,
            generation: &mut generation,
            team: &mut team,
            trace: &mut *trace,
            stops: eos,
            screen,
            simd,
            next_token,
        };
        match speculate {
            Some((max_draft, min_match)) => self.decode_speculative(&mut decode, max_draft, min_match)?,
            None => self.decode_plain(&mut decode, teacher_tokens, score)?,
        }
        trace.decode_end();
        drop(team);
        crate::model::report_decode_phases(self.config.tuning.phases);
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.;
        self.finish_result(Page {
            tokens: generation.tokens,
            reason: generation.reason,
            width: prepared.width,
            height: prepared.height,
            input_tokens: tokens.len(),
            teacher_forced: !teacher_tokens.is_empty(),
            budget_clamped,
            timings: Timings {
                image_decode_ms: 0.,
                preprocessing_ms: prep_ms,
                image_projection_ms: Some(image_projection_ms),
                transformer_prefill_ms: Some(transformer_prefill_ms),
                prefill_ms,
                decode_ms,
                total_ms: prep_ms + start.elapsed().as_secs_f64() * 1000.,
                time_to_first_token_ms: prep_ms + prefill_ms,
            },
        })
    }

    /// Runs exported canonical patches/tokens, bypassing image decode/resize so the
    /// reference comparison can isolate model execution from image preparation.
    pub fn trace_reference(
        &self,
        fixture_path: impl AsRef<Path>,
        max_new_tokens: usize,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        use safetensors::{Dtype, SafeTensors};
        let bytes = std::fs::read(fixture_path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let tokens_tensor = tensors.tensor("tokens")?;
        let read_ids = |name: &str| -> Result<Vec<u32>> {
            let tensor = tensors.tensor(name)?;
            match tensor.dtype() {
                Dtype::I64 => tensor
                    .data()
                    .chunks_exact(8)
                    .map(|x| u32::try_from(i64::from_le_bytes(x.try_into().unwrap())).context("invalid token id"))
                    .collect(),
                Dtype::U32 => Ok(tensor
                    .data()
                    .chunks_exact(4)
                    .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                    .collect()),
                dtype => anyhow::bail!("unsupported {name} dtype {dtype:?}"),
            }
        };
        ensure!(tokens_tensor.shape().len() <= 2, "invalid token tensor");
        let tokens = read_ids("tokens")?;
        let patches = tensors.tensor("patches")?;
        ensure!(patches.dtype() == Dtype::F32, "expected F32 patches");
        let patches = patches
            .data()
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let positions_tensor = tensors.tensor("pos_hw")?;
        ensure!(positions_tensor.dtype() == Dtype::F32, "expected F32 positions");
        let all_positions = positions_tensor
            .data()
            .chunks_exact(8)
            .map(|x| {
                [
                    f32::from_le_bytes(x[..4].try_into().unwrap()),
                    f32::from_le_bytes(x[4..].try_into().unwrap()),
                ]
            })
            .collect::<Vec<_>>();
        ensure!(all_positions.len() == tokens.len(), "position/token count mismatch");
        let positions_hw = tokens
            .iter()
            .zip(all_positions)
            .filter_map(|(&token, pos)| (token == self.model.config.img_id).then_some(pos))
            .collect();
        let teacher_tokens = if tensors.names().contains(&"teacher_tokens") {
            read_ids("teacher_tokens")?
        } else {
            Vec::new()
        };
        ensure!(
            teacher_tokens.is_empty() || max_new_tokens <= teacher_tokens.len(),
            "same-prefix trace requested {max_new_tokens} outputs, but fixture has only {} teacher tokens",
            teacher_tokens.len()
        );
        let prepared = PreparedImage {
            width: 0,
            height: 0,
            patches,
            positions_hw,
        };
        let (computed_temporal, computed_spatial) = positions(&tokens, &prepared.positions_hw, &self.model.config)?;
        let reference_temporal = read_ids("pos_t")?;
        ensure!(
            computed_temporal.iter().map(|&x| x as u32).eq(reference_temporal),
            "temporal position parity failure"
        );
        let reference_spatial = tensors.tensor("pos_hw")?;
        ensure!(
            reference_spatial.shape() == [tokens.len(), 2],
            "expected [S,2] spatial coordinates"
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
        self.run_prepared_scoped(prepared, tokens, &options, 0., trace, &teacher_tokens)
    }
}

fn validate_prepared_bounds(prepared: &PreparedImage, options: &GenerationOptions) -> Result<()> {
    ensure!(
        prepared.width <= options.max_dimension as usize && prepared.height <= options.max_dimension as usize,
        "minimum-area alignment conflicts with requested bounds: prepared {}x{} exceeds max_dimension={}; explicitly increase max_dimension",
        prepared.width,
        prepared.height,
        options.max_dimension
    );
    Ok(())
}

fn argmax(logits: &[f32]) -> Result<u32> {
    ensure!(!logits.is_empty(), "nonfinite or empty logits");
    let mut best = 0;
    let mut finite = logits[0].is_finite();
    // Match torch.argmax's first-index tie break. One pass also checks
    // finiteness; any nonfinite value rejects the step exactly as before.
    for i in 1..logits.len() {
        finite &= logits[i].is_finite();
        if logits[i] > logits[best] {
            best = i;
        }
    }
    ensure!(finite, "nonfinite or empty logits");
    Ok(best as u32)
}

/// Greedy token for `row`: the screened head's exact choice, or the argmax of
/// that row's full FP32 logits.
fn select(next: &Next<'_>, row: usize, vocab: usize) -> Result<u32> {
    match next {
        Next::Tokens(tokens) => Ok(tokens[row]),
        Next::Logits(logits) => argmax(&logits[row * vocab..(row + 1) * vocab]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn argmax_handles_ties_and_rejects_nan() {
        assert_eq!(argmax(&[2., 3., 3.]).unwrap(), 1);
        assert!(argmax(&[f32::NAN]).is_err());
    }

    #[test]
    fn minimum_area_alignment_cannot_exceed_runtime_image_bound() {
        let options = GenerationOptions {
            min_dimension: 16,
            max_dimension: 32,
            max_new_tokens: 1,
            fit_budget: false,
            route: false,
        };
        options.validate().unwrap();
        let prepared = prepare_rgb(&RgbImage::new(33, 49), 16, 32).unwrap();
        assert!(prepared.width > 32 || prepared.height > 32);
        let error = validate_prepared_bounds(&prepared, &options).unwrap_err().to_string();
        assert!(error.contains("minimum-area alignment conflicts"));
        assert!(error.contains("max_dimension=32"));
        assert!(
            validate_prepared_bounds(
                &prepared,
                &GenerationOptions {
                    max_dimension: 128,
                    ..options
                }
            )
            .is_ok()
        );
    }
}
