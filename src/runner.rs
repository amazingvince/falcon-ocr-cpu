use crate::{
    config::{CacheLayout, GenerationOptions, HeadMode, RunnerConfig, WeightLayout},
    model::{BatchWorkspace, Model, Next, Session, image_range, positions},
    preprocess::{PreparedImage, prepare_file_timed, prepare_rgb},
    tokenizer::OcrTokenizer,
    trace::{NoTrace, PrefixedTrace, Trace},
};
use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Eos,
    Length,
    /// Stopped by the opt-in repetition stop (`Runner::set_repetition_stop`).
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
    pub precision: String,
    pub backend: String,
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
    pub timings: Timings,
}

fn legacy_cache_layout() -> CacheLayout {
    CacheLayout::Expanded
}

/// `--decode-threads`: a fixed count, or `auto` ([`Runner::set_decode_threads_auto`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeThreads {
    Auto,
    Fixed(usize),
}
impl std::str::FromStr for DecodeThreads {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        match value.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Self::Fixed(n)),
            _ => Err(format!("expected `auto` or a positive thread count, got `{value}`")),
        }
    }
}
impl std::fmt::Display for DecodeThreads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Fixed(n) => write!(f, "{n}"),
        }
    }
}

pub struct Runner {
    model: Arc<Model>,
    tokenizer: OcrTokenizer,
    pool: rayon::ThreadPool,
    config: RunnerConfig,
    head: HeadMode,
    /// Opt-in stop for degenerate repetition loops (`crate::repetition`).
    repetition_stop: bool,
    /// Spin-waiting workers for decode steps; prefill keeps using `pool`.
    team: crate::team::Team,
    /// Automatic decode thread count (`set_decode_threads_auto`): one team
    /// per candidate size and the tuner choosing between them.
    auto_threads: Option<AutoThreads>,
    /// Speculative decoding (`set_speculation`): at most this many drafted
    /// tokens per step and the minimum n-gram match; `None` is off.
    speculation: Option<(usize, usize)>,
}

struct AutoThreads {
    teams: Vec<crate::team::Team>,
    tuner: std::sync::Mutex<crate::tune::Tuner>,
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
        let (index, team) = match &self.runner.auto_threads {
            Some(auto) => {
                let index = auto.tuner.lock().unwrap().current();
                (index, &auto.teams[index])
            }
            None => (0, &self.runner.team),
        };
        if index != self.index {
            self.entered = None;
            self.entered = team.enter();
            self.index = index;
        }
    }
    /// Report the step just run on the selected team.
    fn record(&self, ms: f64) {
        if let Some(auto) = &self.runner.auto_threads {
            auto.tuner.lock().unwrap().record(self.index, ms);
        }
    }
}

impl Runner {
    pub fn new(
        model: Arc<Model>,
        model_dir: impl AsRef<Path>,
        config: RunnerConfig,
    ) -> Result<Self> {
        config.validate()?;
        if model.attempt_profile() != crate::attempt::Profile::Reference {
            ensure!(
                config.cache_layout == CacheLayout::Compact,
                "attempt profiles require compact source caches"
            );
            ensure!(
                config.batch_size <= 8,
                "attempt profiles support 1..=8 active rows"
            );
            ensure!(
                config.weight_layout == WeightLayout::Unpacked,
                "attempt profiles do not combine with phase-packed FP32 copies"
            );
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()?;
        let tokenizer = OcrTokenizer::load(model_dir.as_ref())?;
        if config.weight_layout == WeightLayout::PhasePacked {
            model.prepare_phase_packed();
        }
        let team = crate::team::Team::new(pool.current_num_threads())?;
        Ok(Self {
            model,
            tokenizer,
            pool,
            config,
            head: HeadMode::Full,
            repetition_stop: false,
            team,
            auto_threads: None,
            speculation: None,
        })
    }
    /// Stop greedy generation once it repeats a cycle of at most 128 tokens
    /// for at least `max(256, 4 * cycle)` tokens (`FinishReason::Repetition`).
    /// Output up to that step is unchanged. Off by default; never applied to
    /// teacher-forced traces.
    pub fn set_repetition_stop(&mut self, enabled: bool) {
        self.repetition_stop = enabled;
    }
    pub fn repetition_stop(&self) -> bool {
        self.repetition_stop
    }
    /// Size the decode spin team separately from the prefill pool. Prefill is
    /// compute-bound and gains from SMT siblings; decode is memory-bound and
    /// loses when both siblings of a core stream (on a 16-core/32-thread
    /// Ryzen 7950X: prefill 32 threads 22% faster, decode 32 threads 60%
    /// slower than 16). `threads` must be between 1 and the pool size.
    pub fn set_decode_threads(&mut self, threads: usize) -> Result<()> {
        ensure!(
            (1..=self.pool.current_num_threads()).contains(&threads),
            "decode threads must be between 1 and the runner's thread count"
        );
        self.team = crate::team::Team::new(threads)?;
        self.auto_threads = None;
        Ok(())
    }
    /// Choose the decode thread count automatically: the first decode steps
    /// of this runner time a few team sizes (`crate::tune`) and keep the
    /// smallest one within 2% of the fastest. Tokens never depend on it.
    pub fn set_decode_threads_auto(&mut self) -> Result<()> {
        let sizes = crate::tune::candidate_sizes(
            crate::cpu::physical_cores(),
            crate::cpu::logical_cpus(),
            self.pool.current_num_threads(),
        );
        let physical = crate::cpu::physical_cores().min(self.pool.current_num_threads());
        let start = sizes.iter().position(|&s| s >= physical).unwrap_or(sizes.len() - 1);
        let teams = sizes
            .iter()
            .map(|&size| crate::team::Team::new(size))
            .collect::<std::io::Result<Vec<_>>>()?;
        self.auto_threads = Some(AutoThreads {
            teams,
            tuner: std::sync::Mutex::new(crate::tune::Tuner::new(sizes, start)),
        });
        Ok(())
    }
    /// Speculative decoding for single-page generation with the attempt
    /// profiles' split cache: up to `max_draft` tokens (at most 7) drafted
    /// from earlier output (`crate::draft`, n-gram match of at least
    /// `min_match` tokens) are verified in one multi-row step. Every row is
    /// bitwise the single-row step, so tokens are unchanged; `max_draft = 0`
    /// turns it off. Teacher forcing, tracing and batches never speculate.
    pub fn set_speculation(&mut self, max_draft: usize, min_match: usize) {
        self.speculation = (max_draft > 0).then_some((
            max_draft.min(crate::head_screen::MAX_ROWS - 1),
            min_match.max(1),
        ));
    }
    /// The automatically chosen decode thread count, once tuning finished.
    pub fn decode_threads_chosen(&self) -> Option<usize> {
        self.auto_threads
            .as_ref()
            .and_then(|auto| auto.tuner.lock().unwrap().chosen())
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
    pub fn recognize_file(
        &self,
        path: impl AsRef<Path>,
        options: &GenerationOptions,
    ) -> Result<OcrResult> {
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
        let (prepared, decode_ms) =
            prepare_file_timed(path.as_ref(), options.min_dimension, options.max_dimension)?;
        validate_prepared_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000. - decode_ms;
        let mut result =
            self.run_prepared_scoped(prepared, tokens, options, prep_ms, trace, &[])?;
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
        let (prepared, _) =
            prepare_file_timed(path.as_ref(), options.min_dimension, options.max_dimension)?;
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
        let prepared = prepare_rgb(image, options.min_dimension, options.max_dimension)?;
        validate_prepared_bounds(&prepared, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        let prep_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut result =
            self.run_prepared_scoped(prepared, tokens, options, prep_ms, trace, &[])?;
        result.timings.total_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(result)
    }
    /// Bounded batches with independent prefill and shared decode projections.
    /// Results retain input order. Batch size one uses the single-image path.
    pub fn recognize_batch(
        &self,
        images: &[RgbImage],
        options: &GenerationOptions,
    ) -> Result<Vec<OcrResult>> {
        self.recognize_batch_with_trace(images, options, &mut NoTrace)
    }

    /// Trace phase boundaries cover each chunk's shared decode loop; tensor
    /// captures identify request prefills and compacted batch decode rows.
    pub fn recognize_batch_with_trace(
        &self,
        images: &[RgbImage],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<Vec<OcrResult>> {
        options.validate()?;
        let mut results = Vec::with_capacity(images.len());
        for (chunk_index, chunk) in images.chunks(self.config.batch_size).enumerate() {
            if chunk.len() == 1 {
                let prefix = format!("request.{}", chunk_index * self.config.batch_size);
                let mut scoped = PrefixedTrace::new(trace, &prefix);
                results.push(self.recognize_with_trace(&chunk[0], options, &mut scoped)?);
                continue;
            }
            let started = Instant::now();
            let mut inputs = Vec::with_capacity(chunk.len());
            for image in chunk {
                let prep_start = Instant::now();
                let prepared = prepare_rgb(image, options.min_dimension, options.max_dimension)?;
                validate_prepared_bounds(&prepared, options)?;
                let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
                inputs.push(BatchInput {
                    prepared,
                    tokens,
                    image_decode_ms: 0.,
                    preprocessing_ms: prep_start.elapsed().as_secs_f64() * 1000.,
                });
            }
            let mut outputs = self.pool.install(|| {
                self.run_batch_chunk(
                    inputs,
                    options,
                    started,
                    trace,
                    chunk_index * self.config.batch_size,
                )
            })?;
            results.append(&mut outputs);
        }
        Ok(results)
    }

    /// Batch PNG/JPEG files without losing source-mode resize semantics.
    pub fn recognize_files<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
    ) -> Result<Vec<OcrResult>> {
        self.recognize_files_with_trace(paths, options, &mut NoTrace)
    }
    pub fn recognize_files_with_trace<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<Vec<OcrResult>> {
        options.validate()?;
        let mut results = Vec::with_capacity(paths.len());
        for (chunk_index, chunk) in paths.chunks(self.config.batch_size).enumerate() {
            if chunk.len() == 1 {
                results.push(self.recognize_file_with_trace(&chunk[0], options, trace)?);
                continue;
            }
            let started = Instant::now();
            let mut inputs = Vec::with_capacity(chunk.len());
            for path in chunk {
                let prep_start = Instant::now();
                let (prepared, image_decode_ms) =
                    prepare_file_timed(path.as_ref(), options.min_dimension, options.max_dimension)
                        .with_context(|| format!("preparing {}", path.as_ref().display()))?;
                validate_prepared_bounds(&prepared, options)?;
                let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
                inputs.push(BatchInput {
                    prepared,
                    tokens,
                    image_decode_ms,
                    preprocessing_ms: prep_start.elapsed().as_secs_f64() * 1000. - image_decode_ms,
                });
            }
            let mut outputs = self.pool.install(|| {
                self.run_batch_chunk(
                    inputs,
                    options,
                    started,
                    trace,
                    chunk_index * self.config.batch_size,
                )
            })?;
            results.append(&mut outputs);
        }
        Ok(results)
    }

    fn seal_for_attempt(&self, session: &mut Session, trace: &mut dyn Trace) -> Result<()> {
        let mode = self.model.attempt_profile().prefix_mode();
        if mode != crate::attempt::PrefixMode::Reference {
            let before = session.cache_bytes();
            let t = Instant::now();
            session.seal_prefix(&self.model.config, mode)?;
            trace.prefix_sealed(
                before,
                session.cache_bytes(),
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }

    fn run_batch_chunk(
        &self,
        inputs: Vec<BatchInput>,
        options: &GenerationOptions,
        chunk_started: Instant,
        trace: &mut dyn Trace,
        request_offset: usize,
    ) -> Result<Vec<OcrResult>> {
        let c = &self.model.config;
        let simd = self.config.backend.simd();
        let stops = self.tokenizer.stop_ids();
        let count = inputs.len();
        let mut sessions = Vec::with_capacity(count);
        let mut states = Vec::with_capacity(count);
        let mut hidden = Vec::new();
        // Validate every budget before expensive prefill or KV allocation.
        for input in &inputs {
            options.check_budget(input.tokens.len(), c.max_seq_len)?;
        }
        for (index, input) in inputs.into_iter().enumerate() {
            let prefill_started = Instant::now();
            let (image_start, image_end) = image_range(&input.tokens, c)?;
            let (pos_t, pos_hw) = positions(&input.tokens, &input.prepared.positions_hw, c)?;
            let mut session = Session::new(
                c,
                input.tokens.len() + options.max_new_tokens,
                input.tokens.len(),
                image_start,
                image_end,
                simd,
                self.config.cache_layout,
            )?;
            if let Some(previous) = sessions.last_mut() {
                session.reuse_workspace_from(previous);
            }
            let image_projection_ms = self.model.embed(
                &input.tokens,
                Some(&input.prepared.patches),
                simd,
                &mut hidden,
            )?;
            let phase = if trace.enabled() {
                format!("request.{}.prefill", request_offset + index)
            } else {
                String::new()
            };
            let transformer_started = Instant::now();
            let logits =
                self.model
                    .forward(&mut hidden, &pos_t, &pos_hw, &mut session, trace, &phase)?;
            let transformer_prefill_ms = transformer_started.elapsed().as_secs_f64() * 1000.0;
            let token = argmax(logits)?;
            if !stops.contains(&token) && options.max_new_tokens > 1 {
                self.seal_for_attempt(&mut session, trace)?;
            }
            let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.;
            let first_token_ms = chunk_started.elapsed().as_secs_f64() * 1000.;
            let mut generated = Vec::with_capacity(options.max_new_tokens);
            generated.push(token);
            let reason = if stops.contains(&token) {
                FinishReason::Eos
            } else {
                FinishReason::Length
            };
            let finished = reason == FinishReason::Eos || options.max_new_tokens == 1;
            states.push(BatchState {
                width: input.prepared.width,
                height: input.prepared.height,
                input_tokens: input.tokens.len(),
                generated,
                repetition: crate::repetition::RepetitionStop::new(),
                reason,
                finished,
                timings: Timings {
                    image_decode_ms: input.image_decode_ms,
                    preprocessing_ms: input.preprocessing_ms,
                    image_projection_ms: Some(image_projection_ms),
                    transformer_prefill_ms: Some(transformer_prefill_ms),
                    prefill_ms,
                    decode_ms: 0.,
                    time_to_first_token_ms: first_token_ms,
                    total_ms: if finished { first_token_ms } else { 0. },
                },
            });
            if finished && self.model.attempt_profile().memory_hygiene() {
                trace.cache_retired(session.retire_cache());
            }
            sessions.push(session);
        }
        // All earlier sessions handed their scratch to the next prefill. Joint
        // decode has its own small workspace and no longer needs this allocation
        // or the largest prefix's hidden-state buffer.
        if let Some(last) = sessions.last_mut() {
            last.release_workspace();
        }
        drop(hidden);
        let mut active = Vec::with_capacity(count);
        for (index, state) in states.iter().enumerate() {
            if !state.finished {
                active.push(index);
            }
        }
        let mut tokens = Vec::with_capacity(count);
        let mut batch = BatchWorkspace::new(active.len(), c);
        let screen = self.head == HeadMode::Screened;
        if screen {
            batch.reserve_screened_head(&self.model, active.len());
        }
        let decode_started = Instant::now();
        let mut step = 0;
        let mut team = DecodeTeam::new(self);
        trace.decode_start();
        while !active.is_empty() {
            tokens.clear();
            for &index in &active {
                tokens.push(*states[index].generated.last().unwrap());
            }
            let phase = if trace.enabled() {
                format!("batch.{request_offset}.decode.{step}")
            } else {
                String::new()
            };
            if trace.enabled() {
                let ids = active
                    .iter()
                    .map(|&index| (request_offset + index) as f32)
                    .collect::<Vec<_>>();
                trace.tensor(&format!("{phase}.request_indices"), &[active.len()], &ids)?;
            }
            team.select();
            let step_started = Instant::now();
            let next = self.model.decode_batch_next(
                &tokens,
                &active,
                &mut sessions,
                &mut batch,
                self.config.weight_layout,
                trace,
                &phase,
                screen,
            )?;
            let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            team.record(step_ms);
            trace.decode_step(
                active.len(),
                step_ms,
                sessions.iter().map(Session::cache_bytes).sum(),
            );
            for (row, &index) in active.iter().enumerate() {
                let token = select(&next, row, c.vocab_size)?;
                let state = &mut states[index];
                state.generated.push(token);
                if stops.contains(&token) {
                    state.reason = FinishReason::Eos;
                    state.finished = true;
                } else if self.repetition_stop && state.repetition.push(&state.generated) {
                    state.reason = FinishReason::Repetition;
                    state.finished = true;
                }
                if state.generated.len() == options.max_new_tokens {
                    state.finished = true;
                }
                if state.finished {
                    state.timings.decode_ms = decode_started.elapsed().as_secs_f64() * 1000.;
                    state.timings.total_ms = chunk_started.elapsed().as_secs_f64() * 1000.;
                    if self.model.attempt_profile().memory_hygiene() {
                        trace.cache_retired(sessions[index].retire_cache());
                    }
                }
            }
            active.retain(|&index| !states[index].finished);
            step += 1;
        }
        trace.decode_end();
        drop(team);
        crate::model::report_decode_phases();
        states
            .into_iter()
            .map(|state| {
                let text = self.tokenizer.decode(&state.generated)?;
                Ok(OcrResult {
                    text,
                    output_tokens: state.generated.len(),
                    token_ids: state.generated,
                    finish_reason: state.reason,
                    width: state.width,
                    height: state.height,
                    input_tokens: state.input_tokens,
                    precision: if self.model.attempt_profile().is_exact() {
                        "fp32".into()
                    } else {
                        format!("attempt3/{}", self.model.attempt_profile().label())
                    },
                    backend: format!("rust-gemm/{:?}", simd.resolved()).to_lowercase(),
                    cache_layout: self.config.cache_layout,
                    weight_layout: self.config.weight_layout,
                    packed_weight_bytes: self.model.packed_weight_bytes(),
                    weight_packing_ms: self.model.weight_packing_ms(),
                    teacher_forced: false,
                    timings: state.timings,
                })
            })
            .collect()
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
        self.pool.install(|| {
            self.run_prepared(prepared, tokens, options, prep_ms, trace, teacher_tokens)
        })
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
        options.check_budget(tokens.len(), c.max_seq_len)?;
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
        )?;
        let mut hidden = Vec::new();
        let image_projection_ms =
            self.model
                .embed(&tokens, Some(&prepared.patches), simd, &mut hidden)?;
        let screen = self.head == HeadMode::Screened && teacher_tokens.is_empty();
        if screen {
            session.reserve_screened_head(&self.model);
        }
        let transformer_started = Instant::now();
        let next = self.model.forward_next(
            &mut hidden,
            &pos_t,
            &pos_hw,
            &mut session,
            trace,
            "prefill",
            screen,
        )?;
        let transformer_prefill_ms = transformer_started.elapsed().as_secs_f64() * 1000.0;
        let score = trace.scores_teacher();
        let mut next_token = if let Some(&forced) = teacher_tokens.first() {
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
        if (teacher_tokens.len() > 1 || !self.tokenizer.stop_ids().contains(&next_token))
            && options.max_new_tokens > 1
        {
            self.seal_for_attempt(&mut session, trace)?;
        }
        if self.model.attempt_profile().memory_hygiene() {
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
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut reason = FinishReason::Length;
        let mut repetition = crate::repetition::RepetitionStop::new();
        let stop_loops = self.repetition_stop && teacher_tokens.is_empty();
        let mut team = DecodeTeam::new(self);
        trace.decode_start();
        let speculate = self
            .speculation
            .filter(|_| teacher_tokens.is_empty() && !trace.enabled() && session.is_split());
        if let Some((max_draft, min_match)) = speculate {
            let mut drafter = crate::draft::NgramDrafter::new(min_match);
            let mut draft = Vec::with_capacity(max_draft);
            let mut inputs = Vec::with_capacity(max_draft + 1);
            let mut positions = Vec::with_capacity(max_draft + 1);
            // (single steps, their ms, verify steps, their ms, drafted, accepted)
            let mut stats = (0_usize, 0.0_f64, 0_usize, 0.0_f64, 0_usize, 0_usize);
            let mut policy = crate::draft::DraftPolicy::new();
            'decode: loop {
                let token = next_token;
                generated.push(token);
                if stops.contains(&token) {
                    reason = FinishReason::Eos;
                    break;
                }
                if stop_loops && repetition.push(&generated) {
                    reason = FinishReason::Repetition;
                    break;
                }
                if generated.len() == options.max_new_tokens {
                    break;
                }
                // Accepted drafts plus the next token stay within the budget
                // and the cache.
                let limit = max_draft
                    .min(options.max_new_tokens - generated.len() - 1)
                    .min(session.remaining_capacity().saturating_sub(1));
                if policy.should_draft() {
                    drafter.propose(&generated, limit, &mut draft);
                } else {
                    draft.clear();
                }
                team.select();
                let step_started = Instant::now();
                if draft.is_empty() {
                    self.model.embed(&[token], None, simd, &mut hidden)?;
                    let next = self.model.forward_next(
                        &mut hidden,
                        &[session.next_position],
                        &[[f32::NAN; 2]],
                        &mut session,
                        trace,
                        "",
                        screen,
                    )?;
                    let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
                    team.record(step_ms);
                    stats.0 += 1;
                    stats.1 += step_ms;
                    policy.single_step(step_ms);
                    next_token = select(&next, 0, c.vocab_size)?;
                    trace.decode_step(1, step_ms, session.cache_bytes());
                    continue;
                }
                inputs.clear();
                inputs.push(token);
                inputs.extend_from_slice(&draft);
                let rows = inputs.len();
                positions.clear();
                positions.extend((0..rows).map(|r| session.next_position + r));
                let kept = session.len;
                self.model.embed(&inputs, None, simd, &mut hidden)?;
                let next = self
                    .model
                    .verify_next(&mut hidden, &positions, &mut session, screen)?;
                // predicted[r]: the greedy token after inputs[..=r].
                let mut predicted = [0_u32; crate::head_screen::MAX_ROWS];
                for (r, slot) in predicted[..rows].iter_mut().enumerate() {
                    *slot = select(&next, r, c.vocab_size)?;
                }
                let accepted = draft
                    .iter()
                    .zip(&predicted)
                    .take_while(|(d, p)| d == p)
                    .count();
                session.truncate(kept + 1 + accepted)?;
                let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
                stats.2 += 1;
                stats.3 += step_ms;
                stats.4 += draft.len();
                stats.5 += accepted;
                policy.verify_step(step_ms, draft.len(), accepted);
                trace.decode_step(rows, step_ms, session.cache_bytes());
                for &token in &draft[..accepted] {
                    generated.push(token);
                    if stops.contains(&token) {
                        reason = FinishReason::Eos;
                        break 'decode;
                    }
                    if stop_loops && repetition.push(&generated) {
                        reason = FinishReason::Repetition;
                        break 'decode;
                    }
                }
                next_token = predicted[accepted];
            }
            if std::env::var_os("FALCON_OCR_PHASES").is_some() {
                let (singles, single_ms, verifies, verify_ms, drafted, accepted) = stats;
                eprintln!(
                    "speculation: {} tokens; {singles} single steps ({:.2} ms), {verifies} verify steps ({:.2} ms),                      drafted {drafted}, accepted {accepted} ({:.0}%)",
                    generated.len(),
                    single_ms / singles.max(1) as f64,
                    verify_ms / verifies.max(1) as f64,
                    100.0 * accepted as f64 / drafted.max(1) as f64,
                );
            }
        }
        for step in 0..options.max_new_tokens {
            if speculate.is_some() {
                break;
            }
            let token = next_token;
            generated.push(token);
            if stops.contains(&token) && teacher_tokens.is_empty() {
                reason = FinishReason::Eos;
                break;
            }
            if stop_loops && repetition.push(&generated) {
                reason = FinishReason::Repetition;
                break;
            }
            if step + 1 == options.max_new_tokens {
                break;
            }
            team.select();
            let step_started = Instant::now();
            self.model.embed(&[token], None, simd, &mut hidden)?;
            let phase = if trace.enabled() {
                format!("decode.{step}")
            } else {
                String::new()
            };
            let next = self.model.forward_next(
                &mut hidden,
                &[session.next_position],
                &[[f32::NAN; 2]],
                &mut session,
                trace,
                &phase,
                screen,
            )?;
            let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            team.record(step_ms);
            next_token = if let Some(&forced) = teacher_tokens.get(step + 1) {
                if score {
                    trace.teacher_step(step + 1, forced, select(&next, 0, c.vocab_size)?);
                    if let Next::Logits(logits) = &next {
                        trace.teacher_logits(step + 1, &logits[..c.vocab_size]);
                    }
                }
                forced
            } else {
                select(&next, 0, c.vocab_size)?
            };
            trace.decode_step(1, step_ms, session.cache_bytes());
        }
        trace.decode_end();
        drop(team);
        crate::model::report_decode_phases();
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.;
        let text = self.tokenizer.decode(&generated)?;
        Ok(OcrResult {
            text,
            output_tokens: generated.len(),
            token_ids: generated,
            finish_reason: reason,
            width: prepared.width,
            height: prepared.height,
            input_tokens: tokens.len(),
            precision: if self.model.attempt_profile().is_exact() {
                "fp32".into()
            } else {
                format!("attempt3/{}", self.model.attempt_profile().label())
            },
            backend: format!("rust-gemm/{:?}", simd.resolved()).to_lowercase(),
            cache_layout: self.config.cache_layout,
            weight_layout: self.config.weight_layout,
            packed_weight_bytes: self.model.packed_weight_bytes(),
            weight_packing_ms: self.model.weight_packing_ms(),
            teacher_forced: !teacher_tokens.is_empty(),
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
                    .map(|x| {
                        u32::try_from(i64::from_le_bytes(x.try_into().unwrap()))
                            .context("invalid token id")
                    })
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
        ensure!(
            positions_tensor.dtype() == Dtype::F32,
            "expected F32 positions"
        );
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
        ensure!(
            all_positions.len() == tokens.len(),
            "position/token count mismatch"
        );
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
        let (computed_temporal, computed_spatial) =
            positions(&tokens, &prepared.positions_hw, &self.model.config)?;
        let reference_temporal = read_ids("pos_t")?;
        ensure!(
            computed_temporal
                .iter()
                .map(|&x| x as u32)
                .eq(reference_temporal),
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

struct BatchInput {
    prepared: PreparedImage,
    tokens: Vec<u32>,
    image_decode_ms: f64,
    preprocessing_ms: f64,
}
struct BatchState {
    width: usize,
    height: usize,
    input_tokens: usize,
    generated: Vec<u32>,
    repetition: crate::repetition::RepetitionStop,
    reason: FinishReason,
    finished: bool,
    timings: Timings,
}

fn validate_prepared_bounds(prepared: &PreparedImage, options: &GenerationOptions) -> Result<()> {
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
        };
        options.validate().unwrap();
        let prepared = prepare_rgb(&RgbImage::new(33, 49), 16, 32).unwrap();
        assert!(prepared.width > 32 || prepared.height > 32);
        let error = validate_prepared_bounds(&prepared, &options)
            .unwrap_err()
            .to_string();
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
