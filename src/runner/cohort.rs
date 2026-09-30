//! Fixed-cohort batches (`Runner::recognize_batch`, `Runner::recognize_files`):
//! each chunk of `batch_size` pages is prepared and prefilled one page at a
//! time, then decodes in joint steps to the chunk's smallest fitted budget.
//! A chunk of one page takes the single-page path.
use super::{
    DecodeTeam, OcrResult, Runner, Timings, argmax,
    generate::{Generation, Page},
    select, validate_prepared_bounds,
};
use crate::{
    config::{GenerationOptions, HeadMode},
    model::{BatchWorkspace, Session, image_range, positions},
    preprocess::{PreparedImage, decode_file, first_resize_rgb, prepare_first},
    router::Route,
    trace::{NoTrace, PrefixedTrace, Trace},
};
use anyhow::{Context, Result};
use image::RgbImage;
use std::{path::Path, time::Instant};

impl Runner {
    /// Bounded batches with independent prefill and shared decode projections.
    /// Results retain input order. Batch size one uses the single-image path.
    pub fn recognize_batch(&self, images: &[RgbImage], options: &GenerationOptions) -> Result<Vec<OcrResult>> {
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
            let mut routes = Vec::with_capacity(chunk.len());
            for image in chunk {
                let prep_start = Instant::now();
                let first_at = |max| first_resize_rgb(image, options.min_dimension, max);
                let (prepared, route) = self.prepare_batch_page(&first_at, options)?;
                routes.push(route);
                let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
                inputs.push(BatchInput {
                    prepared,
                    tokens,
                    image_decode_ms: 0.,
                    preprocessing_ms: prep_start.elapsed().as_secs_f64() * 1000.,
                });
            }
            let outputs = self.pool.install(|| {
                self.run_batch_chunk(inputs, options, started, trace, chunk_index * self.config.batch_size)
            })?;
            for (output, route) in outputs.into_iter().zip(routes) {
                results.push(self.finish_batch_page(output, route, options, trace)?);
            }
        }
        Ok(results)
    }

    /// Batch PNG/JPEG files without losing source-mode resize semantics.
    pub fn recognize_files<P: AsRef<Path>>(&self, paths: &[P], options: &GenerationOptions) -> Result<Vec<OcrResult>> {
        self.recognize_files_with_trace(paths, options, &mut NoTrace)
    }
    pub fn recognize_files_with_trace<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<Vec<OcrResult>> {
        options.validate()?;
        // Each call is one document for cross-page drafts.
        if let Some(mut history) = self.document_history() {
            history.clear();
        }
        let mut results = Vec::with_capacity(paths.len());
        for (chunk_index, chunk) in paths.chunks(self.config.batch_size).enumerate() {
            if chunk.len() == 1 {
                results.push(self.recognize_file_with_trace(&chunk[0], options, trace)?);
                continue;
            }
            let started = Instant::now();
            let mut inputs = Vec::with_capacity(chunk.len());
            let mut routes = Vec::with_capacity(chunk.len());
            for path in chunk {
                let prep_start = Instant::now();
                let (source, image_decode_ms) =
                    decode_file(path.as_ref()).with_context(|| format!("preparing {}", path.as_ref().display()))?;
                let first_at = |max| source.first_resize(options.min_dimension, max);
                let (prepared, route) = self
                    .prepare_batch_page(&first_at, options)
                    .with_context(|| format!("preparing {}", path.as_ref().display()))?;
                routes.push(route);
                let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
                inputs.push(BatchInput {
                    prepared,
                    tokens,
                    image_decode_ms,
                    preprocessing_ms: prep_start.elapsed().as_secs_f64() * 1000. - image_decode_ms,
                });
            }
            let outputs = self.pool.install(|| {
                self.run_batch_chunk(inputs, options, started, trace, chunk_index * self.config.batch_size)
            })?;
            for (output, route) in outputs.into_iter().zip(routes) {
                results.push(self.finish_batch_page(output, route, options, trace)?);
            }
        }
        Ok(results)
    }

    /// A batched page's prepared input at its maximum dimension (routed when
    /// `options.route`), with the route and the capped page for the safety net.
    fn prepare_batch_page(
        &self,
        first_at: &dyn Fn(u32) -> Result<RgbImage>,
        options: &GenerationOptions,
    ) -> Result<(PreparedImage, Option<Planned>)> {
        let (first, route) = if options.route {
            let (first, route, capped) = self.plan_route(first_at)?;
            (first, Some((route, capped)))
        } else {
            (first_at(options.max_dimension)?, None)
        };
        let prepared = prepare_first(&first)?;
        let max_dimension = route.as_ref().map_or(options.max_dimension, |(r, _)| r.max_dimension);
        validate_prepared_bounds(&prepared, &options.at(max_dimension))?;
        Ok((prepared, route))
    }

    /// A batched page's result, after the safety net when it was routed.
    fn finish_batch_page(
        &self,
        output: OcrResult,
        route: Option<Planned>,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        let Some((mut route, capped)) = route else {
            return Ok(output);
        };
        let mut result = self.safety_net(output, &mut route, capped, options, trace)?;
        result.route = Some(route);
        Ok(result)
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
        // Validate every budget before expensive prefill or KV allocation. A
        // batch shares one step loop, so it runs to the smallest fitted budget.
        let mut budget = options.max_new_tokens;
        for input in &inputs {
            budget = budget.min(options.budget(input.tokens.len(), c.max_seq_len)?);
        }
        let budget_clamped = budget != options.max_new_tokens;
        if budget_clamped {
            eprintln!(
                "max_new_tokens lowered to {budget}: the longest input of {} tokens leaves no more room in the {}-token context",
                inputs.iter().map(|i| i.tokens.len()).max().unwrap_or(0),
                c.max_seq_len
            );
        }
        let options = &GenerationOptions {
            max_new_tokens: budget,
            ..options.clone()
        };
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
                self.config.exp,
                self.config.tuning,
            )?;
            if let Some(previous) = sessions.last_mut() {
                session.reuse_workspace_from(previous);
            }
            let image_projection_ms =
                self.model
                    .embed(&input.tokens, Some(&input.prepared.patches), simd, &mut hidden)?;
            let phase = if trace.enabled() {
                format!("request.{}.prefill", request_offset + index)
            } else {
                String::new()
            };
            let transformer_started = Instant::now();
            let logits = self
                .model
                .forward(&mut hidden, &pos_t, &pos_hw, &mut session, trace, &phase)?;
            let transformer_prefill_ms = transformer_started.elapsed().as_secs_f64() * 1000.0;
            let token = argmax(logits)?;
            if !stops.contains(&token) && options.max_new_tokens > 1 {
                self.seal_session(&mut session, trace)?;
            }
            let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.;
            let first_token_ms = chunk_started.elapsed().as_secs_f64() * 1000.;
            let mut generation = Generation::new(options.max_new_tokens, self.repetition_stop);
            let finished = generation.push(token, &stops);
            states.push(BatchState {
                width: input.prepared.width,
                height: input.prepared.height,
                input_tokens: input.tokens.len(),
                generation,
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
            if finished && self.model.profile().memory_hygiene() {
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
                tokens.push(states[index].generation.last());
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
            trace.decode_step(active.len(), step_ms, sessions.iter().map(Session::cache_bytes).sum());
            for (row, &index) in active.iter().enumerate() {
                let token = select(&next, row, c.vocab_size)?;
                let state = &mut states[index];
                state.finished = state.generation.push(token, &stops);
                if state.finished {
                    state.timings.decode_ms = decode_started.elapsed().as_secs_f64() * 1000.;
                    state.timings.total_ms = chunk_started.elapsed().as_secs_f64() * 1000.;
                    if self.model.profile().memory_hygiene() {
                        trace.cache_retired(sessions[index].retire_cache());
                    }
                }
            }
            active.retain(|&index| !states[index].finished);
            step += 1;
        }
        trace.decode_end();
        drop(team);
        crate::model::report_decode_phases(self.config.tuning.phases);
        states
            .into_iter()
            .map(|state| {
                self.finish_result(Page {
                    tokens: state.generation.tokens,
                    reason: state.generation.reason,
                    width: state.width,
                    height: state.height,
                    input_tokens: state.input_tokens,
                    teacher_forced: false,
                    budget_clamped,
                    timings: state.timings,
                })
            })
            .collect()
    }
}

/// A routed page's route and, when routed below the cap, the capped page
/// kept for the safety net.
type Planned = (Route, Option<RgbImage>);

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
    generation: Generation,
    finished: bool,
    timings: Timings,
}
