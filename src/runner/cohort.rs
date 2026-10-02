//! Fixed-cohort batches (`Runner::recognize_batch`, `Runner::recognize_files`
//! and its streaming form with a trace): each chunk of `batch_size` pages is
//! prepared and prefilled one page at a time, then decodes in joint steps to
//! the chunk's smallest fitted budget. A chunk of one page takes the
//! single-page path. Also the page preparation that every multi-page run
//! shares (`Runner::prepare_path`).
use super::{
    DecodeTeam, OcrResult, Runner, Timings, argmax,
    generate::{Generation, Page},
    require_uncropped, select,
    stream::Emitter,
    validate_prepared_bounds,
};
use crate::{
    config::{GenerationOptions, HeadMode},
    model::{BatchWorkspace, Session, image_range, positions},
    preprocess::{Crop, PreparedImage, decode_file, first_resize_rgb, prepare_first_cropped},
    router::Route,
    trace::{NoTrace, PrefixedTrace, Trace},
};
use anyhow::{Context, Result, anyhow};
use image::RgbImage;
use std::{ops::ControlFlow, path::Path, time::Instant};

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
            let first = chunk_index * self.config.batch_size;
            let requests: Vec<usize> = (first..first + chunk.len()).collect();
            let outputs = self
                .pool
                .install(|| self.run_batch_chunk(inputs, options, started, trace, &requests))?;
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
    /// Every page's result in input order, or the first error: pages run
    /// one at a time, or in fixed cohorts of the batch size that decode
    /// jointly. A cohort in which a page cannot be read or prepared stops
    /// before any model work with the first such error (`preparing <path>:
    /// ...`), as a cohort that does not fit the context does; otherwise the
    /// run stops at the first page, in input order, whose run failed, once its
    /// cohort has finished. [`Runner::recognize_files_streaming_with_trace`]
    /// runs the same cohorts but hands each page its own result or error and
    /// runs the rest of a cohort.
    pub fn recognize_files_with_trace<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<Vec<OcrResult>> {
        let mut results = Vec::with_capacity(paths.len());
        let mut failure = None;
        self.stream_cohorts(paths, options, trace, true, &mut |_, result| match result {
            Ok(result) => {
                results.push(result);
                ControlFlow::Continue(())
            }
            Err(error) => {
                failure = Some(error);
                ControlFlow::Break(())
            }
        })?;
        failure.map_or(Ok(results), Err)
    }

    /// Pages in fixed cohorts of the batch size (one at a time for batch
    /// size 1). With `fail_fast` (`recognize_files`), a cohort in which a page
    /// cannot be prepared stops before any model work and that error is
    /// returned as the run's; without it the page's error goes to `on_page`
    /// and the rest of the cohort runs. A page that runs alone is traced
    /// under its input index (`request.{i}`), as `recognize_batch_with_trace`
    /// traces one, so the pages of one trace keep their own tensors.
    pub(super) fn stream_cohorts<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
        fail_fast: bool,
        on_page: &mut dyn FnMut(usize, Result<OcrResult>) -> ControlFlow<()>,
    ) -> Result<()> {
        options.validate()?;
        // Each call is one document for cross-page drafts.
        if let Some(mut history) = self.document_history() {
            history.clear();
        }
        let mut emit = Emitter::new(on_page);
        for (chunk_index, chunk) in paths.chunks(self.config.batch_size).enumerate() {
            let first = chunk_index * self.config.batch_size;
            let flow = if chunk.len() == 1 {
                let prefix = format!("request.{first}");
                let mut scoped = PrefixedTrace::new(trace, &prefix);
                emit.finish(first, self.recognize_file_with_trace(&chunk[0], options, &mut scoped))
            } else {
                self.stream_chunk(chunk, first, options, trace, fail_fast, &mut emit)?
            };
            if flow.is_break() {
                break;
            }
        }
        Ok(())
    }

    /// One cohort of files (`first` is the index of its first page): each
    /// page is prepared and its output budget fitted on its own, so a page
    /// that cannot be read or prepared, or does not fit the context, leaves
    /// the cohort with its error (or, with `fail_fast`, ends the run with it
    /// before any model work), and the others decode jointly (alone when one
    /// is left, as a cohort of one does), traced under their input indices.
    fn stream_chunk<P: AsRef<Path>>(
        &self,
        chunk: &[P],
        first: usize,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
        fail_fast: bool,
        emit: &mut Emitter<'_, Result<OcrResult>>,
    ) -> Result<ControlFlow<()>> {
        let started = Instant::now();
        let mut inputs = Vec::with_capacity(chunk.len());
        let mut pages = Vec::with_capacity(chunk.len());
        for (offset, path) in chunk.iter().enumerate() {
            let path = path.as_ref();
            // A page's budget conflict is its own, the error it gets alone;
            // the cohort then decodes to the smallest budget of those left.
            let page = self
                .prepare_path(path, options)
                .with_context(|| format!("preparing {}", path.display()))
                .and_then(|page| {
                    options.budget(page.input.tokens.len(), self.model.config.max_seq_len)?;
                    Ok(page)
                });
            match page {
                Ok(page) => {
                    inputs.push(page.input);
                    pages.push((first + offset, page.route));
                }
                Err(error) if fail_fast => return Err(error),
                Err(error) => {
                    if emit.finish(first + offset, Err(error)).is_break() {
                        return Ok(ControlFlow::Break(()));
                    }
                }
            }
        }
        if inputs.len() == 1 {
            let (index, route) = pages.remove(0);
            let page = PreparedPage {
                input: inputs.remove(0),
                route,
            };
            let prefix = format!("request.{index}");
            let mut scoped = PrefixedTrace::new(trace, &prefix);
            return Ok(emit.finish(index, self.run_page(page, options, &mut scoped)));
        }
        if inputs.is_empty() {
            return Ok(ControlFlow::Continue(()));
        }
        let requests: Vec<usize> = pages.iter().map(|&(index, _)| index).collect();
        let outputs = self
            .pool
            .install(|| self.run_batch_chunk(inputs, options, started, trace, &requests));
        match outputs {
            Ok(outputs) => {
                for (output, (index, route)) in outputs.into_iter().zip(pages) {
                    let result = self.finish_batch_page(output, route, options, trace);
                    if emit.finish(index, result).is_break() {
                        return Ok(ControlFlow::Break(()));
                    }
                }
            }
            // The cohort's own failure (a trace refusing the crop, or its
            // prefill or joint decode) fails every page in it.
            Err(error) => {
                let message = format!("{error:#}");
                let mut error = Some(error);
                for (index, _) in pages {
                    let error = error.take().unwrap_or_else(|| anyhow!("{message}"));
                    if emit.finish(index, Err(error)).is_break() {
                        return Ok(ControlFlow::Break(()));
                    }
                }
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    /// A prepared page on its own, as `recognize_file` runs it (the safety
    /// net included when routed).
    pub(super) fn run_page(
        &self,
        page: PreparedPage,
        options: &GenerationOptions,
        trace: &mut dyn Trace,
    ) -> Result<OcrResult> {
        let page_options = page.options(options);
        let BatchInput {
            prepared,
            tokens,
            image_decode_ms,
            preprocessing_ms,
        } = page.input;
        let mut result = self.run_prepared_scoped(prepared, tokens, &page_options, preprocessing_ms, trace, &[])?;
        result.timings.image_decode_ms = image_decode_ms;
        result.timings.total_ms += image_decode_ms;
        result.timings.time_to_first_token_ms += image_decode_ms;
        self.finish_batch_page(result, page.route, options, trace)
    }

    /// Read, decode and prepare the page at `path` at its maximum dimension
    /// (routed when `options.route`), with its prompt: where a batched or
    /// pipelined page starts.
    pub(super) fn prepare_path(&self, path: &Path, options: &GenerationOptions) -> Result<PreparedPage> {
        let prep_start = Instant::now();
        let (source, image_decode_ms) = decode_file(path)?;
        let first_at = |max| source.first_resize(options.min_dimension, max);
        let (prepared, route) = self.prepare_batch_page(&first_at, options)?;
        let tokens = self.tokenizer.prompt(prepared.positions_hw.len())?;
        Ok(PreparedPage {
            input: BatchInput {
                prepared,
                tokens,
                image_decode_ms,
                preprocessing_ms: prep_start.elapsed().as_secs_f64() * 1000. - image_decode_ms,
            },
            route,
        })
    }

    /// A batched page's prepared input at its maximum dimension (routed when
    /// `options.route`), with the route and the capped page for the safety net.
    pub(super) fn prepare_batch_page(
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
        let prepared = prepare_first_cropped(&first, options.crop_margins)?;
        let max_dimension = route.as_ref().map_or(options.max_dimension, |(r, _)| r.max_dimension);
        validate_prepared_bounds(&prepared, &options.at(max_dimension))?;
        Ok((prepared, route))
    }

    /// A batched page's result, after the safety net when it was routed.
    pub(super) fn finish_batch_page(
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

    /// Prefill a cohort's pages one after another on the calling thread's
    /// pool, then decode them in joint steps to the smallest fitted budget;
    /// results in input order. `requests` holds each page's input index,
    /// which names it in traces: its prefill is `request.{i}.prefill`, and
    /// the cohort's decode is `batch.{i}` with its first page's index.
    pub(super) fn run_batch_chunk(
        &self,
        inputs: Vec<BatchInput>,
        options: &GenerationOptions,
        chunk_started: Instant,
        trace: &mut dyn Trace,
        requests: &[usize],
    ) -> Result<Vec<OcrResult>> {
        debug_assert_eq!(requests.len(), inputs.len());
        if trace.enabled() {
            require_uncropped(options, "a tensor trace")?;
        }
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
            crate::note!(
                "max_new_tokens lowered to {budget}: the longest input of {} tokens leaves no more room in the {}-token context",
                inputs.iter().map(|i| i.tokens.len()).max().unwrap_or(0),
                c.max_seq_len
            );
        }
        let options = &GenerationOptions {
            max_new_tokens: budget,
            ..options.clone()
        };
        for (input, request) in inputs.into_iter().zip(requests) {
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
                format!("request.{request}.prefill")
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
                crop: input.prepared.crop,
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
                format!("batch.{}.decode.{step}", requests[0])
            } else {
                String::new()
            };
            if trace.enabled() {
                let ids = active.iter().map(|&index| requests[index] as f32).collect::<Vec<_>>();
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
                    crop: state.crop,
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
pub(super) type Planned = (Route, Option<RgbImage>);

/// A prepared page's model input: its patches, its prompt and the time its
/// preparation took.
pub(super) struct BatchInput {
    pub(super) prepared: PreparedImage,
    pub(super) tokens: Vec<u32>,
    pub(super) image_decode_ms: f64,
    pub(super) preprocessing_ms: f64,
}

/// A page read and prepared for the model (`Runner::prepare_path`): its
/// input at the maximum dimension it runs at, and its route when routed.
pub(super) struct PreparedPage {
    pub(super) input: BatchInput,
    pub(super) route: Option<Planned>,
}
impl PreparedPage {
    /// `options` at the maximum dimension this page runs at.
    pub(super) fn options(&self, options: &GenerationOptions) -> GenerationOptions {
        options.at(self
            .route
            .as_ref()
            .map_or(options.max_dimension, |(route, _)| route.max_dimension))
    }
}
struct BatchState {
    width: usize,
    height: usize,
    crop: Option<Crop>,
    input_tokens: usize,
    generation: Generation,
    finished: bool,
    timings: Timings,
}
