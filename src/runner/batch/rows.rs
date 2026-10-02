//! The runner's [`Engine`]: rows of pages decoded jointly, each prefilled
//! as a single page is and finished with its own result.
use std::time::Instant;

use anyhow::{Context, Result};

use super::{
    super::{
        DecodeTeam, Decoder, OcrResult, Prefilled, Runner, Timings,
        cohort::{BatchInput, Planned, PreparedPage},
        generate::{Generation, Page},
        select,
    },
    feed::{Alone, Feed},
    schedule::Engine,
};
use crate::{
    config::{GenerationOptions, HeadMode},
    model::{BatchWorkspace, Session},
    preprocess::Crop,
    trace::NoTrace,
};

/// A page prefilled for a row (`Runner::prefill_row`).
pub(super) struct Started {
    session: Session,
    generation: Generation,
    /// It ended at its first token (a stop token, or a budget of one).
    finished: bool,
    route: Option<Planned>,
    width: usize,
    height: usize,
    crop: Option<Crop>,
    input_tokens: usize,
    budget_clamped: bool,
    /// The preparation and prefill stages.
    timings: Timings,
}

/// A page in its row: everything but its session, which [`Rows`] keeps
/// with the others for the joint step.
pub(super) struct Row {
    session: usize,
    generation: Generation,
    route: Option<Planned>,
    width: usize,
    height: usize,
    crop: Option<Crop>,
    input_tokens: usize,
    budget_clamped: bool,
    timings: Timings,
    /// When its first joint step started.
    decoding: Option<Instant>,
}

/// The runner's [`Engine`]: rows of pages from `feed`, decoded jointly on
/// the runner's pool and decode team.
pub(super) struct Rows<'r, F> {
    runner: &'r Runner,
    options: &'r GenerationOptions,
    feed: F,
    stops: Vec<u32>,
    screen: bool,
    /// Every started page's session, in start order; a finished page's
    /// cache is released and its empty session stays so indices hold.
    sessions: Vec<Session>,
    /// Each page's row while it decodes.
    rows: Vec<Option<Row>>,
    batch: BatchWorkspace,
    /// Scratch of one step: the fed tokens, the sessions, the selections.
    tokens: Vec<u32>,
    indices: Vec<usize>,
    selected: Vec<u32>,
    /// The result of the page that ran alone.
    alone: Option<(usize, OcrResult)>,
}

impl<'r, F: Feed<Prepared = PreparedPage, Item = Started>> Rows<'r, F> {
    /// Rows for `pages` pages from `feed`, as many as the runner's batch size,
    /// with the joint step's workspace reserved.
    pub(super) fn new(runner: &'r Runner, options: &'r GenerationOptions, pages: usize, feed: F) -> Self {
        let rows = runner.config.batch_size;
        let c = &runner.model.config;
        let screen = runner.head == HeadMode::Screened;
        let mut batch = BatchWorkspace::new(rows, c);
        if screen {
            batch.reserve_screened_head(&runner.model, rows);
        }
        Self {
            runner,
            options,
            feed,
            stops: runner.tokenizer.stop_ids(),
            screen,
            sessions: Vec::with_capacity(pages),
            rows: (0..pages).map(|_| None).collect(),
            batch,
            tokens: Vec::with_capacity(rows),
            indices: Vec::with_capacity(rows),
            selected: Vec::with_capacity(rows),
            alone: None,
        }
    }
}

impl<F: Feed<Prepared = PreparedPage, Item = Started>> Engine for Rows<'_, F> {
    type Output = OcrResult;

    fn ready(&mut self, page: usize) -> bool {
        self.feed.ready(page)
    }

    fn start(&mut self, page: usize, alone: bool) -> Result<bool> {
        let started = if alone {
            match self.feed.take_alone(page)? {
                // Nothing would share its steps: it runs as a single page
                // does, drafts included (its tokens are the same either way).
                Alone::Prepared(prepared) => {
                    let result = self.runner.run_page(prepared, self.options, &mut NoTrace)?;
                    self.alone = Some((page, result));
                    return Ok(false);
                }
                Alone::Prefilled(started) => started,
            }
        } else {
            self.feed.take(page)?
        };
        self.sessions.push(started.session);
        self.rows[page] = Some(Row {
            session: self.sessions.len() - 1,
            generation: started.generation,
            route: started.route,
            width: started.width,
            height: started.height,
            crop: started.crop,
            input_tokens: started.input_tokens,
            budget_clamped: started.budget_clamped,
            timings: started.timings,
            decoding: None,
        });
        Ok(!started.finished)
    }

    fn step(&mut self, pages: &[usize], finished: &mut Vec<usize>) -> Result<()> {
        let now = Instant::now();
        self.tokens.clear();
        self.indices.clear();
        for &page in pages {
            let row = self.rows[page].as_mut().context("a decoding page has a row")?;
            row.decoding.get_or_insert(now);
            self.tokens.push(row.generation.last());
            self.indices.push(row.session);
        }
        let runner = self.runner;
        let (tokens, indices, sessions, batch, selected) = (
            &self.tokens,
            &self.indices,
            &mut self.sessions,
            &mut self.batch,
            &mut self.selected,
        );
        let screen = self.screen;
        // On the runner's pool with the decode team entered for this step,
        // as a single page's decode runs.
        runner.pool.install(|| -> Result<()> {
            let team = DecodeTeam::new(runner);
            let started = Instant::now();
            let next = runner.model.decode_batch_next(
                tokens,
                indices,
                sessions,
                batch,
                runner.config.weight_layout,
                &mut NoTrace,
                "",
                screen,
            )?;
            team.record(started.elapsed().as_secs_f64() * 1000.0);
            selected.clear();
            for row in 0..tokens.len() {
                selected.push(select(&next, row, runner.model.config.vocab_size)?);
            }
            Ok(())
        })?;
        for (&page, &token) in pages.iter().zip(&self.selected) {
            let row = self.rows[page].as_mut().context("a decoding page has a row")?;
            if row.generation.push(token, &self.stops) {
                finished.push(page);
            }
        }
        Ok(())
    }

    fn finish(&mut self, page: usize) -> Result<OcrResult> {
        if let Some((_, result)) = self.alone.take_if(|(alone, _)| *alone == page) {
            return Ok(result);
        }
        let row = self.rows[page].take().context("a finished page has a row")?;
        self.sessions[row.session].retire_cache();
        // Later pages draft from it, as in a sequential run: a page that runs
        // alone drafts.
        if self.runner.speculation.is_some()
            && let Some(mut history) = self.runner.document_history()
        {
            history.push_page(&row.generation.tokens);
        }
        let mut timings = row.timings;
        timings.decode_ms = row
            .decoding
            .map_or(0.0, |decoding| decoding.elapsed().as_secs_f64() * 1000.0);
        timings.time_to_first_token_ms = timings.image_decode_ms + timings.preprocessing_ms + timings.prefill_ms;
        timings.total_ms = timings.time_to_first_token_ms + timings.decode_ms;
        let result = self.runner.finish_result(Page {
            tokens: row.generation.tokens,
            reason: row.generation.reason,
            width: row.width,
            height: row.height,
            crop: row.crop,
            input_tokens: row.input_tokens,
            teacher_forced: false,
            budget_clamped: row.budget_clamped,
            timings,
        })?;
        self.runner
            .finish_batch_page(result, row.route, self.options, &mut NoTrace)
    }

    fn abandon(&mut self, page: usize) {
        if let Some(row) = self.rows[page].take() {
            self.sessions[row.session].retire_cache();
        }
    }
}

impl Runner {
    /// Prefill a prepared page for a row as a single page is prefilled
    /// (`Runner::prefill`), with the page's own fitted budget, on the calling
    /// thread's pool. The prefill workspace is released at once, and a page
    /// that ended at its first token releases its cache.
    pub(super) fn prefill_row(&self, page: PreparedPage, options: &GenerationOptions) -> Result<Started> {
        let page_options = page.options(options);
        let PreparedPage { input, route } = page;
        let BatchInput {
            prepared,
            tokens,
            image_decode_ms,
            preprocessing_ms,
        } = input;
        let prefilled = self.prefill(
            prepared,
            tokens,
            &page_options,
            preprocessing_ms,
            &mut NoTrace,
            &[],
            Decoder::Row,
        )?;
        let Prefilled {
            mut session,
            next_token,
            max_new_tokens,
            budget_clamped,
            input_tokens,
            retained: (prepared, ..),
            image_projection_ms,
            transformer_prefill_ms,
            prefill_ms,
            ..
        } = prefilled;
        let mut generation = Generation::new(max_new_tokens, self.repetition_stop);
        let finished = generation.push(next_token, &self.tokenizer.stop_ids());
        if finished {
            session.retire_cache();
        }
        Ok(Started {
            session,
            generation,
            finished,
            route,
            width: prepared.width,
            height: prepared.height,
            crop: prepared.crop,
            input_tokens,
            budget_clamped,
            timings: Timings {
                image_decode_ms,
                preprocessing_ms,
                image_projection_ms: Some(image_projection_ms),
                transformer_prefill_ms: Some(transformer_prefill_ms),
                prefill_ms,
                decode_ms: 0.0,
                total_ms: 0.0,
                time_to_first_token_ms: 0.0,
            },
        })
    }
}
