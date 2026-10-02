//! Continuous batching for `--batch-size N` (`Runner::recognize_files_streaming`
//! and `Runner::recognize_files_pipelined`): up to N pages decode in joint
//! steps, and a page that finishes leaves its row at once, releasing its
//! cache, for the next page, which joins with its own fitted output budget.
//! Results still reach the caller in input order. [`Scheduler`] makes the
//! decisions and [`drive`] runs them against an [`Engine`]: the runner's
//! pages here, a mock in the tests.
//!
//! Tokens are those of each page run alone, which the repository promises
//! for batches (`tests/batch_parity.rs`): each row of a joint step computes
//! that page's single-row step (per-row attention, per-row screened head,
//! multi-row dots bitwise equal to the single-row ones), so a page's tokens
//! do not depend on which pages share its steps or when it joined. The fixed
//! cohorts of `recognize_files`, `recognize_batch` and traced runs decode a
//! chunk to its smallest fitted budget; here every page keeps its own, as it
//! would alone.
//!
//! Rows decode without drafts. A page that starts with no other page
//! decoding or left to start (a single input, or the last page once the
//! others are done) runs as a single page does, drafts included, unless its
//! prefill has already run on the second pool (`--pipeline`): it then
//! decodes as a row on its own. Finished rows feed the document history in
//! the order they finish, so such a page drafts from every earlier page, as
//! in a sequential run, though the history may hold them in another order
//! (drafts never change tokens).
//!
//! Without the pipeline, a page is read and prefilled on the runner's pool
//! when a row frees, which pauses the other rows for that prefill. With it,
//! the prefetch thread reads and prepares the next pages, and the next page
//! prefills on the second pool while the rows decode (see `pipeline.rs` for
//! the pool's size and when it starts); the page joins as soon as a row is
//! free and its prefill is done, or at once when no row is decoding.
use std::{path::Path, sync::mpsc};

use anyhow::Result;

use super::{
    OcrResult, Runner,
    pipeline::{PREFETCH, Pipeline, SidePool},
    stream::Emitter,
};
use crate::config::GenerationOptions;
use feed::{Ahead, Inline};
use rows::Rows;
use schedule::drive;

mod feed;
mod rows;
mod schedule;

impl Runner {
    /// Continuous batching of `paths`, each page read and prefilled as it
    /// starts.
    pub(super) fn stream_rows<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        emit: &mut Emitter<'_, Result<OcrResult>>,
    ) {
        // Each call is one document for cross-page drafts.
        if let Some(mut history) = self.document_history() {
            history.clear();
        }
        let feed = Inline {
            runner: self,
            paths,
            options,
        };
        let _ = drive(
            &mut Rows::new(self, options, paths.len(), feed),
            self.config.batch_size,
            paths.len(),
            emit,
        );
    }

    /// Continuous batching of `paths` with the page pipeline: the prefetch
    /// thread prepares pages, and the next page prefills on the second pool
    /// while the rows decode.
    pub(super) fn stream_rows_pipelined<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        pipeline: &Pipeline,
        emit: &mut Emitter<'_, Result<OcrResult>>,
    ) {
        // Each call is one document for cross-page drafts.
        if let Some(mut history) = self.document_history() {
            history.clear();
        }
        let side_pool = SidePool::new();
        let side = || self.side_pool(&side_pool, pipeline);
        let prefill = |prepared| self.prefill_row(prepared, options);
        std::thread::scope(|scope| {
            let (sender, queue) = mpsc::sync_channel(PREFETCH);
            scope.spawn(move || self.read_pages(paths, options, sender));
            // The queue lives in the feed or in the side prefill that holds
            // it; both end with this scope's work, which ends the prefetch
            // thread even when the run stopped early.
            let feed = Ahead::new(scope, &prefill, &self.pool, &side, paths.len(), queue);
            let _ = drive(
                &mut Rows::new(self, options, paths.len(), feed),
                self.config.batch_size,
                paths.len(),
                emit,
            );
        });
    }
}
