//! The page pipeline (`Runner::recognize_files_pipelined`, `falcon-ocr run
//! --pipeline`). A prefetch thread reads, decodes and prepares the next
//! pages (file read, image decode, first resize, routing, patches, prompt)
//! on a one-thread pool of its own, off the runner's pool, into a queue of
//! [`PREFETCH`] pages. While page k decodes on the decode team, page k+1's
//! prefill (embedding, forward pass, first token, sealing) runs on a second
//! pool on another thread; page k+1 decodes once page k's decode and its own
//! prefill are done.
//!
//! Tokens are those of a sequential run: the halves of a page are the one
//! sequential call split at the prefill/decode boundary, every prefill
//! stage computes each output independently of its pool's thread count, and
//! drafts only read the outputs of earlier pages, which are complete before
//! a page decodes. Pools of 1, 2, 3 and 7 threads agree bit for bit on one
//! quantized layer's stages (`model::pool_tests`: W8 on FP32 and BF16
//! panels and W16; QKV with folded row scales, rotary factors, the fused and
//! unfused QKV passes, FP32 and BF16 attention, the output projection and
//! gated FFN epilogues, sealing in every split format) and on the FP32
//! projections, attention, norms and first-token heads (`kernels::tests`).
//! The remaining steps are elementwise around those kernels (embedding
//! copies, dequantizing into the FP32 GEMM where panels are unavailable,
//! the FP32 gate); the unpanelled quantized path is not run in these tests.
//!
//! The second pool has the runner's threads minus the decode team unless
//! [`Pipeline::prefill_threads`] says otherwise, so the two stages use every
//! thread of the runner between them. An automatic decode team then takes at
//! most half of the runner's threads: the tuner still measures every candidate,
//! with no prefill beside it, and a choice above half becomes the largest
//! candidate within it. On an 8-core, 16-thread host the 8- and 12-thread teams
//! time within 4% of each other, and the 12-thread team leaves the prefill 4
//! threads instead of 8. The second pool is decided once the team size is
//! settled: with a fixed team at once (page 1's prefill overlaps page 0's
//! decode); with `DecodeThreads::Auto` after the tuner has chosen on the first
//! decode steps, which an overlapping prefill would disturb, so overlap starts
//! with the page decoding after the choice (page 1 when page 0 is long enough
//! to finish tuning). Until then, and for the whole run when the default would
//! leave the pool fewer than two threads (a one-thread prefill takes many times
//! longer than the decode it overlaps) or the pool cannot be built, pages
//! prefill on the runner's pool between decodes. The decode team's workers spin
//! between steps and park after 2 ms idle, so they neither lend their cores to
//! the second pool during a decode nor hold them after it; a prefill that
//! outlasts the decode it overlaps keeps only the second pool's threads busy
//! until it ends.
//!
//! While two pages overlap, both caches are alive: the decoding page's sealed
//! cache and the next page's prefill (its FP32 prefix cache and forward
//! workspace), on top of a sequential run's peak.
use std::{
    ops::ControlFlow,
    path::Path,
    sync::{Mutex, OnceLock, mpsc},
};

use anyhow::{Context, Result, ensure};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};

use super::{
    Decode, Decoder, OcrResult, Prefilled, Runner,
    cohort::{BatchInput, Planned, PreparedPage},
    stream::Emitter,
};
use crate::{config::GenerationOptions, trace::NoTrace};

/// Prepared pages the prefetch queue holds. With the page the prefetch
/// thread holds while the queue is full and the page prefilling on the
/// second pool, up to `PREFETCH + 2` prepared pages are in flight ahead of
/// the decoding ones.
pub(super) const PREFETCH: usize = 2;

/// How [`Runner::recognize_files_pipelined`] overlaps pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pipeline {
    /// Threads of the pool that prefills the next page while the current one
    /// decodes, at most the runner's threads; `None`: the runner's threads
    /// minus the decode team, and no overlap when that leaves fewer than
    /// two. With `None` an automatic decode team takes at most half of the
    /// runner's threads (see the module notes).
    pub prefill_threads: Option<usize>,
}

/// The second pool of a run, decided once the decode team's size is
/// settled: the pool, or `None` when there is none (pages then prefill
/// between decodes).
pub(super) type SidePool = OnceLock<Option<ThreadPool>>;

/// Threads of the second pool beside a `team`-thread decode team in a
/// `pool`-thread runner: `requested`, else what the team leaves, or `None`
/// (no overlap) when that default leaves fewer than two.
fn prefill_threads(requested: Option<usize>, pool: usize, team: usize) -> Option<usize> {
    requested.or(Some(pool.saturating_sub(team)).filter(|&left| left >= 2))
}

/// The prefetch queue: pages in input order, prepared or failed.
type Pages = Mutex<mpsc::Receiver<(usize, Result<PreparedPage>)>>;

/// The prefetch thread's end of the queue.
pub(super) type Prefetch = mpsc::SyncSender<(usize, Result<PreparedPage>)>;

/// A page after its prefill, with what its result still needs.
struct Staged {
    prefilled: Prefilled,
    route: Option<Planned>,
    image_decode_ms: f64,
}

impl Runner {
    /// [`Runner::recognize_files_streaming`] with the page pipeline: the next
    /// pages are read and prepared on a prefetch thread, and the next page's
    /// prefill overlaps the current page's decode on a second pool (see the
    /// module notes). Results, tokens and errors are those of
    /// `recognize_files_streaming`; `Timings::total_ms` of a page is the sum
    /// of its stages. With a batch size above 1 the rows refill as pages
    /// finish (`batch/`) and the next page prefills on the second pool
    /// while they decode. Never traced.
    pub fn recognize_files_pipelined<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        pipeline: &Pipeline,
        mut on_page: impl FnMut(usize, Result<OcrResult>) -> ControlFlow<()>,
    ) -> Result<()> {
        options.validate()?;
        let threads = self.pool.current_num_threads();
        ensure!(
            pipeline
                .prefill_threads
                .is_none_or(|prefill| (1..=threads).contains(&prefill)),
            "the page pipeline's prefill threads must be 1..={threads}, the runner's threads"
        );
        if pipeline.prefill_threads.is_none() {
            self.leave_half_to_prefill(threads);
        }
        let mut emit = Emitter::new(&mut on_page);
        if self.config.batch_size > 1 {
            self.stream_rows_pipelined(paths, options, pipeline, &mut emit);
            return Ok(());
        }
        // Each call is one document for cross-page drafts.
        if let Some(mut history) = self.document_history() {
            history.clear();
        }
        std::thread::scope(|scope| -> Result<()> {
            let (sender, receiver) = mpsc::sync_channel(PREFETCH);
            scope.spawn(move || self.read_pages(paths, options, sender));
            // Dropped when this closure returns, which ends the prefetch
            // thread even when the run stopped early.
            let pages: Pages = Mutex::new(receiver);
            let prefill_pool = SidePool::new();
            let mut ahead = None;
            for _ in 0..paths.len() {
                let (index, staged) = match ahead.take() {
                    Some(staged) => staged,
                    None => self.stage(&pages, &self.pool, options)?,
                };
                let staged = match staged {
                    Ok(staged) => staged,
                    Err(error) => {
                        if emit.finish(index, Err(error)).is_break() {
                            break;
                        }
                        continue;
                    }
                };
                let side = if index + 1 < paths.len() {
                    self.side_pool(&prefill_pool, pipeline)
                } else {
                    None
                };
                let flow = std::thread::scope(|overlap| -> Result<ControlFlow<()>> {
                    let next = side.map(|pool| overlap.spawn(|| self.stage(&pages, pool, options)));
                    let flow = emit.finish(index, self.finish_staged(staged, options));
                    if let Some(next) = next {
                        let staged = next.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
                        ahead = Some(staged);
                    }
                    Ok(flow)
                })?;
                if flow.is_break() {
                    break;
                }
            }
            Ok(())
        })
    }

    /// The prefetch thread of a pipelined run: read and prepare `paths` in
    /// order into `queue` until the run stops (the queue closes). It works
    /// on a one-thread pool of its own, so routing (`--max-dimension auto`),
    /// the one preparation step that uses a pool, stays off the runner's
    /// pool while its threads decode (`plan_route`); the route is the same.
    pub(super) fn read_pages<P: AsRef<Path> + Sync>(&self, paths: &[P], options: &GenerationOptions, queue: Prefetch) {
        let read = move || {
            for (index, path) in paths.iter().enumerate() {
                // A closed queue means the run stopped.
                if queue.send((index, self.prepare_path(path.as_ref(), options))).is_err() {
                    break;
                }
            }
        };
        match ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "falcon-reader".into())
            .build()
        {
            Ok(own) => own.install(read),
            // Pages then route on the runner's pool, as outside the pipeline.
            Err(_) => read(),
        }
    }

    /// Let an automatic decode team take at most half of the runner's
    /// `threads`, so that the second pool keeps at least the other half (see
    /// the module notes). The limit stays for the runner's later calls; it is reported
    /// once, when it drops a candidate.
    fn leave_half_to_prefill(&self, threads: usize) {
        if let Decode::Auto(auto) = &self.decode
            && let Some(largest) = auto.tuner.lock().unwrap().limit(threads / 2)
        {
            crate::note!(
                "pipeline: the automatic decode team uses at most {largest} of {threads} threads, so the next \
                 page's prefill keeps the rest"
            );
        }
    }

    /// The pool that prefills the next page beside the decode team: decided
    /// in `side` once the team's size is settled, and `None` before that or
    /// when there is no second pool ([`prefill_threads`], or the pool could
    /// not be built). The decision is made and reported once per run.
    pub(super) fn side_pool<'a>(&self, side: &'a SidePool, pipeline: &Pipeline) -> Option<&'a ThreadPool> {
        if let Some(decided) = side.get() {
            return decided.as_ref();
        }
        let team = self.decode_threads_used()?;
        let pool = self.pool.current_num_threads();
        let decided = match prefill_threads(pipeline.prefill_threads, pool, team) {
            None => {
                crate::note!(
                    "pipeline: the {team}-thread decode team leaves fewer than two of {pool} threads to prefill \
                     beside it; pages prefill between decodes (--prefill-threads N overlaps anyway)"
                );
                None
            }
            Some(threads) => match ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("falcon-prefill-{index}"))
                .build()
            {
                Ok(built) => {
                    crate::note!(
                        "pipeline: the next page prefills on a {threads}-thread pool beside the {team}-thread decode team"
                    );
                    Some(built)
                }
                Err(error) => {
                    crate::note!(
                        "warning: pipeline: the {threads}-thread prefill pool could not be built ({error}); \
                         pages prefill between decodes"
                    );
                    None
                }
            },
        };
        side.get_or_init(|| decided).as_ref()
    }

    /// Take the next prepared page from the queue and prefill it on `pool`.
    fn stage(&self, pages: &Pages, pool: &ThreadPool, options: &GenerationOptions) -> Result<(usize, Result<Staged>)> {
        let (index, page) = pages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .recv()
            .context("the page reader stopped")?;
        let staged = page.and_then(|page| {
            let page_options = page.options(options);
            let BatchInput {
                prepared,
                tokens,
                image_decode_ms,
                preprocessing_ms,
            } = page.input;
            let prefilled = pool.install(|| {
                self.prefill(
                    prepared,
                    tokens,
                    &page_options,
                    preprocessing_ms,
                    &mut NoTrace,
                    &[],
                    Decoder::Page,
                )
            })?;
            Ok(Staged {
                prefilled,
                route: page.route,
                image_decode_ms,
            })
        });
        Ok((index, staged))
    }

    /// Decode a staged page on the runner's pool, then finish it as a batch
    /// page does (the safety net when routed). Its timings are its stages.
    fn finish_staged(&self, staged: Staged, options: &GenerationOptions) -> Result<OcrResult> {
        let Staged {
            prefilled,
            route,
            image_decode_ms,
        } = staged;
        let mut result = self
            .pool
            .install(|| self.decode_prefilled(prefilled, &mut NoTrace, &[]))?;
        let timings = &mut result.timings;
        timings.image_decode_ms = image_decode_ms;
        timings.time_to_first_token_ms = image_decode_ms + timings.preprocessing_ms + timings.prefill_ms;
        timings.total_ms = timings.time_to_first_token_ms + timings.decode_ms;
        self.finish_batch_page(result, route, options, &mut NoTrace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_second_pool_takes_what_the_team_leaves_and_needs_two_threads_by_default() {
        assert_eq!(prefill_threads(None, 32, 12), Some(20));
        assert_eq!(prefill_threads(None, 16, 14), Some(2));
        // One thread left, none, or a team as large as the pool: no overlap.
        for (pool, team) in [(16, 15), (16, 16), (2, 2), (1, 1)] {
            assert_eq!(prefill_threads(None, pool, team), None, "{pool} {team}");
        }
        // An explicit size is kept, even a small one.
        assert_eq!(prefill_threads(Some(1), 16, 16), Some(1));
        assert_eq!(prefill_threads(Some(6), 32, 12), Some(6));
    }
}
