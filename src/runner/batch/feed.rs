//! Where the rows get each page's prefill: read and prefilled as it starts
//! ([`Inline`]), or from the prefetch queue with the next page prefilling on
//! the second pool ([`Ahead`], `--pipeline`).
use std::{
    path::Path,
    sync::mpsc,
    thread::{Scope, ScopedJoinHandle},
};

use anyhow::{Context, Result, anyhow, ensure};
use rayon::ThreadPool;

use super::{super::cohort::PreparedPage, rows::Started};
use crate::{config::GenerationOptions, runner::Runner};

/// Where [`Rows`] gets each page's prefill.
pub(super) trait Feed {
    /// A prepared page.
    type Prepared;
    /// A prefilled page.
    type Item;
    /// Whether `page`'s prefill is done; may start it.
    fn ready(&mut self, page: usize) -> bool;
    /// `page` prefilled, waiting for its prefill if it runs elsewhere.
    fn take(&mut self, page: usize) -> Result<Self::Item>;
    /// `page`, which runs alone: prepared, or prefilled when its prefill has
    /// already started elsewhere.
    fn take_alone(&mut self, page: usize) -> Result<Alone<Self::Prepared, Self::Item>>;
}

/// A page that runs alone ([`Feed::take_alone`]).
pub(super) enum Alone<T, S> {
    /// Not prefilled: it runs as a single page does, drafts included.
    Prepared(T),
    /// Prefilled for a row (on the second pool): it decodes as a row.
    Prefilled(S),
}

/// Pages read and prefilled on the runner's pool as they start.
pub(super) struct Inline<'a, P> {
    pub(super) runner: &'a Runner,
    pub(super) paths: &'a [P],
    pub(super) options: &'a GenerationOptions,
}

impl<P: AsRef<Path>> Feed for Inline<'_, P> {
    type Prepared = PreparedPage;
    type Item = Started;
    fn ready(&mut self, _: usize) -> bool {
        true
    }
    fn take(&mut self, page: usize) -> Result<Started> {
        let prepared = self.runner.prepare_path(self.paths[page].as_ref(), self.options)?;
        self.runner
            .pool
            .install(|| self.runner.prefill_row(prepared, self.options))
    }
    fn take_alone(&mut self, page: usize) -> Result<Alone<PreparedPage, Started>> {
        let prepared = self.runner.prepare_path(self.paths[page].as_ref(), self.options)?;
        Ok(Alone::Prepared(prepared))
    }
}

/// The prefetch queue: pages in input order, prepared or failed.
pub(super) type Queue<T> = mpsc::Receiver<(usize, Result<T>)>;

/// A side prefill: the queue, handed back with the page's index and
/// prefill when it is done.
pub(super) type Side<'scope, T, S> = ScopedJoinHandle<'scope, (Queue<T>, usize, Result<S>)>;

/// Prepared pages from the prefetch queue (`--pipeline`), each prefilled
/// one page ahead on the second pool while the rows decode, or on the
/// runner's pool when it starts if there is no second pool (the decode
/// team's size is not settled yet, or the run has no second pool). Every
/// [`Feed::take`] consumes its own page's queue entry, whether the page,
/// its preparation or its prefill failed. `T` is a prepared page and `S` a
/// prefilled one; the tests use a mock prefill.
pub(super) struct Ahead<'scope, 'env, T, S> {
    scope: &'scope Scope<'scope, 'env>,
    /// Prefill a page on the pool this is called in.
    prefill: &'env (dyn Fn(T) -> Result<S> + Sync),
    /// The runner's pool, for a page that starts without a side prefill.
    main: &'env ThreadPool,
    /// The second pool, when there is one (`Runner::side_pool`).
    side: &'env dyn Fn() -> Option<&'env ThreadPool>,
    pages: usize,
    /// The queue while no side prefill holds it.
    queue: Option<Queue<T>>,
    /// The page prefilling on the second pool.
    running: Option<(usize, Side<'scope, T, S>)>,
}

impl<'scope, 'env, T: Send + 'env, S: Send + 'env> Ahead<'scope, 'env, T, S> {
    /// The feed of `pages` pages from `queue`, with nothing prefilling yet.
    pub(super) fn new(
        scope: &'scope Scope<'scope, 'env>,
        prefill: &'env (dyn Fn(T) -> Result<S> + Sync),
        main: &'env ThreadPool,
        side: &'env dyn Fn() -> Option<&'env ThreadPool>,
        pages: usize,
        queue: Queue<T>,
    ) -> Self {
        Self {
            scope,
            prefill,
            main,
            side,
            pages,
            queue: Some(queue),
            running: None,
        }
    }

    /// Start `page`'s prefill on the second pool, when there is one and
    /// nothing else prefills there.
    fn spawn(&mut self, page: usize) -> bool {
        if self.running.is_some() || page >= self.pages {
            return false;
        }
        let Some(pool) = (self.side)() else {
            return false;
        };
        let Some(queue) = self.queue.take() else {
            return false;
        };
        let prefill = self.prefill;
        let handle = self.scope.spawn(move || match queue.recv() {
            Ok((index, prepared)) => {
                let prefilled = prepared.and_then(|prepared| pool.install(|| prefill(prepared)));
                (queue, index, prefilled)
            }
            Err(_) => (queue, page, Err(anyhow!("the page reader stopped"))),
        });
        self.running = Some((page, handle));
        true
    }

    /// `page`'s entry of the queue, which no side prefill holds.
    fn receive(&self, page: usize) -> Result<T> {
        let queue = self.queue.as_ref().context("the page reader stopped")?;
        let (index, prepared) = queue.recv().context("the page reader stopped")?;
        ensure!(index == page, "the page reader sent page {index} for page {page}");
        prepared
    }
}

impl<'scope, 'env, T: Send + 'env, S: Send + 'env> Feed for Ahead<'scope, 'env, T, S> {
    type Prepared = T;
    type Item = S;

    fn ready(&mut self, page: usize) -> bool {
        match &self.running {
            Some((running, handle)) => *running != page || handle.is_finished(),
            // Nothing prefills ahead: start this page, or prefill it inline.
            None => !self.spawn(page),
        }
    }

    fn take(&mut self, page: usize) -> Result<S> {
        let prefilled = match self.running.take() {
            Some((running, handle)) => {
                let (queue, index, prefilled) = handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                self.queue = Some(queue);
                ensure!(
                    running == page && index == page,
                    "page {page} started while page {index} prefilled"
                );
                prefilled
            }
            None => self
                .receive(page)
                .and_then(|prepared| self.main.install(|| (self.prefill)(prepared))),
        };
        // Prefill the next page while this one decodes.
        self.spawn(page + 1);
        prefilled
    }

    fn take_alone(&mut self, page: usize) -> Result<Alone<T, S>> {
        if self.running.is_some() {
            return self.take(page).map(Alone::Prefilled);
        }
        self.receive(page).map(Alone::Prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::pipeline::PREFETCH;

    /// The pipelined feed with a mock prefill: every page is taken in order
    /// and fails only with its own error, whether the second pool is never
    /// there (as when it could not be built), always there, or comes and goes
    /// between pages. The last page starts alone: prepared, unless its
    /// prefill already started on the second pool.
    #[test]
    fn the_pipelined_feed_takes_every_page_in_order_with_or_without_a_second_pool() {
        let pool = |name: &'static str| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(move |_| name.to_owned())
                .build()
                .unwrap()
        };
        let (main, second) = (pool("main"), pool("second"));
        let places = std::sync::Mutex::new(Vec::new());
        // Page 3 fails to prepare and page 4 to prefill.
        let prefill = |page: usize| -> Result<usize> {
            places
                .lock()
                .unwrap()
                .push(std::thread::current().name().map(str::to_owned));
            ensure!(page != 4, "page 4 did not prefill");
            Ok(10 * page)
        };
        let pages = 7;
        let patterns: [fn(usize) -> bool; 3] = [|_| false, |_| true, |call| call % 3 != 1];
        for (pattern, available) in patterns.into_iter().enumerate() {
            places.lock().unwrap().clear();
            let calls = std::cell::Cell::new(0);
            let side = || {
                calls.set(calls.get() + 1);
                available(calls.get()).then_some(&second)
            };
            let (seen, ahead, lone) = std::thread::scope(|scope| {
                let (sender, queue) = mpsc::sync_channel(PREFETCH);
                scope.spawn(move || {
                    for page in 0..pages {
                        let prepared = if page == 3 {
                            Err(anyhow!("page 3 did not decode"))
                        } else {
                            Ok(page)
                        };
                        if sender.send((page, prepared)).is_err() {
                            break;
                        }
                    }
                });
                let mut feed = Ahead {
                    scope,
                    prefill: &prefill,
                    main: &main,
                    side: &side,
                    pages,
                    queue: Some(queue),
                    running: None,
                };
                let mut seen = Vec::new();
                for page in 0..pages - 1 {
                    // As `drive` asks while other rows decode: until ready.
                    while !feed.ready(page) {
                        std::thread::yield_now();
                    }
                    seen.push(feed.take(page).map_err(|error| error.to_string()));
                }
                // Started alone, as `drive` starts it: without asking.
                let ahead = feed.running.is_some();
                let lone = match feed.take_alone(pages - 1).unwrap() {
                    Alone::Prepared(page) => format!("prepared {page}"),
                    Alone::Prefilled(page) => format!("prefilled {page}"),
                };
                (seen, ahead, lone)
            });
            let expected: Vec<Result<usize, String>> = vec![
                Ok(0),
                Ok(10),
                Ok(20),
                Err("page 3 did not decode".to_owned()),
                Err("page 4 did not prefill".to_owned()),
                Ok(50),
            ];
            assert_eq!(seen, expected, "pattern {pattern}");
            assert_eq!(lone, if ahead { "prefilled 60" } else { "prepared 6" });
            let places = places.lock().unwrap();
            assert_eq!(places.len(), 5 + usize::from(ahead), "every prefilled page once");
            let on = |name: &str| places.iter().filter(|place| place.as_deref() == Some(name)).count();
            match pattern {
                0 => assert_eq!((on("main"), ahead), (5, false)),
                1 => assert_eq!((on("second"), ahead), (6, true)),
                _ => assert!(on("main") > 0 && on("second") > 0),
            }
        }
    }
}
