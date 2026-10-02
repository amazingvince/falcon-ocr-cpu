//! The decisions of continuous batching ([`Scheduler`]) and the loop that
//! runs them against an [`Engine`] ([`drive`]), without model work.
use std::ops::ControlFlow;

use anyhow::{Result, anyhow};

use super::super::stream::Emitter;

/// The decisions of continuous batching, without model work: pages start in
/// input order while fewer than `rows` decode; a page whose prefill is not
/// ready waits while others decode, but not when nothing else would run;
/// a page that finishes or fails frees its row at once. A page that starts
/// with no other page decoding or left to start is alone for its whole run.
pub(crate) struct Scheduler {
    rows: usize,
    pages: usize,
    /// The next page to start.
    next: usize,
    /// The pages decoding, in the order they started.
    active: Vec<usize>,
}

impl Scheduler {
    /// `pages` pages over `rows` rows (at least one), none started.
    pub(crate) fn new(rows: usize, pages: usize) -> Self {
        Self {
            rows: rows.max(1),
            pages,
            next: 0,
            active: Vec::with_capacity(rows),
        }
    }

    /// The next page to start, if a row is free and pages remain, and
    /// `ready(page)` says its prefill is done or no other page is decoding.
    /// It counts as decoding until [`Scheduler::finish`].
    pub(crate) fn admit(&mut self, ready: impl FnOnce(usize) -> bool) -> Option<usize> {
        if self.active.len() >= self.rows || self.next >= self.pages {
            return None;
        }
        if !self.active.is_empty() && !ready(self.next) {
            return None;
        }
        self.active.push(self.next);
        self.next += 1;
        Some(self.next - 1)
    }

    /// The page that started last is the only one decoding, and no page is
    /// left to start: nothing will join it.
    pub(crate) fn alone(&self) -> bool {
        self.active.len() == 1 && self.next >= self.pages
    }

    /// The pages of the next joint step, in the order they started.
    pub(crate) fn active(&self) -> &[usize] {
        &self.active
    }

    /// `page` finished or failed: its row is free.
    pub(crate) fn finish(&mut self, page: usize) {
        self.active.retain(|&active| active != page);
    }

    /// Every page has started and finished.
    pub(crate) fn done(&self) -> bool {
        self.next >= self.pages && self.active.is_empty()
    }
}

/// The model work that [`drive`] schedules.
pub(crate) trait Engine {
    type Output;
    /// Whether `page`'s prefill is done (always, when pages prefill as they
    /// start); asked only while other pages decode.
    fn ready(&mut self, page: usize) -> bool;
    /// Start `page`: `Ok(true)` while it decodes in a row, `Ok(false)` when
    /// it has finished, at its first token or, when `alone` (nothing else
    /// decodes or is left to start), run to its end on its own.
    fn start(&mut self, page: usize, alone: bool) -> Result<bool>;
    /// One joint decode step of `pages`, pushing those that finished.
    fn step(&mut self, pages: &[usize], finished: &mut Vec<usize>) -> Result<()>;
    /// The result of `page`, which finished; its cache is released.
    fn finish(&mut self, page: usize) -> Result<Self::Output>;
    /// Release `page`, which failed.
    fn abandon(&mut self, page: usize);
}

/// Run `pages` pages on `engine`, at most `rows` at a time, handing each
/// result to `emit` in input order. A page whose prefill fails hands over
/// its error; a joint step that fails fails every page in it. Admissions
/// wait while a failure is buffered behind an earlier page, until the
/// callback receives it and decides whether to continue.
pub(crate) fn drive<E: Engine>(
    engine: &mut E,
    rows: usize,
    pages: usize,
    emit: &mut Emitter<'_, Result<E::Output>>,
) -> ControlFlow<()> {
    let mut scheduler = Scheduler::new(rows, pages);
    let (mut active, mut finished) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    while !scheduler.done() {
        while !emit.has_pending_error()
            && let Some(page) = scheduler.admit(|page| engine.ready(page))
        {
            let result = match engine.start(page, scheduler.alone()) {
                Ok(true) => continue,
                Ok(false) => engine.finish(page),
                Err(error) => {
                    engine.abandon(page);
                    Err(error)
                }
            };
            scheduler.finish(page);
            emit.finish(page, result)?;
        }
        active.clear();
        active.extend_from_slice(scheduler.active());
        if active.is_empty() {
            continue;
        }
        finished.clear();
        match engine.step(&active, &mut finished) {
            Ok(()) => {
                for &page in &finished {
                    scheduler.finish(page);
                    let result = engine.finish(page);
                    emit.finish(page, result)?;
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                let mut error = Some(error);
                for &page in &active {
                    scheduler.finish(page);
                    engine.abandon(page);
                    emit.finish(page, Err(error.take().unwrap_or_else(|| anyhow!("{message}"))))?;
                }
            }
        }
    }
    ControlFlow::Continue(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;

    /// Pages that decode a scripted number of tokens.
    #[derive(Default)]
    struct Mock {
        /// Tokens each page decodes after its first (0: it ends at its first
        /// token); `None`: its prefill fails.
        lengths: Vec<Option<usize>>,
        /// Steps before each page's prefill is ready (0: at once).
        ready_at: Vec<usize>,
        /// The joint step that fails (1-based), if any.
        fail_step: Option<usize>,
        /// The page whose finished result fails.
        fail_finish: Option<usize>,
        steps: usize,
        decoded: Vec<usize>,
        /// Pages admitted, with the number of steps already taken.
        started: Vec<(usize, usize)>,
        /// Every step's pages.
        log: Vec<Vec<usize>>,
        abandoned: Vec<usize>,
        /// The pages that started alone (and ran to their end at once).
        alone: Vec<usize>,
    }

    impl Mock {
        fn new(lengths: Vec<Option<usize>>) -> Self {
            Self {
                ready_at: vec![0; lengths.len()],
                decoded: vec![0; lengths.len()],
                lengths,
                ..Self::default()
            }
        }
    }

    impl Engine for Mock {
        type Output = usize;
        fn ready(&mut self, page: usize) -> bool {
            self.steps >= self.ready_at[page]
        }
        fn start(&mut self, page: usize, alone: bool) -> Result<bool> {
            self.started.push((page, self.steps));
            if alone {
                self.alone.push(page);
            }
            match self.lengths[page] {
                None => bail!("page {page} did not prefill"),
                Some(length) if alone => {
                    self.decoded[page] = length;
                    Ok(false)
                }
                Some(length) => Ok(length > 0),
            }
        }
        fn step(&mut self, pages: &[usize], finished: &mut Vec<usize>) -> Result<()> {
            self.steps += 1;
            self.log.push(pages.to_vec());
            if self.fail_step == Some(self.steps) {
                bail!("step {} failed", self.steps);
            }
            for &page in pages {
                self.decoded[page] += 1;
                if Some(self.decoded[page]) == self.lengths[page] {
                    finished.push(page);
                }
            }
            Ok(())
        }
        fn finish(&mut self, page: usize) -> Result<usize> {
            if self.fail_finish == Some(page) {
                bail!("page {page} did not finish");
            }
            Ok(self.decoded[page])
        }
        fn abandon(&mut self, page: usize) {
            self.abandoned.push(page);
        }
    }

    /// `drive` over `mock` with `rows` rows: how it ended, and every page's
    /// result in the order they reached the caller.
    fn run(mock: &mut Mock, rows: usize, stop_at: Option<usize>) -> (ControlFlow<()>, Vec<(usize, String)>) {
        let pages = mock.lengths.len();
        let mut seen = Vec::new();
        let mut on_page = |page: usize, result: Result<usize>| {
            seen.push((
                page,
                result.map_or_else(|error| format!("{error:#}"), |n| n.to_string()),
            ));
            if stop_at == Some(page) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let flow = drive(mock, rows, pages, &mut Emitter::new(&mut on_page));
        (flow, seen)
    }

    #[test]
    fn rows_refill_as_pages_finish_and_every_page_keeps_its_own_length() {
        let lengths = [6, 1, 3, 0, 4, 2, 5];
        let mut mock = Mock::new(lengths.iter().map(|&n| Some(n)).collect());
        let (flow, seen) = run(&mut mock, 3, None);
        assert!(flow.is_continue());
        // A finished page's row goes to the next page before the next step;
        // page 3 ends at its first token and never takes a step.
        let expected: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],
            vec![0, 2, 4],
            vec![0, 2, 4],
            vec![0, 4, 5],
            vec![0, 4, 5],
            vec![0, 6],
            vec![6],
            vec![6],
            vec![6],
            vec![6],
        ];
        assert_eq!(mock.log, expected);
        // Page 6, the last, joined page 0: it did not start alone.
        assert!(mock.alone.is_empty());
        // Results in input order, each page with its own length.
        let expected: Vec<(usize, String)> = lengths.iter().enumerate().map(|(p, n)| (p, n.to_string())).collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn a_page_waits_for_its_prefill_while_others_decode_but_not_alone() {
        let mut mock = Mock::new(vec![Some(4), Some(1), Some(2)]);
        mock.ready_at = vec![0, 0, 3];
        assert!(run(&mut mock, 2, None).0.is_continue());
        assert_eq!(mock.log, [vec![0, 1], vec![0], vec![0], vec![0, 2], vec![2]]);
        // With nothing decoding, the next page starts even if not ready.
        let mut mock = Mock::new(vec![Some(1), Some(1), Some(1)]);
        mock.ready_at = vec![0, 5, 5];
        assert!(run(&mut mock, 1, None).0.is_continue());
        assert_eq!(mock.log, [vec![0], vec![1]]);
        assert_eq!(mock.alone, [2]);
    }

    #[test]
    fn a_page_that_nothing_would_join_runs_alone() {
        // A single input, whatever the rows.
        let mut mock = Mock::new(vec![Some(4)]);
        let (_, seen) = run(&mut mock, 3, None);
        assert_eq!((mock.alone, mock.log.len()), (vec![0], 0));
        assert_eq!(seen, [(0, "4".to_owned())]);
        // The last page, once the others are done; not while one decodes.
        let mut mock = Mock::new(vec![Some(2), Some(2), Some(3)]);
        let (_, seen) = run(&mut mock, 2, None);
        assert_eq!(mock.log, [vec![0, 1], vec![0, 1]]);
        assert_eq!(mock.alone, [2]);
        assert_eq!(seen[2], (2, "3".to_owned()));
        let mut mock = Mock::new(vec![Some(3), Some(1), Some(2)]);
        assert!(run(&mut mock, 2, None).0.is_continue());
        assert!(mock.alone.is_empty());
        assert_eq!(mock.log, [vec![0, 1], vec![0, 2], vec![0, 2]]);
        // A lone page that fails hands over its error.
        let mut mock = Mock::new(vec![Some(1), None]);
        let (_, seen) = run(&mut mock, 1, None);
        assert_eq!(mock.alone, [1]);
        assert_eq!(seen[1], (1, "page 1 did not prefill".to_owned()));
    }

    #[test]
    fn a_failed_prefill_fails_its_page_and_a_failed_step_its_rows() {
        let mut mock = Mock::new(vec![Some(2), None, Some(3), Some(1)]);
        let (_, seen) = run(&mut mock, 2, None);
        assert_eq!(seen[1], (1, "page 1 did not prefill".to_owned()));
        assert_eq!(seen.iter().map(|(p, _)| *p).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!((seen[2].1.as_str(), seen[3].1.as_str()), ("3", "1"));
        assert_eq!(mock.abandoned, [1]);
        assert_eq!(mock.started, [(0, 0), (1, 0), (2, 2), (3, 2)]);
        let mut mock = Mock::new(vec![Some(3), Some(3), Some(1)]);
        mock.fail_step = Some(2);
        let (_, seen) = run(&mut mock, 2, None);
        let failed = "step 2 failed".to_owned();
        assert_eq!(seen, [(0, failed.clone()), (1, failed), (2, "1".to_owned())]);
        assert_eq!(mock.abandoned, [0, 1]);
        // Page 2 then starts alone.
        assert_eq!((mock.log, mock.alone), (vec![vec![0, 1], vec![0, 1]], vec![2]));
    }

    #[test]
    fn a_stop_ends_the_run_at_once() {
        let mut mock = Mock::new(vec![Some(1), Some(5), Some(1)]);
        let (flow, seen) = run(&mut mock, 2, Some(0));
        assert!(flow.is_break());
        assert_eq!(seen, [(0, "1".to_owned())]);
        assert_eq!(mock.log, [vec![0, 1]]);
        // A later failure waits behind page 0. No row may refill until
        // the callback receives it, whether prefill or finishing failed
        // (at the first token or after a decode step).
        for length in [None, Some(0), Some(1)] {
            for stop in [Some(1), None] {
                let mut mock = Mock::new(vec![Some(4), length, Some(1), Some(1)]);
                mock.fail_finish = length.map(|_| 1);
                let (flow, seen) = run(&mut mock, 2, stop);
                assert_eq!(flow.is_break(), stop.is_some());
                assert_eq!(seen[0], (0, "4".to_owned()));
                assert_eq!(seen[1].0, 1);
                assert!(seen[1].1.starts_with("page 1 did not"));
                if stop.is_some() {
                    assert_eq!(seen.len(), 2);
                    assert_eq!(mock.started, [(0, 0), (1, 0)]);
                    assert_eq!(&mock.decoded[2..], [0, 0]);
                } else {
                    assert_eq!(seen.len(), 4);
                    assert_eq!(mock.started, [(0, 0), (1, 0), (2, 4), (3, 4)]);
                    assert_eq!(&mock.decoded[2..], [1, 1]);
                }
            }
        }
    }
}
