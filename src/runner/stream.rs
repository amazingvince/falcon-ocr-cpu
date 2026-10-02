//! Streaming recognition (`Runner::recognize_files_streaming`): each page's
//! result reaches the caller as soon as the page and every earlier page are
//! finished, in input order, and a page that fails hands over its error
//! without stopping the others. `recognize_files` collects the stream.
use std::{collections::BTreeMap, ops::ControlFlow, path::Path};

use anyhow::Result;

use super::{OcrResult, Runner};
use crate::{
    config::GenerationOptions,
    trace::{NoTrace, Trace},
};

/// Items that finish out of order, handed out in index order: `push` stores
/// item `index`, `pop` returns the next index once it has arrived.
pub(crate) struct InOrder<T> {
    next: usize,
    waiting: BTreeMap<usize, T>,
}

impl<T> InOrder<T> {
    /// Nothing arrived yet; item 0 is handed out first.
    pub(crate) fn new() -> Self {
        Self {
            next: 0,
            waiting: BTreeMap::new(),
        }
    }

    /// Store item `index`. Every index arrives once, never below the last
    /// one handed out.
    pub(crate) fn push(&mut self, index: usize, item: T) {
        assert!(
            index >= self.next && !self.waiting.contains_key(&index),
            "item {index} arrived twice"
        );
        self.waiting.insert(index, item);
    }

    /// The next item in index order, once it has arrived.
    pub(crate) fn pop(&mut self) -> Option<(usize, T)> {
        let item = self.waiting.remove(&self.next)?;
        self.next += 1;
        Some((self.next - 1, item))
    }

    /// Items waiting for an earlier one.
    #[cfg(test)]
    fn waiting(&self) -> usize {
        self.waiting.len()
    }
}

/// Hands finished pages to the caller's callback in input order, and
/// remembers when the callback stopped the run.
pub(crate) struct Emitter<'f, T> {
    order: InOrder<T>,
    on_page: &'f mut dyn FnMut(usize, T) -> ControlFlow<()>,
    stopped: bool,
}

impl<'f, T> Emitter<'f, T> {
    /// Hand pages to `on_page`, starting with page 0.
    pub(crate) fn new(on_page: &'f mut dyn FnMut(usize, T) -> ControlFlow<()>) -> Self {
        Self {
            order: InOrder::new(),
            on_page,
            stopped: false,
        }
    }

    /// Page `index` is finished: hand it over with every later page it held
    /// back. `Break` once the callback has stopped the run; pages finished
    /// after that are dropped.
    pub(crate) fn finish(&mut self, index: usize, item: T) -> ControlFlow<()> {
        if self.stopped {
            return ControlFlow::Break(());
        }
        self.order.push(index, item);
        while let Some((index, item)) = self.order.pop() {
            if (self.on_page)(index, item).is_break() {
                self.stopped = true;
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

impl Runner {
    /// [`Runner::recognize_files`] as a stream: `on_page(index, result)`
    /// receives each page's result as soon as the page and every earlier one
    /// are finished, in input order. A page that fails (an unreadable or
    /// corrupt file, a context budget conflict, a runtime error) hands over
    /// its error the same way; the other pages still run. `on_page` returning
    /// `ControlFlow::Break` stops the run: no page starts after it, pages
    /// decoding beside it in other rows stop unfinished, a prefill already
    /// running on the pipeline's second pool completes first, and no further
    /// result reaches `on_page`. The returned error is the run's own (invalid
    /// options). With a batch size above 1, a page that finishes frees its
    /// row for the next page at once (continuous batching, `batch/`), each
    /// page keeps its own fitted output budget, as it would alone, and its
    /// `Timings` are the sum of its own stages (`Timings::total_ms`). Tokens
    /// are those of `recognize_files`, except that its fixed cohorts decode a
    /// batch to the smallest fitted budget in it.
    pub fn recognize_files_streaming<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        mut on_page: impl FnMut(usize, Result<OcrResult>) -> ControlFlow<()>,
    ) -> Result<()> {
        if self.config.batch_size == 1 {
            return self.recognize_files_streaming_with_trace(paths, options, &mut NoTrace, on_page);
        }
        options.validate()?;
        self.stream_rows(paths, options, &mut Emitter::new(&mut on_page));
        Ok(())
    }

    /// [`Runner::recognize_files_streaming`] with a trace, in fixed cohorts
    /// of the batch size (a cohort hands its pages over when it finishes),
    /// as `recognize_files` runs them. Tensors carry each page's input index:
    /// `request.{i}` for a page that runs alone or prefills in a cohort (and
    /// `request.{i}.rerun` for a routed page's safety-net rerun), and
    /// `batch.{i}` for a cohort's joint decode, after its first page.
    pub fn recognize_files_streaming_with_trace<P: AsRef<Path>>(
        &self,
        paths: &[P],
        options: &GenerationOptions,
        trace: &mut dyn Trace,
        mut on_page: impl FnMut(usize, Result<OcrResult>) -> ControlFlow<()>,
    ) -> Result<()> {
        self.stream_cohorts(paths, options, trace, false, &mut on_page)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_order_hands_out_items_by_index_once_every_earlier_one_arrived() {
        let mut order = InOrder::new();
        assert!(order.pop().is_none());
        order.push(2, "c");
        order.push(1, "b");
        assert_eq!((order.pop(), order.waiting()), (None, 2));
        order.push(0, "a");
        assert_eq!(order.pop(), Some((0, "a")));
        assert_eq!(order.pop(), Some((1, "b")));
        assert_eq!(order.pop(), Some((2, "c")));
        assert_eq!((order.pop(), order.waiting()), (None, 0));
        order.push(3, "d");
        assert_eq!(order.pop(), Some((3, "d")));
    }

    #[test]
    #[should_panic(expected = "arrived twice")]
    fn in_order_rejects_an_index_it_already_handed_out() {
        let mut order = InOrder::new();
        order.push(0, ());
        order.pop();
        order.push(0, ());
    }

    #[test]
    fn emitter_delivers_in_order_and_drops_everything_after_a_stop() {
        // Pages finish in the order a batch of three would finish them.
        let mut seen = Vec::new();
        let mut on_page = |index: usize, page: char| {
            seen.push((index, page));
            if page == 'e' {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let mut emit = Emitter::new(&mut on_page);
        assert!(emit.finish(1, 'b').is_continue());
        assert!(emit.finish(2, 'c').is_continue());
        assert!(emit.finish(0, 'a').is_continue());
        assert!(emit.finish(5, 'f').is_continue());
        assert!(emit.finish(4, 'e').is_continue());
        // Page 3 releases 4, whose callback stops the run; 5 is dropped.
        assert!(emit.finish(3, 'd').is_break());
        assert!(emit.finish(6, 'g').is_break());
        assert_eq!(seen, [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]);
    }
}
