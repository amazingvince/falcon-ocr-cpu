//! Stop for degenerate repetition loops, enabled by default.
//!
//! Greedy OCR decoding sometimes locks into a loop that repeats a short token
//! cycle until `max_new_tokens`. On the 67-page v3 calibration set this
//! happens on 9 FP32 pages, which then hold 40% of all decode steps. The stop
//! ends generation once the most recent tokens repeat one period `p <=
//! MAX_PERIOD` for at least `max(MIN_WINDOW, MIN_REPEATS * p)` tokens. Up to
//! that step the output is unchanged. On that calibration set the rule fired
//! only on pages that ran to the token limit (never on one that ended at EOS);
//! the longest periodic window on an EOS page was 74 tokens.
//!
//! A counting loop writes lines that differ only in an incrementing number
//! (`44 女 …`, `45 女 …`, …), so its tokens never repeat exactly. The stop
//! therefore also reads each run of digit tokens (the tokenizer gives every
//! digit its own token) as one number and compares the resulting symbols
//! with those a period earlier: other tokens must be equal, and each number
//! must differ from the one a period earlier by the same step in every
//! cycle. Such a counting window, with at least one nonzero step, ends
//! generation once it spans `MIN_COUNT_REPEATS` whole periods of `p <=
//! MAX_PERIOD` symbols and at least `MIN_WINDOW` tokens; like the exact
//! rule, it only truncates. Over 423 recorded outputs and the 1,651
//! OmniDocBench ground-truth pages, the longest counting window in
//! legitimate text was 20 cycles (a scoring table counting down from 20
//! bonus points) and the next longest 9; the counting loops on record, on
//! three pages, reached 65 to 212 cycles.

use crate::tokenizer::DIGIT_ZERO;

/// Longest repeated cycle that is detected, in tokens (in symbols for a
/// counting loop).
pub(crate) const MAX_PERIOD: usize = 128;
/// Minimum periodic window, in tokens.
pub(crate) const MIN_WINDOW: usize = 256;
/// Minimum number of whole cycles in the window.
pub(crate) const MIN_REPEATS: usize = 4;
/// Minimum number of whole cycles in a counting window.
pub(crate) const MIN_COUNT_REPEATS: usize = 32;

/// Stands for a number in the symbol history; no token has this ID.
const NUMBER: u32 = u32::MAX;
/// Symbols kept: the newest and two periods of the longest cycle before it.
const HISTORY: usize = 2 * MAX_PERIOD + 1;

/// For every period `p <= MAX_PERIOD`, the number of consecutive generated
/// tokens that equal the token `p` positions earlier, and the counting-loop
/// state.
#[derive(Clone, Debug)]
pub(crate) struct RepetitionStop {
    runs: Vec<u32>,
    counting: CountingLoop,
}

impl RepetitionStop {
    /// Start tracking a new page with no repeated runs.
    pub(crate) fn new() -> Self {
        Self {
            runs: vec![0; MAX_PERIOD + 1],
            counting: CountingLoop::new(),
        }
    }

    /// Call once after each token is appended to `generated`. Returns true
    /// when the newest tokens form a loop long enough to stop.
    pub(crate) fn push(&mut self, generated: &[u32]) -> bool {
        let Some((&last, earlier)) = generated.split_last() else {
            return false;
        };
        let exact = self.exact(last, earlier);
        let cycles = self.counting.push(last, earlier.len());
        exact || cycles >= MIN_COUNT_REPEATS
    }

    /// The exact rule: true when `last` closes a periodic window long enough
    /// to stop.
    fn exact(&mut self, last: u32, earlier: &[u32]) -> bool {
        let mut stop = false;
        for p in 1..=MAX_PERIOD.min(earlier.len()) {
            let run = &mut self.runs[p];
            *run = if earlier[earlier.len() - p] == last {
                *run + 1
            } else {
                0
            };
            stop |= *run as usize + p >= MIN_WINDOW.max(MIN_REPEATS * p);
        }
        stop
    }
}

/// A non-digit token, or a number read from a run of digit tokens.
#[derive(Clone, Copy, Debug, Default)]
struct Symbol {
    /// The token, or `NUMBER`.
    token: u32,
    /// The number's value (wrapping past `u64::MAX`).
    value: u64,
    /// Position of the symbol's first token in the generated output.
    start: u32,
}

/// Counting-loop state: the recent symbols and, for every period `p <=
/// MAX_PERIOD` in symbols, the current window that matches `p` symbols
/// earlier. Allocates only when created.
#[derive(Clone, Debug)]
struct CountingLoop {
    /// Ring of the newest `HISTORY` symbols, indexed by symbol count.
    history: Vec<Symbol>,
    /// Symbols completed so far.
    symbols: usize,
    /// The number whose digits are still arriving.
    number: Option<Symbol>,
    /// Consecutive symbols that match the symbol `p` earlier.
    runs: Vec<u32>,
    /// Token position where the current window for `p` starts.
    starts: Vec<u32>,
    /// Matched symbols since the last number that moved (`u32::MAX`: none
    /// in the current window).
    since_step: Vec<u32>,
}

impl CountingLoop {
    fn new() -> Self {
        Self {
            history: vec![Symbol::default(); HISTORY],
            symbols: 0,
            number: None,
            runs: vec![0; MAX_PERIOD + 1],
            starts: vec![0; MAX_PERIOD + 1],
            since_step: vec![u32::MAX; MAX_PERIOD + 1],
        }
    }

    /// Read the token at `position`. Returns the most whole cycles of any
    /// counting window that ends here and spans at least `MIN_WINDOW`
    /// tokens, or 0. A digit only extends the current number, so it
    /// returns 0; the token after the number completes it.
    fn push(&mut self, token: u32, position: usize) -> usize {
        let start = position as u32;
        let digit = token.wrapping_sub(DIGIT_ZERO);
        if digit < 10 {
            let number = self.number.get_or_insert(Symbol {
                token: NUMBER,
                value: 0,
                start,
            });
            number.value = number.value.wrapping_mul(10).wrapping_add(u64::from(digit));
            return 0;
        }
        let number = self.number.take().map_or(0, |number| self.symbol(number, position));
        let other = self.symbol(Symbol { token, value: 0, start }, position + 1);
        number.max(other)
    }

    /// Append `symbol`, whose last token ends before position `end`, and
    /// return the cycles as for `push`.
    fn symbol(&mut self, symbol: Symbol, end: usize) -> usize {
        let j = self.symbols;
        let mut cycles = 0;
        for p in 1..=MAX_PERIOD.min(j) {
            let earlier = self.history[(j - p) % HISTORY];
            let run = self.runs[p] as usize;
            let (matched, moved) = if symbol.token != earlier.token {
                (false, false)
            } else if symbol.token != NUMBER {
                (true, false)
            } else {
                // The window's first cycle sets each number's step; once
                // the earlier number is itself in the window, the step must
                // repeat.
                let step = symbol.value.wrapping_sub(earlier.value);
                let before = self.history[(j + HISTORY - 2 * p) % HISTORY].value;
                (run < p || step == earlier.value.wrapping_sub(before), step != 0)
            };
            if !matched {
                self.runs[p] = 0;
                self.since_step[p] = u32::MAX;
                continue;
            }
            if run == 0 {
                self.starts[p] = earlier.start;
            }
            self.runs[p] += 1;
            self.since_step[p] = if moved { 0 } else { self.since_step[p].saturating_add(1) };
            if (self.since_step[p] as usize) < p && end - self.starts[p] as usize >= MIN_WINDOW {
                cycles = cycles.max((run + 1 + p) / p);
            }
        }
        self.history[j % HISTORY] = symbol;
        self.symbols += 1;
        cycles
    }
}

#[cfg(test)]
mod tests;
