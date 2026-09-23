//! Opt-in stop for degenerate repetition loops.
//!
//! Greedy OCR decoding sometimes locks into a loop that repeats a short token
//! cycle until `max_new_tokens`. On the 67-page v3 calibration set this
//! happens on 9 FP32 pages, which then hold 40% of all decode steps. The stop
//! ends generation once the most recent tokens repeat one period `p <=
//! MAX_PERIOD` for at least `max(MIN_WINDOW, MIN_REPEATS * p)` tokens. Up to
//! that step the output is unchanged. On that calibration set the rule fired
//! only on pages that ran to the token limit (never on one that ended at EOS);
//! the longest periodic window on an EOS page was 74 tokens.

/// Longest repeated cycle that is detected, in tokens.
pub(crate) const MAX_PERIOD: usize = 128;
/// Minimum periodic window, in tokens.
pub(crate) const MIN_WINDOW: usize = 256;
/// Minimum number of whole cycles in the window.
pub(crate) const MIN_REPEATS: usize = 4;

/// For every period `p <= MAX_PERIOD`, the number of consecutive generated
/// tokens that equal the token `p` positions earlier.
#[derive(Clone, Debug)]
pub(crate) struct RepetitionStop {
    runs: Vec<u32>,
}

impl RepetitionStop {
    pub(crate) fn new() -> Self {
        Self {
            runs: vec![0; MAX_PERIOD + 1],
        }
    }

    /// Call once after each token is appended to `generated`. Returns true
    /// when the newest tokens form a loop long enough to stop.
    pub(crate) fn push(&mut self, generated: &[u32]) -> bool {
        let Some((&last, earlier)) = generated.split_last() else {
            return false;
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens kept when the stop fires, or None.
    fn fires_after(tokens: &[u32]) -> Option<usize> {
        let mut stop = RepetitionStop::new();
        let mut generated = Vec::new();
        for &t in tokens {
            generated.push(t);
            if stop.push(&generated) {
                return Some(generated.len());
            }
        }
        None
    }

    fn with_loop(prefix: usize, cycle: &[u32], total: usize) -> Vec<u32> {
        let mut t: Vec<u32> = (1000..1000 + prefix as u32).collect();
        while t.len() < total {
            t.push(cycle[(t.len() - prefix) % cycle.len()]);
        }
        t
    }

    #[test]
    fn short_cycle_stops_once_the_window_is_full() {
        // Period 3 starting after 10 distinct tokens: the periodic window is
        // the loop itself, so the stop fires after 256 loop tokens.
        let tokens = with_loop(10, &[1, 2, 3], 1000);
        assert_eq!(fires_after(&tokens), Some(10 + MIN_WINDOW));
        assert_eq!(fires_after(&tokens[..10 + MIN_WINDOW - 1]), None);
    }

    #[test]
    fn long_cycle_needs_four_repeats() {
        let cycle: Vec<u32> = (0..100).collect();
        let tokens = with_loop(5, &cycle, 1000);
        assert_eq!(fires_after(&tokens), Some(5 + MIN_REPEATS * 100));
    }

    #[test]
    fn ignores_cycles_longer_than_the_limit_and_plain_text() {
        let cycle: Vec<u32> = (0..MAX_PERIOD as u32 + 1).collect();
        assert_eq!(fires_after(&with_loop(0, &cycle, 4000)), None);
        let text: Vec<u32> = (0..4000).map(|i| (i * 7919 % 4093) as u32).collect();
        assert_eq!(fires_after(&text), None);
    }

    #[test]
    fn a_break_in_the_cycle_restarts_the_count() {
        let mut tokens = with_loop(0, &[7, 8], 200);
        tokens.push(9);
        tokens.extend(with_loop(0, &[7, 8], 255));
        assert_eq!(fires_after(&tokens), None);
        tokens.push(tokens[tokens.len() - 2]);
        assert_eq!(fires_after(&tokens), Some(tokens.len()));
    }
}
