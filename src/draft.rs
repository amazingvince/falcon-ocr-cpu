//! Draft tokens for speculative decoding, looked up in the output so far.
//!
//! OCR output repeats structure (table tags, LaTeX, list markers), so the
//! tokens that followed the most recent earlier occurrence of the latest `n`
//! tokens (`n` from [`MAX_N`] down to the minimum match) are a free guess of
//! what comes next. The continuation repeats periodically when it reaches
//! the present, which also covers loops. Earlier pages of the same document
//! ([`DocumentHistory`]) add running headers, names and repeated table
//! headers, for full `MAX_N`-token matches only. Drafts only affect speed:
//! the model verifies every drafted token and keeps exactly its own greedy
//! output.
use std::collections::HashMap;

/// Longest n-gram looked up.
const MAX_N: usize = 4;

pub(crate) struct NgramDrafter {
    min_match: usize,
    /// (n-gram, length) -> position just after its most recent occurrence.
    index: HashMap<([u32; MAX_N], u8), usize>,
    /// Occurrences ending at positions `1..=indexed` are in the index.
    indexed: usize,
}

fn key(gram: &[u32]) -> ([u32; MAX_N], u8) {
    let mut padded = [u32::MAX; MAX_N];
    padded[..gram.len()].copy_from_slice(gram);
    (padded, gram.len() as u8)
}

/// Earlier pages of one document: their tokens (pages separated by
/// [`SEPARATOR`]) and the position after the most recent occurrence of each
/// `MAX_N`-gram. In an offline replay of calibration documents, drafting from
/// full matches here saved about 1.5% more decode time on an 8-page document
/// and cost 0.2% on unrelated pages (shorter matches cost more).
#[derive(Default)]
pub(crate) struct DocumentHistory {
    tokens: Vec<u32>,
    index: HashMap<[u32; MAX_N], usize>,
}

/// Page boundary in [`DocumentHistory`] (never a token id).
const SEPARATOR: u32 = u32::MAX;

impl DocumentHistory {
    /// Append a finished page.
    pub(crate) fn push_page(&mut self, page: &[u32]) {
        self.tokens.push(SEPARATOR);
        let base = self.tokens.len();
        self.tokens.extend_from_slice(page);
        for end in base + MAX_N..self.tokens.len() {
            let mut gram = [0; MAX_N];
            gram.copy_from_slice(&self.tokens[end - MAX_N..end]);
            self.index.insert(gram, end);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.tokens.clear();
        self.index.clear();
    }

    /// Up to `limit` tokens that followed `gram` (`MAX_N` tokens) most
    /// recently, stopping at a page boundary; false when there are none.
    fn continuation(&self, gram: &[u32], limit: usize, out: &mut Vec<u32>) -> bool {
        let mut key = [0; MAX_N];
        key.copy_from_slice(gram);
        let Some(&start) = self.index.get(&key) else {
            return false;
        };
        out.extend(
            self.tokens[start..]
                .iter()
                .take(limit)
                .take_while(|&&t| t != SEPARATOR),
        );
        !out.is_empty()
    }
}

impl NgramDrafter {
    pub(crate) fn new(min_match: usize) -> Self {
        Self {
            min_match: min_match.clamp(1, MAX_N),
            index: HashMap::new(),
            indexed: 0,
        }
    }

    /// Up to `limit` tokens expected to follow `tokens` (the whole output so
    /// far), written to `out` (cleared first; empty when nothing matches):
    /// the longest match on this page, then a full match in `history`.
    pub(crate) fn propose(
        &mut self,
        tokens: &[u32],
        limit: usize,
        history: Option<&DocumentHistory>,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        let len = tokens.len();
        // Index every occurrence that ends before the present, so a match
        // always has at least one known continuation token.
        while self.indexed + 1 < len {
            let end = self.indexed + 1;
            for n in 1..=MAX_N.min(end) {
                self.index.insert(key(&tokens[end - n..end]), end);
            }
            self.indexed = end;
        }
        if limit == 0 {
            return;
        }
        for n in (self.min_match..=MAX_N.min(len)).rev() {
            if let Some(&start) = self.index.get(&key(&tokens[len - n..len])) {
                let period = len - start;
                out.extend((0..limit).map(|j| tokens[start + j % period]));
                return;
            }
            if n == MAX_N
                && let Some(history) = history
                && history.continuation(&tokens[len - n..len], limit, out)
            {
                return;
            }
        }
    }
}

/// When to draft: only while drafted tokens are accepted often enough to
/// pay for verification. A verify step with `d` drafts costs about
/// `single + d * extra`, and each accepted token saves `single`, so drafting
/// pays once the acceptance rate per drafted token exceeds `extra / single`.
/// Both costs and the rate are running averages of this decode; while below
/// the break-even, one probe draft every [`PROBE_EVERY`] steps notices when
/// the output turns repetitive.
pub(crate) struct DraftPolicy {
    single_ms: f64,
    extra_ms: f64,
    rate: f64,
    since_probe: usize,
}

/// Steps between probe drafts while drafting does not pay.
const PROBE_EVERY: usize = 16;
/// Weight of the newest observation in the running averages.
const ALPHA: f64 = 0.15;

impl DraftPolicy {
    pub(crate) fn new() -> Self {
        // Optimistic start: draft until measurements say otherwise.
        Self {
            single_ms: 0.0,
            extra_ms: 0.0,
            rate: 0.6,
            since_probe: 0,
        }
    }

    fn average(old: f64, new: f64) -> f64 {
        if old == 0.0 { new } else { old + ALPHA * (new - old) }
    }

    pub(crate) fn should_draft(&mut self) -> bool {
        let pays = self.single_ms == 0.0
            || self.extra_ms == 0.0
            || self.rate > self.extra_ms / self.single_ms;
        if pays || self.since_probe >= PROBE_EVERY {
            self.since_probe = 0;
            true
        } else {
            self.since_probe += 1;
            false
        }
    }

    pub(crate) fn single_step(&mut self, ms: f64) {
        self.single_ms = Self::average(self.single_ms, ms);
    }

    pub(crate) fn verify_step(&mut self, ms: f64, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        if self.single_ms > 0.0 {
            let extra = ((ms - self.single_ms) / drafted as f64).max(0.0);
            self.extra_ms = Self::average(self.extra_ms, extra);
        }
        self.rate += ALPHA * (accepted as f64 / drafted as f64 - self.rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_the_continuation_of_the_latest_match() {
        let mut drafter = NgramDrafter::new(2);
        let mut out = Vec::new();
        let tokens = [1, 2, 3, 4, 9, 1, 2];
        drafter.propose(&tokens, 3, None, &mut out);
        assert_eq!(out, [3, 4, 9]);
        // A single-token match is below the minimum.
        drafter.propose(&[5, 6, 7, 6], 2, None, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn drafts_from_earlier_pages_on_full_matches_only() {
        let mut history = DocumentHistory::default();
        history.push_page(&[7, 1, 2, 3, 4, 5, 6]);
        let mut drafter = NgramDrafter::new(2);
        let mut out = Vec::new();
        // A full 4-token match continues from the earlier page, up to its end.
        drafter.propose(&[9, 1, 2, 3, 4], 3, Some(&history), &mut out);
        assert_eq!(out, [5, 6]);
        // A match on this page wins; shorter history matches are ignored.
        let mut drafter = NgramDrafter::new(2);
        drafter.propose(&[3, 4, 8, 3, 4], 2, Some(&history), &mut out);
        assert_eq!(out, [8, 3]);
        let mut drafter = NgramDrafter::new(2);
        drafter.propose(&[9, 2, 3, 4], 2, Some(&history), &mut out);
        assert!(out.is_empty());
        // Continuations stop at a page boundary.
        history.push_page(&[8, 9]);
        let mut drafter = NgramDrafter::new(2);
        drafter.propose(&[0, 3, 4, 5, 6], 3, Some(&history), &mut out);
        assert!(out.is_empty());
        let mut drafter = NgramDrafter::new(2);
        drafter.propose(&[0, 2, 3, 4, 5], 3, Some(&history), &mut out);
        assert_eq!(out, [6]);
    }

    #[test]
    fn policy_stops_drafting_below_break_even_and_probes() {
        let mut policy = DraftPolicy::new();
        policy.single_step(9.0);
        // Four drafts, none accepted, each costing 3.6 ms extra (break-even 40%).
        for _ in 0..30 {
            if policy.should_draft() {
                policy.verify_step(9.0 + 4.0 * 3.6, 4, 0);
            } else {
                policy.single_step(9.0);
            }
        }
        let drafted = (0..64).filter(|_| {
            let d = policy.should_draft();
            if !d {
                policy.single_step(9.0);
            }
            d
        });
        // One probe per PROBE_EVERY + 1 steps, depending on the phase.
        assert!((3..=4).contains(&drafted.count()));
        // Loops: every draft accepted keeps drafting on.
        let mut policy = DraftPolicy::new();
        policy.single_step(9.0);
        for _ in 0..30 {
            assert!(policy.should_draft());
            policy.verify_step(9.0 + 4.0 * 3.6, 4, 4);
        }
    }

    #[test]
    fn loops_continue_periodically() {
        let mut drafter = NgramDrafter::new(2);
        let mut out = Vec::new();
        drafter.propose(&[7, 8, 7, 8, 7, 8], 5, None, &mut out);
        assert_eq!(out, [7, 8, 7, 8, 7]);
    }

    #[test]
    fn index_grows_incrementally() {
        let mut drafter = NgramDrafter::new(1);
        let mut out = Vec::new();
        let mut tokens = vec![3, 1, 4];
        drafter.propose(&tokens, 2, None, &mut out);
        assert!(out.is_empty());
        tokens.extend([1, 5, 9, 2, 6, 5, 3, 5]);
        drafter.propose(&tokens, 2, None, &mut out);
        // No longer n-gram matches; the most recent earlier "5" ends at
        // position 9, so the draft copies the two tokens that followed it.
        assert_eq!(out, [3, 5]);
    }
}
