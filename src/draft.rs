//! Draft tokens for speculative decoding, looked up in the output so far.
//!
//! OCR output repeats structure (table tags, LaTeX, list markers), so the
//! tokens that followed the most recent earlier occurrence of the latest `n`
//! tokens (`n` from [`MAX_N`] down to the minimum match) are a free guess of
//! what comes next. The continuation repeats periodically when it reaches
//! the present, which also covers loops. Drafts only affect speed: the model
//! verifies every drafted token and keeps exactly its own greedy output.
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

impl NgramDrafter {
    pub(crate) fn new(min_match: usize) -> Self {
        Self {
            min_match: min_match.clamp(1, MAX_N),
            index: HashMap::new(),
            indexed: 0,
        }
    }

    /// Up to `limit` tokens expected to follow `tokens` (the whole output so
    /// far), written to `out` (cleared first; empty when nothing matches).
    pub(crate) fn propose(&mut self, tokens: &[u32], limit: usize, out: &mut Vec<u32>) {
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
        }
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
        drafter.propose(&tokens, 3, &mut out);
        assert_eq!(out, [3, 4, 9]);
        // A single-token match is below the minimum.
        drafter.propose(&[5, 6, 7, 6], 2, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn loops_continue_periodically() {
        let mut drafter = NgramDrafter::new(2);
        let mut out = Vec::new();
        drafter.propose(&[7, 8, 7, 8, 7, 8], 5, &mut out);
        assert_eq!(out, [7, 8, 7, 8, 7]);
    }

    #[test]
    fn index_grows_incrementally() {
        let mut drafter = NgramDrafter::new(1);
        let mut out = Vec::new();
        let mut tokens = vec![3, 1, 4];
        drafter.propose(&tokens, 2, &mut out);
        assert!(out.is_empty());
        tokens.extend([1, 5, 9, 2, 6, 5, 3, 5]);
        drafter.propose(&tokens, 2, &mut out);
        // No longer n-gram matches; the most recent earlier "5" ends at
        // position 9, so the draft copies the two tokens that followed it.
        assert_eq!(out, [3, 5]);
    }
}
