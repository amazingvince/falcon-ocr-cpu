//! Speculative decoding for single pages on a split cache: up to `max_draft`
//! tokens drafted from earlier output (`crate::draft`, n-gram match of at
//! least `min_match` tokens) are verified in one multi-row step. Every row
//! is bitwise the single-row step, so tokens are unchanged; drafting switches
//! itself off while drafts are rejected too often to pay (normal text) and
//! on for repetitive output (tables, loops). With a trained draft head
//! (`crate::draft_head`, `RunnerConfig::drafter`), every verified position
//! also feeds the head, which drafts while it is confident.
use std::time::Instant;

use anyhow::Result;

use super::{Runner, generate::DecodeLoop, select};

impl Runner {
    pub(super) fn decode_speculative(
        &self,
        d: &mut DecodeLoop<'_, '_>,
        max_draft: usize,
        min_match: usize,
    ) -> Result<()> {
        let c = &self.model.config;
        let mut ngram = crate::draft::NgramDrafter::new(min_match);
        let mut history = self.document_history();
        let mut draft = Vec::with_capacity(max_draft);
        let mut inputs = Vec::with_capacity(max_draft + 1);
        let mut positions = Vec::with_capacity(max_draft + 1);
        // (single steps, their ms, verify steps, their ms, drafted, accepted)
        let mut stats = (0_usize, 0.0_f64, 0_usize, 0.0_f64, 0_usize, 0_usize);
        let mut policy = crate::draft::DraftPolicy::new();
        let drafter = self.config.drafter;
        // The trained head, when the session captured the features it reads.
        let head = self.draft_head.as_deref().filter(|_| d.session.features().is_some());
        let options = crate::draft_head::DraftOptions {
            confidence: self.config.draft_confidence,
            backoff: self.config.tuning.draft_backoff,
            window: self.config.tuning.draft_window,
            kv: self.config.tuning.draft_kv,
            path_gate: self.config.tuning.draft_path_gate,
        };
        let mut head_state = head.map(|_| crate::draft_head::DraftState::new(options));
        // Tokens whose features the latest forward left for the head: the
        // accepted drafts of a verify step, then the next token (row r's
        // output is fed[r]).
        let mut fed = Vec::with_capacity(max_draft + 1);
        'decode: loop {
            let token = d.next_token;
            if d.generation.push(token, d.stops) {
                break;
            }
            if let (Some(h), Some(state)) = (head, head_state.as_mut()) {
                fed.push(token);
                let features = d.session.features().expect("captured");
                let model = &self.model;
                h.push_rows(state, &fed, |r| features.row(r), |t| model.embedding_row(t), d.simd)?;
                fed.clear();
            }
            // Accepted drafts plus the next token stay within the budget
            // and the cache.
            let limit = max_draft
                .min(d.generation.max_new_tokens - d.generation.len() - 1)
                .min(d.session.remaining_capacity().saturating_sub(1));
            // Drafting time counts toward the step (the policy weighs it).
            let step_started = Instant::now();
            draft.clear();
            if policy.should_draft() {
                use crate::config::Drafter;
                if drafter != Drafter::Head || head.is_none() {
                    ngram.propose(&d.generation.tokens, limit, history.as_deref(), &mut draft);
                }
                if draft.is_empty()
                    && drafter != Drafter::Ngram
                    && let (Some(h), Some(state)) = (head, head_state.as_mut())
                {
                    let model = &self.model;
                    h.draft(state, limit, |t| model.embedding_row(t), &mut draft, d.simd)?;
                }
            }
            d.team.select();
            if draft.is_empty() {
                self.model.embed(&[token], None, d.simd, d.hidden)?;
                let next = self.model.forward_next(
                    d.hidden,
                    &[d.session.next_position],
                    &[[f32::NAN; 2]],
                    d.session,
                    d.trace,
                    "",
                    d.screen,
                )?;
                let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
                d.team.record(step_ms);
                stats.0 += 1;
                stats.1 += step_ms;
                policy.single_step(step_ms);
                d.next_token = select(&next, 0, c.vocab_size)?;
                d.trace.decode_step(1, step_ms, d.session.cache_bytes());
                continue;
            }
            inputs.clear();
            inputs.push(token);
            inputs.extend_from_slice(&draft);
            let rows = inputs.len();
            positions.clear();
            positions.extend((0..rows).map(|r| d.session.next_position + r));
            let kept = d.session.len;
            self.model.embed(&inputs, None, d.simd, d.hidden)?;
            let next = self.model.verify_next(d.hidden, &positions, d.session, d.screen)?;
            // predicted[r]: the greedy token after inputs[..=r].
            let mut predicted = [0_u32; crate::head_screen::MAX_ROWS];
            for (r, slot) in predicted[..rows].iter_mut().enumerate() {
                *slot = select(&next, r, c.vocab_size)?;
            }
            let accepted = draft
                .iter()
                .zip(&predicted)
                .take_while(|(drafted, p)| drafted == p)
                .count();
            // Accepted drafts are verified positions for the head (row r's
            // output is draft[r]); they are fed with the next token.
            if head.is_some() {
                fed.extend_from_slice(&draft[..accepted]);
            }
            d.session.truncate(kept + 1 + accepted)?;
            let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            d.team.record_verify(step_ms);
            stats.2 += 1;
            stats.3 += step_ms;
            stats.4 += draft.len();
            stats.5 += accepted;
            policy.verify_step(step_ms, draft.len(), accepted);
            d.trace.decode_step(rows, step_ms, d.session.cache_bytes());
            // Accepted drafts never reach the budget: `limit` left room for
            // them and the next token.
            for &token in &draft[..accepted] {
                if d.generation.push(token, d.stops) {
                    break 'decode;
                }
            }
            d.next_token = predicted[accepted];
        }
        if let Some(history) = history.as_mut() {
            history.push_page(&d.generation.tokens);
        }
        if self.config.tuning.phases {
            let (singles, single_ms, verifies, verify_ms, drafted, accepted) = stats;
            eprintln!(
                "speculation: {} tokens; {singles} single steps ({:.2} ms), {verifies} verify steps ({:.2} ms), \
                 drafted {drafted}, accepted {accepted} ({:.0}%)",
                d.generation.len(),
                single_ms / singles.max(1) as f64,
                verify_ms / verifies.max(1) as f64,
                100.0 * accepted as f64 / drafted.max(1) as f64,
            );
        }
        Ok(())
    }
}
