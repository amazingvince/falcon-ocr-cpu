//! One page's growing output: the stop ladder every decode loop shares, the
//! plain (non-speculative) loop, and the result a finished page becomes.
use std::time::Instant;

use anyhow::Result;

use super::{DecodeTeam, FinishReason, OcrResult, Runner, Timings, select};
use crate::{
    kernels,
    model::{Next, Session},
    repetition::RepetitionStop,
    trace::Trace,
};

/// A page's generated tokens and the stop they reached.
pub(super) struct Generation {
    pub(super) tokens: Vec<u32>,
    repetition: RepetitionStop,
    pub(super) reason: FinishReason,
    /// Stop on repetition loops (off for teacher-forced runs).
    stop_loops: bool,
    pub(super) max_new_tokens: usize,
}

impl Generation {
    pub(super) fn new(max_new_tokens: usize, stop_loops: bool) -> Self {
        Self {
            tokens: Vec::with_capacity(max_new_tokens),
            repetition: RepetitionStop::new(),
            reason: FinishReason::Length,
            stop_loops,
            max_new_tokens,
        }
    }
    /// Record `token`. True once the page is finished: a stop token, a
    /// repetition loop (when enabled), or the output budget (the reason then
    /// stays `Length`). Never allocates.
    pub(super) fn push(&mut self, token: u32, stops: &[u32]) -> bool {
        self.tokens.push(token);
        if stops.contains(&token) {
            self.reason = FinishReason::Eos;
            return true;
        }
        if self.stop_loops && self.repetition.push(&self.tokens) {
            self.reason = FinishReason::Repetition;
            return true;
        }
        self.tokens.len() == self.max_new_tokens
    }
    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }
    /// The newest token.
    pub(super) fn last(&self) -> u32 {
        *self.tokens.last().expect("a page holds its first token")
    }
}

/// The state one decode loop advances for a single page.
pub(super) struct DecodeLoop<'r, 's> {
    pub(super) session: &'s mut Session,
    pub(super) hidden: &'s mut Vec<f32>,
    pub(super) generation: &'s mut Generation,
    pub(super) team: &'s mut DecodeTeam<'r>,
    pub(super) trace: &'s mut dyn Trace,
    /// Stop tokens (empty when teacher forcing, which never stops on EOS).
    pub(super) stops: &'s [u32],
    pub(super) screen: bool,
    pub(super) simd: kernels::Simd,
    /// The token to feed next: the latest selection.
    pub(super) next_token: u32,
}

/// A finished page, before the runner adds its provenance.
pub(super) struct Page {
    pub(super) tokens: Vec<u32>,
    pub(super) reason: FinishReason,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) input_tokens: usize,
    pub(super) teacher_forced: bool,
    pub(super) budget_clamped: bool,
    pub(super) timings: Timings,
}

impl Runner {
    /// One token per step until the page finishes. `teacher` overrides the
    /// model's choices with its tokens (`score` records each choice against
    /// them).
    pub(super) fn decode_plain(&self, d: &mut DecodeLoop<'_, '_>, teacher: &[u32], score: bool) -> Result<()> {
        let c = &self.model.config;
        for step in 0..d.generation.max_new_tokens {
            let token = d.next_token;
            if d.generation.push(token, d.stops) {
                break;
            }
            d.team.select();
            let step_started = Instant::now();
            self.model.embed(&[token], None, d.simd, d.hidden)?;
            let phase = if d.trace.enabled() {
                format!("decode.{step}")
            } else {
                String::new()
            };
            let next = self.model.forward_next(
                d.hidden,
                &[d.session.next_position],
                &[[f32::NAN; 2]],
                d.session,
                d.trace,
                &phase,
                d.screen,
            )?;
            let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            d.team.record(step_ms);
            d.next_token = if let Some(&forced) = teacher.get(step + 1) {
                if score {
                    d.trace.teacher_step(step + 1, forced, select(&next, 0, c.vocab_size)?);
                    if let Next::Logits(logits) = &next {
                        d.trace.teacher_logits(step + 1, &logits[..c.vocab_size]);
                    }
                }
                forced
            } else {
                select(&next, 0, c.vocab_size)?
            };
            d.trace.decode_step(1, step_ms, d.session.cache_bytes());
        }
        Ok(())
    }

    /// The result of a finished page with this runner's provenance.
    pub(super) fn finish_result(&self, page: Page) -> Result<OcrResult> {
        let text = self.tokenizer.decode(&page.tokens)?;
        Ok(OcrResult {
            text,
            output_tokens: page.tokens.len(),
            token_ids: page.tokens,
            finish_reason: page.reason,
            width: page.width,
            height: page.height,
            input_tokens: page.input_tokens,
            precision: self.precision(),
            backend: self.resolved.decode_isa.clone(),
            mode: self.resolved.mode,
            decode_threads: self.decode_threads_used(),
            plan: Some(self.resolved.clone()),
            decode_tuning: self.take_decode_tuning(),
            cache_layout: self.config.cache_layout,
            weight_layout: self.config.weight_layout,
            packed_weight_bytes: self.model.packed_weight_bytes(),
            weight_packing_ms: self.model.weight_packing_ms(),
            teacher_forced: page.teacher_forced,
            budget_clamped: page.budget_clamped,
            timings: page.timings,
        })
    }
}
