//! Escalation (`falcon-ocr run --escalate`): a fast-mode page that the
//! repetition stop ended is read again by a near-exact model, whose result
//! replaces it with an `escalated_from` record of the fast attempt, as the
//! router's safety net records a routed attempt. This is remedy 1 of
//! `research/phase4-hillclimb/attempt3/RESULTS-V3.md` §7, with near-exact
//! in place of exact mode: fast mode's held-out failures on handwriting
//! were loops caused by the 8-bit weights, which the stop ends; rereading
//! only those pages gives them near-exact output (1 changed token in 24,262
//! against FP32) at near-exact cost.
//!
//! Only `FinishReason::Repetition` escalates. On the 200 held-out pages, the
//! loops that the 8-bit weights caused on pages FP32 ends at EOS were ended
//! by the stop (§6), and fast mode ran to the length limit on 7 pages
//! against FP32's 19 (§8), so rereading length stops would pay near-exact
//! cost for the most expensive pages with little to gain. Near-exact pages
//! never escalate to exact mode: near-exact produced FP32's tokens on all
//! 118 English gate pages and differs from FP32 at 1 of 24,262 anchor steps,
//! so a near-exact loop is almost always an FP32 loop, and exact mode would
//! reread it at about twice the cost.
//!
//! Escalation only ever improves a finished page: when the near-exact model
//! cannot be loaded or the rerun fails, the page keeps its fast result with
//! the reason in `OcrResult::escalation_error`, and after a failed load the
//! run stops trying.
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{FinishReason, OcrResult, Runner};
use crate::{
    auto::{Mode, ModelRequest, WeightsPlan, model_files_dir, resolve_weights},
    config::GenerationOptions,
};

/// The fast-mode attempt that an escalated page's near-exact rerun replaced
/// (`OcrResult::escalated_from`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EscalatedAttempt {
    pub mode: Option<Mode>,
    pub finish_reason: FinishReason,
    pub output_tokens: usize,
    pub total_ms: f64,
}

/// Whether `result` is reread in near-exact mode: a fast-mode page that the
/// repetition stop ended.
pub fn needs_escalation(result: &OcrResult) -> bool {
    result.mode == Some(Mode::Fast) && result.finish_reason == FinishReason::Repetition
}

/// The options that reread `fast`'s page: `options` at the size `fast` was
/// read at, so a routed page is reread at its final size (the cap when the
/// safety net fired) without being routed again.
pub fn rerun_options(options: &GenerationOptions, fast: &OcrResult) -> GenerationOptions {
    match &fast.route {
        Some(route) if route.safety_net.is_some() => options.at(crate::router::CAP),
        Some(route) => options.at(route.max_dimension),
        None => options.clone(),
    }
}

/// `rerun` recording the fast attempt it replaced: its total time includes
/// the attempt's, and a routed page keeps its route.
pub fn escalated(fast: OcrResult, mut rerun: OcrResult) -> OcrResult {
    rerun.escalated_from = Some(EscalatedAttempt {
        mode: fast.mode,
        finish_reason: fast.finish_reason,
        output_tokens: fast.output_tokens,
        total_ms: fast.timings.total_ms,
    });
    rerun.timings.total_ms += fast.timings.total_ms;
    rerun.route = fast.route;
    rerun
}

/// The weights of the near-exact escalation model for `request` (a fast
/// mode request): the near-exact packed file beside `--model-file` when there
/// is one (the published files sit together), else what `--mode near-exact`
/// resolves to in the model directory, its packed file or the checkpoint
/// quantized at load. Reads no tensors.
pub fn near_exact_weights(request: &ModelRequest<'_>) -> Result<WeightsPlan> {
    let beside = Mode::NearExact
        .packed_file_name()
        .map(|name| model_files_dir(request).join(name))
        .filter(|path| request.model_file.is_some() && path.is_file());
    resolve_weights(&ModelRequest {
        model_dir: request.model_dir,
        model_file: beside.as_deref(),
        mode: Some(Mode::NearExact),
        ..ModelRequest::default()
    })
}

/// The near-exact runner of `--escalate`, loaded when the first page needs
/// it; pages are reread one at a time.
pub struct Escalation<'a> {
    load: Box<dyn FnMut() -> Result<Runner> + 'a>,
    runner: Option<Runner>,
    /// Why loading failed: later pages keep their fast result without
    /// loading again.
    failed: Option<String>,
}

impl<'a> Escalation<'a> {
    /// `load` builds the near-exact runner (see [`near_exact_weights`]).
    pub fn new(load: impl FnMut() -> Result<Runner> + 'a) -> Self {
        Self {
            load: Box::new(load),
            runner: None,
            failed: None,
        }
    }

    /// `fast` unchanged when it does not need escalation; else the
    /// near-exact rerun of its page at `path` recording it ([`escalated`]),
    /// or, when the model cannot be loaded or the rerun fails, `fast` with
    /// the reason in `escalation_error`.
    pub fn escalate(&mut self, path: &Path, options: &GenerationOptions, mut fast: OcrResult) -> OcrResult {
        if !needs_escalation(&fast) {
            return fast;
        }
        match self.rerun(path, &rerun_options(options, &fast)) {
            Ok(rerun) => escalated(fast, rerun),
            Err(error) => {
                fast.escalation_error = Some(format!("{error:#}"));
                fast
            }
        }
    }

    /// The near-exact result of the page at `path`, loading the runner once.
    fn rerun(&mut self, path: &Path, options: &GenerationOptions) -> Result<OcrResult> {
        if let Some(reason) = &self.failed {
            bail!("the near-exact model did not load: {reason}");
        }
        let runner = match self.runner.take() {
            Some(runner) => runner,
            None => (self.load)().map_err(|error| {
                self.failed = Some(format!("{error:#}"));
                error.context("load the near-exact model")
            })?,
        };
        let rerun = runner
            .recognize_file(path, options)
            .context("reread the page in near-exact mode");
        self.runner = Some(runner);
        rerun
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::{Route, RoutedAttempt};

    /// A result as `mode` would report it, ending with `finish_reason`.
    fn result(mode: Option<Mode>, finish_reason: FinishReason, total_ms: f64) -> OcrResult {
        serde_json::from_value(serde_json::json!({
            "text": "x", "token_ids": [7, 7], "finish_reason": finish_reason, "width": 64, "height": 32,
            "input_tokens": 20, "output_tokens": 2, "precision": "w8-body-kv-q8", "backend": "avx2",
            "mode": mode, "teacher_forced": false,
            "timings": {"image_decode_ms": 1.0, "preprocessing_ms": 2.0, "prefill_ms": 3.0, "decode_ms": 4.0,
                        "total_ms": total_ms, "time_to_first_token_ms": 6.0}
        }))
        .unwrap()
    }

    fn route(max_dimension: u32, safety_net: bool) -> Route {
        Route {
            max_dimension,
            score_768: 0.5,
            score_1024: 0.5,
            line_height_px: 12.0,
            statistics_ms: 12.0,
            safety_net: safety_net.then_some(RoutedAttempt {
                max_dimension,
                finish_reason: FinishReason::Length,
                output_tokens: 9,
                total_ms: 5.0,
            }),
        }
    }

    #[test]
    fn only_fast_mode_repetition_stops_escalate() {
        use FinishReason::*;
        for (mode, reason, expected) in [
            (Some(Mode::Fast), Repetition, true),
            (Some(Mode::Fast), Length, false),
            (Some(Mode::Fast), Eos, false),
            (Some(Mode::NearExact), Repetition, false),
            (Some(Mode::Exact), Repetition, false),
            (None, Repetition, false),
        ] {
            assert_eq!(
                needs_escalation(&result(mode, reason, 1.0)),
                expected,
                "{mode:?} {reason:?}"
            );
        }
    }

    #[test]
    fn a_routed_page_is_reread_at_its_final_size_without_routing() {
        let options = GenerationOptions {
            route: true,
            ..GenerationOptions::default()
        };
        let mut fast = result(Some(Mode::Fast), FinishReason::Repetition, 1.0);
        assert_eq!(rerun_options(&options, &fast).route, options.route);
        fast.route = Some(route(768, false));
        let rerun = rerun_options(&options, &fast);
        assert!(!rerun.route && rerun.max_dimension == 768);
        // The safety net reran the page at the cap: that is the final size.
        fast.route = Some(route(768, true));
        let rerun = rerun_options(&options, &fast);
        assert!(!rerun.route && rerun.max_dimension == crate::router::CAP);
    }

    #[test]
    fn the_rerun_records_the_fast_attempt_and_keeps_the_route() {
        let mut fast = result(Some(Mode::Fast), FinishReason::Repetition, 100.0);
        fast.route = Some(route(1024, false));
        let rerun = result(Some(Mode::NearExact), FinishReason::Eos, 250.0);
        let merged = escalated(fast.clone(), rerun);
        assert_eq!(
            merged.escalated_from,
            Some(EscalatedAttempt {
                mode: Some(Mode::Fast),
                finish_reason: FinishReason::Repetition,
                output_tokens: 2,
                total_ms: 100.0,
            })
        );
        assert_eq!(
            (merged.mode, merged.finish_reason),
            (Some(Mode::NearExact), FinishReason::Eos)
        );
        assert_eq!(merged.timings.total_ms, 350.0);
        assert_eq!(merged.route, fast.route);
        let json = serde_json::to_value(&merged).unwrap();
        assert_eq!(json["escalated_from"]["mode"], "fast");
        assert_eq!(json["escalated_from"]["finish_reason"], "repetition");
        // Pages that did not escalate carry no record.
        assert!(serde_json::to_value(&fast).unwrap().get("escalated_from").is_none());
    }

    #[test]
    fn a_failed_escalation_keeps_the_fast_result_and_loads_only_once() {
        let mut loads = 0;
        let mut escalation = Escalation::new(|| {
            loads += 1;
            bail!("no near-exact model here")
        });
        let options = GenerationOptions::default();
        let page = Path::new("page.png");
        let eos = result(Some(Mode::Fast), FinishReason::Eos, 1.0);
        let passed = escalation.escalate(page, &options, eos.clone());
        assert_eq!((&passed.token_ids, &passed.escalation_error), (&eos.token_ids, &None));
        let looping = result(Some(Mode::Fast), FinishReason::Repetition, 7.0);
        for attempt in [
            "load the near-exact model: no near-exact model here",
            "did not load: no near-exact model here",
        ] {
            let kept = escalation.escalate(page, &options, looping.clone());
            assert_eq!(kept.token_ids, looping.token_ids);
            assert_eq!(
                (kept.mode, kept.finish_reason),
                (Some(Mode::Fast), FinishReason::Repetition)
            );
            assert_eq!((kept.escalated_from, kept.timings.total_ms), (None, 7.0));
            let error = kept.escalation_error.unwrap();
            assert!(error.contains(attempt), "{error}");
        }
        drop(escalation);
        assert_eq!(loads, 1);
        // The record keeps the reason; results that did not escalate carry none.
        let mut kept = looping;
        kept.escalation_error = Some("the near-exact model did not load: gone".to_owned());
        assert_eq!(
            serde_json::to_value(&kept).unwrap()["escalation_error"],
            "the near-exact model did not load: gone"
        );
        assert!(serde_json::to_value(&eos).unwrap().get("escalation_error").is_none());
    }
}
