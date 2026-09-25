//! Command-line pieces shared by `falcon-ocr` and `falcon-ocr-eval`: the
//! runner flags and their layering over a base configuration, and the
//! doctor report.
use anyhow::Result;
use clap::{ArgAction, Args};

use crate::{
    auto::{self, ModelRequest},
    config::{Backend, DecodeThreads, Drafter, ExpMode, HeadMode, RunnerConfig, Speculation, Tuning},
};

/// Runner flags of both binaries. Every flag is optional: an unset flag keeps
/// the base configuration's value ([`RunnerConfig::default`] for
/// recognition, [`RunnerConfig::reference`] for traces and the eval binary).
#[derive(Args, Clone, Debug, Default)]
pub struct RunnerArgs {
    /// Prefill and pool threads [default: all logical CPUs; 16 in falcon-ocr-eval].
    #[arg(long, global = true)]
    pub threads: Option<usize>,
    /// Kernels: `auto` takes the fastest this CPU runs (AVX2 decode, AVX-512
    /// prefill tiles and BF16 prefill attention where present, NEON on
    /// aarch64); `avx2` is 8-lane FP32 everywhere; `scalar` [default: auto].
    #[arg(long, value_enum, global = true)]
    pub backend: Option<Backend>,
    /// Requests decoded jointly, 1..=8 [default: 1].
    #[arg(long, global = true)]
    pub batch_size: Option<usize>,
    /// FP32 greedy head [default: screened for recognition, full for traces
    /// and falcon-ocr-eval]: `screened` selects the same tokens as `full`
    /// through an exact INT8 screen (53 MB extra); traces always record full
    /// logits.
    #[arg(long, value_enum, global = true)]
    pub head: Option<HeadMode>,
    /// Stop a page once it repeats a cycle of at most 128 tokens for at
    /// least max(256, 4 * cycle) tokens (finish_reason "repetition"); output
    /// up to that step is unchanged [default: on for recognition, off for
    /// falcon-ocr-eval]. `--stop-repetition=false` lets loops run to
    /// --max-new-tokens.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", action = ArgAction::Set, global = true)]
    pub stop_repetition: Option<bool>,
    /// Decode threads: a number, `pool` (as many as --threads) or `auto`
    /// [default: auto for recognition, pool for falcon-ocr-eval]. Decode is
    /// memory-bound, so past bandwidth saturation extra threads and SMT
    /// siblings only contend; `auto` times a few team sizes on the first
    /// decode steps and keeps the smallest within 2% of the fastest (tokens
    /// never depend on it). Prefill uses every thread in --threads.
    #[arg(long, global = true)]
    pub decode_threads: Option<DecodeThreads>,
    /// Speculative decoding: verify up to N tokens drafted from earlier output
    /// in one step (0 = off, at most 7) [default: 4 for recognition, off for
    /// falcon-ocr-eval]. Every verified token is the model's own greedy
    /// choice, so outputs are unchanged. Drafting switches itself off while
    /// drafts are rejected too often to pay (normal text) and on for
    /// repetitive output (tables, loops). Single pages with a split KV cache.
    #[arg(long, global = true)]
    pub speculate: Option<usize>,
    /// Minimum n-gram match in the earlier output for a draft [default: 2].
    #[arg(long, global = true)]
    pub speculate_min_match: Option<usize>,
    /// Treat the images of one run as pages of one document: drafts may also
    /// continue full 4-token matches from earlier pages (running headers,
    /// names, repeated table headers). Outputs are unchanged [default: on for
    /// recognition, off for falcon-ocr-eval].
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", action = ArgAction::Set, global = true)]
    pub document_drafts: Option<bool>,
    /// A trained draft head file for speculative decoding
    /// (`research/draft-head`); selects `--drafter head` unless set.
    #[arg(long, global = true)]
    pub draft_head: Option<std::path::PathBuf>,
    /// Draft source: `ngram` (matches in earlier output), `head` (the
    /// --draft-head model) or `both` (n-gram matches first) [default: head
    /// with --draft-head, otherwise ngram]. Outputs are unchanged.
    #[arg(long, value_enum, global = true)]
    pub drafter: Option<Drafter>,
    /// The draft head drafts while its top token's probability is at least
    /// this [default: 0.35, the best of 0.25-0.6 on calibration pages].
    #[arg(long, global = true)]
    pub draft_confidence: Option<f32>,
    /// Prefill exp: `fast` (token-identical on calibration; the recognition
    /// default) or `exact` (the platform expf, which traces use).
    #[arg(long, value_enum, hide = true, global = true)]
    pub exp: Option<ExpMode>,
    /// Experiment knobs as `key=value` (`prefill-bf16=off|attention|all`,
    /// `split-chunks=1..4`, `phases=1`, `prefill-profile=1`); repeatable.
    #[arg(long = "tune", value_name = "KEY=VALUE", hide = true, global = true)]
    pub tune: Vec<String>,
}

impl RunnerArgs {
    /// `base` with every set flag layered on top.
    pub fn apply(&self, base: RunnerConfig) -> Result<RunnerConfig> {
        let min_match = self
            .speculate_min_match
            .or(base.speculation.map(|s| s.min_match))
            .unwrap_or(2);
        let speculation = match self.speculate {
            Some(0) => None,
            Some(max_draft) => Some(Speculation { max_draft, min_match }),
            None => base.speculation.map(|s| Speculation { min_match, ..s }),
        };
        Ok(RunnerConfig {
            threads: self.threads.unwrap_or(base.threads),
            backend: self.backend.unwrap_or(base.backend),
            batch_size: self.batch_size.unwrap_or(base.batch_size),
            head: self.head.unwrap_or(base.head),
            repetition_stop: self.stop_repetition.unwrap_or(base.repetition_stop),
            decode_threads: self.decode_threads.unwrap_or(base.decode_threads),
            speculation,
            document_drafts: self.document_drafts.unwrap_or(base.document_drafts),
            drafter: self.drafter.unwrap_or(if self.draft_head.is_some() {
                Drafter::Head
            } else {
                base.drafter
            }),
            draft_head: self.draft_head.clone().or(base.draft_head.clone()),
            draft_confidence: self.draft_confidence.unwrap_or(base.draft_confidence),
            exp: self.exp.unwrap_or(base.exp),
            tuning: Tuning::from_pairs(&self.tune)?,
            ..base
        })
    }
}

/// Print the doctor report for `request` and `config`: JSON, or the text
/// form with `text`; `load` and `probe` as in [`auto::doctor`].
pub fn print_doctor(
    request: &ModelRequest<'_>,
    config: &RunnerConfig,
    text: bool,
    load: bool,
    probe: bool,
) -> Result<()> {
    let report = auto::doctor(request, config, load, probe)?;
    if text {
        print!("{report}");
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Test {
        #[command(flatten)]
        runner: RunnerArgs,
    }

    #[test]
    fn unset_flags_keep_the_base_and_set_flags_override_it() {
        let none = Test::parse_from(["t"]).runner.apply(RunnerConfig::default()).unwrap();
        let auto = RunnerConfig::default();
        assert_eq!(none.threads, auto.threads);
        assert_eq!(none.head, auto.head);
        assert_eq!(none.speculation, auto.speculation);
        assert_eq!(none.repetition_stop, auto.repetition_stop);
        let reference = Test::parse_from(["t"]).runner.apply(RunnerConfig::reference()).unwrap();
        assert_eq!(reference.head, RunnerConfig::reference().head);
        assert!(reference.speculation.is_none() && !reference.repetition_stop);
        let set = Test::parse_from([
            "t",
            "--threads",
            "3",
            "--stop-repetition=false",
            "--speculate",
            "2",
            "--speculate-min-match",
            "3",
            "--decode-threads",
            "pool",
            "--tune",
            "phases=1",
        ])
        .runner
        .apply(RunnerConfig::default())
        .unwrap();
        assert_eq!(set.threads, 3);
        assert!(!set.repetition_stop);
        assert_eq!(
            set.speculation,
            Some(Speculation {
                max_draft: 2,
                min_match: 3
            })
        );
        assert_eq!(set.decode_threads, DecodeThreads::Pool);
        assert!(set.tuning.phases);
        // Bare boolean flags mean true; `--speculate 0` turns speculation off.
        let bare = Test::parse_from(["t", "--stop-repetition", "--document-drafts", "--speculate", "0"])
            .runner
            .apply(RunnerConfig::reference())
            .unwrap();
        assert!(bare.repetition_stop && bare.document_drafts && bare.speculation.is_none());
    }
}
