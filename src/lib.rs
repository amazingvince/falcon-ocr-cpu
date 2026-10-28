//! CPU inference for the pinned Falcon-OCR v1.5 architecture.
//!
//! Numerical correctness and performance qualification are tracked separately;
//! consult the reference reports before treating a backend as GPU-equivalent.

/// `eprintln!` for a diagnostic that must not stop a run: when stderr is
/// closed (a pager that quit) the message is dropped instead of panicking.
macro_rules! note {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr().lock(), $($arg)*);
    }};
}
pub(crate) use note;

/// Host capabilities, model discovery and execution plans.
pub mod auto;
/// Long runs of `falcon-ocr run`: input lists, records, `--resume`.
pub mod book;
mod buf;
/// Runner options shared by the product and evaluation command lines.
pub mod cli;
/// Model identity and validated inference options.
pub mod config;
/// Logical and physical CPU counts for thread planning.
pub mod cpu;
mod draft;
mod draft_head;
mod head_screen;
/// CPU projection, normalization and attention dispatch.
pub mod kernels;
/// Checkpoint loading, packed files and forward execution.
pub mod model;
/// Experimental AVX2 weight packing for joint decode projections.
pub mod packed_kernels;
/// Page decoding, resizing, cropping and patch construction.
pub mod preprocess;
mod repetition;
/// Image statistics and the opt-in resolution router.
pub mod router;
/// Recognition sessions, generation results and page scheduling.
pub mod runner;
mod simd;
mod team;
/// Pinned tokenizer loading, prompt construction and output decoding.
pub mod tokenizer;
/// Optional execution callbacks and named tensor capture.
pub mod trace;
mod tune;

pub use auto::{HostInfo, Mode, Resolved};
pub use config::{
    Backend, CacheLayout, DecodeThreads, DraftKv, Drafter, ExpMode, GenerationOptions, HeadMode, MaxDimension,
    ModelConfig, PrefillBf16, RunnerConfig, Speculation, Tuning, WeightLayout,
};
pub use model::{Model, WeightsSource};
pub use runner::{FinishReason, OcrResult, Pipeline, Runner, Timings};
pub use tune::TuneReport;

/// Weight and KV-cache storage profiles and integer codecs.
pub mod quant;
