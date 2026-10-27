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

pub mod auto;
/// Long runs of `falcon-ocr run`: input lists, records, `--resume`.
pub mod book;
mod buf;
pub mod cli;
pub mod config;
pub mod cpu;
mod draft;
mod draft_head;
mod head_screen;
pub mod kernels;
pub mod model;
pub mod packed_kernels;
pub mod preprocess;
mod repetition;
pub mod router;
pub mod runner;
mod simd;
mod team;
pub mod tokenizer;
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

pub mod quant;
