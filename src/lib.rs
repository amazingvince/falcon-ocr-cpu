//! CPU inference for the pinned Falcon-OCR v1.5 architecture.
//!
//! Numerical correctness and performance qualification are tracked separately;
//! consult the reference reports before treating a backend as GPU-equivalent.
mod buf;
pub mod config;
pub mod cpu;
mod draft;
mod head_screen;
pub mod kernels;
pub mod model;
pub mod packed_kernels;
pub mod preprocess;
mod repetition;
pub mod runner;
mod simd;
mod team;
pub mod tokenizer;
pub mod trace;
mod tune;

pub use config::{
    Backend, CacheLayout, ExpMode, GenerationOptions, HeadMode, ModelConfig, PrefillBf16, RunnerConfig, Tuning,
    WeightLayout,
};
pub use model::Model;
pub use runner::{FinishReason, OcrResult, Runner, Timings};

pub mod attempt;
