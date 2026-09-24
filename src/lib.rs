//! CPU inference for the pinned Falcon-OCR v1.5 architecture.
//!
//! Numerical correctness and performance qualification are tracked separately;
//! consult the reference reports before treating a backend as GPU-equivalent.
pub mod config;
pub mod cpu;
mod head_screen;
pub mod kernels;
pub mod model;
pub mod packed_kernels;
pub mod preprocess;
mod repetition;
pub mod runner;
mod simd;
mod team;
mod tune;
mod buf;
mod draft;
pub mod tokenizer;
pub mod trace;

pub use config::{
    Backend, CacheLayout, GenerationOptions, HeadMode, ModelConfig, RunnerConfig, WeightLayout,
};
pub use model::Model;
pub use runner::{FinishReason, OcrResult, Runner, Timings};

pub mod attempt;
