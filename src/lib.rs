//! CPU inference for the pinned Falcon-OCR v1.5 architecture.
//!
//! Numerical correctness and performance qualification are tracked separately;
//! consult the reference reports before treating a backend as GPU-equivalent.
pub mod bf16_attention;
pub mod bf16_kernels;
pub mod bf16_model;
pub mod bf16_ops;
pub mod bf16_runner;
pub mod config;
mod head_screen;
pub mod kernels;
pub mod model;
#[cfg(test)]
mod numerical_diagnostics;
pub mod packed_kernels;
pub mod preprocess;
pub mod runner;
pub mod tokenizer;
pub mod trace;

pub use bf16_model::Bf16Model;
pub use bf16_runner::{Bf16Result, Bf16Runner};
pub use config::{
    Backend, CacheLayout, GenerationOptions, HeadMode, ModelConfig, Precision, RunnerConfig,
    WeightLayout,
};
pub use model::Model;
pub use runner::{FinishReason, OcrResult, Runner, Timings};

pub mod attempt;
