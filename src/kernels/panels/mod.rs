//! Quantized prefill GEMM panels: INT8/INT16 codes with FP32 scales, and the
//! AVX512-BF16 variant.
pub(crate) mod panel;
#[cfg(target_arch = "x86_64")]
pub(crate) mod panel_bf16;
