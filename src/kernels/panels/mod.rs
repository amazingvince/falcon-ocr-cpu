//! Quantized prefill GEMM panels: INT8/INT16 codes with FP32 scales, and the
//! AVX512-BF16 and INT16-activation variants for 8-bit codes.
mod driver;
/// FP32 panels dequantized from integer weights.
pub(crate) mod panel;
#[cfg(target_arch = "x86_64")]
/// BF16 panels and AVX512-BF16 projection kernels.
pub(crate) mod panel_bf16;
#[cfg(target_arch = "x86_64")]
/// INT16 panels of 8-bit codes with INT16 activations (VNNI and AVX2).
pub(crate) mod panel_i16;
