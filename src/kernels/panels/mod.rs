//! Quantized prefill GEMM panels: INT8/INT16 codes with FP32 scales, and the
//! INT16-activation variant for 8-bit codes.
mod driver;
/// FP32 panels dequantized from integer weights.
pub(crate) mod panel;
#[cfg(target_arch = "x86_64")]
/// INT16 panels of 8-bit codes with INT16 activations (VNNI and AVX2).
pub(crate) mod panel_i16;
