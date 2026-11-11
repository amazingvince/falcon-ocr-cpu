//! What `kernels::prefill_plan` and the byte budgets read about the body
//! matrices of a loaded model: their common code width, their exception
//! columns and whether the INT16 prefill panels hold them.
use super::Model;
use crate::quant::linear::QuantLinear;

impl Model {
    /// Bit width of the quantized body when every body matrix of every layer
    /// is quantized to one width with panel-shaped output dims (the prefill
    /// GEMM's precondition); `None` for an FP32 or mixed body.
    pub fn body_bits(&self) -> Option<u32> {
        let mut bits = None;
        for layer in &self.layers {
            for w in [&layer.qkv, &layer.wo, &layer.w13, &layer.w2] {
                let q = w.quantized.as_ref().filter(|q| q.panel_shaped())?;
                match bits {
                    None => bits = Some(q.bits()),
                    Some(b) if b == q.bits() => {}
                    Some(_) => return None,
                }
            }
        }
        bits
    }
    /// Bytes of the body's exception columns (indices and FP32 weights),
    /// which a decode step reads on top of the codes; zero without any.
    pub fn exception_bytes(&self) -> usize {
        self.quantized_body().map(QuantLinear::exception_bytes).sum()
    }
    /// Whether a quantized body matrix does not fit the INT16
    /// prefill panels (`QuantLinear::fits_i16_panels`): 16-bit codes, a W8
    /// overlay's groups of 32, or a dimension that does not fill panels.
    pub fn i16_misfit(&self) -> bool {
        self.quantized_body().any(|q| !q.fits_i16_panels())
    }
    /// The body matrices that are quantized.
    fn quantized_body(&self) -> impl Iterator<Item = &QuantLinear> {
        self.layers
            .iter()
            .flat_map(|layer| [&layer.qkv, &layer.wo, &layer.w13, &layer.w2])
            .filter_map(|w| w.quantized.as_deref())
    }
}
