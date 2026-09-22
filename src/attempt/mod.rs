//! Opt-in, model-specific integrated experiment. No profile is quality-qualified.
//! FP32 graph arithmetic is unchanged unless a selected weight/cache is encoded.
pub(crate) mod prefix;
pub mod quant;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    #[default]
    Reference,
    Hygiene,
    SplitF32,
    KvBf16,
    KvQ8,
    W8Body,
    W8All,
    W8BodyKvBf16,
    W8AllKvBf16,
    W8AllKvQ8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixMode {
    Reference,
    SplitF32,
    SplitBf16,
    SplitQ8,
}
impl Profile {
    pub fn quantizes_body(self) -> bool {
        matches!(
            self,
            Self::W8Body | Self::W8All | Self::W8BodyKvBf16 | Self::W8AllKvBf16 | Self::W8AllKvQ8
        )
    }
    pub fn quantizes_head(self) -> bool {
        matches!(self, Self::W8All | Self::W8AllKvBf16 | Self::W8AllKvQ8)
    }
    pub fn prefix_mode(self) -> PrefixMode {
        match self {
            Self::SplitF32 => PrefixMode::SplitF32,
            Self::KvBf16 | Self::W8BodyKvBf16 | Self::W8AllKvBf16 => PrefixMode::SplitBf16,
            Self::KvQ8 | Self::W8AllKvQ8 => PrefixMode::SplitQ8,
            _ => PrefixMode::Reference,
        }
    }
    pub fn memory_hygiene(self) -> bool {
        self != Self::Reference
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Hygiene => "hygiene",
            Self::SplitF32 => "split-f32",
            Self::KvBf16 => "kv-bf16",
            Self::KvQ8 => "kv-q8",
            Self::W8Body => "w8-body",
            Self::W8All => "w8-all",
            Self::W8BodyKvBf16 => "w8-body-kv-bf16",
            Self::W8AllKvBf16 => "w8-all-kv-bf16",
            Self::W8AllKvQ8 => "w8-all-kv-q8",
        }
    }
}

/// Operational counters, not OCR-quality assertions. No per-step heap growth.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Telemetry {
    pub decode_forwards: usize,
    pub decoded_request_rows: usize,
    /// Bin i is the number of joint forwards with i active requests (0..=8).
    pub active_rows_histogram: [usize; 9],
    pub decode_kernel_ms: f64,
    pub prefix_seals: usize,
    pub prefix_seal_ms: f64,
    pub kv_bytes_before_seal_sum: usize,
    pub kv_bytes_after_seal_sum: usize,
    pub kv_allocated_high_water: usize,
    pub retired_cache_bytes_sum: usize,
}
impl crate::trace::Trace for Telemetry {
    fn enabled(&self) -> bool {
        false
    }
    fn tensor(&mut self, _: &str, _: &[usize], _: &[f32]) -> anyhow::Result<()> {
        Ok(())
    }
    fn decode_step(&mut self, rows: usize, ms: f64, kv_bytes: usize) {
        self.decode_forwards += 1;
        self.decoded_request_rows += rows;
        if rows <= 8 {
            self.active_rows_histogram[rows] += 1;
        }
        self.decode_kernel_ms += ms;
        self.kv_allocated_high_water = self.kv_allocated_high_water.max(kv_bytes);
    }
    fn prefix_sealed(&mut self, before: usize, after: usize, ms: f64) {
        self.prefix_seals += 1;
        self.prefix_seal_ms += ms;
        self.kv_bytes_before_seal_sum += before;
        self.kv_bytes_after_seal_sum += after;
        self.kv_allocated_high_water = self.kv_allocated_high_water.max(before).max(after);
    }
    fn cache_retired(&mut self, bytes: usize) {
        self.retired_cache_bytes_sum += bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ablations_do_not_silently_quantize_endpoints() {
        assert!(!Profile::KvBf16.quantizes_body());
        assert!(Profile::W8Body.quantizes_body());
        assert!(!Profile::W8Body.quantizes_head());
        assert!(Profile::W8All.quantizes_head());
        assert_eq!(Profile::W8Body.prefix_mode(), PrefixMode::Reference);
        assert!(!Profile::Reference.memory_hygiene());
    }
    #[test]
    fn occupancy_counts_live_rows_not_configured_batch() {
        use crate::trace::Trace;
        let mut t = Telemetry::default();
        t.decode_step(2, 1.0, 20);
        t.decode_step(1, 2.0, 10);
        assert_eq!(t.decode_forwards, 2);
        assert_eq!(t.decoded_request_rows, 3);
        assert_eq!(t.active_rows_histogram[1..=2], [1, 1]);
    }
}
