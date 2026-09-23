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
    W8BodyKvQ8,
    W8AllKvBf16,
    W8AllKvQ8,
    /// FP32 weights; Q8 KV with per-channel key scales (`SplitQ8Kc`).
    KvQ8Kc,
    /// W8 body weights; Q8 KV with per-channel key scales (`SplitQ8Kc`).
    W8BodyKvQ8Kc,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixMode {
    Reference,
    SplitF32,
    SplitBf16,
    SplitQ8,
    /// Q8 records whose key channels share one scale per 128-record tile
    /// (KIVI-style); values keep per-record 32-element scales.
    SplitQ8Kc,
}
impl Profile {
    pub fn quantizes_body(self) -> bool {
        matches!(
            self,
            Self::W8Body
                | Self::W8All
                | Self::W8BodyKvBf16
                | Self::W8BodyKvQ8
                | Self::W8AllKvBf16
                | Self::W8AllKvQ8
                | Self::W8BodyKvQ8Kc
        )
    }
    pub fn quantizes_head(self) -> bool {
        matches!(self, Self::W8All | Self::W8AllKvBf16 | Self::W8AllKvQ8)
    }
    pub fn prefix_mode(self) -> PrefixMode {
        match self {
            Self::SplitF32 => PrefixMode::SplitF32,
            Self::KvBf16 | Self::W8BodyKvBf16 | Self::W8AllKvBf16 => PrefixMode::SplitBf16,
            Self::KvQ8 | Self::W8BodyKvQ8 | Self::W8AllKvQ8 => PrefixMode::SplitQ8,
            Self::KvQ8Kc | Self::W8BodyKvQ8Kc => PrefixMode::SplitQ8Kc,
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
            Self::W8BodyKvQ8 => "w8-body-kv-q8",
            Self::W8AllKvBf16 => "w8-all-kv-bf16",
            Self::W8AllKvQ8 => "w8-all-kv-q8",
            Self::KvQ8Kc => "kv-q8-kc",
            Self::W8BodyKvQ8Kc => "w8-body-kv-q8-kc",
        }
    }
}

/// Process working-set and commit counters (bytes) at the time of the call.
/// Peak values cover the whole process lifetime, including load/import.
pub fn process_memory() -> serde_json::Value {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let bytes = |key: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok())
                .map(|kb| kb * 1024)
        };
        serde_json::json!({"resident_bytes":bytes("VmRSS:"),"peak_resident_bytes":bytes("VmHWM:")})
    }
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct Counters {
            cb: u32,
            faults: u32,
            sizes: [usize; 8],
        }
        #[link(name = "psapi")]
        unsafe extern "system" {
            fn GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                counters: *mut Counters,
                cb: u32,
            ) -> i32;
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
        }
        let mut counters = Counters {
            cb: std::mem::size_of::<Counters>() as u32,
            faults: 0,
            sizes: [0; 8],
        };
        // SAFETY: ABI matches PROCESS_MEMORY_COUNTERS; the pseudo-handle is valid
        // and the writable struct has the advertised size.
        let ok = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<Counters>() as u32,
            )
        };
        if ok == 0 {
            serde_json::json!({"unavailable":true})
        } else {
            serde_json::json!({"resident_bytes":counters.sizes[1],"peak_resident_bytes":counters.sizes[0],
                "private_commit_bytes":counters.sizes[6],"peak_private_commit_bytes":counters.sizes[7]})
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        serde_json::json!({"unavailable":true})
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
    /// Screened vocabulary-head selections and full-head fallbacks.
    pub head_screen_steps: usize,
    pub head_screen_fallbacks: usize,
    pub head_candidates_sum: usize,
    pub head_candidates_max: usize,
    /// Recomputed-row buckets: 1, 2-4, 5-16, 17-64, 65-256, 257-1024, >1024.
    pub head_candidates_histogram: [usize; 7],
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
    fn head_screen(&mut self, candidates: usize, fallback: bool) {
        self.head_screen_steps += 1;
        self.head_screen_fallbacks += usize::from(fallback);
        self.head_candidates_sum += candidates;
        self.head_candidates_max = self.head_candidates_max.max(candidates);
        let bucket = match candidates {
            0..=1 => 0,
            2..=4 => 1,
            5..=16 => 2,
            17..=64 => 3,
            65..=256 => 4,
            257..=1024 => 5,
            _ => 6,
        };
        self.head_candidates_histogram[bucket] += 1;
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
