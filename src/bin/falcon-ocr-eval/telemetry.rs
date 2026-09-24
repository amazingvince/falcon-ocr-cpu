//! Operational counters and process memory for the eval reports.
use serde::Serialize;

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
            fn GetProcessMemoryInfo(process: *mut std::ffi::c_void, counters: *mut Counters, cb: u32) -> i32;
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
impl falcon_ocr::trace::Trace for Telemetry {
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
    fn occupancy_counts_live_rows_not_configured_batch() {
        use falcon_ocr::trace::Trace;
        let mut t = Telemetry::default();
        t.decode_step(2, 1.0, 20);
        t.decode_step(1, 2.0, 10);
        assert_eq!(t.decode_forwards, 2);
        assert_eq!(t.decoded_request_rows, 3);
        assert_eq!(t.active_rows_histogram[1..=2], [1, 1]);
    }
}
