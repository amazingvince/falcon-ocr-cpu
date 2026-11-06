//! Opt-in wall-clock phase split of forwards (`Tuning::phases`).
use std::time::Instant;

use super::PARALLEL_ROWS;

/// Report Linux resident memory in decimal MB when phase sampling is enabled.
/// Other platforms do not emit memory samples.
pub(crate) fn report_memory(enabled: bool, label: &str) {
    if !enabled {
        return;
    }
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let mb = |key: &str| {
            status
                .lines()
                .find_map(|line| {
                    line.strip_prefix(key)?
                        .split_whitespace()
                        .next()?
                        .parse::<f64>()
                        .ok()
                        .map(|kib| kib * 1024.0 / 1_000_000.0)
                })
                .unwrap_or(0.0)
        };
        crate::note!(
            "memory {label}: rss={:.3} hwm={:.3} anon={:.3} file={:.3}",
            mb("VmRSS:"),
            mb("VmHWM:"),
            mb("RssAnon:"),
            mb("RssFile:")
        );
    }
    #[cfg(not(target_os = "linux"))]
    let _ = label;
}

/// Opt-in wall-clock split of forwards on stderr. Prefill-sized forwards
/// print immediately; decode steps accumulate on the calling thread until
/// `report_decode_phases`. Disabled, a mark is a branch.
pub(super) struct PhaseClock {
    enabled: bool,
    rows: usize,
    last: Instant,
    totals: [f64; PHASES],
}
const PHASES: usize = 9;
const PHASE_NAMES: [&str; PHASES] = [
    "rope_factors",
    "norm+qkv",
    "split+qk_norm+rope",
    "cache_append",
    "attention",
    "wo+residual",
    "norm+w13+gate",
    "w2+residual",
    "final_norm+head",
];
/// Decode-step phase totals and step counts, by rows per step (1 = plain
/// decode, 2..=8 = draft verification).
type RowPhases = [([f64; PHASES], usize); crate::head_screen::MAX_ROWS + 1];
thread_local! {
    static DECODE_PHASES: std::cell::RefCell<RowPhases> =
        const { std::cell::RefCell::new([([0.0; PHASES], 0); crate::head_screen::MAX_ROWS + 1]) };
}
/// Charges the output head (final norm, screen or full logits) of a decode
/// step with `rows` rows to its `final_norm+head` phase when dropped.
pub(super) struct HeadClock {
    rows: usize,
    start: Instant,
}
impl HeadClock {
    /// Start timing a head for the row-count bucket, or return `None` when
    /// phase timing is disabled.
    pub(super) fn new(rows: usize, enabled: bool) -> Option<Self> {
        enabled.then(|| Self {
            rows,
            start: Instant::now(),
        })
    }
}
impl Drop for HeadClock {
    fn drop(&mut self) {
        let ms = self.start.elapsed().as_secs_f64() * 1000.0;
        DECODE_PHASES.with(|cell| {
            if let Some(bucket) = cell.borrow_mut().get_mut(self.rows) {
                bucket.0[PHASES - 1] += ms;
            }
        });
    }
}
impl PhaseClock {
    /// Start a forward phase clock; disabled clocks do not record on drop.
    pub(super) fn new(rows: usize, enabled: bool) -> Self {
        Self {
            enabled,
            rows,
            last: Instant::now(),
            totals: [0.0; PHASES],
        }
    }
    /// Charge the time since the previous mark to the phase that just ended.
    #[inline]
    pub(super) fn mark(&mut self, ended: usize) {
        if self.enabled {
            let now = Instant::now();
            self.totals[ended] += (now - self.last).as_secs_f64() * 1000.0;
            self.last = now;
        }
    }
}
impl Drop for PhaseClock {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        self.mark(PHASES - 1);
        if self.rows >= PARALLEL_ROWS {
            crate::note!("prefill phases rows={}: {}", self.rows, format_phases(&self.totals, 1));
        } else {
            DECODE_PHASES.with(|cell| {
                let mut state = cell.borrow_mut();
                if let Some(bucket) = state.get_mut(self.rows) {
                    for (total, value) in bucket.0.iter_mut().zip(&self.totals) {
                        *total += value;
                    }
                    bucket.1 += 1;
                }
            });
        }
    }
}
fn format_phases(totals: &[f64; PHASES], steps: usize) -> String {
    let per = steps.max(1) as f64;
    PHASE_NAMES
        .iter()
        .zip(totals)
        .map(|(name, ms)| format!("{name}={:.3}ms", ms / per))
        .collect::<Vec<_>>()
        .join(" ")
}
/// Print and reset this thread's accumulated decode-step phases (per step);
/// a no-op unless `enabled`.
pub(crate) fn report_decode_phases(enabled: bool) {
    if !enabled {
        return;
    }
    DECODE_PHASES.with(|cell| {
        let mut state = cell.borrow_mut();
        for (rows, (totals, steps)) in state.iter().enumerate() {
            if *steps == 0 {
                continue;
            }
            let label = if rows == 1 {
                String::new()
            } else {
                format!(" ({rows} rows)")
            };
            crate::note!(
                "decode phases per step{label} over {steps} steps: {}",
                format_phases(totals, *steps)
            );
        }
        *state = [([0.0; PHASES], 0); crate::head_screen::MAX_ROWS + 1];
    });
}
