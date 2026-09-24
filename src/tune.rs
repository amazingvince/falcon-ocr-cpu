//! Automatic decode thread count.
//!
//! Decode is memory-bound: past the point where the cores saturate memory
//! bandwidth, extra threads only add synchronization, and SMT siblings or
//! efficiency cores can make a step slower. Where that point lies depends on
//! the machine, so [`Tuner`] measures it on real decode steps. After a short
//! warm-up it runs each candidate team size for a few consecutive steps,
//! round-robin over several rounds (so the slowly growing context affects
//! every candidate alike), then keeps the smallest size whose median step time
//! is within [`TOLERANCE`] of the fastest. Team size never changes arithmetic:
//! each task computes the same values with any team.

/// Steps before sampling starts (cold caches right after prefill).
const WARMUP: usize = 4;
/// Consecutive steps per candidate in one round.
const RUN: usize = 4;
/// Rounds over all candidates.
const ROUNDS: usize = 3;
/// A smaller team within this fraction of the fastest median is preferred.
const TOLERANCE: f64 = 0.02;

/// Candidate decode team sizes, ascending: half, three quarters and all of
/// the physical cores, plus the midpoint towards the logical CPU count on SMT
/// machines. Every size is at most `limit` (the prefill pool size).
pub(crate) fn candidate_sizes(physical: usize, logical: usize, limit: usize) -> Vec<usize> {
    let physical = physical.max(1);
    let mut sizes = vec![(physical / 2).max(1), (physical * 3 / 4).max(1), physical];
    if logical > physical {
        sizes.push((physical + logical) / 2);
    }
    let mut sizes: Vec<usize> = sizes.into_iter().map(|s| s.clamp(1, limit.max(1))).collect();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The candidate decode team sizes for a `limit`-thread prefill pool on
/// `host`, and the index of the size the tuner starts on (the physical core
/// count). Draft verification is compute-heavy (5-row steps: 7.6 ms on 12 of
/// 16 cores, 9.1 ms on 8), while single steps are bandwidth-bound and time
/// the same on half and three quarters of the cores; with speculation on, the
/// half-core candidate is dropped.
pub(crate) fn auto_candidates(host: &crate::auto::HostInfo, limit: usize, speculating: bool) -> (Vec<usize>, usize) {
    let mut sizes = candidate_sizes(host.physical_cores, host.logical_cpus, limit);
    if speculating && sizes.len() > 1 {
        sizes.remove(0);
    }
    let physical = host.physical_cores.min(limit);
    let start = sizes.iter().position(|&s| s >= physical).unwrap_or(sizes.len() - 1);
    (sizes, start)
}

pub(crate) struct Tuner {
    sizes: Vec<usize>,
    samples: Vec<Vec<f64>>,
    start: usize,
    steps: usize,
    chosen: Option<usize>,
}

impl Tuner {
    /// `sizes` ascending; decoding starts on `sizes[start]`.
    pub(crate) fn new(sizes: Vec<usize>, start: usize) -> Self {
        assert!(!sizes.is_empty() && start < sizes.len());
        let samples = vec![Vec::new(); sizes.len()];
        Self {
            sizes,
            samples,
            start,
            steps: 0,
            chosen: None,
        }
    }

    #[cfg(test)]
    fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Index of the team size to use for the next step.
    pub(crate) fn current(&self) -> usize {
        match self.chosen {
            Some(index) => index,
            None if self.steps < WARMUP => self.start,
            None => ((self.steps - WARMUP) / RUN) % self.sizes.len(),
        }
    }

    /// The chosen team size once tuning has finished.
    pub(crate) fn chosen(&self) -> Option<usize> {
        self.chosen.map(|index| self.sizes[index])
    }

    /// Record the duration of a step that ran on `sizes[index]`.
    pub(crate) fn record(&mut self, index: usize, ms: f64) {
        if self.chosen.is_some() {
            return;
        }
        if self.steps >= WARMUP && index == self.current() {
            self.samples[index].push(ms);
        }
        self.steps += 1;
        if self.samples.iter().all(|s| s.len() >= RUN * ROUNDS) {
            let index = self.pick();
            self.chosen = Some(index);
            let medians: Vec<String> = self
                .sizes
                .iter()
                .zip(&self.samples)
                .map(|(size, s)| format!("{size}:{:.2}", median(s)))
                .collect();
            eprintln!(
                "decode threads: {} (auto; median ms/step {})",
                self.sizes[index],
                medians.join(" ")
            );
        }
    }

    fn pick(&self) -> usize {
        let medians: Vec<f64> = self.samples.iter().map(|s| median(s)).collect();
        let best = medians.iter().copied().fold(f64::INFINITY, f64::min);
        medians
            .iter()
            .position(|&m| m <= best * (1.0 + TOLERANCE))
            .unwrap_or(self.start)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        f64::NAN
    } else if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_cover_half_to_smt_midpoint() {
        assert_eq!(candidate_sizes(16, 32, 32), vec![8, 12, 16, 24]);
        assert_eq!(candidate_sizes(8, 8, 8), vec![4, 6, 8]);
        assert_eq!(candidate_sizes(16, 32, 12), vec![8, 12]);
        assert_eq!(candidate_sizes(1, 1, 1), vec![1]);
    }

    #[test]
    fn auto_candidates_start_on_the_physical_cores_and_drop_the_half_team_when_speculating() {
        let host = |physical, logical| crate::auto::HostInfo {
            os: "test".into(),
            arch: "test".into(),
            logical_cpus: logical,
            physical_cores: physical,
            smt: logical > physical,
            performance_cores: None,
            features: Default::default(),
        };
        assert_eq!(auto_candidates(&host(16, 32), 32, false), (vec![8, 12, 16, 24], 2));
        assert_eq!(auto_candidates(&host(16, 32), 32, true), (vec![12, 16, 24], 1));
        assert_eq!(auto_candidates(&host(8, 8), 8, false), (vec![4, 6, 8], 2));
        assert_eq!(auto_candidates(&host(16, 32), 12, true), (vec![12], 0));
        assert_eq!(auto_candidates(&host(1, 1), 1, true), (vec![1], 0));
    }

    #[test]
    fn picks_smallest_size_within_tolerance() {
        let mut tuner = Tuner::new(vec![8, 12, 16, 24], 2);
        let cost = |size: usize| match size {
            8 => 10.0,
            12 => 9.1,
            16 => 9.0,
            _ => 9.8,
        };
        for _ in 0..WARMUP {
            assert_eq!(tuner.current(), 2);
            tuner.record(2, 50.0);
        }
        while tuner.chosen().is_none() {
            let index = tuner.current();
            tuner.record(index, cost(tuner.sizes()[index]));
        }
        assert_eq!(tuner.chosen(), Some(12));
        assert_eq!(tuner.current(), 1);
    }
}
