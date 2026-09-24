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
    // On a hybrid CPU the performance cores alone are a candidate: a team
    // that never lands on an efficiency core.
    if let Some(p) = host.performance_cores.filter(|&p| p < host.physical_cores) {
        sizes.push(p.clamp(1, limit.max(1)));
        sizes.sort_unstable();
        sizes.dedup();
    }
    let physical = host.physical_cores.min(limit);
    let start = sizes.iter().position(|&s| s >= physical).unwrap_or(sizes.len() - 1);
    (sizes, start)
}

/// What the tuner measured and chose (`Runner::decode_tuning`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TuneReport {
    /// Candidate team sizes, ascending.
    pub candidates: Vec<usize>,
    /// Median milliseconds of a single-row decode step per candidate.
    pub median_ms: Vec<f64>,
    /// Median milliseconds of the draft-verification steps that ran on each
    /// candidate while tuning (`None` when none did). Reporting only: the
    /// choice rests on single-row steps.
    pub verify_median_ms: Vec<Option<f64>>,
    /// The team size chosen.
    pub chosen: usize,
    /// Decode steps the tuner used before choosing (warm-up included).
    pub steps: usize,
}

impl std::fmt::Display for TuneReport {
    /// `decode threads: 12 (auto; median ms/step 12:9.93 16:10.47 24:10.67)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let medians: Vec<String> = self
            .candidates
            .iter()
            .zip(&self.median_ms)
            .map(|(size, ms)| format!("{size}:{ms:.2}"))
            .collect();
        write!(
            f,
            "decode threads: {} (auto; median ms/step {})",
            self.chosen,
            medians.join(" ")
        )?;
        let verify: Vec<String> = self
            .candidates
            .iter()
            .zip(&self.verify_median_ms)
            .filter_map(|(size, ms)| ms.map(|ms| format!("{size}:{ms:.2}")))
            .collect();
        if !verify.is_empty() {
            write!(f, "; verify steps {}", verify.join(" "))?;
        }
        Ok(())
    }
}

pub(crate) struct Tuner {
    sizes: Vec<usize>,
    samples: Vec<Vec<f64>>,
    verify_samples: Vec<Vec<f64>>,
    start: usize,
    steps: usize,
    chosen: Option<usize>,
    report: Option<TuneReport>,
    /// `take_report` handed the report out.
    reported: bool,
}

impl Tuner {
    /// `sizes` ascending; decoding starts on `sizes[start]`.
    pub(crate) fn new(sizes: Vec<usize>, start: usize) -> Self {
        assert!(!sizes.is_empty() && start < sizes.len());
        let samples = vec![Vec::new(); sizes.len()];
        Self {
            verify_samples: vec![Vec::new(); sizes.len()],
            report: None,
            reported: false,
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
            self.report = Some(TuneReport {
                candidates: self.sizes.clone(),
                median_ms: self.samples.iter().map(|s| median(s)).collect(),
                verify_median_ms: self
                    .verify_samples
                    .iter()
                    .map(|s| (!s.is_empty()).then(|| median(s)))
                    .collect(),
                chosen: self.sizes[index],
                steps: self.steps,
            });
        }
    }

    /// Record the duration of a draft-verification step (several rows) that
    /// ran on `sizes[index]` while tuning. Reporting only.
    pub(crate) fn record_verify(&mut self, index: usize, ms: f64) {
        if self.chosen.is_none() && self.steps >= WARMUP {
            self.verify_samples[index].push(ms);
        }
    }

    /// The report, once the tuner has chosen.
    pub(crate) fn report(&self) -> Option<&TuneReport> {
        self.report.as_ref()
    }

    /// The report the first time it is asked for after the choice, so one
    /// page carries it.
    pub(crate) fn take_report(&mut self) -> Option<TuneReport> {
        if self.reported {
            return None;
        }
        let report = self.report.clone()?;
        self.reported = true;
        Some(report)
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

pub(crate) fn median(values: &[f64]) -> f64 {
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
        // Hybrid 6P+8E: the performance cores are a candidate of their own.
        let hybrid = crate::auto::HostInfo {
            performance_cores: Some(6),
            ..host(14, 20)
        };
        assert_eq!(auto_candidates(&hybrid, 20, false), (vec![6, 7, 10, 14, 17], 3));
        assert_eq!(auto_candidates(&hybrid, 20, true), (vec![6, 10, 14, 17], 2));
    }

    #[test]
    fn report_is_built_once_from_single_steps_and_handed_out_once() {
        let mut tuner = Tuner::new(vec![8, 16], 1);
        assert!(tuner.report().is_none() && tuner.take_report().is_none());
        let mut step = 0;
        while tuner.chosen().is_none() {
            let index = tuner.current();
            // 8 threads: 10 ms; 16 threads: 9 ms (within 2%: the smaller wins... no: 10 > 9.18).
            tuner.record(index, if index == 0 { 10.0 } else { 9.0 });
            // Verification steps are slower and never enter the choice.
            tuner.record_verify(index, 100.0);
            step += 1;
            assert!(step < 1000);
        }
        let report = tuner.report().unwrap().clone();
        assert_eq!(report.chosen, 16);
        assert_eq!(report.candidates, vec![8, 16]);
        assert_eq!(report.median_ms, vec![10.0, 9.0]);
        assert_eq!(report.verify_median_ms, vec![Some(100.0), Some(100.0)]);
        assert_eq!(report.steps, step);
        assert_eq!(tuner.take_report(), Some(report.clone()));
        assert_eq!(tuner.take_report(), None);
        assert!(
            report
                .to_string()
                .starts_with("decode threads: 16 (auto; median ms/step 8:10.00 16:9.00)")
        );
        assert!(report.to_string().ends_with("; verify steps 8:100.00 16:100.00"));
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
