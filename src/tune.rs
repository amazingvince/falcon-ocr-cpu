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

/// The most threads an automatic decode team takes beside a page pipeline's
/// prefill in a `threads`-thread runner: half, so that the other half (at
/// least two threads) can prefill beside it; `None` below three threads,
/// where the default prefill pool would get fewer than two and the team
/// keeps every thread.
pub(crate) fn pipeline_half(threads: usize) -> Option<usize> {
    (threads >= 3).then_some(threads / 2)
}

/// The team held in reserve for a page pipeline beside a `threads`-thread
/// runner whose tuner has the candidates `sizes` ([`Tuner::with_reserve`]):
/// [`pipeline_half`] when every candidate is larger.
pub(crate) fn pipeline_reserve(sizes: &[usize], threads: usize) -> Option<usize> {
    pipeline_half(threads).filter(|&half| sizes.first().is_some_and(|&smallest| smallest > half))
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
    /// The largest team the choice may take ([`Tuner::limit`]).
    max: usize,
    /// A size below every candidate, held back until a limit excludes every
    /// one ([`Tuner::with_reserve`]).
    reserve: Option<usize>,
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
            max: usize::MAX,
            reserve: None,
            sizes,
            samples,
            start,
            steps: 0,
            chosen: None,
        }
    }

    /// Hold `size` back for a [`Tuner::limit`] that excludes every candidate,
    /// when it is at least one and below every candidate
    /// ([`pipeline_reserve`]).
    pub(crate) fn with_reserve(mut self, size: Option<usize>) -> Self {
        self.reserve = size.filter(|size| (1..self.sizes[0]).contains(size));
        self
    }

    /// Every team size the tuner can hand out: the candidates and the
    /// reserve.
    pub(crate) fn team_sizes(&self) -> Vec<usize> {
        self.reserve.into_iter().chain(self.sizes.iter().copied()).collect()
    }

    /// The team size of candidate `index`.
    pub(crate) fn size(&self, index: usize) -> usize {
        self.sizes[index]
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
            self.choose();
        }
    }

    /// Let the choice take at most `max` threads, and return the largest
    /// size it may still take when that excludes a candidate. When every
    /// candidate is larger (with speculation the half-core team is not one),
    /// the reserve ([`Tuner::with_reserve`]) becomes the smallest candidate if
    /// it is within `max`, and else the smallest candidate stays. Tuning still
    /// measures every candidate, and the tolerance still compares with the
    /// fastest of them; a choice above `max` becomes the largest candidate
    /// within it, also when it was already made (its report is then handed
    /// out again), except that a choice made before the reserve came in is
    /// dropped, and tuning goes on until the reserve is measured too.
    pub(crate) fn limit(&mut self, max: usize) -> Option<usize> {
        if max >= self.max {
            return None;
        }
        self.max = max;
        if self.sizes[0] > max
            && let Some(reserve) = self.reserve.filter(|&size| size <= max)
        {
            self.reserve = None;
            self.sizes.insert(0, reserve);
            self.samples.insert(0, Vec::new());
            self.verify_samples.insert(0, Vec::new());
            self.start += 1;
            self.chosen = None;
            self.report = None;
            self.reported = false;
        }
        let allowed = self.allowed();
        if allowed + 1 == self.sizes.len() {
            return None;
        }
        if self.chosen.is_some_and(|index| index > allowed) {
            self.choose();
            self.reported = false;
        }
        Some(self.sizes[allowed])
    }

    /// The index of the largest candidate within the limit, else 0.
    fn allowed(&self) -> usize {
        self.sizes.iter().rposition(|&size| size <= self.max).unwrap_or(0)
    }

    /// Choose from the measured candidates within the limit and build the
    /// report.
    fn choose(&mut self) {
        let index = self.pick().min(self.allowed());
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

    /// Tune `tuner` to its choice with `cost` per team size, and return
    /// the sizes it measured.
    fn tune(tuner: &mut Tuner, cost: impl Fn(usize) -> f64) -> Vec<usize> {
        let mut measured = Vec::new();
        while tuner.chosen().is_none() {
            let size = tuner.sizes()[tuner.current()];
            measured.push(size);
            tuner.record(tuner.current(), cost(size));
        }
        measured.sort_unstable();
        measured.dedup();
        measured
    }

    #[test]
    fn a_limit_caps_the_choice_but_not_the_measurement() {
        // 12 threads are the fastest, 8 and 6 more than 2% slower.
        let cost = |size: usize| match size {
            6 => 9.4,
            8 => 9.3,
            _ => 9.0,
        };
        let mut tuner = Tuner::new(vec![6, 8, 12], 1);
        assert_eq!(tuner.limit(8), Some(8));
        assert_eq!((tuner.limit(8), tuner.limit(16), tuner.limit(12)), (None, None, None));
        assert_eq!(tune(&mut tuner, cost), vec![6, 8, 12]);
        // Not 6 threads, which are within 2% of 8 but not of 12.
        assert_eq!(tuner.chosen(), Some(8));
        let report = tuner.report().unwrap();
        assert_eq!((report.candidates.clone(), report.chosen), (vec![6, 8, 12], 8));
        // A choice within the limit stands.
        let mut tuner = Tuner::new(vec![6, 8, 12], 1);
        tuner.limit(8);
        tune(&mut tuner, |size| if size == 6 { 9.0 } else { cost(size) });
        assert_eq!(tuner.chosen(), Some(6));
        // A limit that excludes nothing changes nothing.
        let mut tuner = Tuner::new(vec![6, 8, 12], 1);
        assert_eq!(tuner.limit(12), None);
        tune(&mut tuner, cost);
        assert_eq!(tuner.chosen(), Some(12));
    }

    #[test]
    fn a_limit_after_the_choice_takes_the_largest_candidate_within_it() {
        let mut tuner = Tuner::new(vec![6, 8, 12], 1);
        tune(&mut tuner, |size| match size {
            6 => 9.4,
            8 => 9.3,
            _ => 9.0,
        });
        assert_eq!(tuner.chosen(), Some(12));
        assert!(tuner.take_report().is_some());
        assert_eq!(tuner.limit(8), Some(8));
        assert_eq!((tuner.chosen(), tuner.current()), (Some(8), 1));
        assert_eq!(tuner.take_report().map(|report| report.chosen), Some(8));
        // Below every candidate, without a reserve, the smallest stays.
        assert_eq!(tuner.limit(4), Some(6));
        assert_eq!(tuner.chosen(), Some(6));
    }

    #[test]
    fn a_limit_below_every_candidate_takes_the_reserve() {
        // With speculation an 8-core host has no 4-thread candidate; the
        // pipeline's half of its 8 threads is the reserve, which a limit to
        // 4 makes a candidate, measured like the others.
        assert_eq!(pipeline_reserve(&[6, 8], 8), Some(4));
        let mut tuner = Tuner::new(vec![6, 8], 1).with_reserve(Some(4));
        assert_eq!(tuner.team_sizes(), [4, 6, 8]);
        assert_eq!(tuner.sizes(), [6, 8], "held back until a limit needs it");
        assert_eq!(tuner.limit(4), Some(4));
        assert_eq!((tuner.sizes(), tuner.current()), (&[4, 6, 8][..], 2));
        assert_eq!(tune(&mut tuner, |size| size as f64), vec![4, 6, 8]);
        assert_eq!(tuner.chosen(), Some(4));
        // A reserve above the limit, or a limit some candidate fits, leaves
        // it out.
        let mut tuner = Tuner::new(vec![6, 8], 1).with_reserve(Some(4));
        assert_eq!(tuner.limit(3), Some(6));
        assert_eq!(tuner.sizes(), [6, 8]);
        assert_eq!(pipeline_reserve(&[4, 6, 8], 8), None);
        let tuner = Tuner::new(vec![4, 6, 8], 2).with_reserve(Some(4));
        assert_eq!(tuner.team_sizes(), [4, 6, 8]);
        // After the choice: the choice is dropped until the reserve is
        // measured too.
        let mut tuner = Tuner::new(vec![6, 8, 12], 1).with_reserve(Some(4));
        tune(&mut tuner, |size| size as f64);
        assert!(tuner.take_report().is_some());
        assert_eq!(tuner.limit(4), Some(4));
        assert_eq!((tuner.chosen(), tuner.report()), (None, None));
        assert_eq!(tune(&mut tuner, |size| size as f64), vec![4, 6, 8, 12]);
        let report = tuner.take_report().unwrap();
        assert!(report.median_ms.iter().all(|ms| ms.is_finite()), "{report:?}");
        assert_eq!((report.candidates, report.chosen), (vec![4, 6, 8, 12], 4));
        // No reserve that leaves the prefill fewer than two threads: on two
        // threads the default pool cannot overlap, and the team keeps both.
        assert_eq!(pipeline_reserve(&[2], 2), None);
        assert_eq!(pipeline_reserve(&[2, 3], 3), Some(1));
        assert_eq!(Tuner::new(vec![1], 0).with_reserve(Some(0)).team_sizes(), [1]);
    }
}
