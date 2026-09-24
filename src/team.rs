//! Persistent spin-waiting worker team for decode steps.
//!
//! A decode step runs about a hundred short parallel loops (projections,
//! attention, the vocabulary screen), each a few tens of microseconds long.
//! Between them, Rayon's idle workers go to sleep and must be woken by the OS
//! for the next loop, which costs more than many of the loops themselves. A
//! `Team` keeps `size - 1` workers spinning on an epoch counter while a decode
//! step is in progress. The calling thread publishes a loop, participates in
//! it, and waits for the workers; tasks are claimed from a shared counter.
//! Workers park after `IDLE_SPIN` without work, so an idle runner costs no CPU.
//!
//! Kernels call [`for_each`], which uses the team entered on the current
//! thread (see [`Team::enter`]) and otherwise falls back to Rayon, so prefill
//! and every other caller keep their existing behaviour. Scheduling never
//! changes arithmetic: each task computes the same values either way.
use rayon::prelude::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::{JoinHandle, Thread};
use std::time::{Duration, Instant};

/// Spin this long without work before parking (a decode step's serial gaps
/// are microseconds; the gap between requests is much longer).
const IDLE_SPIN: Duration = Duration::from_millis(2);

/// One published parallel loop; lives on the caller's stack until every
/// worker has finished with it.
struct Job {
    task: *const (dyn Fn(usize) + Sync),
    tasks: usize,
}

struct Shared {
    epoch: AtomicU64,
    job: AtomicPtr<Job>,
    next: AtomicUsize,
    pending: AtomicUsize,
    stop: AtomicBool,
    sleeping: Vec<AtomicBool>,
    threads: OnceLock<Vec<Thread>>,
}

pub(crate) struct Team {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    size: usize,
    /// One caller at a time; a concurrent caller falls back to Rayon.
    busy: Mutex<()>,
}

thread_local! {
    static ACTIVE: Cell<*const Team> = const { Cell::new(std::ptr::null()) };
    /// Set while this thread executes a task; nested loops then run serially.
    static IN_TASK: Cell<bool> = const { Cell::new(false) };
}

/// Makes the team the current thread's executor until dropped.
pub(crate) struct Entered<'a> {
    previous: *const Team,
    _busy: MutexGuard<'a, ()>,
}
impl Drop for Entered<'_> {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(self.previous));
    }
}

impl Team {
    /// A team of `size` participants: the caller plus `size - 1` workers.
    pub(crate) fn new(size: usize) -> std::io::Result<Self> {
        let size = size.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            sleeping: (1..size).map(|_| AtomicBool::new(false)).collect(),
            threads: OnceLock::new(),
        });
        let mut workers = Vec::with_capacity(size - 1);
        for index in 0..size - 1 {
            let shared = shared.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("falcon-decode-{index}"))
                    .spawn(move || worker(&shared, index))?,
            );
        }
        let _ = shared.threads.set(workers.iter().map(|w| w.thread().clone()).collect());
        Ok(Self {
            shared,
            workers,
            size,
            busy: Mutex::new(()),
        })
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Route [`for_each`] on this thread to the team until the guard drops.
    /// Returns `None` (keep using Rayon) if another thread holds the team.
    pub(crate) fn enter(&self) -> Option<Entered<'_>> {
        let busy = self.busy.try_lock().ok()?;
        let previous = ACTIVE.with(|active| active.replace(self as *const Team));
        Some(Entered { previous, _busy: busy })
    }

    fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() || tasks <= 1 {
            run_tasks(tasks, task, None);
            return;
        }
        let shared = &*self.shared;
        // SAFETY: only the lifetime is erased; `run` does not return until
        // every worker has finished with `job` (pending == 0).
        let task_static: &'static (dyn Fn(usize) + Sync) =
            unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(task) };
        let job = Job {
            task: task_static as *const (dyn Fn(usize) + Sync),
            tasks,
        };
        shared.job.store(&job as *const Job as *mut Job, Ordering::Relaxed);
        shared.next.store(0, Ordering::Relaxed);
        shared.pending.store(self.workers.len(), Ordering::Relaxed);
        // SeqCst pairs with the workers' sleeping flag protocol below.
        shared.epoch.fetch_add(1, Ordering::SeqCst);
        if let Some(threads) = shared.threads.get() {
            for (flag, thread) in shared.sleeping.iter().zip(threads) {
                if flag.load(Ordering::SeqCst) {
                    thread.unpark();
                }
            }
        }
        run_tasks(tasks, task, Some(&shared.next));
        while shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(threads) = self.shared.threads.get() {
            for thread in threads {
                thread.unpark();
            }
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Execute tasks claimed from `next` (or all of them in order without one).
fn run_tasks(tasks: usize, task: &(dyn Fn(usize) + Sync), next: Option<&AtomicUsize>) {
    let nested = IN_TASK.with(|flag| flag.replace(true));
    match next {
        Some(next) => loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= tasks {
                break;
            }
            task(index);
        },
        None => (0..tasks).for_each(task),
    }
    IN_TASK.with(|flag| flag.set(nested));
}

fn worker(shared: &Shared, index: usize) {
    let mut seen = 0_u64;
    loop {
        let mut idle_since = Instant::now();
        let mut spins = 0_u32;
        loop {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen {
                seen = epoch;
                break;
            }
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) && idle_since.elapsed() > IDLE_SPIN {
                shared.sleeping[index].store(true, Ordering::SeqCst);
                if shared.epoch.load(Ordering::SeqCst) == seen && !shared.stop.load(Ordering::SeqCst) {
                    // Timeout is only a safety net; the publisher unparks.
                    std::thread::park_timeout(Duration::from_millis(100));
                }
                shared.sleeping[index].store(false, Ordering::SeqCst);
                idle_since = Instant::now();
            }
        }
        // SAFETY: the publisher keeps `job` and its closure alive until
        // `pending` reaches zero, which requires this worker's decrement.
        let job = unsafe { &*shared.job.load(Ordering::Acquire) };
        let task = unsafe { &*job.task };
        run_tasks(job.tasks, task, Some(&shared.next));
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

/// Run `task(0..tasks)` in parallel on the entered team, or on Rayon.
/// Inside a task (nested), runs serially on the current thread.
pub(crate) fn for_each(tasks: usize, task: impl Fn(usize) + Sync + Send) {
    if IN_TASK.with(Cell::get) {
        (0..tasks).for_each(task);
        return;
    }
    let team = ACTIVE.with(Cell::get);
    if team.is_null() {
        (0..tasks).into_par_iter().for_each(task);
    } else {
        // SAFETY: `Entered` keeps the team alive and active on this thread.
        unsafe { &*team }.run(tasks, &task);
    }
}

/// Participants available to [`for_each`] on this thread.
pub(crate) fn threads() -> usize {
    let team = ACTIVE.with(Cell::get);
    if team.is_null() {
        rayon::current_num_threads().max(1)
    } else {
        // SAFETY: as in `for_each`.
        unsafe { &*team }.size()
    }
}

/// Mutable base pointer for tasks that write disjoint parts of one buffer.
#[derive(Clone, Copy)]
pub(crate) struct SharedMut<T>(*mut T);
// SAFETY: users only write disjoint elements from different tasks.
unsafe impl<T: Send> Send for SharedMut<T> {}
unsafe impl<T: Send> Sync for SharedMut<T> {}
impl<T> SharedMut<T> {
    pub(crate) fn new(slice: &mut [T]) -> Self {
        Self(slice.as_mut_ptr())
    }
    /// The base pointer, for writers of interleaved (non-contiguous) parts.
    pub(crate) fn ptr(&self) -> *mut T {
        self.0
    }
    /// # Safety
    /// `start + len` must lie inside the original slice and no other task may
    /// access the range concurrently.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// Split `items` into about two blocks per participant, each a multiple of
/// `granule` and at least `min_block`.
pub(crate) fn block_size(items: usize, granule: usize, min_block: usize) -> usize {
    let target = 2 * threads();
    items.div_ceil(target).next_multiple_of(granule).max(min_block).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn team_runs_every_task_once_and_nests_serially() {
        let team = Team::new(4).unwrap();
        let _entered = team.enter().unwrap();
        for tasks in [0, 1, 2, 7, 64, 1000] {
            let counts: Vec<AtomicU32> = (0..tasks).map(|_| AtomicU32::new(0)).collect();
            for_each(tasks, |i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
                // Nested loops run inline on this thread.
                let inner = AtomicU32::new(0);
                for_each(3, |_| {
                    inner.fetch_add(1, Ordering::Relaxed);
                });
                assert_eq!(inner.load(Ordering::Relaxed), 3);
            });
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }
        assert_eq!(threads(), 4);
        // A second caller cannot enter while the team is busy.
        assert!(team.enter().is_none());
    }

    #[test]
    fn workers_park_and_wake() {
        let team = Team::new(3).unwrap();
        let _entered = team.enter().unwrap();
        std::thread::sleep(IDLE_SPIN * 3);
        let hits = AtomicU32::new(0);
        for_each(10, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 10);
    }
}
