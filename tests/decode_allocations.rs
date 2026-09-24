//! One test per binary keeps the process-wide counter isolated from other tests.
use falcon_ocr::{CacheLayout, GenerationOptions, HeadMode, Model, Runner, RunnerConfig, WeightLayout, trace::Trace};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

struct CountingAllocator;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count(bytes: usize) {
    if ACTIVE.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}
// SAFETY: All operations delegate to System with exactly the supplied layout
// and pointers; the extra counters do not allocate or access allocation memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(p, layout, size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
}
struct Probe;
impl Trace for Probe {
    fn enabled(&self) -> bool {
        false
    }
    fn tensor(&mut self, _: &str, _: &[usize], _: &[f32]) -> anyhow::Result<()> {
        unreachable!()
    }
    fn decode_start(&mut self) {
        CALLS.store(0, Ordering::SeqCst);
        BYTES.store(0, Ordering::SeqCst);
        ACTIVE.store(true, Ordering::SeqCst);
    }
    fn decode_end(&mut self) {
        ACTIVE.store(false, Ordering::SeqCst);
    }
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn warm_fp32_decode_has_no_heap_allocations() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let image = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    for (cache_layout, weight_layout, head) in [
        (CacheLayout::Expanded, WeightLayout::Unpacked, HeadMode::Full),
        (CacheLayout::Expanded, WeightLayout::PhasePacked, HeadMode::Full),
        (CacheLayout::Compact, WeightLayout::Unpacked, HeadMode::Full),
        (CacheLayout::Compact, WeightLayout::PhasePacked, HeadMode::Full),
        (CacheLayout::Compact, WeightLayout::Unpacked, HeadMode::Screened),
        (CacheLayout::Compact, WeightLayout::PhasePacked, HeadMode::Screened),
    ] {
        {
            let mut runner = Runner::new(
                model.clone(),
                "artifacts/model",
                RunnerConfig {
                    threads: 4,
                    cache_layout,
                    weight_layout,
                    ..Default::default()
                },
            )
            .unwrap();
            runner.set_head_mode(head).unwrap();
            runner.recognize(&image, &options).unwrap();
            let result = runner.recognize_with_trace(&image, &options, &mut Probe);
            ACTIVE.store(false, Ordering::SeqCst);
            let result = result.unwrap();
            assert_eq!(result.output_tokens, 17, "test must exercise all smoke decode steps");
            assert_eq!(
                CALLS.load(Ordering::SeqCst),
                0,
                "{cache_layout:?}: {} bytes allocated during decode",
                BYTES.load(Ordering::SeqCst)
            );
            let mut batch_runner = Runner::new(
                model.clone(),
                "artifacts/model",
                RunnerConfig {
                    threads: 4,
                    batch_size: 4,
                    cache_layout,
                    weight_layout,
                    ..Default::default()
                },
            )
            .unwrap();
            batch_runner.set_head_mode(head).unwrap();
            let pages = [image.clone(), image.clone(), image.clone(), image.clone()];
            batch_runner.recognize_batch(&pages, &options).unwrap();
            let result = batch_runner.recognize_batch_with_trace(&pages, &options, &mut Probe);
            ACTIVE.store(false, Ordering::SeqCst);
            assert_eq!(result.unwrap().len(), 4);
            assert_eq!(
                CALLS.load(Ordering::SeqCst),
                0,
                "{cache_layout:?}: {} bytes allocated during batch decode",
                BYTES.load(Ordering::SeqCst)
            );
        }
    }
}
