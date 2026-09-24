//! Explicit diagnostic: output traces belong on a volume with sufficient space.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use falcon_ocr::{
    Backend, CacheLayout, GenerationOptions, Model, Runner, RunnerConfig,
    trace::{TensorTrace, Trace},
};
use image::{Rgb, RgbImage, imageops};
use memmap2::Mmap;
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};

struct HeapAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: HeapAllocator = HeapAllocator;

fn allocated(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
// SAFETY: Each allocation operation delegates to System with unchanged inputs.
// The counters track successful Rust allocations and never access their contents.
unsafe impl GlobalAlloc for HeapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, size) };
        if !new_pointer.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            allocated(size);
        }
        new_pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[derive(Default)]
struct Capture {
    tensors: TensorTrace,
    decode_start_heap_bytes: usize,
}
impl Trace for Capture {
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
        self.tensors.tensor(name, shape, data)
    }
    fn decode_start(&mut self) {
        self.decode_start_heap_bytes = LIVE.load(Ordering::SeqCst);
    }
}

#[test]
#[ignore = "requires pinned model and FOCR_WORKSPACE_TRACE output path; writes a large trace"]
fn mixed_batch_workspace_trace() {
    let output = PathBuf::from(
        std::env::var_os("FOCR_WORKSPACE_TRACE").expect("set FOCR_WORKSPACE_TRACE to an output .safetensors path"),
    );
    std::fs::create_dir_all(output.parent().unwrap()).unwrap();
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let cache_layout = match std::env::var("FOCR_WORKSPACE_CACHE_LAYOUT").as_deref() {
        Ok("compact") => CacheLayout::Compact,
        Ok("expanded") | Err(_) => CacheLayout::Expanded,
        Ok(other) => panic!("unsupported FOCR_WORKSPACE_CACHE_LAYOUT {other}"),
    };
    let runner = Runner::new(
        model,
        "artifacts/model",
        RunnerConfig {
            threads: 4,
            batch_size: 4,
            backend: Backend::Avx2,
            cache_layout,
            ..Default::default()
        },
    )
    .unwrap();
    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let one_line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    let images = [original.clone(), blank, one_line, original];
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    runner.recognize_batch(&images, &options).unwrap();
    let mut capture = Capture::default();
    let start_heap = LIVE.load(Ordering::SeqCst);
    PEAK.store(start_heap, Ordering::SeqCst);
    let results = runner
        .recognize_batch_with_trace(&images, &options, &mut capture)
        .unwrap();
    let peak_heap = PEAK.load(Ordering::SeqCst);
    capture.tensors.save(&output).unwrap();
    let file = std::fs::File::open(&output).unwrap();
    // SAFETY: This diagnostic owns the completed output and does not modify it.
    let map = unsafe { Mmap::map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&map).unwrap();
    let report = serde_json::json!({
        "description": "Rust live heap only, including captured tensors; excludes mmap model weights and native allocations",
        "cache_layout": cache_layout,
        "trace_sha256": format!("{:x}", Sha256::digest(&map)),
        "tensor_count": tensors.len(),
        "start_heap_bytes": start_heap,
        "decode_start_heap_bytes": capture.decode_start_heap_bytes,
        "peak_heap_bytes": peak_heap,
        "token_ids": results.iter().map(|r| &r.token_ids).collect::<Vec<_>>(),
        "finish_reasons": results.iter().map(|r| &r.finish_reason).collect::<Vec<_>>(),
        "input_tokens": results.iter().map(|r| r.input_tokens).collect::<Vec<_>>(),
    });
    std::fs::write(
        output.with_extension("json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Some(baseline) = std::env::var_os("FOCR_WORKSPACE_BASELINE") {
        let baseline_file = std::fs::File::open(baseline).unwrap();
        // SAFETY: The baseline is a previously completed immutable diagnostic.
        let baseline_map = unsafe { Mmap::map(&baseline_file).unwrap() };
        let reference = SafeTensors::deserialize(&baseline_map).unwrap();
        assert_eq!(tensors.len(), reference.len());
        for (name, actual) in tensors.tensors() {
            let expected = reference.tensor(&name).unwrap();
            assert_eq!(actual.dtype(), expected.dtype(), "{name} dtype");
            assert_eq!(actual.shape(), expected.shape(), "{name} shape");
            assert!(actual.data() == expected.data(), "{name} changed tensor bits");
        }
        println!("All {} tensors match baseline bit for bit", tensors.len());
    }
}
