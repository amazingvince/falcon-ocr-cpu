//! Copied integration test only. Uses public Runner APIs; no production hooks.
use anyhow::{Context, Result, ensure};
use falcon_ocr::{
    Backend, CacheLayout, FinishReason, GenerationOptions, Model, OcrResult, Runner, RunnerConfig,
    WeightLayout, trace::Trace,
};
use image::{Rgb, RgbImage, imageops};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

// Same System delegation and decode interval as tests/decode_allocations.rs.
// Only this one ignored test is run in each fresh integration-test process.
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
// SAFETY: Delegate unchanged pointers/layouts to System; atomic counters do not allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Default, Serialize)]
struct HashTrace {
    tensors: BTreeMap<String, TensorIdentity>,
    logits_argmax: Vec<(String, Vec<u32>)>,
    active_request_indices: Vec<(String, Vec<usize>)>,
    decode_rows: Vec<usize>,
    duplicate_head_tensors_checked: usize,
}
#[derive(Serialize)]
struct TensorIdentity {
    dtype: &'static str,
    shape: Vec<usize>,
    elements: usize,
    sha256: String,
}
impl Trace for HashTrace {
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        ensure!(
            cfg!(target_endian = "little"),
            "trace byte identity requires little endian"
        );
        ensure!(
            shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)) == Some(data.len()),
            "invalid tensor shape {name}"
        );
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "nonfinite tensor {name}"
        );
        if name.ends_with(".k") || name.ends_with(".v") {
            ensure!(
                shape.len() == 3 && shape[1..] == [16, 64],
                "unexpected K/V shape {name}"
            );
            let channels = if name.ends_with(".v") || name.contains("decode.") {
                64
            } else {
                32
            };
            for heads in data.chunks_exact(128) {
                ensure!(
                    heads[..channels]
                        .iter()
                        .zip(&heads[64..64 + channels])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "discarded duplicate bits differ: {name}"
                );
            }
            self.duplicate_head_tensors_checked += 1;
        }
        if name.ends_with(".logits") {
            ensure!(
                !shape.is_empty() && *shape.last().unwrap() > 0,
                "empty logits"
            );
            let width = *shape.last().unwrap();
            let chosen = data
                .chunks_exact(width)
                .map(|row| {
                    let mut best = 0;
                    for i in 1..row.len() {
                        if row[i] > row[best] {
                            best = i;
                        }
                    }
                    u32::try_from(best).unwrap()
                })
                .collect();
            self.logits_argmax.push((name.to_owned(), chosen));
        }
        if name.starts_with("batch.") && name.ends_with(".embedding") {
            self.decode_rows.push(shape[0]);
        }
        if name.ends_with(".request_indices") {
            ensure!(
                data.iter().all(|v| *v >= 0. && v.fract() == 0.),
                "invalid active request index"
            );
            self.active_request_indices
                .push((name.to_owned(), data.iter().map(|v| *v as usize).collect()));
        }
        let item = TensorIdentity {
            dtype: "F32-le",
            shape: shape.to_vec(),
            elements: data.len(),
            sha256: digest(bytemuck::cast_slice(data)),
        };
        ensure!(
            self.tensors.insert(name.to_owned(), item).is_none(),
            "duplicate tensor name {name}"
        );
        Ok(())
    }
}

#[derive(Default, Serialize)]
struct AllocationProbe {
    starts: usize,
    ends: usize,
}
impl Trace for AllocationProbe {
    fn enabled(&self) -> bool {
        false
    }
    fn tensor(&mut self, _: &str, _: &[usize], _: &[f32]) -> Result<()> {
        panic!("disabled trace received tensor")
    }
    fn decode_start(&mut self) {
        assert!(!ACTIVE.load(Ordering::SeqCst));
        self.starts += 1;
        // Do not reset at each callback: a future extra chunk cannot erase allocations.
        ACTIVE.store(true, Ordering::SeqCst);
    }
    fn decode_end(&mut self) {
        ACTIVE.store(false, Ordering::SeqCst);
        self.ends += 1;
    }
}
impl Drop for AllocationProbe {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::SeqCst);
    }
}
fn begin_probe() -> AllocationProbe {
    assert!(!ACTIVE.load(Ordering::SeqCst));
    CALLS.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    AllocationProbe::default()
}
fn finish_probe(probe: &AllocationProbe) -> Result<serde_json::Value> {
    ACTIVE.store(false, Ordering::SeqCst);
    ensure!(
        (probe.starts, probe.ends) == (1, 1),
        "expected exactly one full decode interval"
    );
    let calls = CALLS.load(Ordering::SeqCst);
    let bytes = BYTES.load(Ordering::SeqCst);
    ensure!(
        calls == 0 && bytes == 0,
        "decode allocated {calls} calls/{bytes} bytes"
    );
    Ok(
        serde_json::json!({"decode_starts":probe.starts,"decode_ends":probe.ends,"allocation_calls":calls,"requested_bytes":bytes}),
    )
}

fn result_identity(
    result: &OcrResult,
    teacher: bool,
    layout: CacheLayout,
) -> Result<serde_json::Value> {
    ensure!(
        result.teacher_forced == teacher,
        "unexpected generation path"
    );
    ensure!(
        result.precision == "fp32" && result.backend == "rust-gemm/avx2",
        "unexpected precision/backend"
    );
    ensure!(
        result.cache_layout == layout
            && result.weight_layout == WeightLayout::Unpacked
            && result.packed_weight_bytes == 0,
        "unexpected cache/weight layout"
    );
    ensure!(
        result.output_tokens == result.token_ids.len() && result.output_tokens > 0,
        "invalid output count"
    );
    Ok(
        serde_json::json!({"token_ids":result.token_ids,"text":result.text,"finish_reason":result.finish_reason,
        "width":result.width,"height":result.height,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,
        "teacher_forced":result.teacher_forced,"precision":result.precision,"backend":result.backend,
        "weight_layout":result.weight_layout,"packed_weight_bytes":result.packed_weight_bytes}),
    )
}

#[test]
#[ignore = "explicit isolated full-model qualification; requires reviewed plan and pinned assets"]
fn temporal_full_model_qualification() -> Result<()> {
    let output =
        PathBuf::from(std::env::var_os("FOCR_TEMPORAL_MODEL_OUTPUT").context("set output file")?);
    ensure!(!output.exists(), "preserve existing run output");
    let layout_name = std::env::var("FOCR_TEMPORAL_MODEL_LAYOUT")?;
    ensure!(
        ["expanded", "compact", "temporal_candidate"].contains(&layout_name.as_str()),
        "unknown layout"
    );
    // The same source compiles against the genuine production crate; that crate
    // rejects temporal_candidate. No cfg override or production control patch.
    let layout: CacheLayout = serde_json::from_value(serde_json::json!(layout_name))?;
    let model = Arc::new(Model::load("artifacts/model")?);
    let runner = |batch_size| {
        Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                batch_size,
                backend: Backend::Avx2,
                cache_layout: layout,
                weight_layout: WeightLayout::Unpacked,
            },
        )
    };
    let single = runner(1)?;
    let joint = runner(4)?;
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(
        "artifacts/reference/smoke-fp32/metadata.json",
    )?)?;
    let gpu_ids: Vec<u32> = serde_json::from_value(metadata["token_ids"].clone())?;
    ensure!(gpu_ids.len() == 17, "unexpected smoke reference length");
    let mut canonical = HashTrace::default();
    let traced = single.trace_reference(
        "artifacts/reference/smoke-fp32/trace.safetensors",
        17,
        &mut canonical,
    )?;
    ensure!(
        canonical.tensors.len() == 1904 && canonical.logits_argmax.len() == 17,
        "canonical trace inventory differs"
    );
    ensure!(
        canonical
            .logits_argmax
            .iter()
            .all(|(_, ids)| ids.len() == 1),
        "canonical logits rows"
    );
    let argmax: Vec<u32> = canonical
        .logits_argmax
        .iter()
        .map(|(_, ids)| ids[0])
        .collect();
    ensure!(
        argmax == gpu_ids,
        "actual same-prefix argmax differs from fixed smoke decisions"
    );
    ensure!(
        traced.token_ids == gpu_ids && traced.input_tokens == 144,
        "teacher prefix result differs"
    );
    let trace_result = result_identity(&traced, true, layout)?;
    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")?.to_rgb8();
    ensure!(
        (original.width(), original.height()) == (256, 128),
        "canonical image dimensions"
    );
    let line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    let images = [original, blank, line];
    let inputs:Vec<_>=images.iter().zip(["smoke","blank","one_line"]).map(|(image,key)|serde_json::json!({
        "key":key,"width":image.width(),"height":image.height(),"rgb_sha256":digest(image.as_raw())})).collect();
    let options = GenerationOptions {
        min_dimension: 64,
        max_dimension: 256,
        max_new_tokens: 24,
    };
    let sequential: Vec<_> = images
        .iter()
        .map(|image| single.recognize(image, &options))
        .collect::<Result<_>>()?;
    let expected: Vec<_> = sequential
        .iter()
        .map(|r| result_identity(r, false, layout))
        .collect::<Result<_>>()?;
    ensure!(
        sequential[0].token_ids == gpu_ids,
        "free smoke differs from fixed GPU IDs"
    );
    ensure!(
        sequential
            .iter()
            .all(|r| r.finish_reason == FinishReason::Eos),
        "expected full EOS fixture"
    );
    ensure!(
        sequential
            .iter()
            .map(|r| r.output_tokens)
            .collect::<Vec<_>>()
            == [17, 2, 6],
        "fixed mixed fixture must exercise uneven completion"
    );
    let mut mixed = HashTrace::default();
    let mixed_results = joint.recognize_batch_with_trace(&images, &options, &mut mixed)?;
    let actual: Vec<_> = mixed_results
        .iter()
        .map(|r| result_identity(r, false, layout))
        .collect::<Result<_>>()?;
    ensure!(
        actual == expected,
        "independent single/mixed results differ"
    );
    ensure!(
        mixed.decode_rows.contains(&3)
            && mixed.decode_rows.contains(&2)
            && mixed.decode_rows.contains(&1),
        "missing active-row compaction"
    );
    ensure!(
        mixed.decode_rows.len() == 16 && mixed.active_request_indices.len() == 16,
        "mixed decode schedule incomplete"
    );
    // Warm the exact Runner/config before the disabled-trace allocation interval.
    let warm_single = single.recognize(&images[0], &options)?;
    ensure!(
        result_identity(&warm_single, false, layout)? == expected[0],
        "warm single output differs"
    );
    let probe = begin_probe();
    let mut probe = probe;
    let measured_single = single.recognize_with_trace(&images[0], &options, &mut probe);
    ACTIVE.store(false, Ordering::SeqCst);
    let allocation_single = finish_probe(&probe)?;
    ensure!(
        result_identity(&measured_single?, false, layout)? == expected[0],
        "measured single output differs"
    );
    drop(probe);
    let warm_mixed = joint.recognize_batch(&images, &options)?;
    ensure!(
        warm_mixed
            .iter()
            .map(|r| result_identity(r, false, layout))
            .collect::<Result<Vec<_>>>()?
            == expected,
        "warm mixed differs"
    );
    let mut probe = begin_probe();
    let measured_mixed = joint.recognize_batch_with_trace(&images, &options, &mut probe);
    ACTIVE.store(false, Ordering::SeqCst);
    let allocation_mixed = finish_probe(&probe)?;
    ensure!(
        measured_mixed?
            .iter()
            .map(|r| result_identity(r, false, layout))
            .collect::<Result<Vec<_>>>()?
            == expected,
        "measured mixed differs"
    );
    let report = serde_json::json!({"schema_version":1,"status":"completed","cache_layout":layout_name,
        "model_revision":falcon_ocr::config::MODEL_REVISION,"weights_sha256":model.weights_sha256(),
        "runtime":{"precision":"fp32","backend":"avx2","threads":4,"weight_layout":"unpacked","min_dimension":64,"max_dimension":256,"max_new_tokens":24,"mixed_batch_size":4},
        "inputs":inputs,"canonical":{"path_kind":"teacher_forced_same_prefix","result":trace_result,"trace":canonical},
        "independent_free_single":expected,"independent_free_mixed":actual,"mixed_trace":mixed,
        "allocation":{"single":allocation_single,"mixed":allocation_mixed,
            "scope":"Warmed full Runner decode_start/decode_end interval, Trace.enabled=false, one chunk. Includes all worker-thread allocations; excludes load, preprocessing, prefill, final detokenization and result assembly. Calls/requested bytes are not capacity, peak RSS or timing."},
        "performance_measurement":false,"limits":"CPU layout equivalence on fixed smoke/three-input mixed fixtures only; does not resolve GPU hidden-stage numerical gates, corpus quality or performance."});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&report)?)?;
    Ok(())
}
