//! Warm end-to-end recognition from canonical RGB buffers. Image decoding and
//! model loading are measured separately, and never hidden in warm OCR samples.
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use falcon_ocr::{Backend, CacheLayout, GenerationOptions, Model, Runner, RunnerConfig, WeightLayout};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Execution {
    Sequential,
    Joint,
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "artifacts/model")]
    model: PathBuf,
    #[arg(required = true)]
    images: Vec<PathBuf>,
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[arg(long, value_enum, default_value = "auto")]
    backend: Backend,
    #[arg(long, value_enum, default_value = "compact")]
    cache_layout: CacheLayout,
    #[arg(long, value_enum, default_value = "unpacked")]
    weight_layout: WeightLayout,
    /// Compare independent requests with joint decoding in the same binary.
    #[arg(long, value_enum, default_value = "joint")]
    execution: Execution,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
    batches: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 7)]
    repetitions: usize,
    #[arg(long, default_value_t = 256)]
    max_dimension: u32,
    #[arg(long, default_value_t = 64)]
    min_dimension: u32,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    /// Identify the physical CPU; recorded as a user-supplied label.
    #[arg(long)]
    cpu_label: String,
    /// Distinguish native Linux from WSL, VMs or other environments.
    #[arg(long)]
    environment_label: String,
    #[arg(long)]
    output: PathBuf,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.
    } else {
        sorted[sorted.len() / 2]
    }
}

fn memory() -> serde_json::Value {
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
        json!({"resident_bytes":bytes("VmRSS:"),"peak_resident_bytes":bytes("VmHWM:")})
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
        // SAFETY: ABI matches PROCESS_MEMORY_COUNTERS; this process handle is a
        // valid pseudo-handle and the writable struct has the advertised size.
        let ok = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<Counters>() as u32,
            )
        };
        if ok == 0 {
            json!({"unavailable":true})
        } else {
            json!({"resident_bytes":counters.sizes[1],"peak_resident_bytes":counters.sizes[0],
            "private_commit_bytes":counters.sizes[6],"peak_private_commit_bytes":counters.sizes[7]})
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        json!({"unavailable":true})
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repetitions >= 3, "at least three samples required");
    ensure!(
        args.batches.iter().all(|&b| b > 0 && b <= 8),
        "batch sizes must be 1..=8"
    );
    let options = GenerationOptions {
        min_dimension: args.min_dimension,
        max_dimension: args.max_dimension,
        max_new_tokens: args.max_new_tokens,
    };
    options.validate()?;
    let start = Instant::now();
    let mut image_manifest = Vec::new();
    let mut images = Vec::new();
    for path in &args.images {
        let bytes = std::fs::read(path)?;
        ensure!(
            image::guess_format(&bytes)? == image::ImageFormat::Png,
            "benchmark input must be canonical RGB PNG"
        );
        let decoded = image::load_from_memory(&bytes)?;
        let rgb = decoded
            .as_rgb8()
            .context("benchmark input must have RGB8 source mode")?
            .clone();
        image_manifest.push(
            json!({"path":path,"sha256":hash(&bytes),"rgb_sha256":hash(rgb.as_raw()),
            "width":rgb.width(),"height":rgb.height()}),
        );
        images.push(rgb);
    }
    let read_decode_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    let model = Arc::new(Model::load(&args.model)?);
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let loaded_memory = memory();
    let mut cases = Vec::new();
    for &batch in &args.batches {
        let active_batch = match args.execution {
            Execution::Sequential => 1,
            Execution::Joint => batch,
        };
        let runner = Runner::new(
            model.clone(),
            &args.model,
            RunnerConfig {
                threads: args.threads,
                batch_size: active_batch,
                backend: args.backend,
                cache_layout: args.cache_layout,
                weight_layout: args.weight_layout,
                ..RunnerConfig::reference()
            },
        )?;
        let pages = (0..batch).map(|i| images[i % images.len()].clone()).collect::<Vec<_>>();
        for _ in 0..args.warmup {
            runner.recognize_batch(&pages, &options)?;
        }
        let mut samples = Vec::new();
        let mut times = Vec::new();
        let mut expected = None;
        let mut expected_outputs = None;
        for _ in 0..args.repetitions {
            let start = Instant::now();
            let results = runner.recognize_batch(&pages, &options)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let ids = results.iter().map(|r| r.token_ids.clone()).collect::<Vec<_>>();
            // Outside the timed interval: preserve the actual decoder output
            // and require the same stop semantics in every measured repetition.
            let outputs = results
                .iter()
                .map(|r| {
                    ensure!(
                        r.output_tokens == r.token_ids.len(),
                        "output count differs from token IDs"
                    );
                    ensure!(!r.token_ids.is_empty(), "empty generated token sequence");
                    ensure!(
                        !r.token_ids[..r.token_ids.len() - 1]
                            .iter()
                            .any(|id| [11, 263].contains(id)),
                        "tokens after EOS"
                    );
                    let ended = [11, 263].contains(r.token_ids.last().unwrap());
                    ensure!(
                        ended == matches!(r.finish_reason, falcon_ocr::runner::FinishReason::Eos),
                        "EOS/stop mismatch"
                    );
                    ensure!(
                        ended || r.output_tokens == options.max_new_tokens,
                        "length stop before output cap"
                    );
                    Ok(json!({"text":r.text,"finish_reason":r.finish_reason}))
                })
                .collect::<Result<Vec<_>>>()?;
            if let Some(previous) = &expected {
                ensure!(previous == &ids, "nondeterministic output during benchmark");
            } else {
                expected = Some(ids);
            }
            if let Some(previous) = &expected_outputs {
                ensure!(previous == &outputs, "nondeterministic text or stop during benchmark");
            } else {
                expected_outputs = Some(outputs);
            }
            times.push(elapsed);
            samples.push(json!({"wall_ms":elapsed,"pages_per_second":batch as f64*1000./elapsed,
                "emitted_tokens":results.iter().map(|r|r.output_tokens).sum::<usize>(),
                "per_request":results.iter().map(|r|json!({"timings":r.timings,"input_tokens":r.input_tokens,
                    "output_tokens":r.output_tokens,"finish_reason":r.finish_reason,"text":r.text,
                    "teacher_forced":r.teacher_forced,"precision":r.precision,
                    "width":r.width,"height":r.height})).collect::<Vec<_>>()}));
        }
        let median_ms = median(&times);
        cases.push(json!({"batch_size":batch,"active_batch_size":active_batch,"image_indices":(0..batch).map(|i|i%images.len()).collect::<Vec<_>>(),
            "execution":match args.execution {Execution::Sequential=>"independent_sequential",Execution::Joint=>"independent_prefill_joint_decode"},"median_ms":median_ms,
            "median_pages_per_second":batch as f64*1000./median_ms,"samples":samples,"token_ids":expected,"memory_after":memory()}));
    }
    let report = json!({"schema_version":2,"time_unix":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "cpu_label":args.cpu_label,"environment_label":args.environment_label,"os":std::env::consts::OS,
        "arch":std::env::consts::ARCH,"logical_cpus":std::thread::available_parallelism()?.get(),
        "threads":args.threads,"backend":args.backend,"cache_layout":args.cache_layout,"precision":"fp32","options":options,
        "weight_layout":args.weight_layout,"packed_weight_bytes":model.packed_weight_bytes(),"weight_packing_ms":model.weight_packing_ms(),
        "warmup":args.warmup,"repetitions":args.repetitions,"read_decode_ms":read_decode_ms,"verified_model_load_ms":load_ms,
        "model_revision":falcon_ocr::config::MODEL_REVISION,"weights_sha256":model.weights_sha256(),
        "images":image_manifest,"loaded_memory":loaded_memory,"cases":cases,
        "cargo_lock_sha256":hash(include_bytes!("../Cargo.lock")),
        "source_sha256":{"model":hash(include_bytes!("../src/model/mod.rs")),"kernels":hash(include_bytes!("../src/kernels/mod.rs")),
            "runner":hash(include_bytes!("../src/runner.rs")),"preprocess":hash(include_bytes!("../src/preprocess.rs")),
            "packed_kernels":hash(include_bytes!("../src/packed_kernels.rs")),
            "config":hash(include_bytes!("../src/config.rs")),"tokenizer":hash(include_bytes!("../src/tokenizer.rs")),
            "trace":hash(include_bytes!("../src/trace.rs")),"lib":hash(include_bytes!("../src/lib.rs")),
            "harness":hash(include_bytes!("ocr_bench.rs"))},
        "binary_sha256":hash(&std::fs::read(std::env::current_exe()?)?),
        "notes":["Warm RGB-buffer recognition includes resizing and tokenization but excludes file read/decode and model loading.",
            "Peak memory counters are process-lifetime high-water marks, not isolated per-case peaks.",
            "Repeated input pages are explicitly indexed; this is not a corpus quality evaluation.",
            "Sequential and joint modes use the same model arithmetic and preserve request order; compare matched runs on an otherwise idle host."]});
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", args.output.display());
    Ok(())
}
