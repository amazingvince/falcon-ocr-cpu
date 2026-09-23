//! Isolated entry point; `falcon-ocr` and its defaults remain the reference CLI.
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use falcon_ocr::{
    Backend, CacheLayout, GenerationOptions, HeadMode, Model, Runner, RunnerConfig, WeightLayout,
    attempt::{Profile, Telemetry},
    trace::TensorTrace,
};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Falcon-OCR v1.5 integrated CPU experiment (UNQUALIFIED)")]
struct Cli {
    #[arg(long, default_value = "artifacts/model", global = true)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "reference", global = true)]
    profile: Profile,
    #[arg(long, value_enum, default_value = "auto", global = true)]
    backend: Backend,
    #[arg(long, default_value_t = 16, global = true)]
    threads: usize,
    #[arg(long, default_value_t = 1, global = true)]
    batch_size: usize,
    /// Custom W8G64 safetensors overlay, made by attempt3/convert_w8.py.
    #[arg(long, global = true)]
    w8_artifact: Option<PathBuf>,
    /// FP32 greedy head evaluation; `screened` is exact (same tokens as `full`).
    #[arg(long, value_enum, default_value = "full", global = true)]
    head: HeadMode,
    /// Stop a page once it repeats a cycle of at most 128 tokens for at
    /// least max(256, 4 * cycle) tokens (finish_reason "repetition").
    #[arg(long, global = true)]
    stop_repetition: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Doctor,
    Bench {
        #[arg(required = true)]
        images: Vec<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 1536)]
        max_dimension: u32,
        #[arg(long, default_value_t = 64)]
        min_dimension: u32,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 3)]
        samples: usize,
        /// Must not already exist; raw reports are never silently replaced.
        #[arg(long)]
        report: PathBuf,
    },
    Trace {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
    },
}
fn digest(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn write_new(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Cli::parse();
    ensure!(
        (1..=8).contains(&args.batch_size),
        "attempt batch_size must be 1..=8"
    );
    ensure!(
        args.threads > 0,
        "supply an explicit positive thread budget"
    );
    let mut hardware = serde_json::json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "logical_cpus_visible":std::thread::available_parallelism().map(|n|n.get()).unwrap_or(1),
        "processor_identifier":std::env::var("PROCESSOR_IDENTIFIER").ok(),
        "cpu_model_linux":std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s|s.lines().find(|l|l.starts_with("model name")).map(str::to_owned))});
    #[cfg(target_arch = "x86_64")]
    {
        hardware["avx2"] = std::is_x86_feature_detected!("avx2").into();
        hardware["fma"] = std::is_x86_feature_detected!("fma").into();
        hardware["avx512f"] = std::is_x86_feature_detected!("avx512f").into();
        hardware["avx512bf16"] = std::is_x86_feature_detected!("avx512bf16").into();
    }
    if matches!(args.command, Command::Doctor) {
        println!("{}", serde_json::to_string_pretty(&hardware)?);
        return Ok(());
    }
    // Fail invalid output/options before model allocation and any recognition.
    match &args.command {
        Command::Bench {
            samples,
            report,
            max_new_tokens,
            max_dimension,
            min_dimension,
            images,
            ..
        } => {
            ensure!(*samples > 0, "samples must be positive");
            ensure!(!report.exists(), "report already exists");
            GenerationOptions {
                max_new_tokens: *max_new_tokens,
                min_dimension: *min_dimension,
                max_dimension: *max_dimension,
            }
            .validate()?;
            for p in images {
                ensure!(p.is_file(), "missing image {}", p.display());
            }
        }
        Command::Trace {
            fixture, output, ..
        } => {
            ensure!(fixture.is_file(), "missing fixture");
            ensure!(
                !output.exists() && !output.with_extension("json").exists(),
                "trace or sidecar already exists"
            );
        }
        Command::Doctor => unreachable!(),
    }
    let started = Instant::now();
    let model = Arc::new(Model::load_attempt(
        &args.model,
        args.profile,
        args.w8_artifact.as_deref(),
    )?);
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let memory = model.attempt_memory_report();
    let mut runner = Runner::new(
        model.clone(),
        &args.model,
        RunnerConfig {
            threads: args.threads,
            batch_size: args.batch_size,
            backend: args.backend,
            cache_layout: CacheLayout::Compact,
            weight_layout: WeightLayout::Unpacked,
        },
    )?;
    runner.set_head_mode(args.head)?;
    runner.set_repetition_stop(args.stop_repetition);
    eprintln!(
        "profile={} load/import={:.1}ms; experimental quality is NOT qualified",
        args.profile.label(),
        load_ms
    );
    match args.command {
        Command::Bench {
            images,
            max_new_tokens,
            max_dimension,
            min_dimension,
            warmup,
            samples,
            report,
        } => {
            let options = GenerationOptions {
                max_new_tokens,
                max_dimension,
                min_dimension,
            };
            let inputs = images
                .iter()
                .map(|p| Ok(serde_json::json!({"path":p,"sha256":digest(p)?})))
                .collect::<Result<Vec<_>>>()?;
            for _ in 0..warmup {
                runner.recognize_files(&images, &options)?;
            }
            let mut records = Vec::with_capacity(samples);
            for index in 0..samples {
                let mut telemetry = Telemetry::default();
                let t = Instant::now();
                let outputs =
                    runner.recognize_files_with_trace(&images, &options, &mut telemetry)?;
                let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
                records.push(serde_json::json!({"index":index,"wall_ms":wall_ms,"outputs":outputs,"telemetry":telemetry}));
            }
            let report_value = serde_json::json!({"schema":"falcon-ocr-attempt3-report-v1","profile":args.profile,
                "quality_qualified":false,"model_revision":falcon_ocr::config::MODEL_REVISION,
                "weights_sha256":model.weights_sha256(),"binary_sha256":digest(&std::env::current_exe()?)?,
                "hardware":hardware,"threads":args.threads,"backend":args.backend,"batch_size":args.batch_size,
                "head":args.head,"screened_head_bytes":model.screened_head_bytes(),
                "stop_repetition":args.stop_repetition,
                "schedule":"fixed cohorts; layer-major decode; opt-in completed-cache retirement; no refill",
                "options":options,"inputs":inputs,"warmup":warmup,"load_and_import_ms":load_ms,
                "memory_policy":memory,"process_memory":falcon_ocr::attempt::process_memory(),"samples":records,
                "timing_scope":"warm model, original encoded files to all returned text; load/import excluded; report serialization excluded",
                "limits":"KV counters are persistent-allocation snapshots, not OS RSS or transient conversion peaks. Source F32 mapping remains."});
            write_new(&report, &report_value)
                .with_context(|| format!("write {}", report.display()))?;
            println!(
                "{}",
                serde_json::json!({"report":report,"profile":args.profile,"samples":samples})
            );
        }
        Command::Trace {
            fixture,
            output,
            max_new_tokens,
        } => {
            let mut trace = TensorTrace::default();
            let result = runner.trace_reference(fixture, max_new_tokens, &mut trace)?;
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            trace.save(&output)?;
            write_new(
                &output.with_extension("json"),
                &serde_json::json!({"profile":args.profile,"quality_qualified":false,"output":result,"memory_policy":memory}),
            )?;
        }
        Command::Doctor => unreachable!(),
    }
    Ok(())
}
