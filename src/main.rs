use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use falcon_ocr::{
    Backend, Bf16Model, Bf16Runner, CacheLayout, GenerationOptions, HeadMode, Model, Precision,
    Runner, RunnerConfig, WeightLayout, trace::TensorTrace,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

/// What the FP32 runner optimizes for (see `attempt3/RESULTS-V2.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    /// FP32 weights and caches. Tokens identical to the FP32 reference on
    /// the 67-page calibration set.
    Exact,
    /// 16-bit body weights and 16-bit KV cache (absmax scale per 64 weights
    /// or 32 cache values, quantized at load): about 1.9x faster decode than
    /// exact. Not bitwise: 1 token in 24,262 differed from FP32 when
    /// teacher-forcing 55 calibration pages (KL 9e-8 per token).
    NearExact,
    /// 8-bit body weights (W8G64) and 8-bit KV cache: about 3x faster
    /// decode. Ground-truth neutral on calibration pages that end at EOS,
    /// but different tokens on most pages; pair with --stop-repetition.
    Fast,
}

/// GPTQ overlay that `--mode fast` loads from the model directory by default.
const DEFAULT_OVERLAY: &str = "w8-gptq.safetensors";

#[derive(Parser)]
#[command(version, about = "Falcon-OCR v1.5 CPU runner")]
struct Cli {
    #[arg(long, default_value = "artifacts/model", global = true)]
    model: PathBuf,
    /// Prefill and pool threads [default: all logical CPUs].
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// FP32 runner mode.
    #[arg(long, value_enum, default_value = "exact", global = true)]
    mode: Mode,
    /// W8 overlay for `--mode fast`. Default: `<model>/w8-gptq.safetensors`
    /// when present (GPTQ, built by attempt3/make_gptq_overlay.sh), otherwise
    /// round-to-nearest quantization at load.
    #[arg(long, global = true)]
    w8_artifact: Option<PathBuf>,
    /// BF16 is an experimental single-request graph, separate from FP32 defaults.
    #[arg(long, value_enum, default_value = "fp32", global = true)]
    precision: Precision,
    /// Under BF16, avx512 requires AVX-512F and AVX-512BF16; avx2 is unsupported.
    #[arg(long, value_enum, default_value = "auto", global = true)]
    backend: Backend,
    #[arg(long, default_value_t = 1, global = true)]
    batch_size: usize,
    /// FP32 defaults to `compact`; the experimental BF16 graph defaults to `expanded`.
    #[arg(long, value_enum, global = true)]
    cache_layout: Option<CacheLayout>,
    /// Experimental extra weight copy for AVX2 batch decode; prefill/row1 unchanged.
    #[arg(long, value_enum, default_value = "unpacked", global = true)]
    weight_layout: WeightLayout,
    /// FP32 greedy head [default: screened]: `screened` selects the same
    /// tokens as `full` through an exact INT8 screen (53 MB extra); traces
    /// always record full logits.
    #[arg(long, value_enum, global = true)]
    head: Option<HeadMode>,
    /// Stop a page once it repeats a cycle of at most 128 tokens for at
    /// least max(256, 4 * cycle) tokens (finish_reason "repetition").
    #[arg(long, global = true)]
    stop_repetition: bool,
    /// Decode threads, or `auto` [default: auto]. Decode is memory-bound, so
    /// past bandwidth saturation extra threads and SMT siblings only contend;
    /// `auto` times a few team sizes on the first decode steps and keeps the
    /// smallest within 2% of the fastest (tokens never depend on it).
    /// Prefill uses every logical CPU in --threads.
    #[arg(long, global = true)]
    decode_threads: Option<falcon_ocr::runner::DecodeThreads>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Show CPU capabilities without loading weights.
    Doctor,
    /// Verify the checkpoint hash and tensor contract.
    Inspect,
    /// Recognize full-page PNG/JPEG images; emits JSON lines in input order.
    Run {
        #[arg(required = true)]
        images: Vec<PathBuf>,
        #[arg(long, default_value_t = 8192)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 64)]
        min_dimension: u32,
        #[arg(long, default_value_t = 1536)]
        max_dimension: u32,
        #[arg(long)]
        text: bool,
    },
    /// Capture tensors using canonical GPU-exported input patches.
    Trace {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
    },
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Command::Doctor) {
        let mut features = serde_json::json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
            "logical_cpus":falcon_ocr::cpu::logical_cpus(),"physical_cores":falcon_ocr::cpu::physical_cores(),
            "precision":cli.precision,"gpu_parity_qualified":false});
        #[cfg(target_arch = "x86_64")]
        {
            features["avx2"] = std::is_x86_feature_detected!("avx2").into();
            features["fma"] = std::is_x86_feature_detected!("fma").into();
            features["avx512f"] = std::is_x86_feature_detected!("avx512f").into();
            features["avx512bf16"] = std::is_x86_feature_detected!("avx512bf16").into();
            features["avx512vnni"] = std::is_x86_feature_detected!("avx512vnni").into();
        }
        println!("{}", serde_json::to_string_pretty(&features)?);
        return Ok(());
    }
    if cli.precision == Precision::Bf16 {
        return execute_bf16(cli);
    }
    anyhow::ensure!(
        cli.w8_artifact.is_none() || cli.mode == Mode::Fast,
        "--w8-artifact applies to --mode fast"
    );
    let start = Instant::now();
    // Exact recognition uses the split FP32 cache (bit-identical to compact,
    // about 7% faster decode); traces and batches keep the reference loader.
    let split_exact = matches!(cli.command, Command::Run { .. })
        && cli.batch_size == 1
        && cli
            .cache_layout
            .is_none_or(|layout| layout == CacheLayout::Compact)
        && cli.weight_layout == WeightLayout::Unpacked;
    let model = Arc::new(match cli.mode {
        Mode::Exact if split_exact => {
            Model::load_attempt(&cli.model, falcon_ocr::attempt::Profile::SplitF32, None)?
        }
        Mode::Exact => Model::load(&cli.model)?,
        Mode::NearExact => {
            Model::load_attempt(&cli.model, falcon_ocr::attempt::Profile::W16BodyKvQ16, None)?
        }
        Mode::Fast => {
            let overlay = cli
                .w8_artifact
                .clone()
                .or_else(|| Some(cli.model.join(DEFAULT_OVERLAY)).filter(|p| p.is_file()));
            match &overlay {
                Some(path) => eprintln!("fast mode: W8 weights from {}", path.display()),
                None => eprintln!(
                    "fast mode: no {DEFAULT_OVERLAY} in the model directory; quantizing                      round-to-nearest at load (GPTQ is closer to FP32:                      attempt3/make_gptq_overlay.sh)"
                ),
            }
            Model::load_attempt(
                &cli.model,
                falcon_ocr::attempt::Profile::W8BodyKvQ8,
                overlay.as_deref(),
            )?
        }
    });
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    if matches!(cli.command, Command::Inspect) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"config":model.config(),
            "weights_sha256":model.weights_sha256(),"load_ms":load_ms}))?
        );
        return Ok(());
    }
    eprintln!("Verified model loaded in {load_ms:.1} ms");
    let threads = cli.threads.unwrap_or_else(falcon_ocr::cpu::logical_cpus);
    let mut runner = Runner::new(
        model,
        &cli.model,
        RunnerConfig {
            threads,
            backend: cli.backend,
            batch_size: cli.batch_size,
            cache_layout: cli.cache_layout.unwrap_or(CacheLayout::Compact),
            weight_layout: cli.weight_layout,
        },
    )?;
    runner.set_head_mode(cli.head.unwrap_or(HeadMode::Screened))?;
    runner.set_repetition_stop(cli.stop_repetition);
    match cli
        .decode_threads
        .unwrap_or(falcon_ocr::runner::DecodeThreads::Auto)
    {
        falcon_ocr::runner::DecodeThreads::Auto => runner.set_decode_threads_auto()?,
        falcon_ocr::runner::DecodeThreads::Fixed(n) if n != threads => runner.set_decode_threads(n)?,
        falcon_ocr::runner::DecodeThreads::Fixed(_) => {}
    }
    // Recognition uses the fast exp (token-identical on calibration);
    // FALCON_OCR_EXP=exact keeps the platform exp. Traces always keep it, so
    // they stay bit-comparable with the recorded references.
    let exact_exp = std::env::var("FALCON_OCR_EXP").is_ok_and(|v| v == "exact")
        || matches!(cli.command, Command::Trace { .. });
    falcon_ocr::kernels::set_exp_mode(if exact_exp {
        falcon_ocr::kernels::ExpMode::Exact
    } else {
        falcon_ocr::kernels::ExpMode::Fast
    });
    match cli.command {
        Command::Run {
            images,
            max_new_tokens,
            min_dimension,
            max_dimension,
            text,
        } => {
            let options = GenerationOptions {
                max_new_tokens,
                min_dimension,
                max_dimension,
            };
            for result in runner
                .recognize_files(&images, &options)
                .context("recognize input images")?
            {
                if text {
                    println!("{}", result.text);
                } else {
                    println!("{}", serde_json::to_string(&result)?);
                }
            }
        }
        Command::Trace {
            fixture,
            output,
            max_new_tokens,
        } => {
            let mut trace = TensorTrace::default();
            let result = runner.trace_reference(fixture, max_new_tokens, &mut trace)?;
            trace.save(&output)?;
            std::fs::write(
                output.with_extension("json"),
                serde_json::to_vec_pretty(&result)?,
            )?;
            println!("{}", serde_json::to_string(&result)?);
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn execute_bf16(cli: Cli) -> Result<()> {
    anyhow::ensure!(
        cli.head.is_none_or(|head| head == HeadMode::Full),
        "the experimental BF16 graph has no screened head"
    );
    anyhow::ensure!(
        cli.mode == Mode::Exact && cli.w8_artifact.is_none(),
        "--mode fast and --w8-artifact apply to the FP32 runner"
    );
    let config = RunnerConfig {
        threads: cli.threads.unwrap_or(16),
        backend: cli.backend,
        batch_size: cli.batch_size,
        cache_layout: cli.cache_layout.unwrap_or(CacheLayout::Expanded),
        weight_layout: cli.weight_layout,
    };
    Bf16Runner::validate_config(&config)?;
    let start = Instant::now();
    let model = Arc::new(Bf16Model::load(&cli.model)?);
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    if matches!(cli.command, Command::Inspect) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"config":model.config(),
            "precision":"bf16","experimental":true,"gpu_parity_qualified":false,
            "weights_sha256":model.weights_sha256(),"weight_tensor_bytes":model.weight_tensor_bytes(),"load_ms":load_ms}))?
        );
        return Ok(());
    }
    eprintln!("Experimental BF16 model loaded in {load_ms:.1} ms; GPU qualification is incomplete");
    let runner = Bf16Runner::new(model, &cli.model, config)?;
    match cli.command {
        Command::Run {
            images,
            max_new_tokens,
            min_dimension,
            max_dimension,
            text,
        } => {
            let options = GenerationOptions {
                max_new_tokens,
                min_dimension,
                max_dimension,
            };
            for output in runner
                .recognize_files(&images, &options)
                .context("recognize input images with experimental BF16")?
            {
                if text {
                    println!("{}", output.result.text);
                } else {
                    println!("{}", serde_json::to_string(&output)?);
                }
            }
        }
        Command::Trace {
            fixture,
            output,
            max_new_tokens,
        } => {
            let mut trace = TensorTrace::default();
            let result = runner.trace_reference(fixture, max_new_tokens, &mut trace)?;
            trace.save(&output)?;
            std::fs::write(
                output.with_extension("json"),
                serde_json::to_vec_pretty(&result)?,
            )?;
            println!("{}", serde_json::to_string(&result)?);
        }
        _ => unreachable!(),
    }
    Ok(())
}
