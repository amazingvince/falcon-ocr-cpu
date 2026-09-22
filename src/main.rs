use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use falcon_ocr::{
    Backend, Bf16Model, Bf16Runner, CacheLayout, GenerationOptions, HeadMode, Model, Precision,
    Runner, RunnerConfig, WeightLayout, trace::TensorTrace,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

#[derive(Parser)]
#[command(version, about = "Falcon-OCR v1.5 CPU runner")]
struct Cli {
    #[arg(long, default_value = "artifacts/model", global = true)]
    model: PathBuf,
    #[arg(long, default_value_t = 16, global = true)]
    threads: usize,
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
    /// FP32 greedy head: `screened` selects the same tokens through an exact
    /// INT8 screen (53 MB extra); traces always record full logits.
    #[arg(long, value_enum, default_value = "full", global = true)]
    head: HeadMode,
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
            "logical_cpus":std::thread::available_parallelism().map(|n|n.get()).unwrap_or(1),
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
    let start = Instant::now();
    let model = Arc::new(Model::load(&cli.model)?);
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
    let mut runner = Runner::new(
        model,
        &cli.model,
        RunnerConfig {
            threads: cli.threads,
            backend: cli.backend,
            batch_size: cli.batch_size,
            cache_layout: cli.cache_layout.unwrap_or(CacheLayout::Compact),
            weight_layout: cli.weight_layout,
        },
    )?;
    runner.set_head_mode(cli.head)?;
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
        cli.head == HeadMode::Full,
        "the experimental BF16 graph has no screened head"
    );
    let config = RunnerConfig {
        threads: cli.threads,
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
