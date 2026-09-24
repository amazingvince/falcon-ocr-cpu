use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use falcon_ocr::{
    CacheLayout, GenerationOptions, Mode, Runner, RunnerConfig, WeightLayout,
    auto::{ModelRequest, load_model, resolve_weights},
    cli::{RunnerArgs, print_doctor},
    model::WeightsSource,
    trace::TensorTrace,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

#[derive(Parser)]
#[command(version, about = "Falcon-OCR v1.5 CPU runner")]
struct Cli {
    /// Model directory: the FP32 checkpoint and/or the published kernel-ready
    /// files (`falcon-ocr-v1.5-<mode>.safetensors`).
    #[arg(long, default_value = "artifacts/model", global = true)]
    model: PathBuf,
    /// What to optimize for [default: near-exact]. Loads the kernel-ready
    /// file for the mode from --model when present, otherwise the FP32
    /// checkpoint, quantized at load (fast mode needs the GPTQ overlay).
    #[arg(long, value_enum, global = true)]
    mode: Option<Mode>,
    /// W8 overlay for `--mode fast` [default: `<model>/w8-gptq.safetensors`,
    /// built by tools/make_gptq_overlay.sh].
    #[arg(long, global = true)]
    w8_artifact: Option<PathBuf>,
    /// Let `--mode fast` quantize round-to-nearest at load when no GPTQ
    /// overlay exists (about three times the changed tokens of GPTQ).
    #[arg(long, global = true)]
    allow_rtn: bool,
    /// KV cache layout [default: compact].
    #[arg(long, value_enum, global = true, hide = true)]
    cache_layout: Option<CacheLayout>,
    /// Experimental extra weight copy for AVX2 batch decode; prefill/row1 unchanged.
    #[arg(long, value_enum, default_value = "unpacked", global = true, hide = true)]
    weight_layout: WeightLayout,
    /// Kernel-ready model file written by `pack` (near-exact or fast; the
    /// file decides the mode). Mapped and used in place: fast startup, no
    /// FP32 checkpoint needed. The tokenizer is read from the file's folder
    /// when it holds tokenizer.json, otherwise from --model.
    #[arg(long, global = true)]
    model_file: Option<PathBuf>,
    /// With --model-file: check the digest of every tensor first.
    #[arg(long, global = true)]
    verify_model_file: bool,
    #[command(flatten)]
    runner: RunnerArgs,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// What `auto` does here: the host, the model files found and the plan
    /// the other flags produce. Reads no tensors unless --load.
    Doctor {
        /// Plain text instead of JSON.
        #[arg(long)]
        text: bool,
        /// Load the model (timed) and build the screened head.
        #[arg(long)]
        load: bool,
        /// Measure memory read bandwidth (512 MB, five passes) and the decode
        /// floor it implies for the plan.
        #[arg(long)]
        probe: bool,
    },
    /// Verify the checkpoint hash and tensor contract.
    Inspect,
    /// Write a kernel-ready model file for --mode near-exact (default) or fast.
    Pack {
        #[arg(long)]
        output: PathBuf,
    },
    /// Recognize full-page PNG/JPEG images; emits JSON lines in input order.
    Run {
        #[arg(required = true)]
        images: Vec<PathBuf>,
        /// Output token cap [default: 8192, lowered with a warning when the
        /// page's input tokens leave less room in the 16384-token context; an
        /// explicit value that does not fit is an error].
        #[arg(long)]
        max_new_tokens: Option<usize>,
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
    let start = Instant::now();
    let batch_size = cli.runner.batch_size.unwrap_or(1);
    // Exact recognition uses the split FP32 cache (bit-identical to compact,
    // about 7% faster decode); traces and batches keep the reference loader.
    let split_exact = matches!(cli.command, Command::Run { .. } | Command::Doctor { .. })
        && batch_size == 1
        && cli.cache_layout.is_none_or(|layout| layout == CacheLayout::Compact)
        && cli.weight_layout == WeightLayout::Unpacked;
    let mode = match (&cli.command, cli.mode) {
        // Traces stay bit-comparable with the recorded FP32 references.
        (Command::Trace { .. }, Some(mode)) if mode != Mode::Exact => {
            bail!("trace compares tensors with the FP32 reference and runs in exact mode; drop --mode {mode}")
        }
        (Command::Trace { .. }, _) => Some(Mode::Exact),
        (Command::Pack { .. }, Some(Mode::Exact)) => {
            bail!("pack writes a near-exact or fast model file from the checkpoint")
        }
        (Command::Pack { .. }, mode) => {
            ensure!(
                cli.model_file.is_none(),
                "pack reads the FP32 checkpoint, not a kernel-ready file"
            );
            Some(mode.unwrap_or(Mode::NearExact))
        }
        (_, mode) => mode,
    };
    let request = ModelRequest {
        model_dir: &cli.model,
        model_file: cli.model_file.as_deref(),
        mode,
        w8_artifact: cli.w8_artifact.as_deref(),
        allow_rtn: cli.allow_rtn,
        split_exact,
        from_checkpoint: matches!(cli.command, Command::Pack { .. }),
        ..ModelRequest::default()
    };
    // Traces stay bit-comparable with the recorded references: the reference
    // configuration (platform exp, full head, no speculation) on every
    // logical CPU. Recognition uses the automatic one. The flags layer on top.
    let base = if matches!(cli.command, Command::Trace { .. }) {
        RunnerConfig {
            threads: 0,
            ..RunnerConfig::reference()
        }
    } else {
        RunnerConfig::default()
    };
    let config = RunnerConfig {
        cache_layout: cli.cache_layout.unwrap_or(CacheLayout::Compact),
        weight_layout: cli.weight_layout,
        ..cli.runner.apply(base)?
    };
    if let Command::Doctor { text, load, probe } = &cli.command {
        return print_doctor(&request, &config, *text, *load, *probe);
    }
    let plan = resolve_weights(&request)?;
    if matches!(plan.source, WeightsSource::Checkpoint { rtn: true, .. }) {
        eprintln!("fast mode: quantizing round-to-nearest at load (--allow-rtn); the GPTQ overlay is closer to FP32");
    }
    let model = Arc::new(load_model(&plan, cli.verify_model_file)?);
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    if let Command::Pack { output } = &cli.command {
        model.prepare_screened_head()?;
        model.write_packed(output)?;
        eprintln!(
            "wrote {} ({:.0} MB, {})",
            output.display(),
            std::fs::metadata(output)?.len() as f64 / 1e6,
            model.profile().label()
        );
        return Ok(());
    }
    if matches!(cli.command, Command::Inspect) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"config":model.config(),
            "weights_sha256":model.weights_sha256(),"load_ms":load_ms,"weights":plan.source,
            "profile":plan.profile,"mode":plan.mode}))?
        );
        return Ok(());
    }
    let tokenizer_dir = plan.source.tokenizer_dir(&cli.model);
    let runner = Runner::new(model, &tokenizer_dir, config)?;
    eprintln!("{} | loaded in {load_ms:.0} ms", runner.resolved());
    match cli.command {
        Command::Run {
            images,
            max_new_tokens,
            min_dimension,
            max_dimension,
            text,
        } => {
            let options = GenerationOptions {
                max_new_tokens: max_new_tokens.unwrap_or(8192),
                min_dimension,
                max_dimension,
                // Without an explicit cap, long pages get what the context leaves.
                fit_budget: max_new_tokens.is_none(),
            };
            for result in runner
                .recognize_files(&images, &options)
                .context("recognize input images")?
            {
                if let Some(tuning) = &result.decode_tuning {
                    eprintln!("{tuning}");
                }
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
            std::fs::write(output.with_extension("json"), serde_json::to_vec_pretty(&result)?)?;
            println!("{}", serde_json::to_string(&result)?);
        }
        _ => unreachable!(),
    }
    Ok(())
}
