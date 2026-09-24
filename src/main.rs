use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use falcon_ocr::{
    Backend, CacheLayout, DecodeThreads, ExpMode, GenerationOptions, HeadMode, Mode, Runner, RunnerConfig, Speculation,
    Tuning, WeightLayout,
    auto::{ModelRequest, load_model, resolve_weights},
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
    /// Prefill and pool threads [default: all logical CPUs].
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// What to optimize for [default: near-exact]. Loads the kernel-ready
    /// file for the mode from --model when present, otherwise the FP32
    /// checkpoint, quantized at load (fast mode needs the GPTQ overlay).
    #[arg(long, value_enum, global = true)]
    mode: Option<Mode>,
    /// W8 overlay for `--mode fast` [default: `<model>/w8-gptq.safetensors`,
    /// built by attempt3/make_gptq_overlay.sh].
    #[arg(long, global = true)]
    w8_artifact: Option<PathBuf>,
    /// Let `--mode fast` quantize round-to-nearest at load when no GPTQ
    /// overlay exists (about three times the changed tokens of GPTQ).
    #[arg(long, global = true)]
    allow_rtn: bool,
    /// Kernels: `auto` takes the fastest this CPU runs (AVX2 decode, AVX-512
    /// prefill tiles and BF16 prefill attention where present, NEON on
    /// aarch64); `avx2` is 8-lane FP32 everywhere; `scalar`.
    #[arg(long, value_enum, default_value = "auto", global = true)]
    backend: Backend,
    #[arg(long, default_value_t = 1, global = true)]
    batch_size: usize,
    /// KV cache layout [default: compact].
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
    /// least max(256, 4 * cycle) tokens (finish_reason "repetition"); output
    /// up to that step is unchanged. `--stop-repetition=false` lets loops run
    /// to --max-new-tokens.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, global = true)]
    stop_repetition: bool,
    /// Decode threads: a number, `pool` (as many as --threads) or `auto`
    /// [default: auto]. Decode is memory-bound, so past bandwidth saturation
    /// extra threads and SMT siblings only contend; `auto` times a few team
    /// sizes on the first decode steps and keeps the smallest within 2% of the
    /// fastest (tokens never depend on it). Prefill uses every thread in
    /// --threads.
    #[arg(long, global = true)]
    decode_threads: Option<DecodeThreads>,
    /// Speculative decoding: verify up to N tokens drafted from earlier output
    /// in one step (0 = off, at most 7) [default: 4]. Every verified token is
    /// the model's own greedy choice, so outputs are unchanged. Drafting
    /// switches itself off while drafts are rejected too often to pay (normal
    /// text) and on for repetitive output (tables, loops). Single pages with a
    /// split KV cache (every `run` mode).
    #[arg(long, default_value_t = 4, global = true)]
    speculate: usize,
    /// Minimum n-gram match in the earlier output for a draft.
    #[arg(long, default_value_t = 2, global = true)]
    speculate_min_match: usize,
    /// Treat the images of one `run` as pages of one document: drafts may also
    /// continue full 4-token matches from earlier pages (running headers,
    /// names, repeated table headers). Outputs are unchanged.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, global = true)]
    document_drafts: bool,
    /// Kernel-ready model file written by `pack` (near-exact or fast; the
    /// file decides the mode). Mapped and used in place: fast startup, no
    /// FP32 checkpoint needed. The tokenizer is read from the file's folder
    /// when it holds tokenizer.json, otherwise from --model.
    #[arg(long, global = true)]
    model_file: Option<PathBuf>,
    /// With --model-file: check the digest of every tensor first.
    #[arg(long, global = true)]
    verify_model_file: bool,
    /// Prefill exp: `fast` (the default for `run`; token-identical on
    /// calibration) or `exact` (the platform expf, which `trace` always uses).
    #[arg(long, value_enum, hide = true, global = true)]
    exp: Option<ExpMode>,
    /// Experiment knobs as `key=value` (`prefill-bf16=off|attention|all`,
    /// `split-chunks=1..4`, `phases=1`, `prefill-profile=1`); repeatable.
    #[arg(long = "tune", value_name = "KEY=VALUE", hide = true, global = true)]
    tune: Vec<String>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Show CPU capabilities without loading weights.
    Doctor,
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
    if matches!(cli.command, Command::Doctor) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"host": falcon_ocr::HostInfo::detect()}))?
        );
        return Ok(());
    }
    let start = Instant::now();
    // Exact recognition uses the split FP32 cache (bit-identical to compact,
    // about 7% faster decode); traces and batches keep the reference loader.
    let split_exact = matches!(cli.command, Command::Run { .. })
        && cli.batch_size == 1
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
    let plan = resolve_weights(&ModelRequest {
        model_dir: &cli.model,
        model_file: cli.model_file.as_deref(),
        mode,
        w8_artifact: cli.w8_artifact.as_deref(),
        allow_rtn: cli.allow_rtn,
        split_exact,
        from_checkpoint: matches!(cli.command, Command::Pack { .. }),
    })?;
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
            model.attempt_profile().label()
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
    // Traces stay bit-comparable with the recorded references: the reference
    // configuration (platform exp, full head, no speculation). Recognition
    // uses the automatic one, with the flags above layered on top.
    let base = if matches!(cli.command, Command::Trace { .. }) {
        RunnerConfig::reference()
    } else {
        RunnerConfig::default()
    };
    let config = RunnerConfig {
        threads: cli.threads.unwrap_or(0),
        backend: cli.backend,
        batch_size: cli.batch_size,
        cache_layout: cli.cache_layout.unwrap_or(CacheLayout::Compact),
        weight_layout: cli.weight_layout,
        exp: cli.exp.unwrap_or(base.exp),
        tuning: Tuning::from_pairs(&cli.tune)?,
        head: cli.head.unwrap_or(base.head),
        speculation: (cli.speculate > 0).then_some(Speculation {
            max_draft: cli.speculate,
            min_match: cli.speculate_min_match,
        }),
        document_drafts: cli.document_drafts,
        repetition_stop: cli.stop_repetition,
        decode_threads: cli.decode_threads.unwrap_or(base.decode_threads),
    };
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
