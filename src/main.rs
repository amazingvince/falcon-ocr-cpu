use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use falcon_ocr::{
    CacheLayout, GenerationOptions, MaxDimension, Mode, OcrResult, Pipeline, Runner, RunnerConfig, WeightLayout,
    auto::{ModelRequest, WeightsPlan, load_model, model_files_dir, resolve_weights, with_default_draft_head},
    book::{self, Existing, Sink, Summary},
    cli::{RunnerArgs, print_doctor},
    model::WeightsSource,
    quant::Kv,
    runner::{
        ControlFlow,
        escalate::{self, Escalation},
    },
    trace::TensorTrace,
};
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

/// `eprintln!` for a diagnostic that must not stop a run: when stderr is
/// closed (`run … 2>&1 | less`, and the pager quit) the message is dropped
/// instead of panicking.
macro_rules! note {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr().lock(), $($arg)*);
    }};
}

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
    /// built by tools/make_gptq_overlay.sh]. Applied to the FP32 checkpoint;
    /// given explicitly, it is used even when --model holds a packed fast file.
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
    /// Research: the KV cache storage for run and doctor, replacing the
    /// mode's own (`--mode fast --kv-cache q8r` runs w8-body-kv-q8r); results
    /// name the profile that ran.
    #[arg(long, value_enum, global = true, hide = true)]
    kv_cache: Option<Kv>,
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
    /// Recognize full-page PNG/JPEG images: one JSON line per page, in input
    /// order, each printed as soon as its page and every earlier one are
    /// done.
    Run {
        /// Page images, in order [required unless --list].
        #[arg(required_unless_present = "list")]
        images: Vec<PathBuf>,
        /// Also read input paths from FILE, after the positional images: one
        /// per line; blank lines and lines starting with `#` are skipped;
        /// relative paths resolve against the current directory.
        #[arg(long, value_name = "FILE")]
        list: Option<PathBuf>,
        /// Output token cap [default: 8192, lowered with a warning when the
        /// page's input tokens leave less room in the 16384-token context; an
        /// explicit value that does not fit is an error].
        #[arg(long)]
        max_new_tokens: Option<usize>,
        /// Smallest side in pixels: the first resize scales a page with a
        /// side below it so that its shorter side is this, then caps its
        /// longer side at --max-dimension.
        #[arg(long, default_value_t = 64)]
        min_dimension: u32,
        /// Resolution cap in pixels, or `auto`: the resolution router picks
        /// 768, 1024 or 1536 per page (fast mode; changes the output) and
        /// reruns a routed page at 1536 if it loops or hits the length limit.
        #[arg(long, default_value = "1536")]
        max_dimension: MaxDimension,
        /// Cut blank page margins after the first resize, keeping PAD pixels
        /// around the content: `--crop-margins` keeps 24, `--crop-margins=PAD`
        /// (with `=`) sets PAD. Fewer image tokens, text at the same size.
        /// Changes the model input, so off by default; each result reports
        /// its crop (docs/MODES.md).
        #[arg(long, value_name = "PAD", num_args = 0..=1, require_equals = true, default_missing_value = "24")]
        crop_margins: Option<u32>,
        /// Print each page's text instead of its JSON record (with --output,
        /// the records still go to the file, and the run goes on if stdout
        /// closes).
        #[arg(long)]
        text: bool,
        /// Append the JSON records to FILE instead of printing them, flushed
        /// after every page. An existing FILE must hold only records; a last
        /// record cut short by a crash is removed first.
        #[arg(long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Skip inputs that already have a successful record in the --output
        /// file (a page whose record was cut short by a crash runs again).
        #[arg(long, requires = "output")]
        resume: bool,
        /// Record a page that fails as {"path", "page", "error"} (with --text
        /// and no --output, only its error on stderr) and go on; the exit
        /// code is non-zero at the end if any page failed. Without it the
        /// run stops at the first failing page. Either way an input the run
        /// reads (any but those --resume skips) that is missing or not a PNG
        /// or JPEG stops the run before the model loads.
        #[arg(long)]
        keep_going: bool,
        /// Read, prepare and prefill the next page while the current pages
        /// decode (tokens unchanged).
        #[arg(long)]
        pipeline: bool,
        /// Threads that prefill the next page with --pipeline, at most
        /// --threads [default: --threads minus the decode team, an automatic
        /// team then taking at most half of 3 or more --threads; when that
        /// leaves fewer than 2, pages prefill between decodes instead]. Given
        /// explicitly, it leaves an automatic team unlimited.
        #[arg(long, requires = "pipeline")]
        prefill_threads: Option<usize>,
        /// Reread fast-mode pages that the repetition stop ended with the
        /// near-exact model (loaded when first needed; the packed near-exact
        /// file, else the checkpoint).
        #[arg(long)]
        escalate: bool,
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
    ensure!(
        cli.kv_cache.is_none() || matches!(cli.command, Command::Run { .. } | Command::Doctor { .. }),
        "--kv-cache applies to run and doctor; trace, pack and inspect use the mode's own cache"
    );
    let request = ModelRequest {
        model_dir: &cli.model,
        model_file: cli.model_file.as_deref(),
        mode,
        w8_artifact: cli.w8_artifact.as_deref(),
        allow_rtn: cli.allow_rtn,
        split_exact,
        from_checkpoint: matches!(cli.command, Command::Pack { .. }),
        kv_cache: cli.kv_cache,
        ..ModelRequest::default()
    };
    let config = runner_config(&cli)?;
    // The published draft head next to the model files drafts unless a
    // drafter was chosen; traces run without speculation, so never use it.
    let explicit = cli.runner.drafter.is_some() || cli.runner.draft_head.is_some();
    let config = with_default_draft_head(config, &model_files_dir(&request), explicit);
    if let Command::Doctor { text, load, probe } = &cli.command {
        return print_doctor(&request, &config, *text, *load, *probe);
    }
    // `run` checks its options and inputs, then the weights (and those
    // `--escalate` needs); the model loads and the runner checks its
    // configuration before the output file is created or repaired, so a
    // run that cannot start leaves no output.
    let job = RunJob::new(&cli.command, &config)?;
    let plan = resolve_weights(&request)?;
    let escalation = match &job {
        Some(job) if job.escalate => Some(escalation(&plan, &request, &config, &cli.model, cli.verify_model_file)?),
        _ => None,
    };
    if matches!(plan.source, WeightsSource::Checkpoint { rtn: true, .. }) {
        note!("fast mode: quantizing round-to-nearest at load (--allow-rtn); the GPTQ overlay is closer to FP32");
    }
    let model = Arc::new(load_model(&plan, cli.verify_model_file)?);
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    if let Command::Pack { output } = &cli.command {
        model.prepare_screened_head()?;
        model.write_packed(output)?;
        note!(
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
    note!("{} | loaded in {load_ms:.0} ms", runner.resolved());
    if let Some(job) = job {
        let output = job.open_output()?;
        return job.run(&runner, output, escalation);
    }
    match cli.command {
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

/// The runner configuration the flags ask for. Traces stay bit-comparable
/// with the recorded references: the reference configuration (platform exp,
/// full head, no speculation) on every logical CPU. Recognition uses the
/// automatic one. The flags layer on top; the default draft head is added
/// later, where the model files are known.
fn runner_config(cli: &Cli) -> Result<RunnerConfig> {
    let base = if matches!(cli.command, Command::Trace { .. }) {
        RunnerConfig {
            threads: 0,
            ..RunnerConfig::reference()
        }
    } else {
        RunnerConfig::default()
    };
    Ok(RunnerConfig {
        cache_layout: cli.cache_layout.unwrap_or(CacheLayout::Compact),
        weight_layout: cli.weight_layout,
        ..cli.runner.apply(base)?
    })
}

/// What a `run` command asks of every page and of the page pipeline: its
/// generation options (checked by `RunJob::new`) and `--pipeline`; `None`
/// for other commands.
fn run_settings(command: &Command) -> Option<(GenerationOptions, Option<Pipeline>)> {
    let Command::Run {
        max_new_tokens,
        min_dimension,
        max_dimension,
        crop_margins,
        pipeline,
        prefill_threads,
        ..
    } = command
    else {
        return None;
    };
    let mut options = GenerationOptions {
        max_new_tokens: max_new_tokens.unwrap_or(8192),
        min_dimension: *min_dimension,
        // Without an explicit cap, long pages get what the context leaves.
        fit_budget: max_new_tokens.is_none(),
        crop_margins: *crop_margins,
        ..GenerationOptions::default()
    };
    max_dimension.apply(&mut options);
    let pipeline = pipeline.then_some(Pipeline {
        prefill_threads: *prefill_threads,
    });
    Some((options, pipeline))
}

/// A `run` command checked before the model loads: its options and inputs.
struct RunJob {
    inputs: Vec<PathBuf>,
    options: GenerationOptions,
    text: bool,
    /// `--output` ([`RunJob::open_output`]).
    output: Option<PathBuf>,
    resume: bool,
    keep_going: bool,
    pipeline: Option<Pipeline>,
    escalate: bool,
}

impl RunJob {
    /// The job of `command` when it is `run`, after checking the flags and
    /// that every input it will read (under `--resume`, those without a
    /// successful record) is a readable PNG or JPEG file. Nothing is written.
    fn new(command: &Command, config: &RunnerConfig) -> Result<Option<Self>> {
        let (
            Some((options, pipeline)),
            Command::Run {
                images,
                list,
                text,
                output,
                resume,
                keep_going,
                prefill_threads,
                escalate,
                ..
            },
        ) = (run_settings(command), command)
        else {
            return Ok(None);
        };
        options.validate()?;
        config.validate()?;
        // The runner's pool: --threads, or every logical CPU.
        let threads = match config.threads {
            0 => falcon_ocr::cpu::logical_cpus(),
            threads => threads,
        };
        ensure!(
            prefill_threads.is_none_or(|prefill| (1..=threads).contains(&prefill)),
            "--prefill-threads must be 1..={threads}, the runner's threads"
        );
        ensure!(
            !*escalate || config.repetition_stop,
            "--escalate rereads the pages that the repetition stop ended; drop --stop-repetition=false"
        );
        if let Some(path) = output.as_deref() {
            ensure!(!path.is_dir(), "--output {} is a directory", path.display());
            // The file itself is created after the model loads; a missing
            // folder would only fail then.
            let folder = match path.parent() {
                Some(folder) if !folder.as_os_str().is_empty() => folder,
                _ => Path::new("."),
            };
            ensure!(
                folder.is_dir(),
                "--output {}: the folder {} does not exist",
                path.display(),
                folder.display()
            );
            ensure!(
                !*resume || book::is_file_or_missing(path),
                "--resume reads the records in --output, and {} is not a regular file",
                path.display()
            );
        }
        let mut inputs = images.clone();
        if let Some(list) = list {
            inputs.extend(book::read_list(list)?);
            ensure!(!inputs.is_empty(), "no input images: {} lists none", list.display());
        }
        // A record names its input by its path, as text: a path that is not
        // UTF-8 would lose bytes, and could then share a record with another
        // (which a resumed run would skip).
        if output.is_some()
            && let Some(path) = inputs.iter().find(|input| input.to_str().is_none())
        {
            bail!("{path:?} is not UTF-8, so a record in --output cannot name it; rename it");
        }
        if *resume {
            // Resumed runs match records by path.
            let mut seen = std::collections::HashSet::new();
            if let Some(twice) = inputs.iter().find(|input| !seen.insert(input.to_string_lossy())) {
                bail!(
                    "--resume matches records by input path, and {} is an input twice",
                    twice.display()
                );
            }
        }
        // The output is read (not yet created or repaired) before the model
        // loads, so a file that is not an output (an input image or list) is
        // refused first. A resumed run reads only the inputs that have no
        // successful record yet, so the others need not exist any more.
        let existing = match output.as_deref() {
            Some(path) => book::read_output(path)?,
            None => Default::default(),
        };
        if let Some(path) = output.as_deref().filter(|_| !*resume) {
            let again = inputs
                .iter()
                .map(|input| input.to_string_lossy())
                .filter(|input| existing.done.contains(input.as_ref()))
                .collect::<std::collections::HashSet<_>>()
                .len();
            if again > 0 {
                note!(
                    "warning: {} already holds records of {again} of these pages; without --resume they run again \
                     and get a second record",
                    path.display()
                );
            }
        }
        let done = if *resume { existing.done } else { Default::default() };
        for input in &inputs {
            if !done.contains(input.to_string_lossy().as_ref()) {
                book::check_input(input)?;
            }
        }
        Ok(Some(Self {
            inputs,
            options,
            text: *text,
            output: output.clone(),
            resume: *resume,
            keep_going: *keep_going,
            pipeline,
            escalate: *escalate,
        }))
    }

    /// `--output`, open for appending, and what it already held; a record
    /// that a crash cut short is removed first. Opened once every other
    /// check has passed.
    fn open_output(&self) -> Result<Option<(File, Existing)>> {
        let Some(path) = &self.output else {
            return Ok(None);
        };
        let (file, existing) = book::open_output(path)?;
        if existing.cut > 0 {
            note!(
                "{}: removed a last record cut short ({} bytes)",
                path.display(),
                existing.cut
            );
        }
        Ok(Some((file, existing)))
    }

    /// Recognize the inputs that still need it, writing each page as soon
    /// as it is done, then print the summary. Fails at the end when a page
    /// failed.
    fn run(
        self,
        runner: &Runner,
        output: Option<(File, Existing)>,
        mut escalation: Option<Escalation<'_>>,
    ) -> Result<()> {
        let (file, existing) = output.map_or((None, None), |(file, existing)| (Some(file), Some(existing)));
        let done = existing.as_ref().filter(|_| self.resume).map(|existing| &existing.done);
        let resolved = runner.resolved();
        let this = book::this_run(
            resolved.mode,
            &runner.precision(),
            resolved.overlay_sha256.as_deref(),
            &self.options,
            book::Kernels::of(resolved),
        );
        if let Some(existing) = existing.as_ref().filter(|_| self.resume)
            && let Some(warning) = book::resume_warning(existing, &this)
        {
            note!("warning: {warning}");
        }
        if let Some(existing) = existing.as_ref().filter(|_| self.resume && self.escalate)
            && let Some(warning) = book::escalation_warning(existing)
        {
            note!("warning: {warning}");
        }
        // (input index, path) of every page that still needs a record.
        let todo: Vec<(usize, &PathBuf)> = self
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, path)| done.is_none_or(|done| !done.contains(path.to_string_lossy().as_ref())))
            .collect();
        let paths: Vec<&PathBuf> = todo.iter().map(|&(_, path)| path).collect();
        let mut summary = Summary {
            skipped: self.inputs.len() - todo.len(),
            ..Summary::default()
        };
        let mut sink = Sink::new(std::io::stdout().lock(), file, self.text, Some(self.options.clone()));
        let mut failure = None;
        let options = &self.options;
        let started = Instant::now();
        let on_page = |index: usize, result: Result<OcrResult>| {
            let (page, path) = todo[index];
            let result = result.map(|result| {
                print_tuning(&result);
                let Some(escalation) = escalation.as_mut().filter(|_| escalate::needs_escalation(&result)) else {
                    return result;
                };
                note!(
                    "page {page} ({}): the repetition stop ended it in fast mode after {} tokens; rereading it in near-exact mode",
                    path.display(),
                    result.output_tokens
                );
                let result = escalation.escalate(path, options, result);
                match &result.escalation_error {
                    Some(error) => note!(
                        "warning: page {page} ({}): kept the fast-mode result: {error}",
                        path.display()
                    ),
                    None => print_tuning(&result),
                }
                result
            });
            let written = match result {
                // Done once its record is written; a page that ran but could
                // not be written failed.
                Ok(result) => {
                    let written = sink.page(path, page, &result);
                    match written {
                        Ok(()) => summary.done += 1,
                        Err(_) => summary.failed += 1,
                    }
                    written
                }
                Err(error) if self.keep_going => {
                    summary.failed += 1;
                    note!("page {page} ({}) failed: {error:#}", path.display());
                    sink.error(path, page, &error)
                }
                Err(error) => {
                    summary.failed += 1;
                    failure = Some(error.context(format!("recognize {}", path.display())));
                    return ControlFlow::Break(());
                }
            };
            written.map_or_else(
                |error| {
                    failure = Some(error.context("write the output"));
                    ControlFlow::Break(())
                },
                ControlFlow::Continue,
            )
        };
        match &self.pipeline {
            Some(pipeline) => runner.recognize_files_pipelined(&paths, options, pipeline, on_page)?,
            None => runner.recognize_files_streaming(&paths, options, on_page)?,
        }
        summary.seconds = started.elapsed().as_secs_f64();
        summary.not_run = todo.len() - summary.done - summary.failed;
        note!("{summary}");
        if let Some(error) = failure {
            return Err(error);
        }
        ensure!(summary.failed == 0, "{} of {} pages failed", summary.failed, todo.len());
        Ok(())
    }
}

/// The decode-thread tuner's choice, on the page that carries it.
fn print_tuning(result: &OcrResult) {
    if let Some(tuning) = &result.decode_tuning {
        note!("{tuning}");
    }
}

/// `run --escalate` in fast mode: the near-exact weights are found now (no
/// tensors read), and the runner loads when a page first needs it, with this
/// run's configuration and one page at a time.
fn escalation(
    plan: &WeightsPlan,
    request: &ModelRequest<'_>,
    config: &RunnerConfig,
    model_dir: &Path,
    verify: bool,
) -> Result<Escalation<'static>> {
    match plan.mode {
        Some(Mode::Fast) => {}
        Some(mode) => bail!(
            "--escalate rereads fast-mode pages in near-exact mode; this run is {} already",
            mode.label()
        ),
        // A research profile (`--kv-cache` other than the mode's own) is
        // compared with, not replaced by, the near-exact model.
        None => bail!(
            "--escalate rereads fast-mode pages in near-exact mode; this run is the research profile {}",
            plan.profile.label()
        ),
    }
    // Only the model files are shared: the near-exact run gets its own
    // weights and cache (no --w8-artifact, no --kv-cache).
    let near_exact = escalate::near_exact_weights(request).context("--escalate needs a near-exact model")?;
    let config = RunnerConfig {
        batch_size: 1,
        ..config.clone()
    };
    let model_dir = model_dir.to_path_buf();
    Ok(Escalation::new(move || {
        let started = Instant::now();
        let model = Arc::new(load_model(&near_exact, verify)?);
        let runner = Runner::new(model, near_exact.source.tokenizer_dir(&model_dir), config.clone())?;
        note!(
            "escalation: {} | loaded in {:.0} ms",
            runner.resolved(),
            started.elapsed().as_secs_f64() * 1000.
        );
        Ok(runner)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use falcon_ocr::DecodeThreads;

    fn parse(arguments: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("falcon-ocr").chain(arguments.iter().copied())).unwrap()
    }

    /// Every `run` flag that changes a page or the pipeline reaches the
    /// options, the pipeline or the runner configuration: a flag dropped on
    /// the way would change the output without an error.
    #[test]
    fn run_flags_reach_the_options_the_pipeline_and_the_config() {
        let cli = parse(&[
            "--batch-size",
            "3",
            "--decode-threads",
            "1",
            "run",
            "page.png",
            "--max-new-tokens",
            "24",
            "--min-dimension",
            "32",
            "--max-dimension",
            "auto",
            "--crop-margins=40",
            "--pipeline",
            "--prefill-threads",
            "1",
        ]);
        let (options, pipeline) = run_settings(&cli.command).unwrap();
        assert_eq!(
            (options.max_new_tokens, options.fit_budget, options.min_dimension),
            (24, false, 32)
        );
        assert_eq!((options.max_dimension, options.route), (1536, true));
        assert_eq!(options.crop_margins, Some(40));
        assert_eq!(
            pipeline,
            Some(Pipeline {
                prefill_threads: Some(1)
            })
        );
        let config = runner_config(&cli).unwrap();
        assert_eq!((config.batch_size, config.decode_threads), (3, DecodeThreads::Fixed(1)));
        // Defaults: the fitted budget, no crop, no pipeline, one page at a time.
        let cli = parse(&["run", "page.png", "--max-dimension", "1024"]);
        let (options, pipeline) = run_settings(&cli.command).unwrap();
        assert_eq!((options.max_new_tokens, options.fit_budget), (8192, true));
        assert_eq!(
            (options.max_dimension, options.route, options.crop_margins),
            (1024, false, None)
        );
        assert_eq!(pipeline, None);
        assert_eq!(runner_config(&cli).unwrap().batch_size, 1);
        // A bare --crop-margins keeps 24 pixels; --pipeline alone sizes its
        // second pool from the decode team.
        let cli = parse(&["run", "page.png", "--crop-margins", "--pipeline"]);
        let (options, pipeline) = run_settings(&cli.command).unwrap();
        assert_eq!(options.crop_margins, Some(24));
        assert_eq!(pipeline, Some(Pipeline::default()));
        assert!(run_settings(&parse(&["doctor"]).command).is_none());
    }
}
