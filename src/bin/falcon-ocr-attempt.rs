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
    /// Kernel-ready model file (`falcon-ocr pack`); overrides --profile's weights.
    #[arg(long, global = true)]
    model_file: Option<PathBuf>,
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
    /// Body matrices kept in FP32 whatever the W8 overlay holds, by name
    /// (comma-separated; `*` matches any characters), e.g.
    /// `layers.3.feed_forward.w2.weight` or `layers.3.*`.
    #[arg(long, value_delimiter = ',', global = true)]
    keep_fp32: Vec<String>,
    /// Round every checkpoint tensor to BF16 precision first (production
    /// serves BF16); arithmetic stays FP32.
    #[arg(long, global = true)]
    weights_bf16: bool,
    /// FP32 greedy head evaluation; `screened` is exact (same tokens as `full`).
    #[arg(long, value_enum, default_value = "full", global = true)]
    head: HeadMode,
    /// Stop a page once it repeats a cycle of at most 128 tokens for at
    /// least max(256, 4 * cycle) tokens (finish_reason "repetition").
    #[arg(long, global = true)]
    stop_repetition: bool,
    /// Decode threads (default: --threads). Decode is memory-bound; on SMT
    /// CPUs one thread per physical core is usually fastest, while prefill
    /// gains from every logical CPU in --threads.
    #[arg(long, global = true)]
    decode_threads: Option<falcon_ocr::runner::DecodeThreads>,
    /// Speculative decoding: verify up to N tokens drafted from earlier output
    /// in one step (0 = off, at most 7). Every verified token is the model's own
    /// greedy choice, so outputs are unchanged. Needs the split cache
    /// (near-exact, fast and attempt profiles); single pages only.
    #[arg(long, default_value_t = 0, global = true)]
    speculate: usize,
    /// Minimum n-gram match in the earlier output for a draft.
    #[arg(long, default_value_t = 2, global = true)]
    speculate_min_match: usize,
    /// Cross-page drafts: the images of one bench sample are one document.
    #[arg(long, global = true)]
    document_drafts: bool,
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
    /// Free-running pages with the FP32 reference profile, accumulating the
    /// mean `XᵀX` of every body projection input (GPTQ calibration).
    CaptureGram {
        #[arg(required = true)]
        images: Vec<PathBuf>,
        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 1536)]
        max_dimension: u32,
        /// Directory for `<tensor>.gram.npy`; must not already exist.
        #[arg(long)]
        output: PathBuf,
    },
    /// Teacher-force each page's reference tokens (from a `bench` report of
    /// the same pages) and count the steps where this profile's greedy choice
    /// differs: a per-step fidelity measure that one early divergence cannot
    /// inflate.
    Agree {
        #[arg(required = true)]
        images: Vec<PathBuf>,
        /// `bench` report whose outputs supply the forced tokens.
        #[arg(long)]
        reference: PathBuf,
        /// Forced steps per page (the reference output is truncated to this).
        #[arg(long, default_value_t = 1024)]
        max_steps: usize,
        #[arg(long, default_value_t = 1536)]
        max_dimension: u32,
        #[arg(long, default_value_t = 64)]
        min_dimension: u32,
        /// Must not already exist.
        #[arg(long)]
        report: PathBuf,
        /// Write this run's top-K next-token log-probabilities per step
        /// (JSON), to score other profiles against (run it with FP32).
        #[arg(long)]
        dump_topk: Option<PathBuf>,
        /// Score KL(reference || this profile) per step against a
        /// `--dump-topk` file of the same pages and steps.
        #[arg(long)]
        reference_topk: Option<PathBuf>,
    },
}

/// Log-probabilities kept per step by `--dump-topk`.
const TOP_K: usize = 32;

/// `log softmax(logits)` in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&l| l as f64 - lse).collect()
}

/// KL(P || Q) with P given by its top-K `(id, log p)` and Q by full
/// log-probabilities; every other token is pooled into one tail bucket on
/// both sides (a lower bound on the full KL, identical for all candidates).
fn kl_topk(reference: &[(u32, f32)], log_q: &[f64]) -> f64 {
    let (mut kl, mut p_top, mut q_top) = (0.0, 0.0, 0.0);
    for &(id, log_p) in reference {
        let (log_p, log_q) = (log_p as f64, log_q[id as usize]);
        let p = log_p.exp();
        kl += p * (log_p - log_q);
        p_top += p;
        q_top += log_q.exp();
    }
    let (p_tail, q_tail) = ((1.0 - p_top).max(0.0), (1.0 - q_top).max(1e-300));
    if p_tail > 0.0 {
        kl += p_tail * (p_tail.ln() - q_tail.ln());
    }
    kl.max(0.0)
}

/// Accumulates `XᵀX` of every projection input over free-running pages (GPTQ
/// calibration). Single decode rows are buffered and multiplied in blocks.
struct GramCapture {
    widths: [(&'static str, usize); 4],
    /// Per (layer, site): f64 sum of XᵀX, row count, pending rows.
    sums: Vec<Vec<f64>>,
    rows: Vec<usize>,
    pending: Vec<Vec<f32>>,
}
impl GramCapture {
    const FLUSH_ROWS: usize = 256;
    fn new(layers: usize, dim: usize, query_dim: usize, ffn_dim: usize) -> Self {
        let widths = [
            ("qkv", dim),
            ("wo", query_dim),
            ("w13", dim),
            ("w2", ffn_dim),
        ];
        let mut sums = Vec::new();
        for _ in 0..layers {
            for (_, w) in widths {
                sums.push(vec![0.0f64; w * w]);
            }
        }
        let n = sums.len();
        Self {
            widths,
            sums,
            rows: vec![0; n],
            pending: vec![Vec::new(); n],
        }
    }
    fn slot(&self, layer: usize, site: &str) -> (usize, usize) {
        let s = self
            .widths
            .iter()
            .position(|(n, _)| *n == site)
            .expect("known projection site");
        (layer * 4 + s, self.widths[s].1)
    }
    fn accumulate(&mut self, slot: usize, width: usize, x: &[f32]) {
        let rows = x.len() / width;
        if rows == 0 {
            return;
        }
        // XᵀX = Xt · Xtᵀ with Xt = [width, rows], through the gemm-backed linear.
        let mut xt = vec![0.0f32; width * rows];
        for r in 0..rows {
            for c in 0..width {
                xt[c * rows + r] = x[r * width + c];
            }
        }
        let mut gram = vec![0.0f32; width * width];
        falcon_ocr::kernels::linear(&xt, width, rows, &xt, width, &mut gram);
        for (acc, g) in self.sums[slot].iter_mut().zip(&gram) {
            *acc += *g as f64;
        }
        self.rows[slot] += rows;
    }
    fn flush(&mut self) {
        for slot in 0..self.pending.len() {
            let pending = std::mem::take(&mut self.pending[slot]);
            let width = self.widths[slot % 4].1;
            self.accumulate(slot, width, &pending);
        }
    }
    /// Mean `XᵀX / rows` per matrix as float32 `.npy` files.
    fn save(&mut self, dir: &Path) -> Result<Vec<(String, usize)>> {
        self.flush();
        std::fs::create_dir_all(dir)?;
        let names = [
            "attention.wqkv",
            "attention.wo",
            "feed_forward.w13",
            "feed_forward.w2",
        ];
        let mut written = Vec::new();
        for slot in 0..self.sums.len() {
            let (layer, site) = (slot / 4, slot % 4);
            let width = self.widths[site].1;
            let name = format!("layers.{layer}.{}.weight", names[site]);
            let n = self.rows[slot].max(1) as f64;
            let mut bytes = Vec::with_capacity(128 + 4 * width * width);
            let header = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({width}, {width}), }}"
            );
            let total = 10 + header.len() + 1;
            let padded = total.div_ceil(64) * 64;
            bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
            bytes.extend_from_slice(&((padded - 10) as u16).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(std::iter::repeat_n(b' ', padded - total));
            bytes.push(b'\n');
            for v in &self.sums[slot] {
                bytes.extend_from_slice(&((*v / n) as f32).to_le_bytes());
            }
            std::fs::write(dir.join(format!("{name}.gram.npy")), bytes)?;
            written.push((name, self.rows[slot]));
        }
        Ok(written)
    }
}
impl falcon_ocr::trace::Trace for GramCapture {
    fn enabled(&self) -> bool {
        false
    }
    fn captures_linear_inputs(&self) -> bool {
        true
    }
    fn linear_input(&mut self, layer: usize, site: &'static str, rows: usize, data: &[f32]) {
        let (slot, width) = self.slot(layer, site);
        let x = &data[..rows * width];
        if rows >= 64 {
            self.accumulate(slot, width, x);
        } else {
            self.pending[slot].extend_from_slice(x);
            if self.pending[slot].len() >= Self::FLUSH_ROWS * width {
                let pending = std::mem::take(&mut self.pending[slot]);
                self.accumulate(slot, width, &pending);
            }
        }
    }
    fn tensor(&mut self, _name: &str, _shape: &[usize], _data: &[f32]) -> Result<()> {
        Ok(())
    }
}

/// Collects the teacher-forced greedy choices of one page, and optionally
/// its top-K log-probabilities (dump) or KL against a reference (score).
#[derive(Default)]
struct Agreement {
    steps: usize,
    flips: Vec<(usize, u32, u32)>,
    dump: Option<Vec<Vec<(u32, f32)>>>,
    reference: Option<Vec<Vec<(u32, f32)>>>,
    kl: Vec<f64>,
}
impl falcon_ocr::trace::Trace for Agreement {
    fn enabled(&self) -> bool {
        false
    }
    fn scores_teacher(&self) -> bool {
        true
    }
    fn teacher_step(&mut self, step: usize, forced: u32, predicted: u32) {
        self.steps += 1;
        if forced != predicted {
            self.flips.push((step, forced, predicted));
        }
    }
    fn teacher_logits(&mut self, step: usize, logits: &[f32]) {
        if self.dump.is_none() && self.reference.is_none() {
            return;
        }
        let log_q = log_softmax(logits);
        if let Some(dump) = &mut self.dump {
            let mut order: Vec<u32> = (0..log_q.len() as u32).collect();
            order.select_nth_unstable_by(TOP_K, |&a, &b| {
                log_q[b as usize].total_cmp(&log_q[a as usize])
            });
            let mut top: Vec<(u32, f32)> = order[..TOP_K]
                .iter()
                .map(|&i| (i, log_q[i as usize] as f32))
                .collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            debug_assert_eq!(dump.len(), step);
            dump.push(top);
        }
        if let Some(reference) = &self.reference
            && let Some(top) = reference.get(step)
        {
            self.kl.push(kl_topk(top, &log_q));
        }
    }
    fn tensor(&mut self, _name: &str, _shape: &[usize], _data: &[f32]) -> Result<()> {
        Ok(())
    }
}

/// Page directory name of an input image path (the corpus page ID).
fn page_id(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
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
        Command::CaptureGram { images, output, .. } => {
            ensure!(!output.exists(), "output directory already exists");
            ensure!(
                args.profile == Profile::Reference,
                "capture the Gram with the FP32 reference profile"
            );
            for p in images {
                ensure!(p.is_file(), "missing image {}", p.display());
            }
        }
        Command::Agree {
            images,
            reference,
            report,
            ..
        } => {
            ensure!(!report.exists(), "report already exists");
            ensure!(reference.is_file(), "missing reference report");
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
    let model = Arc::new(match &args.model_file {
        Some(path) => Model::load_packed(path, false)?,
        None => Model::load_attempt_bf16(
            &args.model,
            args.profile,
            args.w8_artifact.as_deref(),
            &args.keep_fp32,
            args.weights_bf16,
        )?,
    });
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
    runner.set_speculation(args.speculate, args.speculate_min_match);
    runner.set_document_drafts(args.document_drafts);
    match args.decode_threads {
        Some(falcon_ocr::runner::DecodeThreads::Auto) => runner.set_decode_threads_auto()?,
        Some(falcon_ocr::runner::DecodeThreads::Fixed(threads)) => runner.set_decode_threads(threads)?,
        None => {}
    }
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
                "stop_repetition":args.stop_repetition,"decode_threads":args.decode_threads.map(|d| d.to_string()),
                "decode_threads_chosen":runner.decode_threads_chosen(),
                "speculate":args.speculate,"speculate_min_match":args.speculate_min_match,
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
        Command::CaptureGram {
            images,
            max_new_tokens,
            max_dimension,
            output,
        } => {
            let c = model.config();
            let mut capture = GramCapture::new(c.n_layers, c.dim, c.query_dim(), c.ffn_dim);
            let options = GenerationOptions {
                max_new_tokens,
                max_dimension,
                min_dimension: 64,
            };
            for image in &images {
                let t = Instant::now();
                let result = runner.recognize_file_with_trace(image, &options, &mut capture)?;
                eprintln!(
                    "{}: {} tokens in {:.1}s",
                    page_id(image),
                    result.output_tokens,
                    t.elapsed().as_secs_f64()
                );
            }
            let written = capture.save(&output)?;
            println!(
                "{}",
                serde_json::json!({"output":output,"pages":images.len(),"matrices":written.len(),
                    "rows_per_matrix":written.first().map(|w| w.1)})
            );
        }
        Command::Agree {
            images,
            reference,
            max_steps,
            max_dimension,
            min_dimension,
            report,
            dump_topk,
            reference_topk,
        } => {
            let reference_top: Option<std::collections::HashMap<String, Vec<Vec<(u32, f32)>>>> =
                match &reference_topk {
                    Some(path) => {
                        let value: serde_json::Value =
                            serde_json::from_slice(&std::fs::read(path)?)?;
                        Some(serde_json::from_value(value["pages"].clone())?)
                    }
                    None => None,
                };
            let mut dumped = serde_json::Map::new();
            let (mut kl_sum, mut kl_steps) = (0.0_f64, 0_usize);
            let reference_report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&reference)?)?;
            let mut forced = std::collections::HashMap::new();
            let inputs = reference_report["inputs"]
                .as_array()
                .context("reference inputs")?;
            let outputs = reference_report["samples"][0]["outputs"]
                .as_array()
                .context("reference outputs")?;
            for (input, output) in inputs.iter().zip(outputs) {
                let path = PathBuf::from(input["path"].as_str().context("reference path")?);
                let ids = output["token_ids"]
                    .as_array()
                    .context("reference token_ids")?
                    .iter()
                    .map(|v| v.as_u64().map(|x| x as u32).context("token id"))
                    .collect::<Result<Vec<u32>>>()?;
                forced.insert(page_id(&path), ids);
            }
            let started = Instant::now();
            let (mut total_steps, mut total_flips) = (0usize, 0usize);
            let mut pages = Vec::with_capacity(images.len());
            for image in &images {
                let id = page_id(image);
                let ids = forced
                    .get(&id)
                    .with_context(|| format!("page {id} missing from the reference report"))?;
                let teacher = &ids[..ids.len().min(max_steps)];
                let options = GenerationOptions {
                    max_new_tokens: teacher.len(),
                    max_dimension,
                    min_dimension,
                };
                let mut agreement = Agreement {
                    dump: dump_topk.as_ref().map(|_| Vec::new()),
                    reference: match &reference_top {
                        Some(pages) => Some(
                            pages
                                .get(&id)
                                .with_context(|| {
                                    format!("page {id} missing from the top-K reference")
                                })?
                                .clone(),
                        ),
                        None => None,
                    },
                    ..Agreement::default()
                };
                let t = Instant::now();
                runner.score_teacher_file(image, teacher, &options, &mut agreement)?;
                if reference_top.is_some() {
                    ensure!(
                        agreement.kl.len() == agreement.steps,
                        "top-K reference for {id} has fewer steps than this run"
                    );
                }
                let page_kl: f64 = agreement.kl.iter().sum();
                kl_sum += page_kl;
                kl_steps += agreement.kl.len();
                if let Some(dump) = agreement.dump.take() {
                    dumped.insert(id.clone(), serde_json::to_value(dump)?);
                }
                let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
                total_steps += agreement.steps;
                total_flips += agreement.flips.len();
                eprintln!(
                    "{id}: {} flips / {} steps, mean KL {:.3e} in {:.1}s",
                    agreement.flips.len(),
                    agreement.steps,
                    page_kl / agreement.kl.len().max(1) as f64,
                    wall_ms / 1000.0
                );
                pages.push(serde_json::json!({"page":id,"path":image,"steps":agreement.steps,
                    "flips":agreement.flips.len(),
                    "kl_mean":(!agreement.kl.is_empty()).then(|| page_kl / agreement.kl.len() as f64),
                    "kl_max":agreement.kl.iter().copied().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))),
                    "first_flip":agreement.flips.first().map(|f| f.0),
                    "flip_steps":agreement.flips.iter().map(|f| serde_json::json!([f.0,f.1,f.2])).collect::<Vec<_>>(),
                    "wall_ms":wall_ms}));
            }
            let per_thousand = 1000.0 * total_flips as f64 / total_steps.max(1) as f64;
            let kl_mean = (kl_steps > 0).then(|| kl_sum / kl_steps as f64);
            if let Some(path) = &dump_topk {
                write_new(
                    path,
                    &serde_json::json!({"k":TOP_K,"profile":args.profile,
                    "max_steps":max_steps,"pages":dumped}),
                )
                .with_context(|| format!("write {}", path.display()))?;
            }
            let report_value = serde_json::json!({"schema":"falcon-ocr-attempt3-agree-v1","profile":args.profile,
                "reference":reference,"w8_artifact":args.w8_artifact,"max_steps":max_steps,
                "exp_mode":format!("{:?}", falcon_ocr::kernels::exp_mode()),
                "binary_sha256":digest(&std::env::current_exe()?)?,"threads":args.threads,"backend":args.backend,
                "steps":total_steps,"flips":total_flips,"flips_per_1000":per_thousand,
                "kl_mean":kl_mean,"reference_topk":reference_topk,"keep_fp32":args.keep_fp32,
                "weights_bf16":args.weights_bf16,
                "wall_ms":started.elapsed().as_secs_f64() * 1000.0,"pages":pages});
            write_new(&report, &report_value)
                .with_context(|| format!("write {}", report.display()))?;
            println!(
                "{}",
                serde_json::json!({"report":report,"profile":args.profile,"steps":total_steps,
                    "flips":total_flips,"flips_per_1000":per_thousand,"kl_mean":kl_mean})
            );
        }
        Command::Doctor => unreachable!(),
    }
    Ok(())
}
