//! Resumable free-running CPU evaluation using a frozen canonical-image manifest.
#[path = "support/corpus_record.rs"]
mod corpus_record;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use falcon_ocr::{
    Backend, Bf16Model, Bf16Runner, GenerationOptions, Model, Precision, Runner, RunnerConfig,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "artifacts/model")]
    model: PathBuf,
    #[arg(long, default_value = "reference/corpus-smoke-lock-v1.json")]
    manifest: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[arg(long, value_enum, default_value = "auto")]
    backend: Backend,
    #[arg(long, value_enum, default_value = "fp32")]
    precision: Precision,
    #[arg(long, default_value_t = 1536)]
    max_dimension: u32,
    #[arg(long, default_value_t = 4096)]
    max_new_tokens: usize,
    #[arg(long)]
    resume: bool,
    /// Restrict to the first N manifest entries without changing their order.
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long)]
    cpu_label: String,
    #[arg(long)]
    environment_label: String,
    /// Optional checked source/tool snapshot produced by capture_rust_build.py.
    #[arg(long)]
    build_manifest: Option<PathBuf>,
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
enum Runtime {
    Fp32(Runner),
    Bf16(Bf16Runner),
}
impl Runtime {
    fn recognize_file(&self, path: &std::path::Path, options: &GenerationOptions) -> Result<Value> {
        match self {
            Self::Fp32(runner) => Ok(serde_json::to_value(runner.recognize_file(path, options)?)?),
            Self::Bf16(runner) => Ok(serde_json::to_value(runner.recognize_file(path, options)?)?),
        }
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    let precision_name = match args.precision {
        Precision::Fp32 => "fp32",
        Precision::Bf16 => "bf16",
    };
    let output_root = args.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "artifacts/cpu/corpus-smoke-{precision_name}-{}",
            args.max_new_tokens
        ))
    });
    let bytes = std::fs::read(&args.manifest)?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    let pages = manifest["pages"].as_array().context("missing pages")?;
    ensure!(!pages.is_empty(), "manifest has no pages");
    ensure!(args.limit != Some(0), "limit must be positive");
    let options = GenerationOptions {
        max_new_tokens: args.max_new_tokens,
        max_dimension: args.max_dimension,
        ..Default::default()
    };
    options.validate()?;
    let source = json!({"model":hash(include_bytes!("../src/model.rs")),"kernels":hash(include_bytes!("../src/kernels.rs")),
        "runner":hash(include_bytes!("../src/runner.rs")),"preprocess":hash(include_bytes!("../src/preprocess.rs")),
        "tokenizer":hash(include_bytes!("../src/tokenizer.rs")),"config":hash(include_bytes!("../src/config.rs")),
        "harness":hash(include_bytes!("corpus_eval.rs")),"lock":hash(include_bytes!("../Cargo.lock")),
        "corpus_record":hash(include_bytes!("support/corpus_record.rs")),
        "cargo_manifest":hash(include_bytes!("../Cargo.toml")),"toolchain":hash(include_bytes!("../rust-toolchain.toml")),
        "lib":hash(include_bytes!("../src/lib.rs")),"trace":hash(include_bytes!("../src/trace.rs")),
        "packed_kernels":hash(include_bytes!("../src/packed_kernels.rs")),
        "bf16_model":hash(include_bytes!("../src/bf16_model.rs")),"bf16_runner":hash(include_bytes!("../src/bf16_runner.rs")),
        "bf16_kernels":hash(include_bytes!("../src/bf16_kernels.rs")),"bf16_ops":hash(include_bytes!("../src/bf16_ops.rs")),
        "bf16_attention":hash(include_bytes!("../src/bf16_attention.rs"))});
    let binary_sha256 = hash(&std::fs::read(std::env::current_exe()?)?);
    let build_manifest_sha256 = if let Some(path) = &args.build_manifest {
        let data = std::fs::read(path)?;
        let build: Value = serde_json::from_slice(&data)?;
        ensure!(
            build["status"] == "complete",
            "build snapshot did not complete"
        );
        ensure!(
            build["source_unchanged_during_build"] == true,
            "build sources changed during compilation"
        );
        ensure!(
            build["binary_sha256"] == binary_sha256,
            "build snapshot belongs to a different executable"
        );
        for (key, expected) in source.as_object().unwrap() {
            let path = match key.as_str() {
                "harness" => "examples/corpus_eval.rs".to_string(),
                "corpus_record" => "examples/support/corpus_record.rs".to_string(),
                "lock" => "Cargo.lock".to_string(),
                "cargo_manifest" => "Cargo.toml".to_string(),
                "toolchain" => "rust-toolchain.toml".to_string(),
                name => format!("src/{name}.rs"),
            };
            ensure!(
                build["source_sha256"][&path] == *expected,
                "build snapshot source mismatch: {path}"
            );
        }
        let archive = path
            .parent()
            .context("build manifest has no directory")?
            .join(
                build["source_archive"]
                    .as_str()
                    .context("missing source archive")?,
            );
        ensure!(
            build["source_archive_sha256"] == hash(&std::fs::read(archive)?),
            "build source archive changed"
        );
        Some(hash(&data))
    } else {
        None
    };
    let contract = json!({"manifest_sha256":hash(&bytes),"options":options,"threads":args.threads,"backend":args.backend,
        "precision":args.precision,"source_sha256":source,"cpu_label":args.cpu_label,"environment_label":args.environment_label,
        "model_revision":falcon_ocr::config::MODEL_REVISION,"weights_sha256":falcon_ocr::config::WEIGHTS_SHA256,
        "config_sha256":falcon_ocr::config::CONFIG_SHA256,"binary_sha256":binary_sha256,
        "build_manifest_sha256":build_manifest_sha256,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "prompt":falcon_ocr::tokenizer::PLAIN_PROMPT,"greedy_policy":"first token index attaining the maximum finite logit"});
    let contract_hash = hash(&serde_json::to_vec(&contract)?);
    std::fs::create_dir_all(&output_root)?;
    let run_path = output_root.join("run.json");
    if run_path.exists() {
        ensure!(
            args.resume,
            "output already contains a run; use --resume or another output directory"
        );
        let previous: Value = serde_json::from_slice(&std::fs::read(&run_path)?)?;
        ensure!(
            previous["contract_sha256"] == contract_hash,
            "resume contract changed: use another output directory"
        );
    }
    let start = Instant::now();
    let runner_config = RunnerConfig {
        threads: args.threads,
        backend: args.backend,
        ..Default::default()
    };
    let runner = match args.precision {
        Precision::Fp32 => Runtime::Fp32(Runner::new(
            Arc::new(Model::load(&args.model)?),
            &args.model,
            runner_config,
        )?),
        Precision::Bf16 => {
            Bf16Runner::validate_config(&runner_config)?;
            Runtime::Bf16(Bf16Runner::new(
                Arc::new(Bf16Model::load(&args.model)?),
                &args.model,
                runner_config,
            )?)
        }
    };
    if !run_path.exists() {
        std::fs::write(
            &run_path,
            serde_json::to_vec_pretty(
                &json!({"contract":contract,"contract_sha256":contract_hash,
        "os":std::env::consts::OS,"arch":std::env::consts::ARCH,"verified_load_and_runner_ms":start.elapsed().as_secs_f64()*1000.,
        "planned_pages":pages.len(),"teacher_forced":false,
        "build_provenance_scope":if args.build_manifest.is_some() {"executing binary, compiled source hashes, archived build manifest"} else {"executing binary and compiled source hashes; compiler/build archive absent"},
        "qualification":"free_running_corpus_outputs_pending_independent_parity_and_quality_assessment"}),
            )?,
        )?;
    }
    let invocation_directory = output_root.join("invocations");
    std::fs::create_dir_all(&invocation_directory)?;
    let invocation_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let invocation_path = invocation_directory.join(format!("{invocation_id}.json"));
    let mut invocation = json!({"contract_sha256":contract_hash,"resume":args.resume,"limit":args.limit,
        "selected_pages":pages.len().min(args.limit.unwrap_or(usize::MAX)),"planned_pages":pages.len(),
        "verified_load_and_runner_ms":start.elapsed().as_secs_f64()*1000.,"status":"running"});
    std::fs::write(&invocation_path, serde_json::to_vec_pretty(&invocation)?)?;
    let mut errors = 0;
    let mut completed = 0;
    let mut resumed = 0;
    for (index, page) in pages
        .iter()
        .take(args.limit.unwrap_or(usize::MAX))
        .enumerate()
    {
        let path = PathBuf::from(
            page["canonical_path"]
                .as_str()
                .context("missing canonical path")?,
        );
        let key = path
            .parent()
            .and_then(|p| p.file_name())
            .context("missing page key")?;
        let output = output_root.join(key).with_extension("json");
        if output.exists() && args.resume {
            let previous: Value = serde_json::from_slice(&std::fs::read(&output)?)?;
            ensure!(
                previous["contract_sha256"] == contract_hash,
                "page resume contract changed"
            );
            ensure!(
                previous["id"] == page["id"]
                    && previous["input_sha256"] == page["canonical_png_sha256"]
                    && previous["ground_truth_sha256"] == page["ground_truth_sha256"],
                "page resume identity changed"
            );
            if previous["result"].is_object() && previous.get("error").is_none() {
                corpus_record::validate_result(&previous["result"], &options, precision_name)
                    .context("invalid completed page on resume")?;
                resumed += 1;
                continue;
            }
        }
        let image_bytes = std::fs::read(&path)?;
        ensure!(
            page["canonical_png_sha256"] == hash(&image_bytes),
            "image checksum changed: {}",
            path.display()
        );
        let started = Instant::now();
        let mut record = json!({"id":page["id"],"category":page["category"],"input_sha256":page["canonical_png_sha256"],
            "ground_truth_sha256":page["ground_truth_sha256"],"contract_sha256":contract_hash,"teacher_forced":false});
        match runner.recognize_file(&path, &options) {
            Ok(result) => {
                corpus_record::validate_result(&result, &options, precision_name)?;
                completed += 1;
                record["result"] = result;
            }
            Err(error) => {
                errors += 1;
                record["error"] = format!("{error:#}").into();
            }
        }
        record["wall_ms"] = (started.elapsed().as_secs_f64() * 1000.).into();
        let temporary = output.with_extension("partial");
        std::fs::write(&temporary, serde_json::to_vec_pretty(&record)?)?;
        // Each record is checkpointed; a resume skips only successful matching contracts.
        if output.exists() {
            std::fs::remove_file(&output)?;
        }
        std::fs::rename(temporary, &output)?;
        eprintln!(
            "{}/{} {}: {}",
            index + 1,
            pages.len(),
            key.to_string_lossy(),
            if record.get("error").is_some() {
                "error".to_string()
            } else {
                format!(
                    "{} tokens in {:.1}s",
                    record["result"]["output_tokens"],
                    started.elapsed().as_secs_f64()
                )
            }
        );
    }
    invocation["status"] = if errors == 0 {
        "completed_selected_pages"
    } else {
        "failed"
    }
    .into();
    invocation["new_completed_pages"] = completed.into();
    invocation["resumed_completed_pages"] = resumed.into();
    invocation["failed_pages"] = errors.into();
    std::fs::write(&invocation_path, serde_json::to_vec_pretty(&invocation)?)?;
    ensure!(
        errors == 0,
        "{errors} pages failed; inspect per-page records"
    );
    Ok(())
}
