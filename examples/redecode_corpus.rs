//! Replay only text decoding from preserved inference token IDs. No model forward
//! runs here; new records bind both the original inference and the Rust decoder.
#[path = "support/corpus_record.rs"]
mod corpus_record;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use falcon_ocr::tokenizer::OcrTokenizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "artifacts/model")]
    model: PathBuf,
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    resume: bool,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.source != args.output, "source and output must differ");
    let source_run_path = args.source.join("run.json");
    let source_run_bytes = std::fs::read(&source_run_path)?;
    let source_run: Value = serde_json::from_slice(&source_run_bytes)?;
    let manifest_bytes = std::fs::read(&args.manifest)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let source_contract = &source_run["contract"];
    let options: falcon_ocr::GenerationOptions =
        serde_json::from_value(source_contract["options"].clone())?;
    options.validate()?;
    let precision = source_contract["precision"]
        .as_str()
        .context("source precision missing")?;
    ensure!(
        source_contract.is_object(),
        "source run has no inference contract"
    );
    ensure!(
        source_contract.get("postprocessing_replay").is_none(),
        "replay must start from original inference, not another replay"
    );
    ensure!(
        source_contract["manifest_sha256"] == hash(&manifest_bytes),
        "source manifest differs"
    );
    ensure!(
        source_run["contract_sha256"] == hash(&serde_json::to_vec(source_contract)?),
        "source contract hash differs"
    );
    ensure!(
        source_run["teacher_forced"] == false,
        "source must be free-running"
    );
    let tokenizer = OcrTokenizer::load(&args.model)?;
    let decoder_binary = std::env::current_exe()?;
    let decoder_source_path = args.output.join("decoder-source.rs");
    let tokenizer_asset_sha256: serde_json::Map<String, Value> =
        ["tokenizer.json", "tokenizer_config.json"]
            .into_iter()
            .map(|name| {
                let path = args.model.join(name);
                Ok((
                    path.to_string_lossy().into_owned(),
                    Value::String(hash(&std::fs::read(path)?)),
                ))
            })
            .collect::<Result<_>>()?;
    let replay = json!({"schema_version":1,"source_run_path":source_run_path,
        "source_run_sha256":hash(&source_run_bytes),"source_contract_sha256":source_run["contract_sha256"],
        "decoder_binary_path":decoder_binary,"decoder_binary_sha256":hash(&std::fs::read(&decoder_binary)?),
        "decoder_source_path":decoder_source_path,"decoder_source_sha256":hash(include_bytes!("../src/tokenizer.rs")),
        "harness_source_sha256":hash(include_bytes!("redecode_corpus.rs")),
        "record_validator_sha256":hash(include_bytes!("support/corpus_record.rs")),
        "cargo_lock_sha256":hash(include_bytes!("../Cargo.lock")),
        "tokenizer_asset_sha256":tokenizer_asset_sha256,
        "policy":"Pinned Transformers 5.14.1 BPE decode without WordPiece cleanup; preserve special tokens, remove two EOS spellings, Python 3.12 outer strip",
        "scope":"Text-only Rust replay from original saved IDs; source_sha256 and timings describe original inference, not a new forward run"});
    let mut contract = source_contract.clone();
    contract["postprocessing_replay"] = replay;
    let contract_hash = hash(&serde_json::to_vec(&contract)?);
    if args.output.exists() {
        ensure!(
            args.resume,
            "output exists; use --resume or a new directory"
        );
        ensure!(
            args.output.canonicalize()? != args.source.canonicalize()?,
            "output resolves to source"
        );
        let prior: Value = serde_json::from_slice(&std::fs::read(args.output.join("run.json"))?)?;
        ensure!(
            prior["contract_sha256"] == contract_hash,
            "replay resume contract changed"
        );
        ensure!(
            std::fs::read(&decoder_source_path)? == include_bytes!("../src/tokenizer.rs"),
            "preserved decoder source changed"
        );
    } else {
        std::fs::create_dir_all(&args.output)?;
        std::fs::write(&decoder_source_path, include_bytes!("../src/tokenizer.rs"))?;
        let mut run = source_run.clone();
        run["contract"] = contract;
        run["contract_sha256"] = contract_hash.clone().into();
        run["derived_text_replay"] = true.into();
        run["qualification"] = "Original free-running inference with separately bound corrected Rust text decoding; not a fresh model run".into();
        std::fs::write(
            args.output.join("run.json"),
            serde_json::to_vec_pretty(&run)?,
        )?;
    }
    let mut completed = 0;
    let mut changed = 0;
    let mut missing = Vec::new();
    let mut errors = Vec::new();
    for page in manifest["pages"]
        .as_array()
        .context("missing manifest pages")?
    {
        let image = PathBuf::from(
            page["canonical_path"]
                .as_str()
                .context("missing image path")?,
        );
        let key = image
            .parent()
            .and_then(|p| p.file_name())
            .context("missing page key")?;
        let source_path = args.source.join(key).with_extension("json");
        if !source_path.exists() {
            missing.push(key.to_string_lossy().into_owned());
            continue;
        }
        let bytes = std::fs::read(&source_path)?;
        let original: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            original["contract_sha256"] == source_run["contract_sha256"],
            "original page contract differs"
        );
        ensure!(
            original["id"] == page["id"]
                && original["input_sha256"] == page["canonical_png_sha256"]
                && original["ground_truth_sha256"] == page["ground_truth_sha256"],
            "original page identity differs"
        );
        if original.get("error").is_some() || !original["result"].is_object() {
            errors.push(key.to_string_lossy().into_owned());
            continue;
        }
        let mut record = original.clone();
        corpus_record::validate_result(&original["result"], &options, precision)?;
        let ids: Vec<u32> = serde_json::from_value(original["result"]["token_ids"].clone())?;
        let text = tokenizer.decode(&ids)?;
        changed += usize::from(original["result"]["text"] != text);
        record["result"]["text"] = text.into();
        record["contract_sha256"] = contract_hash.clone().into();
        record["postprocessing_replay"] = json!({"source_record_path":source_path,
            "source_record_sha256":hash(&bytes),"source_contract_sha256":source_run["contract_sha256"],
            "original_text":original["result"]["text"],
            "original_text_sha256":hash(original["result"]["text"].as_str().context("source text missing")?.as_bytes()),
            "inference_reexecuted":false});
        let destination = args.output.join(key).with_extension("json");
        if destination.exists() {
            ensure!(args.resume, "page already exists");
            let prior: Value = serde_json::from_slice(&std::fs::read(destination)?)?;
            ensure!(
                prior == record,
                "source or decoder changed for existing replay record"
            );
        } else {
            let temporary = destination.with_extension("partial");
            std::fs::write(&temporary, serde_json::to_vec_pretty(&record)?)?;
            std::fs::rename(temporary, destination)?;
        }
        completed += 1;
    }
    let report = json!({"contract_sha256":contract_hash,"replayed_pages":completed,"changed_text_pages":changed,
        "missing_source_pages":missing,"failed_source_pages":errors,"derived_text_replay":true,"inference_reexecuted":false});
    // A progress snapshot may be refreshed; run identity and existing pages are immutable.
    std::fs::write(
        args.output.join("replay-status.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        errors.is_empty(),
        "source inference contains failed records"
    );
    Ok(())
}
