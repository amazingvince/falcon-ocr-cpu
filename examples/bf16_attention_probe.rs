//! Same-input BF16 attention export for the immutable GPU local contract.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use falcon_ocr::{
    bf16_attention::{self, Parameters},
    bf16_kernels::Backend,
    trace::{TensorTrace, Trace},
};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(
        long,
        default_value = "artifacts/reference/attention-operators-bf16.safetensors"
    )]
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Diagnostic intermediates for selected failing and control prefill heads.
    #[arg(long)]
    stages: Option<PathBuf>,
    #[arg(long, default_value = "auto")]
    backend: String,
    #[arg(long, default_value_t = 2)]
    threads: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    let backend = match args.backend.as_str() {
        "auto" => Backend::Auto,
        "scalar" => Backend::Scalar,
        "avx512bf16" => Backend::Avx512Bf16,
        _ => anyhow::bail!("unsupported backend"),
    };
    backend.validate().map_err(anyhow::Error::msg)?;
    let bytes = std::fs::read(&args.fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(args.fixture.with_extension("json"))?)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let read = |name: &str| -> Result<Vec<bf16>> {
        let tensor = tensors.tensor(name)?;
        ensure!(tensor.dtype() == Dtype::BF16, "{name}: expected BF16");
        Ok(tensor
            .data()
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
            .collect())
    };
    let mut trace = TensorTrace::default();
    let mut stages = TensorTrace::default();
    let mut cases = Vec::new();
    for case in metadata["cases"].as_array().context("cases")? {
        let name = case.as_str().context("case name")?;
        let qview = tensors.tensor(&format!("{name}.q"))?;
        let kview = tensors.tensor(&format!("{name}.k"))?;
        ensure!(
            qview.shape().len() == 3 && qview.shape()[2] == 64,
            "expected Q [S,H,64]"
        );
        let (rows, heads, keys) = (qview.shape()[0], qview.shape()[1], kview.shape()[0]);
        let params = tensors.tensor(&format!("{name}.params"))?;
        ensure!(
            params.dtype() == Dtype::I64 && params.shape() == [3],
            "expected offset/image start/end"
        );
        let params: Vec<_> = params
            .data()
            .chunks_exact(8)
            .map(|v| usize::try_from(i64::from_le_bytes(v.try_into().unwrap())))
            .collect::<std::result::Result<_, _>>()?;
        let p = Parameters {
            query_len: rows,
            kv_len: keys,
            heads,
            query_offset: params[0],
            image_start: params[1],
            image_end: params[2],
            capacity: 256,
        };
        // These six frozen toy-fixture cases were classified at capacity256.
        ensure!(keys <= 256, "fixture exceeds frozen block-mask capacity");
        let q = read(&format!("{name}.q"))?;
        let k = read(&format!("{name}.k"))?;
        let v = read(&format!("{name}.v"))?;
        let sinks = read(&format!("{name}.sinks"))?;
        if args.stages.is_some() {
            let selected: &[(usize, usize)] = match name {
                "prefill.layer.0" => &[(0, 0), (13, 1), (13, 2), (53, 8)],
                "prefill.layer.17" => &[(54, 10), (124, 11)],
                "prefill.layer.19" => &[(41, 6)],
                _ => &[],
            };
            for &(row, head) in selected {
                bf16_attention::diagnostics::trace_prefill_head(
                    &q,
                    &k,
                    &v,
                    p,
                    row,
                    head,
                    backend,
                    &mut stages,
                    &format!("{name}.row{row}.head{head}"),
                )?;
            }
        }
        let mut out = vec![bf16::ZERO; q.len()];
        let mut raw = out.clone();
        let mut lse = vec![0.; rows * heads];
        pool.install(|| {
            bf16_attention::attention(&q, &k, &v, &sinks, p, &mut out, &mut raw, &mut lse, backend)
        });
        for (label, values) in [("raw", &raw), ("scaled", &out)] {
            trace.tensor(
                &format!("{name}.{label}"),
                &[rows, heads, 64],
                &values.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            )?;
        }
        let mut head_major = vec![0.; lse.len()];
        for row in 0..rows {
            for head in 0..heads {
                head_major[head * rows + row] = lse[row * heads + head];
            }
        }
        trace.tensor(&format!("{name}.lse"), &[heads, rows], &head_major)?;
        cases.push(name.to_owned());
        eprintln!("Exported {name} [{rows},{heads},64] x {keys}");
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    trace.save(&args.output)?;
    if let Some(path) = args.stages {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        stages.save(path)?;
    }
    std::fs::write(
        args.output.with_extension("json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "contract_sha256":format!("{:x}",Sha256::digest(std::fs::read("reference/bf16-local-contract-v1.json")?)),
            "export_sha256":format!("{:x}",Sha256::digest(std::fs::read(&args.output)?)),
            "kernel_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/bf16_attention.rs"))),
            "matrix_kernel_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/bf16_kernels.rs"))),
            "probe_sha256":format!("{:x}",Sha256::digest(include_bytes!("bf16_attention_probe.rs"))),
            "cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.lock"))),
            "os":std::env::consts::OS,"arch":std::env::consts::ARCH,"debug_assertions":cfg!(debug_assertions),
            "backend":format!("{:?}",backend.resolved()),"threads":args.threads,"cases":cases,
            "interpretation":"Same-input BF16 blockwise CPU attention candidate. Raw/scaled outputs are exact BF16 values stored as F32, LSE is head-major F32. Frozen contract assessment is separate; no graph parity claim."
        }))?,
    )?;
    Ok(())
}
