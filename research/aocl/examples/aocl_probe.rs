//! Isolated AOCL-DLP FP32 GEMM accuracy probe. No production backend changes,
//! timing measurements, or changes to the frozen model tolerance policy.

#[path = "support/aocl_dynamic.rs"]
mod aocl_dynamic;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use falcon_ocr::config::{CONFIG_SHA256, WEIGHTS_SHA256};
use falcon_ocr::kernels::{self, Simd};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

const AOCL_REVISION: &str = "abb63d85ed7a6d559ea42b5db648e2585ac9ecb8";
const FIXTURE_SHA256: &str = "ce7345c219d8923182ff66e9aad2f4d6bad3c193a19c3445a850c2d3a90e5417";
const MANIFEST_SHA256: &str = "148b9d4286bf413b7c9c693b920290298d5f46cd137449ac4e394430495a8962";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    library: PathBuf,
    #[arg(long)]
    expected_library_sha256: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "artifacts/model")]
    model: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/reference/layer-operators-fp32.safetensors"
    )]
    fixture: PathBuf,
    #[arg(long, default_value = "reference/layer-operators-fp32-manifest.json")]
    manifest: PathBuf,
    #[arg(long, default_value = "artifacts/aocl-dlp")]
    aocl_source: PathBuf,
    /// Command-local line-ending interpretation for a shared Windows checkout.
    #[arg(long, value_parser = ["true", "false", "input"])]
    git_autocrlf: Option<String>,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_digest(path: impl AsRef<Path>) -> Result<String> {
    Ok(digest(&std::fs::read(path)?))
}

fn git(source: &Path, args: &[&str], autocrlf: Option<&str>) -> Result<String> {
    let mut command = Command::new("git");
    if let Some(value) = autocrlf {
        command.args(["-c", &format!("core.autocrlf={value}")]);
    }
    let output = command.arg("-C").arg(source).args(args).output()?;
    ensure!(
        output.status.success(),
        "git query failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn float_digest(values: &[f32]) -> String {
    #[cfg(target_endian = "little")]
    {
        digest(bytemuck::cast_slice(values))
    }
    #[cfg(target_endian = "big")]
    {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
}

fn matrix(tensors: &SafeTensors<'_>, name: &str, rows: usize, columns: usize) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(tensor.dtype() == Dtype::F32, "{name}: expected FP32");
    ensure!(
        tensor.shape() == [rows, columns],
        "{name}: expected [{rows},{columns}], found {:?}",
        tensor.shape()
    );
    let count = rows.checked_mul(columns).context("matrix shape overflow")?;
    ensure!(
        tensor.data().len() == count.checked_mul(4).context("matrix bytes overflow")?,
        "{name}: invalid byte length"
    );
    let values: Vec<_> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "{name}: nonfinite fixture input"
    );
    Ok(values)
}

fn difference(actual: &[f32], expected: &[f32]) -> Result<Value> {
    ensure!(
        !actual.is_empty() && actual.len() == expected.len(),
        "difference shape mismatch"
    );
    let (mut maximum, mut squared, mut peak) = (0.0f64, 0.0f64, 0.0f64);
    let (mut worst, mut different_bits, mut different_values) = (0usize, 0usize, 0usize);
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite operator result at {index}"
        );
        let error = (f64::from(a) - f64::from(b)).abs();
        if error > maximum {
            maximum = error;
            worst = index;
        }
        squared += error * error;
        peak = peak.max(f64::from(b).abs());
        different_bits += usize::from(a.to_bits() != b.to_bits());
        different_values += usize::from(a != b);
    }
    Ok(
        json!({"elements":actual.len(), "max_abs":maximum, "rms_abs":(squared / actual.len() as f64).sqrt(),
        "different_bit_patterns":different_bits, "different_values":different_values,
        "reference_peak_abs":peak, "worst_index":worst, "worst_actual":actual[worst], "worst_expected":expected[worst]}),
    )
}

fn known_answers(library: &aocl_dynamic::Aocl) -> Result<Value> {
    // A[2,3] and B[4,3], deliberately rectangular and nonsymmetric to detect
    // swapped strides, wrong transposition, and the row/column-major convention.
    let input = [1., 2., 3., -1., 0., 2.];
    let weight = [4., 5., 6., -2., 1., 3., 0., 7., -1., 2., -3., 1.];
    let product = [32., 9., 11., -1., 8., 8., -2., 0.];
    let mut output = [f32::NAN; 8];
    library.linear(&input, 2, 3, &weight, 4, &mut output, 1., 0.)?;
    ensure!(
        output == product,
        "AOCL row-major/transposed-weight known answer failed: {output:?}"
    );
    // beta=0 must ignore the NaN contents of C above. alpha=0 with beta!=0
    // exercises the scale-only path using otherwise valid matrix pointers.
    let initial = [1., 2., 3., 4., 5., 6., 7., 8.];
    output = initial;
    library.linear(&input, 2, 3, &weight, 4, &mut output, 0., 2.)?;
    ensure!(
        output == initial.map(|x| 2. * x),
        "AOCL alpha=0/beta=2 known answer failed"
    );
    output = initial;
    library.linear(&input, 2, 3, &weight, 4, &mut output, 0.5, -1.)?;
    let expected: Vec<_> = product
        .iter()
        .zip(initial)
        .map(|(&value, old)| 0.5 * value - old)
        .collect();
    ensure!(
        output.as_slice() == expected,
        "AOCL alpha/beta accumulation known answer failed"
    );
    ensure!(
        library
            .linear(&input[..5], 2, 3, &weight, 4, &mut output, 1., 0.)
            .is_err(),
        "invalid A shape was accepted"
    );
    ensure!(
        library
            .linear(&input, 2, 3, &weight[..11], 4, &mut output, 1., 0.)
            .is_err(),
        "invalid B shape was accepted"
    );
    ensure!(
        library
            .linear(&input, 2, 3, &weight, 4, &mut output[..7], 1., 0.)
            .is_err(),
        "invalid C shape was accepted"
    );
    ensure!(
        library
            .linear(&[], usize::MAX, 2, &[], 4, &mut [], 1., 0.)
            .is_err(),
        "overflowing shape was accepted"
    );
    Ok(
        json!({"rectangular_orientation":true, "beta_zero_ignores_nan_c":true,
        "alpha_zero_beta_two":true, "nonunit_alpha_negative_beta":true,
        "input_weight_output_shape_errors_rejected":true, "overflow_rejected":true,
        "shape":{"m":2,"n":4,"k":3}, "alpha_one_beta_zero_expected":product}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "output already exists; choose a new diagnostic report"
    );
    Simd::Avx2.validate().map_err(anyhow::Error::msg)?;
    let source_revision = git(
        &args.aocl_source,
        &["rev-parse", "HEAD"],
        args.git_autocrlf.as_deref(),
    )?;
    ensure!(
        source_revision == AOCL_REVISION,
        "AOCL source revision differs from pinned ABI"
    );
    let source_changes = git(
        &args.aocl_source,
        &["status", "--porcelain", "--untracked-files=no"],
        args.git_autocrlf.as_deref(),
    )?;
    ensure!(
        source_changes.is_empty(),
        "AOCL tracked source modifications need explicit qualification"
    );
    let header = args
        .aocl_source
        .join("include/classic/aocl_gemm_interface_apis.h");
    let types = args.aocl_source.join("include/classic/dlp_base_types.h");
    let library = aocl_dynamic::Aocl::load(&args.library, &args.expected_library_sha256)?;
    let answers = known_answers(&library)?;
    let fixture_bytes = std::fs::read(&args.fixture)?;
    ensure!(
        digest(&fixture_bytes) == FIXTURE_SHA256,
        "GPU operator fixture hash differs"
    );
    let manifest_bytes = std::fs::read(&args.manifest)?;
    ensure!(
        digest(&manifest_bytes) == MANIFEST_SHA256,
        "GPU operator manifest hash differs"
    );
    let metadata: Value = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        metadata["output_sha256"].as_str() == Some(FIXTURE_SHA256),
        "manifest does not bind fixture"
    );
    let fixture = SafeTensors::deserialize(&fixture_bytes)?;
    let config_hash = file_digest(args.model.join("config.json"))?;
    ensure!(config_hash == CONFIG_SHA256, "model configuration differs");
    let model_file = File::open(args.model.join("model.safetensors"))?;
    // SAFETY: This read-only probe owns the file and mapping for every tensor
    // access. The pinned model file must not be mutated during the probe.
    let model_map = unsafe { MmapOptions::new().map(&model_file)? };
    let weights_hash = digest(&model_map);
    ensure!(
        weights_hash == WEIGHTS_SHA256,
        "full checkpoint hash differs"
    );
    let weights = SafeTensors::deserialize(&model_map)?;
    let cases = metadata["cases"]
        .as_array()
        .context("fixture cases missing")?;
    ensure!(cases.len() == 7, "expected all seven independent GPU cases");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let mut results = Vec::new();
    for case in cases {
        let name = case["name"].as_str().context("case name")?;
        let layer = case["layer"].as_u64().context("case layer")?;
        let rows = usize::try_from(case["rows"].as_u64().context("case rows")?)?;
        let prefix = format!("layers.{layer}.");
        ensure!(
            case["weight_prefix"].as_str() == Some(&prefix),
            "case weight prefix differs"
        );
        for (stage, source, expected, weight_name, width, channels) in [
            (
                "qkv",
                "attention_norm.expected",
                "qkv.expected",
                "attention.wqkv.weight",
                768,
                2048,
            ),
            (
                "wo",
                "attention.scaled",
                "attention_projection.expected",
                "attention.wo.weight",
                1024,
                768,
            ),
            (
                "w13",
                "ffn_norm.expected",
                "w13.expected",
                "feed_forward.w13.weight",
                768,
                4608,
            ),
            (
                "w2",
                "gate.expected",
                "w2.expected",
                "feed_forward.w2.weight",
                2304,
                768,
            ),
        ] {
            let input_key = format!("{name}.{source}");
            let expected_key = format!("{name}.{expected}");
            let weight_key = format!("{prefix}{weight_name}");
            let input = matrix(&fixture, &input_key, rows, width)?;
            let expected = matrix(&fixture, &expected_key, rows, channels)?;
            let weight = matrix(&weights, &weight_key, channels, width)?;
            let count = rows
                .checked_mul(channels)
                .context("output shape overflow")?;
            let mut aocl = vec![f32::NAN; count];
            let mut rust = vec![f32::NAN; count];
            library.linear(&input, rows, width, &weight, channels, &mut aocl, 1., 0.)?;
            pool.install(|| {
                kernels::linear_with_simd(
                    &input,
                    rows,
                    width,
                    &weight,
                    channels,
                    &mut rust,
                    Simd::Avx2,
                )
            });
            results.push(json!({"case":name,"layer":layer,"operator":stage,
                "shape":{"m":rows,"n":channels,"k":width},
                "input_key":input_key,"weight_key":weight_key,"gpu_expected_key":expected_key,
                "output_sha256":{"aocl":float_digest(&aocl),"rust":float_digest(&rust),"gpu":float_digest(&expected)},
                "aocl_vs_gpu":difference(&aocl,&expected)?, "rust_vs_gpu":difference(&rust,&expected)?,
                "aocl_vs_rust":difference(&aocl,&rust)?}));
        }
    }
    ensure!(results.len() == 28, "incomplete operator coverage");
    ensure!(
        file_digest(&args.library)? == library.sha256(),
        "AOCL DLL changed during probe"
    );
    let report = json!({"schema_version":1,"status":"complete","case_count":cases.len(),"operator_count":results.len(),
        "qualification":"Isolated equal-input FP32 GEMM diagnostic under concurrent functional load; no timing measurements, no full-model parity claim, and no frozen tolerance change.",
        "execution":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"rust_threads":4,
            "rust_simd":"AVX2/FMA for <=8 rows; gemm crate independently dispatches larger prefill GEMM",
            "aocl_threads":"Library build uses OpenMP off; this probe calls AOCL serially without changing process-global thread state"},
        "library":{"path":args.library.canonicalize()?,"sha256":library.sha256(),
            "expected_sha256":args.expected_library_sha256,"symbol":"aocl_gemm_f32f32f32of32",
            "abi":"extern C; md_t=int64_t; order R, transa N, transb T; lda=k, ldb=k, ldc=n; normal-memory N,N; alpha=1 beta=0; metadata=null"},
        "aocl_source":{"path":args.aocl_source.canonicalize()?,"revision":source_revision,"tracked_changes":source_changes,
            "git_autocrlf_command_override":args.git_autocrlf,
            "gemm_header_sha256":file_digest(header)?,"base_types_header_sha256":file_digest(types)?},
        "reference":{"fixture":args.fixture,"fixture_sha256":FIXTURE_SHA256,"manifest":args.manifest,"manifest_sha256":MANIFEST_SHA256,
            "gpu_environment":metadata["environment"],"source_trace_sha256":metadata["source_trace_sha256"],
            "weights_sha256":weights_hash,"config_sha256":config_hash},
        "probe":{"binary_sha256":file_digest(std::env::current_exe()?)?,
            "source_sha256":digest(include_bytes!("aocl_probe.rs")),
            "ffi_source_sha256":digest(include_bytes!("support/aocl_dynamic.rs")),
            "rust_kernels_source_sha256":digest(include_bytes!("../src/kernels.rs")),
            "cargo_lock_sha256":digest(include_bytes!("../Cargo.lock")),
            "cargo_manifest_sha256":digest(include_bytes!("../Cargo.toml"))},
        "output_digest_encoding":"Contiguous row-major IEEE-754 FP32 values serialized as little-endian bytes, including signed zero bit patterns",
        "known_answers":answers,"operators":results});
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    // create_new prevents accidental replacement even if another process wrote
    // the destination after the initial existence check.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    serde_json::to_writer_pretty(file, &report)?;
    println!(
        "{}",
        json!({"status":"complete","cases":cases.len(),"operators":results.len(),"output":args.output})
    );
    Ok(())
}
