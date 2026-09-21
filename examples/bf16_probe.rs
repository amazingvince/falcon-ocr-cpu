//! Compare BF16×BF16→FP32 CPU projections with independently captured GPU
//! operators. Timings are opt-in; this tool does not qualify a BF16 model graph.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use falcon_ocr::bf16_kernels::{self, Backend, PackedLinear};
use falcon_ocr::trace::{TensorTrace, Trace};
use half::bf16;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// BF16 inputs/weights and FP32 expected outputs, with a JSON weight map.
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Export native BF16-rounded candidate values for frozen per-element gates.
    #[arg(long)]
    export: Option<PathBuf>,
    #[arg(long, default_value = "auto")]
    backend: String,
    #[arg(long, default_value_t = 2)]
    threads: usize,
    #[arg(long, default_value_t = 8)]
    oracle_rows: usize,
    #[arg(long)]
    filter: Option<String>,
    /// Explicit opt-in: allocate/pack first, then measure warm operator calls.
    #[arg(long)]
    timing: bool,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 7)]
    repetitions: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
}

fn bfloats(tensor: TensorView<'_>) -> Result<Vec<bf16>> {
    ensure!(tensor.dtype() == Dtype::BF16, "expected BF16 operands");
    let data: Vec<_> = tensor
        .data()
        .chunks_exact(2)
        .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
        .collect();
    ensure!(data.iter().all(|v| v.is_finite()), "nonfinite BF16 operand");
    Ok(data)
}

fn floats(tensor: TensorView<'_>) -> Result<Vec<f32>> {
    ensure!(
        tensor.dtype() == Dtype::F32,
        "expected FP32 operator outputs"
    );
    let data: Vec<_> = tensor
        .data()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(data.iter().all(|v| v.is_finite()), "nonfinite GPU output");
    Ok(data)
}

fn stats(reference: &[f64], candidate: &[f32]) -> Value {
    assert_eq!(reference.len(), candidate.len());
    assert!(!reference.is_empty());
    let mut max = 0.0_f64;
    let mut index = 0;
    let mut squared = 0.0;
    let mut peak = 0.0_f64;
    let mut changed = 0;
    for (i, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        let delta = (a - b as f64).abs();
        if delta > max {
            max = delta;
            index = i;
        }
        squared += delta * delta;
        peak = peak.max(a.abs());
        changed += usize::from(a != b as f64);
    }
    json!({"max_abs":max,"peak_error_index":index,"rms_abs":(squared/reference.len() as f64).sqrt(),
        "reference_peak_abs":peak,"different_elements":changed})
}

fn selected_rows(rows: usize, target: usize, worst: [usize; 2]) -> Vec<usize> {
    if rows <= target {
        return (0..rows).collect();
    }
    let mut selected = BTreeSet::from([0, rows - 1]);
    for i in 1..target.saturating_sub(1) {
        selected.insert(i * (rows - 1) / (target - 1));
    }
    selected.extend(worst);
    selected.into_iter().collect()
}

fn sampled(values: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}

fn measure(args: &Args, mut operation: impl FnMut()) -> Value {
    // Even without timing, execute once to produce the comparison tensor.
    operation();
    if !args.timing {
        return Value::Null;
    }
    for _ in 0..args.warmup {
        operation();
    }
    let mut samples = Vec::with_capacity(args.repetitions);
    for _ in 0..args.repetitions {
        let start = Instant::now();
        for _ in 0..args.iterations {
            operation();
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / args.iterations as f64);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5;
    json!({"median_ms":median,"samples_ms":samples})
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.threads > 0 && args.oracle_rows > 0,
        "threads/oracle rows must be positive"
    );
    ensure!(
        !args.timing || (args.repetitions > 0 && args.iterations > 0),
        "timing counts must be positive"
    );
    let backend = match args.backend.as_str() {
        "auto" => Backend::Auto,
        "scalar" => Backend::Scalar,
        "avx512bf16" => Backend::Avx512Bf16,
        _ => bail!("backend must be auto, scalar, or avx512bf16"),
    };
    backend.validate().map_err(anyhow::Error::msg)?;
    let bytes = std::fs::read(&args.fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let metadata_bytes = std::fs::read(args.fixture.with_extension("json"))?;
    let metadata: Value = serde_json::from_slice(&metadata_bytes)?;
    let fixtures = metadata
        .get("fixtures")
        .unwrap_or(&metadata["operators"])
        .as_array()
        .context("sidecar requires fixtures with name and weight_key")?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let mut weights: HashMap<String, (Vec<bf16>, PackedLinear)> = HashMap::new();
    let mut packing_reports = Vec::new();
    let mut reports = Vec::new();
    let mut export = TensorTrace::default();
    for fixture in fixtures {
        let name = fixture["name"].as_str().context("fixture name")?;
        if args
            .filter
            .as_ref()
            .is_some_and(|filter| !name.contains(filter))
        {
            continue;
        }
        let weight_key = fixture["weight_key"].as_str().context("weight_key")?;
        let input_view = tensors.tensor(&format!("{name}.input"))?;
        ensure!(
            input_view.shape().len() == 2,
            "input must have shape [rows,K]: {name}"
        );
        let (rows, input_dim) = (input_view.shape()[0], input_view.shape()[1]);
        let weight_view = tensors.tensor(weight_key)?;
        ensure!(
            weight_view.shape().len() == 2 && weight_view.shape()[1] == input_dim,
            "weight must have shape [N,K]: {name}"
        );
        let output_dim = weight_view.shape()[0];
        let expected_view = tensors.tensor(&format!("{name}.expected"))?;
        ensure!(
            expected_view.shape() == [rows, output_dim]
                && rows > 0
                && input_dim > 0
                && output_dim > 0,
            "expected must have nonempty shape [rows,N]: {name}"
        );
        let input = bfloats(input_view)?;
        let expected = floats(expected_view)?;
        if !weights.contains_key(weight_key) {
            let data = bfloats(weight_view)?;
            let started = args.timing.then(Instant::now);
            let packed = PackedLinear::new(&data, input_dim, output_dim);
            let packing_ms = started.map(|start| start.elapsed().as_secs_f64() * 1000.0);
            packing_reports.push(json!({"weight_key":weight_key,
                "source_tensor_bytes":data.len()*2,"packed_tensor_bytes":packed.packed_weight_bytes(),"packing_ms":packing_ms}));
            weights.insert(weight_key.to_owned(), (data, packed));
        }
        let (weight, packed) = &weights[weight_key];
        let mut direct = vec![0.0; expected.len()];
        let mut packed_output = vec![0.0; expected.len()];
        let direct_timing = pool.install(|| {
            measure(&args, || {
                bf16_kernels::linear(
                    black_box(&input),
                    rows,
                    input_dim,
                    black_box(weight),
                    output_dim,
                    black_box(&mut direct),
                    backend,
                );
                black_box(&direct);
            })
        });
        let packed_timing = pool.install(|| {
            measure(&args, || {
                packed.linear(
                    black_box(&input),
                    rows,
                    black_box(&mut packed_output),
                    backend,
                );
                black_box(&packed_output);
            })
        });
        ensure!(
            direct.iter().chain(&packed_output).all(|v| v.is_finite()),
            "nonfinite CPU output: {name}"
        );
        let expected_f64: Vec<_> = expected.iter().map(|&v| v as f64).collect();
        let direct_gpu = stats(&expected_f64, &direct);
        let packed_gpu = stats(&expected_f64, &packed_output);
        let selected = selected_rows(
            rows,
            args.oracle_rows,
            [
                direct_gpu["peak_error_index"].as_u64().unwrap() as usize / output_dim,
                packed_gpu["peak_error_index"].as_u64().unwrap() as usize / output_dim,
            ],
        );
        // BF16 operands convert to F64 exactly. Retain the F64 sum; do not
        // round this oracle to FP32 before calculating either device's error.
        let mut oracle = vec![0.0_f64; selected.len() * output_dim];
        pool.install(|| {
            oracle
                .par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(index, dst)| {
                    let src =
                        &input[selected[index] * input_dim..(selected[index] + 1) * input_dim];
                    for (channel, value) in dst.iter_mut().enumerate() {
                        let w = &weight[channel * input_dim..(channel + 1) * input_dim];
                        *value = src
                            .iter()
                            .zip(w)
                            .map(|(x, w)| x.to_f64() * w.to_f64())
                            .sum();
                    }
                })
        });
        let bf16_cast_differences = |candidate: &[f32]| {
            candidate
                .iter()
                .zip(&expected)
                .filter(|&(a, b)| bf16::from_f32(*a).to_bits() != bf16::from_f32(*b).to_bits())
                .count()
        };
        if args.export.is_some() {
            for (label, values) in [("direct", &direct), ("packed", &packed_output)] {
                let rounded: Vec<_> = values.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
                export.tensor(&format!("{name}.{label}"), &[rows, output_dim], &rounded)?;
            }
        }
        reports.push(json!({
            "operator":name,"weight_key":weight_key,
            "shape":{"rows":rows,"input_dim":input_dim,"output_dim":output_dim},
            "weight_tensor_bytes":weight.len()*2,"packed_weight_tensor_bytes":packed.packed_weight_bytes(),
            "direct_vs_gpu_full":direct_gpu,"packed_vs_gpu_full":packed_gpu,
            "oracle_rows":selected,"oracle_values_dtype":"float64",
            "direct_vs_f64_sampled":stats(&oracle,&sampled(&direct,&selected,output_dim)),
            "packed_vs_f64_sampled":stats(&oracle,&sampled(&packed_output,&selected,output_dim)),
            "gpu_vs_f64_sampled":stats(&oracle,&sampled(&expected,&selected,output_dim)),
            "posthoc_bf16_cast_different_elements":{"direct":bf16_cast_differences(&direct),"packed":bf16_cast_differences(&packed_output)},
            "timing":{"direct":direct_timing,"packed":packed_timing},
        }));
        eprintln!("Compared {name} [{rows},{input_dim}] x [{output_dim},{input_dim}]");
    }
    ensure!(!reports.is_empty(), "no matching BF16 operators");
    let report = json!({
        "schema_version":1,"fixture":args.fixture,
        "fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "fixture_metadata_sha256":format!("{:x}",Sha256::digest(&metadata_bytes)),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/bf16_kernels.rs"))),
        "probe_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("bf16_probe.rs"))),
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.lock"))),
        "backend_requested":format!("{:?}",backend),"backend_resolved":format!("{:?}",backend.resolved()),
        "threads":args.threads,"debug_assertions":cfg!(debug_assertions),
        "target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH,
        "packing":packing_reports,
        "timing_configuration":if args.timing { json!({"warmups":args.warmup,"samples":args.repetitions,"iterations_per_sample":args.iterations}) } else { Value::Null },
        "interpretation":"Experimental BF16 operands, FP32 outputs. F64 oracles use identical BF16 operands. Posthoc BF16 cast comparisons round the FP32 reference output and do not establish equivalence to native GPU BF16-output GEMM or the HF model. Timing excludes construction, allocation and packing; no model speed promotion or GPU BF16 graph parity is claimed.",
        "operators":reports,
    });
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    if let Some(path) = &args.export {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        export.save(path)?;
        let mut metadata = report.clone();
        metadata["export_sha256"] = format!("{:x}", Sha256::digest(std::fs::read(path)?)).into();
        metadata["contract_sha256"] = format!(
            "{:x}",
            Sha256::digest(std::fs::read("reference/bf16-local-contract-v1.json")?)
        )
        .into();
        metadata["interpretation"] = "Direct/packed candidate FP32 accumulation is explicitly rounded to BF16 once and stored exactly as FP32. Frozen native-output bounds must be assessed separately; export alone is not qualification.".into();
        std::fs::write(
            path.with_extension("json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
    }
    println!(
        "Compared {} BF16 operators; report: {}",
        reports.len(),
        args.output.display()
    );
    Ok(())
}
