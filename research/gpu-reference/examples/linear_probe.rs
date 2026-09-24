//! Isolate linear projection accuracy on independently exported GPU inputs.
//! Reports full-output GPU comparison, sampled-row F64 oracles, and warm CPU
//! timings. This never changes the frozen full-model tolerance policy.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use falcon_ocr::kernels::{self, Simd};
use falcon_ocr::packed_kernels::PhasePackedLinear;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "auto")]
    simd: String,
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[arg(long, default_value_t = 32)]
    oracle_rows: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 7)]
    repetitions: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    /// Optional substring limits the inspected operators.
    #[arg(long)]
    filter: Option<String>,
    /// Diagnostic only: split the reduction into GEMM chunks, then sum their
    /// FP32 partials pairwise. Zero uses the production kernel unchanged.
    #[arg(long, default_value_t = 0)]
    split_k: usize,
    /// Experimental persisted phase packing, restricted to AVX2 and <=8 rows.
    #[arg(long, conflicts_with = "split_k")]
    phase_packed: bool,
    /// Compare accuracy once, without warmups or any timing measurements.
    #[arg(long)]
    no_timing: bool,
    /// Use the first N captured rows; skip fixtures containing fewer rows.
    /// Enables short WO/W2 batches without recomputing any reference tensors.
    #[arg(long)]
    batch_rows: Option<usize>,
}

fn floats(tensor: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    ensure!(
        tensor.dtype() == Dtype::F32,
        "expected FP32 operator fixture"
    );
    Ok(tensor
        .data()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn stats(reference: &[f32], candidate: &[f32]) -> Value {
    let mut max = 0.0_f64;
    let mut index = 0usize;
    let mut sum_squared = 0.0_f64;
    let mut reference_peak = 0.0_f64;
    let mut changed = 0usize;
    for (i, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        let delta = a as f64 - b as f64;
        if delta.abs() > max {
            max = delta.abs();
            index = i;
        }
        reference_peak = reference_peak.max((a as f64).abs());
        sum_squared += delta * delta;
        changed += usize::from(a != b);
    }
    json!({"max_abs":max,"peak_error_index":index,"rms_abs":(sum_squared/reference.len() as f64).sqrt(),
        "reference_peak_abs":reference_peak,"different_elements":changed})
}

fn oracle_rows(rows: usize, target: usize, worst_row: usize) -> Vec<usize> {
    if rows <= target {
        return (0..rows).collect();
    }
    let mut selected = BTreeSet::new();
    // Retain the beginning and ending boundaries (including image registers
    // on the smoke fixture), then distribute the remaining samples uniformly.
    let edge = (target / 4).max(1);
    selected.extend(0..edge.min(rows));
    selected.extend(rows.saturating_sub(edge)..rows);
    let interior = target.saturating_sub(selected.len());
    for index in 1..=interior {
        selected.insert(index * (rows - 1) / (interior + 1));
    }
    // Report this explicit extra row; never omit the worst full-output error
    // merely because it fell between the deterministic samples.
    selected.insert(worst_row);
    selected.into_iter().collect()
}

fn sampled(data: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| data[row * width..(row + 1) * width].iter().copied())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn split_k_linear(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weights: &[f32],
    output_dim: usize,
    output: &mut [f32],
    chunk: usize,
    partials: &mut [f32],
) {
    let matrix_size = rows * output_dim;
    let chunks = input_dim.div_ceil(chunk);
    assert_eq!(partials.len(), chunks * matrix_size);
    let threads = rayon::current_num_threads();
    for part in 0..chunks {
        let start = part * chunk;
        let count = chunk.min(input_dim - start);
        // SAFETY: This diagnostic receives shape-validated tensors; matrix row
        // strides preserve the full input width while reducing one K interval.
        // Partial outputs occupy disjoint, fully allocated contiguous blocks.
        unsafe {
            gemm::gemm(
                rows,
                output_dim,
                count,
                partials.as_mut_ptr().add(part * matrix_size),
                1,
                output_dim as isize,
                false,
                input.as_ptr().add(start),
                1,
                input_dim as isize,
                weights.as_ptr().add(start),
                input_dim as isize,
                1,
                0.0_f32,
                1.0_f32,
                false,
                false,
                false,
                if threads > 1 {
                    gemm::Parallelism::Rayon(threads)
                } else {
                    gemm::Parallelism::None
                },
            );
        }
    }
    let mut stride = 1;
    while stride < chunks {
        for part in (0..chunks).step_by(2 * stride) {
            if part + stride < chunks {
                let (before, after) = partials.split_at_mut((part + stride) * matrix_size);
                let left = &mut before[part * matrix_size..(part + 1) * matrix_size];
                let right = &after[..matrix_size];
                left.par_iter_mut()
                    .zip(right.par_iter())
                    .for_each(|(a, b)| *a += *b);
            }
        }
        stride *= 2;
    }
    output.copy_from_slice(&partials[..matrix_size]);
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.threads > 0 && args.oracle_rows > 0,
        "threads and oracle-rows must be positive"
    );
    ensure!(
        args.batch_rows.is_none_or(|rows| rows > 0),
        "batch-rows must be positive"
    );
    ensure!(
        args.no_timing || (args.repetitions > 0 && args.iterations > 0),
        "timing counts must be positive"
    );
    let simd = match args.simd.as_str() {
        "auto" => Simd::Auto,
        "scalar" => Simd::Scalar,
        "avx2" => Simd::Avx2,
        "avx512" => Simd::Avx512,
        _ => bail!("simd must be auto, scalar, avx2, or avx512"),
    };
    simd.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        !args.phase_packed || simd.resolved() == Simd::Avx2,
        "phase-packed candidate requires AVX2/FMA"
    );
    let bytes = std::fs::read(&args.fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let metadata_path = args.fixture.with_extension("json");
    let metadata: Value = if metadata_path.exists() {
        serde_json::from_slice(&std::fs::read(&metadata_path)?)?
    } else {
        Value::Null
    };
    let mut names: Vec<_> = tensors
        .names()
        .into_iter()
        .filter_map(|name| name.strip_suffix(".input"))
        .filter(|name| {
            args.filter
                .as_ref()
                .is_none_or(|filter| name.contains(filter))
        })
        .collect();
    names.sort();
    ensure!(!names.is_empty(), "no matching operators");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let mut cached_weights: HashMap<String, Arc<[f32]>> = HashMap::new();
    let mut cached_packed: HashMap<String, Arc<PhasePackedLinear>> = HashMap::new();
    let mut packing_reports = Vec::new();
    let mut skipped_large_batches = Vec::new();
    let mut skipped_short_fixtures = Vec::new();
    let mut different_bits = 0usize;
    let mut reports = Vec::new();
    for name in names {
        let view = tensors.tensor(&format!("{name}.input"))?;
        let input_dim = *view.shape().last().context("scalar linear input")?;
        let mut input = floats(view)?;
        let source_rows = input.len() / input_dim;
        let rows = args.batch_rows.unwrap_or(source_rows);
        if rows > source_rows {
            skipped_short_fixtures.push(json!({"operator":name,"source_rows":source_rows}));
            continue;
        }
        input.truncate(rows * input_dim);
        if args.phase_packed && rows > 8 {
            skipped_large_batches.push(json!({"operator":name,"rows":rows}));
            continue;
        }
        let mut expected = floats(tensors.tensor(&format!("{name}.expected"))?)?;
        let direct_weight = format!("{name}.weight");
        let weight_name = if tensors.tensor(&direct_weight).is_ok() {
            direct_weight
        } else {
            metadata
                .get("fixtures")
                .unwrap_or(&metadata["operators"])
                .as_array()
                .and_then(|ops| ops.iter().find(|op| op["name"].as_str() == Some(name)))
                .and_then(|op| op["weight_key"].as_str())
                .with_context(|| format!("no weight mapping for {name}"))?
                .to_owned()
        };
        let weight_view = tensors.tensor(&weight_name)?;
        ensure!(
            weight_view.shape().len() == 2 && weight_view.shape()[1] == input_dim,
            "linear weight shape mismatch for {name}"
        );
        let output_dim = weight_view.shape()[0];
        ensure!(
            rows > 0 && expected.len() == source_rows * output_dim,
            "linear output shape mismatch for {name}"
        );
        expected.truncate(rows * output_dim);
        let weights = if let Some(weights) = cached_weights.get(&weight_name) {
            Arc::clone(weights)
        } else {
            let weights: Arc<[f32]> = floats(weight_view)?.into();
            cached_weights.insert(weight_name.clone(), Arc::clone(&weights));
            weights
        };
        let packed = if args.phase_packed {
            if let Some(packed) = cached_packed.get(&weight_name) {
                Some(Arc::clone(packed))
            } else {
                let started = (!args.no_timing).then(Instant::now);
                let packed = Arc::new(PhasePackedLinear::new(&weights, input_dim, output_dim));
                let packing_ms = started.map(|start| start.elapsed().as_secs_f64() * 1000.0);
                packing_reports.push(json!({"weight_key":weight_name,
                    "packed_tensor_bytes":packed.packed_weight_bytes(),"packing_ms":packing_ms}));
                cached_packed.insert(weight_name.clone(), Arc::clone(&packed));
                Some(packed)
            }
        } else {
            None
        };
        let mut candidate = vec![0.0; expected.len()];
        let mut partials = if args.split_k > 0 {
            vec![0.0; expected.len() * input_dim.div_ceil(args.split_k)]
        } else {
            Vec::new()
        };
        let mut samples_ms = Vec::with_capacity(args.repetitions);
        pool.install(|| {
            let mut operation = || {
                if let Some(packed) = &packed {
                    packed.linear_avx2(black_box(&input), rows, black_box(&mut candidate));
                } else if args.split_k == 0 {
                    kernels::linear_with_simd(
                        black_box(&input),
                        rows,
                        input_dim,
                        black_box(&weights),
                        output_dim,
                        black_box(&mut candidate),
                        simd,
                    );
                } else {
                    split_k_linear(
                        black_box(&input),
                        rows,
                        input_dim,
                        black_box(&weights),
                        output_dim,
                        black_box(&mut candidate),
                        args.split_k,
                        black_box(&mut partials),
                    );
                }
                black_box(candidate[0]);
            };
            if args.no_timing {
                operation();
                return;
            }
            for _ in 0..args.warmup {
                operation();
            }
            for _ in 0..args.repetitions {
                let start = Instant::now();
                for _ in 0..args.iterations {
                    operation();
                }
                samples_ms.push(start.elapsed().as_secs_f64() * 1000.0 / args.iterations as f64);
            }
        });
        ensure!(
            candidate
                .iter()
                .chain(&expected)
                .all(|value| value.is_finite()),
            "nonfinite linear results for {name}"
        );
        let exact_comparison = if args.phase_packed {
            let mut baseline = vec![0.0; candidate.len()];
            pool.install(|| {
                kernels::linear_with_simd(
                    &input,
                    rows,
                    input_dim,
                    &weights,
                    output_dim,
                    &mut baseline,
                    Simd::Avx2,
                )
            });
            let count = candidate
                .iter()
                .zip(&baseline)
                .filter(|&(a, b)| a.to_bits() != b.to_bits())
                .count();
            different_bits += count;
            json!({"different_bits":count,"elements":candidate.len()})
        } else {
            Value::Null
        };
        let gpu_error = stats(&expected, &candidate);
        let worst_row = gpu_error["peak_error_index"].as_u64().unwrap() as usize / output_dim;
        let selected = oracle_rows(rows, args.oracle_rows, worst_row);
        let mut ideal = vec![0.0; selected.len() * output_dim];
        pool.install(|| {
            ideal
                .par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(selected_index, out)| {
                    let row = selected[selected_index];
                    let src = &input[row * input_dim..(row + 1) * input_dim];
                    for (channel, value) in out.iter_mut().enumerate() {
                        let weight = &weights[channel * input_dim..(channel + 1) * input_dim];
                        *value = src
                            .iter()
                            .zip(weight)
                            .map(|(&x, &w)| x as f64 * w as f64)
                            .sum::<f64>() as f32;
                    }
                })
        });
        let mut sorted = samples_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if sorted.is_empty() {
            0.0
        } else if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        };
        reports.push(json!({
            "operator":name,"weight_key":weight_name,
            "shape":{"rows":rows,"input_dim":input_dim,"output_dim":output_dim},
            "fixture_source_rows":source_rows,
            "diagnostic_scratch_bytes":partials.len()*4,
            "phase_packed_vs_production_avx2":exact_comparison,
            "candidate_vs_gpu_full":gpu_error,
            "oracle_rows":selected,
            "candidate_vs_f64_sampled":stats(&ideal,&sampled(&candidate,&selected,output_dim)),
            "gpu_vs_f64_sampled":stats(&ideal,&sampled(&expected,&selected,output_dim)),
            "timing":if args.no_timing { Value::Null } else { json!({"median_ms":median,"samples_ms":samples_ms,"useful_gflops_per_second":(2*rows*input_dim*output_dim) as f64/(median*1e6)}) },
        }));
    }
    let report = json!({
        "schema_version":1,"fixture":args.fixture,"fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/kernels.rs"))),
        "packed_kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/packed_kernels.rs"))),
        "probe_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("linear_probe.rs"))),
        "simd":format!("{:?}",simd),"threads":args.threads,"debug_assertions":cfg!(debug_assertions),
        "target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH,
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.lock"))),
        "candidate":if args.phase_packed {"experimental_phase_packed_avx2"} else if args.split_k==0 {"production"} else {"diagnostic_split_k_pairwise_fp32"},
        "split_k":args.split_k,
        "packing":packing_reports,
        "phase_packed_vs_production_total_different_bits":different_bits,
        "skipped_operators_above_phase_packed_batch_limit":skipped_large_batches,
        "first_n_fixture_rows":args.batch_rows,"skipped_fixtures_with_insufficient_rows":skipped_short_fixtures,
        "timing_configuration":if args.no_timing { Value::Null } else { json!({"warmups":args.warmup,"samples":args.repetitions,"iterations_per_sample":args.iterations}) },
        "interpretation":"Complete FP32 outputs compared to independent GPU projection. F64 is a mathematical oracle on explicitly listed rows, all output channels. Timings, when enabled, exclude allocation, weight packing and validation; one-time packing is reported separately. Phase-packed candidates additionally require bit equality with production AVX2. Frozen full-model tolerances remain unchanged.",
        "operators":reports,
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    ensure!(
        !reports.is_empty(),
        "no matching operators after row selection"
    );
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        different_bits == 0,
        "phase-packed output differs from production AVX2 in {different_bits} elements"
    );
    Ok(())
}
