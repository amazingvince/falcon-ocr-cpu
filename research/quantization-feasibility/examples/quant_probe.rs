//! Isolated weight-only quantization diagnostic; no runner/quality/timing path.
#[path = "../experiments/quantization/q4_reference.rs"]
#[allow(dead_code)]
mod q4_reference;
#[path = "../experiments/quantization/q8_reference.rs"]
#[allow(dead_code)]
mod q8_reference;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::File, io::Write, path::PathBuf};

const WEIGHTS: &str = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16";
const FIXTURE: &str = "ce7345c219d8923182ff66e9aad2f4d6bad3c193a19c3445a850c2d3a90e5417";
const SIDECAR: &str = "148b9d4286bf413b7c9c693b920290298d5f46cd137449ac4e394430495a8962";

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "artifacts/model/model.safetensors")]
    checkpoint: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/reference/layer-operators-fp32.safetensors"
    )]
    fixture: PathBuf,
    /// New JSON path; existing evidence is never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
fn float_hash(data: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in data {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
fn floats(view: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    ensure!(view.dtype() == Dtype::F32, "FP32 tensor required");
    let values: Vec<_> = view
        .data()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite source tensor"
    );
    Ok(values)
}

#[derive(Default)]
struct Difference {
    count: usize,
    max: f64,
    squares: f64,
    reference_squares: f64,
    reference_peak: f64,
}
impl Difference {
    fn add(&mut self, candidate: f64, reference: f64) {
        assert!(candidate.is_finite() && reference.is_finite());
        let delta = candidate - reference;
        self.count += 1;
        self.max = self.max.max(delta.abs());
        self.squares += delta * delta;
        self.reference_squares += reference * reference;
        self.reference_peak = self.reference_peak.max(reference.abs());
    }
    fn report(&self) -> Value {
        json!({"elements":self.count,"max_abs":self.max,"rms_abs":(self.squares/self.count as f64).sqrt(),
            "reference_rms":(self.reference_squares/self.count as f64).sqrt(),"reference_peak":self.reference_peak,
            "relative_l2":if self.reference_squares>0. {Some((self.squares/self.reference_squares).sqrt())}else{None}})
    }
}
fn stats(actual: &[f64], reference: &[f64]) -> Value {
    assert_eq!(actual.len(), reference.len());
    let mut diff = Difference::default();
    for (&a, &b) in actual.iter().zip(reference) {
        diff.add(a, b);
    }
    diff.report()
}

enum Quant {
    Q4(q4_reference::Q4Linear),
    Q8(q8_reference::Q8Linear),
}
impl Quant {
    fn new(bits: usize, weights: &[f32], n: usize, k: usize, g: usize) -> Result<Self> {
        Ok(if bits == 4 {
            Self::Q4(
                q4_reference::Q4Linear::quantize(weights, n, k, g).map_err(anyhow::Error::msg)?,
            )
        } else {
            Self::Q8(
                q8_reference::Q8Linear::quantize(weights, n, k, g).map_err(anyhow::Error::msg)?,
            )
        })
    }
    fn row(&self, index: usize, out: &mut [f32]) {
        match self {
            Self::Q4(q) => q.dequantize_row(index, out),
            Self::Q8(q) => q.dequantize_row(index, out),
        }
    }
    fn metadata(&self) -> Value {
        let (bytes, codes, scales) = match self {
            Self::Q4(q) => (q.payload_bytes(), hash(q.packed_codes()), q.scales()),
            Self::Q8(q) => (
                q.payload_bytes(),
                hash(&q.codes().iter().map(|&i| i as u8).collect::<Vec<_>>()),
                q.scales(),
            ),
        };
        json!({"payload_bytes":bytes,"codes_sha256":codes,"scales_f32le_sha256":float_hash(scales),"scale_count":scales.len()})
    }
}

fn row_indices(rows: usize) -> Vec<usize> {
    [
        0,
        1.min(rows - 1),
        rows / 4,
        rows / 2,
        3 * rows / 4,
        rows.saturating_sub(2),
        rows - 1,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect()
}
fn channel_indices(n: usize, qkv: bool) -> Vec<usize> {
    let mut set: BTreeSet<_> = (0..4.min(n)).chain(n.saturating_sub(4)..n).collect();
    for part in 1..=3 {
        let first = (part * n / 4) / 2 * 2;
        set.insert(first);
        if first + 1 < n {
            set.insert(first + 1);
        }
    }
    if qkv {
        set.extend([1022, 1023, 1024, 1025, 1534, 1535, 1536, 1537]);
    }
    set.into_iter().collect()
}
fn sampled_dots(
    input: &[f32],
    weight: &[f32],
    k: usize,
    rows: &[usize],
    channels: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut single = Vec::new();
    let mut double = Vec::new();
    let mut bounds = Vec::new();
    let ku = k as f64 * (f32::EPSILON as f64 / 2.);
    for &row in rows {
        for &channel in channels {
            let mut f = 0.0f32;
            let mut d = 0.0f64;
            let mut abs = 0.0f64;
            for (&x, &w) in input[row * k..(row + 1) * k]
                .iter()
                .zip(&weight[channel * k..(channel + 1) * k])
            {
                f = x.mul_add(w, f);
                let p = x as f64 * w as f64;
                d += p;
                abs += p.abs();
            }
            single.push(f as f64);
            double.push(d);
            bounds.push(ku / (1. - ku) * abs);
        }
    }
    (single, double, bounds)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "refusing to overwrite existing report"
    );
    let checkpoint_file = File::open(&args.checkpoint)?;
    // SAFETY: read-only frozen artifacts; complete contents are hashed before
    // reading tensors and again before publishing the report.
    let checkpoint = unsafe { Mmap::map(&checkpoint_file)? };
    let fixture = std::fs::read(&args.fixture)?;
    let sidecar_path = args.fixture.with_extension("json");
    let sidecar_bytes = std::fs::read(&sidecar_path)?;
    ensure!(
        hash(&sidecar_bytes) == SIDECAR,
        "fixture sidecar identity mismatch before case/weight selection"
    );
    let sidecar: Value = serde_json::from_slice(&sidecar_bytes)?;
    ensure!(hash(&checkpoint) == WEIGHTS, "checkpoint identity mismatch");
    ensure!(
        hash(&fixture) == FIXTURE && sidecar["output_sha256"] == FIXTURE,
        "fixture identity mismatch"
    );
    let model = SafeTensors::deserialize(&checkpoint)?;
    let tensors = SafeTensors::deserialize(&fixture)?;
    let cases = sidecar["cases"].as_array().context("missing cases")?;
    ensure!(cases.len() == 7, "expected seven fixed diagnostic cases");
    let mappings = [
        (
            "qkv",
            "attention.wqkv.weight",
            "attention_norm.expected",
            "qkv.expected",
        ),
        (
            "wo",
            "attention.wo.weight",
            "attention.scaled",
            "attention_projection.expected",
        ),
        (
            "w13",
            "feed_forward.w13.weight",
            "ffn_norm.expected",
            "w13.expected",
        ),
        (
            "w2",
            "feed_forward.w2.weight",
            "gate.expected",
            "w2.expected",
        ),
    ];
    let mut operators = Vec::new();
    for case in cases {
        let case_name = case["name"].as_str().context("case name")?;
        let prefix = case["weight_prefix"].as_str().context("weight prefix")?;
        for &(op, suffix, input_suffix, expected_suffix) in &mappings {
            let weight_key = format!("{prefix}{suffix}");
            let input_key = format!("{case_name}.{input_suffix}");
            let expected_key = format!("{case_name}.{expected_suffix}");
            let view = model.tensor(&weight_key)?;
            ensure!(view.shape().len() == 2, "matrix shape");
            let (n, k) = (view.shape()[0], view.shape()[1]);
            let weight_hash = hash(view.data());
            let weights = floats(view)?;
            let input_view = tensors.tensor(&input_key)?;
            let input_hash = hash(input_view.data());
            let input = floats(input_view)?;
            ensure!(input.len() % k == 0, "input shape");
            let rows = input.len() / k;
            ensure!(
                rows > 0 && Some(rows as u64) == case["rows"].as_u64(),
                "source rows"
            );
            let expected_view = tensors.tensor(&expected_key)?;
            let expected_hash = hash(expected_view.data());
            let expected = floats(expected_view)?;
            ensure!(expected.len() == rows * n, "expected shape");
            let selected_rows = row_indices(rows);
            let selected_channels = channel_indices(n, op == "qkv");
            let gpu: Vec<f64> = selected_rows
                .iter()
                .flat_map(|&r| selected_channels.iter().map(move |&c| (r, c)))
                .map(|(r, c)| expected[r * n + c] as f64)
                .collect();
            let (baseline, original64, _) =
                sampled_dots(&input, &weights, k, &selected_rows, &selected_channels);
            let mut variants = Vec::new();
            for bits in [4, 8] {
                for group in [64, 128] {
                    let quant = Quant::new(bits, &weights, n, k, group)?;
                    let mut reconstruction = Difference::default();
                    let mut restored = vec![0.; n * k];
                    for channel in 0..n {
                        let row = &mut restored[channel * k..(channel + 1) * k];
                        quant.row(channel, row);
                        for (&value, &original) in
                            row.iter().zip(&weights[channel * k..(channel + 1) * k])
                        {
                            reconstruction.add(value as f64, original as f64);
                        }
                    }
                    let (candidate, quant64, bounds) =
                        sampled_dots(&input, &restored, k, &selected_rows, &selected_channels);
                    let violations = candidate
                        .iter()
                        .zip(&quant64)
                        .zip(&bounds)
                        .filter(|((a, b), bound)| (**a - **b).abs() > **bound)
                        .count();
                    ensure!(
                        violations == 0,
                        "FP32 FMA exceeded independent F64 arithmetic bound"
                    );
                    variants.push(json!({"format":format!("w{bits}a32_g{group}"),"bits":bits,"group_size":group,
                    "storage":quant.metadata(),"reconstructed_f32le_sha256":float_hash(&restored),
                    "weight_reconstruction_error_full_matrix":reconstruction.report(),
                    "sampled_output_fp32":candidate,"sampled_output_f64_dequantized":quant64,
                    "fp32_accumulation_error_bound":bounds,"fp32_accumulation_bound_violations":violations,
                    "fp32_vs_dequantized_f64":stats(&candidate,&quant64),
                    "quantization_only_f64_vs_original_f64":stats(&quant64,&original64),
                    "candidate_vs_gpu_sampled":stats(&candidate,&gpu)}));
                }
            }
            operators.push(json!({"case":case_name,"operator":op,"weight_key":weight_key,"input_key":input_key,"expected_key":expected_key,
                "shape":{"source_rows":rows,"out_dim":n,"in_dim":k},"weight_f32le_sha256":weight_hash,
                "input_f32le_sha256":input_hash,"gpu_expected_f32le_sha256":expected_hash,
                "sample_rows":selected_rows,"sample_channels":selected_channels,"sampled_gpu":gpu,
                "sampled_original_fp32_ascending_fma":baseline,"sampled_original_f64":original64,
                "original_fp32_vs_gpu_sampled":stats(&baseline,&gpu),"gpu_vs_original_f64_sampled":stats(&gpu,&original64),"variants":variants}));
            eprintln!("captured {case_name}.{op} [{n},{k}]");
        }
    }
    ensure!(
        hash(&checkpoint) == WEIGHTS,
        "checkpoint changed during capture"
    );
    let report = json!({"schema_version":1,"scope":"equal-input isolated scalar arithmetic diagnostic; not calibration, quality qualification or timing",
        "checkpoint":args.checkpoint,"checkpoint_sha256":WEIGHTS,"fixture":args.fixture,"fixture_sha256":FIXTURE,
        "fixture_sidecar":sidecar_path,"fixture_sidecar_sha256":hash(&sidecar_bytes),"original_trace_sha256":sidecar["source_trace_sha256"],
        "precision":"FP32 activations, dequantization and ascending-K FMA; weight-only quantization",
        "sampling_policy":"Rows: sorted unique {0,1,R/4,R/2,3R/4,R-2,R-1}. Channels: first4,last4 and adjacent even/odd pairs at quartiles; QKV also includes both sides of Q/K/V boundaries. Selection independent of errors/outputs.",
        "sample_layout":"row-major over explicit sample_rows then sample_channels",
        "reconstruction_scope":"All values of each of 28 actual checkpoint matrices, for both bits4/8 and group64/128. Scalar dot outputs sampled only.",
        "source_sha256":{"probe":hash(include_bytes!("quant_probe.rs")),"q4":hash(include_bytes!("../experiments/quantization/q4_reference.rs")),"q8":hash(include_bytes!("../experiments/quantization/q8_reference.rs")),"cargo_lock":hash(include_bytes!("../Cargo.lock"))},
        "binary_sha256":hash(&std::fs::read(std::env::current_exe()?)?),"target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH,
        "threads":1,"timing":Value::Null,"model_integration":false,"quality_qualification":false,"operators":operators});
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::options()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!("{}", args.output.display());
    Ok(())
}
