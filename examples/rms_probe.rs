//! Compare a candidate RMSNorm kernel with independently exported GPU operators.
//! The sequential baseline is preserved here for diagnosis; no tolerance policy
//! is generated or modified by this probe.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use falcon_ocr::kernels;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
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

fn error(reference: &[f32], candidate: &[f32]) -> Value {
    let mut max = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut changed = 0usize;
    for (&a, &b) in reference.iter().zip(candidate) {
        let d = a as f64 - b as f64;
        max = max.max(d.abs());
        sum_squared += d * d;
        changed += usize::from(a != b);
    }
    json!({"max_abs":max,"rms_abs":(sum_squared/reference.len() as f64).sqrt(),"different_elements":changed})
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut names: Vec<_> = tensors
        .names()
        .into_iter()
        .filter_map(|name| name.strip_suffix(".input"))
        .collect();
    names.sort();
    ensure!(
        !names.is_empty(),
        "fixture contains no <name>.input tensors"
    );
    let mut reports = Vec::new();
    for name in names {
        let view = tensors.tensor(&format!("{name}.input"))?;
        let width = *view.shape().last().context("scalar fixture")?;
        let input = floats(view)?;
        let reference = floats(tensors.tensor(&format!("{name}.expected"))?)?;
        ensure!(
            input.len() == reference.len() && input.len() % width == 0,
            "fixture shape mismatch"
        );
        let mut candidate = vec![0.0; input.len()];
        kernels::rms_norm(&input, &mut candidate, width, f32::EPSILON, None);
        let mut sequential = vec![0.0; input.len()];
        let mut ideal = vec![0.0; input.len()];
        for (row, src) in input.chunks_exact(width).enumerate() {
            let scale = (src.iter().map(|v| v * v).sum::<f32>() / width as f32 + f32::EPSILON)
                .sqrt()
                .recip();
            let ideal_scale = (src.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / width as f64
                + f32::EPSILON as f64)
                .sqrt()
                .recip();
            for (index, value) in src.iter().enumerate() {
                sequential[row * width + index] = *value * scale;
                ideal[row * width + index] = (*value as f64 * ideal_scale) as f32;
            }
        }
        reports.push(json!({
            "operator": name, "rows": input.len()/width, "width": width,
            "candidate_vs_gpu": error(&reference, &candidate),
            "sequential_vs_gpu": error(&reference, &sequential),
            "candidate_vs_f64": error(&ideal, &candidate),
            "sequential_vs_f64": error(&ideal, &sequential),
            "gpu_vs_f64": error(&ideal, &reference),
        }));
    }
    let report = json!({
        "schema_version":1,
        "fixture":args.fixture,
        "fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/kernels.rs"))),
        "precision":"FP32 square, pairwise FP32 sum; no dtype widening",
        "interpretation":"Independent GPU operator and F64 mathematical comparisons only; frozen full-model tolerances are unchanged.",
        "operators":reports,
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
