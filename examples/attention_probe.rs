//! Isolate attention from inherited model error using independent GPU Q/K/V.
//! This diagnostic never creates or changes the frozen full-model tolerances.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use falcon_ocr::kernels::{self, Simd};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "auto")]
    simd: String,
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// Validate 16/8-style compact storage against the expanded CPU oracle.
    #[arg(long)]
    compact_prefix_length: Option<usize>,
}

fn floats(tensor: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    ensure!(tensor.dtype() == Dtype::F32, "expected F32 fixture");
    Ok(tensor
        .data()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn error(reference: &[f32], candidate: &[f32]) -> Value {
    let mut max = 0.0_f64;
    let mut index = 0usize;
    let mut sum_squared = 0.0_f64;
    let mut changed = 0usize;
    let mut reference_peak = 0.0_f64;
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

#[allow(clippy::too_many_arguments)]
fn attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    queries: usize,
    keys: usize,
    heads: usize,
    dim: usize,
    offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; q.len()];
    let mut scores = vec![0.0_f64; keys];
    for query in 0..queries {
        let absolute = offset + query;
        for head in 0..heads {
            for (key, score) in scores.iter_mut().enumerate() {
                if key <= absolute
                    || (absolute >= image_start
                        && absolute < image_end
                        && key >= image_start
                        && key < image_end)
                {
                    let start_q = (query * heads + head) * dim;
                    let start_k = (key * heads + head) * dim;
                    *score = (0..dim)
                        .map(|d| q[start_q + d] as f64 * k[start_k + d] as f64)
                        .sum::<f64>()
                        / (dim as f64).sqrt();
                } else {
                    *score = f64::NEG_INFINITY;
                }
            }
            let max = scores.iter().copied().fold(sinks[head] as f64, f64::max);
            let denominator = (sinks[head] as f64 - max).exp()
                + scores.iter().map(|s| (s - max).exp()).sum::<f64>();
            for score in &mut scores {
                *score = (*score - max).exp() / denominator;
            }
            for d in 0..dim {
                output[(query * heads + head) * dim + d] = scores
                    .iter()
                    .enumerate()
                    .map(|(key, score)| *score * v[(key * heads + head) * dim + d] as f64)
                    .sum::<f64>() as f32;
            }
        }
    }
    output
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    let simd = match args.simd.as_str() {
        "auto" => Simd::Auto,
        "scalar" => Simd::Scalar,
        "avx2" => Simd::Avx2,
        "avx512" => Simd::Avx512,
        _ => bail!("simd must be auto, scalar, avx2, or avx512"),
    };
    simd.validate().map_err(anyhow::Error::msg)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let bytes = std::fs::read(&args.fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut names: Vec<_> = tensors
        .names()
        .into_iter()
        .filter_map(|name| name.strip_suffix(".expected"))
        .collect();
    names.sort();
    ensure!(!names.is_empty(), "no attention operators found");
    let mut reports = Vec::new();
    for name in names {
        let qview = tensors.tensor(&format!("{name}.q"))?;
        let kview = tensors.tensor(&format!("{name}.k"))?;
        ensure!(
            qview.shape().len() == 3 && kview.shape().len() == 3,
            "expected [S,H,D] Q/K"
        );
        let (queries, heads, dim) = (qview.shape()[0], qview.shape()[1], qview.shape()[2]);
        let keys = kview.shape()[0];
        let q = floats(qview)?;
        let k = floats(kview)?;
        let v = floats(tensors.tensor(&format!("{name}.v"))?)?;
        let sinks = floats(tensors.tensor(&format!("{name}.sinks"))?)?;
        let expected = floats(tensors.tensor(&format!("{name}.expected"))?)?;
        let params = tensors.tensor(&format!("{name}.params"))?;
        ensure!(
            params.dtype() == Dtype::I64 && params.data().len() == 24,
            "expected three i64 parameters"
        );
        let params: Vec<_> = params
            .data()
            .chunks_exact(8)
            .map(|v| {
                usize::try_from(i64::from_le_bytes(v.try_into().unwrap()))
                    .context("negative attention parameter")
            })
            .collect::<Result<_>>()?;
        let (offset, image_start, image_end) = (params[0], params[1], params[2]);
        let mut candidate = vec![0.0; q.len()];
        pool.install(|| {
            kernels::attention_with_simd(
                &q,
                &k,
                &v,
                queries,
                keys,
                heads,
                dim,
                offset,
                image_start,
                image_end,
                &sinks,
                &mut candidate,
                simd,
            )
        });
        let mut cache_comparison = Value::Null;
        if let Some(prefix_len) = args.compact_prefix_length {
            ensure!(
                heads % 2 == 0 && prefix_len <= keys,
                "invalid compact GQA shape or prefix length"
            );
            let kv_heads = heads / 2;
            let mut compact_k = Vec::with_capacity((keys - prefix_len) * kv_heads * dim);
            let mut compact_v = Vec::with_capacity(keys * kv_heads * dim);
            for token in 0..keys {
                for head in 0..kv_heads {
                    let begin = (token * heads + head * 2) * dim;
                    for d in 0..dim {
                        ensure!(
                            v[begin + d].to_bits() == v[begin + dim + d].to_bits(),
                            "reference value pair differs: {name}, token {token}, KV head {head}"
                        );
                        if token >= prefix_len {
                            ensure!(
                                k[begin + d].to_bits() == k[begin + dim + d].to_bits(),
                                "generated reference key pair differs: {name}, token {token}, KV head {head}"
                            );
                        }
                    }
                    compact_v.extend_from_slice(&v[begin..begin + dim]);
                    if token >= prefix_len {
                        compact_k.extend_from_slice(&k[begin..begin + dim]);
                    }
                }
            }
            let prefix_k = &k[..prefix_len * heads * dim];
            let mut compact_out = vec![0.0; q.len()];
            pool.install(|| {
                kernels::attention_compact_with_simd(
                    &q,
                    prefix_k,
                    &compact_k,
                    &compact_v,
                    queries,
                    prefix_len,
                    keys,
                    heads,
                    kv_heads,
                    dim,
                    offset,
                    image_start,
                    image_end,
                    &sinks,
                    &mut compact_out,
                    simd,
                )
            });
            let bit_exact = compact_out
                .iter()
                .zip(&candidate)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            ensure!(
                bit_exact,
                "compact attention differs from expanded CPU oracle for {name}"
            );
            let full_bytes = (k.len() + v.len()) * 4;
            let compact_bytes = (prefix_k.len() + compact_k.len() + compact_v.len()) * 4;
            cache_comparison = json!({
                "prefix_length":prefix_len,"query_heads":heads,"kv_heads":kv_heads,
                "expanded_cpu_output_bit_exact":bit_exact,
                "expanded_cache_payload_bytes":full_bytes,"compact_cache_payload_bytes":compact_bytes,
                "cache_payload_bytes_saved":full_bytes-compact_bytes,
                "cache_payload_fraction_saved":1.0-compact_bytes as f64/full_bytes as f64,
                "interpretation":"Measured tensor payload lengths, excluding weights, allocator overhead, temporary validation copies, and process RSS. No throughput promotion is claimed."
            });
            candidate = compact_out;
        }
        let ideal = attention_f64(
            &q,
            &k,
            &v,
            queries,
            keys,
            heads,
            dim,
            offset,
            image_start,
            image_end,
            &sinks,
        );
        let mut result = json!({
            "operator": name,
            "shape":{"queries":queries,"keys":keys,"heads":heads,"head_dim":dim},
            "candidate_vs_gpu":error(&expected,&candidate),
            "candidate_vs_f64":error(&ideal,&candidate),
            "gpu_vs_f64":error(&ideal,&expected),
            "cache_comparison":cache_comparison,
        });
        if let Ok(dense) = tensors.tensor(&format!("{name}.dense_expected")) {
            let dense = floats(dense)?;
            result["candidate_vs_dense_gpu"] = error(&dense, &candidate);
            result["dense_gpu_vs_f64"] = error(&ideal, &dense);
        }
        reports.push(result);
    }
    let report = json!({
        "schema_version":1,"fixture":args.fixture,"fixture_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/kernels.rs"))),
        "probe_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("attention_probe.rs"))),
        "simd":format!("{:?}",simd),"threads":args.threads,
        "interpretation":"Identical-input operators isolate attention rounding from inherited model errors. F64 is a mathematical oracle. Frozen full-model tolerances remain unchanged.",
        "operators":reports,
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
