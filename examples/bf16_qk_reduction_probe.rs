//! Diagnostic-only QK reduction hypotheses. Production/model dispatch is untouched.
#[path = "support/bf16_qk_attention.rs"]
mod candidate_attention;
#[path = "support/bf16_qk_candidates.rs"]
mod qk_candidates;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use falcon_ocr::{
    bf16_attention::{self, Parameters},
    bf16_kernels::Backend,
    trace::{TensorTrace, Trace},
};
use half::bf16;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(
        long,
        default_value = "artifacts/reference/attention-operators-bf16.safetensors"
    )]
    fixture: PathBuf,
    #[arg(long, default_value = "reference/bf16-local-contract-v1.json")]
    contract: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/reference/bf16-local-contract-v1.safetensors"
    )]
    bounds: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/reference/bf16-attention-tiles.safetensors"
    )]
    tiles: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    export: Option<PathBuf>,
    #[arg(long, default_value_t = 2)]
    threads: usize,
}
fn bfloats(t: TensorView<'_>) -> Result<Vec<bf16>> {
    ensure!(t.dtype() == Dtype::BF16, "BF16 operands required");
    Ok(t.data()
        .chunks_exact(2)
        .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
        .collect())
}
fn floats(t: TensorView<'_>) -> Result<Vec<f32>> {
    match t.dtype() {
        Dtype::BF16 => Ok(bfloats(t)?.into_iter().map(|v| v.to_f32()).collect()),
        Dtype::F32 => Ok(t
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()),
        dtype => anyhow::bail!("unsupported float tensor {dtype:?}"),
    }
}
fn compare(actual: &[f32], expected: &[f32], bound: &[f32], rms_bound: Option<f64>) -> Value {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(bound.len(), actual.len());
    let mut max = 0f64;
    let mut squared = 0.;
    let mut different = 0;
    let mut violations = Vec::new();
    let mut finite = true;
    for (i, ((&a, &b), &bound)) in actual.iter().zip(expected).zip(bound).enumerate() {
        finite &= a.is_finite() && b.is_finite();
        let delta = (a as f64 - b as f64).abs();
        max = max.max(delta);
        squared += delta * delta;
        different += usize::from(a != b);
        if !a.is_finite() || delta > bound as f64 {
            violations.push(i);
        }
    }
    let rms = (squared / actual.len() as f64).sqrt();
    json!({"passed":finite&&violations.is_empty()&&rms_bound.is_none_or(|b|rms<=b),"elements":actual.len(),
        "different_elements":different,"max_abs":max,"rms_abs":rms,"rms_bound":rms_bound,
        "bound_violations":violations.len(),"first_violation_indices":violations.iter().take(32).collect::<Vec<_>>()})
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    Backend::Avx512Bf16.validate().map_err(anyhow::Error::msg)?;
    let input_bytes = std::fs::read(&args.fixture)?;
    let fixture = SafeTensors::deserialize(&input_bytes)?;
    let bounds_bytes = std::fs::read(&args.bounds)?;
    let bounds = SafeTensors::deserialize(&bounds_bytes)?;
    let contract_bytes = std::fs::read(&args.contract)?;
    let contract: Value = serde_json::from_slice(&contract_bytes)?;
    ensure!(
        format!("{:x}", Sha256::digest(&contract_bytes))
            == "43106ccc50516c492e7180b726526fede13b19c4d04b0397bb5b7679446083d1",
        "frozen contract changed"
    );
    for (path, bytes) in [(&args.fixture, &input_bytes), (&args.bounds, &bounds_bytes)] {
        let key = path.to_string_lossy().replace('\\', "/");
        ensure!(
            contract["sources"][&key].as_str()
                == Some(format!("{:x}", Sha256::digest(bytes)).as_str()),
            "frozen input hash mismatch {key}"
        );
    }
    let tiles_bytes = std::fs::read(&args.tiles)?;
    let tiles = SafeTensors::deserialize(&tiles_bytes)?;
    let tile_meta: Value =
        serde_json::from_slice(&std::fs::read(args.tiles.with_extension("json"))?)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let mut variants = Vec::new();
    let mut export = TensorTrace::default();
    for candidate in qk_candidates::ALL {
        let mut checks = serde_json::Map::new();
        let mut tile_checks = Vec::new();
        for case in contract["attention_cases"]
            .as_array()
            .context("attention_cases")?
        {
            let name = case["name"].as_str().context("case name")?;
            let qv = fixture.tensor(&format!("{name}.q"))?;
            let kv = fixture.tensor(&format!("{name}.k"))?;
            let (rows, heads, keys) = (qv.shape()[0], qv.shape()[1], kv.shape()[0]);
            let pvec = fixture.tensor(&format!("{name}.params"))?;
            let params: Vec<_> = pvec
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
            let q = bfloats(qv)?;
            let k = bfloats(kv)?;
            let v = bfloats(fixture.tensor(&format!("{name}.v"))?)?;
            let sinks = bfloats(fixture.tensor(&format!("{name}.sinks"))?)?;
            let mut out = vec![bf16::ZERO; q.len()];
            let mut raw = out.clone();
            let mut lse = vec![0.; rows * heads];
            pool.install(|| {
                candidate_attention::attention(
                    &q,
                    &k,
                    &v,
                    &sinks,
                    p,
                    &mut out,
                    &mut raw,
                    &mut lse,
                    Backend::Avx512Bf16,
                    candidate,
                )
            });
            if matches!(candidate, qk_candidates::Candidate::Baseline) {
                let mut check = vec![bf16::ZERO; q.len()];
                let mut checkraw = check.clone();
                let mut checklse = vec![0.; lse.len()];
                pool.install(|| {
                    bf16_attention::attention(
                        &q,
                        &k,
                        &v,
                        &sinks,
                        p,
                        &mut check,
                        &mut checkraw,
                        &mut checklse,
                        Backend::Avx512Bf16,
                    )
                });
                ensure!(
                    check == out
                        && checkraw == raw
                        && checklse
                            .iter()
                            .zip(&lse)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "diagnostic baseline differs from production {name}"
                );
            }
            for (stage, values) in [("raw", &raw), ("scaled", &out)] {
                let actual: Vec<_> = values.iter().map(|v| v.to_f32()).collect();
                let expected = floats(
                    fixture.tensor(
                        case[stage]["expected_key"]
                            .as_str()
                            .context("expected_key")?,
                    )?,
                )?;
                let bound = floats(
                    bounds.tensor(case[stage]["bound_key"].as_str().context("bound_key")?)?,
                )?;
                checks.insert(
                    format!("{name}.{stage}"),
                    compare(
                        &actual,
                        &expected,
                        &bound,
                        case[stage]["rms_bound"].as_f64(),
                    ),
                );
                if args.export.is_some() {
                    export.tensor(
                        &format!("{}.{}.{}", candidate.name(), name, stage),
                        &[rows, heads, 64],
                        &actual,
                    )?;
                }
            }
            let mut head_major = vec![0.; lse.len()];
            for row in 0..rows {
                for head in 0..heads {
                    head_major[head * rows + row] = lse[row * heads + head];
                }
            }
            let expected = floats(fixture.tensor(&format!("{name}.lse"))?)?;
            let bound =
                vec![case["lse_max_abs_bound"].as_f64().context("lse bound")? as f32; lse.len()];
            checks.insert(
                format!("{name}.lse"),
                compare(&head_major, &expected, &bound, None),
            );
            if args.export.is_some() {
                export.tensor(
                    &format!("{}.{}.lse", candidate.name(), name),
                    &[heads, rows],
                    &head_major,
                )?;
            }
            for probe in tile_meta["probes"]
                .as_array()
                .context("tile probes")?
                .iter()
                .filter(|p| p["case"].as_str() == Some(name))
            {
                let row = probe["query_row"].as_u64().unwrap() as usize;
                let head = probe["head"].as_u64().unwrap() as usize;
                for tile in probe["tiles"].as_array().unwrap() {
                    let start = tile["key_start"].as_u64().unwrap() as usize;
                    let count = tile["valid_keys"].as_u64().unwrap() as usize;
                    let key = format!(
                        "{}.tile{}.qk",
                        probe["name"].as_str().unwrap(),
                        tile["index"].as_u64().unwrap()
                    );
                    let expected = floats(tiles.tensor(&key)?)?;
                    let query = &q[(row * heads + head) * 64..(row * heads + head + 1) * 64];
                    let actual: Vec<_> = (start..start + count)
                        .map(|i| {
                            candidate.dot(
                                query,
                                &k[(i * heads + head) * 64..(i * heads + head + 1) * 64],
                            )
                        })
                        .collect();
                    let comparison = compare(&actual, &expected[..count], &vec![0.; count], None);
                    tile_checks.push(json!({"key":key,"comparison":comparison}));
                }
            }
        }
        let passed = checks.values().filter(|v| v["passed"] == true).count();
        let violations: usize = checks
            .values()
            .map(|v| v["bound_violations"].as_u64().unwrap() as usize)
            .sum();
        let qk_exact: usize = tile_checks
            .iter()
            .map(|v| {
                (v["comparison"]["elements"].as_u64().unwrap()
                    - v["comparison"]["different_elements"].as_u64().unwrap())
                    as usize
            })
            .sum();
        eprintln!(
            "{}: {passed}/{} gates; {violations} element violations; {qk_exact}/720 tile QK exact",
            candidate.name(),
            checks.len()
        );
        variants.push(json!({"name":candidate.name(),"rationale":candidate.rationale(),"passed_gates":passed,
            "element_bound_violations":violations,"tile_qk_exact":qk_exact,"checks":checks,"tile_qk":tile_checks}));
    }
    if let Some(path) = args.export {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        export.save(path)?;
    }
    let report = json!({"schema_version":1,"contract_sha256":format!("{:x}",Sha256::digest(&contract_bytes)),
        "fixture_sha256":format!("{:x}",Sha256::digest(&input_bytes)),"tile_fixture_sha256":format!("{:x}",Sha256::digest(&tiles_bytes)),
        "probe_sha256":format!("{:x}",Sha256::digest(include_bytes!("bf16_qk_reduction_probe.rs"))),
        "candidate_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/bf16_qk_candidates.rs"))),
        "attention_copy_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/bf16_qk_attention.rs"))),
        "production_attention_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/bf16_attention.rs"))),
        "threads":args.threads,"os":std::env::consts::OS,"debug_assertions":cfg!(debug_assertions),
        "baseline_entire_fixture_bit_exact":true,"production_changed":false,"bounds_changed":false,"timings_collected":false,
        "scope":"Only QK dot changes; P*V, tile schedule, exp2, denominator reduction and sink scale retain AVX512BF16 baseline. All6 full attention fixtures and all18 raw/scaled/LSE gates are evaluated. Tile QK diagnostics are720 values across5 existing GPU probes.","candidates":variants});
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(args.output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
