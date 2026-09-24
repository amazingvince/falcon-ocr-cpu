//! Reproducible warm operator benchmarks; not a whole-model promotion gate.
//!
//! Example:
//! cargo run --release --example kernel_bench -- --threads 16 --simd avx2
//! Run alternatives separately on an otherwise idle machine. JSON goes to
//! stdout; optional --output saves the same report directly without shell
//! encoding conversions.

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use falcon_ocr::kernels::{self, Simd};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SimdChoice {
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

impl From<SimdChoice> for Simd {
    fn from(value: SimdChoice) -> Self {
        match value {
            SimdChoice::Auto => Self::Auto,
            SimdChoice::Scalar => Self::Scalar,
            SimdChoice::Avx2 => Self::Avx2,
            SimdChoice::Avx512 => Self::Avx512,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Suite {
    All,
    Linear,
    Attention,
}

#[derive(Parser)]
#[command(about = "Warm FP32 Falcon-OCR numerical-kernel benchmarks (JSON)")]
struct Args {
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[arg(long, value_enum, default_value_t = SimdChoice::Auto)]
    simd: SimdChoice,
    #[arg(long, value_enum, default_value_t = Suite::All)]
    suite: Suite,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 7)]
    repetitions: usize,
    /// Number of invocations in each timed sample.
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    /// Prefill sequence length; dimensions of actual model layers stay fixed.
    #[arg(long, default_value_t = 512)]
    sequence: usize,
    #[arg(long, default_value_t = 4096)]
    decode_context: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn measure(args: &Args, useful_flops: f64, mut operation: impl FnMut()) -> Value {
    for _ in 0..args.warmup {
        operation();
    }
    let mut samples_ms = Vec::with_capacity(args.repetitions);
    for _ in 0..args.repetitions {
        let start = Instant::now();
        for _ in 0..args.iterations {
            operation();
        }
        samples_ms.push(start.elapsed().as_secs_f64() * 1e3 / args.iterations as f64);
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let median_ms = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    json!({
        "median_ms": median_ms,
        "min_ms": sorted[0],
        "max_ms": sorted[sorted.len() - 1],
        "mean_ms": samples_ms.iter().sum::<f64>() / samples_ms.len() as f64,
        "useful_gflops_per_second": useful_flops / (median_ms * 1e6),
        "samples_ms_per_invocation": samples_ms,
    })
}

fn linear_case(
    args: &Args,
    simd: Simd,
    name: &str,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Value {
    let input = data(rows * input_dim, 61);
    let weights = data(output_dim * input_dim, 19);
    let mut output = vec![0.0; rows * output_dim];
    let timing = measure(args, (2 * rows * input_dim * output_dim) as f64, || {
        kernels::linear_with_simd(
            black_box(&input),
            rows,
            input_dim,
            black_box(&weights),
            output_dim,
            black_box(&mut output),
            simd,
        );
        black_box(output[0]);
    });
    json!({
        "name": name,
        "operator": "linear",
        "shape": {"rows": rows, "input_dim": input_dim, "output_dim": output_dim},
        "weight_bytes": weights.len() * 4,
        "output_checksum_f64": output.iter().map(|&x| x as f64).sum::<f64>(),
        "timing": timing,
    })
}

fn attention_case(args: &Args, simd: Simd, decode: bool) -> Value {
    const HEADS: usize = 16;
    const DIM: usize = 64;
    let (queries, keys, offset, image_start, image_end) = if decode {
        (
            1,
            args.decode_context,
            args.decode_context - 1,
            1,
            args.decode_context / 2,
        )
    } else {
        (args.sequence, args.sequence, 0, 1, args.sequence - 7)
    };
    let q = data(queries * HEADS * DIM, 71);
    let k = data(keys * HEADS * DIM, 83);
    let v = data(keys * HEADS * DIM, 117);
    let sinks = data(HEADS, 24);
    let mut output = vec![0.0; q.len()];
    let visible_pairs: usize = (offset..offset + queries)
        .map(|pos| {
            if pos >= image_start && pos < image_end {
                image_end
            } else {
                pos + 1
            }
        })
        .sum();
    let timing = measure(args, (4 * visible_pairs * HEADS * DIM) as f64, || {
        kernels::attention_with_simd(
            black_box(&q),
            black_box(&k),
            black_box(&v),
            queries,
            keys,
            HEADS,
            DIM,
            offset,
            image_start,
            image_end,
            &sinks,
            black_box(&mut output),
            simd,
        );
        black_box(output[0]);
    });
    json!({
        "name": if decode {"attention_decode"} else {"attention_prefill"},
        "operator": "attention",
        "shape": {"query_len": queries, "kv_len": keys, "heads": HEADS, "head_dim": DIM},
        "mask": {"query_offset": offset, "image_start": image_start, "image_end_exclusive": image_end},
        "expanded_kv_bytes": (k.len() + v.len()) * 4,
        "output_checksum_f64": output.iter().map(|&x| x as f64).sum::<f64>(),
        "timing": timing,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    ensure!(
        args.repetitions > 0 && args.iterations > 0,
        "repetitions and iterations must be positive"
    );
    ensure!(
        args.sequence >= 9 && args.sequence <= 16384,
        "sequence must be between 9 and 16384"
    );
    ensure!(
        args.decode_context >= 4 && args.decode_context <= 16384,
        "decode-context must be between 4 and 16384"
    );
    let simd = Simd::from(args.simd);
    simd.validate().map_err(anyhow::Error::msg)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let cases = pool.install(|| {
        let mut results = Vec::new();
        if matches!(args.suite, Suite::All | Suite::Linear) {
            // Pinned v1.5 dimensions: Q=1024,K=512,V=512; interleaved FFN=4608.
            for rows in [1, 2, 4, 8] {
                for (name, input, output) in [
                    ("qkv_decode", 768, 2048),
                    ("ffn_gate_decode", 768, 4608),
                    ("ffn_down_decode", 2304, 768),
                ] {
                    results.push(linear_case(&args, simd, name, rows, input, output));
                }
            }
            results.push(linear_case(&args, simd, "vocabulary_decode", 1, 768, 65536));
            for (name, input, output) in [
                ("qkv_prefill", 768, 2048),
                ("ffn_gate_prefill", 768, 4608),
                ("ffn_down_prefill", 2304, 768),
            ] {
                results.push(linear_case(&args, simd, name, args.sequence, input, output));
            }
        }
        if matches!(args.suite, Suite::All | Suite::Attention) {
            results.push(attention_case(&args, simd, false));
            results.push(attention_case(&args, simd, true));
        }
        results
    });
    let report = json!({
        "schema_version": 1,
        "benchmark": "falcon-ocr-fp32-kernels",
        "timestamp_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "platform": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
            "cpu_identifier": std::env::var("PROCESSOR_IDENTIFIER").ok(),
            "logical_processors": std::thread::available_parallelism().map(|n| n.get()).ok()},
        "configuration": {"threads": args.threads, "simd_requested": format!("{:?}", simd),
            "simd_resolved": format!("{:?}", simd.resolved()),
            "avx2_available": Simd::Avx2.validate().is_ok(), "avx512_available": Simd::Avx512.validate().is_ok(),
            "debug_assertions": cfg!(debug_assertions), "warmup_invocations": args.warmup,
            "sample_count": args.repetitions, "invocations_per_sample": args.iterations,
            "prefill_sequence": args.sequence, "decode_context": args.decode_context,
            "kernel_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../src/kernels.rs"))),
            "dependency_lock_sha256": format!("{:x}", Sha256::digest(include_bytes!("../Cargo.lock"))),
            "data_generator": "LCG 1664525/1013904223, seeds in source, FP32 uniform [-1,1)"},
        "interpretation": "Warm operator timings only. Allocation/initialization is excluded. Linear batches above eight rows and multiquery optimized attention use gemm's independent ISA dispatch. Useful attention FLOPs exclude masked pairs. No whole-model speedup or GPU parity is established.",
        "cases": cases,
    });
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{encoded}");
    Ok(())
}
