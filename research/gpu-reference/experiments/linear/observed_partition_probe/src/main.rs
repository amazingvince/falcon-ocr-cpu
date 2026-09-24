//! One diagnostic: production-style GEMM on the observed W2 K ranges.
//! Does not select partitions, alter a model, measure speed, or infer GPU order.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

const N: usize = 144;
const M: usize = 768;
const K: usize = 2304;
const SPLITS: usize = 14;
const CAPTURE_SHA: &str = "697b717ac9949c1f90bf94eaca8f074164ec40d10f7917ca8bd4ecf3c5a7a7f4";
const KERNELS_SHA: &str = "19f1f18e164fffcab63dd1d747aae76f3a4a75dfcc15edebacd46b5dc32f7f16";
const MAIN_SOURCE: &str = include_str!("main.rs");
const MANIFEST_SOURCE: &str = include_str!("../Cargo.toml");
const LOCK_SOURCE: &str = include_str!("../Cargo.lock");

fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn decode(data: &[u8]) -> Result<Vec<f32>> {
    ensure!(data.len() % 4 == 0, "Invalid F32 byte length");
    let values: Vec<_> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "Nonfinite input/reference"
    );
    Ok(values)
}

fn ranges() -> Vec<(usize, usize)> {
    (0..SPLITS)
        .map(|slot| (slot * 165, ((slot + 1) * 165).min(K)))
        .collect()
}

fn gather(matrix: &[f32], rows: usize, width: usize, start: usize, end: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), rows * width);
    assert!(start < end && end <= width);
    let mut out = Vec::with_capacity(rows * (end - start));
    for row in matrix.chunks_exact(width) {
        out.extend_from_slice(&row[start..end]);
    }
    out
}

fn linear(
    input: &[f32],
    rows: usize,
    width: usize,
    weights: &[f32],
    channels: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), rows * width);
    assert_eq!(weights.len(), channels * width);
    assert_eq!(output.len(), rows * channels);
    let in_stride = isize::try_from(width).unwrap();
    let out_stride = isize::try_from(channels).unwrap();
    let threads = rayon::current_num_threads();
    let parallelism = if threads == 1 {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(threads)
    };
    // Identical GEMM arguments to production kernels.rs's rows>8 non-Scalar
    // branch at KERNELS_SHA: cs precedes rs; alpha*dst + beta*lhs*rhs.
    // Partitions are gathered bit-for-bit into contiguous rows before this call.
    // SAFETY: lengths/strides above cover every pointer access, with exclusive output.
    unsafe {
        gemm::gemm(
            rows,
            channels,
            width,
            output.as_mut_ptr(),
            1,
            out_stride,
            false,
            input.as_ptr(),
            1,
            in_stride,
            weights.as_ptr(),
            in_stride,
            1,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            parallelism,
        );
    }
}

fn compare(actual: &[f32], expected: &[f32]) -> Value {
    assert_eq!(actual.len(), expected.len());
    let mut mismatches = 0_usize;
    let mut max_error = 0.0_f64;
    let mut squared_error = 0.0_f64;
    let mut worst = 0_usize;
    let mut nonfinite = 0_usize;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        mismatches += usize::from(a.to_bits() != b.to_bits());
        if !a.is_finite() || !b.is_finite() {
            nonfinite += 1;
            continue;
        }
        let error = (a as f64 - b as f64).abs();
        squared_error += error * error;
        if error > max_error {
            max_error = error;
            worst = index;
        }
    }
    json!({"elements":actual.len(),"bit_mismatches":mismatches,"nonfinite_pairs":nonfinite,
           "max_abs_error":max_error,"rms_error":(squared_error/actual.len() as f64).sqrt(),
           "worst_flat_index":worst,"worst_actual_bits":format!("{:08x}",actual[worst].to_bits()),
           "worst_expected_bits":format!("{:08x}",expected[worst].to_bits())})
}

fn save(path: &Path, values: &[f32]) -> Result<Value> {
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&data)?;
    Ok(json!({"bytes":data.len(),"sha256":hash(&data)}))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 4,
        "Usage: observed-w2-partitions CAPTURE_DIRECTORY NEW_OUTPUT_DIRECTORY THREADS(1..4)"
    );
    let capture = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let threads: usize = args[3].to_str().context("Invalid thread count")?.parse()?;
    ensure!(
        (1..=4).contains(&threads),
        "Diagnostic is limited to 1..4 threads"
    );
    ensure!(!output.exists(), "Preserve existing output directory");
    let report_bytes = fs::read(capture.join("report.json"))?;
    ensure!(
        hash(&report_bytes) == CAPTURE_SHA,
        "Pinned capture report differs"
    );
    let report: Value = serde_json::from_slice(&report_bytes)?;
    ensure!(
        report["status"] == "captured_layout_observed_conditionally"
            && report["source_and_artifact_closure_unchanged"] == true,
        "Capture did not pass"
    );
    let observed = report["layout_observation"]["conditional_k_membership"]
        .as_array()
        .context("Missing membership")?;
    ensure!(observed.len() == K, "Membership width differs");
    for (slot, (start, end)) in ranges().iter().copied().enumerate() {
        ensure!(
            observed[start..end]
                .iter()
                .all(|v| v.as_u64() == Some(slot as u64)),
            "Observed range differs"
        );
    }
    let mut input_identities = serde_json::Map::new();
    let mut operands = Vec::new();
    for name in [
        "control/input.f32le",
        "control/weight.f32le",
        "control/output.f32le",
        "control/workspace.bin",
    ] {
        let data = fs::read(capture.join(name))?;
        let meta = &report["artifacts"][name];
        ensure!(
            meta["sha256"] == hash(&data) && meta["bytes"] == data.len(),
            "Captured operand changed: {name}"
        );
        input_identities.insert(name.to_owned(), meta.clone());
        let interpreted = if name.ends_with("workspace.bin") {
            &data[..SPLITS * N * M * 4]
        } else {
            &data
        };
        operands.push(decode(interpreted)?);
    }
    let [input, weights, expected, gpu_partials]: [Vec<f32>; 4] = operands.try_into().unwrap();
    ensure!(
        input.len() == N * K
            && weights.len() == M * K
            && expected.len() == N * M
            && gpu_partials.len() == SPLITS * N * M,
        "Operand size differs"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut unsplit = vec![0.0_f32; N * M];
    let mut partials = vec![0.0_f32; SPLITS * N * M];
    pool.install(|| {
        linear(&input, N, K, &weights, M, &mut unsplit);
        for (slot, (start, end)) in ranges().into_iter().enumerate() {
            let x = gather(&input, N, K, start, end);
            let w = gather(&weights, M, K, start, end);
            linear(
                &x,
                N,
                end - start,
                &w,
                M,
                &mut partials[slot * N * M..(slot + 1) * N * M],
            );
        }
    });
    let mut fold = vec![0.0_f32; N * M];
    for part in partials.chunks_exact(N * M) {
        for (dst, &value) in fold.iter_mut().zip(part) {
            *dst += value;
        }
    }
    let per_slot: Vec<_> = ranges().into_iter().enumerate().map(|(slot,(start,end))| {
        json!({"slot":slot,"k_start":start,"k_end_exclusive":end,
               "comparison":compare(&partials[slot*N*M..(slot+1)*N*M],&gpu_partials[slot*N*M..(slot+1)*N*M])})
    }).collect();
    fs::create_dir(output)?;
    let outputs = json!({"cpu-partials.f32le":save(&output.join("cpu-partials.f32le"),&partials)?,
                         "cpu-unsplit.f32le":save(&output.join("cpu-unsplit.f32le"),&unsplit)?,
                         "cpu-ascending-fold.f32le":save(&output.join("cpu-ascending-fold.f32le"),&fold)?});
    for (name, meta) in &input_identities {
        ensure!(
            hash(&fs::read(capture.join(name))?) == meta["sha256"],
            "Input changed during probe: {name}"
        );
    }
    ensure!(
        fs::read(capture.join("report.json"))? == report_bytes,
        "Capture report changed during probe"
    );
    let current_binary = std::env::current_exe()?;
    let result = json!({"schema_version":1,"status":"diagnostic_completed","threads":threads,
        "capture_report_sha256":CAPTURE_SHA,"production_kernels_comparison_source_sha256":KERNELS_SHA,
        "probe_source_sha256":hash(MAIN_SOURCE.as_bytes()),"manifest_sha256":hash(MANIFEST_SOURCE.as_bytes()),
        "lock_sha256":hash(LOCK_SOURCE.as_bytes()),"binary_sha256":hash(&fs::read(current_binary)?),
        "gemm_version":"0.19.0","gemm_parallelism":if threads==1{"None"}else{"Rayon(explicit threads)"},
        "gather":"bit-preserving contiguous input/weight rows per fixed observed K range; in_stride=Kpart",
        "partial_shape":[SPLITS,N,M],"inputs":input_identities,"outputs":outputs,"per_slot":per_slot,
        "all_partials_vs_gpu":compare(&partials,&gpu_partials),"unsplit_vs_gpu_output":compare(&unsplit,&expected),
        "ascending_fold_vs_gpu_output":compare(&fold,&expected),
        "qualification":"One fixed observed-partition diagnostic. No alternate partitions, arithmetic or order sweep; no timing or model-quality claim. Partial membership is conditional scratch evidence; ascending output equivalence does not prove GPU reduction order."});
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("report.json"))?
        .write_all(serde_json::to_string_pretty(&result)?.as_bytes())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"partials":result["all_partials_vs_gpu"],
        "unsplit":result["unsplit_vs_gpu_output"],"fold":result["ascending_fold_vs_gpu_output"]}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_range_coverage_and_tail() {
        let r = ranges();
        assert_eq!(r.len(), 14);
        assert_eq!(r[13], (2145, 2304));
        assert!(r[..13].iter().all(|&(a, b)| b - a == 165));
        assert_eq!(
            r.iter().flat_map(|&(a, b)| a..b).collect::<Vec<_>>(),
            (0..K).collect::<Vec<_>>()
        );
    }
    #[test]
    fn gather_preserves_bits_and_row_boundaries() {
        let x: Vec<_> = (0..3 * 171)
            .map(|i| f32::from_bits(0x3f000000 + i))
            .collect();
        let y = gather(&x, 3, 171, 12, 171);
        for r in 0..3 {
            for c in 0..159 {
                assert_eq!(y[r * 159 + c].to_bits(), x[r * 171 + 12 + c].to_bits());
            }
        }
    }
    #[test]
    fn gemm_abi_matches_independent_exact_dot() {
        let (rows, channels, width) = (9, 7, 11);
        let x: Vec<_> = (0..rows * width)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.125)
            .collect();
        let w: Vec<_> = (0..channels * width)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.25)
            .collect();
        let mut out = vec![f32::NAN; rows * channels];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| linear(&x, rows, width, &w, channels, &mut out));
        for r in 0..rows {
            for c in 0..channels {
                let oracle = (0..width)
                    .map(|k| x[r * width + k] as f64 * w[c * width + k] as f64)
                    .sum::<f64>() as f32;
                assert_eq!(out[r * channels + c].to_bits(), oracle.to_bits());
            }
        }
    }
}
