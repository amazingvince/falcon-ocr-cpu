//! Standalone CPU scale capture. No model, GPU call, or inference backend.
use std::{env, fs, io::{self, Write}};
include!("production_pairwise.rs");

fn cuda_shaped_sum(source: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 128];
    for (thread, sum) in lanes.iter_mut().enumerate() {
        for start in (thread * 4..source.len()).step_by(512) {
            for &x in &source[start..start + 4] {
                *sum = x.mul_add(x, *sum);
            }
        }
    }
    for warp in lanes.chunks_exact_mut(32) {
        for offset in [16, 8, 4, 2, 1] {
            for lane in 0..offset { warp[lane] += warp[lane + offset]; }
        }
    }
    (lanes[0] + lanes[64]) + (lanes[32] + lanes[96])
}

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args_os().collect();
    assert_eq!(args.len(), 3, "input.bin output.bin");
    let bytes = fs::read(&args[1])?;
    assert!(!bytes.is_empty() && bytes.len() % (768 * 4) == 0);
    let values: Vec<_> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(values.iter().all(|x| x.is_finite()));
    let mut output = fs::OpenOptions::new().write(true).create_new(true).open(&args[2])?;
    for row in values.chunks_exact(768) {
        // Order: production, production+once-rounded rsqrt, CUDA-shaped,
        // CUDA-shaped+once-rounded rsqrt. No variance is inferred from GPU rstd.
        for sum in [sum_squares_pairwise(row), cuda_shaped_sum(row)] {
            let variance = sum / 768.0 + f32::EPSILON;
            for scale in [variance.sqrt().recip(), (1.0 / (variance as f64).sqrt()) as f32] {
                assert!(sum.is_finite() && variance.is_finite() && scale.is_finite());
                for value in [sum, variance, scale] { output.write_all(&value.to_le_bytes())?; }
            }
        }
    }
    Ok(())
}
