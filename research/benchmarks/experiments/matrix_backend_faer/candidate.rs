//! Isolated faer 0.24.4 FP32 adapter. No production integration.
//!
//! Calls the public matmul API with an unchanged row-major weight payload and a
//! transposed view. There is no adapter-owned packed format or physical transpose.

use anyhow::{Context, Result, ensure};
use faer::{Accum, MatMut, MatRef, Par};
use serde_json::{Value, json};

const THREADS: usize = 16;

pub struct Prepared {
    weights: Box<[f32]>,
    k: usize,
    n: usize,
}

fn check_pool() -> Result<()> {
    ensure!(
        rayon::current_thread_index().is_some() && rayon::current_num_threads() == THREADS,
        "faer candidate requires execution inside the caller's 16-thread Rayon pool"
    );
    Ok(())
}

fn detected_cpu_features() -> Value {
    #[cfg(target_arch = "x86_64")]
    {
        json!({
            "avx2": std::arch::is_x86_feature_detected!("avx2"),
            "fma": std::arch::is_x86_feature_detected!("fma"),
            "avx512f": std::arch::is_x86_feature_detected!("avx512f"),
            "avx512bw": std::arch::is_x86_feature_detected!("avx512bw"),
            "avx512dq": std::arch::is_x86_feature_detected!("avx512dq"),
            "avx512vl": std::arch::is_x86_feature_detected!("avx512vl"),
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Value::Null
    }
}

impl Prepared {
    pub fn new(weights: &[f32], k: usize, n: usize) -> Result<Self> {
        check_pool()?;
        ensure!(k > 0 && n > 0, "faer candidate requires nonzero K and N");
        let count = k.checked_mul(n).context("weight element count overflow")?;
        let bytes = count.checked_mul(size_of::<f32>()).context("weight byte count overflow")?;
        ensure!(bytes <= isize::MAX as usize, "weight view exceeds isize::MAX bytes");
        ensure!(weights.len() == count, "weight length does not match [N,K]");
        Ok(Self {
            weights: weights.into(),
            k,
            n,
        })
    }

    pub fn run(&self, input: &[f32], m: usize, out: &mut [f32]) -> Result<()> {
        check_pool()?;
        ensure!(m > 0, "faer candidate requires nonzero M");
        let input_len = m.checked_mul(self.k).context("input element count overflow")?;
        let output_len = m.checked_mul(self.n).context("output element count overflow")?;
        ensure!(input_len <= isize::MAX as usize / size_of::<f32>(), "input view too large");
        ensure!(output_len <= isize::MAX as usize / size_of::<f32>(), "output view too large");
        ensure!(input.len() == input_len, "input length does not match [M,K]");
        ensure!(out.len() == output_len, "output length does not match [M,N]");

        let a = MatRef::from_row_major_slice(input, m, self.k);
        let weights = MatRef::from_row_major_slice(self.weights.as_ref(), self.n, self.k);
        let b = weights.transpose(); // [K,N], row stride 1, column stride K.
        let dst = MatMut::from_row_major_slice_mut(out, m, self.n);
        // Replace does not read old output values: alpha=1, mathematical beta=0.
        // Par sets a work budget within the caller's pool; it creates no pool.
        faer::linalg::matmul::matmul(dst, Accum::Replace, a, b, 1.0f32, Par::rayon(THREADS));
        Ok(())
    }

    pub fn info(&self) -> Value {
        json!({
            "library": "faer",
            "version": "0.24.4",
            "released_source_commit": "0539947ffb757a739d7e703a7d2fa0c792a909c1",
            "features": ["std", "rayon"],
            "default_features": false,
            "mode": "fp32_public_matmul_transposed_weight_view",
            "effective_kernel": null,
            "isa_policy": "library runtime dispatch; public matmul API does not expose actual selected kernel",
            "detected_cpu_features": detected_cpu_features(),
            "detected_features_scope": "CPU availability only, not proof of selected GEMV/GEMM instructions",
            "precision": "f32 operands, f32 output, f32 arithmetic; no quantization or dtype conversion",
            "reduction_policy": "library-defined order; no bit-identity promise versus existing CPU or GPU kernels",
            "alpha": 1.0,
            "beta": 0.0,
            "output_accumulation": "Accum::Replace; previous output values are not read",
            "bias": false,
            "shape_weights": [self.n, self.k],
            "weight_layout": "row-major [N,K]; RHS [K,N] view strides [1,K]",
            "input_layout": "row-major [M,K]",
            "output_layout": "row-major [M,N]",
            "prepacked_weights": false,
            "retained_weight_payload_bytes": self.weights.len() * size_of::<f32>(),
            "retained_bytes_scope": "owned weight payload only; allocator overhead excluded",
            "scratch_bytes": null,
            "scratch_policy": "library scratch/internal packing may allocate or retain TLS buffers; not measured here",
            "thread_count": THREADS,
            "thread_policy": "caller-owned 16-thread Rayon pool with explicit Par::rayon(16); no global configuration change",
            "single_row_dispatch": "public matmul; released source specializes M=1 through matrix/vector dispatch",
            "multirow_dispatch": "public matmul; x86/std may use private-gemm-x86; actual resolved source bound by capture lockfile",
        })
    }
}
