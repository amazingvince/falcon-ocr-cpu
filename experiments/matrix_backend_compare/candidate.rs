//! Isolated RTen 0.26.0 FP32 adapter. No production integration.
//!
//! The first candidate deliberately keeps both operands unpacked: RTen's
//! released implementation dispatches M=1 to GEMV only for that combination.
//! Matrix views transpose the retained row-major weights without moving data.

use anyhow::{Context, Result, ensure};
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmOptions};
use rten_tensor::Matrix;
use serde_json::{Value, json};

const THREADS: usize = 16;

pub struct Prepared {
    // GemmExecutor is Sync but !Send. Construct and consume Prepared inside the
    // caller's ThreadPool::install closure; do not add an unsafe Send impl.
    executor: GemmExecutor<f32, f32, f32>,
    weights: Box<[f32]>,
    k: usize,
    n: usize,
}

fn check_pool() -> Result<()> {
    ensure!(
        rayon::current_thread_index().is_some() && rayon::current_num_threads() == THREADS,
        "RTen candidate requires execution inside the caller's 16-thread Rayon pool"
    );
    Ok(())
}

impl Prepared {
    pub fn new(weights: &[f32], k: usize, n: usize) -> Result<Self> {
        check_pool()?;
        ensure!(k > 0 && n > 0, "RTen candidate requires nonzero K and N");
        let count = k.checked_mul(n).context("weight element count overflow")?;
        let bytes = count.checked_mul(size_of::<f32>()).context("weight byte count overflow")?;
        ensure!(bytes <= isize::MAX as usize, "weight view exceeds isize::MAX bytes");
        ensure!(weights.len() == count, "weight length does not match [N,K]");
        Ok(Self {
            executor: GemmExecutor::new(),
            weights: weights.into(),
            k,
            n,
        })
    }

    pub fn run(&self, input: &[f32], m: usize, out: &mut [f32]) -> Result<()> {
        check_pool()?;
        ensure!(m > 0, "RTen candidate requires nonzero M");
        let input_len = m.checked_mul(self.k).context("input element count overflow")?;
        let output_len = m.checked_mul(self.n).context("output element count overflow")?;
        ensure!(input_len <= isize::MAX as usize / size_of::<f32>(), "input view too large");
        ensure!(output_len <= isize::MAX as usize / size_of::<f32>(), "output view too large");
        ensure!(input.len() == input_len, "input length does not match [M,K]");
        ensure!(out.len() == output_len, "output length does not match [M,N]");

        let a = Matrix::from_data([m, self.k], input);
        let weight_view = Matrix::from_data([self.n, self.k], self.weights.as_ref());
        let b = weight_view.transposed();
        self.executor
            .gemm(
                out,
                GemmInputA::Unpacked(a),
                GemmInputB::Unpacked(b),
                GemmOptions {
                    alpha: 1.0,
                    beta: 0.0,
                    bias: None,
                    a_quant: None,
                    b_quant: None,
                },
            )
            .map_err(|error| anyhow::anyhow!("RTen FP32 gemm failed: {error:?}"))
    }

    pub fn info(&self) -> Value {
        json!({
            "library": "rten-gemm",
            "version": "0.26.0",
            "mode": "fp32_unpacked_b_transposed_view",
            "effective_kernel": self.executor.kernel_name(),
            "isa_policy": "library runtime default; AVX512 before AVX2/FMA; no public override",
            "precision": "f32 operands, f32 output, f32 accumulation; no quantization",
            "alpha": 1.0,
            "beta": 0.0,
            "bias": false,
            "shape_weights": [self.n, self.k],
            "weight_layout": "row-major [N,K]; RHS [K,N] view strides [1,K]",
            "input_layout": "row-major [M,K]",
            "output_layout": "row-major [M,N]",
            "prepacked_weights": false,
            "retained_weight_payload_bytes": self.weights.len() * size_of::<f32>(),
            "retained_bytes_scope": "owned weight payload only; executor and allocator overhead excluded",
            "scratch_bytes": null,
            "scratch_policy": "library-owned TLS packing buffers for GEMM; not measured; no allocation-free claim",
            "thread_count": THREADS,
            "thread_policy": "caller-owned Rayon pool; library uses current pool; construction and calls stay inside it",
            "single_row_dispatch": "RTen gemv because A and B are both unpacked",
            "multirow_dispatch": "RTen tiled gemm with internal packing",
        })
    }
}
