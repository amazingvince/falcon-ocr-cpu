# First external FP32 matrix adapter

Use **rten-gemm 0.26.0, with both operands unpacked**, in the isolated operator probe. The user's explicit Rust upgrade makes its dependency graph eligible for a Rust 1.94 build. This is an API/source selection, not a speed or numerical qualification. The adapter is [candidate.rs](../../../benchmarks/experiments/matrix_backend_compare/candidate.rs); the release inspection receipt is [matrix-backend-release-source-v1.json](../../../../reference/matrix-backend-release-source-v1.json).

The first mode keeps the existing operation `Y[M,N] = X[M,K] * W[N,K]^T`. It owns one unmodified FP32 weight payload, uses views for the transpose, and exposes the library's actual selected kernel name. No inference framework, model conversion, quantization, or production backend change is needed.

## Released versions and compatibility

I read the official `.crate` archives in memory, including normalized manifests, `.cargo_vcs_info.json`, and relevant implementation files. This resolves an ambiguity in the top-level RTen manifest: `rten-gemm` declares no MSRV, but its **mandatory `rten-simd 0.26.0` dependency declares Rust 1.94.0**. Stock Rust 1.92 was therefore ineligible. This is a released-package requirement, not an inference from repository `main`. All five inspected RTen 0.26.0 packages identify commit `14cbf4620476f402237bb29f731a244d3b6bed4d`. They use edition 2024 and MIT OR Apache-2.0 licensing. [Released GEMM archive](https://static.crates.io/crates/rten-gemm/rten-gemm-0.26.0.crate), [released SIMD archive](https://static.crates.io/crates/rten-simd/rten-simd-0.26.0.crate).

| Candidate | Minimal direct dependencies | Source-level platform assessment |
|---|---|---|
| RTen 0.26.0 | `rten-gemm = "=0.26.0"`, `rten-tensor = "=0.26.0"`; no optional feature selection needed | Rust SIMD, Rayon, no external BLAS/CMake dependency in the inspected packages. x86 implementation is architecture-gated rather than Linux-only. Native MSVC and Linux builds still require actual confirmation. |
| faer 0.24.4 | `faer = { version = "=0.24.4", default-features = false, features = ["std", "rayon"] }` | Released MSRV 1.84.0, MIT. Its x86 `std` path adds a generated-assembly build script through `private-gemm-x86`; this review does not certify MSVC assembly/build behavior. |

Faer's published source commit is `0539947ffb757a739d7e703a7d2fa0c792a909c1`. The inspected `private-gemm-x86 0.1.20` and `spindle 0.2.6` manifests also declare Rust 1.84 and MIT; these are source samples, not a substitute for the actual resolved lockfile. [faer release](https://static.crates.io/crates/faer/faer-0.24.4.crate), [private-gemm-x86 release](https://static.crates.io/crates/private-gemm-x86/private-gemm-x86-0.1.20.crate), [spindle release](https://static.crates.io/crates/spindle/spindle-0.2.6.crate).

## RTen call and execution policy

The released `gemm_impl` selects its dedicated GEMV path only when `M == 1` **and A and B are both `Unpacked`**. Passing `PackedBMatrix` would instead select tiled GEMM even for B1, so prepacking is a separate future mode. The default FP32 executor tries AVX-512 before AVX2/FMA; its kernel-selection enum is private. Record `kernel_name()` and avoid describing the candidate as forced AVX2. The standalone GEMM reads Rayon's current pool, including its block sizing; full RTen runtime environment settings do not replace installing the caller's pool. [Released implementation](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-gemm/src/lib.rs), [public executor API](https://docs.rs/rten-gemm/latest/rten_gemm/struct.GemmExecutor.html).

The concrete adapter call is:

```rust
let executor = GemmExecutor::<f32, f32, f32>::new();
let a = Matrix::from_data([m, k], input);       // strides [K, 1]
let weights_view = Matrix::from_data([n, k], weights);
let b = weights_view.transposed();             // [K,N], strides [1, K]
executor.gemm(out, GemmInputA::Unpacked(a), GemmInputB::Unpacked(b),
    GemmOptions {
        alpha: 1.0, beta: 0.0, bias: None,
        a_quant: None, b_quant: None,
    })?;
```

`Matrix` comes from the direct `rten-tensor` dependency. `out` is an existing row-major `M*N` FP32 slice. The adapter checks all lengths and checked products before constructing views. Both operand types, output, and selected kernel are FP32; it never invokes block-quantized compute mode or BF16 operations. [Tensor view source](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-tensor/src/tensor.rs), [FP32 x86 kernels](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-gemm/src/kernels/x86_64.rs).

Construct and retain `Prepared` **inside** the same `ThreadPool::install` closure used for the probe. The executor is `Sync` but not `Send`; there is no unsafe trait override. This adapter requires the caller's 16-thread pool. Weight copying and executor construction are setup work; GEMV/GEMM, internal packing and scheduling remain inside the measured call when measurement is later authorized.

The API does not promise allocation-free execution. Tiled GEMM retains thread-local packing buffers, which can allocate or grow. The report therefore records owned weight payload bytes, leaves internal scratch bytes unknown, and makes no zero-allocation claim. A future packed variant can call `executor.prepack_b(b)` once and then `GemmInputB::Packed(&packed)`; packed data is validated against kernel/blocking compatibility and adds retained storage. [Packing implementation](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-gemm/src/prepack.rs), [packing buffer](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-gemm/src/packing.rs).

## Why faer is second here

Faer's direct public call is also straightforward:

```rust
faer::linalg::matmul::matmul(
    MatMut::from_row_major_slice_mut(out, m, n), Accum::Replace,
    MatRef::from_row_major_slice(input, m, k),
    MatRef::from_row_major_slice(weights, n, k).transpose(),
    1.0f32, Par::rayon(16),
);
```

No persistent packed-weight object is accepted by that public function. It specializes single-row products; our contiguous-per-output weight layout reaches the row-major matrix/vector path after transposition. Larger x86/std products can reach `private-gemm-x86`, so testing faer is not necessarily retesting the existing `gemm 0.19` call. Its ISA choice is also internal and may differ between GEMV and GEMM. [Public matmul](https://docs.rs/faer/0.24.4/faer/linalg/matmul/fn.matmul.html), [released dispatch source](https://docs.rs/faer/0.24.4/src/faer/linalg/matmul/mod.rs.html).

`Par::rayon(16)` supplies a budget, not a new private OS thread pool. The inspected Spindle implementation uses the current Rayon pool. Its GEMM implementation uses retained TLS packing memory; other paths can allocate scratch. Preserve the same caller pool and report these costs rather than claiming a fixed scratch size. [Par API](https://docs.rs/faer/0.24.4/faer/enum.Par.html), [Spindle source](https://docs.rs/spindle/0.2.6/src/spindle/lib.rs.html).

The initial probe should preserve the original same-compiler kernel control, fixed saved real operands, and separate FP64 arithmetic-error versus GPU-output differences. Neither backend preserves the existing summation order by API contract. Build success, operator numerical results, and later timing are separate evidence; none was established by this source research.
