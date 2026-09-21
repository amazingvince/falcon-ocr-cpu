# Isolated faer FP32 matrix adapter

This is one source-only adapter for the existing real-operand matrix probe. It
does not modify the RTen experiment, production runner, frozen main, or baseline
kernels. Build, accuracy, allocation and timing evidence remain pending.

The parent capture will reuse the frozen main/control workload under Rust 1.94.0:
17 distinct B1 cases (four projections at four captured decoder layers plus the
vocabulary projection) and four optional M144 projections. No synthetic repeated
rows or full-page performance inference is introduced. The original probe's
independent FP64 coverage, saved CPU vocabulary control, full GPU comparisons,
allocation intervals and control/candidate/control protocol must remain intact.

## Interface and dependency

`candidate.rs` provides the unchanged interface:

```rust
Prepared::new(weights: &[f32], k: usize, n: usize) -> anyhow::Result<Prepared>
Prepared::run(&self, input: &[f32], m: usize, out: &mut [f32]) -> anyhow::Result<()>
Prepared::info(&self) -> serde_json::Value
```

Replace the two RTen dependencies in the separate parent-owned manifest with:

```toml
faer = { version = "=0.24.4", default-features = false, features = ["std", "rayon"] }
```

The adapter also uses the probe's existing pinned `anyhow`, `rayon` and
`serde_json`. The parent owns dependency resolution, the separate lockfile and
capture. No build or package resolution was performed during adapter preparation.

## Arithmetic, memory and threading contract

Preparation copies the unchanged row-major `W[N,K]` once into `Box<[f32]>`.
Every call constructs borrowed `X[M,K]`, `W[N,K].transpose()` and `Y[M,N]` views.
The RHS has strides `[1,K]`; there is no per-call physical weight transpose.
The public `matmul` receives `Accum::Replace`, FP32 alpha 1 and `Par::rayon(16)`.
Replace does not read old output values and corresponds to beta 0. This is one
compute mode for all cases, with no adapter prepacking or backend override.

Both construction and execution reject calls outside the caller's existing
16-thread Rayon pool. The explicit parallel budget does not create another pool
or change a global setting. The parent main can continue constructing and using
all adapters inside its existing `ThreadPool::install` closure. Shape products,
slice lengths and addressable byte counts are checked before mutable output use.

The public API does not reveal its actual selected ISA/kernel. `info()` therefore
reports `effective_kernel: null`, the dispatch policy and detected CPU features
as availability only. Single-row and multirow paths can select different code.
Internal allocation/TLS packing retention is unknown until observed; the reported
owned payload excludes allocator overhead. Constructor copy cost stays separate,
while any internal call-time packing/scheduling remains inside the measured call.

Faer changes reduction order. Arithmetic-envelope checks are not GPU parity or
full-model qualification, and the prefill FP64 oracle covers only the original
three fixed rows. No production integration or timing claim follows from this
source preparation. Native Windows and Linux build behavior must be verified,
including the actually resolved generated-assembly dependencies.

## Primary source basis

Reviewed 2026-09-20, using the earlier
[released-package inspection](../../docs/research/fp32-matrix-adapter-2026-09-20.md)
and the versioned public API. Faer 0.24.4 identifies release source commit
`0539947ffb757a739d7e703a7d2fa0c792a909c1`; the capture must bind the actual
resolved dependency graph rather than assume previously sampled transitive
versions.

- [faer 0.24.4 public matmul and Replace semantics](https://docs.rs/faer/0.24.4/faer/linalg/matmul/fn.matmul.html)
- [faer 0.24.4 dispatch source](https://docs.rs/faer/0.24.4/src/faer/linalg/matmul/mod.rs.html)
- [faer 0.24.4 parallelism API](https://docs.rs/faer/0.24.4/faer/enum.Par.html)
- [official released archive](https://static.crates.io/crates/faer/faer-0.24.4.crate)

Only `candidate.rs` and this note are owned by this source task. The parent owns
the new main copy, manifest/lock, preparation, build, output and run receipts.
