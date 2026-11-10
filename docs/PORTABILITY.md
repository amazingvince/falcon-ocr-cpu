# Portability

Status: current as of 2026-10-29. Runs and is measured on x86-64 (Windows,
Linux under WSL, native Linux). aarch64 compiles (CI cross-checks Linux and
runs the NEON unit tests on macOS runners); the NEON kernels are bitwise the
portable ones in those tests, but no aarch64 machine has yet run the model
with weights.

- Quality is measured against FP32's token choices; kernel tests separately
  check bitwise arithmetic. [Mode fidelity results](MODES.md).
- Targets: x86 AVX2/FMA, AVX-512 (as `auto` on AVX-512F and AVX512-BF16
  CPUs), aarch64 NEON, and a scalar fallback.

## What limits speed on each class of CPU

Decode streams every weight and the whole image KV cache once per token: its
floor is bytes per token divided by read bandwidth. Prefill is compute-bound.

| Machine class | Read bandwidth | FP32 compute | Limits decode | Limits prefill |
|---|---:|---:|---|---|
| Ryzen 7950X, DDR5 (measured) | 50-54 GB/s | 1.9-2.3 TFLOP/s | bytes | FMA throughput, exp |
| Laptops (DDR4, LPDDR5) | 30-90 GB/s | 0.3-1 TFLOP/s | bytes | FMA throughput |
| Intel with two AVX-512 FMA units | 60-300 GB/s | 2× AVX2 per core | bytes | FMA; 16-lane tiles pay off |
| Apple M4 / Pro / Max | 120 / 273 / 410-546 GB/s | 0.6-1.5 TFLOP/s NEON | dequantization compute | NEON FMA |
| Graviton 3/4 | 300+ GB/s per socket | many cores | dequantization compute | FMA; BF16/I8MM |

Only the 7950X row is measured here. The other rows are hardware-class
estimates, not measurements of this runner on those machines.

Fewer bytes per token is the universal decode lever (16-bit or 8-bit weights,
the INT8 head screen, Q16/Q8 caches). On high-bandwidth machines the decode
bottleneck moves to dequantization: those need integer dot products
(SDOT/I8MM on ARM, VNNI on x86) with per-token activation quantization, a
lossy step that would be validated by token agreement like the others.
Thread handoffs must be cheap everywhere (the spin team); dynamic task
claiming already tolerates heterogeneous cores, and `HostInfo` counts
performance cores so the decode tuner can try a team of only those.

## Code structure

Each ISA implements the same 8-lane `simd::Simd` interface
([operations and reduction trees](ARCHITECTURE.md#kernels-one-source-per-operation-instantiated-per-isa)).

| Implementation | Registers |
|---|---|
| `Avx2` | one `__m256`; `exp` is the platform-exact vector exp of `kernels/exp.rs` |
| `Avx2Fast` | `Avx2` with the portable fast exp: bitwise equal to NEON and Portable |
| `Neon` | two `float32x4_t` |
| `Portable` | `[f32; 8]` with `mul_add`: the oracle |

The x86 entry wrappers carry `#[target_feature(enable = "avx2,fma")]`;
NEON needs none. Runtime detection (`kernels::Simd::resolved`,
`prefill_plan`) lets one baseline-target binary serve every CPU. Kernels
with the same exp choice share the portable oracle's arithmetic; the
`gemm` projections and BF16 kernels are token-checked numerical deviations.

Recognition uses the portable polynomial exp in prefill attention
(token-identical to the platform exp on all calibration pages) and in fast
mode's decode attention over its 8-bit cache; on x86 the other decode
attention and traces use a vector exp that matches the platform's
`f32::exp`, with scalar fallback for uncertain lanes. RoPE also uses `powf`
and `sin_cos`. UCRT, glibc and Apple libm can differ, so platform-exact
arithmetic is not a cross-platform bitwise guarantee. The
[trace-hash gates](DEVELOPMENT.md#the-check) use Windows baselines. NEON
always uses the polynomial exp.

## Kernel inventory

| Kernel | Phase | AVX2 | AVX-512 (`auto`) | NEON | Scalar |
|---|---|---|---|---|---|
| FP32 GEMV, 1-8 rows (`kernels::linear_with_simd`) | decode | generic | same as AVX2 | generic | scalar |
| 8/16-bit GEMV and fused GLU, 1-8 rows (`quant/linear.rs`, `quant/linear/dot.rs`) | decode | generic | same as AVX2 | generic | scalar |
| W8 exception columns after the GEMV (`quant/linear.rs`, research overlays) | decode | one fused multiply-add per column (FMA instruction) | same as AVX2 | `mul_add` | same as AVX2 |
| Decode attention, compact and expanded (`attention/decode64.rs`, `online.rs`) | decode | generic head | same as AVX2 | generic head | function-pointer head |
| Split F32/Q16/Q8 cache scan and verify rows (`quant/kv/decode.rs`) | decode | generic | same as AVX2 | generic | generic loop |
| Rotated Q8R/Q4R caches (research): Hadamard rotation of queries and outputs (`quant/rotation.rs`), Q4 codes unpacked to the 8-bit loads | decode | scalar rotation, generic scan | same as AVX2 | scalar rotation, generic scan | generic loop |
| INT8 head screen (`head_screen.rs`) | decode | generic | same as AVX2 | generic | full head |
| Prefill attention tiles (`attention/prefill64.rs`) | prefill | generic | 16-lane `wide` tiles, bitwise | generic | tiled GEMM (`gemm` crate) |
| BF16 prefill attention (`prefill64/bf16.rs`, fast mode) | prefill | - | AVX512-BF16 | - | - |
| Panel GEMM for quantized projections (`panels/panel.rs`) | prefill | generic | same as AVX2 | generic | `gemm` crate |
| BF16 panel GEMM (`panels/panel_bf16.rs`, opt-in; not for bodies with exception columns) | prefill | - | AVX512-BF16 | - | - |
| FP32 projections of exact mode | prefill | `gemm` crate | `gemm` crate | `gemm` crate (NEON) | `gemm` crate |
| Fused prefill QKV row (`model/fused.rs`) | prefill | generic | same as AVX2 | generic | portable row |
| Vector exp | both | fast in recognition's prefill attention and fast mode's 8-bit decode attention; platform-exact in traces and other decode attention | same | fast | fast |
| Thread defaults (`cpu.rs`) | both | logical CPUs for prefill, tuned team for decode | same | `hw.physicalcpu`, `hw.perflevel0.physicalcpu` | Linux sysfs (`cpu_core`, `cpu_capacity`) |

`--backend avx2` forces 8-lane FP32 kernels everywhere (no wide tiles, no
BF16); `--backend scalar` replaces the vector kernels and the projections'
GEMM for numerical debugging.

## Running on Apple Silicon

`tools/m4_check.sh` is the checklist: build, unit tests (NEON versus
portable), the smoke trace against the GPU tolerances, token agreement on the
calibration pages against the stored FP32 outputs, and a speed report against
the machine's measured bandwidth. `--backend auto` selects NEON; the panel
GEMM, prefill tiles, decode attention, cache scans and the fused row all have
NEON instantiations. Prefill projections of exact mode use the `gemm` crate's
own NEON kernels. Fast mode's BF16 kernels are AVX-512-only: on ARM, fast
mode runs its FP32 prefill.

## Follow-ups

SDOT/I8MM W8 decode kernels for high-bandwidth ARM; an Accelerate backend for
prefill projections on macOS; FP32 AVX-512 panel tiles for Intel hosts
without BF16; the first weights run on an M4 (numbers into MODES.md and
PERFORMANCE.md). Other unmeasured work: [next steps](NEXT-STEPS.md).
