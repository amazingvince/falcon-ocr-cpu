# Falcon-OCR CPU kernel libraries — 2026-09-20

The strongest design is a model-specific Rust runner with separate kernels for
prefill projections, single-token projections, and attention. Keep the existing
custom attention semantics and use a small matrix-backend boundary. No reviewed
library establishes that replacing the whole runner would be faster or preserve
its numerical behavior.

This is a source and primary-documentation review, accessed **2026-09-20**. No
dependencies were installed, builds or operators executed, profiles captured,
or benchmarks run for this review. Versions below are research candidates;
existing source and binary pins remain unchanged. Recommendations rank fit and
integration effort, not measured speed.

## The dimensions matter more than the library label

The pinned [configuration](../../../../artifacts/model/config.json) and
[model graph](../../../../src/model/) have 22 layers, residual width 768, 16 query
heads, 8 KV heads, head width 64, FFN width 2304 and vocabulary 65536. For
`X[M,K] * W[N,K]^T`, the actual shapes are:

| Operation | M during prefill | M during single-request decode | K | N | FP32 weight bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image projector | Image patches | Not used | 768 | 768 | 2,359,296 |
| QKV, per layer | Prefix length P | 1 | 768 | 2048 | 6,291,456 |
| Attention output, per layer | P | 1 | 1024 | 768 | 3,145,728 |
| Interleaved gate/up, per layer | P | 1 | 768 | 4608 | 14,155,776 |
| FFN down, per layer | P | 1 | 2304 | 768 | 7,077,888 |
| Vocabulary output | Last hidden row only | 1 | 768 | 65536 | 201,326,592 |

All five decode reduction lengths are multiples of 256. The vocabulary matrix
is large but its input is one 768-element vector; treating it as a large square
GEMM is misleading. The separate embedding matrix is a selected-row lookup.
W13's gate/up channels are interleaved, so a fusion expecting two contiguous
halves requires an explicit layout adapter.

The transformer projections plus vocabulary contain 219,021,312 actively used
weights per incremental token: **876,085,248 bytes (835.5 MiB)** and about
438,042,624 FLOPs, counting an FMA as two operations. Idealized B1 GEMV intensity
is roughly 0.5 FLOP per weight byte, before input/output traffic. Prefill reuses
each weight across P rows and can amortize packing; B1 often cannot. These are
analytic operand counts, not measured memory traffic or a proven bottleneck.

For attention, [current kernels](../../../../src/kernels/) use query tiles of 32 and
key tiles of 128. The two products are `(M,N,K)=(32,128,64)` for QK and
`(32,64,128)` for PV, plus tails. They run serial GEMMs inside outer Rayon
parallelism. Single-query attention instead scans keys with 64-element dots and
64-element scaled value additions. At attended length S, QK+PV across all layers
cost approximately `4*22*16*64*S = 90,112*S` FLOPs. Dense prefill attention is
approximately `90,112*P^2`; the model's bidirectional image block prevents using
a universal causal-half estimate. Actual masks and padded tiles modify executed
work. At P=6544, the dense count is about 3.859 TFLOPs, versus 2.208 TFLOPs for
the four layer projections.

Compact prefix KV stores 16 K heads and 8 V heads, while generated KV stores
8 of each. Head-specific spatial RoPE prevents whole-head prefix-K sharing.
The accepted fixed64 experiment preserves this layout and the existing online
softmax; see [its results](../../../benchmarks/experiments/attention64_compact/RESULTS-V1.md).
A matrix library does not remove these attention storage and traversal costs.

## Current versions and compatibility evidence

Dates in this table are publication dates where the source gives them. GitHub
release pages displaying only month/day are recorded that way rather than
inventing a timestamp. Moving documentation is not a captured dependency.

| Library | Current published evidence inspected | Practical implication |
| --- | --- | --- |
| `gemm` | [0.19.0, 2025-11-14](https://docs.rs/crate/gemm/0.19.0) | Already pinned locally; no newer release was shown. |
| `faer` | [0.24.4, 2026-06-24; declared MSRV 1.84](https://docs.rs/crate/faer/latest) | Plausible Rust 1.92 candidate, with a materially different x86 backend and dedicated matrix-vector paths. |
| `pulp` | [0.22.3, 2026-06-20](https://docs.rs/crate/pulp/latest) | SIMD abstraction and dispatch, not a matrix library. |
| AOCL-BLAS | [5.3.2; release page August 20](https://github.com/amd/blis/releases/tag/5.3.2) | AMD-tuned BLAS; release explicitly reports some GEMV regressions. |
| AOCL-DLP | [5.3.2; release page August 20](https://github.com/amd/aocl-dlp/releases) | Inference-oriented GEMM/reorder/low-precision APIs; distinct from AOCL-BLAS. |
| Upstream BLIS | [2.1; release notes identify improvements dated June 25, 2026](https://github.com/flame/blis/releases) | Framework, small/skinny GEMM and plugin route; release heading/date are inconsistent, so use a source pin. |
| oneDNN | [3.13.2; release page August 26](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13.2) | Use the 3.13 MatMul primitive, not its newly deprecated BLAS-like API. |
| LIBXSMM | [2.1.0; release page July 25, short commit 7944bf3](https://github.com/libxsmm/libxsmm/releases) | Current release is 2.x; historical claims that 1.17 is latest are stale. |
| OpenBLAS | [0.3.34; release page July 16](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.34) | Useful conventional BLAS comparator, with native Windows and Linux routes. |

The project remains on Rust **1.92.0**. Only existing `gemm` and the already
built native AOCL adapter have local toolchain evidence. Faer's declared MSRV
fits, but its complete selected dependency graph and Windows/WSL build still
need verification. Do not silently upgrade Rust to adopt a moving framework
branch. Rust bindings, native library, compiler, features and transitive lock
must each be pinned; a crate version alone does not identify the loaded BLAS.

## Ranked choices by useful scope

| Priority | Choice and first useful scope | Why | Cost / unresolved point |
| --- | --- | --- | --- |
| Baseline | Existing `gemm` plus current custom B1/attention kernels | Known CPU arithmetic, bounded threading, existing exact-output evidence | Retain as comparator; it is not proof of optimality. |
| 1: new Rust alternative | `faer` 0.24.4 for the four prefill projections; separately assess its row-major GEMV | Small Rust integration boundary; actual alternative x86 implementation | Different reduction order, implicit ISA dispatch, scratch/packing and thread behavior need qualification. |
| 1: existing native alternative | Pinned AOCL-DLP adapter for projections | Windows/Linux adapter and real-operand evidence already exist | Earlier W2 whole-graph substitution worsened numerical gates; do not repeat it as a claimed parity fix. |
| 2 | oneDNN 3.13 MatMul for B1 and prefill, with retained primitives and user scratch | Current release explicitly improves AVX2 GEMV-like shapes; reusable layouts | Larger descriptor/JIT/reorder/thread integration; verify selected AMD path. |
| 3 | LIBXSMM 2.1 fixed attention GEMM tiles | Both 32×128×64 products fit its small-kernel niche exactly | Windows ABI/current API validation, scaling placement, tails and numerical order. |
| 4 | AOCL-BLAS / upstream BLIS or OpenBLAS through CBLAS | Independent conventional GEMM/GEMV controls | Extra build/runtime ownership; vendor name does not establish a B1 or tile win. |
| Engineering option | `pulp` for future portable custom SIMD | Runtime dispatch and common SIMD interface | No automatic speed gain; generic reductions can change the exact tree. |

This is a shortlist, not a request to run every combination. A future comparison
should choose a phase and one competing backend, then retain the successful
backend only for that phase. Single-page decode and prefill need not share it.

### gemm and faer

Our `linear_with_simd` uses allocation-free dots for M<=8; larger non-scalar
calls use `gemm` with one Rayon budget. Thus updating a GEMM backend alone does
not change today's B1 path. `gemm` supports strides and explicit `None`/`Rayon`
parallelism. Its convention is `alpha*dst + beta*lhs*rhs`, and its low-level
signature places column stride before row stride. The current projection call
uses alpha=0, beta=1; these values must be swapped in meaning for ordinary BLAS.
HF `[N,K]` weights already represent a logical RHS `[K,N]` with row stride 1
and column stride K. No full physical transpose is inherently needed.
[API](https://docs.rs/gemm/0.19.0/gemm/fn.gemm.html),
[parallelism](https://docs.rs/gemm/0.19.0/gemm/enum.Parallelism.html).

`gemm`'s default features include `std` and `rayon`; `x86-v4` is separate.
The runner's AVX2 flag resolves our custom kernels, not necessarily every
library's internal dispatch. Capture enabled Cargo features as well as the CLI
label. [Features](https://docs.rs/crate/gemm/0.19.0/features).

Faer 0.24.4 is not merely a nicer wrapper around our same `gemm` call. Its
released matmul source first handles matrix-vector shapes with dedicated
row/column-major routines. General x86-64/std products dispatch through
`private_gemm_x86`, choosing AVX-512F before AVX2+FMA; tiny products use
`nano_gemm`, with other paths falling back to `gemm`. Use the public `matmul`
over borrowed `MatRef`/`MatMut`, explicit `Accum` and `Par`; do not depend directly
on a private crate API. The source's ISA choice requires explicit reporting if
the surrounding runner still says AVX2. These are source facts, not evidence
that its AVX-512 path wins on Zen 4.
[Released implementation](https://docs.rs/faer/0.24.4/src/faer/linalg/matmul/mod.rs.html),
[public API](https://docs.rs/faer/0.24.4/faer/linalg/matmul/fn.matmul.html).

Pulp's `Arch::dispatch` / `WithSimd` offers a portable custom-kernel boundary.
Dispatch once around a complete kernel, as in the fixed64 specialization, rather
than once per key. Preserve the existing four accumulator histories and final
horizontal tree explicitly; a vector-width-dependent generic sum changes them.
Current default and optional ISA features must be inspected independently of
the host's capabilities.
[API](https://docs.rs/pulp/0.22.3/pulp/),
[features](https://docs.rs/crate/pulp/latest/features).

### AOCL and BLIS

AOCL-BLAS 5.3.2 adds GEMV kernels/threading while acknowledging regressions in
some GEMV cases. Therefore “newest AMD library” is not evidence for this exact
five-shape B1 workload. Upstream BLIS 2.x offers plugins/control-tree extensions
for custom packing and kernels, but modifying its internals would be a larger
project than retaining our existing Rust attention.
[AOCL release](https://github.com/amd/blis/releases/tag/5.3.2),
[BLIS framework changes](https://github.com/flame/blis/releases).

AOCL-DLP 5.3.2 separately documents Windows MSVC/clang-cl, Linux/Windows CMake
presets, F32/BF16 packing work, and group-quantized S8×S4 operations. The 5.3
release includes BF16×U4 weight-only and AVX-512 GEMV work. These formats and
activation dtypes are not equivalent to our W4A32 reference. An INT4 API name
does not mean native four-bit multiply instructions or unchanged FP32 inputs.
[DLP releases](https://github.com/amd/aocl-dlp/releases),
[datatype/API overview](https://github.com/amd/aocl-dlp).

The local [AOCL experiment](../../../aocl/docs/AOCL.md) uses the separate weekly revision
`abb63d85ed7a6d559ea42b5db648e2585ac9ecb8`, not an assumed 5.3.2 binary. Its
checked `aocl_gemm_f32f32f32of32` adapter already covers Windows MSVC and WSL
GCC, correct row-major/transposed weights, `md_t=int64_t`, and serial native
execution. Reuse that boundary instead of introducing a model framework.
The prior whole-graph W2 replacement produced 47 failed hidden checks versus
10 baseline failures, despite local W2 improvements and unchanged 17 argmax
decisions. Native performance remains a separate question; there is no native
speed result in that probe.

### oneDNN

The 3.13 release specifically improves FP32/BF16/FP16 unit-dimension MatMul on
AVX2 and deprecates `dnnl::sgemm` in favor of the MatMul primitive. That gives
a concrete B1 reason to inspect it, rather than extrapolating AMX server
numbers to this CPU. Use fixed shapes where possible, retain primitives and
packed immutable weights, and separate descriptor/JIT/reorder cost from steady
execution while charging setup to cold/page measurements appropriately.
[Release](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13),
[MatMul](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_matmul.html).

Set FP math mode and accumulation mode explicitly to strict, and request
deterministic execution for repeatability. Strict controls datatype conversion
and accumulator precision; deterministic controls repeated execution on the
same platform/environment. Neither promises our CPU bits or PyTorch's summation
order. [FP math](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_attributes_fpmath_mode.html),
[accumulation](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_attributes_accumulation_mode.html),
[determinism](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_attributes_deterministic.html).

Use user-owned scratch per executing runner/task: the documented default
library scratch mode has execution-thread restrictions, and shared primitive
execution requires care. Select SEQ for calls inside our parallel tile loop,
or let oneDNN own the whole operator's thread budget. Its threadpool integration
is a C++ interoperability interface, so sharing Rayon is an adapter task, not
an automatic property of Rust FFI. Windows and Linux builds are separate
native artifacts. [Scratchpad](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_attributes_scratchpad.html),
[build options](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_build_options.html),
[threadpool](https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_threadpool.html).

Generic SDPA/GQA or gated-MLP fusion is not the first integration target:
Falcon's image mask, head-specific spatial K, interleaved squared-ReLU gate,
learned zero-value sink and separate LSE/sigmoid rounding need exact mapping.
Start with standalone MatMul and leave these model operations in Rust.

### LIBXSMM

Its documented small-matrix heuristic, cube-root(MNK) around 64 or less,
matches both current full attention tiles exactly: `32*128*64=64^3`.
Generate and retain functions before execution; use the outer Rayon schedule
and include tail shapes. Large projections should remain blocked GEMM, not a
single enormous unrolled JIT kernel. QK's 1/8 scale and PV's accumulated output
also need their original operation placement.
[Small-matrix scope](https://libxsmm.readthedocs.io/en/latest/),
[matrix multiplication interface](https://libxsmm.readthedocs.io/en/latest/libxsmm_mm/).

Version evidence matters here: current release **2.1.0** follows 2.0.0, whose
notes explicitly add NN/NT/TN/TT FP32/FP64/BF16 GEMMs. Older documentation's
TransA=N restriction cannot be applied universally to 2.x. The exact released
dispatch API, alpha/beta support and input strides must be checked before an
adapter is frozen. The compatibility wiki's Windows restrictions and ABI
wrapper discussion date to 2023; they establish an unresolved validation item,
not proof that 2.1.0 is unsupported. Linux is the simpler initial JIT
integration prospect; native Windows needs its own ABI and generated-code
checks. [Current releases](https://github.com/libxsmm/libxsmm/releases),
[dated compatibility notes](https://github.com/libxsmm/libxsmm/wiki/Compatibility).

### OpenBLAS and Rust bindings

OpenBLAS is a useful independent SGEMM/SGEMV control. Its current README maps
Zen 4 to Skylake-X kernels; CPU recognition should not be confused with a
dedicated Zen 4 kernel. It documents Windows builds as well as Linux and
different thread controls for OpenMP versus its native threading. Select one
thread owner and record the actual loaded configuration.
[Official README](https://github.com/OpenMathLib/OpenBLAS/blob/develop/README.md),
[runtime controls](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/).

`cblas-sys` 0.3.0 (2025-05-28) supplies C bindings, not an implementation.
`openblas-src` 0.10.16 (2026-05-08) can provide the native library, but its
Windows route requires `system`/vcpkg rather than building OpenBLAS itself.
That is a limitation of the crate's integration, not a claim OpenBLAS cannot
be built on Windows. A small checked CBLAS loader, similar to our AOCL adapter,
would keep the selected DLL/SO identity visible and avoid unneeded LAPACK.
[CBLAS bindings](https://docs.rs/crate/cblas-sys/latest),
[source-provider integration](https://docs.rs/crate/openblas-src/latest).

## 7950X and numerical constraints

AMD lists 16 cores/32 threads, 64 MiB L3 and two DDR5 memory channels for the
7950X. The model's active weight and KV payloads substantially exceed L3.
Zen 4 implements AVX-512 through 256-bit execution datapaths; wider instructions
can change register pressure or instruction count without doubling arithmetic
throughput. EPYC core-count/bandwidth measurements do not transfer to this Ryzen.
[7950X specifications](https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-9-7950x.html),
[AMD discussion of Zen 4 and Zen 5 execution datapaths](https://www.amd.com/en/blogs/2025/leadership-hpc-performance-with-5th-generation-amd.html).

The [existing host ISA probe](../../../../reference/quantization-feasibility-layouts.json)
observed AVX2/FMA, AVX-512F/BW/VL/VNNI/BF16, but no AVX-VNNI, AVX-512FP16 or AMX.
AVX-512 VNNI and VEX AVX-VNNI are different feature bits. Guard the actual
instructions plus OS state, including on WSL; never select a kernel merely from
the string “AVX-512.” WSL functional evidence does not establish bare-metal
Linux performance.

The current CPU-preserving contract includes four independent AVX2 dot
accumulators, their chronological FMAs, the same pairwise/horizontal merge,
128-key attention tile order, online normalizer, separate sink sigmoid and
per-value AXPY order. Reassociation, splitting K, changing tile sizes,
vector exp, BF16 conversion and fused post-ops may change bits. Different
backends can all be valid FP32 implementations while failing this contract.
The existing frozen GPU hidden-tensor policy has ten open failures; its bounds
must not be loosened to install a library. Operator FP64 error, current-CPU
bit equality, GPU graph gates and OCR outputs answer different questions.

## A small integration contract for later work

Keep an immutable model-level backend plan for each `(phase,M,K,N,dtype,layout)`;
each runner owns execution scratch. A useful conceptual boundary is
`linear(input[M,K], weight[N,K], out[M,N], scratch, parallelism)`, with separate
prepared-weight construction and measured byte counts. Attention tile adapters
need explicit strides and accumulation semantics. Resolve dispatch outside
inner loops; retain a native library/JIT lifetime as long as any function
pointer or prepared weight exists.

For a future chosen backend, the necessary evidence is limited and concrete:

1. Pin release/source, Cargo features/lock, compiler, native binary, selected
   ISA, integer ABI and threading runtime. Check known nonsymmetric shapes,
   beta-zero destinations and real strided layouts.
2. Compare actual saved operands for all four projection shapes, vocabulary,
   and the selected M values; include attention tails if that is the target.
   Preserve native outputs and independent error calculations. Report packing,
   scratch and persistent memory separately.
3. Run unchanged same-prefix graph/output checks before combining backends.
   An operator improvement is not a graph-parity result. Retain the old backend
   for unsupported shapes and preserve warm decode allocation checks.
4. Once a backend is selected for an experiment, measure the chosen phase and full pages with
   matching thread budgets and bracketed controls. Packing/JIT/init belong in
   their appropriate cold or page costs. No speed ranking follows from this
   research alone.

The recently tested paired-GEMV and split-temporal experiments missed the
existing 5% whole-page target despite exact outputs; their results constrain
speculation about small local changes. They do not eliminate matrix-library,
cache-locality or precision opportunities. The proposed hybrid architecture
allows those questions to be answered independently without abandoning the
model-specific arithmetic and validation already established.
