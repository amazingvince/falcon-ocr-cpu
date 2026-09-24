# CPU kernels, quantization, and image libraries: source review

Research date: 2026-09-20. Target: the pinned Falcon-OCR v1.5 model on the Ryzen 7950X, native Windows and Linux. This review did not build libraries, run models, or measure performance. Repository `main`/`master` links describe the source inspected on this date, not a frozen release or an adoption decision. Capture a commit, dependency lock, compiler, enabled features, and actual selected kernel before an experiment.

The most useful next dependencies are **standalone SIMD/math or matrix kernels**, particularly RTen's supporting crates and an isolated MLAS W4 operator. Moving the complete model to Candle, tract, RTen, or ONNX Runtime would add graph, state, and preprocessing work without establishing a faster kernel. Keep the existing runner and its verified image/token/cache contracts while measuring a small operator substitution.

## Ranked candidates

| Priority | Candidate | Concrete first use | Main integration risk |
|---|---|---|---|
| 1 | `rten-vecmath` + `rten-simd`; SLEEF as independent alternative | SIMD exponentials inside the existing attention tile, retaining its accumulation order | Approximation, subnormal handling, and compiler FMA choices change numerical results |
| 2 | Existing project W4A32 kernel; `rten-gemm` and MLAS as comparators | Decode GEMV on actual projection inputs, with prepacked immutable weights | Quantization error, unpack cost, activation conversion, and extra scratch/pools |
| 3 | `rten-gemm` INT8; MLAS; selected ggml/Candle kernels | A separately labeled W8A8 or W4A8 experiment | Saturating non-VNNI dot products; calibration; a different arithmetic contract |
| 4 | BF16 storage/conversion and a feature-verified native kernel | Separate BF16 model mode or cache-storage experiment | BF16 storage does not imply a fast BF16 dot instruction or upstream dtype parity |
| 5 | `tract-linalg` | A small prepacked FP32/INT8 comparison if the first alternatives disappoint | Packing, assembly/toolchain support, kernel dispatch, and changed reduction order |
| 6 | `fast_image_resize`, `pic-scale` | Reuse SIMD structure around the existing exact coefficients | “Bicubic” does not establish Pillow pixel equality; low whole-page time share |
| Keep | Pinned `turbojpeg`/libjpeg-turbo | Existing qualified JPEG path | Decoder flags and library upgrades can alter pixels |

This ordering follows the current single-page objective. The latest saved compact control spends about 50.46 s decoding, 12.26 s in prefill, and 54.55 ms in preprocessing. Eliminating that preprocessing entirely would save only about 0.087% of this particular page's 62.82 s latency. This is an Amdahl calculation from one saved workload, not a claim about all images. The temporal-cache candidate's 3.23–3.32% gain remains below the fixed 5% speed target. See the [independent saved-result review](../../../../reference/benchmarks/windows-temporal-candidate-fullpage-review-v1.json).

## What “INT4” must mean in this runner

For a row of weights, a symmetric group quantizer reconstructs `w_hat[j] = scale[g] * q[j]`. The current isolated W4A32 experiment multiplies that scale into each weight in FP32 and then performs FP32 FMA with the original activation. Changing this to `scale * sum(x*q)` changes rounding even with identical codes. W4A8 additionally quantizes the activation and usually accumulates integer products; it is a different experiment, not an implementation detail of W4A32.

With one FP32 scale per group and no zero point, W4 needs `0.5 + 4/g` bytes/weight: 4.5 bits at group 64 or 4.25 bits at group 128. That is about 7.11x or 7.53x less matrix storage than FP32 before alignment, auxiliary sums, packing, and retained originals. W8 with the same scale convention costs `1 + 4/g` bytes/weight. These ratios apply only to quantized matrices. The existing [quantization accounting](../../../quantization-feasibility/docs/QUANTIZATION.md) already shows why selectively quantizing transformer matrices gives a much smaller whole-model reduction.

Separate three errors in every report:

1. **Format error:** original FP32 weights versus independently reconstructed quantized weights.
2. **Operator arithmetic error:** candidate versus FP64 sums using those same reconstructed weights and the same actual activations. Retain the established arithmetic bound.
3. **Model/quality change:** logits, independent generation, EOS, and OCR metrics after errors propagate through the graph.

For B1 decode, reduce weight reads and reuse an unpacked vector across live rows where applicable. For large prefill, a good packed GEMM can outperform a nibble-unpack GEMV scheme; use separate paths if measurements support it. Packing all weights back into FP32 defeats persistent storage savings. Keeping both original and packed copies must be included in RSS/capacity accounting.

## Source findings: reusable kernels versus complete runtimes

### RTen: strongest direct Rust reuse candidate

The inspected `rten-gemm` exports `GemmExecutor`, prepacked matrices, quantization parameters, and block-quantized interfaces. Its public scalar types include FP32 and integer GEMM inputs/outputs. The source provides a `may_saturate()` query rather than pretending all INT8 implementations have identical range behavior. This is directly usable below the existing model API; no ONNX graph is required. Packing lifetime, layout conversion, allocation, and the interaction of `rten-parallel`/Rayon with the owned runner pool need an explicit adapter. [GEMM implementation](https://raw.githubusercontent.com/robertknight/rten/main/rten-gemm/src/lib.rs).

The x86 implementation contains distinct AVX2/AVX-512 kernels and optional AVX-512 VNNI integer dot products. The non-VNNI path uses `maddubs` followed by `madd`, with possible intermediate saturation; selecting AVX-512 alone does not remove this hazard. Kernel constructors check ISA support. These are concrete source-level reasons to evaluate RTen's INT8 kernels rather than infer capability from its model-format support. [x86 kernels](https://raw.githubusercontent.com/robertknight/rten/main/rten-gemm/src/kernels/x86_64.rs).

`rten-simd` separates ISA capability objects from vector types, exports explicit AVX2/AVX-512 ISA types, and supports dispatch and slice tails. Its default dispatch prefers wider vectors; our benchmark must retain an explicit AVX2 choice because “widest” is not evidence of lowest latency on this CPU. Inlining through the generic SIMD layer matters at hot call sites. [SIMD implementation and interface](https://github.com/robertknight/rten/blob/main/rten-simd/src/lib.rs).

Published **rten-gemm 0.26.0** documentation also exposes `BlockQuantizedGemm`, `PackedAMatrix`, `PackedBMatrix`, and `ComputeMode`, whose purpose is to choose whether the LHS is quantized. Thus the public kernel interface is not merely an unreleased-main claim. Its internal block packing implementation was not successfully retrieved in this pass; do not assume it matches our Q4 bitstream. [Published 0.26.0 API](https://docs.rs/rten-gemm/0.26.0/rten_gemm/).

The inspected workspace and `rten-gemm` manifests likewise say 0.26.0; the **root** manifest requires Rust **1.94**, newer than this project's declared 1.92 floor. The GEMM subcrate does not declare that same floor in the inspected manifest, so verify the exact standalone dependency closure instead of asserting that it inherits the runtime's MSRV. Pin a compatible release/toolchain; any compiler change needs a same-compiler control. [Workspace manifest](https://raw.githubusercontent.com/robertknight/rten/main/Cargo.toml), [GEMM manifest](https://raw.githubusercontent.com/robertknight/rten/main/rten-gemm/Cargo.toml).

### MLAS through a narrow native adapter; `ort` only for a larger graph experiment

MLAS has distinct SQ4 FP32-compute and SQ4 INT8-compute dispatches, block-size checks, weight packing, and workspace queries. Its availability function checks the actual dispatch pointers. FP32 W4 support includes an M=1 kernel and a dequantize-to-SGEMM route; this distinction fits the decode/prefill split. [Quantized dispatch](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/mlas/lib/qnbitgemm.cpp).

The actual AVX2 source wires `SQ4BitGemmM1Kernel_CompFp32_avx2`, separate INT8 kernels, and dequantization kernels. It explicitly keeps this translation unit free of the AVX-VNNI compiler requirement; VNNI is a separate dispatch option. This makes MLAS a useful W4A32 operator comparator on an AVX2 baseline. It is an internal native interface, so a pinned C++ shim with checked sizes/layouts is preferable to assuming a stable public Rust API. [AVX2 implementation](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2.cpp).

The inspected `ort` main manifest says **2.0.0-rc.13**, targets ONNX Runtime **1.30**, and defaults to downloading/copying runtime binaries. Pin the actual runtime DLL/shared object as well as the wrapper. A complete ORT migration needs a faithful graph export for image-prefix masking, spatial/temporal positions, sink denominators, and persistent KV state. Calling a fresh tiny ONNX graph for every projection is not a sensible default; engine overhead and transfers must be measured. [Rust wrapper manifest](https://raw.githubusercontent.com/pykeio/ort/main/Cargo.toml).

ORT's quantization guide distinguishes dynamic activation quantization, calibrated static quantization, and constant-weight `MatMulNBits`; it supports RTN and optional more elaborate weight quantizers. It also documents the AVX2 U8S8 saturation hazard and reduced-range alternatives. Quantized ONNX support is therefore not a guarantee of the desired arithmetic. [Official quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

### Candle and ggml: useful quantized formats, different activation arithmetic

Candle's actual GGML traits map Q4_0 to Q8_0 activation dots and Q4_K to Q8_K. Its `matmul` converts FP32 input into the selected dot type, reuses a thread-local scratch buffer, and uses its own barrier pool. This is not our current W4A32 contract. Importing it wholesale also changes thread ownership. Q4_0's 32-element blocks and Q4_K's 256-element superblocks are not the project's group-64/128 signed-nibble format. [Quantized implementation](https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/quantized/k_quants.rs).

Candle's quantized x86 module is gated by compile-time `target_feature="avx2"` in the inspected source. A portable application must not mistake that for our explicit runtime backend selection. The workspace currently says **0.11.0** and already depends on `gemm` **0.19.0**, the version this project uses; swapping to Candle FP32 alone is not a new matrix engine. [Quantized module](https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/quantized/mod.rs), [workspace dependencies](https://raw.githubusercontent.com/huggingface/candle/main/Cargo.toml).

ggml's CPU type table likewise pairs Q4_0 with Q8_0 and Q4_K with Q8_K, and allocates/converts activation workspace when input and dot types differ. Its x86 quantized routines are concrete C/C++ kernel references; a carefully bound operator shim is a smaller task than porting this model into a GGUF runtime. Falcon language-model support elsewhere does not establish support for Falcon-OCR's custom image/rotary/attention contract. [CPU dispatch](https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-cpu/ggml-cpu.c), [x86 quantized implementation](https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-cpu/arch/x86/quants.c).

### tract and AOCL: secondary operator candidates

The inspected tract x86 dispatcher selects FMA/AVX-512/VNNI-related implementations with CPU feature checks. It also contains AMX modules, which do not justify using AMX on this machine. The generic AVX2 plug is empty in this source: the useful implementation must be identified by the actual FMA/integer kernel, not an “AVX2 supported” label. Main currently identifies itself as **0.23.8-pre**. `tract-linalg` is the relevant smaller experiment; complete ONNX graph adoption is not required. Q4 implementation details and native Windows assembly build behavior remain unqualified here. [x86 dispatcher](https://github.com/sonos/tract/blob/main/linalg/src/x86_64_fma.rs), [workspace manifest](https://raw.githubusercontent.com/sonos/tract/main/Cargo.toml).

AOCL-DLP is already a locally built, pinned native comparator. Its supported GEMM family makes BF16/INT8/INT4 worth checking against their individual ISA/packing requirements, with OpenMP disabled or explicitly budgeted. Prior FP32 evidence is mixed: better equal-input prefill W2 error did not translate into a better full trace, so no general AOCL accuracy/speed claim follows. Reuse the existing loader/build provenance rather than add another unpinned BLAS dependency. [AMD API documentation](https://amd.github.io/aocl-dlp/api/gemm/index.html), [project AOCL results](../../../aocl/docs/AOCL.md).

## SIMD exponentials: a focused experiment, not a softmax replacement

`rten-vecmath::Exp` uses range reduction and an FMA polynomial; its source states/tests a maximum 1 ULP difference against Rust's `f32::exp`. The exhaustive test is ignored by default, so this review has not reproduced that guarantee. `ReducedRangeExp` assumes nonpositive inputs and truncates near -87.67 rather than retaining the full subnormal range. `Silu` uses division, while `Sigmoid` uses a reciprocal operation: reusing a nearby activation helper can change arithmetic beyond `exp`. [Actual exp/activation implementation and tests](https://raw.githubusercontent.com/robertknight/rten/main/rten-vecmath/src/exp.rs).

SLEEF is the native alternative with documented vector accuracy variants. Its `_u10` suffix denotes a **1.0 ULP** bound, not 10 ULP. The FP32 implementation also uses range reduction, polynomial evaluation and exponent reconstruction. Neither library promises equality with Windows scalar libm or CUDA's approximate exponential instruction. Use an array-pointer C wrapper if needed rather than expose compiler-specific vector ABI types across Rust FFI. [SLEEF x86 API](https://raw.githubusercontent.com/shibatch/sleef/master/docs/x86.xhtml), [FP32 implementation](https://raw.githubusercontent.com/shibatch/sleef/master/src/libm/sleefsimdsp.c).

SLEEF's current CMake source declares 4.0.0 and enables several components by default, including OpenMP-related configuration. Build only the selected math component and explicitly control optional dependencies/threads in an experiment. `wide` is another pure Rust reference, but its exp implementation and compile-time feature branches require their own qualification; `mul_add` falls back to separate operations without FMA, and approximate reciprocal/rsqrt helpers must not silently replace division/sqrt. [SLEEF build configuration](https://raw.githubusercontent.com/shibatch/sleef/master/CMakeLists.txt), [`wide` FP32x8 source](https://raw.githubusercontent.com/Lokathor/wide/main/src/f32x8_.rs).

Proposed narrow change: evaluate eight exponentials together inside the existing 128-key tile, but retain the original denominator addition order, sink treatment, and ordered value updates. Do not simultaneously change the online softmax algorithm or reciprocal. Qualify finite values, underflow, masks and the actual scalar tail. Nonpositive score-minus-maximum inputs do not establish that every other exp call, such as a sink-related expression, has the same domain.

For reasoning about approximation, if every positive unnormalized weight has relative error at most epsilon < 1, normalizing them gives an L1 probability error bounded by `2*epsilon/(1-epsilon)`. This follows by bounding both each numerator and the denominator; a sink can be included as an extra zero-valued entry. This is an analytic local bound, not a model guarantee. It does not apply unchanged when values underflow to zero. FP32 reduction error, near-tied logits and autoregressive feedback still require actual comparisons.

## BF16, INT8 and KV traffic on this CPU

Use the verified AVX2/FMA path as the portable experiment baseline. AVX-512F, AVX-512VNNI, AVX-VNNI, and AVX-512BF16 are distinct features. Detect the required feature and OS register-state support in the actual Windows/Linux execution environment, and record the selected kernel. This review neither assumes nor denies BF16 support from the Ryzen product name, and makes no AMX assumption. AVX-512 width alone is not a throughput result on Zen 4.

BF16 storage can be loaded and widened to FP32 without native BF16 dot products, but that trades bandwidth against conversion cost and loses mantissa bits. Native BF16 dot products can change pairwise accumulation and rounding. Matching the upstream BF16 model additionally requires its conversion boundaries around normalization, rotary transforms, attention and projections. The existing [BF16 findings](../../../bf16-graph/docs/BF16.md) remain separate from FP32 qualification.

INT8 VNNI accumulates small integer products into INT32; zero-point compensation and scales still matter. Without VNNI, a `maddubs` path can saturate a pair before the INT32 sum. For full-range u8/i8, two `255*127` terms exceed signed 16-bit range. Restricted weights, a non-saturating widening implementation, or a verified VNNI path are explicit alternatives. Include activation range scans/quantization and compensation costs in timings.

Weight quantization does not reduce KV-cache reads. With 22 layers and head width 64, the compact FP32 layout's prefix K16 + V8 occupies `22*64*(16+8)*4 = 135168` bytes per prefix token. Its generated K8 + V8 occupies 90112 bytes/token. For the frozen 6544-token prefix, that is 884539392 bytes (843.5625 MiB) before generated tokens, allocator reserve, or scratch. Temporal sharing reduces the prefix representation to 112640 bytes/token, or 702.96875 MiB for the same prefix. These are layout arithmetic, not measured DRAM transfers. [Model/cache contract](../PLAN.md), [temporal candidate assessment](../../../benchmarks/docs/PERFORMANCE.md).

Attention revisits this growing state for every generated token. Reuse between paired heads, cache residency, worker scheduling and traversal order determine actual traffic. A claimed 7x matrix compression cannot imply 7x whole-run speed when attention dominates. KV quantization is a distinct future experiment: KIVI's analysis motivates different grouping for keys and values, but its results on language models do not establish safety for this model's spatially rotated image prefix. Start with more conservative storage precision and measured key/value error before 2-bit cache proposals. [KIVI paper](https://arxiv.org/abs/2402.02750).

## Calibration and model-specific constraints

Use the frozen 64-page calibration split separately from the 200 evaluation pages. Collect actual image-prefill and generated-token inputs; a short text-only calibration cannot represent the image projection, checkpoint-stored head-specific spatial frequencies, blank pages, or long OCR outputs. Keep the fixed source/model/normalization contracts. [Corpus split qualification](../../../corpus-qualification/docs/corpus-v3-review.md).

Start with the existing deterministic groupwise W4A32 format. If error is too large, activation-aware scales or selective higher precision are better motivated than a blind group-size sweep. AWQ uses activation information to choose weight scaling; GPTQ minimizes reconstruction error using approximate second-order information. Their published benefits are reasons to test, not evidence for this checkpoint. [AWQ paper](https://arxiv.org/abs/2306.00978), [GPTQ paper](https://arxiv.org/abs/2210.17323).

SmoothQuant transfers activation difficulty into weights through compensating channel scaling. For this model, apply any such identity at a linear boundary where the inverse scaling is explicit. Do not assume it commutes through RMS normalization, Q/K normalization, spatial RoPE or the interleaved `ReLU(gate)^2 * up` FFN. Fusing a new scale into adjacent weights may change rounding even when the real-number identity holds. [SmoothQuant paper](https://proceedings.mlr.press/v202/xiao23c.html), [pinned graph details](../PLAN.md).

Keep image projection, final vocabulary projection, normalization/softmax and other sensitive operations in FP32 initially. Decide which additional matrices to quantize using calibration, then report quality on the untouched evaluation split. Report complete/partial counts, cap/EOS changes, absolute CER/WER/structure metrics and category deltas; never select a new acceptance metric after seeing quantized results. Operator success alone remains operator evidence.

## Image processing: preserve the established pixels

`fast_image_resize` implements a Catmull-Rom cubic with A=-0.5 and support 2, matching the continuous Pillow bicubic kernel shape; its default filter is Lanczos3. Matching the polynomial is insufficient: pixel-center convention, antialias support, coefficient normalization/quantization, edge handling and intermediate rounding must also match. Prefer an isolated SIMD inner-loop port around the already qualified coefficients over changing the resize contract. [Actual filter source](https://raw.githubusercontent.com/Cykooz/fast_image_resize/main/src/convolution/filters.rs).

`pic-scale` has interesting SIMD/threading code, but its integer support source uses precision 15, while the Pillow-derived uint8 implementation uses 22-bit coefficients. Its weight code also performs its own quantization/correction. It therefore cannot be assumed pixel-identical even with the same filter name. Its inspected manifest says 0.7.11 with Rust 1.89; those build facts do not resolve pixel semantics. [Precision constants](https://raw.githubusercontent.com/awxkee/pic-scale/master/src/support.rs), [weight generation](https://raw.githubusercontent.com/awxkee/pic-scale/master/src/filter_weights.rs), [manifest](https://raw.githubusercontent.com/awxkee/pic-scale/master/Cargo.toml), [Pillow resampler](https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c).

The current source preserves mode-dependent first resizing before RGB conversion and two quantized resize stages. Palette/bilevel handling, alpha, 16-bit modes, CMYK and rounding cannot be replaced by “decode everything to RGB, then resize.” Existing Windows/Linux fixtures already cover 75 PNG/JPEG cases. The pinned `turbojpeg=1.5.1` decoder uses the accurate integer IDCT and fancy upsampling path; faster DCT, IDCT scaling or fast upsampling would be a new pixel contract. Keep the vendored decoder qualified against Pillow before any upgrade. [Current preprocessing](../../../../src/preprocess.rs), [fixture status](../../../benchmarks/docs/STATUS.md), [libjpeg-turbo primary documentation](https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/README.md).

## Bounded next work

1. **SIMD exp:** prototype RTen `Exp` at one existing attention call site; compare with a fixed SLEEF choice if useful. Freeze input domain, arithmetic policy and error gates first. Keep cache/GEMV changes out of that comparison.
2. **Quantized decode:** retain the checked W4A32 bitstream and independent FP64 reconstruction; compare its actual matrix shapes against one RTen/MLAS adapter. Account for prepacking, scratch, retained originals and full-run quality separately. Introduce W4A8/W8A8 only as named alternatives.
3. **KV precision:** pursue only after attributing the remaining time/traffic; keep cache quantization independent of weight quantization so error and speed remain interpretable.

No framework replacement, image-library switch, precision promotion, or new dependency is justified by this source review alone. Pinning a small operator experiment is the next decision; the current Windows evidence and future Linux measurements must remain separate.
