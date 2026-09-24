# INT8 / INT4 feasibility, 2026-09-20

> **Archived 2026-09-24.** Feasibility notes of 2026-09-20. What shipped is described in [docs/MODES.md](../../../docs/MODES.md): 16-bit and GPTQ 8-bit body weights, Q16/Q8 KV caches and an INT8 screened head; no 4-bit format was adopted.

Start with an isolated **W4A32** experiment: groupwise INT4 weights, FP32
activations/dequantization/accumulation, and the existing FP32 graph around each
linear operator. Compare a W8A32 control before adding activation quantization.
This separates weight compression from the still-unqualified BF16 graph. Nothing
in this investigation changes the runner, its default dtype, or frozen parity
bounds. There are no quantized model quality or throughput results yet.

The standalone [scalar reference](../experiments/quantization/q4_reference.rs)
implements the format and operator contract below. Four tests passed using Rust
1.92 on native Windows: known packed codes and ties-to-even, zero/subnormal and
invalid input handling, row/group tails, reconstruction error, and independent
FP64 dot comparisons at K=768/1024/2304 and batch=1/2/4/8. Output-channel counts
are deliberately small/odd for these bounded tests; they are not full-model
matrix benchmarks. No production source or Cargo dependency was changed.

The later [bounded operator experiment](../experiments/quantization/RESULTS-V1.md)
adds the W8A32 control. All 112 matrix/format reconstructions match independent
Python code, scale and reconstructed-weight bytes, and all 8,880 sampled outputs
pass independent arithmetic checks. These samples cover only 0.0375% of possible
outputs for seven synthetic cases; they select no group size, clipping rule,
layer exceptions or quality budget. Full-model quality and speed remain unmeasured.

## Hardware and implementation choices

The native [feature probe](../experiments/quantization/isa_probe.rs) observed
AVX2/FMA, AVX-512F/BW/VL/VNNI/BF16; it observed no AVX-VNNI, AVX-512FP16, AMX-INT8
or AMX-TILE. These are runtime results from this host, not an inference from a
generic AVX-512 label. The result and compiler identity are preserved in the
[layout/evidence report](../../../reference/quantization-feasibility-layouts.json).

| Candidate | Why try it | Important distinction |
|---|---|---|
| Custom Rust W4A32, group 64 then 128 | Smallest integration surface; existing FP32 activations and graph; inspect unpack/dequant costs directly | Compression alone does not guarantee faster GEMV or prefill |
| Custom Rust W8A32 control | Fewer quantization levels lost; simpler widening/conversion | Still float arithmetic after loading int8 weights |
| GGML Q4_0 / Q4_K through a pinned C operator adapter | Existing packed formats and x86 kernels; useful independent comparator | Typical quantized dots also quantize activations; preserve Falcon's graph in our runner |
| Candle quantized CPU operators | Rust implementation, QTensor/QMatMul interfaces and GGML types | Its Q4_0 dot uses Q8_0 operands; this is not the W4A32 reference semantics |
| Custom W8A8 using AVX-512 VNNI | Integer dot products and reusable quantized activations across output channels | Activation scaling/clipping adds another accuracy decision |
| AOCL-DLP BF16×S4 and INT8 | Concrete optional native GEMM/reorder alternative | BF16×S4 changes activation precision; validate its scale/layout contract and thread ownership |

GGML defines Q4_0 as 32 weights plus one FP16 scale (18 bytes); Q4_K uses 256
weights with scale/min metadata (144 bytes). Both cost 4.5 bits/weight but encode
different quantizers. Q8_0 costs 34 bytes/32 weights. Our custom signed [-7,7],
FP32-scale, adjacent-nibble representation is **not binary-compatible** with
either Q4 format. [GGML format definitions](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h)
and [reference quantizers](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c).

Candle's `GgmlType` associates Q4_0 with Q8_0 dot operands and K-quants with Q8K;
the inspected x86 dispatch includes compile-time `target_feature="avx2"` paths.
Its public `QTensor`/`QMatMul` is a useful optional comparison harness, but check
the pinned version's allocation, batching, threading and target-feature behavior
before depending on it. Do not replace this model's custom mask, sinks or rotary
logic with an unrelated generic Falcon implementation.
[Candle quantized types](https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/k_quants.rs),
[public operators](https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/mod.rs).

AOCL-DLP documents BF16×S4/S8 and integer GEMM variants, reordered weights and
scale/zero-point operations. Its README explicitly lists `u8s4s32os32` as reorder
APIs without a GEMM, so an INT4 label alone does not establish the desired
integer execution path. Verify the selected function and group-scale layout in
a pinned source revision. [Supported combinations](https://github.com/amd/aocl-dlp),
[GEMM API](https://amd.github.io/aocl-dlp/api/gemm/index.html).
Current build instructions cover Windows MSVC/clang-cl and Linux GCC/Clang, with
optional OpenMP. Build single-threaded for calls inside the runner's Rayon pool,
or let AOCL own the entire budget; never nest two full thread teams.
[AOCL build instructions](https://github.com/amd/aocl-dlp/blob/dev/BUILD.md).

Research URLs above refer to moving branches inspected on this date. The subsequent
[isolated AOCL comparison](../../aocl/docs/AOCL.md) pins source and library builds on Windows and
WSL Linux; it does not add a production backend. Pin any further native candidate
and archive source/build identity before comparison. Existing `half`, `safetensors`, `rayon` and
`std::arch` suffice for the first custom experiment. An optional Candle harness
or C FFI adapter belongs in its own dependency/build target.

## VNNI details that affect correctness

Rust 1.92 can use the stable `_mm512_dpbusd_epi32` and
`_mm256_dpbusd_epi32` intrinsics. The latter requires AVX-512VNNI **and VL**, which
this host has. It is different from `_mm256_dpbusd_avx_epi32`, which requires
AVX-VNNI and is unavailable here. Both supported widths should eventually be
measured; no claim that 512 bits is faster follows from feature detection.
[512-bit intrinsic](https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_dpbusd_epi32.html),
[256-bit EVEX intrinsic](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_epi32.html),
[distinct AVX-VNNI intrinsic](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_avx_epi32.html).

For signed symmetric activation code `a` and signed weight code `w`, use
unsigned `u=a+128`, then recover `sum(a*w) = sum(u*w)-128*sum(w)`; precompute the
weight sum per output channel/group. Apply the activation and weight scales only
at the intended group boundary. At the largest model reduction, K=2304,
`2304*255*127=74,615,040`, safely within int32. This bound must be rechecked if a
future kernel accumulates additional rows/groups into one integer accumulator.
This is a proposed W8A8 contract, not an implemented path.

An AVX2 fallback must not blindly substitute the saturating adjacent-pair
`maddubs` operation for VNNI: two products can reach `2*255*127=64,770` before
widening. Use a proven widening/split implementation. INT4 likewise needs nibble
unpacking and either float dequantization, BF16 conversion, or an integer-dot
scheme; this host does not supply an INT4 matrix primitive that eliminates those
steps. [AVX2 multiply/add intrinsic](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_maddubs_epi16.html).
The instruction's signed-word saturation is specified in the
[Intel instruction manual, PMADDUBSW](https://cdrdv2-public.intel.com/868137/325462-089-sdm-vol-1-2abcd-3abcd-4.pdf).

## Exact model shapes and payloads

These values were computed from the actual pinned safetensors header by
[`describe_layout.py`](../experiments/quantization/describe_layout.py), without
loading/quantizing the tensor data. The full FP32 tensor payload is
**1,079,777,664 bytes**; the 11,800-byte safetensors header makes the file
1,079,789,464 bytes. The table's W4 column is group 64 with FP32 scales:
`N*K/2 + 4*N*(K/64)` bytes. All real K values divide 32/64/128/256 exactly.

| Weight, row-major [N,K] | Count | FP32 bytes each | W4 code bytes each | W4 scale bytes each | W4 total each |
|---|---:|---:|---:|---:|---:|
| QKV [2048,768] | 22 | 6,291,456 | 786,432 | 98,304 | 884,736 |
| WO [768,1024] | 22 | 3,145,728 | 393,216 | 49,152 | 442,368 |
| interleaved W13 [4608,768] | 22 | 14,155,776 | 1,769,472 | 221,184 | 1,990,656 |
| W2 [768,2304] | 22 | 7,077,888 | 884,736 | 110,592 | 995,328 |
| vocabulary output [65536,768] | 1 | 201,326,592 | 25,165,824 | 3,145,728 | 28,311,552 |
| token embedding [65536,768] | 1 | 201,326,592 | 25,165,824 | 3,145,728 | 28,311,552 |
| image projector [768,768] | 1 | 2,359,296 | 294,912 | 36,864 | 331,776 |

For the 88 transformer matrices, group-64 W4 replaces 674,758,656 bytes with
94,887,936 bytes. Keeping embeddings, output, projector and the 6,528 bytes of
other parameters in FP32 gives **499,906,944 bytes (476.75 MiB)** total: 2.16×
smaller payload, not 8×. Group 128 with FP32 scales gives 494,635,392 total bytes
(471.72 MiB); group 32 gives 510,450,048 (486.80 MiB). Group-64 W8/F32 scales
gives 584,251,776 bytes (557.19 MiB).

Also quantizing the output matrix with group-64 W4 would give 326,891,904 bytes
(311.75 MiB); quantizing every 2D tensor would give 151,849,344 (144.81 MiB).
Those are accounting scenarios, not recommended initial quality settings.
The output matrix is large and read for every generated token; test it separately
because quantization directly affects logit rankings. Embedding lookup touches
selected rows rather than streaming the whole matrix per token, so its storage
saving has a different latency implication. Embedding/output tensors are separate
checkpoint entries; do not assume weight tying.

Payload calculations exclude allocator/container metadata, alignment, ISA-specific
packing, temporary tiles, KV caches and any retained FP32 mapping. Keeping an
entire dequantized/phase-packed copy would defeat the intended resident-memory
reduction. The detailed report separates these exclusions from measured memory.

A full bit-pattern scan also rules out lossless BF16 storage of the original
FP32 checkpoint: 269,940,284 of 269,944,416 values have nonzero lower 16 bits,
and none of its 115 tensors is entirely BF16-representable. All values are
finite. This does not predict conversion error or OCR quality; it establishes
that BF16 weight storage changes values. The scan verifies the pinned checkpoint
hash before and after reading; see
[`checkpoint-bf16-storage-audit-v1.json`](../../../reference/checkpoint-bf16-storage-audit-v1.json).

## Experimental interface and kernel sequence

The implemented standalone `Q4Linear` has `quantize(weights, N, K, group_size)`,
`dequantize_row(row, output)`, `linear_f32(input, rows, output)` and
`payload_bytes()`. It uses finite weights, ties-to-even, independent K groups per
output channel, scale zero for all-zero groups, two's-complement [-7,7] codes,
and the low nibble for the earlier adjacent element. Odd rows/tails are padded
with zero codes and never share scales across rows. FP32 reconstruction occurs
before the FP32 FMA; changing this scale-placement arithmetic is another operator
candidate. It is not a proposed stable checkpoint ABI.

Next add an isolated `PackedQ4Linear` view with explicit format/group/scale dtype,
runtime backend, reusable scratch requirements, and row count. Store U8 codes
and F32 scales in a separate safetensors experiment file with original checkpoint
digest, quantizer source digest, tensor name/shape, packing/rounding version,
calibration digest and layer-selection policy. The production loader must not
silently accept it as the original F32 checkpoint.

1. AVX2 unpack/dequant into bounded tiles; keep F32 activations. Group 64/128 and
   output-channel tiles of 8/16 are initial candidates, not tuned winners.
2. Decode rows 1/2/4/8: load/unpack a weight tile once and reuse it across live
   rows. Preserve interleaved gate/up rows and all QKV head ordering.
3. Prefill: dequantize a bounded [K-block,N-block] panel and reuse it across a
   query-row tile. Avoid one full expanded weight copy per invocation. Compare
   against the current optimized F32 GEMM; decode savings may not transfer.
4. Add W8A32, then W8A8/VNNI and a pinned GGML/Candle operator as independently
   named candidates. Compare W4A16/BF16 or AOCL only with explicit activation
   conversion/output rounding, without borrowing the HF BF16 qualification label.

Keep RMSNorm, squared-ReLU gate arithmetic, rotary positions, sinks, softmax,
residuals and KV storage in the selected unquantized graph initially. FP32
full-cache/compact-cache correctness remains unchanged. Quantizing weights does
not reduce long-context KV memory.

## Calibration and gates before any promotion

Use only the **v3 64-page calibration lock**, never the v1 smoke or 200-page
evaluation split, for group/clipping/scale/layer choices. Include both image
prefill and free/teacher-prefix decode activations, especially W2 inputs after
the squared gate. Our absmax weight quantizer itself requires no data; selecting
exceptions or clipping from activation effects does. Start with per-layer
sensitivity on calibration pages and preserve FP32 for sensitive matrices.
Activation-aware scaling (AWQ) and activation-outlier redistribution
(SmoothQuant) are later candidates, not evidence that this architecture will
retain quality. Its unit RMSNorms and squared, interleaved gate require explicit
algebra/rounding validation before folding transformations into the graph.
[AWQ paper](https://arxiv.org/abs/2306.00978),
[SmoothQuant paper](https://arxiv.org/abs/2211.10438).

- **Format and arithmetic:** exact packing, row/group tails, code/sign convention,
  zero groups, nonfinite rejection, scale representation and overflow tests.
  Compare optimized kernels to independently dequantized FP64 sums and a scalar
  candidate; compare quantization error separately against original FP32 weights.
  Cover all four projection families plus output/image projection, real N/K and
  rows 1/2/4/8 plus realistic prefill sizes. Include allocation checks.
- **Frozen operator baseline:** use identical actual FP32 activations, exported
  independently reconstructed quantized weights and CPU/GPU candidate outputs.
  Set accumulation-error bounds from that independent baseline before judging
  SIMD variants. Existing FP32/BF16 model gates remain unchanged; quantization
  error is not a reason to widen either gate.
- **Model behavior and quality:** record same-prefix logit error/margins, argmax
  disagreements, independently free-running output IDs/text, EOS/caps and 8k/16k
  boundaries. Freeze a separate quantized quality budget before inspecting held-out
  200-page text/table/formula metrics and category regressions. No such acceptance
  budget or quantized model result is claimed here; assembled CER alone is not the
  official quality gate.
- **Performance and memory:** after accuracy gates and an exclusive timing
  window, measure packing/quantization/load cost, resident bytes, decode/prefill
  latency and batch 1/2/4/8 with cold/warm samples and source/binary/ISA/thread
  identity. Apply the plan's >=5% targeted improvement and <=5% regression rule
  only to measured comparable workloads. Native Windows and future bare-metal
  Linux need separate reports; WSL supplies functionality checks, not Linux
  bare-metal numbers.
