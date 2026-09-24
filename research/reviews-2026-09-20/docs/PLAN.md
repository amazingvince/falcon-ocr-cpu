# Implementation and acceptance plan

This file records the active goal. Completion requires the implementation,
qualification and reports below; a successful smoke example alone is insufficient.

User decision, 2026-09-20: continue on the available local machine and leave
bare-metal Linux performance measurements pending. Native Windows performance
and Linux functionality/parity under WSL remain in scope. A physical Linux
benchmark host is not a blocker for continuing the other stages.

## Model contract

Use the updated v1.5 checkpoint pinned in `reference/manifest.json`, the official
repository revision there, and explicit precision in every comparison. The older
paper's settings do not override the current model card's larger output/image
settings. Full-page plain OCR is the initial path, with no layout detector.

The actual checkpoint has 22 layers, hidden width 768, 16 query heads, 8 projected
KV heads, head dimension 64 and FFN width 2304. Query width is 1024, not the hidden
width. QKV weights are `[2048,768]`; FFN gate/up rows are interleaved in
`[4608,768]`, with `ReLU(gate)^2 * up`. Learned attention sinks contribute only
to the softmax denominator. Input and Q/K normalization have implicit unit
weights; the final learned normalization has a separate epsilon.

Preprocessing has two Pillow bicubic uint8 stages with truncation/ties-to-even
dimension rules. Preserve pixel/channel patch order and FP64-to-FP32 scaling.
Use the pinned tokenizer and image register/class/end markers. Prompt starts at
image class token 244, without a BOS. Plain extraction ends with OCR_PLAIN.
The attention mask is causal except the bidirectional image prefix interval.

Normalize Q/K before rotary positions and repeat K heads before spatial RoPE.
Every paired head's learned spatial frequencies differ: reducing all rotated
prefix keys to eight heads is incorrect. Temporal positions do not advance for
patches/registers/image-end markers. Nonpatch spatial positions are NaN.

## Delivery stages

1. Reproducible strict FP32 GPU export, same-prefix tensors, free-running output,
   fixed corpus and evaluation harness. Compare HF, the official engine and vLLM.
2. Correct Rust FP32 model, image preparation, library and CLI on Windows/Linux.
3. Measured FP32 performance: bounded tiled attention, reusable scratch, no decode
   allocations, packed weights, dedicated GEMV/GEMM/batch paths, last-position
   vocabulary projection, compatible fusion and runtime SIMD dispatch.
4. BF16 backend with explicit upstream dtype boundaries and its own qualification.
5. Lower-priority INT8/INT4 experiments reporting speed, memory and OCR quality
   separately. Quantization is not allowed to silently replace the baseline.

Public controls cover threads, precision, backend, batching, dimensions and output
budget. Reject unsupported choices clearly. Results include text, IDs, EOS/length,
dimensions, counts and timings. Immutable weights may be shared; runners own their
execution resources. Preserve input order and per-request EOS in mixed batches.

## Performance work

Benchmark batches 1/2/4/8 at realistic image and output lengths. Break out image
decode, resize/patch preparation, image projection, prefill and token decode.
Report first-token latency, total latency, throughput and peak/resident memory,
including cold load separately. Trace/export runs are not benchmarks.

Start with blocked FP32 attention and exact computation. Avoid allocating a
sequence-squared score matrix. Reuse buffers, precompute rotary factors and pack
weights once when measurements justify their memory cost. Keep one thread budget:
do not nest Rayon workers and a separate OpenMP thread team.

AVX-512 is a candidate, not a guaranteed win over AVX2 on Zen 4. Promote changes
only after repeatable >=5% improvement in the targeted primary workload without
>5% regression in the other primary workload. Record compiler flags, dependencies,
source hashes, CPU/OS, topology, threads, dimensions, context, repetitions and all
samples. Windows and bare-metal Linux are separate performance targets; WSL can
establish Linux functionality but cannot supply bare-metal Linux measurements.

Validate a cache layout with 16 prefix key heads, 8 generated-text key heads and
8 value heads before reducing memory. Preserve both paths for comparison until
cache tensors, logits and generated output agree within frozen gates. At FP32,
the initial fully expanded K/V cache costs 180224 bytes per cached token across
all 22 layers; this becomes material near 16k context.

## Rust and native candidates

| Component | Current choice / next experiment | Why and qualification |
|---|---|---|
| Weight/token assets | `safetensors`, `memmap2`, `sha2`, `tokenizers` | Read-only mapped weights and verified tokenizer contract |
| FP32 matrices | `gemm` plus model-specific `std::arch` GEMV | Measure actual 1/2/4/8-row and prefill shapes |
| Threading | `rayon` owned pool | Bound concurrent work and avoid nested pools |
| SIMD portability | `std::arch`; consider `pulp` | Explicit ISA guard and scalar oracle |
| Scratch | reusable Vec arenas; consider `dyn-stack` | Allocation-free hot path must be measured |
| BF16/quantized native GEMM | [AOCL-DLP](https://github.com/amd/aocl-dlp) via optional FFI | AMD-optimized BF16/INT8/INT4 APIs and reordered weights; check Windows/Linux builds and OpenMP policy |
| Images | `image` decode and Pillow-compatible resampling | Exact canonical pixels precede speed optimization |
| Resize candidate | [fast_image_resize](https://github.com/Cykooz/fast_image_resize) | SIMD RGB/RGBA support; qualify coefficients, rounding, alpha and mode semantics first |
| JPEG candidate | libjpeg-turbo / matching decoder | Pixel fidelity must be compared to the pinned Pillow decode |
| Measurement | operator harness, end-to-end runner harness; perf/AMD uProf/ETW | Separate compute, bandwidth, cache and scheduling limits |

AOCL-DLP documents FP32, BF16, INT8 and INT4 variants with architecture-specific
requirements, plus weight reorder APIs. It is a concrete optional backend to
benchmark, not an assumed speedup. Its OpenMP execution must share the runner's
thread budget. See [API docs](https://amd.github.io/aocl-dlp/api/gemm/index.html)
and [build instructions](https://github.com/amd/aocl-dlp/blob/dev/BUILD.md).

INT4 begins with groupwise weight-only quantization of large linear layers,
with FP32 accumulation and separate calibration data. Test group size, symmetric
versus asymmetric scaling, selective higher-precision layers and quantize/dequant
cost. BF16-weight/activation variants and INT8/VNNI may outperform INT4 depending
on batch size. Keep normalization, softmax and rotary arithmetic appropriately
precise. Do not extrapolate ideal weight-size reduction into tokens/second.

## Qualification gates

Freeze a 200-page diverse evaluation corpus with a 24-page smoke subset and
separate calibration data. Include plain paragraphs, tiny text, tables, multiple
columns, receipts, formulas, multilingual pages, low contrast/noise, rotated text,
blank/sparse pages and dense long output. Record provenance, licenses and expected
text. A dataset of partial facts is useful but does not replace full transcriptions.

Require exact discrete token/patch/layout/index behavior. Compare floating spatial
coordinates against independent FP32 reference-derived tolerances. Capture model
embeddings, Q/K/V, attention, hidden states, caches and last-position logits on
identical teacher prefixes. Disable TF32, including FlexAttention's own setting.
Freeze numerical tolerances from strict GPU/operator baselines before optimized
comparisons; do not widen them merely to pass a candidate. Require matching argmax
where the winner margin exceeds twice the measured maximum logit error, and
investigate near ties. Also compare free-running text and EOS independently.

Nonquantized modes may regress OCR quality by at most 0.25 percentage points
overall and 1 point per category on the fixed corpus. Report CER/WER and any
structure-specific metrics with normalization rules fixed in advance. Quantized
models get distinct accuracy/speed/memory reports and explicit acceptance decisions.

Exercise 8k output/16k context boundaries, exact caps, early EOS, empty/invalid
inputs, insufficient budget, mixed-batch stop lengths, caches, backend/platform
agreement and memory pressure. A toy fixture cannot establish long-context parity.

## GPU references

HF revision and official source are pinned in `reference/manifest.json`. The
inspected AMD64 serving image is
`ghcr.io/tiiuae/falcon-ocr@sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd`.
Run vLLM with an explicit dtype: serving documentation and the inspected entrypoint
have different defaults. Source inspection is not a completed serving comparison.

For BF16, account for the observed distinction that the official engine preserves
golden spatial frequencies in FP32 whereas HF `.to(BF16)` casts them. Select and
record the exact reference behavior before implementing a BF16 CPU target.
