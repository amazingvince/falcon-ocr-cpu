# CPU math and kernel review — 2026-09-20

The proposed solution is a model-specific Rust execution engine with separate
prefill and decode kernels, an explicit attention-cache layout, and replaceable
matrix backends. The current profile motivated two attention experiments.
Removing repeated accumulator traffic made the page slower; head-contiguous
prefix storage gave only a small improvement below the promotion threshold.
Shape-specific matrix-library comparisons are the next performance priority.
Quantized weights and caches offer a larger subsequent change, with
a separate accuracy contract. Proposed gains cannot be inferred from static code.

Execution update: the probability-staging experiment below has now been built
and measured. It preserved the checked CPU tensors and page outputs and achieved
the intended register allocation, but made the tested page 2.01–2.04% slower.
Retain the existing compact runner; see the
[staged-attention results](../experiments/attention64_staged/RESULTS-V1.md).
The new compact profile has now been captured and examined. It retains strict
audit/export failures, but its explicitly attributed samples identify QK and PV
as the dominant compact-head regions; see the
[compact profile results](../experiments/profiling/COMPACT-PROFILE-RESULTS-V1.md).
The subsequent head-contiguous prefix-cache experiment preserved its checked
tensors and outputs, but improved the full page by only 0.61–1.40%, missing the
5% target. It remains unpromoted; see the
[cache-layout results](../experiments/head_contiguous_prefix/RESULTS-V1.md).
The earlier canceled elevation attempt remains recorded separately.

This review inspects the current runner, preserved experiments, the pinned model
configuration and implementation, and current primary library sources. It does
not change inference code, dependencies, precision, or acceptance thresholds.
The implementation goal remains separate from this design review.

The upstream [model card](https://huggingface.co/tiiuae/Falcon-OCR),
[technical report](https://arxiv.org/abs/2603.27365), and
[official implementation](https://github.com/tiiuae/Falcon-Perception) describe
an early-fusion transformer processing image patches and text together. This
review keeps the pinned v1.5 assets and direct full-page OCR path. The published
vLLM throughput uses GPU serving under concurrency; it cannot predict CPU B1
latency. The server's BF16 default also differs from our strict FP32 reference.

## What the measurements actually establish

The combined compact-cache/fixed64-attention candidate completed a controlled
single-page bracket at 63.102 seconds, a 6.15–6.40% reduction against fresh compact
controls. Later unchanged controls ran around 62.8 seconds. A paired-output GEMV
experiment improved whole-page time by only 1.16–1.29%; split temporal-key storage
improved it by 3.23–3.32%, with about 141 MiB lower peak process memory. Both latter
experiments missed the existing 5% speed target. These are separate brackets;
their percentage improvements cannot be added.

The recent control stage medians were roughly 12.3 seconds prefill and 50.4
seconds decode. The original expanded-cache profile assigned 51.86% of sampled
CPU work to non-GEMM attention and 26.85% to linear projections. That trace has
documented audit/export limitations and does not measure the current compact
candidate. Sample shares are not wall-time fractions or DRAM measurements.

Evidence: [combined attention](../experiments/attention64_compact/RESULTS-V1.md),
[paired GEMV](../experiments/gemv_pair/RESULTS-V1.md),
[temporal storage](../experiments/attention64_temporal/RESULTS-V1.md),
[original profile](../experiments/profiling/FULLPAGE-RESULTS-V1.md).

The subsequent compact profile reproduced all three full-page outputs exactly.
Its XML diagnostic attributes 2,692,193 samples to the runner: 44.70% to
non-GEMM attention call paths, 31.50% to linear call paths, and 12.65% to
GEMM-attention paths. One exported sample has another process's root; the ETL
audit separately retains one sample after process exit, and the exporter exited
1. These failures remain explicit; matching aggregate sample totals do not
prove an event-level correspondence between the two representations.

An independent retained-ETLX instruction scan maps all 214 observed leaf RVAs
inside the pinned compact head to exact disassembly instruction starts. QK
regions contain 643,657 samples (23.91% of all in-bounds samples), and PV update
blocks contain 491,637 (18.26%). This supports an arithmetic-preserving layout
experiment. It does not prove memory stalls, bandwidth saturation, or the gain
from any change. See the [address analysis](../experiments/profiling/COMPACT-IP-RESULTS-V1.md).

The baseline runner has exact saved GPU output agreement on 200 pages. That
does not transfer automatically to each experimental backend. Ten intermediate
FP32 numerical comparisons remain outside their frozen bounds. Existing
experiments establish narrower operator/trace/output comparisons; arithmetic
changes still need their own validation.

A subsequent fixed CPU-prefix/GPU-suffix diagnostic reproduced 94 control
arrays and decomposed the layer-9 value error without changing any tolerance.
At the original failing coordinate, the embedding and block-0 conditional
terms were +0.002848 and +0.004446; block 1 canceled 0.001581, while the local
layer-9 RMS/QKV term was only +0.00000763. This directs the parallel numerical
investigation toward earlier stages. These are ordered substitutions through
a nonlinear suffix, not independent kernel error estimates or a correctness
fix. All ten failures remain open. See the [complete signed result and
review](../experiments/fp32_prefix_suffix_telescope/RESULTS-V1.md).

## The actual graph determines the useful kernels

Pinned dimensions are hidden width D=768, layers L=22, query heads H=16, KV heads
J=8, head width d=64, FFN width F=2304, and vocabulary V=65536. Query width is
1024, which differs from the residual width. Shapes below use row-major weights
`[output, input]` and one active token.

| Projection | Weight shape | Weights read by the mathematical operation |
| --- | --- | ---: |
| Combined Q/K/V | 2048 × 768 | 1,572,864 |
| Attention output | 768 × 1024 | 786,432 |
| Interleaved gate/up | 4608 × 768 | 3,538,944 |
| FFN down | 768 × 2304 | 1,769,472 |
| Vocabulary, once after 22 layers | 65536 × 768 | 50,331,648 |

One layer contains 7,667,712 projection weights. Each incremental token uses
219,021,312 projection weights including the vocabulary head: 876,085,248 bytes
at FP32, and approximately 438,042,624 floating-point operations when an FMA is
counted as two operations. The vocabulary head accounts for 22.98% of these
linear weights. The separate embedding table is a row lookup during decoding;
the whole 270M-parameter checkpoint is not multiplied for every token.

Prefill multiplies thousands of rows by these weights; decode multiplies one
row. For B1 GEMV, ignoring the small input/output vectors and cache reuse,
arithmetic intensity is about two FLOPs per four-byte weight, or 0.5 FLOP/byte.
Prefill reuses each weight across many rows. This is why one choice of library,
packing, and thread count need not win both phases.

Implementation anchors: [configuration](../artifacts/model/config.json),
[model graph](../src/model.rs), [operators](../src/kernels.rs).

For the measured page, prefix length P=6544 and emitted-token count T=1140.
Prefill emits the first token, leaving T−1=1139 incremental steps with cache
lengths P+1 through P+1139. Average attended length is therefore 7114.
Decode QK and PV together cost approximately

```text
4 × L × H × d × S = 90,112 × S FLOPs per incremental step
```

At S=7114 this is 641,056,768 FLOPs, before normalization, RoPE, softmax and
sinks. Together with the linear projections, it is about 1.079 GFLOPs per step.
These are analytic operation counts, not measured retired instructions.

Long output increases this cost: total incremental attention is proportional to
`(T−1)P + T(T−1)/2`. The second term grows quadratically with output length.
An implementation tuned only on 17-token smoke output misses this behavior.
For prefill, the dense all-pairs attention upper count is about 3.859 TFLOPs at
P=6544; the hybrid mask removes some work, while padded GEMM tiles can still
compute masked scores. Layer projections add about 2.208 TFLOPs. Exact executed
instruction counts require the actual mask, tiling and kernel paths.

## Cache structure is part of the mathematics

Let g(h)=floor(h/2) identify the shared KV head. After normalization, each key has
32 temporal and 32 spatial components. Temporal rotations are shared by the
paired query heads, but the checkpoint-stored spatial frequencies depend on h.
The pinned code registers those frequencies as a persistent buffer; it does
not establish that they are learned parameters. Thus:

```text
K_prefix[j,h] = concat(K_temporal[j,g(h)], K_spatial[j,h])
V[j,h]       = V_unique[j,g(h)]
```

Generated text has no spatial rotation, so its complete key can use eight unique
heads. Image-prefix spatial keys must retain sixteen head-specific versions.
Simply giving a generic GQA kernel eight rotated image keys is incorrect.
The tested temporal layout shares the first 32 components and retains all
sixteen spatial halves. It reduces storage without removing any dot products.

The current compact prefix uses `[token, head, 64]`. A worker following one head
reads 256 useful key bytes, then advances 4096 bytes to the next key; its value
stride is 2048 bytes. Each vector already uses whole cache lines. Changing this
layout is a locality/prefetch/TLB hypothesis, not a claim that the current loop
wastes most cache-line bytes.

For P prefix positions and G generated positions, compact FP32 KV payload is

```text
4 × L × d × [P(H+J) + G(2J)] bytes.
```

At P=6544 and G=570 (the average live generated length above), the unique payload
is 935,903,232 bytes. Independent head loops logically consume 1,282,113,536 KV
bytes per step because shared values, and shared generated keys, are read by
both query heads. Cache reuse may avoid some lower-level traffic. These figures
describe unique data and logical operands, respectively; neither is measured
DRAM traffic. Reserved capacity is different again: P6544/G4096 reserves
1,253,638,144 bytes of compact KV payload across the layers.

The idealized active-weight-plus-unique-KV payload at the average step is about
1.812 GB. This explains why weight compression alone does not address the entire
decode workload. It is not a bandwidth lower bound for every call: the hierarchy
can retain data, reload it, or duplicate it across cores.

## Preserve the distinction between algebra and floating-point execution

Attention scores are `s_j = dot(q,k_j)/8`. The sink is a learned scalar a per
head. In real arithmetic the output can be written as

```text
y = sum_j exp(s_j) v_j / (exp(a) + sum_j exp(s_j)).
```

The pinned reference instead computes attention over the real values, calculates
`LSE=logsumexp(s)`, and multiplies by `sigmoid(LSE−a)`. Those forms are
algebraically equal but have different FP32 rounding. Retain the reference
operation boundaries for the faithful implementation.

The existing online softmax already carries bounded per-head state `(m,l,u)`:

```text
m_new = max(m, max(scores_in_tile))
r     = exp(m − m_new)                         # 0 for the initial empty state
l_new = r*l + sum_j exp(s_j − m_new)
u_new = r*u + sum_j exp(s_j − m_new)*v_j
LSE   = m + log(l)
y     = (u/l) * sigmoid(LSE − a)
```

It avoids a sequence-squared score allocation today. The IO-aware tiling idea
in [FlashAttention](https://arxiv.org/abs/2205.14135) is relevant, but adopting
that name or its GPU package does not add an optimization we already have.
Prefill currently uses 32-query × 128-key tiles with serial GEMM inside outer
Rayon tasks. Decode uses one query and fixed64 vector operations.

The image's K/V tensors are already reused across output tokens. The remaining
scan depends on the new query: `softmax(q*K^T)*V` cannot generally be factored
into `q*(K^T*V)` because of the query-dependent exponential and normalization.
Linear attention, key pruning and patch removal would introduce model
approximations with separate OCR quality requirements. They are not algebraic
shortcuts for preserving this checkpoint's dense attention.

Changing tile boundaries, summation trees, FMA contraction, exponential
implementation, or dividing by a denominator versus multiplying by its
reciprocal can change numerical results. Splitting a long key sequence among
workers and merging online states is algebraically valid, but introduces a new
reduction order. Classify that as an arithmetic change.

The FFN also needs its exact graph: split *interleaved* gate/up outputs, compute
`relu(gate)^2 * up`, then multiply by W2 and add the residual. It is not a SiLU
gate. Input and Q/K RMSNorm use FP32 epsilon in this runner; final learned
normalization uses the configuration's 1e−5. Substituting a library's default
epsilon or a generic activation would change the model.

## Concrete attention experiments

**Smallest new kernel hypothesis: stage probabilities, then accumulate V in registers.**
The preserved compact candidate's assembly makes the opportunity concrete:
`0x1400b1a28` calls scalar `expf`; `0x1400b1a3b..0x1400b1ae8` then performs eight
256-bit value loads, eight FMAs reading the output accumulator from memory, and
eight output stores for each key. This is 256 output bytes read and 256 written
per key at the instruction level, mostly expected to be hot data. It is not
evidence of 512 extra DRAM bytes per key. The PDB's function extent and direct-call
symbols identify the actual function; objdump's nearest-export label is unrelated.
See the [preserved assembly](../artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/head-disassembly.txt)
and [symbol receipt](../artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/symbol-receipt.json).

After finding the tile maximum and rescaling, overwrite its existing 128-logit
buffer with the same scalar `expf` results, adding probabilities to the denominator
in exactly the previous order. Then run a separate PV loop with eight explicit
YMM output accumulators. Load the output once, process keys in the same order
using the same per-channel FMAs, and store once per tile. Keep the final division,
log, sigmoid and sink multiply unchanged. This removes no mathematical operation;
it offers the compiler a call-free PV region. It should need no additional tile
allocation. Assembly must confirm retained registers, and exact CPU comparisons
must establish the intended arithmetic before measuring page latency.
In particular, retain the separately rounded rescale multiply before the first
FMA, and preserve the normal floating-point environment and nonaliasing assumptions.
Moving operations also moves floating-point exception timing; no equivalence is
claimed for applications that observe trapping/exception order.

**First layout hypothesis: pack the immutable prefix by head and key tile.**
Use a layout such as `[head, tile, key_in_tile, 64]` for keys and the corresponding
eight-head value layout. Keep generated entries in append-friendly blocks.
Pack each layer once as prefill hands off to decode, then reuse it for every
generated token. Read keys in exactly the previous sequence order, keep the
128-key softmax tiles and dot/AXPY reduction order, and initially change no
precision or thread policy. Packing must be included in page latency. Reuse or
release construction buffers so a second permanent full KV copy does not hide
the cost. Prefill can retain its existing workspace/ordering for the first
isolated experiment.

Tile boundaries must follow absolute sequence positions across the
prefix/generated split. Here P=6544 is 16 positions into a 128-key tile; a later
full boundary tile contains 16 prefix keys and 112 generated keys. Restarting
online softmax tiles at the generated segment would change rounding despite
using the same nominal tile size.

This is an experiment on access pattern, with unchanged asymptotic work and
payload size. The current profile refresh should confirm whether attention is
still the right target. Record cycles, cache/TLB evidence when available, and
unprofiled page latency rather than inferring a gain from contiguous arrays.

**Second hypothesis: compute paired query heads together to reuse V loads.**
Each pair retains separate Q, spatial K, scores, maxima, denominators and output
accumulators. A loaded value vector can update two independent outputs. Temporal
key data can also be reused in the split representation. Preserve each head's
arithmetic order initially. The tradeoff is substantial: eight paired tasks
replace sixteen independent tasks, potentially leaving half the physical cores
without attention work. Compare appropriate eight- and sixteen-thread controls;
do not assume shared loads beat the lost parallelism. A split-key merge would
recover parallelism only by introducing a separate arithmetic experiment.

**Third hypothesis: improve tile kernels and keep more state in registers.**
Specialize the actual 64-wide operations and inspect generated assembly for
spill/reload traffic and scalar calls. Wider SIMD can help register use or
instruction count, but doubling vector width does not guarantee twice the
throughput on Zen 4. AMD describes Zen 4's 256-bit execution datapaths in its
[architecture discussion](https://www.amd.com/en/blogs/2025/leadership-hpc-performance-with-5th-generation-amd.html).
Compare AVX2 and AVX-512 with explicit CPUID/OS-state dispatch. An EVEX/AVX-512VL
candidate could keep 256-bit reduction groups while using additional registers;
whether LLVM actually does so is an assembly and benchmark question.

The existing [host feature probe](../reference/quantization-feasibility-layouts.json)
observed AVX-512BF16 and AVX-512VNNI, as well as F/BW/VL, on this Windows host.
It did not observe AVX-VNNI, AVX-512FP16 or AMX. These distinct feature names
matter when choosing integer/BF16 intrinsics; new deployments still need runtime
detection. Hardware availability does not qualify the BF16 graph's numerics.

Vector exponentials are another candidate if the updated profile supports it.
Unit-weight 64-wide Q/K RMSNorm bounds each real-arithmetic vector norm by 8;
orthogonal RoPE preserves that norm. Consequently real scores lie within
[-8,8] and ordinary online-softmax exponent arguments within [-16,0]. A guarded
specialization could exploit that range. Floating-point normalization/rotation
needs an error allowance, masked values need explicit handling, and the learned
sink's exponential is a separate operation. This argument neither proves
bitwise equality with scalar `expf` nor authorizes a faster approximate function
without numerical checks.

## Shape-specific matrix-library experiments

Keep a small backend boundary for projections and attention GEMMs. Backend
selection should be fixed when constructing a runner/plan, outside element
loops. Allocate scratch and pack immutable weights once where the backend needs
it. Preserve one bounded thread budget; a serial native call inside Rayon and
a threaded native call outside it are distinct scheduling choices.

Evaluate four full-page projection shapes at M=P, plus decode M=1 and relevant
small batches. Include vocabulary `[M,768] × [768,65536]` separately. Attention
uses QK shapes `[32,64] × [64,128]` and PV `[32,128] × [128,64]`, including tails.
Those small fixed shapes may benefit from specialized kernels even when a
library's large GEMM is excellent.

Our `gemm` 0.19.0 backend already provides optimized matrix multiplication and
explicit parallelism. Its scaling convention is
`dst = alpha*dst + beta*lhs*rhs`, opposite the common naming in BLAS. Any adapter
must preserve scalar placement, strides, and destination initialization.
[API documentation](https://docs.rs/gemm/0.19.0/gemm/fn.gemm.html).

The detailed current crate/native-library recommendations accompany this review
in [matrix and SIMD libraries](research/cpu-kernel-libraries-2026-09-20.md) and
[quantization, vector math and image libraries](research/cpu-quant-image-libraries-2026-09-20.md).
Those notes distinguish published versions from moving repository branches.
At the user's request, the project now pins Rust 1.94.0. Earlier performance
results remain Rust 1.92 evidence; new library probes compile their control and
candidate together under 1.94. A library's current `main` branch is not
automatically usable under our lockfile.

The short list is deliberately small:

| Candidate | Useful comparison | Main integration condition |
| --- | --- | --- |
| [`gemm` 0.19.0](https://docs.rs/gemm/0.19.0/gemm/) | Existing FP32 GEMM control | Already used; retain it while testing alternatives |
| [`faer` 0.24.4](https://docs.rs/faer/0.24.4/faer/) | Native Rust dense projections and GEMV | Adapt views/strides and explicit parallelism; recheck reduction behavior |
| [`rten-gemm` 0.26.0](https://docs.rs/rten-gemm/0.26.0/rten_gemm/) | Inference-oriented FP32/INT8/block-quantized kernels and reusable packing | Check compiler requirement, selected kernel, activation-compute mode and thread ownership |
| [`AOCL-BLAS`](https://github.com/amd/blis/releases) | Zen-oriented native FP32 GEMV/GEMM | Separate from prior AOCL-DLP trial; test exact release, layout and threading |
| [`oneDNN`](https://github.com/uxlfoundation/oneDNN/releases) | Native MatMul primitives and reduced-precision paths | Cache primitive/reorder setup; keep FP32 math mode explicit |
| [`LIBXSMM` 2.1.0](https://github.com/libxsmm/libxsmm/releases/tag/2.1.0) | Fixed small QK/PV tile kernels | Pin current API; validate generated code, scaling/transpose and native Windows support |

`pulp` is useful for implementing our own portable SIMD dispatch; it is already
present transitively at 0.22.3. Adding it as a direct dependency is an engineering
choice, not an automatic optimization. RTen's `rten-simd` and `rten-vecmath` are
additional reusable components. Their existence makes a small kernel comparison
more attractive than assuming a complete runtime migration is necessary.

Faer 0.24.4 declares Rust 1.84. Inspection of the published RTen 0.26.0 packages
confirmed that mandatory `rten-simd` requires Rust 1.94, even though the GEMM
manifest omits that floor. The user-authorized compiler update enables this
candidate. See the [released-source adapter review](research/fp32-matrix-adapter-2026-09-20.md)
for exact API, precision, layout and threading constraints; dependency resolution
alone is not a successful build or performance result.

A previous AOCL-DLP prefill-W2 intervention increased failed intermediate checks
from 10 to 47 while retaining all 17 smoke argmax results. It was not a speed
comparison. This does not disqualify every native library, but demonstrates why
local dot accuracy and short output equality are insufficient evidence for a
whole-graph backend change. See the
[intervention report](../reference/aocl-w2-intervention-summary-v1.json).

## Quantization: separate weights, activations, and caches

Weight-only W4A32 stores blockwise integers and scales, reconstructs a small
working tile, and accumulates in FP32. W4A8 additionally quantizes activations
and can use integer dot-product instructions. These are different error and
performance tradeoffs. BF16 storage with FP32 arithmetic is different again
from BF16 dot products and from reproducing the GPU's BF16 graph boundaries.
Avoid full-matrix dequantization on every token: it defeats the traffic saving.

An illustrative symmetric group-64 format with one FP32 scale uses
`0.5 + 4/64 = 0.5625` bytes per weight before padding/metadata. Applied to the
219,021,312 active linear weights it would occupy about 123.2 MB instead of
876.1 MB. Keeping FP32 KV unchanged leaves about 1.059 GB of idealized active
weights plus unique KV, compared with 1.812 GB originally: only a 1.71× payload
ratio despite the roughly 7.1× weight reduction. Neither ratio predicts runtime;
unpacking, scaling, accumulation and cache behavior matter.

KV compression deserves its own experiment because every token repeatedly
reads the image prefix. Start with a conservative storage format before more
aggressive KV INT4. Keys and values can require different quantizers. The
[KIVI paper](https://arxiv.org/abs/2402.02750) studies this asymmetry for other
language models; its reported quality/speed results are not Falcon-OCR results.
Calibrate using representative OCR documents, retain independent evaluation
pages, and examine generated long-context caches as well as prefill tensors.

For attention sensitivity, a perturbed query/key gives
`delta_s = (q·delta_k + delta_q·k + delta_q·delta_k)/8`.
Softmax then changes every weight, and large values can magnify a small weight
error. Squared-ReLU gating and later residual blocks can amplify earlier errors
further. Qualify layer outputs, teacher-forced logits and free-running OCR,
including near ties and long output. Weight quantization, activation quantization,
KV quantization and vector-exp changes should begin as separate experiments.

Squared ReLU offers another model-specific possibility: nonpositive gates create
zero FFN activations for finite up values. Sparse W2 multiplication could skip
their contributions. W2 accounts for 38,928,384 weights across the layers, about
17.8% of the active linear-weight count, so the opportunity is bounded even
before accounting for attention. Actual activation sparsity has not been
measured in this review. Element skips in the current row-major matrix may save
little cache traffic; a column/block layout could expose useful contiguous
loads at the cost of indexing and a different accumulation schedule. Treat
sparsity-aware W2 as a later measured experiment, with zero/sign and reduction
semantics checked explicitly.

## Execution structure and lower-priority work

The runner can own a resolved plan resembling:

```text
Model: mapped source weights + optional shared backend-specific packed weights
Runner: selected kernels + bounded worker pool + reusable scratch
Session: immutable prefix KV + append-only generated KV + positions + stop state
Prefill: exact image preparation -> large projections -> tiled hybrid attention
Decode: B1 projections -> fixed64 cache scan -> FFN -> final norm -> full-vocab argmax
```

Use whole-operator dispatch, not per-element virtual calls. The diagnostic path
can continue exposing logits/tensors; the ordinary greedy path does not need a
vocabulary softmax. All vocabulary scores still need evaluation for the current
exact greedy algorithm. Removing most vocabulary rows would require a separate
provably safe pruning method or a quality-changing approximation.

Topology-aware scheduling is worth measuring: the 7950X has two CCDs, each with
its own L3. Stable ownership may improve locality, but pinning every Rayon
worker without considering work stealing is not sufficient. AMD's
[Ryzen optimization guide](https://gpuopen.com/gdc-presentations/2024/GDC2024_AMD_Ryzen_Processor_Software_Optimization.pdf)
describes this topology. [`core_affinity`](https://docs.rs/core_affinity/0.8.3/core_affinity/)
can pin threads, while [`hwlocality`](https://docs.rs/hwlocality/latest/hwlocality/)
provides richer topology/binding facilities with native hwloc integration.
Logical CPU IDs must not be assumed to encode physical core/CCD placement.

Image preparation was about 54–61 milliseconds in the existing full-page
measurement, compared with tens of seconds of inference. Optimizing it cannot
produce a 5% warm-page improvement on that workload. Keep the exact Pillow-like
resize contract; use fast image crates only after checking pixel rounding,
antialiasing, alpha and two-stage behavior. Image-file decoding and model loading
are excluded from that warm recognition benchmark and should be timed separately
for cold CLI startup.

Image *resolution* has a much larger effect than resize implementation: patch
count scales approximately with pixel area, dense image prefill with its square,
and prefix attention during decode with patch count times output length.
Reducing each image dimension by a factor r therefore scales these leading
terms roughly as r^4 and r^2, respectively. This also changes the model input
and can lose small text. It is an explicit quality/performance setting, not a
parity-preserving kernel optimization or an automatic way to fit context.

Changing the allocator cannot materially improve a decode path already checked
to allocate zero times. Huge pages, software prefetch and custom schedulers need
an observed translation/cache/synchronization problem first. A model-wide tensor
framework migration would also require reimplementing this model's hybrid mask,
head-specific spatial rotations and sink semantics; individual reusable kernels
can be evaluated at a much smaller boundary.

## Recommended order

1. Retain the compact/fixed64 control. The profile is captured with explicit
   audit/export limitations. Scalar-probability staging was about 2% slower;
   head-contiguous prefix storage improved the page by only 0.61–1.40%.
   Both preserved their checked outputs, but neither met the promotion target.
2. Compare a short matrix-backend shortlist on actual operands/shapes, then
   integrate one winner in isolation. Keep prefill and B1 choices independent;
   charge packing to the appropriate loading or page interval and validate any
   changed reduction order before full-page timing.
3. Use hardware counters or a bounded kernel experiment to distinguish compute,
   cache/TLB, bandwidth and scheduling limits before choosing further attention
   changes. Paired-head reuse and vector math remain separate hypotheses.
4. Qualify reduced-precision storage and quantized linear kernels; evaluate KV
   compression independently before combining successful candidates.

Use the existing operator, whole-graph, output and benchmark infrastructure.
The ten numerical discrepancies remain a parallel investigation. Broader corpus,
long-output, mixed-request and cross-platform checks remain required before
promotion; bare-metal Linux timing is still deferred as the user requested.

The companion [mathematical derivation](research/cpu-model-math-2026-09-20.md)
adds source-level graph details and transformation constraints.
