# Falcon-OCR CPU mathematics and execution opportunities

This source review concerns the pinned v1.5 checkpoint and the direct, single-image OCR path. It derives work and storage counts; it does not measure performance, execute a model, change a numerical bound, or qualify a new backend. The practical priority is attention scheduling/layout experiments that preserve arithmetic, followed by shape-specific matrix work. Quantization and vector transcendental functions need separate numerical qualification.

The model revision is `fe757d59ecd79d4d68760162306a70a015761ad9`; the weights SHA256 is `3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16`. The [configuration][config] is authoritative: residual width 768, 22 layers, 16 query heads, eight KV heads, head width 64, FFN width 2304, and vocabulary 65536. Query width is **1024**, not 768. The paper explains the early-fusion architecture and image/text attention distinction, but its general description does not replace this checkpoint's code and processor. [Falcon Perception technical report](https://arxiv.org/html/2603.27365v1).

## Exact graph and its consequences

Write the unit-affine RMS normalization of an n-vector as

\[
R_\epsilon(x)=x/r,\qquad r=\sqrt{\frac1n\sum_i x_i^2+\epsilon}.
\]

For a block input X, the pinned [attention implementation][up-attn] computes

\[
[Q_0,K_0,V_0]=R_\epsilon(X)W_{qkv}^{T},\quad
Q=R_\epsilon(Q_0),\quad K=R_\epsilon(K_0).
\]

The first normalization is per residual row of width 768. Q/K normalization is separately per head of width 64, with **no learned affine scale**; V is not normalized. Q/K/V widths before repeating are 1024/512/512. These functional normalizations omit `eps`; the pinned strict-FP32 runtime uses FP32 machine epsilon. This is distinct from the final learned normalization's explicit `1e-5`. The [Rust implementation][rust-forward] makes that distinction explicit. Its [RMS reduction][rust-rms] uses its own FP32 summation tree and `sqrt().recip()`, which need not reproduce CUDA's reduction and reciprocal-square-root bits.

After attention, with concatenation of all 16 output heads,

\[
U=X+O W_o^T,\qquad
[g,u]_{\mathrm{interleaved}}=R_\epsilon(U)W_{13}^{T},
\]
\[
a_i=\max(g_i,0)^2 u_i,\qquad X'=U+aW_2^T.
\]

The [gate kernel][up-ffn] reads alternating gate/up channels: `gate, up, gate, up, ...`; it is squared ReLU, not SiLU or a split of the matrix into two halves. It selects zero when `gate > 0` is false, then evaluates the two products. The [block][up-block] contains two separate residual additions. FP32 fusion/reassociation can change their result; BF16 introduces further actual cast boundaries and must be treated as a separate graph.

After layer 21, the final normalization is `R_1e-5(x) * gamma`; a separate, untied `[65536,768]` weight matrix produces logits. Greedy decoding needs argmax, not vocabulary softmax. It still requires the current full-vocabulary projection. The Rust path already projects only the final prefill row, a major existing saving compared with evaluating all prefill vocabulary logits. Removing the positive final RMS scale preserves real-arithmetic argmax, but moving it through gamma and the projection changes FP rounding and fails to preserve diagnostic logits; a 768-element norm is a poor target for that tradeoff. [Final norm and output definitions][up-final]; [Rust final-row projection][rust-final].

Image input has no separate deep vision encoder: each RGB 16×16 patch is a 768-vector, multiplied by a bias-free `[768,768]` projector. Its result replaces the image-token embedding. Image resize, channel order, byte conversion, patch order and normalization remain part of the model input contract. For instance, folding `(u/255 - 0.5)/0.5` to `2u/255 - 1` is algebraic equivalence, not an FP32 bit-identity argument. [Image scatter/projector][up-image].

## RoPE explains what can and cannot be shared

The first 32 channels of each 64-vector use temporal RoPE; the last 32 use spatial RoPE only for image patch positions. For a pair `(a,b)`, rotation by theta is `(a cos(theta) - b sin(theta), a sin(theta) + b cos(theta))`. In real arithmetic this is orthogonal.

Temporal pair j, for j=0…15, uses

\[
\omega_j=10000^{-2j/32},\qquad \theta_j=t\omega_j.
\]

The denominator is **32**, the temporal half-width. The frequency table is computed in the pinned code on CPU as complex64 and subsequently moved to the model device. Regenerating it using another `pow`/`sin` implementation need not reproduce its bits. [Temporal frequencies and multiplication][rope-temporal]; [Rust factors][rust-rope].

The spatial angle is

\[
\theta_{h,j}=p_y G_{h,j,0}+p_x G_{h,j,1}.
\]

G is the checkpoint-stored `[16,16,2]` golden-frequency **buffer**, shared across layers and specific to the query head. The local source does not establish that it is a learned parameter. Use those saved values; do not regenerate them from a generic formula. [Spatial projection and rotation][rope-spatial]; [buffer registration][up-final].

The processor's grid coordinates matter: for a grid W patches wide and H high, `xlim=sqrt(W/H)` and `ylim=sqrt(H/W)`. Coordinates are inclusive linspaces on `[-xlim,xlim]` and `[-ylim,ylim]`, flattened in patch order. They are not normalized by `max(W,H)`. Only patch tokens get finite spatial coordinates. Registers, class/end markers and text have NaN spatial coordinates and bypass spatial rotation. Temporal positions advance for the image class token, but **not** for patches, four image registers, or the image end token. Thus the image patches share one temporal position; replacing it with their flattened sequence indices changes the checkpoint's model. [Processor positions][positions].

Let `g(h)=floor(h/2)`. The [upstream repetition][up-repeat] duplicates eight normalized K heads into adjacent pairs **before** head-specific spatial rotation. Therefore:

\[
K_{\mathrm{prefix},h}=[K^t_{g(h)},K^{xy}_{h}],\qquad
V_h=V_{g(h)}.
\]

Prefix temporal K halves and all V vectors can be shared. Rotated image spatial K halves generally cannot. Generated text has no spatial rotation, so its complete K vectors can be shared. Feeding only eight rotated image K heads to an ordinary GQA kernel is incorrect. Conventional GQA motivates sharing KV, but this extra spatial rotation changes the sharing boundary. [GQA paper](https://arxiv.org/abs/2305.13245).

The [existing compact cache][rust-cache] stores prefix K at 16 heads, all V at eight heads, and generated K at eight heads. The tested temporal-prefix layout further shares its first half. Its per-token payload is:

| Representation | Prefix FP32 values | Generated FP32 values |
| --- | ---: | ---: |
| Fully expanded K and V | 2048 | 2048 |
| Compact K/V | 1536 | 1024 |
| Shared temporal prefix K | 1280 | 1024 |

The last row is `8*32 + 16*32 + 8*64`. It saves 16⅔% of compact prefix KV bytes, or 25% of prefix K bytes. It does not remove dot products. The existing temporal-layout benchmark missed its speed target despite exact functional comparisons; storage arithmetic is not a new performance result. [Temporal experiment results][temporal-results].

For the current AVX2 dot, the first and second 32-channel halves update the **same four eight-lane accumulators**, followed by a fixed reduction. Computing two independently reduced half-dots and adding them changes rounding. A split storage layout can preserve the original sequence of FMAs while loading from two addresses. [AVX2 dot][rust-dot].

## Hybrid attention, sinks and stable reduction

For the supported single-image, unpadded document, visible keys satisfy

\[
j\le i\quad\text{or}\quad i,j\in[a,b),
\]

where a is the image-class position and b is the image-end position. The image class, registers and patches are mutually visible. The OR with causal attention also permits preceding text/BOS; it is not image-only attention. Packed documents and left padding require the additional native conditions. [Pinned mask composition][mask].

For one head and a fixed query, let `s_j=q·k_j/8`, excluding masked keys. The native computation obtains sink-free attention and its natural log-sum-exp:

\[
Z=\sum_j e^{s_j},\quad L=\log Z,\quad o_0=Z^{-1}\sum_j e^{s_j}v_j.
\]
\[
o=\sigma(L-a_h)o_0
 =\frac{\sum_j e^{s_j}v_j}{e^{a_h}+\sum_j e^{s_j}}.
\]

The learned sink acts like an additional logit with a zero value vector. The right-hand equation explains the model, but its direct implementation would alter FP rounding. Retain the observed sink-free output, LSE, sigmoid and multiply sequence; the sink's logit is not a regular cached token. [Native sink operation][up-attn].

The [existing CPU kernels][rust-attn] already implement an online softmax. For each key tile, with running maximum m, denominator l and vector numerator u:

\[
m'=\max(m,\max_j s_j),\quad \alpha=e^{m-m'},
\]
\[
p_j=e^{s_j-m'},\quad l'=\alpha l+\sum_j p_j,\quad
u'=\alpha u+\sum_j p_jv_j.
\]

The first empty state uses alpha=0 explicitly. At the end, `L=m+log(l)` and `o=(u/l)*sigmoid(L-a)`. This avoids storing an S×S matrix. Prefill already uses 32-query × 128-key tiles and GEMMs with outer Rayon scheduling. It is not awaiting a first implementation of tiled attention. [Online normalizer](https://arxiv.org/html/1805.02867v1); [FlashAttention](https://arxiv.org/abs/2205.14135).

Online-state merging is associative over real numbers; finite-precision merging is order-dependent. Changing key tiles, splitting keys between workers, moving denominator scaling, substituting reciprocal multiplication, reassociating the PV accumulation, or replacing transcendental functions requires numerical validation. Query tiling can also change a selected GEMM implementation, even if the symbolic operation stays the same. GPU FlashAttention performance results do not predict CPU performance. Work partitioning is a separate problem from the mathematical identity. [FlashAttention-2](https://arxiv.org/abs/2307.08691).

### A useful, qualified bound for exponentials

The unit-affine width-64 RMS norm implies, in real arithmetic,

\[
\|R_\epsilon(x)\|_2^2
 =\frac{64\sum_i x_i^2}{\sum_i x_i^2+64\epsilon}\le64.
\]

Orthogonal RoPE preserves this bound, so Cauchy–Schwarz gives `|q·k/8|≤8`. For finite visible scores, ordinary online probability arguments and noninitial rescaling arguments lie in `[-16,0]`. This gives a model-specific reason to investigate a guarded vector exponential on a limited interval.

It does **not** prove that actual FP32 inputs obey those exact endpoints: reduction error, approximate CUDA reciprocal square root, trigonometric rounding, complex multiplication and dot products need an error allowance or an actual bound. Masked `-inf`, the first tile and nonfinite inputs require explicit handling/fallback. The learned-sink exponential `exp(a-L)` is outside this interval argument and must retain its separate route. A fast interval approximation is not bit-equivalent to scalar libm, even if its relative error is tiny. Keep scalar exp/log comparisons and unchanged acceptance gates; classify this as an arithmetic candidate, not a free layout change.

## Work, operand loads and stored bytes

Count an FMA as two FLOPs. Ignore small normalizations, elementwise gates, RoPE, nonlinear functions, addresses and reductions in the dense matrix count.

| Matrix | Shape | Weights per layer |
| --- | --- | ---: |
| QKV | 2048×768 | 1,572,864 |
| Attention output | 768×1024 | 786,432 |
| Interleaved gate/up | 4608×768 | 3,538,944 |
| FFN down | 768×2304 | 1,769,472 |
| Total | | 7,667,712 |

Across 22 layers plus the vocabulary head, one incremental token applies **219,021,312 weights**, or **438,042,624 FLOPs**. A hypothetical single read of every active FP32 weight is **876,085,248 bytes = 835.5 MiB**. The vocabulary head alone is 192 MiB, 22.98% of that weight count. The embedding table is a lookup, not another full matrix multiplication. The patch projector is prefill-only. These are analytic operands, not measured DRAM bytes: the memory hierarchy may retain data or reload it.

QK and PV together cost `4*22*16*64*S = 90,112*S` FLOPs per incremental step. For the measured shape P=6544 prefix tokens and T=1140 emitted tokens, prefill emits the first token; there are 1139 incremental calls, attending 6545…7683 positions. Their mean S is **7114**, with mean generated-cache length G=570. Therefore:

| Quantity at the average incremental step | Amount |
| --- | ---: |
| Attention QK+PV | 641,056,768 FLOPs |
| Dense linear + attention leading work | 1,079,099,392 FLOPs |
| Logical KV operand loads, independent head loops | 1,282,113,536 bytes / 1222.71875 MiB |
| Unique compact KV payload across 22 layers | 935,903,232 bytes / 892.546875 MiB |
| Unique temporal-shared KV payload | 788,480,000 bytes / 751.953125 MiB |

The logical-load expression is `22*16*(64+64)*4*S`; each head consumes K and V. Unique compact payload is `22*4*(1536*P+1024*G)`. Temporal sharing replaces 1536 by 1280, saving **140.59375 MiB** at this P. Logical operand loads, unique live storage, reserved capacity, cache-line transactions and DRAM traffic are distinct quantities. None of this table measures the last two.

Across an output of T tokens, attention work is proportional to `(T-1)P+T(T-1)/2`, not simply P*T. For prefill, let I be the number of positions inside the image's bidirectional interval. The visible-pair count is

\[
M=P(P+1)/2+I(I-1)/2.
\]

For the existing plain prompt, P includes 16 nonpatch positions and I includes the class plus four registers and all patches. At P=6544, patches=6528 and I=6533: **M=42,752,018**. Attention's useful leading work is then `90,112*M = 3,852,469,846,016` FLOPs, about 3.852 TFLOPs. Dense layer projections add 2.208 TFLOPs, patch projection about 7.701 GFLOPs, and a final-row vocabulary projection 0.101 GFLOPs. Tiled kernels can compute additional masked/padded lanes; these are useful mathematical work counts, not instruction counts. Different prompt tokens or image arrangements change I and the formula's applicability.

## CPU transformations worth testing

### 1. Head-major, blocked immutable prefix

Use prefix K `[head, key_tile, key, 64]` and V `[kv_head, key_tile, key, 64]`, or an equivalent aligned layout. Keep the 16 existing head tasks, score order, 128-key softmax boundaries, each dot's accumulator tree and PV key order. A worker currently consumes 256 K bytes then advances **4096 bytes**; its V stride is **2048 bytes**. Head-major storage makes those per-head vectors contiguous. On ordinary 4-KiB pages, 6544 stride-4096 accesses potentially touch 6544 pages for one head, versus roughly 409 pages for contiguous 256-byte vectors, before alignment/tails and huge-page effects.

This is a locality, prefetch and translation hypothesis. Each existing vector uses complete cache lines, and all heads collectively use the other vectors. The layout does not itself reduce unique payload or mathematical operand bytes. Shared caches, concurrent head traversal and translation behavior decide the benefit; no DRAM saving or speedup is established.

Copy the immutable prefix once per layer at the prefill/decode handoff, ideally replacing the existing cache copy rather than retaining an extra permanent full cache. Charge packing to page latency. Keep existing prefill workspaces and arithmetic initially. Keep generated K/V append-friendly and preserve the original softmax tile across the prefix/tail boundary; making that storage boundary a new numerical tile changes results. Require byte reconstruction of K/V plus same-input attention, whole-trace and output comparisons before timing.

### 2. Reuse V across paired query heads, with explicit scheduling costs

The paired heads have different Q, scores, maxima, denominators and outputs. One loaded V vector can feed both output accumulations. Two independent prefix-head loops logically load 1024 bytes per key (two K and two V vectors); sharing V reduces this to 768 bytes, **25% fewer logical loads**. Sharing the common temporal K half as well gives 640 bytes, **37.5% fewer**. For generated text, full K sharing plus V sharing gives 512 bytes. These are load-instruction opportunities, not corresponding DRAM reductions: the old pair may already hit shared cache data.

QK and PV arithmetic still occur separately for both heads. Pairing reduces 16 head tasks to eight, potentially reducing B1 utilization on a 16-core CPU. B2 supplies more independent tasks but is a different workload. Do not assume that shared data compensates for lost parallelism.

Register pressure also matters. Two 64-element FP32 outputs consume all sixteen AVX2 YMM registers before probabilities, values or temporary state. A possible exact-order schedule computes probabilities into a small tile buffer, then processes 32 output channels for each head, using eight accumulator registers, two broadcasts and a value temporary. It must retain the same per-channel key/FMA sequence and tile rescaling. This adds loop/buffer traffic and is a hypothesis, not an accepted kernel. The QK temporal loads may be shared while maintaining two independent four-accumulator dot trees; do not sum independent temporal/spatial half-dots.

### 3. Separate scalar transcendental calls from register-resident PV

The current online loop interleaves scalar exp calls with 64-wide output updates. ABI clobbers and live state can force spills. Stage the **same** probability results and denominator additions in a 128-float tile, then execute PV with the original key-order FMAs. A 512-byte per-head probability buffer is small compared with the tile's K/V operands. This may let the compiler retain output channels without changing exp implementation. Generated assembly must establish the actual effect; no claim is made that current calls necessarily spill or that staging wins. Preserve denominator addition order, rescaling and final division/sink order.

The root's subsequent review of the [preserved compact-head assembly](../../../../artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/head-disassembly.txt) confirms eight memory-source output FMAs and eight output stores per key at `0x1400b1a3b..0x1400b1ae8`, after the scalar exp call at `0x1400b1a28`. This establishes actual per-key output read/write instructions, without attributing them uniquely to ABI spills or claiming DRAM traffic. The existing logits buffer can be overwritten by probabilities after its maximum has been found, avoiding a second 512-byte buffer. Explicit eight-register PV accumulation is a small concrete candidate; a newly compiled candidate must establish that those registers remain live across the call-free key loop.

### 4. Matrix and prefill changes should follow actual shapes

Prefill reuses weights over thousands of rows; B1 decode streams large matrices for one row. They need separate dispatch/packing choices. Already-tiled QK `[32,64]*[64,128]` and PV `[32,128]*[128,64]` are quite different from W13, W2 and the vocabulary matrix. A cache layout can be exact in stored values while a new GEMM backend changes reduction order. Keep those experiments separate.

Normalization of K before repetition offers a small exact-work saving: the native graph normalizes eight width-64 heads, while the current expanded Rust workspace repeats first and normalizes identical copies. Normalizing once and copying can preserve those bits, but this is tiny relative to the large matrices and attention scan. Likewise, allocator replacement has little to offer the already allocation-free warmed decode interval.

### 5. Squared-ReLU sparsity is a conditional W2 opportunity

For finite gate/up values, a nonpositive gate produces a zero activation (its sign can depend on the up value). Skipping zero columns of W2 could avoid useful work. Across 22 layers, W2 contains **38,928,384 weights**, or **17.7738%** of active linear weights; attention and the other projections remain. Actual activation sparsity and contiguous zero-block patterns have **not been measured** here.

Elementwise skips in row-major weights may save few cache-line fetches while adding branches. A column/block layout could stream selected output-column vectors, at the cost of dynamic indexing, packing and scheduling. Preserving the existing accumulation tree is harder than retaining the mathematical sum: zero FMAs can affect signed-zero results, and moving nonzero contributions between lanes or four accumulators changes rounding. NaN/infinite exceptional operands also invalidate a blanket zero-product simplification. Treat this as a measured-sparsity-dependent later candidate, not a replacement for the dense kernel by inspection.

## Quantization and numerical sensitivity

Weight-only int4 reduces traffic, but Rust and the CPU ISA still need unpacking, scale application and an efficient dot path. For illustration only, symmetric groups of 64 with one FP16 scale require `0.5+2/64=0.53125` bytes/weight before padding and metadata: **116,355,072 bytes (110.96484375 MiB)** for the active linear matrices, versus 835.5 MiB FP32. FP32 scales give a different format/cost. Full-matrix dequantization on every token forfeits much of this advantage. KV compression is an independent lever because the unique prefix cache is comparable to the active linear weight payload at this workload.

For perturbed Q/K with Euclidean errors bounded by eta_q and eta_k, the real norm bound gives

\[
|\delta s|\le\eta_q+\eta_k+\eta_q\eta_k/8.
\]

For softmax p, its first-order perturbation is

\[
\delta p_i=p_i(\delta s_i-\sum_jp_j\delta s_j).
\]

Consequently `||delta p||_1 <= 2||delta s||_infinity` to first order is a conservative bound. Include the sink as an additional zero-value category, with an unchanged sink perturbation of zero. Attention-output error additionally depends on value magnitudes and value quantization; a conservative first-order term is `2*max_j ||v_j|| * ||delta s||_infinity`, plus direct value error. This is a local sensitivity argument, not an OCR-quality guarantee.

Squared-ReLU's positive branch has derivatives `2*g*u` with respect to g and `g^2` with respect to u. Large values can amplify small incoming changes. RMSNorm's Jacobian is `I/r - xx^T/(n*r^3)`; its sensitivity also depends on the input. Later projections, residuals and near-tied logits prevent a simple local quantization bound from proving free-running token identity. Calibrate weight, activation and KV formats separately on the calibration split, then retain independent corpus, same-prefix logit, greedy-output and long-context checks. BF16 storage, BF16 dot operands with FP32 accumulation, and the native BF16 graph are different contracts.

## What the remaining FP32 discrepancies mean

The real equations are coherent; the relevant differences are in their floating-point execution. CUDA RMS reductions/`rsqrtf`, CPU sum trees/`sqrt().recip()`, cuBLAS projection partitioning, FMA contraction, transcendental implementations and operation boundaries all matter. Higher accuracy against an F64 oracle does not guarantee closer agreement with the pinned GPU. Exact GPU emulation may be possible at a particular boundary but is not automatically the best CPU engine.

The saved 200-page output agreement is strong evidence for the tested FP32 path, while ten frozen intermediate comparisons remain outside bounds. Neither statement cancels the other. Existing causal crossovers show that most of the selected layer-9 V discrepancy arrives in earlier state; the downstream terms can cancel each other. That evidence does not prove a specific kernel bug or justify relaxing a threshold. [Downstream decomposition][downstream]; [observed RMS reference][rms-observed]; [rejected exact-rsqrt intervention][rms-intervention].

The next exact-preserving performance candidate should therefore change data placement and instruction scheduling while retaining arithmetic. Changes to reduction trees, vector math, precision or the mathematical input need their own isolated evidence. The broader execution-engine recommendations and measured-result limitations are in the [CPU kernel review][review].

[config]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/config.json
[up-repeat]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/modeling_falcon_ocr.py:118
[up-attn]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/modeling_falcon_ocr.py:139
[up-ffn]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/modeling_falcon_ocr.py:185
[up-block]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/modeling_falcon_ocr.py:248
[up-final]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/modeling_falcon_ocr.py:339
[up-image]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/modeling_falcon_ocr.py:407
[rope-temporal]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/rope.py:5
[rope-spatial]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/rope.py:58
[positions]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/processing_falcon_ocr.py:240
[mask]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/model/attention.py:103
[rust-forward]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/model.rs:267
[rust-final]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/model.rs:441
[rust-cache]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/model.rs:737
[rust-rope]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/model.rs:943
[rust-rms]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/kernels.rs:189
[rust-dot]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/kernels.rs:1007
[rust-attn]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/src/kernels.rs:307
[temporal-results]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/experiments/attention64_temporal/RESULTS-V1.md
[downstream]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/reference/fp32-crossover-downstream-decomposition-v1.md
[rms-observed]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/reference/rms-observed-rstd-v1.json
[rms-intervention]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/reference/rms-gpu-rsqrt-width768-intervention-v1.json
[review]: C:/Users/amazi/Documents/ChatGPT/falcon-ocr/docs/CPU_MATH_AND_KERNEL_REVIEW.md
