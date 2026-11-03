# Modes

Status: current as of 2026-11-03. Measurements: Ryzen 9 7950X (16 cores,
32 threads, DDR5), Windows 11, the journal benchmark page (6,544 input tokens),
unless a section names the Ryzen 7 7700X (8 cores, 16 threads), Linux.

FP32 is 32-bit floating point; KV is the attention key/value cache. GPTQ
quantizes weights using calibration activations.

## The metric

Closeness to FP32 is measured with `falcon-ocr-eval agree`: every calibration
page is teacher-forced along the FP32 tokens, and the steps where the
configuration's own greedy choice differs are counted (flips), together with
the KL divergence of its distribution. A flip at step 30 of a free-running
page turns everything after it into edits, so free-running comparisons
understate agreement. Teacher forcing compares each step on the same input.

The original anchor is the 55 calibration pages outside the GPTQ capture
set: 24,262 steps at up to 512 per page, or 78,282 with `agree --max-steps
8192`. Its FP32 reference is
`artifacts/phase4/checks/calibration-reference.json`.

The exception-column and rotated-cache results use the Ryzen 7 7700X's own
FP32 reference: the 52 pages of its calibration lock outside the capture
set, 22,726 steps at up to 512 per page. On this set the reference profile
has 0 flips, near-exact 1 (KL 7.1e-8) and fast mode 67 (KL 3.0e-4). Compare
these arms with each other, not with the 24,262-step results. KL is over
each step's 32 most likely FP32 tokens (`falcon-ocr-eval agree
--reference-topk`); every arm uses `tools/agree_queue.sh`'s flags (`--exp
fast --backend avx2`). Those 8-bit arms ran before fast mode's INT16
prefill projections became the default, so they used FP32 projection
panels. [DEVELOPMENT.md](DEVELOPMENT.md#token-agreement) gives the commands
for either reference and the flag that reproduces those arms.

Character error rate (CER) counts character insertions, deletions and
substitutions relative to the reference text's length. Micro CER pools the
edit counts and reference character counts across pages before dividing.
CER changes use percentage points (pt).

Quality gates use budgets registered before the run and fresh held-out
pages. A post-hoc diagnostic can explain a failure but does not replace its
recorded verdict; new qualification needs fresh pages.

## The three modes

| | `exact` | `near-exact` (default) | `fast` |
|---|---|---|---|
| Body weights | FP32 | 16-bit codes, FP32 absmax scale per 64 inputs | 8-bit GPTQ act-order codes, FP32 scale per 64 inputs (`w8-gptq.safetensors`) |
| KV cache after sealing | FP32 split records | 16-bit codes, BF16 scale per 32 values | 8-bit codes, BF16 scale per 32 values |
| Prefill projections | `gemm` crate, FP32 | FP32 panel GEMM | INT16 activations against the 8-bit codes on x86 CPUs with AVX2, FP32 panel GEMM elsewhere |
| Prefill attention | FP32 | FP32 | BF16 products with the static softmax on AVX512-BF16 CPUs, FP32 elsewhere |
| Bytes per token (weights + head) | 876 MB | 401 MB | 233 MB |
| KV bytes per position | 113 KB | 58 KB | 30 KB |
| Against FP32, 24,262 steps | **0 flips** | KL 9e-8, **1 flip** | KL 2.7e-4, **63 flips** |
| Journal page | 38 s | 21.5 s | 12.6 s (14 s without AVX512-BF16) |
| Profile label | `split-f32` (`reference` for traces and batches) | `w16-body-kv-q16` | `w8-body-kv-q8` |

Fast mode's flips and journal-page times predate its current prefill
kernels, the INT16 projections and the static softmax
([below](#int16-projections-and-static-softmax)). On the Ryzen 7 7700X
they move its flips on that host's 22,726-step anchor from 67 to 66 and
take the journal page's transformer prefill from 5.3 to 3.7 s
([PERFORMANCE.md](PERFORMANCE.md#int16-projections-and-static-softmax)).

Every mode maps its packed file when present. Without one, exact reads the
FP32 checkpoint; near-exact and fast quantize it at load, and fast needs its
GPTQ overlay. See [Packed model files](#packed-model-files).

Every mode uses the exact screened head and the portable polynomial exp in
prefill attention, which changes rounding but no token against the platform
exp on all 67 calibration pages (93,249 tokens); traces use the platform exp.
Fast mode also uses it in decode attention over its 8-bit cache, as the NEON
kernels always do (`--tune decode-exp=exact|fast` overrides): teacher-forced
on the 55 anchor pages up to 8,192 steps (78,282 steps) it has 94 flips
against 92 with the platform exp, the English gate below is unchanged, and
verification steps' attention is 3-10% faster.

Exact mode therefore gives FP32's tokens, not its bits. Under the reference
configuration (`RunnerConfig::reference()`: the platform exp and the full
head, which `trace` uses and `falcon-ocr-eval` starts from), the tests pin a
CPU smoke trace, and the split FP32 cache is bitwise the compact CPU kernel.
GPU validation establishes token agreement. The
[frozen smoke comparison](../reference/windows-rust-smoke-fp32-sinks-pairwise.json)
retains ten intermediate-tensor tolerance failures.

Under the automatic configuration of `run`, the polynomial exp moves logits
in their last bits: teacher-forced along the
1,295 FP32 tokens of a 1418 × 1224 page (Ryzen 7 7700X), 4.8% of exact
mode's top-32 log-probabilities were bit-identical to the platform exp's,
the largest difference 1.1e-4, with no flip.

Tokens held on everything measured: the GPU smoke page under both
configurations (`tests/gpu_parity.rs`, `tests/modes.rs`), all 67 calibration
pages and the four full benchmark pages.

Storage types were chosen against FP32 on the anchor set (GPU harness):
INT16 with a scale per 64 inputs (1 flip, KL 1.7e-7) beat FP16 (11 flips)
and BF16 (51 flips) at the same bytes; GPTQ INT8 (63 flips) beat
round-to-nearest INT8 (168 flips); 4-bit weights cost 10-12% output error
and are out. An IEEE FP16 KV cache was 1.2% faster than Q16 but raised
near-exact's flips from 1 to 4, so it was rejected.

## Fast mode's GPTQ overlay

Fast mode uses GPTQ act-order quantization of the 88 body matrices, with
scales per 64 inputs. `tools/make_gptq_overlay.sh` builds
`artifacts/model/w8-gptq.safetensors` in about 20 minutes: it captures the
activation Gram matrices (mean XᵀX) of every projection input on 12
calibration pages (`tools/gptq-calibration-pages.txt`, `falcon-ocr-eval
capture-gram`), then GPTQ-quantizes the 88 body matrices with act-order
(`tools/w8_variants.py --method gptq --act-order`). The projection inputs are
extremely anisotropic: one input channel of the layer-10 FFN output
projection (`W2`) carries 614,663× the median energy, which is why plain
rounding drifts three times as much. The overlay used for the published fast
file has SHA-256 `60808bbf…`; its codes and scales are included in that
packed file. Without an overlay, `--allow-rtn` quantizes round-to-nearest at
load (2.7 → 8.6 flips per 1,000 steps). Build instructions are in
[DEVELOPMENT.md](DEVELOPMENT.md).

## Exception columns (experimental)

Exception columns preserve selected weights in FP32 alongside the 8-bit
codes. What GPTQ leaves of fast mode's drift sits mostly in W2 (69% of the
8-bit weights' KL against production,
[RESULTS-V3.md](../research/phase4-hillclimb/attempt3/RESULTS-V3.md) §8),
whose squared-ReLU inputs carry the extreme channels. A weight-only
quantized product errs by about Σⱼ Δwⱼ xⱼ. A channel with a huge
activation multiplies its column's rounding error, and GPTQ can move that
error only onto columns whose inputs correlate with it.

An exception column is not quantized: its FP32 weights are stored beside
the codes, which are zero there. A W2 column costs 3 KB and a column of the
fused FFN gate/up projection (`W13`) 18 KB, against about 1.7 MB more for a
whole W2 kept in BF16.

Every product uses the FP32 columns: prefill and other large products have
them in place (INT16 panels hold codes only, so a body with exception
columns keeps FP32 projection panels instead of fast mode's INT16 ones, as
its plan reports), and decode adds one fused multiply-add per exception
column and output row after the 8-bit kernel, in ascending column order, so
draft verification stays bitwise the single-row steps. `doctor` and every
result's plan count the extra bytes.

Round-to-nearest quantization computes scales and codes with the exception
columns zeroed and keeps their original values. GPTQ never quantizes them
and processes them last, so they absorb every quantized column's error
compensation and end at the least-squares optimum for the chosen codes
under GPTQ's damped Gram (as in OWQ and SpQR). Their stored values are
therefore not the checkpoint's. [DEVELOPMENT.md](DEVELOPMENT.md) gives the
proxy, selection and build commands.

Such overlays have the format `falcon-ocr-attempt3-w8g64-v2`
(`{name}.__w8_exc_cols` I32 `[k]`, `{name}.__w8_exc_vals` F32 `[out, k]`),
and packed files made from them `falcon-ocr-kernel-v2`; binaries that predate
exception columns refuse both rather than compute with zeroed columns.
Overlays and packed files without exception columns keep their v1 formats
(re-packing gives the published files' tensors); an overlay with a tensor
that no body matrix uses is refused. An explicit `--w8-artifact` always
reads the checkpoint, even when a packed fast file sits in the model
directory.

Exception columns are opt-in. On the Ryzen 7 7700X, GPTQ's mean proxy error
over the 22 W2 matrices varies with the number of columns:

| FP32 columns per W2 | Mean proxy error |
|---|---:|
| 0 | 9.96e-4 |
| 1 | 8.12e-4 |
| 2 | 7.67e-4 |
| 4 | 7.14e-4 |
| 8 | 6.82e-4 |
| 16 | 6.37e-4 |

Source: Grams captured on that host by `tools/make_gptq_overlay.sh`.
The added storage ranges from 68 KB to 1.1 MB per decode step.

With 4 (+271 KB, 0.15% of fast mode's weight bytes per token) and the same
Grams, fast mode gave FP32's 1,295 tokens on the 1418 × 1224 sample page,
where the overlay without exception columns first differs at token 1,264.
On the 7700X anchor ([The metric](#the-metric)):

| Overlay | Flips | KL against FP32 |
|---|---:|---:|
| Published, no exception columns | 67 | 3.04e-4 |
| Rebuilt on the 7700X, no exception columns | 64 | 3.04e-4 |
| Same Grams, 4 W2 exception columns | 63 | 1.84e-4 |

Four exception columns reduce KL by 39%; flips do not resolve a difference
of that size. Decode time per token did not measurably change
(8.24 against 8.28 ms on the journal page,
[PERFORMANCE.md](PERFORMANCE.md#kv-caches-and-exception-columns)).

## INT16 projections and static softmax

Two kernels change fast mode's prefill rounding; both have been its
defaults since 2026-11-03 ([the decision](#the-decision)). Neither touches
decode, and exact and near-exact never use them. `doctor` and every
result's plan name the kernels (`panel-i16-vnni`, `panel-i16-avx2`,
`bf16-static`); `--tune prefill-projections=fp32` and `--tune
prefill-softmax=online` (both binaries) restore the previous ones
(`panel-avx2`, `bf16`).

The prefill projections of an 8-bit body run with INT16 activations. Each
row is quantized per 64-input group to INT16 with one FP32 step, the
group's absmax / 32,767 (round to nearest, ties to even; the RMS-norm row
factor multiplies the step). The GPTQ codes are sign-extended into INT16
panels, so a group's dot product is an exact INT32 sum
(64 × 32,767 × 127 < 2^31). Per group and output the sum converts to FP32
and joins the accumulator as `fma(sum, s_w · s_a, acc)`, in ascending group
order. The AVX512-VNNI kernel (`vpdpwssd`, plan label `panel-i16-vnni`; the
`auto` backend on CPUs with AVX512F and VNNI) and the AVX2 kernel
(`vpmaddwd`, `panel-i16-avx2`; other x86 CPUs with AVX2, and `--backend
avx2`) agree bit for bit with each other and with a scalar oracle, because
the integer part is exact. A body with exception columns keeps FP32 panels,
aarch64 keeps them too (it has no INT16 kernel yet), and `--backend scalar`
and x86 CPUs without AVX2 run the projections through the `gemm` crate.

The static softmax replaces the online softmax of fast mode's BF16 prefill
attention (plan label `bf16-static`). It needs AVX512-BF16 and AVX512BW: a
CPU with AVX512-BF16 but not AVX512BW keeps the online softmax (`bf16`),
and one without AVX512-BF16, or `--backend avx2`, runs FP32 attention.
Falcon-OCR RMS-normalizes every query and key head without a weight before
the rotary positions, which preserve norms, so |q| and |k| are at most 8
and, with the 1/8 score scale, every score lies in [-8, 8]; the largest
|score| over all 22 layers and 16 heads of the journal page is 7.992.
`exp(score)` needs no maximum subtraction: the kernel has no running
maximum, no block-maximum pass and no rescale of the output rows, and key
tiles add without correction. The exp (a degree-3 polynomial with relative
error 7.5e-5, scaled by `vscalefps`) runs in the QK epilogue. The
denominator sums the BF16-rounded probabilities that PV multiplies, so the
weights PV applies sum to one.

Each adds little error. An INT16 activation moves by at most half a step,
1/65,534 of its group's absmax, where BF16 rounds each value by up to 2^-8
of itself (8 significant bits); the weights keep their GPTQ codes, so the
activation rounding is the only new error. The static softmax adds the
polynomial's 7.5e-5, below the up to 2^-8 that BF16 rounding gives every
probability before PV, and the rounded denominator cancels the common part
of that rounding. Exp overflows only beyond |score| of about 87, so the
bound of 8 leaves a factor of ten.

### Teacher-forced agreement

Agreement is on the 7700X anchor ([The metric](#the-metric)): 22,726
steps, `falcon-ocr-eval --exp fast --threads 16 --profile w8-body-kv-q8`
with the published GPTQ overlay, `agree --max-steps 512` and
`--reference-topk`. The arms ran while both kernels were opt-in knobs, on
the default backend, so even the first row has BF16 prefill attention; the
exception-column tables above ran `--backend avx2` and give 67 flips and
KL 3.04e-4 for the same overlay.

| Prefill arm | Flips | KL against FP32 |
|---|---:|---:|
| Previous default (FP32 panels, online softmax) | 67 | 2.94e-4 |
| BF16 projections (removed) | 79 | 3.51e-4 |
| INT16 projections | 65 | 2.98e-4 |
| Static softmax | 61 | 2.91e-4 |
| Both (the current default) | 66 | 2.99e-4 |

KL moves +1.4% with INT16 projections, -1.0% with the static softmax and
+1.7% with both, against +19% for BF16 projections; flips (65, 61 and 66
against 67) do not resolve differences of that size. Tokens held: on the
journal page (6,544 input tokens) the previous default, each kernel and
both give the same 1,140 tokens ending at EOS, and the smoke page gives the
receipt's 17 tokens under each. Their speed is in
[PERFORMANCE.md](PERFORMANCE.md#int16-projections-and-static-softmax).

The BF16 projections (plan label `panel-bf16`) were the other way to speed
up the 8-bit projections. They matched the INT16 ones' prefill speed on the
7700X but raised KL by 19% where the INT16 ones raise it by 1.4%, and they
needed AVX512-BF16, where the INT16 kernel also runs on AVX2 CPUs, so they
were removed ([PERFORMANCE.md](PERFORMANCE.md#rejected-or-unadopted)).

### Free-running validation

Before the switch, one binary ran the previous defaults ("old") and the
two kernels ("new") in fresh processes on a Ryzen 7 7700X with the
published packed files and otherwise default settings, and near-exact, the
FP32 stand-in ([Fast against near-exact](#fast-against-near-exact)), on the
deep-fried pages. The test images and their deep-fried copies are in
[tests/fixtures/stress](../tests/fixtures/stress/README.md); per-page
results are in
[reference/fast-defaults-free-running-v1.json](../reference/fast-defaults-free-running-v1.json).

| Pages | Old and new token-identical | Other results |
|---|---:|---|
| 4 test images and an earlier deep-fried copy of 3 of them | 5 of 7 | The 4 originals identical |
| The 52 anchor pages ([The metric](#the-metric)) | 28 of 52 | Micro CER 40.67% old, 40.31% new; EOS on 44, the repetition stop on the same 8 |
| 45 deep-fried pages: 3 test images and the first 12 anchor pages (degraded and handwriting) at 3 levels, `--max-new-tokens 4096` | 25 of 45 | Old and new each token-identical to near-exact on 16 |

The two test images that differ are deep-fried JPEGs, and each difference
is one character of hard-to-read text, judged against the image: on the
wiki screenshot old reads the hexadecimal line
`464fdaf1672d7bef5e891c6e227/` exactly and new drops one `f`; on the book
scan new reads the margin label "String for Rip Cord Support" and old
"Sting for". Of the 24 anchor pages that differ, CER (distance to one
reference text, not a quality verdict) is lower for new on 10, higher on
11 and equal on 3. The nine deep-fried test images are token-identical in
7 cases and have text similarity 0.993 and 0.999 in the other two. Micro
CER of the 36 deep-fried anchor pages is 65.09% old, 77.65% new and
76.62% near-exact; one page, a counting loop that new and near-exact share
([Loops and the repetition stop](#loops-and-the-repetition-stop)),
accounts for the difference, so that aggregate is no quality comparison.

### The decision

Both kernels became fast mode's defaults, and the BF16 projections were
removed, on 2026-11-03, before the held-out gate below had a verdict, on
three grounds: agreement with FP32 is unchanged within what flips resolve
(66 against 67 flips, KL +1.7%); free running found no systematic
difference (one character of hard-to-read text each way, and a loop that
near-exact shares); and the two kernels cut fast mode's transformer
prefill by 31% on the 7700X. The fast-mode results in the next three
sections ran its earlier prefill kernels.

### Held-out gate

`fast-defaults-gate-v1` compares the new kernels with the previous
defaults on fresh pages. Its budget,
[reference/fast-defaults-gate-v1-budget.json](../reference/fast-defaults-gate-v1-budget.json),
was registered before either arm had output on its pages: 156 English
OmniDocBench pages from the router development pool, one per document
family and none from a family in corpus v1-v3 or the English gate (formulas
42, tables 41, multi-column 30, other 43), with figure regions that have no
ground truth painted white on 66. Both arms ran `--max-new-tokens 4096`;
the reference is the previous fast defaults. The criteria are the English
gate's: overall micro CER at most 0.25 pt above the reference, at most
1.0 pt above it in every category with at least 10 pages where the
reference ends at EOS, and no repetition stop on such a page.

The gate passed
([reference/fast-defaults-gate-v1-results.json](../reference/fast-defaults-gate-v1-results.json)).
Micro CER on the pages where the reference ends at EOS:

| Pages | Count | Previous defaults | INT16 and static | Change |
|---|---:|---:|---:|---:|
| All | 151 | 21.654% | 21.629% | -0.025 pt |
| Formulas | 42 | 33.501% | 33.489% | -0.012 pt |
| Multi-column | 30 | 12.323% | 12.323% | 0 |
| Other | 43 | 14.440% | 14.440% | 0 |
| Tables | 36 | 18.358% | 18.291% | -0.066 pt |

The other 5 pages, all tables, reach the 4,096-token cap in both arms; no
page changes its stop reason, and the repetition stop fires on none. 148
of the 156 pages are token-identical (144 of the 151 EOS pages), with 289
token edits in the EOS pages' 182,447 tokens. The gate's binary predates
the counting rule of the
[repetition stop](#loops-and-the-repetition-stop), which ends none of the
gate's 312 outputs (both arms) when they are replayed through it.

## Static softmax in the FP32 tiles (experimental)

`--tune prefill-softmax-fp32=static` (both binaries; default `online`)
gives the FP32 prefill tiles the static softmax of fast mode's BF16
attention ([above](#int16-projections-and-static-softmax)), on the same
bound: every score lies in [-8, 8], so `exp(score)` needs no running
maximum. It applies to 16- and 8-bit bodies on x86 CPUs with AVX2 and FMA:
near-exact's prefill attention, and fast mode's where that runs in FP32
(`--backend avx2`, CPUs without AVX512-BF16, `--tune prefill-bf16=off`).
Exact mode, aarch64 and `--backend scalar` keep the online softmax. The
plan names the kernel `avx512-wide-static`, or `avx2-static` under
`--backend avx2` and on CPUs without AVX-512F; the two agree bit for bit.
It is a knob of its own because `prefill-softmax` defaults to `static` for
fast mode's BF16 attention and would otherwise move those FP32-attention
fast runs too. The kernel is in
[ARCHITECTURE.md](ARCHITECTURE.md#kernels-one-source-per-operation-instantiated-per-isa).
It is not bitwise the online softmax: it always evaluates the polynomial
exp of `--exp fast` (the automatic configuration's), and on the score
itself, where the online softmax takes the rounded `score - max`.

Agreement is on the 7700X anchor ([The metric](#the-metric)): 22,726
steps, `falcon-ocr-eval --exp fast --backend avx2 --threads 16 --profile
w16-body-kv-q16`, `agree --max-steps 512` and `--reference-topk`; the
online row is the near-exact result quoted there.

| Near-exact, FP32 softmax | Flips | KL against FP32 |
|---|---:|---:|
| Online (the default) | 1 | 7.15e-8 |
| Static | 1 | 7.24e-8 |

Both flip the same step (step 329 of page `40408ccf4389cd81`, to the same
token) and no other. KL rises 1.3%; per page the static softmax's KL is
0.90 to 1.53 times the online one's, median 1.00, and lower on 29 of the
52 pages. On the `auto` backend (`avx512-wide`, `avx512-wide-static`),
pages `705b5079f112131e` and `40408ccf4389cd81` give each softmax the same
per-page flips and KL as `--backend avx2`, as the bitwise equal tile widths
require. Tokens held: on the journal page the static softmax gives
near-exact's 1,140 tokens ending at EOS on either tile width, and the
smoke page gives the receipt's 17 tokens with the knob in near-exact
(`avx512-wide-static`, `avx2-static`), fast mode (`bf16-static`, and
`avx2-static` under `--backend avx2`) and exact mode, whose plan stays
`avx512-wide`. Its speed is in
[PERFORMANCE.md](PERFORMANCE.md#static-softmax-in-the-fp32-tiles). It
changes near-exact's rounding, so it stays opt-in until a held-out gate in
near-exact mode decides ([NEXT-STEPS.md](NEXT-STEPS.md#still-open)).

## BF16 attention in near-exact mode (experimental)

`--tune prefill-bf16-w16=attention` (both binaries; default `off`) gives
near-exact's 16-bit body fast mode's BF16 prefill attention
([above](#int16-projections-and-static-softmax)) with the softmax that
`--tune prefill-softmax` selects: plan label `bf16-static` by default,
`bf16` with `online`. It runs exactly where fast mode's does: the `auto`
backend on CPUs with AVX512-BF16 (the static softmax also needs
AVX512BW), with `--tune prefill-bf16` on. `--backend avx2`, other CPUs and
`--tune prefill-bf16=off` keep the FP32 tiles, and exact mode's FP32 body
never takes it. Only the attention changes: the projections stay
`panel-avx2`, and the BF16 key and value copies are scratch beside the
FP32 compact cache that the Q16 cache is sealed from. Under `--escalate`,
a fast run with the knob rereads its
loops with BF16 attention too. On the smoke page near-exact with the knob
(`bf16-static`) gives the receipt's 17 tokens ending at EOS. Its agreement
with FP32 is not measured yet; it changes near-exact's rounding, so it
stays opt-in until agreement and a held-out gate in near-exact mode decide
([NEXT-STEPS.md](NEXT-STEPS.md#still-open)).

## Fast mode's held-out result

Fast mode is not qualified for handwritten or degraded pages; use
`near-exact` there. On the 200 held-out pages it was within budget overall
(micro CER -0.16 pt) and on 5 of 7 categories, and **failed** on handwriting
(+5.4 pt, three pages sent into loops by the 8-bit weights) and degraded
scans (+1.4 pt of spread drift). The repetition stop fired on five pages
where FP32 ends at EOS; four are 8-bit-induced loops, and on the fifth FP32
itself emits a 2,232-token hallucination.

A post-hoc comparison with production BF16 serving found that FP32 itself
would fail the same per-category budget against production. On 25
handwriting and 25 degraded pages, a blinded LLM judge rated the four
systems' content equal except on loop pages.

Sources: `reference/phase4-quality-budget.json` (the one registered run),
`research/phase4-hillclimb/attempt3/RESULTS-V3.md` §6 and §8.

## Fast mode on English print

Fast mode passed the English gate overall and failed its ordinary-page
category limit. The set contains 118 English OmniDocBench pages outside
corpus v1-v3 and every v3 document, one page per document. The model is
trained on English only, and the handwriting and degraded-scan failures
above were on Chinese pages. The English gate uses the same criteria:

| English gate comparison with FP32 | Micro CER change | Verdict |
|---|---:|---|
| Overall | +0.01 pt | Pass |
| Formulas | -0.06 pt | Pass |
| Tables | +0.02 pt | Pass |
| Multi-column | -0.08 pt | Pass |
| Slides | 0.00 pt | Pass |
| Ordinary pages (19) | +2.48 pt | Fail |

No page hit the repetition stop. One math-book page accounts for +2.17 pt:
fast mode transcribes its three commutative diagrams as LaTeX arrays that
match the page, FP32 writes only the equation numbers, and the ground truth
marks the diagrams as empty figures. English OmniDocBench has almost no
handwriting or degraded scans, so those stay unqualified.

Source: `reference/english-gate-v1-*`, including
`reference/english-gate-v1-results.json`.

In a post-hoc diagnostic, every figure region that the ground truth leaves
out is painted white on all systems' input (`mask_excluded.py`, 35 of the
118 pages). Fast mode is -0.11 pt against FP32 overall and +0.02 pt on
ordinary pages. Future gates on OmniDocBench pages use figure-masked pages
from the start. The original gate remains failed on ordinary pages; the
diagnostic supports the practical recommendation of fast mode for printed
English.

Source: `reference/english-gate-v1-masked-results.json`.

## Fast against near-exact

Near-exact (the default) produced exactly FP32's tokens on all 118 English
gate pages (110,516 tokens, figure-masked, default settings), so on printed
English fast mode's cost is its difference from FP32: 93 of the 118 pages
token-identical; half of the other 25 differ by at most 6 characters (a
fifth only in whitespace, quotes or markdown); 112 pages within 0.5 pt of
the near-exact CER against the ground truth, 3 worse and 3 better (2 each by
more than 2 pt, tiny-print pages where one early token changes the rest);
-0.11 pt overall; no loops. For about 1.7× the speed that is no measurable
loss on printed English.

The cost shows on hard pages: on the held-out set's handwriting and
degraded scans (all Chinese, which the model reads poorly even in FP32),
the 8-bit weights tipped uncertain pages into loops or drift.
Use near-exact for handwriting, degraded scans and anything the model finds
hard to read.

## Resolution routing

`--max-dimension auto` chooses 768, 1024 or 1536 per page. It changes the
model input and output, so it stays opt-in and exact/near-exact comparisons
use a fixed resolution. See
[resolution routing](ARCHITECTURE.md#resolution-routing) for the runtime
behavior.

The router **failed** its held-out English gate (+0.29 pt overall against a
+0.25 pt limit); the figure-masked diagnostic does not replace that verdict.
The [development-set results](../research/resolution-router/README.md#development-set-results)
and [held-out English gate](../research/resolution-router/README.md#held-out-english-gate-failed)
include the timing, category comparisons, diagnostics and receipts.

## Margin cropping

`run --crop-margins` or `--crop-margins=PAD`
(`GenerationOptions::crop_margins`, off by default) cuts the blank margins of
each page after the processor's first, aspect-preserving resize, keeping PAD
pixels (default 24, at that resize's scale) around the content; the second
resize and the patches then run unchanged on the cut page. Text keeps its size
in pixels and only the image token count drops, which shortens prefill
(attention grows with the square of the token count) and every decode step.
PAD must be below `--max-dimension`, since a larger padding never crops the
page; with `auto` it must be below 768, so that it can crop a page routed to
any size.

The content box comes from integer luma statistics
(`preprocess::margin_crop`): the background is the 90th luma percentile and
must be at least 160, ink is at least 64 levels darker, and a row or column
is content when it holds two ink pixels within a run of at least four such
lines, so dust specks up to 3 pixels across are ignored. A page stays whole
when:

- it has no content;
- its background is dark (inverted pages, dark slides);
- the crop would remove less than 10% of the area;
- the crop's second resize would do more than round each side to whole
  patches;
- the crop would keep as many patches as the whole page (small pages, where
  rounding takes back what the crop removed).

Dark scan borders, gutter shadows and punch holes count as ink and keep
their side of the page. Ink is counted per row and per column over the whole
page, so a mark that runs the length of one edge makes every row (or column)
content: the crop keeps that side and both adjacent margins whole and cuts
only the opposite margin. A mark that covers only part of an edge (a punch
hole, a short border) keeps its side and the rows or columns it covers.

Marks fainter than the ink threshold (light pencil) or thinner than four
pixels are cut off when they lie beyond the padding, except that a
full-length mark two or three pixels thick, though cut off itself, still
makes every row (or column) content and keeps the margins at its two ends.

On a 1188 × 1536 page with a text block at x 180-1000, y 200-1350 and
default padding:

| Border | Image tokens after cropping |
|---|---:|
| None | 4,050 (7,104 uncropped) |
| 18-pixel gutter shadow down the right edge | 6,144 |
| 12-pixel black border down the left edge | 6,144 |
| 12-pixel black band along the top | 6,364 |

Each result reports its crop (`crop`: `x`, `y`, `width`, `height` and the
first-resize `first_width` and `first_height`, all in first-resize pixels);
`width` and `height` stay those of the model input. Routed pages
(`--max-dimension auto`) are routed on the whole page and crop the resize
they run at, as does the safety-net rerun. `run --pipeline`, `--batch-size
N` and the `--escalate` rerun prepare each page the same way, so every path
reports the same crop. The library refuses the option for tensor traces and
for teacher scoring (`Runner::score_teacher_file`, which `falcon-ocr-eval
agree` uses; that binary has no crop flag), since both compare with the
whole page.

It is opt-in because it changes the model input. TII's layout pipeline feeds
cropped regions to the model, so crops are in distribution. On synthetic book
pages (a text block with about 15% side margins) it removes 42-44% of the
image tokens at 1536 (6,048 → 3,400-3,485), 40% at 1024 and 35% at 768;
pages whose content fills the page are not cropped. The router's page-time
model (fast mode with the draft head,
`research/resolution-router/analyze_agreement.py`) puts 6,048 → 3,485 image
tokens at about 37% less time for a page with 700 output tokens. This is an
estimate, not a timing measurement.

On 24 pages of the Internet Archive scan
`cu31924014450716` (Strunk's *The Elements of Style*, 1920), rendered at
956 × 1536 by `tools/pdf_to_pages.py`, in near-exact mode on a Ryzen 7 7700X,
it cropped every page, removed 16% of the input tokens (138,624 → 115,991)
and 17-18% of the run's time in each of two rounds (+21% pages per hour).
Every page kept its words. 10 kept their tokens; 6 differ only in markup
(heading and emphasis marks, superscript notation, quote style), 4 add or
drop one period, and 1 joins a word hyphenated across a line ("semicolon").
One page gives its two-column word lists as tables where the whole page
gave one column after the other.

The other two changes are corrections, checked against the page images:
the cropped pages read "come" (a damaged glyph in the scan) and the note
number "6-7" where the whole pages read "could" and "6".

[DEVELOPMENT.md](DEVELOPMENT.md) describes how to compare cropped and whole
pages on a book before choosing this option.

## Loops and the repetition stop

The stop (`--stop-repetition`, on by default) has two rules. Either ends
the page with `finish_reason: "repetition"`; output before the stop is
unchanged (`tests/modes.rs`), but the page ends early. The exact rule fires
once the output repeats a cycle of at most 128 tokens for at least
`max(256, 4 × cycle)` tokens. On the calibration pages it never fired on a
page that ends normally and cut decode work by about a third. It also ends
genuine FP32 loops, even one that FP32 itself ends at EOS after a
2,232-token hallucination (a held-out page, where the stopped output read
better: CER 1.90 → 0.83), so there the default output is shorter than
FP32's.

The counting rule ends loops whose lines differ only in numbers that
count, such as `44 女 …`, `45 女 …`. The pinned tokenizer gives every
digit its own token (`OcrTokenizer::load` checks IDs 539-548), so the rule
reads each run of digit tokens as one number and compares each symbol, a
number or another token, with the one a period of at most 128 symbols
earlier. Other tokens must be equal, each number must differ from its
counterpart a period earlier by the same step in every cycle, and at least
one step must be nonzero, which leaves exact loops to the exact rule. The
rule fires once such a window holds 32 whole cycles and spans at least 256
tokens (`MIN_COUNT_REPEATS`, `MIN_WINDOW` in `src/repetition.rs`).

The threshold comes from recorded text: 423 distinct output sequences
under `artifacts/` (calibration, anchor, book, stress and probe runs) and
the 1,651 OmniDocBench ground-truth pages, tokenized with tables in the
model's style. The longest counting window in legitimate text is 20
cycles, a scoring table counting down from 20 bonus points (v3 page
`aa8e1119117b1822`, which ends at EOS); the next longest is 9, and a
threshold of 16 would cut the scoring table. The three counting loops on
record reach 65, 77 and 212 cycles. At 32 the rule ends 4 of these 2,074
outputs, none of which ends at EOS:

| Page | Output | Tokens before | Counting rule |
|---|---|---:|---:|
| Mild deep-fried `ad43a4d35e59199e` | Fast and near-exact (identical): rows 44 to 255 | 4,096 (cap) | 695 |
| Anchor `3b0add7f7d14d45c` | Fast mode's current kernels: a list `(1)` to `(100)` of one date, `1980年12月24日 星期五` | 2,247 (exact rule) | 728 |
| `3ae00f1496cda3ff` | FP32, CPU reference and PyTorch GPU cross-check: the years 1981, 1982, … | 512 and 384 (budgets) | 308 |

The deep-fried page is a degraded Traditional Chinese page
([free-running validation](#free-running-validation)); its counting loop
now ends in row 75. Fast mode's previous prefill kernels fell into an exact
two-line loop there instead, which the exact rule ended at 314 tokens. In
each output the loop continues past the cut, and fast mode's outputs of
`3ae00f1496cda3ff` end at EOS without the year run. Both rules together
cost about 0.3 µs per token (one run on a busy 7700X), against about 9 ms
per decode step.

Legitimate content that matches a rule is cut: a 256-token page of
identical lines, or 32 or more rows identical apart from counting numbers
(a long blank numbered form, a number chart). Pass
`--stop-repetition=false` for such pages.

`run --escalate` rereads fast-mode pages that either rule ends with the
near-exact model, and under `--max-dimension auto` the router's
[safety net](ARCHITECTURE.md#resolution-routing) reruns them at 1536.
[Long runs](ARCHITECTURE.md#long-runs) describes model selection,
batching, records and failure handling.

The repetition stop ends the 8-bit-induced loops seen on held-out
handwriting. Rereading only those pages gives near-FP32 output while
limiting the near-exact cost to looping pages. Length stops do not
escalate: fast mode reaches the length limit less often than FP32 (7
against 19 of the 200 held-out pages), so rerunning them would pay for the
most expensive pages with little to gain. Near-exact pages do not escalate
to exact mode: near-exact matched FP32 on all 118 English gate pages, so its
loops are almost always FP32's own.

This follows the loop-rerun proposal (remedy 1) in
`research/phase4-hillclimb/attempt3/RESULTS-V3.md` §7, using near-exact
instead of exact. Its effect on those spent held-out pages is unmeasured.
On the 64 v3 calibration pages (published files, Ryzen 7 7700X), the stop
ends 9 fast-mode pages and `--escalate` rereads all 9 with the tokens of
plain near-exact runs. Two end at EOS (516 and 1,858 tokens, where fast
looped at 335 and 2,139); the other 7 also loop in near-exact mode.

## Packed model files

`falcon-ocr pack --output F` writes a kernel-ready SafeTensors file.
Select `--mode exact` for FP32, `--mode near-exact` (default) for INT16, or
`--mode fast` for GPTQ INT8. The published filenames identify body weight
storage, independently of the mode names and KV cache choices above:

| File | Mode | Size |
|---|---|---:|
| `falcon-ocr-v1.5-fp32.safetensors` | `exact` | 1,133 MB |
| `falcon-ocr-v1.5-int16-g64.safetensors` | `near-exact` | 806 MB |
| `falcon-ocr-v1.5-int8-gptq-g64.safetensors` | `fast` | 638 MB |
| `falcon-ocr-v1.5-draft-head.safetensors` | Shared draft head | 94 MB |

INT16 means scaled integer codes, not FP16 or BF16. GPTQ INT8 describes
this runner's group-64 quantizer, not a GGUF encoding. All models retain
FP32 embeddings, vocabulary head, norms, projector and sinks, plus the INT8
head screen and its bounds. These are runner-specific model files, not
Transformers checkpoints.

FP32 body tensors use `falcon-ocr-kernel-fp32-v1`, with `weight_bits=32`
and no quantization group. INT16 and INT8 keep `falcon-ocr-kernel-v1`, or
`falcon-ocr-kernel-v2` with exception columns. Metadata records the profile,
source checkpoint and tensor digest. The formats are independent of the
filenames; older binaries reject the FP32 format.

`--model-file F`, or the mode's catalog file in `--model`, maps tensors in
place. In the recorded quantized-file comparison, load fell from 2.2 s to
10 ms and peak resident memory from 2.9 to 1.8 GB, with identical tokens to
the checkpoint loader (`tests/modes.rs`). The files are on
[BEE-spoke-data/falcon-ocr-v1.5-cpu](https://huggingface.co/BEE-spoke-data/falcon-ocr-v1.5-cpu).

Upgrade the runner and download the catalog files together. The renamed
INT16 and INT8 files replace the old `near-exact` and `fast` filenames;
the mode flags stay the same. Older binaries can load the quantized files
through an explicit `--model-file`, but cannot discover the renamed files
automatically or load the FP32 format. Old cached copies can be removed
after refreshing the model directory.

## Research profiles

`falcon-ocr-eval --profile` takes any `Weights × Kv` pair: `reference`,
`split-f32`, `kv-q16`, `kv-q8`, `kv-q8r`, `kv-q4r`, `w16-body-compact`,
`w16-body`, `w16-body-kv-q16`, `w16-body-kv-q8`, `w16-body-kv-q8r`,
`w16-body-kv-q4r`, `w8-body`, `w8-body-split-f32`, `w8-body-kv-q16`,
`w8-body-kv-q8`, `w8-body-kv-q8r`, `w8-body-kv-q4r`. Only the three above are
modes of the CLI. Its hidden `--kv-cache compact|f32-split|q16|q8|q8r|q4r`
(run and doctor only) replaces the KV half of whatever profile the other
flags resolve to, packed files included: `--mode fast --kv-cache q8r` runs
`w8-body-kv-q8r`, and the result's `mode` (null), `precision` and `plan` name
that profile. With `--model-file`, `falcon-ocr-eval --profile` likewise picks
the cache for the file's weights (it must name them), and its reports record
the profile that ran and the file. The checkpoint loader's `--w8-artifact`,
`--keep-fp32` and `--weights-bf16` are refused beside `--model-file`, as is
`capture-gram`, which needs the FP32 reference profile.

### Rotated KV cache (experimental)

`q8r` and `q4r` rotate cache blocks before 8- or 4-bit quantization. Their
representation, decode transforms and chunk-rounding behavior are described
in [KV cache formats](ARCHITECTURE.md#kv-cache-formats).

The uses studied: a more faithful fast mode (`w8-body-kv-q8r`), a cheaper
cache for near-exact (`w16-body-kv-q8r` streams 23% fewer bytes per decode
step than `w16-body-kv-q16` on the journal page), and a 4-bit cache for fast
mode (`w8-body-kv-q4r`). With the published files, every rotated profile
gives the smoke page's reference tokens. Comparisons across caches pin the
decode exp: under `--exp fast` the 8-bit and rotated caches default to the
polynomial decode exp and `q16` to the platform one, a switch that alone
moved fast mode from 92 to 94 flips (above), about the size of the effect
under study; pass `--tune decode-exp=exact|fast`. On the 7700X anchor
([The metric](#the-metric)):

| Body weights | Cache | Decode exp | Flips | KL |
|---|---|---|---:|---:|
| FP32 | `q8` | polynomial | 22 | 2.94e-5 |
| FP32 | `q8r` | polynomial | 22 | 2.60e-5 |
| 16-bit (near-exact's) | `q16` | platform | 1 | 7.1e-8 |
| 16-bit | `q8r` | platform | 18 | 2.48e-5 |
| 8-bit GPTQ (fast mode's) | `q8` | polynomial | 67 | 3.04e-4 |
| 8-bit GPTQ | `q8r` | polynomial | 61 | 2.93e-4 |
| 8-bit GPTQ | `q4r` | polynomial | 335 | 9.04e-3 |

Rotation lowers the 8-bit cache's KL by 11% with FP32 weights and by 4% in
fast mode, and its flips stay within noise (22 → 22, 67 → 61), as the static
study of the pinned weights predicted
([research/kv-rotation](../research/kv-rotation/README.md)): the cache
blocks are nearly Gaussian except the temporal key halves of late layers.
None of the three uses holds up: `q8r` would gain fast mode a few percent of
KL at no measurable decode cost; on near-exact's weights `q8r` decodes 21%
faster per token (7% with the draft head), but an 8-bit cache, rotated or
not, flips 18 times where `q16` flips once; and `q4r`, whose codes decode
through the 8-bit loads, is 15% slower per token than `q8` and multiplies
fast mode's flips by 5 and its KL by 30
([PERFORMANCE.md](PERFORMANCE.md#kv-caches-and-exception-columns) has the
timing). The profiles stay for research.
