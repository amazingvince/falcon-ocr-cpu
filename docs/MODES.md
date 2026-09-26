# Modes

Status: current as of 2026-09-25. Measurements: Ryzen 9 7950X (16 cores,
32 threads, DDR5), Windows 11, the journal benchmark page (6,544 image tokens).

## The metric

Closeness to FP32 is measured with `falcon-ocr-eval agree`: every calibration
page is teacher-forced along the FP32 tokens, and the steps where the
configuration's own greedy choice differs are counted (flips), together with
the KL divergence of its distribution. A flip at step 30 of a free-running
page turns everything after it into edits, so free-running comparisons
understate agreement; per-step agreement does not. The anchor set is the 55
calibration pages outside the GPTQ capture set, 24,262 steps
(`artifacts/phase4/checks/calibration-reference.json`).

## The three modes

| | `exact` | `near-exact` (default) | `fast` |
|---|---|---|---|
| Body weights | FP32 | 16-bit codes, FP32 absmax scale per 64 inputs, quantized at load | 8-bit GPTQ act-order codes, FP32 scale per 64 inputs (`w8-gptq.safetensors`) |
| KV cache after sealing | FP32 split records | 16-bit codes, BF16 scale per 32 values | 8-bit codes, BF16 scale per 32 values |
| Prefill projections | `gemm` crate, FP32 | FP32 panel GEMM | FP32 panel GEMM (`--tune prefill-bf16=all`: BF16) |
| Prefill attention | FP32 | FP32 | BF16 products on AVX512-BF16 CPUs, FP32 elsewhere |
| Bytes per token (weights + head) | 876 MB | 401 MB | 233 MB |
| KV bytes per position | 113 KB | 58 KB | 30 KB |
| Against FP32, 24,262 steps | bitwise | KL 9e-8, **1 flip** | KL 2.7e-4, **63 flips** |
| Journal page | 38 s | 21.5 s | 12.6 s (14 s without AVX512-BF16) |
| Profile label | `split-f32` (`reference` for traces and batches) | `w16-body-kv-q16` | `w8-body-kv-q8` |

Every mode uses the exact screened head and the portable vector exp in
prefill attention (token-identical to the platform exp on all calibration
pages; traces use the platform exp). Fast mode also uses it in decode
attention over its 8-bit cache, as the NEON kernels always do
(`--tune decode-exp=exact|fast` overrides): teacher-forced on the 55 anchor
pages up to 8,192 steps (78,282 steps) it has 94 flips against 92 with the
platform exp, the English gate below is unchanged, and verification steps'
attention is 3–10% faster. Exact mode is bit-identical to the FP32
reference: the smoke trace and the GPU parity test hold under both the
reference and the automatic configuration.

Storage types were chosen against FP32 on the anchor set (GPU harness):
INT16 with a scale per 64 inputs (1 flip, KL 1.7e-7) beat FP16 (11 flips)
and BF16 (51 flips) at the same bytes; GPTQ INT8 (63 flips) beat
round-to-nearest INT8 (168 flips); 4-bit weights cost 10–12% output error
and are out. An IEEE FP16 KV cache was 1.2% faster than Q16 but raised
near-exact's flips from 1 to 4, so it was rejected.

## Fast mode's GPTQ overlay

`tools/make_gptq_overlay.sh` builds `artifacts/model/w8-gptq.safetensors` in
about 20 minutes: it captures the mean XᵀX of every projection input on 12
calibration pages (`tools/gptq-calibration-pages.txt`, `falcon-ocr-eval
capture-gram`), then GPTQ-quantizes the 88 body matrices with act-order
(`tools/w8_variants.py --method gptq --act-order`). The projection inputs are
extremely anisotropic (one channel of a layer-10 W2 input carries 614,663×
the median energy), which is why plain rounding drifts three times as much.
The published overlay has SHA-256 `60808bbf…`; the published fast packed file
contains it. Without an overlay, `--allow-rtn` quantizes round-to-nearest at
load (2.7 → 8.6 flips per 1,000 steps).

## Fast mode's held-out result

Fast mode ran once on the 200 held-out pages against a pre-registered budget
(`reference/phase4-quality-budget.json`). It was within budget overall
(micro CER −0.16 pt) and on 5 of 7 categories, and **failed** on handwriting
(+5.4 pt, three pages sent into loops by the 8-bit weights) and degraded
scans (+1.4 pt of spread drift). The repetition stop fired on five pages
where FP32 ends at EOS; four are 8-bit-induced loops, and on the fifth FP32
itself emits a 2,232-token hallucination. Fast mode is therefore **not
qualified** for handwritten or degraded pages; use `near-exact` there. A
post-hoc comparison with production BF16 serving found that FP32 itself
would fail the same per-category budget against production, and a blinded
LLM judge rated the four systems' content equal except on loop pages.
Details: `research/phase4-hillclimb/attempt3/RESULTS-V3.md` §6 and §8.

## Fast mode on English print

The model is trained on English only, and every failure above sat on Chinese
pages, so fast mode also ran a fresh English gate: 118 English OmniDocBench
pages never used before (outside corpus v1–v3 and every v3 document, one page
per document; `reference/english-gate-v1-*`), pre-registered with the same
criteria. It passed overall (micro CER +0.01 pt), on formulas (−0.06 pt),
tables (+0.02), multi-column (−0.08) and slides (0.00), with no repetition
stop, and **failed** the per-category limit on ordinary pages (+2.48 pt over
19 pages). One math-book page accounts for +2.17 pt: fast mode transcribes its
three commutative diagrams as LaTeX arrays that match the page, FP32 writes
only the equation numbers, and the ground truth marks the diagrams as empty
figures. The recorded verdict stays a fail; read as a quality statement, fast
mode matches FP32 on printed English. English OmniDocBench has almost no
handwriting or degraded scans, so those stay unqualified. Details:
`reference/english-gate-v1-results.json`.

A later, post-hoc check confirms the reading: with every figure region that
the ground truth leaves out painted white on all systems' input
(`mask_excluded.py`, 35 of the 118 pages), fast mode is −0.11 pt against FP32
overall and +0.02 pt on ordinary pages. Future gates on OmniDocBench pages
use figure-masked pages from the start
(`reference/english-gate-v1-masked-results.json`).

## Resolution routing

`--max-dimension auto` lets the runner choose each page's maximum dimension:
768, 1024 or 1536. Prefill and decode both scale with the image token count,
and most printed English pages read as well at a lower resolution. The router
(`src/router`, ~12 ms per page) computes 26 image statistics of the page after
the processor's first resize at 1536 (size, ink, contrast, a projection-profile
estimate of the text lines, column gaps, and how much the ink changes when the
page is scaled to 1024 or 768 and back), scores two gradient-boosted tree
models, and takes the smallest resolution whose model says the page is
routable and at which the median text line stays at least 8 px tall. A routed
page is exactly the input `--max-dimension 768` or `1024` would give it. If
the routed run loops or reaches the length limit, the page is rerun at 1536
(the safety net; the result's `route.safety_net` records the first attempt).

On the router development set (389 English OmniDocBench pages with ground
truth, none in any corpus or the English gate; fast mode with the draft head)
it routed 73 pages to 768 and 185 to 1024, reran 2, and changed micro CER
against the ground truth from 20.89% to 20.20% (−0.69 pt; 95% bootstrap
[−1.35, −0.18]) while saving about a quarter of the CPU time (24.7% from the
fixed-resolution runs; 26.4% measured over the part of the end-to-end run
before other load appeared on the host). Lower resolution often reads better
(tables, textbooks, multi-column pages); dense small print (magazines,
newspapers) stays at 1536. The router was tuned on this set.

On the held-out English gate pages (the 118 pages above, against fast mode at
1536, pre-registered with the same criteria) the router **failed**: overall
+0.29 pt (limit +0.25), ordinary +1.45 pt and slides +6.66 pt (limit +1.0);
formulas +0.90, tables −0.11 and multi-column −0.53 pt passed, with no
repetition stop and the same route mix as the development set. About two
thirds of the overall change comes from two pages where the routed run
transcribes a figure as an HTML table that the ground truth leaves out; the
rest is real: a sidebar dropped at 768 and weaker LaTeX at 1024. The
development set's improvement did not replicate, so read the router as about a
quarter less CPU time for roughly +0.3 pt CER on printed English. It stays
opt-in; it changes the output, so exact and near-exact comparisons use a fixed
resolution. With figures masked (the post-hoc check above) the router is
+0.06 pt against fast at 1536 and within every category limit, formulas
(+0.86 pt) being the closest, so the figure artifact explains the failure; a
verdict still needs fresh pages. Details: `research/resolution-router/README.md`,
`reference/router-english-gate-v1-results.json`.

## Loops and the repetition stop

The stop (`--stop-repetition`, on by default) ends a page once it repeats a
cycle of at most 128 tokens for at least `max(256, 4 × cycle)` tokens, with
`finish_reason: "repetition"`; output before the stop is unchanged. On the
calibration pages it never fired on a page that ends normally and cut decode
work by about a third; it also ends genuine FP32 loops. Legitimately periodic
content longer than that (a 256-token page of identical lines) would be cut:
pass `--stop-repetition=false`.

## Packed model files

`falcon-ocr --mode near-exact|fast pack --output F` writes one safetensors
file holding every tensor in the layout the kernels read (FP32 embedding,
head, norms, projector and sinks; body codes and scales; the INT8 head screen
and its bounds), with the recipe and a tensor digest in its metadata.
`--model-file F` (or a `falcon-ocr-v1.5-<mode>.safetensors` in `--model`)
maps it and uses it in place: load 2.2 s → 10 ms, peak resident memory
2.9 → 1.8 GB, tokens identical to the checkpoint loader (gated by
`tests/modes.rs`). The published files are on Hugging Face at
`amazingvince/falcon-ocr-v1.5-cpu`.

## Research profiles

`falcon-ocr-eval --profile` takes any `Weights × Kv` pair: `reference`,
`split-f32`, `kv-q16`, `kv-q8`, `w16-body-compact`, `w16-body`,
`w16-body-kv-q16`, `w16-body-kv-q8`, `w8-body`, `w8-body-split-f32`,
`w8-body-kv-q16`, `w8-body-kv-q8`. Only the three above are modes of the
CLI.
