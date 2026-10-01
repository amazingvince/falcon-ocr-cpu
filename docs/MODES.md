# Modes

Status: current as of 2026-09-30. Measurements: Ryzen 9 7950X (16 cores,
32 threads, DDR5), Windows 11, the journal benchmark page (6,544 input tokens).

## The metric

Closeness to FP32 is measured with `falcon-ocr-eval agree`: every calibration
page is teacher-forced along the FP32 tokens, and the steps where the
configuration's own greedy choice differs are counted (flips), together with
the KL divergence of its distribution. A flip at step 30 of a free-running
page turns everything after it into edits, so free-running comparisons
understate agreement; per-step agreement does not. The anchor set is the 55
calibration pages outside the GPTQ capture set, 24,262 steps at up to 512
per page (`artifacts/phase4/checks/calibration-reference.json`; `agree
--max-steps 8192` gives 78,282).

## The three modes

| | `exact` | `near-exact` (default) | `fast` |
|---|---|---|---|
| Body weights | FP32 | 16-bit codes, FP32 absmax scale per 64 inputs, quantized at load | 8-bit GPTQ act-order codes, FP32 scale per 64 inputs (`w8-gptq.safetensors`) |
| KV cache after sealing | FP32 split records | 16-bit codes, BF16 scale per 32 values | 8-bit codes, BF16 scale per 32 values |
| Prefill projections | `gemm` crate, FP32 | FP32 panel GEMM | FP32 panel GEMM (`--tune prefill-bf16=all`: BF16) |
| Prefill attention | FP32 | FP32 | BF16 products on AVX512-BF16 CPUs, FP32 elsewhere |
| Bytes per token (weights + head) | 876 MB | 401 MB | 233 MB |
| KV bytes per position | 113 KB | 58 KB | 30 KB |
| Against FP32, 24,262 steps | **0 flips** | KL 9e-8, **1 flip** | KL 2.7e-4, **63 flips** |
| Journal page | 38 s | 21.5 s | 12.6 s (14 s without AVX512-BF16) |
| Profile label | `split-f32` (`reference` for traces and batches) | `w16-body-kv-q16` | `w8-body-kv-q8` |

Every mode uses the exact screened head and the portable polynomial exp in
prefill attention, which changes rounding but no token against the platform
exp on all 67 calibration pages (93,249 tokens); traces use the platform exp.
Fast mode also uses it in decode attention over its 8-bit cache, as the NEON
kernels always do (`--tune decode-exp=exact|fast` overrides): teacher-forced
on the 55 anchor pages up to 8,192 steps (78,282 steps) it has 94 flips
against 92 with the platform exp, the English gate below is unchanged, and
verification steps' attention is 3–10% faster.

Exact mode therefore gives FP32's tokens, not its bits. Under the reference
configuration (`RunnerConfig::reference()`: the platform exp and the full
head, which `trace` uses and `falcon-ocr-eval` starts from) it is bitwise the
FP32 reference: the smoke trace hash is pinned, and the split FP32 cache is
bitwise the compact one. Under the automatic configuration of `run` the
polynomial exp moves logits in their last bits: teacher-forced along the
1,295 FP32 tokens of a 1418 × 1224 page (Ryzen 7 7700X), 4.8% of exact
mode's top-32 log-probabilities were bit-identical to the platform exp's,
the largest difference 1.1e-4, with no flip. Tokens held on everything
measured: the GPU smoke page under both configurations
(`tests/gpu_parity.rs`, `tests/modes.rs`), all 67 calibration pages and the
four full benchmark pages.

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

## Exception columns (experimental)

What GPTQ leaves of fast mode's drift sits mostly in W2 (69% of the 8-bit
weights' KL against production, `RESULTS-V3.md` §8), whose squared-ReLU
inputs carry the extreme channels. A weight-only quantized product errs by
about Σⱼ Δwⱼ xⱼ, so a channel with a huge activation multiplies its
column's rounding error, and GPTQ can move that error only onto columns
whose inputs correlate with it. An exception column is not quantized: its
FP32 weights are stored beside the codes, which are zero there. A W2 column
costs 3 KB and a W13 column 18 KB, against about 1.7 MB more for a whole W2
kept in BF16.

Every product uses the FP32 columns: prefill and other large products have
them in place (BF16 panels hold codes only, so under `--tune
prefill-bf16=all` a body with exception columns keeps FP32 projection panels,
as its plan reports), and decode adds one fused multiply-add per exception
column and output row after the 8-bit kernel, in ascending column order, so
draft verification stays bitwise the single-row steps. `doctor` and every
result's plan count the extra bytes.

```sh
python tools/w8_proxy.py --gram-dir artifacts/w8/gram --include "feed_forward\.w2" --jobs 16
EXCEPTIONS=4 REUSE_GRAMS=1 bash tools/make_gptq_overlay.sh   # artifacts/model/w8-gptq-exc4.safetensors
falcon-ocr --mode fast --w8-artifact artifacts/model/w8-gptq-exc4.safetensors run page.png
falcon-ocr --mode fast --w8-artifact artifacts/model/w8-gptq-exc4.safetensors pack --output fast-exc4.safetensors
```

`tools/w8_proxy.py` prints, per matrix, the share of the input energy in the
top k channels and the activation-weighted output error
√(tr(ΔW G ΔWᵀ) / tr(W G Wᵀ)) of round-to-nearest and of GPTQ with k
exception columns (k = 0, 1, 2, 4, 8, 16), quantized exactly as the
overlay builder does, so N can be chosen in minutes before any model run.
`tools/w8_variants.py --exceptions N --exceptions-include REGEX` keeps the N
highest-energy input columns of each matching matrix (`--exception-select
error` weights the energy by the column's expected rounding error). Round to
nearest computes scales and codes with those columns zeroed and keeps their
original values; GPTQ never quantizes them and processes them last, so they
absorb every quantized column's error compensation and end at the
least-squares optimum for the chosen codes under GPTQ's damped Gram (as in
OWQ and SpQR): their stored values are not the checkpoint's.

Such overlays have the format `falcon-ocr-attempt3-w8g64-v2`
(`{name}.__w8_exc_cols` I32 `[k]`, `{name}.__w8_exc_vals` F32 `[out, k]`),
and packed files made from them `falcon-ocr-kernel-v2`; binaries that predate
exception columns refuse both rather than compute with zeroed columns.
Overlays and packed files without exception columns keep their v1 formats
(re-packing gives the published files' tensors); an overlay with a tensor
that no body matrix uses is now refused. An explicit `--w8-artifact` always
reads the checkpoint, even when a packed fast file sits in the model
directory. To measure a v2 overlay offline, use `tools/w8_proxy.py
--overlay`.

Fast mode's defaults are unchanged. Measured on a Ryzen 7 7700X with the
Grams of `tools/make_gptq_overlay.sh` captured there: over the 22 W2
matrices GPTQ's mean proxy error falls from 9.96e-4 with no exception
columns to 8.12e-4, 7.67e-4, 7.14e-4, 6.82e-4 and 6.37e-4 with 1, 2, 4, 8
and 16, at 68 KB to 1.1 MB more per decode step. With 4 (+271 KB, 0.15%
of fast mode's weight bytes per token) and the same Grams, fast mode gave
FP32's 1,295 tokens on the 1418 × 1224 sample page, where the overlay
without exception columns first differs at token 1,264.

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

## Fast against near-exact

Near-exact (the default) produced exactly FP32's tokens on all 118 English
gate pages (110,516 tokens, figure-masked, default settings), so on printed
English fast mode's cost is its difference from FP32: 93 of the 118 pages
token-identical; half of the other 25 differ by at most 6 characters (a
fifth only in whitespace, quotes or markdown); 112 pages within 0.5 pt of
the near-exact CER against the ground truth, 3 worse and 3 better (2 each by
more than 2 pt, tiny-print pages where one early token changes the rest);
−0.11 pt overall; no loops. For about 1.7× the speed that is no measurable
loss on printed English. The cost shows on hard pages: on the held-out set's
handwriting and degraded scans (all Chinese, which the model reads poorly
even in FP32) the 8-bit weights tipped uncertain pages into loops or drift.
Use near-exact for handwriting, degraded scans and anything the model finds
hard to read.

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
[−1.35, −0.18]) while saving a quarter of the CPU time (25.1% measured on a
quiet host, 58.0 → 43.5 minutes, the two settings alternated in four chunks;
routed pages 38.5% faster). Lower resolution often reads better
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

## Margin cropping

`run --crop-margins` or `--crop-margins=PAD`
(`GenerationOptions::crop_margins`, off by default) cuts the blank margins of
each page after the processor's first, aspect-preserving resize, keeping PAD
pixels (default 24, at that resize's scale) around the content; the second
resize and the patches then run unchanged on the cut page. Text keeps its size
in pixels and only the image token count drops, which shortens prefill
(attention grows with the square of the token count) and every decode step.
The content box comes from integer luma statistics
(`preprocess::margin_crop`): the background is the 90th luma percentile and
must be at least 160, ink is at least 64 levels darker, and a row or column is
content when it holds two ink pixels within a run of at least four such lines,
so dust specks up to 3 pixels across are ignored. A page stays whole when it
has no content, when its background is dark (inverted pages, dark slides),
when the crop would remove less than 10% of the area, or when the crop's
second resize would do more than round each side to whole patches. Dark scan
borders, gutter shadows and punch holes count as ink and keep their side of
the page; marks fainter than the ink threshold (light pencil) or thinner than
four pixels are cut off when they lie beyond the padding. Routed pages
(`--max-dimension auto`) are routed on the whole page and crop the resize they
run at, as does the safety-net rerun. `run --pipeline`, `--batch-size N` and
the `--escalate` rerun prepare each page the same way, so every path reports
the same crop.

It is opt-in because it changes the model input. TII's layout pipeline feeds
cropped regions to the model, so crops are in distribution, but nothing has
been measured on real pages here. On synthetic book pages (a text block with
about 15% side margins) it removes 42–44% of the image tokens at 1536 (6,048 →
3,400–3,485), 40% at 1024 and 35% at 768; pages whose content fills the page
are not cropped. The router's page-time model (fast mode with the draft head,
`research/resolution-router/analyze_agreement.py`) puts 6,048 → 3,485 image
tokens at about 37% less time for a page with 700 output tokens. Each result
reports its crop (`crop`: `x`, `y`, `width`, `height` and the first-resize
`first_width` and `first_height`, all in first-resize pixels); `width` and
`height` stay those of the model input. The library refuses the option for
tensor traces and for teacher scoring (`Runner::score_teacher_file`, which
`falcon-ocr-eval agree` uses; that binary has no crop flag), since both
compare with the whole page.

To evaluate it, run a few dozen of your own book pages twice in the same mode,
with and without `--crop-margins`, and compare per page: identical token ids,
the character edit distance between the two texts over the uncropped text's
length (the router's disagreement measure,
`research/resolution-router/analyze_agreement.py`), the stop reason, CER
against ground truth where you have it, and `total_ms`. Read the pages that
changed: a lost page number or marginal note is the failure to look for.

## Loops and the repetition stop

The stop (`--stop-repetition`, on by default) ends a page once it repeats a
cycle of at most 128 tokens for at least `max(256, 4 × cycle)` tokens, with
`finish_reason: "repetition"`; output before the stop is unchanged
(`tests/modes.rs`), but the page ends early. On the calibration pages it never
fired on a page that ends normally and cut decode work by about a third. It
also ends genuine FP32 loops, even one that FP32 itself ends at EOS after a
2,232-token hallucination (a held-out page, where the stopped output read
better: CER 1.90 → 0.83), so there the default output is shorter than FP32's.
Legitimately periodic content longer than that (a 256-token page of identical
lines) would be cut: pass `--stop-repetition=false`.

`run --escalate` (fast mode) rereads every page that the stop ended with the
near-exact model: the near-exact packed file next to `--model-file`, else
what `--mode near-exact` finds in the model directory (its packed file or the
checkpoint quantized at load), loaded when a page first needs it. Pages are
reread one at a time, and in a batch the other rows wait meanwhile. The
near-exact result replaces the page's record, which keeps the fast attempt as
`escalated_from` (`mode`, `finish_reason`, `output_tokens`, `total_ms`, like
the router's `safety_net`; the page's `total_ms` includes it); a routed page
is reread at the size its fast run ended at. When the near-exact model
cannot be loaded or the rerun fails, the page keeps its fast result with the
reason in `escalation_error` and a warning on stderr, and after a failed
load the run stops trying.

This is remedy 1 of `research/phase4-hillclimb/attempt3/RESULTS-V3.md` §7,
with near-exact in place of exact mode: the loops that the 8-bit weights
caused on held-out handwriting were ended by the stop, so rereading those
pages gives them near-FP32 output, and only looping pages pay near-exact
cost. Only the repetition stop escalates: fast mode ran to the length limit
less often than FP32 (7 against 19 of the 200 held-out pages), so rereading
length stops would pay for the most expensive pages with little to gain.
Near-exact pages never escalate to exact mode: near-exact matched FP32 on all
118 English gate pages, so its loops are almost always FP32's own. The
held-out pages are spent, so the effect there is not measured; with the
published files, a fast-mode page of repeated lines that the stop ends is
reread with a plain near-exact run's tokens.

## Packed model files

`falcon-ocr --mode near-exact|fast pack --output F` writes one safetensors
file holding every tensor in the layout the kernels read (FP32 embedding,
head, norms, projector and sinks; body codes and scales, and any exception
columns; the INT8 head screen and its bounds), with the recipe and a tensor
digest in its metadata (format `falcon-ocr-kernel-v1`, or `-v2` with
exception columns).
`--model-file F` (or a `falcon-ocr-v1.5-<mode>.safetensors` in `--model`)
maps it and uses it in place: load 2.2 s → 10 ms, peak resident memory
2.9 → 1.8 GB, tokens identical to the checkpoint loader (gated by
`tests/modes.rs`). The published files are on Hugging Face at
`amazingvince/falcon-ocr-v1.5-cpu`.

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
the profile that ran and the file.

### Rotated KV cache (experimental)

`q8r` and `q4r` store every 32-value block of the cache (the temporal and
spatial key halves and the two value halves, prefix and generated positions)
as `H D x` before quantizing it: `H` is the 32 × 32 Sylvester Hadamard matrix
and `D` a fixed ±1 diagonal per block kind (`src/quant/rotation.rs`). `q8r`
then uses `q8`'s codes and BF16 scale per 32 values; `q4r` uses 4-bit codes
(-7..=7) with the same scale. An outlier channel no longer sizes the step of
the 31 other values of its block. Decode rotates each query instead of the
cache, `(2^-5 H D q) · (H D k) = q · k`, and un-rotates each head's output
once, `o = 2^-5 D H o'`, so the records and kernels are `q8`'s: `q8r` streams
the same 170 bytes per record and adds 32-point transforms, four per head, row
and layer (two query halves, two output halves), four per group for each
appended position and five per record when the prefix is sealed; its output is
bitwise the compact kernel over the dequantized rotated records with the
rotated query, followed by the inverse rotation. `q4r` takes 90 bytes per
record and decodes its codes through the 8-bit loads (bitwise tested on every
instruction set; not yet a vector nibble unpack).

The uses under study: a more faithful fast mode (`w8-body-kv-q8r`), a cheaper
cache for near-exact (`w16-body-kv-q8r` streams 23% fewer bytes per decode
step than `w16-body-kv-q16` on the journal page), and a 4-bit cache for fast
mode (`w8-body-kv-q4r`). With the published files, every rotated profile
gives the smoke page's reference tokens. Comparisons across caches pin the
decode exp: under `--exp fast` the 8-bit and rotated caches default to the
polynomial decode exp and `q16` to the platform one, a switch that alone
moved fast mode from 92 to 94 flips (above), about the size of the effect
under study; pass `--tune decode-exp=exact|fast`.
