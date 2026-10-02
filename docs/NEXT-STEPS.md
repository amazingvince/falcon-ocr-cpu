# Next steps

Status: current as of 2026-10-02. What still needs other hardware or a
decision, and proposals that are not implemented. The numbers live in the
documents that own them; this page links to them.

## Still open

- **Gates outside CI.** `tests/weight_layout.rs` and `tools/check.sh
  --smoke` need the GPU smoke outputs (`metadata.json`, `trace.safetensors`
  from the frozen reference), as `tests/gpu_parity.rs` and
  `tests/cache_layout.rs` do, but the self-hosted `weights` CI job, whose
  machine holds those outputs, runs only the latter two. The reference's
  preflight pins the RTX 4090 the outputs were recorded on, so no other
  machine can remake them.
- **The 55-page anchor.** The Ryzen 7 7700X agreement numbers in
  [MODES.md](MODES.md#the-metric) (exception columns, rotated caches) are on
  that host's 52-page anchor, so they compare arms with each other. Scored
  against upstream's `artifacts/phase4/checks/calibration-reference.json`
  ([DEVELOPMENT.md](DEVELOPMENT.md#token-agreement)), the same arms would
  give 24,262-step numbers comparable with the modes table.
- **Held-out gates** decide every default that would change tokens: margin
  cropping for books, `--max-dimension 1280`, a rotated cache and exception
  columns. Held-out pages are single-use, so each needs a pre-registered
  budget before its run.
- **Windows and aarch64.** The book runs, the page pipeline and the KV-cache
  timings ([PERFORMANCE.md](PERFORMANCE.md#on-a-ryzen-7-7700x)) were
  measured on one Linux desktop; no aarch64 machine has run the model with
  weights ([PORTABILITY.md](PORTABILITY.md)).

## Proposals (not implemented)

Estimates come from measured numbers in this repository; they are not
measurements. Agreement rules out two other ideas: a vector nibble unpack
for `q4r` (its 4-bit codes flip 5 times as often as fast mode's 8-bit cache)
and a faster near-exact on `q8r` (an 8-bit cache flips 18 times where
near-exact flips once; [MODES.md](MODES.md#rotated-kv-cache-experimental)).

### Exception columns as fast mode's default

- Why: 4 FP32 columns per W2 cut fast mode's KL against FP32 by 39% (3.04e-4
  → 1.84e-4 on the 7700X anchor) for 271 KB more per decode step, 0.15% of
  its weight bytes ([MODES.md](MODES.md#exception-columns-experimental)).
- Estimate: no flip gain at that anchor's size (64 → 63); the KL gain is
  what a held-out gate would have to confirm in output quality.
- Risks: a new packed format (`falcon-ocr-kernel-v2`) and republished files;
  the proxy curve flattens after 4 columns (7.14e-4 → 6.37e-4 at 16), and
  other matrices (QKV carries 16% of the 8-bit cost) are untested.
- First measurement: agree at 8, 16 and with QKV included, then the
  held-out English gate with a pre-registered budget.

### GPTQ damping relative to the median

- Why: `tools/w8_variants.py` damps each Gram by 1% of its mean diagonal; in
  the layer-10 W2 Gram, whose largest diagonal entry is 614,663 times the
  median, that is 4.9 times the median diagonal entry (the Grams captured on
  the 7700X), which weakens the error compensation of every ordinary column.
- Estimate: none (it changes GPTQ's result).
- Risks: fitting the 12 capture pages more closely; the proxy is in-sample.
- First measurement: `tools/w8_proxy.py --damp` with smaller values on the
  W2 Grams, then agree.

### W8A8 rotated INT8 prefill

- Why: prefill is compute-bound (about 6.1 TFLOP for the journal page:
  2.2 in the projections, 3.9 in attention), and INT8 dot products (VNNI)
  could speed up the projections, 1.3 s of fast mode's 2.9 s prefill
  ([PERFORMANCE.md](PERFORMANCE.md)). On book pages with `--pipeline` the
  prefill sets the pace, so its time is the run's.
- Estimate: at most those 1.3 s; BF16 projections (`--tune
  prefill-bf16=all`) already save 0.6 s of them and were rejected for +17%
  KL.
- Risks: per-token INT8 activations meet the massive channels (one layer-10
  W2 input channel carries 614,663 times the median energy). A Hadamard
  rotation spreads that channel but the others then share an 8-bit step it
  sizes; likely lossier than the rejected BF16 projections.
- First measurement: simulate W8A8 with rotated activations in the GPU
  harness (`research/phase4-hillclimb/attempt3/gpu_harness.py`) on the anchor
  pages; go further only if its KL beats the BF16 projections'.

### Dead FFN channel pruning

- Why: in the pinned checkpoint 6,108 of the 50,688 FFN channels (12.1%;
  none in layer 0, 18–19% in layers 16–18) have a W2 column norm below
  1/1000 of their layer's median, and their gate and up rows (interleaved in
  `w13`) are a median 460–500 and 1,600–2,900 times below their layer's: a
  dead channel's output is about 1e-12 of a typical one's.
- Estimate: 12% of the FFN's bytes and FLOPs, 8% of the body: about 0.3 ms
  of fast mode's 8.5 ms per token (12% of its 2.3 ms in W13 and W2) and
  0.55 ms of near-exact's 15 ms, 3–4% of a journal page in either mode.
- Risks: rounding-level, not bitwise, so agreement decides; the W8 overlay
  must be rebuilt on the pruned matrices; per-layer FFN widths need the
  loader (it requires 2,304), the packed format and the panel shapes to
  accept them. Exact mode stays unpruned.
- First measurement: a research profile that zeroes those channels at load
  (same shapes, no speedup) and agree for near-exact and fast; flips and KL
  must stay within noise before any change of shape.

### `--max-dimension 1280` per book

- Why: on 64 calibration pages in near-exact mode 1280 took 20% less time
  than 1536. Its CER was better on the 48 pages that end at EOS at 1536,
  1280 and 1024 but worse on the 55 that end at EOS at 1536, with 9
  repetition stops against 7
  ([PERFORMANCE.md](PERFORMANCE.md#rejected-or-unadopted) has the numbers).
- Estimate: about 20% of a book's time where its print reads as well at
  1280, on top of margin cropping.
- Risks: more loops and lost small print; the per-page router failed the
  held-out English gate (+0.29 pt). A choice per book needs a check per
  book.
- First measurement: compare a few dozen pages of the book at 1536 and at
  1280, as for margin cropping; adopt it for that book if no page lost
  content and no loop was added.
