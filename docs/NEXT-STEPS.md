# Next steps

Status: current as of 2026-11-03. Unresolved qualification and hardware
work, followed by research proposals.

## Still open

- **Weight variants and containers.** [Additional dtypes and quantizers](https://github.com/amazingvince/falcon-ocr-cpu/issues/15)
  and [GGUF export/loading](https://github.com/amazingvince/falcon-ocr-cpu/issues/16)
  are separate follow-ups to the FP32, INT16 and GPTQ INT8 catalog.

- **Reference-dependent gates.** `tools/check.sh --smoke` needs the GPU
  smoke outputs (`metadata.json`, `trace.safetensors` from the frozen
  reference), as `tests/gpu_parity.rs`, `tests/cache_layout.rs` and
  `tests/weight_layout.rs` do, but the self-hosted `weights` CI job, whose
  machine holds those outputs, runs only the three tests. The reference's
  preflight pins the RTX 4090 the outputs were recorded on, so no other
  machine can remake them.
- **The 55-page anchor.** The Ryzen 7 7700X agreement numbers in
  [MODES.md](MODES.md#the-metric) (exception columns, rotated caches) are on
  that host's 52-page anchor, so they compare arms with each other. Scored
  against upstream's `artifacts/phase4/checks/calibration-reference.json`
  ([DEVELOPMENT.md](DEVELOPMENT.md#token-agreement)), the same arms would
  give 24,262-step numbers comparable with the modes table.
- **Default qualification.** Held-out gates decide defaults that change
  tokens: margin cropping for books, `--max-dimension 1280`, a rotated cache
  and exception columns. The gate method is in
  [MODES.md](MODES.md#the-metric).
- **Windows and aarch64.** The book runs, the page pipeline and the KV-cache
  timings ([PERFORMANCE.md](PERFORMANCE.md#on-a-ryzen-7-7700x)) were
  measured on one Linux desktop; no aarch64 machine has run the model with
  weights ([PORTABILITY.md](PORTABILITY.md)).

## Proposals (not implemented)

These proposals are not scheduled work or accepted defaults. Their speed
estimates are projections from measurements, not measured speedups.
Agreement does not support pursuing a vector nibble unpack for `q4r`
(its 4-bit codes flip 5 times as often as fast mode's 8-bit cache) or a
faster near-exact on `q8r` (18 flips against near-exact's 1). See
[MODES.md](MODES.md#rotated-kv-cache-experimental).

### Exception columns as fast mode's default

Compare agreement at 8 and 16 exception columns and with QKV included,
then run a fresh held-out English gate before changing the default or
republishing files. QKV carries 16% of the 8-bit cost and is untested.
The [exception-column results](MODES.md#exception-columns-experimental)
give the existing W2 proxy curve, fidelity and format support.

### GPTQ damping relative to the median

`tools/w8_variants.py` damps each activation Gram matrix by 1% of its mean
diagonal. In the layer-10 W2 Gram captured on the 7700X, whose largest
diagonal entry is 614,663 times the median, this adds 4.9 times the median
diagonal entry. It weakens error compensation for ordinary columns.

Smaller damping changes GPTQ's result; there is no estimated gain. A first
comparison would use `tools/w8_proxy.py --damp` on the W2 Grams, then token
agreement. The proxy is in-sample, so it can reward fitting the 12 capture
pages more closely without improving other pages.

Source: `artifacts/w8/gram/layers.10.feed_forward.w2.weight.gram.npy` on
the 7700X.

### INT16 prefill projections: what remains

The INT16 projections, fast mode's default since 2026-11-03
([MODES.md](MODES.md#int16-projections-and-static-softmax)), cut the
journal page's projection time from 2.4 to 1.2 s on the 7700X for +1.4% KL,
where BF16 projections cost +19%
([PERFORMANCE.md](PERFORMANCE.md#int16-projections-and-static-softmax)).
Two pieces are left:

- **A NEON kernel.** aarch64 keeps the FP32 panels for fast mode.
- **Exception columns in the INT16 epilogue.** A body with exception
  columns keeps FP32 panels, so it runs without the INT16 speedup.

INT8 activations, which would double the MAC rate again with `vpdpbusd`,
were simulated exactly and rejected: on two anchor pages they raise fast
mode's KL against FP32 up to 20-fold, and on QKV, WO and W13, which have
no outlier channels, the loss comes from the 8-bit step's resolution, so
neither a per-group outlier fallback nor a Hadamard rotation would rescue
them ([PERFORMANCE.md](PERFORMANCE.md#rejected-or-unadopted)).

### Dead FFN channel pruning

In the pinned checkpoint, 6,108 of the 50,688 FFN channels (12.1%; none in
layer 0, 18-19% in layers 16-18) have a W2 column norm below 1/1000 of
their layer's median. For these channels, per-layer median gate and up row
norms (interleaved in `w13`) are about 440-511 and 978-2,918 times below
their layer's medians. A dead channel's output is estimated at about 1e-12
of a typical one's.

Source: column and row L2 norms of the pinned checkpoint,
`artifacts/model/model.safetensors`.

Pruning projects a saving of 12% of the FFN's bytes and FLOPs, 8% of the
body: about 0.3 ms of fast mode's 8.5 ms/token (12% of its 2.3 ms in W13
and W2) and 0.55 ms of near-exact's 15 ms/token, or 3-4% of a journal page
in either mode. It is not bitwise, so token agreement decides.

A first comparison would zero these channels at load, keeping shapes and
speed unchanged, and measure flips and KL for near-exact and fast. Only if
both stay within noise would shapes change. The W8 overlay must then be
rebuilt; the loader (which requires FFN width 2,304), packed format and
panel shapes must accept per-layer widths. Exact mode would stay unpruned.

### `--max-dimension 1280` per book

The projected saving is about 20% of a book's time, on top of margin
cropping, where its print reads as well at 1280; the
[calibration comparison](PERFORMANCE.md#rejected-or-unadopted) shows the
loop and small-print risks. Compare a few dozen of that book's pages at
1536 and 1280, as for margin cropping, and adopt 1280 only if no page loses
content and no loop is added.

### Repetition stop for counting loops

The repetition stop compares tokens exactly, so a loop whose lines differ
only in an incrementing number never fires it: fast and near-exact mode
both write such rows to the token limit on a deep-fried anchor page
([MODES.md](MODES.md#loops-and-the-repetition-stop)). A stop that treats
tokens made only of digits as equal when it compares a token with the one
a period earlier would end such loops. Like the current stop it only
truncates, so it changes no token before the step where it fires. The
risk is legitimately numbered content: a numbered list or table whose rows
repeat apart from the number would be cut. A first comparison would check
on the calibration pages and the
[stress pages](../tests/fixtures/stress/README.md) that it never fires on a
page that ends at EOS, as the current stop was checked.
