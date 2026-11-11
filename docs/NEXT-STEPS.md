# Next steps

Status: current as of 2026-11-11. Unresolved qualification and hardware
work, followed by research proposals.

## Still open

- **Weight variants and containers.** [Additional dtypes and quantizers](https://github.com/amazingvince/falcon-ocr-cpu/issues/15)
  and [GGUF export/loading](https://github.com/amazingvince/falcon-ocr-cpu/issues/16)
  are separate follow-ups to the FP32, INT16 and GPTQ INT8 catalog.

- **Reference-dependent gates.** The [fixture download](DEVELOPMENT.md#tests)
  supplies the original GPTQ overlay and GPU smoke record for local checks.
  `tools/check.sh --smoke` and the bitwise trace test still use
  [Windows baselines](DEVELOPMENT.md#the-check). The tokenizer-decode,
  RoPE and draft-head parity checks need separate fixtures.
- **The 55-page anchor.** The Ryzen 7 7700X agreement numbers in
  [MODES.md](MODES.md#the-metric) (exception columns, rotated caches, fast
  mode's prefill kernels, near-exact's static softmax) are on that host's
  52-page anchor, so they compare arms with each other. Scored against
  upstream's `artifacts/phase4/checks/calibration-reference.json`
  ([DEVELOPMENT.md](DEVELOPMENT.md#token-agreement)), the same arms would
  give 24,262-step numbers comparable with the modes table, whose
  near-exact row predates the static softmax.
- **Default qualification.** Held-out gates decide defaults that change
  tokens: margin cropping for books, `--max-dimension 1280`, a rotated
  cache, exception columns and whole layers at Q8. The gate method is in
  [MODES.md](MODES.md#the-metric). Fresh pages are nearly spent: after
  [near-exact's gate](MODES.md#near-exacts-held-out-gate), 60
  OmniDocBench document families stay fresh (tables 54, other 6), none of
  them English, so a gate on any other category or on English pages needs
  pages from another source. The spent page manifests and locks are
  [archived with checksums and source mappings](../reference/prefill-gate-provenance-v2.md).
- **Windows and aarch64.** The book runs, the page pipeline, the KV-cache
  timings and peak memory
  ([PERFORMANCE.md](PERFORMANCE.md#on-a-ryzen-7-7700x)) were measured on one
  Linux desktop; no aarch64 machine has run the model with weights
  ([PORTABILITY.md](PORTABILITY.md)). The generic static-softmax NEON tiles
  are test instantiations, not recognition defaults; enabling them needs
  native token qualification.

## Proposals (not implemented)

These proposals are not scheduled work or accepted defaults. Their speed
estimates are projections from measurements, not measured speedups.
Agreement does not support pursuing a vector nibble unpack for `q4r`
(its 4-bit codes flip 5 times as often as fast mode's 8-bit cache) or a
faster near-exact on `q8r` (18 flips against near-exact's 1). See
[MODES.md](MODES.md#rotated-kv-cache-experimental).

### Exception columns as fast mode's default

The comparison at 8 and 16 exception columns and with QKV is
[done](MODES.md#exception-counts-and-damping): mean KL falls, but flips and
pages identical to exact mode do not improve, and QKV gains nothing. A
default therefore needs a reason beyond mean KL, a republished v2 file and a
fresh held-out English gate.

### `--tune kv-layers` as near-exact's default

[Whole layers at Q8](MODES.md#whole-layer-q8-in-the-q16-cache-experimental)
would gain about 1.5 to 3% of a page
([PERFORMANCE.md](PERFORMANCE.md#kv-caches-and-exception-columns)). As a
default it would regenerate near-exact's fixtures and supersede its
held-out results.

### GPTQ damping relative to the median

`tools/w8_variants.py` damps each activation Gram matrix by a fraction of
its mean diagonal, 1% by default. In the layer-10 W2 Gram captured on the
7700X, whose largest diagonal entry is 614,663 times the median, this adds
4.9 times the median diagonal entry. Smaller fractions do not help: on the
[10-page set](MODES.md#exception-counts-and-damping) 1e-4 to 3e-3 raise mean
KL 1.431 to 3.103 times. A damping tied to the median instead of the mean is
untried, and there is no estimated gain.

Source: `artifacts/w8/gram/layers.10.feed_forward.w2.weight.gram.npy` on
the 7700X.

### INT16 prefill projections: what remains

The INT16 projections, fast mode's default since 2026-11-03
([MODES.md](MODES.md#int16-projections-and-static-softmax)), cut the
journal page's projection time from 2.4 to 1.2 s on the 7700X for +1.2% KL,
where BF16 projections cost +19%
([PERFORMANCE.md](PERFORMANCE.md#int16-projections-and-static-softmax)).
One piece is left: a NEON kernel, since aarch64 keeps the FP32 panels for
fast mode.

INT8 activations, which would double the MAC rate again with `vpdpbusd`,
were simulated exactly and rejected: on two anchor pages they raise fast
mode's KL against FP32 up to 20-fold, and on QKV, WO and W13, which have
no outlier channels, the loss comes from the 8-bit step's resolution, so
neither a per-group outlier fallback nor a Hadamard rotation would rescue
them ([PERFORMANCE.md](PERFORMANCE.md#rejected-or-unadopted)).

### FFN channel pruning

In the pinned checkpoint, 6,108 of the 50,688 FFN channels (12.1%; none in
layer 0, 18-19% in layers 16-18) have a W2 column norm below 1/1000 of
their layer's median. For these channels, per-layer median gate and up row
norms (interleaved in `w13`) are about 440-511 and 978-2,918 times below
their layer's medians. A dead channel's output is estimated at about 1e-12
of a typical one's.

Source: column and row L2 norms of the pinned checkpoint,
`artifacts/model/model.safetensors`.

The prototype, at tag `checkpoint/ffn-prune-2026-11-06`, runs near-exact on
the checkpoint route; [PERFORMANCE.md](PERFORMANCE.md#rejected-or-unadopted)
has its measurements. It kept every token on the [10-page
set](MODES.md#the-metric), and fast mode with the channels below 0.1 zeroed
has KL 0.907 times base and flips 1 to 0. Pruning cannot reach the product
route without a new packed format: `body_shapes` takes one `ffn_dim`,
`ModelConfig::validate` pins 2304 and scales are shaped `input / 64`, so
per-layer widths need a new format string and republished files, which is
the upstream author's call. Two more constraints follow:

- **Width.** INT16 panels need `width % 64 == 0`, so each layer removes
  channels in multiples of 64: 6,080 in all at a threshold of 0.1 and 5,440
  at 1e-3.
- **The W8 overlay.** It must be rebuilt, because regrouping changes 44% of
  the kept W2 scales. Start from a threshold of 1e-3: 209 of the 6,605
  columns below 0.1 still hold live W8 codes, while all 6,108 below 1e-3 are
  already zero in the overlay.

Channels below each threshold, as a fraction of the layer's median W2 column
norm:

| Threshold | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Channels | 70 | 2,552 | 6,108 | 6,438 | 6,475 | 6,532 | 6,605 |

Exact mode would stay unpruned.

### `--max-dimension 1280` per book

The projected saving is about 20% of a book's time, on top of margin
cropping, where its print reads as well at 1280; the
[calibration comparison](PERFORMANCE.md#rejected-or-unadopted) shows the
loop and small-print risks. Compare a few dozen of that book's pages at
1536 and 1280, as for margin cropping, and adopt 1280 only if no page loses
content and no loop is added.
