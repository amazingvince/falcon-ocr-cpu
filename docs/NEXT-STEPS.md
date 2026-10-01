# Next steps

Status: current as of 2026-10-01. What the features of this release were
validated with, what still needs other hardware or a decision, and proposals
that are not implemented. The numbers live in the documents that own them;
this page links to them.

## Validated on real weights

On a Ryzen 7 7700X (8 cores, 16 threads, AVX-512 with BF16), Linux, with the
pinned checkpoint, the published packed files and a GPTQ overlay:

- **Every run path keeps a page's tokens.** With the published files and
  the default configuration (screened head, draft head), 16 calibration and
  book pages gave each page the same tokens run one after another, with
  `--pipeline`, `--batch-size 4`, `--batch-size 3 --pipeline` and
  `--max-dimension auto`, in near-exact and fast mode; so did every timed
  arm on a book (cropped arms against the cropped sequential run), and every
  cache on the journal page with and without speculation.
- **The ignored gates that need no GPU outputs pass**: `batch_parity`
  (including the default configuration with near-exact and fast weights),
  `batch_trace`, `decode_allocations`, `head_screen`, `margin_crop`,
  `negative_inputs` and `run_cli`. `modes` passes its packed round trip,
  repetition, routing and speculation tests. Its journal test fails here
  only on the overlay's hash, since the local overlay was rebuilt from the
  published fast file; the recorded journal tokens matched.
- **Throughput on a book** ([PERFORMANCE.md](PERFORMANCE.md#book-runs)):
  `--pipeline` gave 12–20% more pages per hour and `--batch-size 4` 2–8%;
  `--crop-margins` gave 18–21% more, and every page kept its words
  ([MODES.md](MODES.md#margin-cropping)). Fast mode with `--crop-margins
  --pipeline` and the decode team pinned to 8 threads read 701–709 pages per
  hour against 506–511. Beside the pipeline an automatic decode team now
  takes at most half of the threads: the tuner had once chosen 12 of 16 and
  left the prefill 4.
- **Long runs.** `--keep-going` records a truncated PNG's error and goes on;
  without it the run stops at that page. An interrupted run resumed with
  `--resume` runs only the missing pages and repairs a cut-off last record
  (`tests/run_cli.rs`).
- **Escalation.** Over the 64 calibration pages in fast mode, `--escalate`
  reread the 9 pages that the repetition stop ended, each with a plain
  near-exact run's tokens
  ([MODES.md](MODES.md#loops-and-the-repetition-stop)).
- **Agreement against FP32** on 52 calibration pages
  ([MODES.md](MODES.md#the-metric)): `q8r` lowers fast mode's KL by 4%
  and `q4r` multiplies it by 30
  ([MODES.md](MODES.md#rotated-kv-cache-experimental)); 4 exception columns
  per W2 lower it by 39%
  ([MODES.md](MODES.md#exception-columns-experimental)). Only `q4r` costs
  measurable decode time, 15% per token
  ([PERFORMANCE.md](PERFORMANCE.md#kv-caches-and-exception-columns)).
- **Packed files.** Re-packing near-exact from the checkpoint, and fast from
  an overlay reconstructed from the published file, gives the published
  files' metadata values and tensors. A v2 file with exception columns
  passes `--verify-model-file` and gives its overlay's tokens.

## Still open

- **The GPU smoke outputs** (`metadata.json`, `trace.safetensors` from the
  frozen reference) gate `tests/gpu_parity.rs`, `tests/cache_layout.rs`,
  `weight_layout` and `tools/check.sh --smoke`. The reference's preflight
  pins the RTX 4090 it was recorded on, so the self-hosted `weights` runner
  is where these run.
- **The anchor at 55 pages.** The agreement numbers here are on 52
  calibration pages (the local calibration lock holds 64; upstream's anchor
  has 55), so they compare arms with each other. The `weights` runner's
  `artifacts/phase4/checks/calibration-reference.json` gives the 24,262-step
  numbers the modes table quotes.
- **Held-out gates** decide every default that would change tokens: margin
  cropping for books, `--max-dimension 1280`, a rotated cache and exception
  columns. Held-out pages are single-use, so each needs a pre-registered
  budget before its run.
- **Windows and aarch64.** Every timing here is Linux on one desktop; no
  aarch64 machine has run the model with weights
  ([PORTABILITY.md](PORTABILITY.md)).

## Setup

The commands below reproduce the agreement numbers from the repository root
(bash; `falcon-ocr-eval` from `cargo build --release --locked`); the timing
tools are listed in [PERFORMANCE.md](PERFORMANCE.md#measuring).

```sh
B=target/release
W8=artifacts/model/w8-gptq.safetensors
# The anchor pages: the calibration pages outside the GPTQ capture set.
python - > anchor-pages.txt <<'EOF'
import json
from pathlib import PureWindowsPath as P  # reads / and \
capture = {P(path).parent.name for path in open("tools/gptq-calibration-pages.txt").read().split()}
for page in json.load(open("artifacts/phase4/checks/calibration-reference.json"))["inputs"]:
    if (name := P(page["path"]).parent.name) not in capture:
        print(f"artifacts/corpus/v3/{name}/canonical-rgb.png")
EOF
# FP32 top-K log-probabilities along the reference tokens, to score KL against
$B/falcon-ocr-eval --profile reference agree $(cat anchor-pages.txt) \
  --reference artifacts/phase4/checks/calibration-reference.json --max-steps 512 \
  --report fp32.json --dump-topk fp32-topk.json
# One agreement arm: flips and KL against FP32 (tools/agree_queue.sh runs several)
AGREE_BIN=$B/falcon-ocr-eval AGREE_ARGS="--reference-topk fp32-topk.json" \
  bash tools/agree_queue.sh agree anchor-pages.txt 512 fast=w8-body-kv-q8=$W8
```

The FP32 arm has 0 flips by construction; an arm's `kl_mean` is
KL(FP32 ‖ arm) over the top 32 tokens per step.

## Proposals (not implemented)

Estimates come from measured numbers in this repository; they are not
measurements. Two earlier proposals were settled by the agreement runs and
are dropped: a vector nibble unpack for `q4r` (its 4-bit codes flip 5 times
as often as fast mode's 8-bit cache) and a faster near-exact on `q8r` (an
8-bit cache flips 18 times where near-exact flips once;
[MODES.md](MODES.md#rotated-kv-cache-experimental)).

### Exception columns as fast mode's default

- Why: 4 FP32 columns per W2 cut fast mode's KL against FP32 by 39% (3.04e-4
  → 1.84e-4 on the 7700X anchor) for 271 KB more per decode step, 0.15% of
  its weight bytes ([MODES.md](MODES.md#exception-columns-experimental)).
- Estimate: no flip gain at this anchor's size (64 → 63); the KL gain is
  what a held-out gate would have to confirm in output quality.
- Risks: a new packed format (`falcon-ocr-kernel-v2`) and republished files;
  the proxy curve flattens after 4 columns (7.14e-4 → 6.37e-4 at 16), and
  other matrices (QKV carries 16% of the 8-bit cost) are untested.
- First measurement: agree at 8, 16 and with QKV included, then the
  held-out English gate with a pre-registered budget.

### GPTQ damping relative to the median

- Why: `tools/w8_variants.py` damps each Gram by 1% of its mean diagonal; in
  the layer-10 W2 Gram, whose largest diagonal entry is 614,663 times the
  median, that is 4.9 times the median diagonal entry (this host's Grams),
  which weakens the error compensation of every ordinary column.
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

- Why: on 64 calibration pages in near-exact mode, 1280 took 20% less time
  than 1536; CER against the ground truth went 16.42 → 15.69% on the 48
  pages that end at EOS at 1536, 1280 and 1024, but 19.10 → 20.67% on the 55
  that end at EOS at 1536, with 9 repetition stops against 7
  (`research/phase4-hillclimb/attempt3/HILLCLIMB.md`, T14).
- Estimate: about 20% of a book's time where its print reads as well at
  1280, on top of margin cropping.
- Risks: more loops and lost small print; the per-page router failed the
  held-out English gate (+0.29 pt). A choice per book needs a check per
  book.
- First measurement: compare a few dozen pages of the book at 1536 and at
  1280, as for margin cropping; adopt it for that book if no page lost
  content and no loop was added.
