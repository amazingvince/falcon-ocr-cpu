# Decode attention sparsity

A block-skipping decode kernel is not worth building for fast mode. Decode
attention over the image prefix is diffuse: 90% of a KV group's image mass
needs 145 to 184 of the page's roughly 210 image blocks of 32 positions. An
oracle that knows every block's exact softmax mass can skip 1.4% of the image
blocks (1.4% of the decode KV bytes) at an attention-output error of about
half the Q8 KV cache's; skipping 8.4% already costs four times Q8's error.
Every cheap bound tested costs more bytes for its summary than it saves at
that fidelity.

This GPU study runs the checkpoint's own PyTorch code in FP32 on 13 pages,
512 decode steps each.

| File | What |
|---|---|
| `decode_attention_sparsity.py` | The study: `run` (GPU, per page) and `summarize` (numpy) |

The recorded run of 2026-11-03 is in the ignored
`artifacts/decode-attention-sparsity-2026-11-03/`. It holds `run.sh` (the
three jobs), their logs, one JSON and one NPZ per page (every per-step
array) and `summary.json` and `summary.md` (the tables below).

## To repeat

The run needs CUDA, the FP32 checkpoint directory with its Python sources
(`artifacts/model`, from `scripts/fetch_reference.py`), and PyTorch with
`transformers`, `einops`, `triton`, `requests`, Pillow and `safetensors`.
Run from the repository root and write into a new directory:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 python3 \
  research/decode-attention-sparsity/decode_attention_sparsity.py run \
  --model artifacts/model --output artifacts/decode-attention-sparsity/run \
  artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png
python3 research/decode-attention-sparsity/decode_attention_sparsity.py \
  summarize artifacts/decode-attention-sparsity/run \
  --cpu-reference artifacts/verify/pr/agree/reference.json \
  --output artifacts/decode-attention-sparsity/summary.json
```

`--cpu-reference` (optional) names a `falcon-ocr-eval` FP32 report whose
tokens the GPU reference is compared with. A page takes about 2.2 minutes on
an RTX 5090: a 10 s reference, a 42 s statistics pass and twelve
teacher-forced policy passes of 4 to 12 s each.

## Method

**Pages.** The journal page `3f294b5e60a0c2d4` and twelve of the 52 anchor
pages, chosen to cover prose, formulas, tables, multi-column layouts,
newspapers, a slide and a colour textbook in English and Chinese:
`3207aa8d68652b99`, `4a756837631f4f69`, `754444afd4659b07`,
`1044f76c94729af0`, `02a1f828bad3aae2`, `63fb8f6a1ef3ce0d`,
`355f51c4a08c3807`, `1306da2487430257`, `8e0aa09f9c52d390`,
`0dfe20ce673bd8ec`, `f19e1efde5ef9f33` and `d4cfdb30756abda0`. Each page's
canonical RGB PNG is preprocessed by the checkpoint's `process_batch` at
`max_dimension` 1536. That gives 6,448 to 7,216 input tokens and 201 to 225
image blocks. Greedy decoding runs up to 512 tokens or EOS: 6,138 decode
steps in all, with three pages ending at EOS.

**Reference.** The model's own forward (FlexAttention with
`FLOAT32_PRECISION='ieee'`, TF32 off) decodes greedily and keeps every
decode step's log-probabilities. At the first and last decode steps a
wrapper around the decode FlexAttention call rebuilds every layer's output
from `softmax([sink, q·k/8])` over the cached keys and compares it with
FlexAttention's output times `sigmoid(lse - sink)`.

**Statistics pass.** A teacher-forced re-run of the reference tokens
computes attention exactly from the cached keys and values. At every step,
layer and query head, the probabilities over the sink and all keys are split
into the image positions and the rest. The image positions are grouped into
consecutive blocks of 32 in cache order. The rest covers the class token, the
four registers, the image end, the prompt and the generated text. A kernel
always reads those positions; their mass is reported as "other". A block's
mass for a KV group is the larger of its two query heads' sums, because a
skipped block is skipped for both heads of the GQA pair.

- *Oracle.* Drop the image blocks whose group mass is below tau and
  renormalize the softmax (the sink stays). The pass reports the skipped
  fraction, the dropped mass and the relative error of the attention output
  against the exact one. "Head" error uses one head's 64 values; "layer"
  error uses the 1,024 values of all 16 heads. "Values-only" keeps the full
  denominator. That models a two-pass kernel that reads every key and then
  skips the values of light blocks.
- *Q8.* The same output from keys and values quantized as the runner's Q8
  split cache: absmax codes per 32 values (each key half, each value half)
  against a BF16 scale rounded up, codes rounded to nearest even
  (`src/quant/kv/codec.rs`; the quantizer matches
  `research/kv-rotation/kv_rotation_study.py` exactly).
- *Bounds.* Each bound is an upper bound on a block's max score `q·k/8`,
  computed from a per-block summary or sketch.
  - `ball`: `q·c + |q| r`, with `c` the BF16 mean key and `r` the largest
    distance from `c`.
  - `split`: the smaller of `ball` and the same bound applied to the
    temporal and spatial key halves separately.
  - `box`: `sum_d max(q_d lo_d, q_d hi_d)`, with per-dimension minima and
    maxima rounded outward to BF16.
  - `sketch4` and `sketch2`: a 4-bit or 2-bit absmax sketch of every key,
    with its worst-case error added.

  A kernel reads the always-read positions first, then visits image blocks
  in descending bound order. It skips the rest once
  `bound < lse_run + ln(tau / 32)`, where `lse_run` is the log-sum-exp of
  the sink and every score read so far. A skipped block then holds at most
  tau of the final mass. `ball-max` is `ball` with the test as first posed,
  the running max score in place of `lse_run`
  (`exp(-delta) * 32 <= tau`). No bound was ever below a block's true
  maximum (0 violations in 6,138 steps x 22 layers x 16 heads x ~210 blocks
  for each bound).
- *Locality.* For each step, layer and group the pass records the most
  attended block, how many blocks hold 90% and 99% of the image mass, and
  the image mass per patch row.

**Bytes.** A Q8 split record is 170 bytes per position and KV group: the
temporal key half shared by the pair, each head's spatial half, the value,
and five BF16 scales. A 32-position block therefore costs 5,440 bytes per
group, and a position costs 29,920 bytes over the 22 layers. Summary costs
per block and group, with the temporal half shared, are:

- `ball`: 200 bytes (3.7% of the block).
- `split`: 212 bytes (3.9%).
- `box`: 384 bytes (7.1%).
- `sketch4`: 1,728 bytes (31.8%).
- `sketch2`: 960 bytes (17.6%).

Keys are per query head (each head rotates its own spatial half), so a
group's summary covers 96 key values, not 64. The net saving is the bytes of
skipped records minus the summaries of every image block, as a share of the
whole decode KV stream: image and always-read positions, averaged over
steps.

**Policy passes.** Teacher-forced runs apply one policy at every decode step
in every layer: the Q8 cache (all positions, prefix and generated), an
oracle drop, a bound-based skip, or Q8 plus a skip. They count argmax flips
against the reference tokens and the mean KL divergence from the
reference's log-probabilities over all decode steps. This is the GPU
analogue of the CPU agreement metric. The policy's own cache is used for
generated positions.

## Results

**Verification.** Rebuilt outputs match FlexAttention to a max abs error of
1.7e-5 on outputs whose largest values are 28 to 56. The check covers all 22
layers at the first decode step of every page and at the last step of the
ten pages that ran to 512 tokens. The log-sum-exp matches to 1.9e-6. The
exact teacher-forced recomputation gives 0 flips, a mean KL below 1e-9 in
magnitude, and log-probabilities within 1.6e-4 of the reference. For all
12 anchor pages the GPU reference tokens equal the CPU exact-mode tokens of
`artifacts/verify/pr/agree/reference.json` over their full length.

The run used the `misc` conda environment (torch 2.13.0+cu130,
transformers 5.19.0, Pillow 11.3.0). The checkpoint's code is loaded through
transformers' remote-code loader.

**Oracle versus Q8.** Means are over 13 pages, all steps, 22 layers and 8
groups (skip) or 16 heads (error); errors are relative.

| tau | Image blocks skipped | Dropped mass, mean / p99 | Head error, mean / p99 | Layer error, mean / p99 | Values-only layer error |
|---|---:|---:|---:|---:|---:|
| 1e-2 | 99.6% | 0.265 / 0.731 | 0.369 / 2.00 | 0.356 / 1.29 | 0.422 |
| 3e-3 | 91.3% | 0.206 / 0.473 | 0.279 / 1.34 | 0.253 / 0.779 | 0.328 |
| 1e-3 | 39.1% | 3.9e-2 / 0.134 | 6.1e-2 / 0.307 | 6.3e-2 / 0.207 | 8.7e-2 |
| 3e-4 | 8.4% | 2.8e-3 / 2.4e-2 | 8.5e-3 / 0.121 | 9.2e-3 / 7.4e-2 | 1.1e-2 |
| 1e-4 | 1.4% | 1.8e-4 / 4.7e-3 | 8.9e-4 / 2.5e-2 | 1.0e-3 / 1.5e-2 | 1.1e-3 |
| 1e-5 | 0.0004% | 7e-9 / 0 | 7e-8 / 0 | 1e-7 / 0 | 1e-7 |
| Q8 cache | - | - | 2.5e-3 / 1.2e-2 | 2.3e-3 / 6.5e-3 | - |

Q8's output error, 2.3e-3, is below its 5.9e-3 block error because
independent rounding errors average out over positions. Dropping is a
systematic error: each light block still carries about 1e-3 of the mass,
while the output averages near-cancelling values. Of the taus that skip
anything, only 1e-4 stays below the Q8 error on average, and its p99 head
error (2.5e-2) is still twice Q8's. At 3e-4 the mean error is already four
times Q8's.

**Bounds.** Skip fraction of image blocks and net bytes saved on image
blocks:

| Bound | Summary share | Skip at 1e-3 | Net at 1e-3 | Skip at 1e-2 | Net at 1e-2 | Layer error at 1e-2 |
|---|---:|---:|---:|---:|---:|---:|
| `ball-max` | 3.7% | 0.000% | -3.7% | 0.09% | -3.6% | 1.9e-4 |
| `ball` | 3.7% | 0.10% | -3.6% | 3.6% | -0.1% | 6.2e-3 |
| `split` | 3.9% | 0.26% | -3.6% | 5.0% | +1.1% | 9.3e-3 |
| `box` | 7.1% | 0.73% | -6.3% | 11.1% | +4.0% | 2.6e-2 |
| `sketch4` | 31.8% | 2.1% | -29.7% | 39.2% | +7.4% | 7.1e-2 |
| `sketch2` | 17.6% | 0.00% | -17.6% | 0.17% | -17.5% | 4.4e-4 |
| oracle | 0 | 39.1% | +39.1% | 99.6% | +99.6% | 0.356 |

At tau = 1e-4 no bound skips more than 0.012% of blocks.

On the whole decode KV stream, the oracle saves 1.37% at tau = 1e-4 and
37.6% at 1e-3. The values-only two-pass kernel saves 0.55% and 15.0%. Every
bound loses 3.4% to 30.6% at either tau.

Skipping needs a block's bound to sit more than ln(32 / tau) = 10.4 nats
(at 1e-3) below the running log-sum-exp, on scores bounded by +-8 (the keys
and queries are RMS-normalized, |k| = |q| = 8). Each head's golden RoPE also
rotates the spatial key half by patch position, so even repeated content
gives different keys within a block. The block radii were not recorded. The
skips that do happen at 1e-2 are paid for with errors 3 to 32 times Q8's.

**Teacher-forced flips and KL** over 13 pages and 6,138 decode steps:

| Policy | Image blocks skipped | Flips | Pages with flips | KL mean | KL p99 |
|---|---:|---:|---:|---:|---:|
| exact recomputation | 0 | 0 | 0 | ~1e-10 | 1.3e-7 |
| Q8 KV (fast mode's floor) | 0 | 5 | 3 | 1.96e-5 | 3.9e-4 |
| oracle 1e-3 | 34.8% | 47 | 9 | 3.30e-3 | 6.7e-2 |
| oracle 3e-4 | 8.3% | 10 | 5 | 1.58e-4 | 3.0e-3 |
| oracle 1e-4 | 1.4% | 2 | 1 | 1.79e-5 | 8.5e-5 |
| values-only 1e-4 | 1.4% | 2 | 1 | 1.56e-5 | 8.1e-5 |
| `ball-max` 1e-3 | 0.0001% | 0 | 0 | ~1e-10 | 1.4e-7 |
| `split` 1e-3 | 0.26% | 0 | 0 | 2.3e-7 | 3.8e-6 |
| `box` 1e-3 | 0.73% | 2 | 1 | 3.75e-6 | 4.4e-5 |
| `sketch4` 1e-3 | 2.1% | 4 | 3 | 1.81e-5 | 3.2e-4 |
| Q8 + oracle 1e-4 | 1.4% | 4 | 3 | 4.84e-5 | 4.5e-4 |
| Q8 + `sketch4` 1e-3 | 2.1% | 6 | 3 | 3.70e-5 | 6.8e-4 |

On top of Q8, the oracle at 1e-4 and `sketch4` at 1e-3 raise the mean KL by
2.5 and 1.9 times for 1.4% and 2.1% of the image blocks. The skip
fractions in this table come from the policy runs themselves.

## Where the mass goes

- **Layers.** In layers 1 to 12, 74% to 91% of the mass goes to the
  always-read positions and 9% to 24% to the image. Oracle skip at 1e-3 is
  43% to 88% there, but at 1e-4 only layers 9 to 12 skip more than 1%
  (3% to 13%). Layer 0 gives the sink 48% of its mass, but its image mass
  is diffuse: 10% skip at 1e-3, none at 1e-4. Layers 13 to 21 give 35% to
  49% to the image and skip 2% to 24% at 1e-3 and none at 1e-4.
- **Heads.** Within a layer the KV groups differ widely. For example, oracle
  skip at 1e-3 ranges from 26% to 96% across layer 11's groups and from 5%
  to 61% across layer 16's.
- **Steps.** Sparsity grows slowly as text accumulates. Oracle skip at
  1e-3 / 1e-4 is 31.7% / 0.7% over steps 0 to 31, 35.1% / 1.0% over 32 to
  127, 38.2% / 1.3% over 128 to 255 and 42.3% / 1.8% over 256 to 511.
- **Pages.** At 1e-3, oracle skip ranges from 25% on the dense Chinese
  newspaper `1306da2487430257` (0.1% at 1e-4) to 46% on the colour textbook
  `d4cfdb30756abda0` (3.1%). Prose, formula, table and multi-column pages
  sit between 37% and 44% (0.8% to 2.3% at 1e-4).
- **Locality.** A few heads follow the text being read. On every page, some
  group's most attended patch row tracks the step, with a Pearson
  correlation of 0.74 to 0.998. The most local groups of layers 11, 13, 16
  and 21 keep 21% to 32% of their image mass within +-2 patch rows of that
  pointer row, against 5.2% for uniform attention. The pointer's own group
  is included in its layer. Layer averages stay at 5% to 15%: most groups
  attend to the whole page, so a locality window would drop mass that
  matters.

## Recommendation

Do not build a block-skipping decode kernel for fast mode.

- **The oracle is the ceiling.** Even a perfect bound saves only 1.4% of
  the decode-attention bytes at an output error half of Q8's. Decode
  attention takes 4.2 ms of a fast-mode step of 8.5 ms
  ([PERFORMANCE.md](../../docs/PERFORMANCE.md)), so that is about 0.06 ms
  per token before any summary reads or second pass. In teacher forcing,
  combined with Q8, it still raises the mean KL 2.5 times.
- **The practical bounds lose bytes.** At tau = 1e-3 the best,
  `sketch4`, skips 2.1% of blocks for a sketch that costs 32% of the
  record. The cheap centroid bound (`split`, 3.9% overhead) skips 0.26%.
  The test as first posed (`ball-max`) skips nothing.
- **The summary-free design saves nothing either.** The two-pass,
  values-only design (exact Q8 scores, then skip the values of light
  blocks) needs no summary. It saves 0.55% of the bytes at 1e-4.
- **Fidelity cost.** Any tau that saves real bytes (3e-4 and above) costs
  4 to 28 times Q8's output error and 8 to 170 times its KL.

If decode-attention bytes are to come down, the diffuse, low-weight image
prefix suggests coarser value or key precision rather than skipping:
rounding error averages out across positions, while dropped blocks bias the
output. That is a separate experiment.
