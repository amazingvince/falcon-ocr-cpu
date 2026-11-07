# Performance

Status: current as of 2026-11-07. Host, unless a section says otherwise:
Ryzen 9 7950X (16 cores, 32 threads, DDR5, AVX-512 with BF16), Windows 11.
Page: the journal benchmark page, 6,544 input tokens. Earlier measurements on
the tiny 256×128 fixture are archived in
`research/benchmarks/docs/PERFORMANCE.md`.

Raw 7950X receipts live under `artifacts/phase4/` on that host; they are not
included in the repository. [DEVELOPMENT.md](DEVELOPMENT.md) gives the
measurement commands and timing protocol.

## The roofline

Decode streams every body weight, the vocabulary head (or its INT8 screen)
and the whole KV cache once per token, so its floor is bytes per token
divided by achievable read bandwidth; `falcon-ocr doctor --probe` measures
that bandwidth (about 50-54 GB/s here) and prints the floor for the plan.
Prefill is compute-bound (about 6.1 TFLOP per full page).

| Mode | Weights + head per token | KV per position | Decode floor at 52 GB/s | Measured, journal page |
|---|---:|---:|---:|---:|
| exact | 876 MB | 113 KB | 31 ms | 29 ms |
| near-exact | 401 MB | 58 KB | 15 ms | 15 ms |
| fast | 233 MB | 30 KB | 8.2 ms | 8.5 ms |

(Floors add the 6,544-position cache: 740, 380 and 196 MB.) Decode runs at
about 90% of the bandwidth floor in every mode; the remaining decode lever is
fewer bytes, and the 4-bit formats cost too much fidelity for this model.

Where a fast-mode page went with its earlier prefill kernels (prefill
2.9 s, decode 8.5 ms/token): prefill attention 1.3 s (BF16 tiles, online
softmax), projections 1.3 s (FP32 panels), split/RoPE/append 0.1 s; decode
attention 4.2 ms, W13 1.5, head 1.0, W2 0.8, QKV 0.7, WO 0.4 ms. The FP32
prefill attention tiles run QK at about 75% and PV at about 90% of FP32
peak. The panel GEMM throughput probe measures 1.7-1.8 TFLOP/s at prefill
shapes. FP32 prefill takes about 4.1 s against fast mode's 2.9 s with BF16
attention. Fast mode's current defaults, INT16 projections and the static
softmax, halve the projections and shorten the attention; their
measurements are on the Ryzen 7 7700X
([below](#int16-projections-and-static-softmax)).

## Accepted changes (journal page, interleaved fresh processes, 3 rounds)

| Change | Effect |
|---|---|
| Split FP32 cache for exact mode (bitwise) | 42.8 → 39.6 s |
| Hybrid threads: prefill on all logical CPUs, decode on physical cores | fast -9%, exact -4% |
| Fast exp (portable polynomial, token-identical) in prefill attention | prefill 7.6 → 6.6 s |
| Screened INT8 head (exact) | head 3 → 1 ms per token |
| Spin team for decode loops | about 3 ms per token |
| 16-bit body weights + 16-bit KV (near-exact) | exact 38.0 → 21.9 s, 1 flip |
| GPTQ act-order overlay (fast) | 8.6 → 2.7 flips per 1,000 steps at the same speed |
| Automatic decode team size | decode 8.8 → 8.5 ms/token here; chooses 12 of 16 cores |
| Speculative decoding with the adaptive policy | looping page 44 → 19 s, table page 45 → 30 s, prose ±0 |
| Panel GEMM for quantized prefill projections | projections 1.73 → 1.39 s |
| Fused prefill QKV row (split, norms, RoPE, append in one pass) and folded norms/residuals | prefill 4.27 → 4.08 s |
| BF16 prefill attention with fused AVX-512 softmax (fast mode) | attention 2.44 → 1.28 s, flips 64 → 63 |
| Kernel-ready packed files | load 2.2 s → 10 ms, peak memory -1.1 GB |
| 16-lane AVX-512 prefill tiles (bitwise) | attention -9%; under 1% end to end on Zen 4 |
| BF16 scales in the Q8/Q16 caches | decode -2% |
| Multi-row quantized dots and shared record decoding for verify steps | 5-row step 30 → 16 ms |
| INT16 prefill projections and static softmax (fast mode; Ryzen 7 7700X, [below](#int16-projections-and-static-softmax)) | transformer prefill 5.28 → 3.65 s, flips 67 → 66 |
| Static softmax in the FP32 prefill tiles (near-exact; Ryzen 7 7700X, [below](#near-exact-prefill-attention)) | transformer prefill 7.91 → 7.16 s, flips 1 → 1 |
| Each layer's KV sealed right after its prefill attention, lazily zeroed forward buffers and split tail, fixed glibc thresholds (Ryzen 7 7700X, 300 tokens, [below](#peak-memory)) | Peak RSS 1,864 → 1,095 MB near-exact, 1,699 → 754 MB fast, 2,216 → 1,918 MB exact; tokens identical |

## Decode exponential

The portable polynomial exp in fast mode's decode attention (8-bit caches
under `ExpMode::Fast`; `--tune decode-exp`) saves verification work:

| Rows per verification step | Attention time change |
|---|---:|
| 2 | -3% |
| 3 | -7% |
| 4 | -9% |
| 5 | -10% |

Single-row steps are unchanged. With the draft head, page totals change by
-2.1% on calibration pages and -0.6% on held-out pages. Flips against FP32
change from 92 to 94 of 78,282 steps.

## Trained draft head

Fast mode, packed model, 12 decode threads on the Ryzen 9 7950X: 16 English
held-out pages, two rounds on a quiet machine. Every arm preserves the
no-speculation tokens:

| Drafter | Page total | Decode | Decode speedup vs none | Acceptance |
|---|---:|---:|---:|---:|
| No speculation | 184.5 s | 139.9 s | 1.00× | |
| N-gram drafts (`--drafter ngram`) | 172.7 s | 128.0 s | 1.09× | 34% |
| Stage-1 head, FP32 drafter KV, confidence 0.45 | 153.9 s | 109.2 s | 1.28× | 76% |
| Stage-2 head, INT8 drafter KV, confidence 0.35 | 139.9 s | 95.1 s | 1.47× | 75% |
| Stage-2 head, rank-384 vocabulary head | 139.2 s | 94.5 s | 1.48× | 72% |

A separate run compares the stage-2 initializer with the published stage-3
head, continued on 48k pages. The latter cuts held-out page time another
1.9%; on the calibration pages its total changes from 182.0 to 176.4 s:

| Drafter | Page total | Decode speedup vs none | Acceptance |
|---|---:|---:|---:|
| No speculation | 181.2 s | 1.00× | |
| Stage-2 head | 138.3 s | 1.46× | |
| Published stage-3 head | 135.7 s | 1.50× | 76% |

The published head gives 1.5-1.9× decode speedup across modes against no
speculation and cuts page totals by 21-26% against n-gram drafts. The other
modes use the same 16 pages and preserve tokens in every arm:

| Mode | No speculation | N-gram drafts | Published head | Decode speedup vs none | Acceptance |
|---|---:|---:|---:|---:|---:|
| Near-exact (2 rounds) | 334.8 s | 277.4 s | 208.1 s | 1.87× | 77% |
| Exact (1 round) | 612.6 s | 531.0 s | 392.5 s | 1.67× | |

Source: the 2026-09-26 comparison and the fast-mode `cpu-ab/` records;
their location, training stages and recipe are in
[research/draft-head](../research/draft-head/README.md).

Calibration sweeps use 16 English pages. INT8 drafter keys and values reduce
decode time (-1.7%) against FP32 with unchanged acceptance. Image attention,
vocabulary size, low-rank heads, attention windows and gating changes each
gain 2% or less:

| Alternative | Measurement or limit |
|---|---|
| Confidence 0.25-0.45 (`--draft-confidence`) or product-of-probabilities gate vs 0.35 | Within 2% |
| 6 drafts per step vs 4 | No gain |
| 24k draft vocabulary vs 16k | Same acceptance, larger head |
| Attention window of 256 positions | -4 points acceptance |
| Rank-384 vocabulary head | -1%, within noise on held-out pages |
| 16 decode threads vs 12 | Slower with and without speculation; +3% with it |
| Next-token-prediction training only | 23% acceptance; overconfident on later chain steps |

A verification row costs about 1.1 ms, 13% of an 8.8 ms target step, mostly
attention over image positions. A target step streams about 400 MB and
evicts the drafter. Cold draft steps cost 0.67 ms with FP32 drafter KV,
0.51 ms with INT8, and 0.29 ms with INT8 and a rank-256 head; their cost
follows bytes rather than FLOPs.

The GPU cost model predicts 1.59× decode, while held-out CPU decode reaches
1.47× because of failed first drafts and cold caches. Page totals improve
less: prefill takes about 2.8 s per page, roughly a third of the total.

## Rejected or unadopted

| Idea | Measurement or limit |
|---|---|
| 32 decode threads | +50% time per token; SMT siblings contend |
| `gemm` crate's AVX-512 kernels | Under 2% gain on Zen 4 |
| 4-bit screened head | Falls back to the full head on every step |
| Per-channel Q8 key scales | 1.2% gain, flip-neutral |
| 4-way chunked FP32 cache scan | Not adopted |
| Key-transposed decode attention | Decode is already at the bandwidth floor |
| Head-major K/V copies; interleaved FP32 softmax | Within noise; FP32 prefill attention is near its ceiling |
| IEEE FP16 KV for near-exact | Flips 1 → 4 |
| BF16 prefill projections (`panel-bf16`; removed 2026-11-03) | Transformer prefill 4,048 ms against the INT16 projections' 4,066 and the FP32 panels' 5,247 ms (7700X, journal page, 300 tokens, `--tune prefill-profile=1`, median of 5 interleaved rounds); 26.5-28 MACs per cycle per core against INT16's 24.5-27.4; but +19% KL against FP32 where INT16 adds 1.2% ([MODES.md](MODES.md#teacher-forced-agreement)), and AVX512-BF16 only |
| BF16 prefill attention for near-exact's 16-bit body (`--tune prefill-bf16-w16=attention`; removed 2026-11-03) | Transformer prefill 4,864 ms against the FP32 tiles' 7,157 with the static softmax (-32%) and 7,909 with the online one (-38.5%) (7700X, journal page, [below](#near-exact-prefill-attention)); but 13 flips and KL 9.07e-6 against FP32 where the online FP32 tiles have 1 flip and 7.15e-8, and it failed near-exact's held-out gate: micro CER +0.503 pt overall and +5.230 pt on slides, and two repetition stops on pages the reference ends at EOS ([MODES.md](MODES.md#bf16-attention-in-near-exact-mode-rejected)) |
| `exp_fast` in the FP32 static 8-lane tiles (`Simd::exp_fast` with its range checks, in place of the unchecked `exp8` and the masked tiles' compare and AND) | The same bits (every oracle held) but about 5% slower: `static_softmax_timing_probe` (one layer at the journal page's shape, 16 threads, median of 8 calls per process, 3 rounds of fresh processes alternating order; 7700X) took 283.8, 294.4 and 285.4 ms against `exp8`'s 277.8, 285.2 and 286.6 ms, and static over online per process was 0.941, 0.930 and 0.945 against 0.903, 0.916 and 0.868; the range checks cost three compares and blends per vector on every tile |
| Block-skipping decode attention (skip 32-position image blocks whose softmax mass is negligible) | Not built: decode attention over the image prefix is diffuse. On 13 pages and 6,138 FP32 decode steps on the GPU ([research/decode-attention-sparsity/](../research/decode-attention-sparsity/README.md)), an oracle that knows each block's mass skips 1.4% of the image blocks at mass 1e-4 (2 flips and mean KL 1.79e-5 against the Q8 cache's 5 and 1.96e-5) and 8.4% at 3e-4 at four times Q8's output error; every cheap bound (centroid, box, 4- or 2-bit key sketch) costs 3.7-31.8% of the KV bytes for its summaries and skips at most 2.1% at mass 1e-3 |
| INT8 QK for BF16 prefill attention (`vpdpbusd`, per-(token, head) absmax Q and K) | Logit error RMS 3.8× BF16's and attention output error 2.5×, to save at most about 0.5 s per page (7700X, estimated: by its share of thread cycles QK takes about 1.0 s of the 2.15 s BF16 attention, and `vpdpbusd` does twice `vdpbf16ps`'s multiply-adds); not built |
| INT8 activations in the 8-bit prefill projections (u8 × s8 `vpdpbusd`) | Simulated exactly: flips 5 → 40 and KL 20× on two anchor pages (7700X); not built (below) |
| Speculation depths 5-7 (`--speculate 5` to `7`, its maximum) against the default 4 | Journal page with the draft head, `--decode-threads 8`, decode medians of 3 interleaved rounds (7700X): fast 6.39 s at depth 4 and 6.29-6.37 s at 5-7 (9.12 s without speculation), near-exact 9.71 and 9.62-9.69 s (16.16 s); at most 1.6% less decode, with no trend across depths and about the spread between one arm's rounds (0.8-1.4%, 5% in one arm); page totals within 0.7% of depth 4; the same 1,140 tokens in every run |
| Huge pages on the weights, `MADV_HUGEPAGE` on the file mapping | Decode 2.68% faster in fast mode and 1.67% in near-exact, below the 3% rule; worst page total 1.015, over its 1.01 limit; journal RSS 1.010 and 1.008 (7700X; rule below) |
| Huge pages on the weights, in an anonymous copy | Decode 2.74% and 1.41% faster, below the 3% rule; worst page total 1.001; journal RSS 1.240 and 1.218, over its 1.05 limit (7700X; rule below) |
| Heap allocator aligning blocks of 2 MiB and up, with `MADV_HUGEPAGE` | Decode 0.86% and 0.26% faster, below the 3% rule; worst page total 0.980; journal RSS 1.000 and 1.006 (7700X; rule below). Page totals improved 2-5% in all 8 configurations before the per-layer seal, mostly in prefill. Re-screened on the seal with `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (no code; 77% huge where the allocator reached 95% in a scratch test): minor faults 284k → 44k, but total ratios near-exact 1.017 (journal) and 0.978 (sample page), fast 0.989 and 0.988, and near-exact's sample-page peak RSS +11%. The overlap with sealing took most of the gain |
| Decode team pinned to one logical CPU per core | Decode 0.27% faster in fast mode and 0.04% slower in near-exact; worst page total 1.007; journal RSS 1.001 and 1.000 (7700X; rule below) |
| A KV store mixing Q8 keys and Q16 values within a layer (per-layer, per-kind masks; tag `checkpoint/kv-mixed-store-2026-11-06`) | 20.28% of KV bytes, mean KL 9.89e-7, 1 flip, decode -9.7% without speculation and -6.1% with the draft head, against [whole layers](#kv-caches-and-exception-columns)' 19.83% and -9.2% and -5.4% in the same sweep (7700X, before the per-layer seal, `--tune decode-exp=exact`): about 175 more non-test lines for 0.45 points of bytes and 0.5 points of decode ([the study](../research/kv-layer-precision/README.md)) |
| FFN channel pruning (near-exact, checkpoint route; below 0.1 of each layer's median W2 column L2 norm, weakest channels in multiples of 8: 6,536 channels; tag `checkpoint/ffn-prune-2026-11-06`) | Body 347.9 → 316.9 MB; journal page, `--speculate 0`, medians of 3 (7700X): prefill 7,007 → 6,805 ms, decode 13.48 → 12.99 ms/token; projected 1,140-token cost 22,370 → 21,619 ms (-3.36%; a formula, not a measured page), estimated 2.7-3.3% with the draft head, which was not run on a pruned body; mean KL 1.03 times base, 0 flips, 10 of 10 pages token-identical. It needs a new packed format ([NEXT-STEPS.md](NEXT-STEPS.md#ffn-channel-pruning)) |
| GPTQ overlay with 16 W2 exception columns and damping 0.03 | Mean KL 0.567 times the published overlay's, but flips 3 against 1 and 5 of 10 pages identical to exact mode against 7 ([MODES.md](MODES.md#exception-counts-and-damping)). Journal page, checkpoint route, `--decode-threads 8 --speculate 0`, medians of 3, 1,140 tokens (7700X): transformer prefill 3,651 → 3,670 ms (+0.5%), decode 7.86 → 7.78 ms/token (-0.9%), peak RSS unchanged |
| QKV exception columns (4 per QKV matrix beside 8 per W2) | Mean KL 5.25e-5 against 5.28e-5 and flips 2 against 1 for 721,248 more bytes per token, on 10 pages (7700X, [MODES.md](MODES.md#exception-counts-and-damping)) |

The INT8-activation simulation quantized each 64-input activation group to
8-bit levels (absmax / 127, the same rounding) and ran it through the INT16
kernels, whose integer sums equal a `vpdpbusd` kernel's. Teacher-forced on
anchor pages `4a756837631f4f69` and `705b5079f112131e` (1,024 steps against
the 7700X FP32 reference), the current defaults give 5 flips and KL
8.18e-4; 8-bit activations give 40 flips and KL 1.66e-2 in all four
projections, 32 and 1.24e-2 in all but W2, and 7 and 1.10e-3 (QKV), 7 and
1.61e-3 (WO), 21 and 5.66e-3 (W13) or 11 and 1.96e-3 (W2) in one. QKV
alone, on one page, adds 1.33e-5 to the 52-page anchor's mean KL, most of
a 5% allowance on the default's 2.99e-4, so no full run was made. The
cause is resolution, not outliers: in the GPTQ calibration Grams
(`artifacts/w8/gram/`, from their diagonals) the strongest channel's RMS in
a 64-input group is at most 6 (WO), 10 (QKV) and 13 (W13) times the group's
median, yet those fail, W13 worst; only W2 has outlier groups (30% of its
792 above 32×, up to 7,942× in layer 2), so neither a per-group outlier
fallback nor a Hadamard rotation would rescue INT8.

The huge-page and pinning rows come from one screen on the 7700X. Each arm
ran against its own baseline in one sweep: three interleaved rounds of 8
configurations (fast and near-exact, the journal page and the 1418 × 1224
sample page, with and without speculation), with private packed files
evicted and warmed per process. An arm had to make decode with `--speculate
0` at least 3% faster on the journal page in both modes, keep page totals
at most 1.01 times the baseline, load at most 500 ms longer and RSS at most
1.05 times the baseline, and show that its huge pages or pins took effect.
The rows give decode gains and RSS as fast mode, then near-exact.
`MADV_HUGEPAGE` on a file mapping takes effect only when the pages enter
the page cache under the advised mapping (`CONFIG_READ_ONLY_THP_FOR_FS`
off, ext4). The knobs and code are at tag
`checkpoint/hugepages-2026-11-06`.

Multi-page batching (`--batch-size`) is available: on a scanned book it gives
2-8% more pages per hour where the page pipeline gives 12-20%
([below](#book-runs)). Fixed cohorts gave +14% on short pages in fast mode
without the draft head.

On 64 calibration pages in near-exact mode, `--max-dimension 1280` is 20%
faster than 1536, but its quality depends on which pages complete:

| Pages compared | CER at 1536 | CER at 1280 |
|---|---:|---:|
| 48 pages that end at EOS at 1536, 1280 and 1024 | 16.42% | 15.69% |
| 55 pages that end at EOS at 1536 | 19.10% | 20.67% |

There are 9 repetition stops at 1280 against 7 at 1536. It has no held-out
validation, so the default remains 1536. Source:
`research/phase4-hillclimb/attempt3/HILLCLIMB.md`, T14.

Resolution-router timing and quality results are in
[the router study](../research/resolution-router/README.md).

Bitwise-identical decode-attention alternatives:

| Idea | Measurement or limit |
|---|---|
| One scale load per 32 cached values and both heads' PV per value load in verification | -1% of 3-5-row attention, no end-to-end gain |
| Same PV with register accumulators in single-row decode | 21% faster on cache-resident data, **25% slower** in real decode streaming from memory |
| Eight keys' horizontal sums at once | Register spills on AVX2's 16 registers |
| Runtime choice between two instantiations inside one `#[target_feature]` function | Exact path 53% slower; separate entry functions avoid this |

Source: [research/draft-head](../research/draft-head/README.md) and its
kernel probes.

## On a Ryzen 7 7700X

Host: Ryzen 7 7700X (8 cores, 16 threads, DDR5, AVX-512 with BF16), Linux.
The published packed files with the draft head beside them and the default
configuration; fresh processes one after another on a quiet host except
the tuner-limit and reserve comparisons identified below. Raw scripts and
records live under `artifacts/` on that host only:

- `verify/pr/`, whose `timing/` holds the book runs and the KV-cache and
  exception-column timings;
- `gate-2026-11-02/` for fast mode's prefill kernels: the five agreement
  arms (`arms/`), the four-arm prefill A/B (`ab-prefill/`) and the
  whole-page A/B with the verification rows (`ab-combo/`);
- `prefill-int16-2026-11-02/` for the INT16 and BF16 projections: their A/B
  (`ab-locked-1/`) and the projection-GEMM throughput probes;
- `prefill-softmax-2026-11-02/` for fast mode's static softmax: the
  score-bound, polynomial and INT8-QK analysis (`analysis-results.txt`) and
  the stage-cycle A/Bs of the first and the shipped kernel (`ab1/`,
  `ab2/`);
- `act8-2026-11-03/` for the INT8-activation simulation;
- `verify-attn-2026-11-02/` for the verification rows: the page runs
  (`runs/`) and the kernel probes (`timing.log`);
- `fp32-static-2026-11-03/` and `queue-2026-11-03/timing/` for the FP32
  tiles' static softmax, and `queue-2026-11-03/spec/` for the speculation
  depths;
- `w16-bf16-attn-2026-11-03/` for near-exact's removed BF16 attention, and
  `profile-2026-11-03/` for the profiled whole pages;
- `integration-2026-11-06/` for peak memory (`sweep2/`, the busier `sweep/`,
  `tiebreak/` with its rule and `trim/`), the whole-layer Q8 cache (`kv/`,
  `kv-whole/`) and the huge-page re-screen (`thp/`), and
  `peak-memory-2026-11-05/` for the per-change ledger;
- `hugepages-pinning-2026-11-05/`, `ffn-prune-2026-11-05/` and
  `gptq-exc-2026-11-05/` for those rejected rows.

### Book runs

24 pages of a scanned book ([MODES.md](MODES.md#margin-cropping); 956 × 1536,
5,776 input tokens and on average 423 output tokens each). Pages per hour,
one figure per round; every arm without `--crop-margins` gave every page the
sequential run's tokens:

| Arm | Fast | Near-exact |
|---|---:|---:|
| Sequential | 489, 506, 506, 511 | 342, 342 |
| `--pipeline` | 566, 566 | 409, 409 |
| `--pipeline --decode-threads 8` | 585, 586 | - |
| `--batch-size 4` | 528, 529 | 350, 350 |
| `--batch-size 4 --decode-threads 8` | 532, 532 | - |
| `--batch-size 4 --pipeline` | 533, 380 | 380, 378 |
| `--batch-size 4 --pipeline --decode-threads 8` | 548, 549 | - |
| `--crop-margins` | 602, 605 | 416, 415 |
| `--crop-margins --pipeline --decode-threads 8` | 701, 709 | - |

The cropped arms give the model fewer input tokens (138,624 → 115,991 over
the 24 pages) and so change some pages' tokens
([MODES.md](MODES.md#margin-cropping)); the pipelined cropped arm matched the
cropped sequential run.

These pages are prefill-heavy. With fast-mode drafts, three sequential
rounds spend 4.6 s per page in prefill and 2.4-2.5 s in decode
(5.7-5.9 ms/token). A fourth round, with 6 decode threads, spends 2.7 s in
decode (6.5 ms/token).

| Stage | Fast | Near-exact |
|---|---:|---:|
| Pipelined prefill | 6.0-6.2 s on 8 threads | 8.4 s on 10 threads |
| Sequential decode | 5.7-5.9 ms/token | 9.1 ms/token |
| Decode beside prefill | 8.8-9.2 ms/token | 14.3 ms/token |

The pipeline goes at the pace of its prefills: with the decode team pinned
to 8, 24 × 6.05 s = 145 s against 148 s measured. Prefill and decode share
memory bandwidth and the decode team's SMT siblings. Continuous batching
gains less: its rows decode without drafts, giving up the draft head's
speedup, and without the pipeline a joining page's prefill pauses the other
rows.

#### Automatic decode team

The automatic team is limited to half the runner's threads beside a
pipeline on three or more threads. A choice above that becomes the largest
candidate within the limit; smaller choices stand. The limit preserves
prefill capacity when the tuner's similar timings would otherwise leave too
few threads. [ARCHITECTURE.md](ARCHITECTURE.md#execution-order) describes
the pool and reserve behavior.

The tuner measures before prefill overlaps decode. Across the book runs,
fast mode's 8- and 12-thread candidates time within 4.3% on single steps
and 7.4% on 4-row batch steps. In the second `--batch-size 4 --pipeline`
round it chose 12, leaving 4 prefill threads and giving 380 pages/hour;
pinning decode to 8 gives 548 and 549 pages/hour.

The limit's two-round A/B ran on a busier host; tokens were identical:

| Configuration | Decode threads per round | Pages/hour per round |
|---|---|---|
| Without limit | 8, 12 | 566, 372 |
| With limit | 8, 8 (otherwise 12) | 530, 515 |

The three 8-thread rounds use one configuration: their 515-566 spread is
host noise. The A/B isolates the limit's effect through the 12-thread
round. Batched and long-output runs chose the same teams in both binaries.
Removing larger candidates before tuning was rejected: omitting the
fastest candidate lets the 2% tolerance accept 6 threads, whose verification
steps are slower. Pinning `--decode-threads 8` gives the pipeline about 3%
more throughput because overlap starts without waiting for tuning.

With speculation, an 8-thread run on 8 cores has 6- and 8-thread candidates;
neither fits the half-thread limit. A 4-thread reserve supplies the missing
choice. On the book's first four pages (379-425 output tokens each), with
`--mode fast --threads 8 --pipeline`:

| Decode team / prefill pool | Time, round 1 | Time, round 2 |
|---|---:|---:|
| 6 / 2 threads, without reserve | 59.3 s | 48.7 s |
| 4 / 4 threads, with reserve | 35.5 s | 35.2 s |
| Sequential, 8 threads | 32.9 s | - |

These two interleaved rounds ran at a load average of about 5 and gave
identical tokens. Even with the reserve, the pipeline does not pay at 8
threads on pages this short.

#### Memory

Peak resident memory in fast mode; pipelined and batched arms pin decode
to 8 threads:

| Arm | Peak resident memory |
|---|---:|
| Sequential | 2,243-2,300 MiB |
| `--pipeline` | 3,404-3,413 MiB |
| `--pipeline --crop-margins` | 3,822-3,831 MiB |
| `--batch-size 4` | 3,899-3,900 MiB |
| `--batch-size 4 --pipeline` | 3,348-4,275 MiB |

These figures predate sealing per layer; [Peak memory](#peak-memory) has
the current behaviour.

### Peak memory

A page's memory used to peak at the end of its last prefill layer, with
every layer's FP32 compact KV beside the forward workspace. Each layer now
[seals right after its attention](ARCHITECTURE.md#kv-cache-formats), so one
layer's FP32 buffers exist at a time and the sealed records grow in their
place. Fresh workspace buffers and the split tail also come from `vec!`,
whose untouched pages stay out of RSS, the patches go after the embedding,
and fixed glibc thresholds return large frees to the OS.

The build before these changes against the build with them: three
interleaved rounds of fresh processes, `--model artifacts/packed
--decode-threads 8` and default speculation, with the arms' order swapped
in the second round, on a quiet host (largest unrelated process 2.8% CPU).
The journal page is capped at 300 tokens (`--max-new-tokens 300`) and the
1418 × 1224 sample page (`falcon-ocr-comparison.png`) runs at the default
budget. Peak RSS comes from `/usr/bin/time -v`, and MB is 10^6 bytes.
Medians, old → new; tokens were identical in every pair. `prefill_ms` is
the like-for-like prefill column, since `transformer_prefill_ms` now
includes the seal:

| Mode | Page | Peak RSS, MB | Ratio | `prefill_ms` | Ratio | Decode, ms/token | `total_ms` | Ratio | Minor faults |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| near-exact | journal | 1,864 → 1,095 | 0.587 | 7,319 → 7,240 | 0.989 | 7.77 → 7.87 | 9,697 → 9,712 | 1.002 | 479k → 283k |
| near-exact | sample | 1,933 → 1,117 | 0.578 | 7,991 → 7,707 | 0.964 | 8.06 → 8.37 | 18,498 → 18,788 | 1.016 | 635k → 292k |
| fast | journal | 1,699 → 754 | 0.444 | 3,816 → 3,634 | 0.952 | 5.08 → 5.03 | 5,398 → 5,242 | 0.971 | 443k → 233k |
| fast | sample | 1,760 → 769 | 0.437 | 4,099 → 3,834 | 0.935 | 5.30 → 5.20 | 11,026 → 10,638 | 0.965 | 544k → 240k |
| exact | journal | 2,216 → 1,918 | 0.866 | 8,268 → 8,219 | 0.994 | 13.73 → 13.88 | 12,500 → 12,526 | 1.002 | 584k → 544k |
| exact | sample | 2,842 → 1,958 | 0.689 | 9,387 → 8,787 | 0.936 | 14.53 → 14.70 | 28,567 → 27,941 | 0.978 | 815k → 577k |

The timing columns carry scatter. An earlier sweep of the same binaries on
a busier host gave total ratios of 0.990 and 0.978 in near-exact (journal,
sample), 0.968 and 0.970 in fast mode and 1.003 and 0.986 in exact, with
decode ratios of 0.98 to 1.004. Fast mode gains about 3%, and exact stays
within about 2% either way.

Near-exact's sample page disagreed most between the sweeps, with decode
ratios of 1.039 and 0.993, so six more rounds settled near-exact: the same
binaries, flags and pages, the arms' order swapped each round, and a rule
fixed before the run. A difference counts as real if the median ratio passes
1% and at least five of the six rounds fall on that side. New/old, median
and range:

| Page | Decode, ms/token | `prefill_ms` | `total_ms` |
|---|---:|---:|---:|
| journal | 0.992 (0.967-1.016) | 0.991 (0.982-1.134) | 0.995 (0.986-1.105) |
| sample | 1.000 (0.979-1.012) | 0.954 (0.941-0.998) | 0.984 (0.974-0.993) |

Near-exact's decode did not change. The sample page's prefill and total are
faster in all six rounds, by 4.6% and 1.6%, and pass the rule; nothing on the
journal page does, and its 1.134 and 1.105 come from one run. Tokens matched
in every pair. Absolute times ran up to 19% slower in the first three rounds,
with no cause in `ps`; the ratios compare runs within a round.

With `--pipeline`, near-exact, on the first three pages of the [10-page
set](MODES.md#the-metric), one run each: peak RSS 3,627 → 1,335 MB, wall time
32.13 → 31.86 s, identical tokens.

Each change's share, cumulative from top to bottom, one run each: the last
VmHWM that `--tune phases=1` sampled, in MB. The split tail matters at the
sample page's default budget; the journal page's 300 tokens barely use it.

| Change | Near-exact, journal | Fast, journal | Near-exact, sample | Fast, sample |
|---|---:|---:|---:|---:|
| Before | 1,862 | 1,699 | 1,933 | 1,759 |
| Seal per layer | 1,370 | 1,020 | 1,738 | 1,222 |
| Free the patches after the embedding | 1,368 | 1,020 | 1,716 | 1,202 |
| Lazily zeroed workspace | 1,156 | 806 | 1,494 | 980 |
| Lazily zeroed split tail | 1,155 | 806 | 1,286 | 811 |

glibc raises its mmap threshold after each large free, so a later page's
prefill buffers came from a heap that kept the freed pages. Both binaries
now call `cpu::return_large_frees_to_os` first in `main`, which sets
`M_MMAP_THRESHOLD` and `M_TRIM_THRESHOLD` to 4 MiB
([GNU/Linux only](PORTABILITY.md#allocator-and-memory-samples)). Against
the last row above, a sequential run of the same three pages goes from
2,170 to 1,211 MB in near-exact and from 1,260 to 845 MB in fast mode (last
sampled VmHWM).

The trim threshold is set too because setting only `M_MMAP_THRESHOLD`
freezes it at 128 KiB, which returned per-thread pack buffers on every free.
Medians of 3 interleaved rounds, the mmap threshold alone against both:

| Mode | Pages | Minor faults | System time |
|---|---|---:|---:|
| near-exact | journal | 699k → 284k | 1.81 → 1.11 s |
| near-exact | three pages | 1,922k → 643k | 5.21 → 2.50 s |
| fast | journal | 340k → 237k | - |
| fast | three pages | 847k → 510k | - |

Peak RSS moves under 1%, and `total_ms` improves 0.2 to 1.0%.

### KV caches and exception columns

The journal page's first 300 tokens (`--max-new-tokens 300`): decode
milliseconds per token, median of three rounds. `--model-file` picks the
weights and the hidden `--kv-cache` the cache; every arm gave the same 300
tokens, with and without speculation:

| Weights | Cache | `--speculate 0` | Draft head |
|---|---|---:|---:|
| fast | `q8` | 8.28 | 6.35 |
| fast | `q8r` | 8.37 | 6.39 |
| fast | `q4r` | 9.54 | 7.18 |
| fast with 4 W2 exception columns | `q8` | 8.24 | - |
| near-exact | `q16` | 13.84 | 8.62 |
| near-exact | `q8r` | 11.30 | 7.51 |
| near-exact | `q8r`, `--tune decode-exp=exact` | 10.98 | 8.01 |

Rounds differ by up to 5%, so the rotation's 32-point transforms and the
exception columns' fused multiply-adds cost nothing measurable. `q4r`
streams fewer bytes than `q8` but decodes its codes through the 8-bit loads,
and is 15% slower. On near-exact's weights `q8r` streams 23% fewer bytes per
step than `q16`; with the platform exp that its agreement run used, it
decodes 21% faster without speculation and 7% faster with the draft head.
The exception-column file had no draft head beside it, so its speculative
runs used n-gram drafts and are left out. Their fidelity is in
[MODES.md](MODES.md#rotated-kv-cache-experimental) and
[MODES.md](MODES.md#exception-columns-experimental).

`--tune kv-layers=0,2,3,5,6,7,8,9,10` seals nine whole layers at Q8 in
near-exact's Q16 cache;
[MODES.md](MODES.md#whole-layer-q8-in-the-q16-cache-experimental) has its
fidelity and why it stays opt-in. Journal page, packed near-exact file,
`--max-new-tokens 300 --decode-threads 8`, medians of 3 interleaved rounds,
without the knob → with it:

| Speculation | Decode, ms/token | Change | `total_ms` | Change |
|---|---:|---:|---:|---:|
| `--speculate 0` | 13.87 → 12.76 | -8.0% | 11,492 → 11,141 | -3.1% |
| Draft head | 7.62 → 7.26 | -4.6% | 9,642 → 9,494 | -1.5% |

Peak RSS falls from 1,095 to 1,019 MB (-7%), since sealing lets the smaller
cache reach the peak. On a binary from before the per-layer seal, with
`--tune decode-exp=exact` pinned, the same set gave decode -9.2% and -5.4%
with RSS unchanged; its tokens on the 10 pages are identical to the unpinned
run's on all 10.

### INT16 projections and static softmax

Fast mode's default prefill kernels since 2026-11-03;
[MODES.md](MODES.md#int16-projections-and-static-softmax) has their
fidelity, token checks and the decision. Journal page, `--mode fast
--speculate 0 --decode-threads 8 --max-new-tokens 300 --tune phases=1`,
three interleaved rounds of fresh processes with no other work on the host,
while both kernels were opt-in knobs. On a current build the previous
default is `--tune prefill-projections=fp32 --tune prefill-softmax=online`,
and each single-kernel arm sets one of the two. Transformer prefill is the
median [range] in milliseconds; the attention and projection columns are
its phase split:

| Arm | Transformer prefill | Attention | Projections | Against previous default |
|---|---:|---:|---:|---:|
| Previous default (FP32 panels, online softmax) | 5,284 [5,122-5,312] | 2,618 | 2,411 | |
| INT16 projections | 4,103 [4,097-4,109] | 2,610 | 1,245 | -22% |
| Static softmax | 4,832 [4,812-4,836] | 2,161 | 2,420 | -9% |
| Both (the current default) | 3,652 [3,634-3,654] | 2,151 | 1,249 | -31% |

The kernels act on different stages and add: INT16 activations halve the
projections (2,411 to 1,245 ms) and leave attention alone, and the static
softmax takes 0.46 s off attention and leaves the projections alone.

On the whole journal page with the draft head (`--mode fast
--decode-threads 8`, three interleaved rounds of fresh processes), a build
with both kernels and the [verification-row change](#verification-rows)
took 10.36 s against 12.64 s for the build before both changes (medians):
prefill 5.61 → 3.94 s, decode 6.98 → 6.37 s, the same 1,140 tokens.
Near-exact, which neither prefill kernel touches, went from 18.77 to
18.10 s in the same rounds.

Projection GEMM at m = 6,544 on 16 threads, in multiply-accumulates (MACs)
per cycle per core:

| Kernel | MACs per cycle per core |
|---|---:|
| FP32 panels | about 14.3 |
| INT16 panels, AVX512-VNNI | 24.5-27.4 |
| INT16 panels, AVX2 | 19-20 |

Measured per-core instruction throughput on this Zen 4: a zmm FMA issues
once per cycle and a ymm FMA twice, and zmm `vpdpwssd`, `vpdpbusd` and
`vdpbf16ps` once each. The FP32 panels' ceiling is therefore 16 MACs per
cycle (two ymm FMAs of 8 lanes) and the INT16-VNNI kernel's 32 (one
instruction of 16 lanes × 2 products): the FP32 panels reach 89% of theirs
and the INT16-VNNI panels 77-86%. The removed BF16 panels ran 26.5-28
([Rejected or unadopted](#rejected-or-unadopted)).

The static softmax removes most of the softmax stage. In the shipped
kernel's A/B (`--tune prefill-profile=1`, journal page, 4 interleaved
rounds; `prefill-softmax-2026-11-02/ab2/`) the online kernel's stage took
44.8e9 thread cycles, 25% of attention; fused into the QK epilogue, the exp
adds 12.8e9 (QK 68.1e9 → 80.9e9), and the profile reports the softmax
inside the qk share and 0% for the softmax itself.

### Near-exact prefill attention

The FP32 tiles' static softmax has been near-exact's default since
2026-11-03. Fast mode's BF16 attention for the 16-bit body, measured beside
it, failed the same held-out gate and was removed.
[MODES.md](MODES.md#static-softmax-in-the-fp32-tiles) has their fidelity,
token checks and [held-out gate](MODES.md#near-exacts-held-out-gate).
Journal page, `--mode near-exact --speculate 0 --decode-threads 8
--max-new-tokens 8 --tune phases=1`, three interleaved rounds of fresh
processes with no other work on the host, while both were opt-in knobs; on
a current build the previous default is `--tune
prefill-softmax-fp32=online`. Transformer prefill is the median [range] in
milliseconds; attention is its phase split:

| Prefill attention | Plan | Transformer prefill | Attention |
|---|---|---:|---:|
| FP32 tiles, online softmax (the previous default) | `avx512-wide` | 7,909 [7,887-7,934] | 5,234 |
| FP32 tiles, static softmax (the default) | `avx512-wide-static` | 7,157 [7,154-7,173] | 4,478 |
| BF16, static softmax (removed) | `bf16-static` | 4,864 [4,840-4,869] | 2,174 |

The other phase lines sum to 2.54-2.55 s in every arm (transformer prefill
less attention is 2.68-2.69 s). The static softmax takes 0.76 s (14%) off
attention, 9.5% of transformer prefill; an earlier run of the two FP32 arms
gave 7,903 and 7,116 ms, within 0.6% of these. BF16 attention takes 3.06 s
(58%) off attention and 38.5% off transformer prefill, 2.29 s more than the
static softmax; its attention phase is about fast mode's 2,151 ms with the
same kernel ([above](#int16-projections-and-static-softmax)).
Teacher-forced against FP32 the static softmax has the online softmax's 1
flip and 1.3% more KL, and BF16 attention 13 flips and 127 times the KL;
on the held-out gate's 207 pages the static softmax changed no token and
BF16 attention raised the gated micro CER by 0.503 pt. Under `--tune
prefill-profile=1`, on a busy host and so an indication only, the softmax
stage (in the static tiles: masking, exp and denominators) takes 24.9% and
24.5% of FP32 attention's thread cycles in two online runs and 7.3% with
the static softmax.

For scale, one profiled run each of the whole journal page with the draft
head (`--decode-threads 8 --tune phases=1 --tune prefill-profile=1`, which
add synchronization) took 17.1 s in near-exact with the static softmax
(prefill 7.4 s, decode 9.6 s) and 10.3 s in fast mode (3.9 and 6.3 s).
BF16 attention's 2.3 s against the static softmax was about 13% of that
near-exact page.

### Verification rows

Two bitwise changes make drafted rows cheaper: the AVX-512 row kernels of
split-cache decode attention
([ARCHITECTURE.md](ARCHITECTURE.md#decode-machinery)), and quantizing each
appended position's KV codes without a libm `roundeven` call per element
(the same codes, ties included). The journal page with the draft head and
`--decode-threads 8 --tune phases=1`, before and after, interleaved fresh
processes (fast mode 4 rounds, near-exact 3); every run gave the same 1,140
tokens. Decode attention per step by rows in the step, medians in ms:

| Mode | 1 | 2 | 3 | 4 | 5 | Per extra row | Decode, page |
|---|---:|---:|---:|---:|---:|---:|---:|
| fast, before | 4.14 | 6.30 | 7.99 | 9.68 | 11.54 | 1.85 | 7,260 |
| fast, after | 4.08 | 5.14 | 6.15 | 7.31 | 8.72 | 1.16 | 6,419 |
| near-exact, before | 7.53 | 8.88 | 10.54 | 12.34 | 14.55 | 1.76 | 10,601 |
| near-exact, after | 7.51 | 8.27 | 9.10 | 10.20 | 11.86 | 1.09 | 9,922 |

Decode time falls 11.6% in fast mode and 6.4% in near-exact. Appending one
position per row (`cache_append`) went from 0.18 to 0.06 ms per row in
single-row steps and from 0.17-0.18 to 0.04-0.05 ms per row in steps of 2
to 5 rows. `verify_rows_probe` (22 layers, 8 threads) puts each extra row
at 1.76 → 1.07 ms for `q8`, 1.71 → 1.04 for `q16` and 0.78 → 0.47 for
`f32-split`; single-row decode attention
(`decode_attention_bandwidth_probe`) does not slow down. On one thread, in
an 8-row step, an extra `q8` row still costs about 27 cycles per group and
key (6.2 ms over 22 layers, 8 KV groups and 6,552 keys, at about 5 GHz),
against the 16 cycles that its 256 multiply-adds take at 16 FMA lanes per
cycle. Not adopted: a batched horizontal sum of eight keys on 8-lane
vectors (about 3% at 5-8 rows), two rows per PV pass sharing value loads
(no gain; the kernels are FMA-bound) and 16-key sub-tiles (mixed).

## Measuring

See [measurement commands and protocol](DEVELOPMENT.md#measuring-performance-and-input-changes).
