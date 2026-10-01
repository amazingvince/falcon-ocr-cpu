# Performance

Status: current as of 2026-10-01. Host, unless a section says otherwise:
Ryzen 9 7950X (16 cores, 32 threads, DDR5, AVX-512 with BF16), Windows 11.
Page: the journal benchmark page, 6,544 input tokens. Earlier measurements on
the tiny 256×128 fixture are archived in
`research/benchmarks/docs/PERFORMANCE.md`.

## The roofline

Decode streams every body weight, the vocabulary head (or its INT8 screen)
and the whole KV cache once per token, so its floor is bytes per token
divided by achievable read bandwidth; `falcon-ocr doctor --probe` measures
that bandwidth (about 50–54 GB/s here) and prints the floor for the plan.
Prefill is compute-bound (about 6.1 TFLOP per full page).

| Mode | Weights + head per token | KV per position | Decode floor at 52 GB/s | Measured, journal page |
|---|---:|---:|---:|---:|
| exact | 876 MB | 113 KB | 31 ms | 29 ms |
| near-exact | 401 MB | 58 KB | 15 ms | 15 ms |
| fast | 233 MB | 30 KB | 8.2 ms | 8.5 ms |

(Floors add the 6,544-position cache: 740, 380 and 196 MB.) Decode runs at
about 90% of the bandwidth floor in every mode; the remaining decode lever is
fewer bytes, and the 4-bit formats cost too much fidelity for this model.

Where a fast-mode page goes (prefill 2.9 s, decode 8.5 ms/token):
prefill attention 1.3 s (BF16 tiles), projections 1.3 s (FP32 panels; 0.7 s
with `--tune prefill-bf16=all`), split/RoPE/append 0.1 s; decode attention
4.2 ms, W13 1.5, head 1.0, W2 0.8, QKV 0.7, WO 0.4 ms. The FP32 prefill
attention tiles run QK at about 75% and PV at about 90% of FP32 peak.

## Accepted changes (journal page, interleaved fresh processes, 3 rounds)

| Change | Effect |
|---|---|
| Split FP32 cache for exact mode (bitwise) | 42.8 → 39.6 s |
| Hybrid threads: prefill on all logical CPUs, decode on physical cores | fast −9%, exact −4% |
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
| Kernel-ready packed files | load 2.2 s → 10 ms, peak memory −1.1 GB |
| 16-lane AVX-512 prefill tiles (bitwise) | attention −9%; under 1% end to end on Zen 4 |
| BF16 scales in the Q8/Q16 caches | decode −2% |
| Multi-row quantized dots and shared record decoding for verify steps | 5-row step 30 → 16 ms |
| Portable polynomial exp in fast mode's decode attention (8-bit caches under `ExpMode::Fast`; `--tune decode-exp`) | verification attention −3/−7/−9/−10% at 2/3/4/5 rows, single rows unchanged; page totals with the draft head −2.1% (calibration) and −0.6% (held-out); flips vs FP32 92 → 94 of 78,282 |
| Trained draft head (`--draft-head`, stage-2 head, INT8 drafter KV; 16 English held-out pages, fast mode) | 184.5 → 139.9 s page total, decode 1.47× (n-gram drafts: 172.7 s, 1.09×); stage-3 head (48k pages), a later run: 181.2 → 135.7 s, decode 1.50×; tokens identical. The published head in the other modes (same pages): near-exact 277.4 → 208.1 s against n-gram drafts (decode 1.87× vs none), exact 531.0 → 392.5 s (1.67×); used automatically when it sits next to the model files |

## Rejected or unadopted

32 threads for decode (+50% per token: SMT siblings contend); the `gemm`
crate's AVX-512 kernels (under 2% on Zen 4); a 4-bit screened head (fell back
to the full head on every step); per-channel Q8 key scales (1.2%, flip
neutral); a 4-way chunked FP32 cache scan; key-transposed decode attention
(decode is already at the bandwidth floor); head-major K/V copies and an
interleaved FP32 softmax (within noise: the FP32 prefill attention is near
its ceiling); IEEE FP16 KV for near-exact (1 → 4 flips); BF16 prefill
projections by default (+17% KL for 0.6 s). Multi-page batching
(`--batch-size`) is available but gains little: on a scanned book it gave
2–8% more pages per hour where the page pipeline gave 12–20%
([below](#book-runs)); fixed cohorts gave +14% on short pages in fast mode,
before the draft head. `--max-dimension 1280` was 20% faster on 64
calibration pages (near-exact); its CER against the ground truth was 16.42 →
15.69% on the 48 pages that end at EOS at 1536, 1280 and 1024, but 19.10 →
20.67% on the 55 that end at EOS at 1536, with 9 repetition stops against 7.
It is not validated on held-out pages; the default stays 1536
(`research/phase4-hillclimb/attempt3/HILLCLIMB.md`, T14). The per-page
resolution router (`--max-dimension auto`,
[MODES.md](MODES.md#resolution-routing)) saves 25% of the CPU time on its
development set (quiet host); it improved CER on its development set but
failed the held-out English gate (+0.29 pt), so it stays opt-in.

Draft head (research/draft-head, 16 calibration pages): other confidence
thresholds and a product-of-probabilities gate (within 2%), 6 drafts per step
(no gain over 4), a 24k draft vocabulary (same acceptance, larger head), an
attention window of 256 positions (−4 points acceptance), a rank-384
vocabulary head (−1%, within noise on held-out pages), 16 decode threads with
speculation (+3%), and a head trained on next-token prediction only (23%
acceptance: overconfident on later chain steps).

Decode attention kernels (bitwise-identical rewrites, research/draft-head
probes): one scale load per 32 cached values and both heads' PV per value
load in the verification kernel (−1% of 3–5-row attention, no end-to-end
gain); the same PV with register accumulators in the single-row kernel
(21% faster on cache-resident data, **25% slower** in real decode, where the
cache streams from memory); eight keys' horizontal sums at once (register
spills on AVX2's 16 registers). Two instantiations of the kernel behind a
runtime choice inside one `#[target_feature]` function made the exact path
53% slower; each instantiation now has its own entry function.

## On a Ryzen 7 7700X

Host: Ryzen 7 7700X (8 cores, 16 threads, DDR5, AVX-512 with BF16), Linux.
The published packed files with the draft head beside them and the default
configuration; fresh processes one after another on a quiet host.

### Book runs

24 pages of a scanned book ([MODES.md](MODES.md#margin-cropping); 956 × 1536,
5,776 input tokens and on average 423 output tokens each). Pages per hour,
one figure per round; every arm without `--crop-margins` gave every page the
sequential run's tokens:

| Arm | Fast | Near-exact |
|---|---:|---:|
| Sequential | 489, 506, 506, 511 | 342, 342 |
| `--pipeline` | 566, 566 | 409, 409 |
| `--pipeline --decode-threads 8` | 585, 586 | – |
| `--batch-size 4` | 528, 529 | 350, 350 |
| `--batch-size 4 --decode-threads 8` | 532, 532 | – |
| `--batch-size 4 --pipeline` | 533, 380 | 380, 378 |
| `--batch-size 4 --pipeline --decode-threads 8` | 548, 549 | – |
| `--crop-margins` | 602, 605 | 416, 415 |
| `--crop-margins --pipeline --decode-threads 8` | 701, 709 | – |

The cropped arms give the model fewer image tokens (138,624 → 115,991 over
the 24 pages) and so change some pages' tokens
([MODES.md](MODES.md#margin-cropping)); the pipelined cropped arm matched the
cropped sequential run.

These pages are prefill-heavy: a sequential fast-mode page spends 4.6 s in
prefill and 2.7 s in decode (6.5 ms per token with drafts). The pipeline
prefills the next page on the threads the decode team leaves (8 in fast
mode, 10 in near-exact), where it takes 6.1 s (near-exact: 8.4 s), and the
run goes at the pace of those prefills: with the team pinned to 8,
24 × 6.05 s = 145 s against 148 s measured. The decode beside them slows
from 6.5 to 8.8 ms per token (near-exact: 9.1 to 14.3 ms), since the two
share memory bandwidth and the decode team's SMT siblings. Continuous batching gains less: its rows decode
without drafts, giving up the draft head's speedup, and without the pipeline
a joining page's prefill pauses the other rows.

The decode tuner chooses the team as if no prefill ran beside it. In fast
mode its 8- and 12-thread candidates time within 4% of each other, on single
steps and on 4-row batch steps, and in the second round of `--batch-size 4
--pipeline` it chose 12, which left the prefill pool 4 threads (380 pages per
hour). Pinning the team to 8 makes that arm repeatable and gives the pipeline
3% more. Peak resident memory in fast mode: 2.2–2.3 GB sequential, 3.4 GB
with `--pipeline` (3.9 GB with `--crop-margins`), 3.9 GB with `--batch-size
4` and 3.3–4.3 GB with both.

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
| fast with 4 W2 exception columns | `q8` | 8.24 | – |
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

## Measuring

- `falcon-ocr doctor --probe --text`: bandwidth and the decode floor.
- `falcon-ocr-eval bench <pages> --profile <profile> --samples 3 --report R`:
  timed recognition with telemetry (`--tune phases=1` prints the phase split
  of decode steps and prefill; `--tune prefill-profile=1` the prefill
  attention stage cycles).
- `python research/phase4-hillclimb/attempt3/ab.py`: interleaved A/B of two binaries
  on the same pages; compare arms within one run only, on a quiet host.
- `python research/draft-head/cpu_ab.py`: interleaved A/B of drafter settings
  with page totals and token identity against no speculation. A running GPU
  job (training, vLLM) slows CPU decode by about 30%; time only on a quiet
  host.
- `cargo test --release --lib panels::panel::tests::prefill_throughput_probe
  -- --ignored --nocapture`: the panel GEMM against the `gemm` crate at
  prefill shapes (1.7–1.8 TFLOP/s here).
- `cargo run --release --example ocr_bench`: warm RGB-buffer recognition
  with hashes of everything involved, for reproducible reports.

Timing A/Bs are unreliable while other jobs run; every number above came
from a quiet host. The 7950X receipts are under `artifacts/phase4/`, the
7700X runs' scripts and records under `artifacts/verify/pr/timing/`.
