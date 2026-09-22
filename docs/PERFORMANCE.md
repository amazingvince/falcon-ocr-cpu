# Performance evidence

The current measurements establish improvements on a small fixture. They do not
establish representative full-page throughput or bare-metal Linux performance.
The user deferred bare-metal Linux measurements; WSL is used for functionality.

## Same-binary Windows mode comparison

Native Windows, Ryzen 9 7950X (16 cores/32 logical processors), Rust 1.92.0 MSVC,
16 Rayon threads, AVX2, FP32. The input is the canonical 256x128 three-line image:
144 prompt tokens and 17 emitted tokens. Groups repeat that same image. Each case
has two warmups and seven measured samples, excluding file decode and model load.
The mode order is sequential, joint/expanded, joint/compact, sequential again.

All generated token IDs match. Sequential median drift between controls is
at most 1.28%. Project CPU/GPU jobs were paused; normal interactive applications
remained open, so these are local development measurements, not a dedicated host.

| Pages per group | Sequential control medians | Joint / expanded median | Joint / compact median | Expanded throughput gain |
|---:|---:|---:|---:|---:|
| 1 | 486–492 ms | 498 ms | 473 ms | 1.2–2.5% slower |
| 2 | 983–988 ms | 813 ms | 786 ms | 1.21–1.22x |
| 4 | 1932–1934 ms | 1264 ms | 1249 ms | 1.53x |
| 8 | 3860–3865 ms | 1995 ms | 1980 ms | 1.93–1.94x |

Joint decoding reaches 4.01 pages/second for the repeated eight-page group versus
about 2.07 sequentially on this fixture. Prefill still runs independently.
The single-page result stays within the planned 5% regression limit here.
Representative full-page and mixed-length workloads remain necessary before
closing the overall performance gate.

Compact caches reduce the joint group median by 3.25%, 1.16% and 0.77% at groups
2/4/8 in this run. Those batch gains are below the 5% promotion target; compact
therefore remains opt-in. Peak process resident memory over each complete mode
run is 1.359 GB for joint/expanded and 1.305 GB for joint/compact. These are
process-lifetime counters including mapped weights, not isolated case peaks.
The separate bit-exact cache test measures the Rust heap saving directly.

The full report and retained samples are in
[`reference/benchmarks/windows-mode-comparison.json`](../reference/benchmarks/windows-mode-comparison.json)
and the four `quiet-modes-*.json` files beside it. The report includes binary,
source, dependency, image and weight hashes, all timings and token IDs. Reproduce
each mode using `examples/ocr_bench.rs`, `--execution sequential|joint`,
`--cache-layout expanded|compact`, `--backend avx2`, `--threads 16`,
`--batches 1,2,4,8`, `--warmup 2` and `--repetitions 7`, then run
`python scripts/compare_benchmark_modes.py` with the recorded filenames.

The earlier 562 ms single-page development baseline had concurrent GPU-corpus CPU
orchestration. It is retained for provenance and is not used as the control here.

## Packed FP32 weights

The opt-in `--weight-layout phase-packed` runner preserves AVX2 reduction order
while sharing weight loads among 2–8 live decode rows. Single-row decode and
prefill keep their existing paths. A Model packs once and shares the packed copy
between Runners. Windows and WSL tests preserve single/mixed/batch2/4/8 traces,
outputs and zero-allocation warm decoding; see `reference/weight-layout-parity.json`.

A later native Windows same-binary run uses the same tiny image, 16 threads and
expanded caches. Two unpacked controls bracket two packed repetitions, each with
two warmups and seven samples per group. Every output token matches.

| Pages per group | Unpacked medians | Packed medians | Latency reduction against both controls |
|---:|---:|---:|---:|
| 1 | 488–508 ms | 488–490 ms | Within measurement variation |
| 2 | 796 ms | 651–655 ms | 17.7–18.1% |
| 4 | 1301–1312 ms | 922–932 ms | 28.4–29.7% |
| 8 | 1966–2009 ms | 1525–1559 ms | 20.7–24.1% |

Control drift is at most 4.09% (single page); batch drift is at most 2.12%.
The packed copy adds 876,085,248 bytes (835.5 MiB), takes 131–135 ms to prepare,
and increases peak resident memory from about 1.36 GB to 2.24 GB across the full
mode runs. It remains opt-in pending representative full-page
and mixed-length evidence. These results measure an additional improvement over
joint decoding; they should not be multiplied by gains from an older binary.

Reproduce with `scripts/benchmark_packed_windows.ps1` and summarize with
`scripts/compare_packed_benchmarks.py`. Retained samples and hashes are in
`reference/benchmarks/quiet-packed-*.json`; the summary is
`reference/benchmarks/windows-packed-comparison.json`.

## Promotion bracket: fixed-width attention and compact default (2026-09-21)

Commit `95daabc0` makes `--cache-layout compact` the FP32 default and routes
single-query, 64-wide-head AVX2 attention through `src/kernels/attention64.rs`
(the kernels measured in isolation under `experiments/attention64` and
`experiments/attention64_compact`). The promotion bracket ran three fresh
`examples/ocr_bench.rs` processes per case in the order control-before,
candidate, control-after on an idle native Windows host (Ryzen 9 7950X, Rust
1.94.0 MSVC, 16 threads, FP32 AVX2, unpacked weights). The control is the
previous default binary (commit `5275f467`, expanded cache); the candidate is the
new default (compact cache, fixed-width attention). Each process performed two
warmups and three measured recognitions (seven for the small fixture). Every
measured token ID, text, stop reason and dimension is identical across the three
processes and, where a pinned GPU record exists, identical to it.

| Case | Control before | Candidate | Control after | Drift | Candidate vs controls |
|---|---:|---:|---:|---:|---:|
| Journal page, batch 1 (6,544-token prefix, 1,140 output tokens, 1536 cap) | 73.403 s | **62.719 s** | 74.167 s | 1.04% | −14.56% / −15.44% |
| Mixed pair, joint batch 2 (sparse room 6 tokens + table page 2,280 tokens) | 159.286 s | **131.764 s** | 157.735 s | 0.98% | −17.28% / −16.46% |
| Small fixture, joint group 1 | 0.493 s | 0.485 s | 0.497 s | 0.82% | −1.58% / −2.38% |
| Small fixture, joint group 2 | 0.800 s | 0.772 s | 0.787 s | 1.58% | −3.43% / −1.90% |
| Small fixture, joint group 4 | 1.298 s | 1.256 s | 1.267 s | 2.47% | −3.25% / −0.86% |
| Small fixture, joint group 8 | 2.047 s | 1.961 s | 2.087 s | 1.94% | −4.20% / −6.02% |

Journal-page stage medians: prefill 12.68 / 12.34 / 12.72 s and decode
60.73 / 51.58 / 61.30 s for control-before / candidate / control-after; the gain
is in decode, as the isolated experiments predicted. The target workloads beat
both controls by more than the 5% promotion threshold, and no secondary workload
regresses (the small-fixture groups move by −0.9% to −6.0%). Raw processes,
commands and the evaluation are in
`artifacts/benchmarks/attention64-promotion-v1/bracket/` (`bracket.json`,
`bracket.md`), produced by `scripts/promotion_bracket.py`; the control and
candidate binaries are preserved under `artifacts/builds/control-5275f467/` and
`artifacts/builds/candidate-95daabc0/`. This bracket qualifies the new default on
native Windows only; bare-metal Linux performance remains deferred.

An external head-to-head against the Rust/GGML implementation
`pszemraj/falcon-ocr.rs` is recorded separately in [`COMPARISON.md`](COMPARISON.md).

## Full-page single-request measurement

The first full-page group completed on native Windows with the Ryzen 9 7950X,
FP32 AVX2 and 16 threads. One journal page has a 6,544-token prefix and generates
1,140 tokens to EOS. Six fresh processes use the frozen order: sequential A,
expanded A, compact, phase-packed, expanded B, sequential B. Each performs two
warmups and three measured recognitions. Other project inference, scoring,
tests, builds and profiling were stopped during the bracket; ordinary desktop
services and lightweight source authoring/review remained.

Every measured token ID, literal text, stop, dimension and count matches across
all modes and repetitions. Sequential control drift is -0.60%; expanded control
drift is +0.76%, both within the frozen 5% limit.

| Mode | Median page latency | Peak resident memory | Peak private commit |
|---|---:|---:|---:|
| Sequential controls | 73.581–74.024 s | 2.973–2.974 GB | 2.449–2.450 GB |
| Expanded-cache controls | 74.310–74.876 s | 2.974 GB | 2.450 GB |
| Compact cache | 67.829 s | 2.576 GB | 1.784 GB |
| Phase-packed weights, expanded cache | 75.549 s | 3.851 GB | 3.328 GB |

Memory uses decimal GB and process-lifetime high-water marks, including loading
and warmup. It is not isolated KV allocation or steady-state resident memory.

Compact caching reduces page latency by **8.7–9.4% against both expanded
controls**, passing the target for this group. Peak resident memory falls by
about 398 MB and peak private commit by about 665 MB. Median decode time falls
from 61.6–62.1 s to 55.4 s; prefill remains about 12.4–12.6 s. Image preprocessing
is about 54–61 ms, with the image projection taking about 7 ms inside prefill.
Stage medians are nested observations and need not sum to the enclosing median.

Packed weights provide no single-page gain here: latency is 0.9–1.7% higher
than the expanded controls. The additional packed payload is 835.5 MiB and takes
130.4 ms to construct, outside the warm samples. With one active row, the runner
uses its original single-row kernels; this group cannot measure packed batch
compute gains. The batch API with one page also uses that single-row path.

This is one selected full-page group, with three measured samples per process,
not corpus throughput or complete batch-matrix coverage. Neither compact nor
packed defaults are promoted. The other primary workloads, outstanding GPU
numerical checks and deferred bare-metal Linux measurements remain separate.
The [sanitized comparison](../reference/benchmarks/windows-fullpages-b1-v1.json)
retains samples, stage measurements, memory counters and source/build identities.
Its SHA256 is `ef25cf55760ec893616f023ab050c60b6a2e996cd700f109d8ceeb10b7803e0e`.
The complete raw bracket stays in `artifacts/benchmarks/fullpages-b1-window-v1`.
An [independent saved-artifact review](../reference/benchmarks/windows-fullpages-b1-independent-review-v1.json)
passed 838 checks across 69 files, including all measured outputs, control drift,
medians and the preserved source archives. It repeated no benchmark work.

## Pending experiments

The [frozen realistic protocol](REALISTIC_BENCHMARKS.md) now has a fresh
source-bound Windows build and a validated 72-process plan, with the prescribed
minimum of three measurements after two warmups. The full 72-process matrix has
not run. See `reference/benchmarks/realistic-fp32-v2-planning-validation.json`.
The separate six-process batch-one group above is complete, using
`artifacts/benchmarks/fullpages-b1-window-v1/plan.json`, the same binary and
sampling policy.
The next selected group, `mixed-sparse-first` batch two, began on 2026-09-20
after its [independent planning review](../reference/benchmarks/mixed-sparse-first-b2-planning-review-v1.json).
Its new plan is `artifacts/benchmarks/mixed-sparse-first-b2-window-v1/plan.json`,
SHA256 `18a4b1a2ffa7b9d58ee7003ec3ec0ee9e65e8abd21e6dadb3a13f97b31a84c4f`.
It pairs the sparse-room image and full numerical table in that order. The
attempt was deliberately stopped during its first control when the user moved
the main effort to single-page profiling and optimization. No timing result is
accepted. The [interruption receipt](../reference/benchmarks/mixed-sparse-first-b2-interrupted-v1.json)
preserves the partial attempt; any later run needs a fresh complete bracket.
Later groups will use separate complete control/candidate brackets; the current
runner cannot resume an interrupted bracket. These selected groups do not imply
complete matrix coverage. See the
[scheduling review](../reference/realistic-benchmark-scheduling-review-v1.md).
New FP32 results separately measure the image projector and transformer prefill,
both as subintervals of the existing prefill time. The timing-only instrumentation
preserves the previous 1,904-tensor smoke trace byte for byte; single/mixed output
and timing checks pass on Windows and WSL. Historical reports are unchanged.

The first native full-page CPU trace is captured and analyzed. All three
recognitions match the saved baseline's 1,140 IDs, literal text and EOS, and the
owned WPR session stopped cleanly. Of 3,118,232 in-bounds samples, 99.9469% have
attached stacks; reported ETW event loss is zero. A separate buffer-loss counter
is unavailable. One additional kernel-only sample falls 189.9 microseconds after
process stop: the strict audit remains failed, while its negligible contribution
allows explicitly qualified exploratory hotspot ranking.

Resolved call paths assign 51.86% of whole-process CPU samples to non-GEMM
attention, 26.85% to linear projections and 11.68% to GEMM attention. Shared
`dot_avx2` and `axpy_avx2` leaves account for 46.30% and 25.66%, respectively;
these overlapping views must not be added. The first isolated experiment targets
fixed-size single-query attention operations while preserving arithmetic order.
Its completed native Windows bracket measured **68.444 s per page**, compared
with unchanged controls at **75.570 and 72.931 s**: a **6.15–9.43% latency
reduction**. All nine measured outputs match exactly, and six operator checks
plus the complete 1,904-tensor smoke trace remain bit-identical. Whole-page
control drift is 3.49%, within the preset 5% limit. Stage timings varied more:
control prefill changed by +17.45% and decode by −6.92%, limiting precise
phase-level claims. See the [candidate result](../experiments/attention64/RESULTS-V1.md)
and [bound receipt](../reference/benchmarks/windows-attention64-fullpage-v1.json).
This is one page and one bracket, with no default promotion. The separate
compact-cache gain cannot be added without a combined experiment.

That combined experiment is now complete: **63.102 s per page** with compact
caches and fixed-width attention, versus fresh compact controls at **67.414 and
67.239 s**. This is **6.15–6.40% additional latency reduction over compact caches
alone**, with 0.259% control drift and all nine measured outputs exact. Decode
latency fell by 8.10–8.24%; prefill differed by −0.13% to +2.21% versus controls.
Eleven operator tests, the byte-identical 1,904-tensor compact smoke and eight
zero-allocation warm decode intervals passed before timing. See the
[combined result](../experiments/attention64_compact/RESULTS-V1.md). There is no
fresh expanded control in that bracket; the separate historical gains cannot
be added. The same candidate now also passes Linux-under-WSL operator, complete
same-platform smoke-tensor and allocation checks; see the
[Linux functional receipt](../reference/linux-attention64-compact-functional-v1.json).
The frozen sparse/table mixed-B2 follow-up also passed: **128.912 s per pair**
versus **142.475/141.837 s** compact controls, a **9.11–9.52% latency reduction**
with **0.448% control drift**. All 18 measured request outputs match the pinned
GPU IDs, literal text and EOS stops (6 and 2,280 output tokens per pair); the
[independent review](../reference/benchmarks/windows-attention64-compact-mixed-b2-review-v1.json)
checked the saved results and executable/source bindings. The sparse request
finishes early, so the long singleton tail dominates this pair; this is not a
sustained B2/B4/B8 throughput result. Broader workload, full-corpus and bare-metal
Linux performance qualification remain pending, and production defaults are
unchanged.

An isolated paired-output AVX2 GEMV follow-up preserved all operator/smoke
outputs and zero-allocation checks, but measured **62.041 s** versus
**62.849/62.766 s** combined-attention controls: only **1.16–1.29%** lower
latency, below the unchanged 5% target. All nine full-page outputs remained
exact and control drift was 0.132%. The candidate is unpromoted; see the
[GEMV result](../experiments/gemv_pair/RESULTS-V1.md). This bounded result does
not prove that all GEMV optimization is unhelpful or that bandwidth is the
limiting resource.

A subsequent explicit temporal-prefix cache candidate matched thirteen operator
checks, 4,048 single/mixed tensor digests and two zero-allocation intervals. Its
full-page bracket measured **60.730 s** versus **62.818/62.758 s** controls:
**3.23–3.32%** lower latency, with all nine outputs exact and **0.096%** drift.
This also misses the frozen 5% speed target and remains unpromoted. It saves an
analytic **140.594 MiB** of reserved KV payload for this prefix/capacity; storage
reduction is distinct from the speed target and hardware memory traffic. See
the [temporal-cache result](../experiments/attention64_temporal/RESULTS-V1.md) and
[independent review](../reference/benchmarks/windows-temporal-candidate-fullpage-review-v1.json).
Further variants or platform/batch qualification are not queued for it. A fresh
profile of the combined-attention compact baseline is the next performance
step; the original expanded profile need not reflect the current hotspots.

See the [profile results and limitations](../experiments/profiling/FULLPAGE-RESULTS-V1.md),
[parsed stacks](../reference/benchmarks/windows-fullpage-profile-stacks-v1.json),
and [boundary review](../reference/windows-profile-boundary-review-v1.md).
The matching frozen PDB resolved project code; Windows system frames remain
unresolved. PerfView returned an exception after writing a parseable XML ZIP,
so exporter success is not claimed. The raw capture and export remain preserved
in `D:/falcon-ocr-rust-builds/profiles/fullpage-expanded-v2`.
The earlier canceled elevation attempt is preserved separately; the subsequent
Windows approval succeeded and no further profiling approval is pending.
The [prospective criteria](../experiments/profiling/PROFILE_ACCEPTANCE.md) and
original failed audit remain unchanged. CPU samples are neither exact phase
markers nor hardware bandwidth counters; profiled timings do not replace the
unprofiled controls.

- Full-size documents with natural long outputs, mixed completion lengths and
  stage measurements, including image preparation and peak memory.
- AVX2 versus explicit AVX-512, thread-count sweeps, and full-page packed matrices.
- Qualified BF16 graph and its direct/packed operators; optional AOCL-DLP.
- INT8 and groupwise INT4 with independent calibration and OCR-quality reports.

Kernel-only speedups and theoretical weight/cache byte reductions do not establish
whole-model speedups. Alternatives must preserve correctness and beat matched
controls by at least 5% in their target workload without regressing the other
primary workload by more than 5% before promotion.
