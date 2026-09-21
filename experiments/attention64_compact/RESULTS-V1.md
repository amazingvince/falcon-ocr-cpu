# Compact caches with fixed-width attention

The combined candidate reduced full-page latency by **6.15–6.40% compared with
compact caches alone**, reaching **63.102 seconds per page** in one native
Windows bracket. All nine measured recognitions matched the saved baseline's
1,140 IDs, literal text, EOS, dimensions and counts. The two unchanged compact
controls differed by only **0.259%**.

| Process | Whole-page median | Prefill median | Decode median |
|---|---:|---:|---:|
| Compact control before | 67.414 s | 12.707 s | 54.633 s |
| Compact + fixed-width attention | **63.102 s** | 12.691 s | **50.209 s** |
| Compact control after | 67.239 s | 12.417 s | 54.720 s |

Each fresh process performed two unmeasured warmups followed by three measured
recognitions. Candidate samples were 62.970, 63.233 and 63.102 seconds. The
candidate beat both controls by the preset 5% target, and absolute control drift
passed the preset 5% limit. This is an isolated experimental result; it does not
by itself authorize default promotion.

Decode latency fell by 8.10–8.24%, with only 0.159% drift between control decode
medians. Prefill control drift was −2.282%; the candidate's prefill ranged from
0.126% faster to 2.206% slower than the two controls. Stage medians are separate
statistics and need not sum to the median page time. This bracket shows much
less stage variation than the earlier expanded-cache experiment.

## Implementation and parity

The copied project retains the previously measured expanded-cache specialization
and adds its fixed64 AVX2/FMA operations to compact single-query attention. It
extracts the original compact head body, replacing only the dot and AXPY call
targets. Prefix keys retain all 16 distinct heads; generated keys and values
retain eight. Prefix/generated addressing, grouped head selection, 128-key
tiles, image/causal visibility, FMA/reduction order, exponentials and sink
scaling remain unchanged. Only `src/kernels.rs` differs among the frozen
control's archived benchmark sources.

A [compiled-head inspection](../../artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/symbol-receipt.json)
using the matching executable and PDB confirms inline fixed-width dot/AXPY
operations, no indirect calls, and the preserved prefix/generated-key branch.

Validation completed before timing:

- Eleven bit-exact operator tests: six prior expanded cases and five compact
  cases, including zero/full prefixes, 127/128/129 boundaries, the actual 6,544
  prefix and 16,384 total context, grouped heads and unaffected dispatch paths.
  Compact results match both the original compact implementation and an
  independently expanded cache evaluated by the original expanded path.
- The complete compact smoke trace is byte-identical to the saved CPU control:
  1,904 tensors, SHA256
  `e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309`.
  All 17 teacher IDs, literal text and stopping behavior also match.
- The existing unchanged allocation integration check passed all eight warmed
  decode intervals with zero heap allocations, covering expanded/compact,
  unpacked/packed and single/mixed-batch paths.
- All nine measured full-page outputs agree with the frozen baseline; warmup
  outputs are not exported. The benchmark checks per-repetition token equality
  internally and records each measured text, stop, dimensions and counts.

The ten existing GPU intermediate numerical mismatches remain open. This work
changes no tolerance and has not rerun the combined candidate over the full
200-page corpus.

### Linux under WSL

The unchanged candidate source archive also passed a fresh Linux build with
Rust 1.92.0. All eleven operator tests passed, and the compact smoke matched
all 1,904 tensor payloads against the saved original Linux control, plus all
17 teacher IDs, text and length stop. The unchanged allocation test passed
eight warmed decode intervals with zero allocations. A separate saved-output
check verified every tensor payload and 23 output-file hashes.

See the [Linux functional receipt](../../reference/linux-attention64-compact-functional-v1.json)
and [capture report](../../artifacts/diagnostics/attention64-compact-linux-v1/report.json).
These are same-platform checks under WSL; they establish neither Windows/Linux
tensor equality nor bare-metal Linux performance.

## Mixed-B2 regression result

The frozen sparse-room/table pair completed in **128.912 seconds**, compared
with **142.475 and 141.837 seconds** for the two unchanged compact controls.
That is a **9.519% and 9.112%** latency reduction against the respective
controls. Control drift was **−0.4478%**, within the preset 5% limit; the
candidate met the preset 5% gain target against both controls.

| Process | Median pair time | Three measured pair times |
|---|---:|---|
| Compact control before | 142.475 s | 143.281, 140.637, 142.475 s |
| Compact + fixed-width attention | **128.912 s** | 126.265, 129.142, 128.912 s |
| Compact control after | 141.837 s | 143.949, 141.837, 140.839 s |

Each process used two unmeasured warmups and three measured repetitions,
native Windows FP32 AVX2, 16 threads, joint batch two, compact caches,
unpacked weights, dimensions 64–1536 and output cap 4096. All **18 measured
request outputs** preserved the exact 6/2,280 token vectors, literal texts,
EOS stops, counts and frozen CPU dimensions. Token vectors are stored once
per case; the unchanged compiled harness enforces their equality on every
measured repetition. Warmup outputs are not individually exported.

The sparse page ends after six tokens, so this exercises a short shared B2
decode phase followed by a long single-request tail. It establishes neither
sustained B2 throughput nor B4/B8 performance. Per-request decode and total
intervals can include waiting and overlap; they are not additive service costs
or isolated kernel timings. The GPU references match IDs/text/stops and prefix
counts, but do not record observed prepared dimensions, and the historical
table reference retains its incomplete startup-provenance limitation.

This is the incremental fixed-width-attention result against fresh compact
controls. It must not be added to historical expanded/compact percentages and
does not promote the candidate or close the ten GPU intermediate gates.
See the [portable mixed-B2 receipt](../../reference/benchmarks/windows-attention64-compact-mixed-b2-v1.json)
and [independent review](../../reference/benchmarks/windows-attention64-compact-mixed-b2-review-v1.json),
which passed 789 checks across 28 bound files. The frozen plan is
`904841b075f91c3f2109b00db07f4a697fadbb35bfef0a5345f8956e081407cc`;
the independent-review SHA256 is
`b123307b018f741c458438e6c0513d6da0f7a66f26e18fecc2833489f0a3f324`.

## Reproduction and limits

The workload is the same journal page as the earlier baseline: 1653×2339 RGB,
prepared at 1088×1536, 6,544 prefix tokens and 1,140 emitted tokens ending in EOS.
The run uses the pinned v1.5 weights, FP32 AVX2, 16 threads, unpacked weights,
compact cache and batch one on the native Windows Ryzen 9 7950X. The recorded
`joint` harness option matches the frozen compact control; batch one enters the
single-page runner path. Model verification/loading and file decoding precede
the measured recognitions. All project jobs were idle and no WSL distribution
was running during the bracket; normal desktop/system services remained.

This directly measures the added benefit of fixed-width attention **over
compact caches**. The separate historical compact-cache reduction and this
percentage must not be added. There is no fresh expanded-cache control in this
bracket, nor evidence here for other pages, batch throughput, AVX-512 or Linux.
Production source and defaults remain unchanged pending broader qualification.

Evidence:

- `artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json`
- `artifacts/diagnostics/attention64-compact-v1/operators.json`
- `artifacts/diagnostics/attention64-compact-v1/smoke/report.json`
- `artifacts/diagnostics/attention64-compact-v1/allocations/report.json`
- `artifacts/benchmarks/attention64-compact-fullpage-window-v1/plan.json`
- `artifacts/benchmarks/attention64-compact-fullpage-window-v1/comparison.json`
- [Bound result receipt](../../reference/benchmarks/windows-attention64-compact-fullpage-v1.json)
- [Independent saved-result review](../../reference/benchmarks/windows-attention64-compact-fullpage-review-v1.json)

The independent review passed 606 checks across 27 bound files, including all
nine measured outputs, timing calculations, source/build identities and the
three nonoverlapping successful benchmark processes.

Plan SHA256:
`2647716915e89f9bc4cccb51b4fdee6db3ebbaba3fbf8b0628c68ae02c13396a`.
Candidate executable SHA256:
`8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac`.

The [selected mixed-batch control comparison](MIXED-B2-PLAN-V1.md) is now
completed as reported above; broader workload and sustained-batch qualification
remain separate. The bounded
[layer-7 state crossover](../../reference/fp32-crossover-layer7-decomposition-v1.md)
completed with all ten original controls exact and no tolerance changes. It
does not change the scope of this completed timing result.
