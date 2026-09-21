# Fixed-width attention: first full-page result

The isolated AVX2 candidate reduced whole-page latency by **6.15–9.43%** against
unchanged controls in one native Windows bracket. All nine measured recognitions
matched the frozen baseline's 1,140 token IDs, literal text, EOS, dimensions and
counts. This is a successful single-page experiment; production defaults remain
unchanged and broader qualification is pending.

| Process | Whole-page median | Prefill median | Decode median |
|---|---:|---:|---:|
| Unchanged control before | 75.570 s | 10.698 s | 64.754 s |
| Fixed-width attention candidate | **68.444 s** | 12.707 s | **55.617 s** |
| Unchanged control after | 72.931 s | 12.564 s | 60.277 s |

Each process used two unmeasured warmups followed by three measured pages.
Candidate measurements were 68.545, 68.334 and 68.444 seconds. Whole-page control
drift was −3.493%, passing the preset absolute 5% limit. The candidate beat both
controls by the preset 5% target. Stage medians are separate statistics and need
not sum to the median total.

The decode reduction was 7.73–14.11% against the two controls. Stage stability
was weaker than whole-page stability: control prefill changed by +17.45% and
decode by −6.92%. Candidate prefill was 1.14% slower than the final control and
18.79% slower than the first. Therefore the whole-page bracket is the primary
speed result; these stage measurements do not establish a repeatable prefill
regression or a precise decode speedup. They warrant follow-up in subsequent
performance experiments.

## Change and validation

The [CPU profile](../profiling/FULLPAGE-RESULTS-V1.md) put non-GEMM attention at
51.86% of whole-process sampled CPU work. The candidate specializes expanded
cache attention for one query, head width 64 and AVX2/FMA. It moves dispatch
outside the inner key loop and gives dot/AXPY fixed bounds and direct inline
calls. The original per-head attention body is extracted from the frozen source;
only those call targets change. Four dot accumulators, FMA sequence, reduction
tree, 128-key tiles, key order, normalization, exponentials and sinks retain their
original arithmetic.

A bounded [compiled-head inspection](../../artifacts/diagnostics/attention64-v2/benchmark-build/compiled-head/symbol-receipt.json)
confirms eight dot FMAs and the same reduction tree, eight AXPY FMA/store blocks,
and no indirect calls in the specialized head. Scalar exponentials/logarithm
remain direct calls. The preserved candidate PDB matches the executable's
CodeView GUID and age; function attribution uses that PDB, not misleading
nearest-export labels in the disassembler.

The copied project differs from the preserved control source only in
`src/kernels.rs`. Six operator tests passed bit for bit, including tile tails,
causal/image boundaries, 16,384-key context and unaffected dispatch cases.
The candidate's entire 1,904-tensor smoke trace is byte-identical to the saved
CPU control, with all 17 teacher tokens, text and stopping behavior unchanged.
The original ten GPU intermediate numerical mismatches remain open; tolerances
were not changed. This candidate has not rerun the full 200-page GPU corpus.

## Reproduction and scope

- Ryzen 9 7950X, native Windows, FP32 AVX2, 16 threads, sequential batch one,
  expanded cache and unpacked weights. No profiler or competing project jobs
  ran during the benchmark. Normal desktop/system services remained; the WSL
  inspection started its services before the two unmeasured warmups.
- One journal page: original 1653×2339 RGB PNG, prepared 1088×1536, 6,544 prefix
  tokens, 1,140 generated IDs with EOS, output cap 4,096. Input and pinned v1.5
  model hashes are inherited and verified from the frozen workload.
- Model verification/loading and file decoding precede the measured recognitions.
  No added packed-weight copy or cache-layout change is part of this candidate.
- The approximately 9% compact-cache gain was a separate experiment. It cannot
  be added to this result without measuring a combined implementation.
- This is one page and one three-process bracket. It does not establish batch,
  AVX-512, compact-cache or Linux performance, nor default-promotion eligibility.

The executable/source/tool bindings and commands are preserved in:

- `artifacts/diagnostics/attention64-v2/benchmark-build/build.json`
- `artifacts/diagnostics/attention64-v2/operators.json`
- `artifacts/diagnostics/attention64-v2/smoke/report.json`
- `artifacts/benchmarks/attention64-fullpage-window-v1/plan.json`
- `artifacts/benchmarks/attention64-fullpage-window-v1/comparison.json`
- [Portable result receipt](../../reference/benchmarks/windows-attention64-fullpage-v1.json)
- [Independent saved-result review](../../reference/benchmarks/windows-attention64-fullpage-review-v1.json)

Plan SHA256:
`38d1fa812e0cf2d78a05355593f6d236fafdc018d5398754e8c9bb0674e63c28`.
Candidate executable SHA256:
`bcfa0134f33ff8fa75459a16c8f00f62618d6e069ed57e6f533da8cc12d4c1f1`.

The first sandboxed build attempt failed before compilation when creating its
external D: target. That attempt remains in `attention64-v1`; the successful
fresh build and all candidate results are under `attention64-v2`.

The subsequent [compact-cache experiment](../attention64_compact/RESULTS-V1.md)
applied these fixed-width operations to the compact path and measured against
fresh compact controls. Its combined behavior and phase measurements are
reported separately; this original candidate and its evidence remain preserved.
