# Staged-probability attention: native Windows result

The experiment preserves the checked CPU results and produces the intended
register-resident PV loop, but makes this full page **2.01–2.04% slower**.
Retain the existing combined fixed64 compact runner. The candidate misses the
prospective 5% improvement target and is not promoted.

## Measured result

Ryzen 7950X, native Windows, FP32/AVX2, 16 threads, joint B1 (the singleton
recognition path), compact cache and unpacked weights. The frozen journal page
has 6,544 prefix tokens, prepared dimensions 1088×1536, and 1,140 generated
tokens ending in EOS. Image-size limit 1536; output limit 4096.

Each fresh process ran two unmeasured warmups and three measured recognitions.
No other project build, model, profiler or bulk analysis ran during the bracket.
Ordinary OS/user background activity was not excluded. All nine measured
outputs matched the frozen token vector, literal text, stop, counts and dimensions.

| Process | Page median | Decode median | Prefill median |
|---|---:|---:|---:|
| Compact control before | 62.6959 s | 52.5083 s | 10.0854 s |
| Staged candidate | 63.9725 s | 53.9128 s | 9.9668 s |
| Compact control after | 62.7090 s | 52.5287 s | 10.0750 s |

Control drift was **0.0208%**. Candidate page latency increased by 2.0361%
against the first control and 2.0149% against the second. Decode intervals
increased by 2.63–2.67%. Stage intervals are descriptive, not isolated kernel
measurements; their medians need not sum to the page median. This is one page
and one controlled bracket, not a general performance claim for other CPUs.

The [comparison](../../artifacts/benchmarks/attention64-staged-fullpage-window-v1/comparison.json)
records the individual samples, exact-output checks and target decision.
The [prospective plan](../../artifacts/benchmarks/attention64-staged-fullpage-window-v1/plan.json)
fixes the workload, binaries, 5% target and control-drift limit before execution.
The [independent saved-result review](../../reference/benchmarks/windows-attention64-staged-fullpage-review-v1.json)
passed 606 checks across 27 unchanged files, reproducing the timing arithmetic,
process ordering and all nine measured output checks without rerunning inference.

## What was implemented and verified

Only copied `src/kernels.rs` differs from the frozen combined fixed64 compact
source. The compact AVX2 single-query path first calculates scalar probabilities
in the original order, overwrites the existing tile-logit array, and then applies
the values with eight YMM output accumulators. The separately rounded rescale
multiply, ascending-key FMA history, 128-key tile boundaries, denominator,
mask, sink and final normalization are preserved. Other dispatch paths retain
their original implementation.

- Six offline source guards passed.
- All 17 selected Rust operator tests executed and passed, including the 11
  retained tests and six new rescale, cancellation, signed-zero, unaligned,
  prefix/tile/tail, full-context and dispatch tests.
- The full 1,904-tensor saved native CPU trace matched byte for byte, together
  with all 17 teacher-forced IDs and the Length finish reason.
- The unchanged allocation harness passed all eight warmed decode intervals
  with zero allocations.
- The [independent assembly review](RESULTS-ASSEMBLY-REVIEW-V1.md) confirmed
  eight output accumulators, separate rescale multiplies, key-ordered FMAs and
  tile-end stores. The decoded PV loop contains no calls or output
  loads/stores/spills. This proves the compiler transformation, not its speed.
- All nine measured full-page outputs were exact.

These are same-CPU optimization checks. They do not resolve the ten existing
CPU/GPU intermediate numerical mismatches or transfer the baseline's 200-page
GPU agreement to this experimental binary. No new GPU, mixed-batch or Linux
qualification was run for this slower candidate.

## Interpretation and next measurement

Removing repeated output traffic from this loop was a valid optimization
hypothesis, but the complete calculation also needs probability staging,
strided value access, address generation and scheduling around scalar exp.
The output traffic removed was local accumulator traffic; it was not evidence
of reduced DRAM traffic. Assembly counts cannot establish which cost limits
the complete decoder. No specific bottleneck is proven by this regression.

The next useful measurement is the prepared profile of the existing compact
control. Its new capture protocol binds the correct binary/PDB, limits the
recording, and preserves exact-output, PID, lifetime, loss and stack checks.
The launcher was checked with fixed exit-0 and exit-7 children and correctly
reported both outcomes.

Windows reported that the UAC launch was canceled before the launcher started.
No recorder or profiled model run began, and no profile result is claimed.
The [canceled-launch receipt](../../artifacts/diagnostics/fullpage-compact-profile-uac-attempt-v1.json)
is preserved. The prepared plan is
`D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/plan.json`.
A future launch requires Windows elevation. The separately prepared XML
integrity helper has not been exercised on a new capture.

## Reproducibility identities

Sources and launch helpers are in this directory. The capture driver uses a
fresh isolated `D:/falcon-ocr-rust-builds/attention64-staged-v1` build target,
Rust 1.92.0 / LLVM 21.1.3, matching control build flags, and preserved emitted
binaries. The live runner's defaults are not replaced by this experiment.

| Artifact | SHA-256 |
|---|---|
| Preparation | `f848a193ca147b2a1cbe476ab59c6c4b5fce03e078074a9c95043c546e512348` |
| Build | `b76ea8940ac3e03b355be61667f2cc4344dc9e579c5a52bfbd2b81084a072cbe` |
| Operators | `6e54297b181175aad218e882f6649d9fbd1a63fe3b623562837f4c6379f70528` |
| CPU smoke | `128a854c1d26816e9940cd7c76846c7631130fee54635244e8ac27694b2cf061` |
| Allocations | `68382379dd9ba84df13c9c329c7c17597dddc6258aa617731c38b546040afe46` |
| Benchmark build | `9d45b164cc8c24301e95b820f2e9c6edaabd939481a4a0cf1628e09679b3a6b9` |
| Candidate executable | `9414183c6a7989c1899c042edacd018e52ca990f30e7099238833b5f1873e5d9` |
| Benchmark source archive | `75e457330ef8d7ad74950012e6ab14f82b9cd10a929c32979bca22aa8022694d` |
| Assembly receipt | `8894a92ad955c982d6ed49e8f6bdf7862886f301bcb4b9cd9f7d876b0610e530` |
| B1 plan | `45730ca2ed26fe63facd1dafb004701711ace5cc50e8a8b0c4d92fb37ab0d6b0` |
| B1 comparison | `7ab775eb2dd34b78c760d22a9f67796da5ccfd6ec6e45c450e86a13c6c839ab0` |
| Independent B1 review | `61b932790c2a219c306c2b44de1a6b510039d9ed9b5f9b6548e607724cf99d43` |
| Prepared compact profile plan | `530cfc0c5808cc525722379c0e0e0a2518d0f3d7067e62e9819030407080332b` |

The candidate receipts live under `artifacts/diagnostics/attention64-staged-v1`;
the timing bracket lives under `artifacts/benchmarks/attention64-staged-fullpage-window-v1`.
Frozen build inputs and old experiments remain preserved. New result/review
files are post-execution evidence, separate from the original source inventory.
