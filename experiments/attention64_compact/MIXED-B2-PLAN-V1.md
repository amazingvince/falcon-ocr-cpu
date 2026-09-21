# Mixed-B2 comparison protocol

The next native Windows comparison uses the frozen `mixed-sparse-first` pair:
`sparse-room`, then the natural table page `2c243b31d36eb729`. It measures the
combined fixed-width attention candidate against the original implementation,
with compact caches in both. The original interrupted six-mode benchmark is
preserved and is not resumed.

Three fresh processes run control, candidate, control. Each uses two unmeasured
warmups and three measured recognitions, FP32 AVX2, 16 threads, joint batch two,
unpacked weights, dimensions 64–1536 and a 4,096-token output cap. The limit is
1,800 seconds per process. Exact GPU output signatures are frozen before launch:
6 and 2,280 IDs, both EOS, plus literal text and prefix counts. Prepared dimensions
come from the frozen workload and must match every CPU output; the historical
GPU records do not contain observed prepared dimensions. Missing historical GPU
startup fields remain missing.

The comparison is accepted only with exact outputs in every measured request
and no more than 5% absolute drift between control medians. It separately reports
whether candidate latency falls by at least 5% and whether it regresses by more
than 5% against either control. No default promotion follows from this one pair.
The sparse page exits after six tokens, leaving only about five joint decode
forwards before the long single-request tail. Sustained B2, B4 and B8 throughput
remain separate work. Per-request stage intervals overlap in joint decoding;
their medians must not be added to obtain batch wall time.

All builds, tests and other project inference must finish before the timed
window. Normal desktop services may remain and are recorded in the launch
attestation. Input/model/source identity checks run outside that window. A
failed or partial bracket is preserved; rerunning requires a fresh plan.

- [Frozen plan](../../artifacts/benchmarks/attention64-compact-mixed-b2-window-v1/plan.json)
- [Execution helper](../profiling/benchmark_mixed_compact_candidate.py)
- [Protocol regression checks](../profiling/test_benchmark_mixed_compact_candidate.py)
- [Independent source review](../../reference/benchmarks/attention64-compact-mixed-b2-source-review-v1.json)
- [Independent frozen-plan review](../../reference/benchmarks/attention64-compact-mixed-b2-plan-review-v1.json)

Plan SHA256: `904841b075f91c3f2109b00db07f4a697fadbb35bfef0a5345f8956e081407cc`.
This document describes the protocol; completed timing evidence is recorded separately.
