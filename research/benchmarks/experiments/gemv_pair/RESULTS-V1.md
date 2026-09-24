# Paired-output GEMV candidate

Native Windows correctness and full-page output checks passed on 2026-09-20,
but the latency reduction was only **1.16–1.29%**, below the frozen 5% target.
The candidate remains unpromoted. It extends the frozen combined fixed64
attention implementation and is isolated from production source/defaults.

## Correctness and build

The four source guards passed before preparation. The copied runtime sources
change only `src/kernels.rs`, adding the exact five-shape, single-row AVX2/FMA
dispatch and paired-output implementation. Benchmark, CLI, operator and
allocation-test executables were built in a fresh unique
`D:/falcon-ocr-rust-builds/gemv-pair-v1` target with matching control toolchain
and release settings. Emitted Cargo project/source paths, freshness and
preserved executable bytes were checked.

All seven operator tests passed. They include full synthetic outputs of each
shape, unaligned/boundary and cancellation cases, fallback paths and malformed
shapes. The mandatory real-input cases compare all 73,728 output values across
QKV, WO, W13, W2 and the vocabulary projection against the unchanged CPU linear
function; every FP32 bit matches. The first four real inputs are saved prefill
rows evaluated as single-row operators, not a claim about reproducing the
original multirow GEMM's reduction.

The candidate's complete 1,904-tensor teacher-forced smoke file has SHA-256
`e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309`,
identical to the saved native CPU control. All 17 IDs, literal text, dimensions,
counts and length-stop metadata match. The unchanged allocation test passed
with zero heap allocations in each of eight warmed decode intervals across
expanded/compact caches, unpacked/phase-packed weights and single/batch-four
paths. These are bounded functional checks, not performance measurements or
closure of the existing GPU numerical failures.

Evidence is in `artifacts/diagnostics/gemv-pair-v1`:

| Artifact | SHA-256 |
|---|---|
| `preparation.json` | `28c12aaf1f8b3589d8e816d15fc26a08e75daf5dc1c4433ed22ffc509631bf3e` |
| `build.json` | `9cee524cda2db72be503c9be200d69fc8e631d9687331dcc82fa989f9874044a` |
| `operators.json` | `7f1999ac52a0d685042c1e9c102f6a877b6106f15ae4be6e1c7a1a8a9ba029d6` |
| `smoke/report.json` | `7e0436df761d3b91780cc670e3c8955275c1e6d929ebb01317fdcffd74e62963` |
| `allocations/report.json` | `75d7e7bc09f3af355d1624a595d2b456c7dfffe0e6ce4c3627d45cf757b9ebae` |
| `benchmark-build/ocr_bench.exe` | `13e440dfc4c0ce27fc2bc4b2bd9c12ea6446011043670f31bac2379ea512ce25` |

## Compiled code and measurement

The matching PDB resolves `gemv_pair_candidate::channel_block` at
`0x1400b1310`, 762 bytes. Its K loop loads four input vectors once and reuses
them across eight FMAs into two independent four-accumulator groups. Both
original reduction trees are inline. The hot loop has no calls or stack
accesses; Windows nonvolatile XMM register saves/restores occur in the function
prologue/epilogue. The two direct calls are range-check failure paths.
The preserved disassembly and symbol receipt are under
`benchmark-build/compiled-channel-block`. This establishes the intended
compiled structure, not a speedup or memory-bandwidth measurement.
The [manual dataflow review](../../../../reference/gemv-pair-compiled-dataflow-v1.json)
binds the matching PDB, executable and disassembly and distinguishes register
preservation from accumulator spills.

The prospective protocol is [B1-PLAN-V1.md](B1-PLAN-V1.md). Its prepared plan is
`artifacts/benchmarks/gemv-pair-compact-fullpage-window-v1/plan.json`, SHA-256
`c64443002031b9ea36dfcf02ca147598f5e3fd446dc824c4ab42e68cdf5abc36`.
The control is the combined-attention compact-cache build. The completed native
Windows bracket used 16 threads, AVX2 FP32, compact cache, unpacked weights,
two warmups and three measured pages per process:

| Process | Median page time | Measured samples (ms) |
|---|---:|---|
| Control before | 62.849 s | 62849.0994, 63015.8522, 62738.3525 |
| Paired GEMV | 62.041 s | 62040.7347, 61992.8093, 62108.0656 |
| Control after | 62.766 s | 62914.7365, 62747.1278, 62766.1729 |

Control drift was −0.13195%. Whole-page reductions were 1.28620% and 1.15578%
against the respective controls; neither meets 5%. Decode medians were
50.453/49.615/50.345 s and prefill medians 12.308/12.329/12.321 s. Stage timings
are wall intervals and their medians need not add to the whole-page median.

All nine measured outputs exactly match the frozen 1,140 IDs, literal text,
EOS stop, dimensions and counts. Warmup outputs are not separately exported.
The [independent result review](../../../../reference/benchmarks/windows-gemv-pair-compact-fullpage-review-v1.json)
passed, including the below-target decision. The completed comparison is
`artifacts/benchmarks/gemv-pair-compact-fullpage-window-v1/comparison.json`.
This is a small observed gain on one page, not a basis for promotion or an
automatic variant search. No mixed-batch or Linux qualification run is queued
for this candidate; broader evidence remains incomplete. The next experiment
will continue from the stronger combined-attention baseline.
The [plan review](../../../../reference/benchmarks/gemv-pair-compact-fullpage-planning-review-v1.json)
passed before launch. The prelaunch process/memory check found no selected
native or WSL project jobs and 28.187 GiB available physical memory. Build,
test, GPU and analysis work remained stopped until the bracket exited at
18:00:06 UTC.
