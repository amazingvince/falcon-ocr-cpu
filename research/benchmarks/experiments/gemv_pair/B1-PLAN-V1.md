# Paired-output GEMV single-page experiment

Status: prospective protocol. The candidate is source only; no build, operator
result or speedup is established. The attention mixed-B2 timing window must
finish before preparation, builds, tests, tensor inspection or new benchmarks.

The comparison baseline is the combined fixed64 attention implementation with
compact cache, not the original expanded-cache runner. Its frozen build is
`artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json`
(`68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0`).
The accepted historical output and runtime signature is its `candidate.json`
in `artifacts/benchmarks/attention64-compact-fullpage-window-v1`.

## Before measuring

1. Run the four source guards and freeze a copied project from the exact
   combined-attention source archive. Only `src/kernels.rs` may differ from
   that archive's runtime sources. Preserve live source and runtime defaults.
2. Build the benchmark, CLI, operator tests and unchanged allocation test in a
   fresh unique D: target. Require the same Rust/Cargo versions, release flags
   and build environment as the control except for the target directory.
   Check the emitted Cargo artifact's project, source, freshness and preserved
   binary bytes for every executable.
3. Require all seven selected operator tests to pass, including all 73,728
   real-operand outputs and all synthetic vocabulary outputs. Inspect the
   mandatory real-case records; missing inputs or skipped tests cannot pass.
4. Require the candidate's complete 1,904-tensor smoke trace and 17 teacher
   token IDs, literal text and finish reason to equal the saved CPU control.
   Require zero allocations in each of the eight warmed decode intervals.
   This establishes same-platform equivalence on those fixtures, not closure
   of the ten existing GPU intermediate numerical failures.
5. Inspect the matching compiled channel-block function to establish whether
   LLVM retained shared loads and paired independent accumulators, and record
   any spills or calls in its inner loop. Disassembly is diagnostic evidence;
   measured full-page latency decides whether the optimization helps.

## Timed comparison

Reuse the unchanged `research/benchmarks/experiments/profiling/benchmark_compact_candidate.py`
protocol with the combined-attention build as control, its prior candidate
report as the historical baseline, and the newly captured paired-GEMV build
as candidate. Its plan must bind the actual new build, source archive and
binary hashes; this document is not a launch receipt.

Use a fresh output directory and control-before / candidate / control-after
processes. Each process performs two warmups and three measured recognitions
of the frozen prose page, native Windows on the Ryzen 7950X, 16 threads,
explicit AVX2 FP32, compact cache, unpacked weights, joint batch one, maximum
image dimension 1536 and output cap 4096. Keep source/build/model/GPU work
outside the entire bracket. Record a fresh quiet-process and memory check.

All nine measured outputs must have exactly the frozen 1,140 IDs, literal
text, EOS stop, prepared dimensions and prefix count. Warmup outputs are not
exported by this harness and must not be described as individually audited.
Report both control medians, all samples, the candidate median, control drift
and gain against each control. The control drift limit is 5%; the targeted
gain is at least 5% against each control. Preserve failed or interrupted
brackets and never resume their partial timings.

Separate preprocessing, prefill and decode wall intervals support diagnosis;
they do not measure memory bandwidth or hardware counters. Any gain is
incremental over the combined-attention control. Do not add percentages from
historical cache or attention comparisons as though they came from one
controlled experiment.

## After measuring

Independently verify the saved output and source/binary bindings. A useful B1
result proceeds to the frozen mixed-B2 regression workload and Linux
functionality checks. Promotion still requires the applicable regression,
parity and platform evidence; neither this protocol nor a B1 pass promotes
the candidate. Bare-metal Linux performance remains deferred by the user.
