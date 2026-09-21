# Staged-probability compact attention experiment

One isolated FP32 AVX2 experiment, based on the frozen combined expanded/compact
fixed64 source archive. Only copied `src/kernels.rs` changes. Live source,
defaults, prior experiments and frozen benchmark artifacts remain unchanged.

The current compact key loop interleaves scalar exponential/denominator work
with a 64-lane AXPY. The candidate finishes each tile's scalar probabilities
first, reusing its existing 128-F32 logits array. An inline PV loop then keeps
eight independent YMM accumulators across keys and stores them once per tile.
This is an instruction scheduling/storage hypothesis, not a speed result.

## Arithmetic and scope

The patch extracts the accepted compact wrapper, complete head body and dot64
helper from the exact pinned kernel bytes. It changes only the original
rescale/exp/AXPY block and the existing compact AVX2 call target. Dispatch still
requires one query, head width 64, and validated AVX2/FMA. Expanded attention,
prefill GEMM, scalar, AVX-512, other head widths and multi-query paths are intact.

Scalar `exp`, denominator rescaling and denominator addition remain in the
original absolute-key/tile order. Each 128-key tile's maximum, QK dot and scale,
prefix/generated/GQA addresses, learned sink and final scalar division retain
their original source. Eight output vectors receive separate rounded multiply
by rescale followed by the exact previous ascending-key FMA history per lane.
No reduction combines lanes or keys. The output multiplication moves after the
tile's scalar probability calculation, which is data-independent of the output.
The process floating-point exception flags are not an observation contract.

The source expresses eight live PV accumulators but does not prove LLVM keeps
them in registers. Later compiled inspection must resolve
`falcon_ocr::kernels::attention64_staged::compact_head`, check the PV loop for
output spills/reloads or scalar exp calls, and retain any uncertainty from
inlining or partial disassembly. No new vector exponential, normalized
probability rounding, wider SIMD, split-K, cache-layout or thread change is made.

## Patch/capture interface

`patch.py` performs no import-time I/O. `make_sources(original: dict[str,bytes])`
returns the complete source inventory with only `src/kernels.rs` replaced.
`make_patch(original, template, tests)` additionally returns extracted pieces
for source review. The baseline build/archive/kernel SHA-256 pins, exact
changed/added source names, and repository-relative `SOURCE_FILES` are exported.
The capture owner is responsible for freezing the whole baseline archive,
selected source files and any new launch helpers before build/execution.

The baseline is
`artifacts/diagnostics/attention64-compact-v1/benchmark-build/source.zip`, SHA
`f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3`.
It excludes the later paired-GEMV and temporal-storage experiments.

All eleven old attention tests remain verbatim. Six new tests compare with
the unchanged combined compact entry point and/or original AVX2 AXPY:
unaligned PV slices/guards, separately rounded rescaling, cancellation,
increasing tile maxima, negative values and signed zeros, sink extremes,
mask/prefix/tile boundaries, GQA repeats, P6544/full16384/tails and unchanged
dispatch/validation. `TEST_NAMES` contains all **17 full test names**;
`TEST_FILTERS` contains two disjoint module prefixes. `TEST_FILTER` selects their
union. AVX2 is mandatory with no silent skip; AVX-512 unchanged-path cases run
only when supported. Test operands are deterministic synthetic data; there is
no claim that they are real model activations. `INPUT_PINS` is empty for these
operator tests. Model qualification separately binds its real inputs/assets.

Allowed at source preparation: run
`python -m unittest discover -s experiments/attention64_staged -p test_source.py -v`.
Those guards check frozen source/archive identities, exact extraction and
dispatch scope, inventory and the rescale/FMA structure. They do not compile or
execute Rust and cannot establish numerical equality or register allocation.

After review and separate execution release, the capture should build into a
fresh unique target, verify emitted binaries and actually selected test names,
then run the 17 tests. A single compact 17-token same-prefix smoke trace must
match the accepted `e2dad223...85309` tensor baseline and outputs exactly;
the unchanged warmed decode-allocation test must retain zero new allocations.
Benchmarking, if subsequently released, compares against the frozen combined
compact baseline with the existing full-page outputs, matching build flags and
bracketed controls. No result or promotion is implied by source preparation.
