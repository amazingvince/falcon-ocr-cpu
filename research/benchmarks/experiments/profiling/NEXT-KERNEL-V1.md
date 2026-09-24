# Next single-page experiment: paired AVX2 GEMV

Recommendation: one isolated, single-row GEMV specialization that dispatches
once per output block and computes two independent output channels together.
Start from the frozen combined fixed64-attention candidate; retain its cache
layout and all other arithmetic. This is a source-only proposal, with no new
build, test, model run, profile, timing or tensor analysis.

## Evidence and choice

The original whole-process full-page profile attributes 26.8540% of samples
to linear, 51.8606% to attention, 11.6838% to attention GEMM and 6.1815% to
Rayon without an operator. The 46.2971% `dot_avx2` leaf share spans attention
and projections; it is not a GEMV-only measurement. These are baseline
whole-process shares, not warmed-decode shares or a fresh profile after the
attention optimization. See `reference/benchmarks/windows-fullpage-profile-stacks-v1.json`
and `research/benchmarks/experiments/profiling/decode-source-map.md`.

The fixed64 attention experiments already have separate full-page measurements:
expanded improved 6.15–9.43% against its bracketing controls, compact 6.15–6.40%,
with exact outputs. The remaining GEMV path still chooses a dot function pointer
once per linear call, then invokes it once per output channel. It offers a
bounded follow-up without changing model storage or a reduction tree.

Prefer this experiment before adding prefix-temporal cache splitting to the
performance candidate. Splitting saves exactly 256 FP32 values per prefix token
per layer: 16⅔% of compact prefix payload, not total reserved KV. At prefix
6544 that is 140.59375 MiB; generated-cache payload is unchanged. Its tested
correctness-first implementation reconstructs each prefix key into a stack
array before the existing dot64. Reduced payload may help cache behavior, but
the additional reconstruction may offset it; no speed measurement establishes
either effect. It is a separate layout experiment with broader integration
scope. See `reference/prefix-temporal-dedup-design-v1.md`.

## Exact proposed scope

Only `rows == 1`, resolved AVX2/FMA, and the actual Falcon shapes below enter
the new path. Leave scalar/AVX-512, batches 2–8, prefill/GEMM and packed weights
unchanged. Weights remain contiguous HF `[output,input]` FP32 rows.

| Projection | Input width | Output width | Calls per next-token forward |
|---|---:|---:|---:|
| QKV | 768 | 2048 | 22 |
| WO | 1024 | 768 | 22 |
| Interleaved W13 | 768 | 4608 | 22 |
| W2 | 2304 | 768 | 22 |
| Vocabulary | 768 | 65536 | 1 |

This covers 89 calls and 245,760 output dots per next-token forward. Their
logical weight payload is 835.5 MiB, including 192 MiB of vocabulary weights;
that is not a measurement of DRAM traffic or of vocabulary's sample share.

Move runtime dispatch outside the channel loop into one AVX2/FMA function per
Rayon output block. Retain a minimum work unit of 32 output channels, for
example 16 adjacent pairs; avoid accidentally doubling this by applying the
existing `with_min_len(32)` to pairs. Do not separately tune scheduling in this
experiment. Each pair loads four eight-float input vectors once per 32 input
values and uses them for two independent weight rows.

For **each output**, preserve the existing four zero-initialized YMM
accumulators, assignment of input lanes, chronological FMA sequence, input as
the first FMA multiplicand, and weight as the second. Preserve the merge
`(acc0 + acc1) + (acc2 + acc3)`, low/high-half addition, and two horizontal
adds. Never combine the two outputs' reductions. All guarded widths are
divisible by 32 and all output widths are even; unsupported shapes use the
unchanged path. Preserve W13's channel order and perform no packing or
per-token allocation.

## Expected mechanism, risks and stopping rule

Reusing input loads and amortizing call/dispatch overhead are plausible gains,
not measured ones. The input vector is only 3, 4 or 9 KiB and may already be
resident in L1; weight loads are unchanged. Two outputs require eight
accumulators plus input and weight registers, so register pressure, spills or
instruction scheduling can erase the benefit. No factor-of-two speedup or
bandwidth reduction is implied. Preserve a generic fallback and inspect the
compiled pair function for its actual loads, direct calls and spills.

The first validation should compare every output bit against the unchanged
GEMV for these shapes, using frozen real operands and focused cancellation,
signed-zero and boundary cases. Follow with the existing 1904-tensor,
17-decision smoke comparison, exact free outputs, and warmed zero-allocation
check. Same-platform CPU equality is the acceptance target; the ten open GPU
hidden-tensor gates remain open. Any new bit difference or allocation rejects
this candidate without an arithmetic repair sweep.

Only after functional checks should the parent run one quiet, same-layout B1
baseline/candidate/baseline bracket using the existing protocol. Retain its
5% improvement and control-drift criteria; a smaller or inconclusive result
does not justify promotion or automatic variant searches. No implementation
or measurement has been performed for this proposal.
