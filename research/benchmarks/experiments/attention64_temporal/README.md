# Split temporal-prefix keys with fixed64 attention

This is one isolated FP32 candidate, starting from the frozen combined
expanded/compact fixed64 attention build. It excludes the paired-GEMV experiment.
Live sources, defaults, historical evidence and benchmark binaries are unchanged.

The copied runner adds **TemporalCandidate** (`--cache-layout temporal-candidate`,
JSON `temporal_candidate`). Existing Compact and Expanded remain independent
paths, and Expanded remains the default. BF16 retains its existing Expanded-only
restriction. This is source preparation, not numerical or performance acceptance.

## Storage and arithmetic

The two historical files `temporal_candidate.rs` and `temporal_model_tests.rs`
are copied byte-for-byte and pinned in `patch.py`. Prefix K stores 8×32 shared
temporal channels and 16×32 distinct spatial channels per token. Generated K
and all V each retain 8×64 channels. Duplicate bits are validated before any
buffer mutation; the cache rejects unsupported dimensions, partial prefill,
multiquery continuation, noncontiguous appends and capacity overflow.

Prefill borrows the original expanded workspace K and calls the unchanged
combined-baseline compact attention. Single-query AVX2/FMA decode loads the
temporal and spatial halves directly. The temporal half updates four eight-lane
accumulators, then the spatial half updates those same accumulators. Operand
order and final reduction tree match frozen dot64; there are no two independent
dot32 reductions. Generated keys use the frozen contiguous dot64 helper. V uses
the frozen axpy64 helper. All key ordering, 128-key tiles, masks, scale, online
max/denominator, exponentials, accumulation and sink operations are copied from
the frozen compact head. Dispatch occurs before the head loops.

Scalar and explicit AVX512 keep the previously validated stack reconstruction
path. Auto follows the existing resolver. The historical generic adapter cannot
be applied mechanically to the combined baseline: its inherited fixed64 compact
dispatch refers to full prefix K. The new patch explicitly removes both that
dispatch and the inapplicable multiquery GEMM branch from the single-query
adapter, then inserts the split-key dispatch after all validation. Source guards
restore each changed load block and compare the arithmetic text exactly.

At prefix 6544, the 22-layer saving is 147,423,232 bytes (140.59375 MiB): 25% of
prefix K, or 16⅔% of prefix K+V payload. This is an owned Vec payload claim, not
allocator overhead, peak RSS or measured memory traffic. The old baseline did
not reconstruct keys; the direct path prevents reconstruction overhead introduced
by the new layout. Each Q head still loads eight key vectors. Performance remains
an unmeasured hypothesis until a separate quiet bracket.

## Capture interface and scope

`make_sources(original: dict[str, bytes]) -> dict[str, bytes]` preserves every
supplied archive member and returns exactly these changes:

- Changed: `src/config.rs`, `src/lib.rs`, `src/model.rs`, `src/kernels.rs`.
- Added: `src/temporal_candidate.rs`, `src/temporal_model_tests.rs`.

`make_patch(original_kernels, candidate_text, tests_text)` returns patched
kernels, the inlined candidate, and the unchanged generic reconstruction oracle.
The direct module is `kernels::temporal64`. Constants expose the baseline build,
archive and affected-source pins, the exact runtime source delta, this experiment's
seven `SOURCE_FILES`, three disjoint `TEST_FILTERS`, and all 13 full `TEST_NAMES`.
No file copying, building or model execution occurs on importing the patch.
The capture owner binds the entire frozen archive/build and any common external
qualification test/fixture sources before invoking this API. The copied model
test requires unchanged `tests/fixtures/model-config.json`.

The runtime storage module is normal release code; only its introspection and
test helpers are cfg(test). Model integration routes the explicit layout through
both single and batch cache append/attention call sites with error propagation.
The old model tests instantiate Compact and TemporalCandidate separately.

## Checks and later qualification

The 13 focused tests comprise five new fixed64 tests and the unchanged eight
storage/model tests. They compare exact bits against the independent existing
Compact/Expanded and generic reconstructed paths; cover unaligned halves,
cancellation/scales/signed zero, paired heads, zero/full prefixes, 127/128/129
crossovers, image masks, extreme sinks, prefix 6544 and context 16384; and reject
bad shapes before dispatch. Historical tests cover scalar/Auto/available AVX2 and
AVX512, storage NaN payloads, before-mutation rejection, borrowed prefill,
continuation, owned capacity stability and explicit/default layout behavior.

Offline source guards may be run with:

```powershell
C:/Users/amazi/mambaforge/python.exe -m unittest discover -s experiments/attention64_temporal -p test_source.py -v
```

Rust builds, operator execution, same-prefix traces, free outputs and allocation
qualification are separate released phases. Reuse the pinned actual-model
qualification harness for 1904 canonical and 2144 mixed tensor hashes, 17 teacher
decisions, independent free outputs and zero allocations in warmed decode. Its
teacher trace correctly stops at length 17; free runs retain EOS requirements.
The ordinary allocation test selects Expanded/Compact only and does not establish
TemporalCandidate allocation behavior.

A later new benchmark wrapper must declare the four changed/two added sources
and compare frozen combined-attention Compact against TemporalCandidate, without
weakening the existing source closure. Same image/options/binary identities,
two warmups and three measured repetitions for control/candidate/control, every
literal output, both-control gains and drift gates remain mandatory. There is no
default promotion, shipping recommendation, GPU parity claim or tolerance change.
