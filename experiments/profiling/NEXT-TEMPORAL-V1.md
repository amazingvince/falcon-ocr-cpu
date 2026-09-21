# Next bounded candidate: split-prefix storage with direct fixed64 attention

Recommend one copied-project candidate combining the already validated
prefix-temporal storage representation with a direct two-pointer AVX2 dot64.
Start from the frozen **combined fixed64 attention** archive, without the
paired-GEMV candidate: the latter's 1.16–1.29% B1 improvement did not meet its
frozen 5% target. This assessment describes the final explicit-layout route;
the source-only implementation is in `experiments/attention64_temporal`.
This note is not execution or performance evidence.

## Exact arithmetic and integration

For prefix key `t` and query head `h`, use temporal pointer
`temporal[(t*8+h/2)*32]` and spatial pointer `spatial[(t*16+h)*32]`.
Keep four eight-lane accumulators initially zero. Update each with Q[0:32]
and its temporal eight-lane slice, then update the **same accumulators** with
Q[32:64] and its spatial slice. Preserve input/weight FMA operands and order,
`(acc0+acc1)+(acc2+acc3)`, low/high-half addition and the two horizontal adds.
Do not calculate two dot32 reductions and add their scalars.

The existing `experiments/attention64/candidate.rs::dot64` already expresses
that topology. A new inline split helper changes only the key load addresses.
For generated keys, retain its existing contiguous `dot64`; keep `axpy64`
unchanged for all V. Put both inside a feature-qualified temporal head with
dispatch outside the key loop. Preserve 128-key tiles, the prefix/generated
branch, every key's traversal order, maxima, scaling, exponentials, denominator
and PV order, and learned sink. The prefix boundary is not a new softmax tile.
Each distinct Q head still computes all 64 products.

Reuse `experiments/cache_layout/temporal_candidate.rs` storage and append
validation: prefix K is 8×32 temporal plus 16×32 spatial, generated K and all
V are 8×64. All duplicated bits are checked before mutation, including signed
zeros; spatial heads remain distinct. Retain the same fixed-dimension and
single-token-continuation rejection paths. Initial prefill borrows expanded
`work.k` and calls the baseline compact attention, preserving its GEMM operands,
strides and schedule. Scalar and explicit AVX-512 can retain the validated
stack-reconstruction fallback; only resolved AVX2 takes the direct split path.

The copied-project integration points already exist in
`experiments/cache_layout/capture_temporal_candidate.py::patch_copy`:

- `src/config.rs`: add explicit `CacheLayout::TemporalCandidate`. Keep Compact
  as an independent control path and Expanded as the unchanged default.
- `src/lib.rs` and a new storage module: register the copied cache.
- `src/model.rs`: `LayerCache::Temporal`, session allocation, fallible append,
  and pass `current_expanded_k` through both single/batch attention call sites.
- `src/kernels.rs`: validated temporal entry point plus a new fixed64 temporal
  head/helper; leave expanded and compact paths intact.

**Do not apply the old `attention_adapter` mechanically to the new baseline.**
The combined compact function now contains a fixed64 dispatch referring to
`prefix_k`; the old adapter's substitutions do not handle that branch. Reuse
the previously validated generic temporal adapter as the fallback, and derive
the new AVX2 head from the frozen combined `compact_head` with a checked,
limited key-loading substitution. This avoids reintroducing per-key indirect
dot/AXPY calls or silently losing the prior optimization.

The exact source delta against the combined-attention archive is four changed
files (`src/config.rs`, `src/kernels.rs`, `src/model.rs`, `src/lib.rs`) and two
added files (`src/temporal_candidate.rs`, `src/temporal_model_tests.rs`). The
runner and benchmark harness remain identical. `Session::new` selects the new
private Temporal variant only for the explicit TemporalCandidate layout.
Compact and Expanded retain their original construction and attention paths.
The two added files reuse the validated historical storage/model-test source
bytes; the new fixed64 tests are inlined into kernels.rs under `cfg(test)`.

The old storage type, append/attention methods and adapter are already ordinary
runtime code in the diagnostic copy; only its reconstruction helpers and test
modules use `cfg(test)`. Its prior integration tests linked that copied library.
The new module/dispatch must remain compiled in the release benchmark, with no
test-only override controlling activation. The focused Session test instantiates
Compact and TemporalCandidate separately and compares their outputs. This is safe to
adapt prospectively; prior output equality does not itself qualify the new
release binary. Unsupported dimensions/multiquery continuation still fail.

## What may improve, and what remains uncertain

At the selected full-page prefix of 6,544 tokens, persistent payload falls by
`6544*256*22*4 = 147,423,232 bytes = 140.59375 MiB`. That is 25% of prefix K
alone and 16⅔% of prefix K+V, not 16⅔% of the full reserved cache. With 4,096
reserved generated rows, analytic total KV falls from 1,253,638,144 to
1,106,214,912 bytes. Vector metadata, allocator overhead, one-layer expanded
workspace and actual process residency are separate.

Compared with the combined-attention baseline, there was no reconstruction
copy to eliminate: the proposed direct helper prevents the new storage layout
from introducing one. The prospective benefit is the smaller unique key
working set and changed access locality. Each query head still issues eight
key-vector loads; two heads may reuse their temporal data through the cache,
but the instruction count or physical-memory traffic does not automatically
fall by the storage percentage. Separate temporal/spatial streams add address
work and may hurt prefetch/TLB behavior. Runtime duplicate checks and packing
add prefill cost; retain them for this first honest end-to-end measurement.
No counter or post-optimization profile establishes a bandwidth bottleneck.

An alternative with a potentially larger data-reuse surface is processing
each pair of Q heads together to share V loads (and generated/temporal K).
It also reduces 16 independently scheduled head tasks to eight, needs two
online-softmax states, and increases register pressure. The old baseline's
AXPY share motivates investigating it later, but it is a larger orchestration
change with no measured benefit. The validated temporal layout is the more
bounded next experiment; avoid combining the two hypotheses.

## Reuse tests, then one fair B1 bracket

Reuse the eight storage/attention/model-dispatch tests, including duplicate-bit
mutation rejection, supported backend fallbacks, prefill and continuation
validation, mask/tile boundaries and owned Vec-capacity accounting. Add direct
split-dot equality against the original full dot64 on finite cancellation and
signed-zero cases, deliberately unaligned halves, and both heads of each GQA
pair. Extend decode cases to prefix 6,544 and total context 16,384 as already
covered by the combined-attention tests. Compare against both reconstructed
generic temporal attention and unchanged compact/expanded oracles.

Reuse the prior temporal full-model qualification harness for complete
1,904-tensor/17-decision smoke, the 2,144-tensor mixed trace with shrinking
active rows, independent free outputs and warmed allocation intervals. Prior
Windows/Linux results validate the storage design, not the new direct helper;
repeat the bounded checks for the new build. The ordinary unchanged allocation
test selects only Expanded/Compact, so it does not cover TemporalCandidate.
Reuse the unchanged dedicated `temporal_model_qualification.rs` harness, which
already accepts the explicit layout and measures two warmed allocation intervals
per invocation: single-page and mixed-batch decode. Require zero allocation
calls and requested bytes in both intervals for this actual candidate. Keep
the independent Compact control and the teacher trace's
17-token length stop and free-run EOS contracts separate.
Inspect one compiled temporal head for direct split loads, the original FMA
tree, unchanged AXPY and no reconstruction/indirect inner-loop calls.

Fastest fair timing route: one fresh copied candidate build/unique D: target,
with the frozen combined-attention binary as both fresh controls. Keep the
existing `ocr_bench.rs` unchanged; it already accepts the library CacheLayout
enum. Use a **new narrowly adapted** B1 wrapper, preserving the current wrapper.
The control takes `--cache-layout compact` and reports `compact`; the candidate
takes `--cache-layout temporal-candidate` and reports `temporal_candidate`.
The existing wrapper
cannot run unchanged: it permits only a kernels.rs source difference and
identical source inventories. The new plan must allow exactly the four changed
files and two added files above, bind the complete source archive, and keep all
other sources/flags fixed. `benchmark_temporal_candidate.py` implements this
explicit CLI/report mapping and exact source allowlist as a new protocol.

Use the existing frozen prose input, FP32 AVX2, 16 threads, dimensions 64–1536,
cap 4,096, unpacked weights, B1, two warmups and three measured repetitions in
control/candidate/control order. Exact all-nine 1,140-ID/text/EOS/dimension
outputs, <=5% absolute control drift and >=5% improvement against both controls
remain unchanged. Report prefill/decode separately and Vec payload versus
process-memory counters distinctly. Do not time the stack fallback as an
extra variant or carry GEMV pairing into this candidate. Preserve a negative
result; no automatic variant sweep, additive historical gain or default
promotion follows. The ten GPU numerical gates remain separate and open.
