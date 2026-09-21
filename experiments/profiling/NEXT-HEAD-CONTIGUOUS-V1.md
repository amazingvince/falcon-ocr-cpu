# One prefix-only head-contiguous cache experiment

Source-oriented plan,2026-09-20; **not implemented or measured**. Start from the
frozen combined compact/fixed64 build, excluding the rejected GEMV, temporal and
probability-staging candidates. The IP histogram supports concentrating on the
QK/PV regions but does not prove a memory bottleneck or predict a gain.

Baseline: `artifacts/diagnostics/attention64-compact-v1/benchmark-build/source.zip`
SHA`f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3`;
build SHA`68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0`.
The inspected frozen project is the sibling `project/`. Its compact_head is in
`src/kernels.rs:1831`; cache integration is in`src/model.rs:742` onward. A future
patch must verify archive/member pins before using these line/anchor references.

## Storage and scope

Add an explicit experimental `HeadContiguousPrefix` cache-layout option
(`head-contiguous-prefix` CLI, `head_contiguous_prefix` JSON). Preserve Expanded,
Compact and the default unchanged. Initial dimensions are strictly16Q/8KV/64;
accept one full prefix followed by contiguous one-token appends. Reject partial
prefixes, repeated initialization, gaps, capacity overflow and unsupported
dimensions before mutation. Batch decode can still call each session separately
with one query; this is not new multi-query attention.

For prefix lengthP and generated lengthG, store:

| Array | Physical shape | Index for absolute keyk, query headh, KV headj=h/2 |
| --- | --- | --- |
| Prefix K | `[16,P,64]` | `(h*P+k)*64`, whenk<P |
| Prefix V | `[8,P,64]` | `(j*P+k)*64`, whenk<P |
| Generated K | `[G,8,64]` | `((k-P)*8+j)*64` |
| Generated V | `[G,8,64]` | `((k-P)*8+j)*64` |

All16 prefix K heads remain distinct; there is no temporal-half sharing or
whole-head deduplication. Prefix V and generated K/V select the same first head
of each already duplicated GQA pair as Compact. Validate paired bytes in the
focused tests, including signed zeros and NaN payload storage; do not silently
accept an invalid reduction. The initial experiment changes **only prefix
physical order**. Generated K/V keep their present token-major format and append
order, although V moves to a separate generated buffer.

Pack the complete prefix once per layer after the existing RMS/RoPE and before
decode. Reserve each persistent vector at session creation using checked sizes
and fallible allocation. Append generated rows without further reservations.
Head-major prefix segments then advance256 bytes per key, versus4,096 for
current prefix K and2,048 for current V. This is an addressing fact, not a cache
miss or bandwidth measurement.

## Preserve prefill and arithmetic

The current `attention_gemm_compact` reads token-major V with stride512. Passing
head-major V to that function would be wrong. For the first candidate, allocate
one reusable local `P*512` FP32 scratch for the prefill call; fill it with the
unchanged compact-V selection from each layer's expanded `work.v`. Call the
existing compact prefill operator using borrowed expanded `work.k`, empty
generated K, and this token-major compact-V scratch. Drop the scratch when the
prefill call returns. It is reused across22 layers, not allocated per layer or
kept for decode. The retained head-major K/V packing and scratch writes are all
inside the normal page/prefill execution, with no hidden preparation outside
the timing interval. Do not add a transpose/GEMM backend change to this test.

For decode, derive the candidate head body from the frozen compact_head and
replace only K/V slice addressing. Preserve existing AVX2 dot64 and axpy64,
their four-accumulator/FMA/reduction order, scalar exp/log, denominator updates,
rescale rounding, masks, sink application and final division. The generic
scalar/AVX512 path may use the same direct64-element slices, retaining its
current backend arithmetic; it does not need reconstruction or a new SIMD dot.

Critically, keep **one global loop** `start=0,128,256,...`, one running maximum,
denominator and output. AtP6544, the tile beginning6528 contains16 prefix keys
and up to112 generated keys. Select the array separately for each absolute key
inside that same tile. Do not split attention into prefix/generation passes or
restart softmax tiles at the cache boundary. Both QK tile maximum and PV key
order must span the crossing exactly as before.

## Concrete copied-source touchpoints

Expected runtime allowlist: changed`src/config.rs`, `src/lib.rs`, `src/model.rs`,
`src/kernels.rs`; added`src/head_contiguous_prefix.rs` for checked storage and
`src/head_contiguous_prefix_model_tests.rs` for private model integration tests.
No live-source change, Cargo/dependency change or old-receipt rewrite is needed.
The actual patch must declare and check this exact inventory.

- `CacheLayout` in config supplies explicit derived Clap/Serde naming. Existing
  generic CLI/report fields need no relabeling; unsupported BF16 cache selection
  must continue to reject the new layout.
- `LayerCache` adds a variant wrapping the new four-vector storage. Its append
  branch handles complete-prefix packing or one generated row. Existing variants
  and`append_unique_heads` semantics remain unchanged.
- `Session::new` reserves1024P+512P+512Gmax+512Gmax floats per layer. It must not
  keep an extra persistent token-major prefix mirror.
- `Model::forward` at the cache append/attention call adds borrowed current K
  and the candidate-only prefill V scratch as inputs to the dispatch. Traces of
  current Q/K/V stay before packing and retain their existing tensor layout.
- The `decode_batch` cache call supplies its exact single-row slices and no
  prefill scratch. Active-session indices/order and all linear projections are
  unchanged. Different sessions retain independentP/G/capacity values.
- `LayerCache::attention` selects the unchanged compact prefill branch or the
  new prefix-addressing decode adapter. A standalone single-query kernel
  validates four buffer lengths, fixed dimensions and backend before dispatch.
  Copy the current fixed64 head body with exact source anchors; do not let the
  old token-major Compact dispatch accidentally consume head-major arrays.

## Memory and batch implications

Persistent logical FP32 payload remains
`22*4*(1536*P + 1024*Gmax)` bytes, identical to Compact. AtP6544/Gmax4096 this is
**1,253,638,144 bytes (1,195.5625MiB)**, excluding Vec headers, allocator overhead,
workspace and model. There are four persistent vectors per layer rather than
three. The one transient prefill V scratch is **13,402,112 bytes (12.78125MiB)**;
it introduces294,846,464 bytes of additional logical scratch writes over22
layers, not a claim about DRAM traffic. Record actual Vec capacities and process
memory separately; the temporal candidate's141MiB saving does not apply here.

Batch prefills are sequential with shared workspace. The local scratch must be
gone before the next request begins, avoiding aP-sized retained copy per live
session. Persistent memory still sums each session's independent cache budget.
The generated K/V append remains token-major and allocation-free after reserve.
Do not make batch throughput claims from the first single-page experiment.

## Bounded qualification and fair next measurement

1. Storage reconstruction must be bit-exact to Compact for full prefix and
   generated rows, distinct paired prefix keys, unique V selection, capacity
   stability, rejected invalid continuation and overflow before mutation.
2. Attention must match unchanged Compact bit-for-bit for scalar and supported
   AVX2/AVX512: P0 kernel-only, short/full prefixes,127/128/129 crossings,
   P6544 including6528 crossing tile, total context16384, tails, image boundaries,
   sinks, negative values and GQA head selection. SessionP0 remains unsupported.
3. Reuse the preserved operator/smoke/mixed free-output qualification pattern:
   all1,904 canonical tensors plus17 teacher decisions, literal free outputs/
   EOS, and warmed single/mixed decode zero-allocation checks. No GPU tolerance
   changes. Keep both candidate and ordinary Compact selectable in the copy.
4. Inspect the one compiled head to confirm direct head-major prefix indexing,
   unchanged generated branch and existing arithmetic; this is not a speed gate.
5. Then one quiet unchanged-Compact/candidate/unchanged-Compact full-page B1
   bracket, same binary toolchain/flags,16threads,2warmups+3repetitions,1536/4096,
   literal1140-token EOS output equality and frozen5% gain/control-drift rules.
   Use a new explicit-layout protocol with the four-changed/two-added source
   allowlist; the older kernels-only wrapper must not be bypassed. Charge all
   page-time packing/allocation/prefill costs and report stage medians plus memory.

If exact parity, allocation, source closure, resource or output gates fail, stop
that candidate; if the complete page misses the target, record the result
without a layout/block-size/prefetch sweep. This document authorizes no build,
model run, profiling or timing by itself.
