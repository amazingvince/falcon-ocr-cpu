# Prefix temporal-key deduplication: source review only

The pinned FP32 graph permits an additional lossless storage reduction: preserve eight temporal key halves, sixteen spatial key halves, and eight full value heads per prefix token. This is a design finding from source, not an implemented or measured backend. Production and the active functional harness closure remain unchanged.

The parent's [v2 byte audit](prefix-temporal-sharing-byte-audit-v2.json), produced by [audit_temporal_sharing.py](../experiments/cache_layout/audit_temporal_sharing.py), reconstructs all 748 selected K/V tensors bit for bit: twenty-two layers, one 144-token prefix, and sixteen decode steps from the fixed `e2dad223...` Rust trace. It finds 22,528 distinct paired spatial halves, so whole-head prefix deduplication is invalid. Its synthetic changed-duplicate bit is detected. This review inspected the reconstruction and record inventory and independently rehashed its bound files; it did not run new inference or a candidate kernel. These saved bytes support the storage invariant only for the stated trace. The review caught that Python `-O` would strip assertion gates; the author added an explicit optimized-interpreter rejection, reran the unchanged fixture, and preserved the original v1 script/report separately.

The relevant invariants are enforced model dimensions of sixteen query heads, eight original KV heads, and sixty-four channels per head (`src/config.rs:83`). Upstream `_pre_attention_qkv` normalizes each original K head over all sixty-four channels, then `repeat_kv` duplicates each adjacent pair. `apply_3d_rotary_emb` splits channels into temporal `[0,32)` and spatial `[32,64)` halves. Temporal frequencies broadcast across heads; golden spatial frequencies depend on the query-head index. No operation subsequently mixes spatial channels into temporal channels before cache insertion.

The Rust implementation copies the same original K values into paired heads (`src/model.rs:327`), applies identical deterministic width64 RMS reductions to those copies, and performs identical temporal rotations for their first sixteen complex pairs (`src/model.rs:348`, `src/model.rs:972`). Its expansion occurs before normalization, whereas upstream normalization precedes expansion; duplicate input rows still use identical Rust arithmetic. Golden rotations vary only in the second half (`src/model.rs:978`). Batch decode uses the same operations. Q is distinct for each of the sixteen heads and cannot be deduplicated. V has no head-specific rotation and already uses eight-head compact storage. Generated K remains the existing eight-head, sixty-four-channel representation.

A proposed persistent representation is:

| Buffer | Logical shape | FP32 values per token |
| --- | --- | ---: |
| Prefix temporal K | `[prefix_len, 8, 32]` | 256 |
| Prefix spatial K | `[prefix_len, 16, 32]` | 512 |
| Generated K | `[generated_len, 8, 64]` | 512 |
| All V | `[total_len, 8, 64]` | 512 |

For prefix token `t` and query head `h`, reconstruct key channels `[0,32)` from temporal index `(t*8 + h/2)*32` and `[32,64)` from spatial index `(t*16 + h)*32`. Append by copying the already normalized and rotated expanded workspace bits. Do not renormalize halves, recompute rotations, change precision, or merge spatial channels. A future explicit layout should reject other model dimensions until separately supported. Any invariance assertion must compare `to_bits()`, including signed zero, rather than floating-point equality.

The storage change must preserve the existing full 64-channel dot evaluation, not just its algebraic value:

- Scalar decode carries its existing left-to-right accumulator through temporal channels and then spatial channels. Two independent dot32 results followed by addition change rounding.
- AVX2/FMA decode (`src/kernels.rs:1007`) uses four eight-lane accumulators. The first 32 temporal channels update them from zero; the final 32 spatial channels update those same accumulators. Keep the exact FMA operands/order, `(acc0+acc1)+(acc2+acc3)` combination, and horizontal reduction. Merely summing the halves separately breaks this structure.
- AVX-512 decode (`src/kernels.rs:1083`) uses four sixteen-lane accumulators for this dimension. The first two load temporal channels; the final two load spatial channels. Preserve its existing combination and `_mm512_reduce_add_ps` sequence. Do not reuse the AVX2 reduction contract.
- The `1/sqrt(64)` scale remains applied once after the same dot. Attention key order, key tile boundaries, maxima, exp/denominator updates, PV accumulation, learned-sink scaling, and output casts remain unchanged. The split must not become a new softmax tile boundary.

A small correctness-first decode adapter could bit-copy both halves into a stack `[f32;64]` and call the existing selected dot function. It would need no heap allocation, but its copy cost is unmeasured. A later two-pointer SIMD dot can remove the copy while preserving the exact accumulator topology above. Neither design reduces QK arithmetic: both distinct query heads still require all sixty-four products.

For the initial full-prefix prefill, the lowest-risk route is to retain the current expanded `work.k` scratch for the layer, append its bits to split persistent storage, and borrow `work.k` as the prefix argument to the **unchanged existing compact attention function**, together with compact V and an empty generated-K segment. The current prefill already constructs this workspace; this needs no additional full-prefix reconstruction and retains its current GEMM shape, strides, scale, and tile schedule. The cache savings apply to the persistent per-layer storage, not to this existing one-layer workspace. This is a future interface change, not a current implementation.

If generic multiquery continuation later needs to read split cached prefixes, reconstruct one complete original `128×64` key tile (32 KiB) before a single K64 GEMM. Preserve original key order and any prefix/generated boundary within that tile. Do not use two K32 GEMMs and add their results: that changes accumulation/scale rounding. Even a bit-identical gathered matrix can change a library's packing path through its row stride or alignment, so exact-output tests must establish that proposed continuation path rather than assume it. Initial full-prefix prefill can avoid this uncertainty with the borrowed workspace route.

With `P` prefix tokens and `G` reserved generated tokens, the current compact cache uses `22*4*(1536*P + 1024*G)` bytes; the proposed representation uses `22*4*(1280*P + 1024*G)`. The saving is `22*4*256*P` bytes. This is exactly 16⅔% of prefix KV payload, 25% of prefix K alone, and a smaller fraction of a cache that reserves generated tokens.

| Frozen functional input | Prefix tokens | Saving | Proposed reserved KV bytes, cap4096 |
| --- | ---: | ---: | ---: |
| prose | 6,544 | 140.59375 MiB | 1,106,214,912 |
| blank-white | 5,136 | 110.34375 MiB | 947,617,792 |
| sparse-room | 3,088 | 66.34375 MiB | 716,931,072 |
| receipt-cafe | 2,416 | 51.90625 MiB | 641,236,992 |
| Total b4 | 17,184 | 369.1875 MiB | 3,412,000,768 |

For this b4 reservation, current compact payload is 3,799,121,920 bytes, so the saving would be 387,121,152 bytes (10.18975%). These are analytic payload counts, excluding allocations, vector metadata, weights, scratch, page residency and allocator behavior. Separate temporal/spatial streams can increase address work or affect locality; fewer stored bytes do not establish a latency or bandwidth gain.

Before any promotion, require bit-equal stored temporal halves across paired heads at every layer and selected real prefix, then reconstruct complete K/V and compare exact bits. Follow with isolated attention equality against existing compact and expanded oracles for scalar/AVX2/AVX-512, including prefix lengths around128, image-mask boundaries, generated-boundary tiles, long prefixes and strong cancellation. Check the proposed prefill route and decode separately, same-prefix full-model traces, single/mixed free outputs and warmed decode allocations. Keep the current layouts as independent controls. Performance needs a later quiet repeated workload; this design neither changes frozen tolerances nor resolves existing GPU numerical gates.
