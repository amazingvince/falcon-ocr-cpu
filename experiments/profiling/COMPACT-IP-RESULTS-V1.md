# Compact-head instruction samples — 2026-09-20

The saved histogram places most compact-head leaf samples in **QK dataflow
(56.10%)** and **PV update blocks (42.85%)**. This supports investigating the
head's data access and computation together; it does not identify cache misses,
bandwidth saturation, instruction latency, or a cause for the rejected staging
experiment. The strict complete-profile audit remains **failed** on its one
preserved lifetime exception. These are explicitly qualified diagnostic counts.

This review joined only small saved JSON/text files. It ran no ETLX scan,
symbol tool, compiler, model or benchmark. The parent-run helper/build had
already completed; its14 host checks and exact original-audit count joins are
separate evidence from this address interpretation.

## Exact join and exhaustive regions

PID79568/index557 has **2,692,193 in-bounds samples**. **1,147,243** are leaf IPs
inside compact_head (42.6137% of all in-bounds samples). All **214 distinct leaf
RVAs** exactly match instruction-start addresses in the pinned disassembly;
**unmatched leaf RVAs/counts: 0/0**. No nearest-instruction mapping was used.
All89 distinct external-leaf head-frame RVAs also match starts, without implying
they are all call sites. No head-leaf sample carries DPC/ISR/NonProcess flags.

The join is `disassembly VA - 0x140000000 = sample IP - 0x7ff79bdf0000`.
The function extent is `[0xb13f0,0xb1d36)`; module path and PDB
`1797F1F9-EC73-4291-98A8-5C9C8AAF2AE1`, age1, match the closed receipt and trace.
Below every range is **start-inclusive/end-exclusive**. The adjacent ranges
exhaust the entire extent, including padding and error paths. Labels describe
the surrounding code, not a attribution of every stall to its sampled opcode.

| RVA range | Code region | Leaf samples |
| --- | --- | ---: |
| `[b13f0,b16c0)` | Prologue, indexing, zeroing, setup | 226 |
| `[b16c0,b1780)` | Tile iteration and maximum control | 208 |
| `[b1780,b17a0)` | QK loop setup | 30 |
| `[b17a0,b1823)` | QK vector loads and eight FMAs | 601,672 |
| `[b1823,b1845)` | QK reduction and scale | 31,097 |
| `[b1845,b186b)` | Logit store, maximum and loop control | 9,218 |
| `[b186b,b18e0)` | Prefix/generated K addressing and bounds | 1,640 |
| `[b18e0,b1908)` | Rescale-exp call/return surroundings | 21 |
| `[b1908,b19d5)` | Output and denominator rescale | 343 |
| `[b19d5,b1a16)` | PV loop setup and V addressing | 4,271 |
| `[b1a16,b1a3b)` | Probability-exp call, denominator, broadcast | 3,815 |
| `[b1a3b,b1af0)` | Eight PV load/FMA/store blocks, adjacent increment | 491,637 |
| `[b1af0,b1b04)` | PV loop tail | 2,965 |
| `[b1b04,b1d36)` | Final sink/normalization, epilogue, error paths | 100 |
| **Total** | | **1,147,243** |

Thus the complete QK region `[b1780,b18e0)` contains **643,657** samples
(56.1047% of head;23.9083% of all in-bounds). The PV update block contains
**491,637** (42.8538%;18.2616%). The complementary regions contain **11,949**
(1.0415%;0.4438%). Scalar math executing outside the function is **not** counted
in these leaf-region totals.

## Largest instruction locations

| RVA | Samples | Instruction / operand role |
| --- | ---: | --- |
| `b17c5` | 294,962 | `vfmadd132ps 0x20(%rax), %ymm9, %ymm3` — QK, K memory operand |
| `b1a42` | 184,681 | `vfmadd213ps (%rbx), %ymm0, %ymm1` — PV, output accumulator memory operand |
| `b1810` | 135,810 | `vmovups 0xe0(%r14,%rsi,4), %ymm4` — Q-vector load |
| `b1ac1` | 102,539 | `vfmadd213ps 0xc0(%rbx), %ymm0, %ymm1` — PV accumulator operand |
| `b17d1` | 98,400 | `vfmadd132ps 0x60(%rax), %ymm9, %ymm5` — QK K operand |
| `b1a64` | 75,685 | `vfmadd213ps 0x40(%rbx), %ymm0, %ymm1` — PV accumulator operand |
| `b17ea` | 65,346 | `vmovups 0xa0(%r14,%rsi,4), %ymm2` — Q-vector load |
| `b1a8b` | 60,454 | `vfmadd213ps 0x80(%rbx), %ymm0, %ymm1` — PV accumulator operand |

The prominent Q-vector loads are an especially useful warning against reading
sampled memory operands as measured cache misses: the same query is reused
across keys. Sampling skid, instruction scheduling and dependency chains are
unmeasured. The earlier staging candidate removed repeated output stores but
lost whole-page performance; these samples do not reverse that result.

## External leaves remain separately qualified

**46,773** samples have a leaf outside the head and a saved head-frame RVA;
none has multiple distinct head-frame RVAs. **16,013** carry one or more
DPC/ISR/NonProcess flags, retained as a reported subset, not silently removed.
The largest saved RVA is `b1a2d`: **29,679** samples, including380 flagged.
It is exactly the next instruction after the call at`b1a28` to the known expf
target. Similarly `b18ed` has549 (8 flagged), after the rescale-exp call;
`b1b3a` has100 after logf; `b1b4c` has3 after final expf; and `b154f` has52
after memset. The pinned direct-call symbol evidence identifies those static
targets, not the dynamic external leaf for every sample.

This is consistent with call-return ancestry, but the histogram does not retain
the external leaf module/address per bucket. Do not relabel29,679 as proven
expf executions or time. That bucket is1.1024% of all in-bounds observations;
all external-head-frame samples together are1.7374%. Neither is a strict upper
bound on possible exp optimization gains: missing stacks, skid, interactions
and the distinction between CPU sample shares and wall time remain.

## One next isolated hypothesis

A **head-contiguous compact K/V cache** is a better-supported next experiment
than an exp approximation. Present prefix K traversal advances4,096 bytes per
key for one head; generated K and V advance2,048 bytes. Head-contiguous storage
could make the same64-value key/value segments consecutive at256-byte strides.
This changes access layout while retaining the exact four QK accumulator
histories/reduction, ascending key order, eight PV FMAs, scalar exp, tile size,
denominator and sink math. It must retain16 independent prefix K heads and8
generated K/V heads; whole-head prefix deduplication remains invalid.

Use the unchanged combined compact baseline, not the rejected staged/GEMV/
temporal candidates. Borrow expanded prefill work as before, pack retained
prefix storage once, and include packing cost plus generated-token append costs
in any later end-to-end comparison. Head-major capacity/indexing, GQA sharing,
mixed prefix/generated boundaries and warmed zero-allocation behavior need
bit-exact operator/trace/output checks before timing. The existing experiments
supply that protocol, not a predicted gain. This is a locality hypothesis, not
a claim that bandwidth or TLB misses caused the present sample distribution.
Exp approximation would add a numerical change without comparable observed
support and should not be the next optimization on these data.

## Bound evidence

Small input hashes were verified during this review and rechecked at closure.
No large binary, PDB or ETLX was read or rehashed. Their identity claims remain
those of the preserved build/trace receipts. The diagnostic retains the original
one rejected sample and inherited ETLX-hash limitation; see
[boundary review](COMPACT-BOUNDARY-REVIEW-V1.md).

| Input | SHA-256 |
| --- | --- |
| `D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/compact-ip-histogram-v1.json` | `267897446850d8065be8957940efd64dc565111ecea83830e30adb945149acb6` |
| `artifacts/tools/compact-ip-histogram-build-v1/build.json` | `56759932cc1b1a7a0f51e2c01ef4c07bb1a22b04d901e239b38adf8f902486a5` |
| `artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/symbol-receipt.json` | `ea4842b0289c971a72f6e22bb0d5d10b623743376685159e34ef894444f3acf3` |
| Same directory: `head-disassembly.txt` | `3c57260a519cf288bee0f13b09e5a8fc2d3d30d095aa8549e6017d181a2b2343` |
| Same directory: `direct-call-symbols.txt` | `384a9dbd42bb8617e2035f4fe4612c1e9bfc148ce393edd9676e842941f43bf8` |
