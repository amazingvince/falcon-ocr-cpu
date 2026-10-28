# Compiled head-contiguous prefix review

Manual review of one saved function found the intended prefix K/V address
change and preserved fixed64 arithmetic. This is compiled-code evidence, not a
speed claim or a proof about the entire binary. No build, model, test, symbol
rescan or benchmark was run for this review.

The exact PDB procedure is
`falcon_ocr::kernels::head_prefix64::prefix_head`, section 1, decimal offset
728080, 2,495 bytes: VA `[0x1400b2c10,0x1400b35cf)` (RVA
`[0xb2c10,0xb35cf)`). PDB GUID is
`22152B33-7443-448F-BF90-DA9729B7B8C0`, age 1. The objdump
`onig_get_start_by_callout_args` label is an unrelated nearest export and is
not used for attribution.

## Physical addresses

The Windows argument mapping and setup identify `P` at `[rsp+0x398]`, prefix K
base in `r12`, prefix V base at `[rsp+0x368]`, generated K base at
`[rsp+0x378]`, and generated V base at `[rsp+0x388]`. Setup stores `head*P`
at `[rsp+0x98]`, `kv_head*P` at `[rsp+0x90]`, `kv_head*64` at
`[rsp+0x48]`, and `kv_width` at `[rsp+0x40]`. These are element offsets;
the final address uses a factor of four for FP32 bytes.

| Path | Observed instructions | Effective element address |
|---|---|---|
| Prefix K | `0x1400b3093..0x1400b30be`: compare absolute key with P, add `head*P`, shift by 6, bounds check, direct base load | `(head*P+key)*64` |
| Generated K | `0x1400b30d0..0x1400b30fc`: subtract P, multiply by kv_width, add kv_head*64 | `(key-P)*kv_width+kv_head*64` |
| Prefix V | branch `0x1400b333d` to `0x1400b3230..0x1400b3256`: add kv_head*P, shift by 6, select prefix base | `(kv_head*P+key)*64` |
| Generated V | `0x1400b3343..0x1400b3372`: subtract P, multiply by kv_width, add kv_head*64, select generated base | `(key-P)*kv_width+kv_head*64` |

Both K branches join the same dot body at `0x1400b2fc0`. Both V branches
join the same pointer formation/AXPY path at `0x1400b3256`. Prefix accesses
advance by 256 bytes for each key within a head; generated accesses retain
the 512-float token stride. All 16 prefix K heads and 8 V heads remain
addressable. There is no temporary reconstructed key in the reviewed dataflow.

## Arithmetic and key order

QK at `0x1400b2fc0..0x1400b305c` has eight YMM FMAs. The first four
accumulate dimensions 0..31 from zero; the next four add dimensions 32..63 to
those same four accumulator histories. At `0x1400b303e..0x1400b3058`, the
tree is `(acc0+acc1)+(acc2+acc3)`, followed by upper/lower 128-bit addition
and two horizontal adds. Scalar scale multiplication follows at
`0x1400b305c`. No QK accumulator is written to the stack in this body.

`rbp` is the absolute tile start, initialized to zero at `0x1400b2ed7` and
advanced by 128 at `0x1400b2ef0`; each tile length is capped at 128 at
`0x1400b2f3c..0x1400b2f50`. Prefix comparisons change addresses only, not
these counters or online-softmax state. The P=6544 crossover therefore stays
inside the tile starting at 6528. The visible-end setup at
`0x1400b2d20..0x1400b2d36` still selects image_end for an in-image query,
otherwise absolute_query+1.

Online rescaling calls scalar `expf` at `0x1400b3118`. Separate `vmulps`
output rescaling at `0x1400b3170..0x1400b3198` and scalar denominator
`vmulss` at `0x1400b3211` precede the next probability loop; they are not
folded into its FMAs. Per-key probability uses the saved logit, subtraction
and `expf` at `0x1400b3317..0x1400b3329`. Denominator addition remains
one scalar `vaddss` per ascending key at `0x1400b325a`. Eight
`vfmadd213ps`/store groups at `0x1400b3263..0x1400b32fc` update output
dimensions in byte offsets 0,32,...224, retaining key-ordered AXPY. This is
the baseline memory-backed output recurrence, not the rejected staged-PV
experiment.

After all tiles, `logf` at `0x1400b33a8`, the running-max addition, sink
subtraction and `expf` at `0x1400b33bb` form the original sink scale.
`vaddss` then `vdivss` at `0x1400b33e1..0x1400b33e5` implement the
reciprocal of one plus that exponential. Final output normalization uses
`vdivps` by the denominator followed by separate `vmulps` by sink scale
at `0x1400b3430..0x1400b3463`, with corresponding fallback tails retained.

## Calls and stack scope

All 14 parsed calls are direct numeric targets: three static `expf` call
sites, one `logf`, one `memset`, and nine bounds/division panic sites.
Their saved PDB records resolve exact target addresses, including explicit
mangled/demangled aliases for panic routines. No indirect call or tail
transfer appears among the parsed instructions. Dot and AXPY are inline;
this statement does not imply absence of library work in scalar exp/log.

The function reserves `0x2f8` bytes after eight general-register pushes.
`[rsp+0xa0,rsp+0x2a0)` is the deliberate 128-F32 logit array: initialization,
per-key store at `0x1400b3060`, and probability-loop load at
`0x1400b3317` account for its traffic. XMM6..10 saves at offsets
`0x2a0..0x2e0` and matching epilogue loads are Windows callee-save traffic,
not dot/PV accumulator spills. Remaining stack slots hold arguments,
counters and address state. The output AXPY stores target the output buffer.
No additional floating-point accumulator spill is visible in this function.

The saved listing lacks raw instruction bytes, so complete byte-by-byte
decode coverage remains unverified. The claim is limited to the resolved
function and inspected dataflow. Memory-stall cause, hardware traffic,
latency and overall performance cannot be inferred from these instructions.

## Evidence

- Inspector receipt: `artifacts/diagnostics/head-contiguous-prefix-v1/benchmark-build/compiled-prefix-head/symbol-receipt.json`, SHA256 `20874cba1c991caeaee0cf1589e1b3a0c0d9a684b5ff3d180c9acf5543eebec2`.
- Disassembly SHA256 `b0aca644ad714e83e2bc0aaf7e2a8fcce1315de3167c5776ee1cd1eb029f78ca`.
- Direct-target evidence SHA256 `141bbdf22ae99632988807a9da9feafef6edc3b8e68005a1bbd50f239015058c`.
- Recorded benchmark PE SHA256 `5695e415c58432305905d4e7f46cdf958118249ad880816434dd6fc75e2afe52`; matching PDB SHA256 `ef0c1f723e723bb9b66ed111c4ee36c63aa05b9515c1c68be96c1a79d36c6893`. This manual review relies on the inspector's binary/PDB closure and does not rehash either large file.
- `COMPILED-REVIEW-V1.json` binds the small files actually reviewed and rechecks their unchanged bytes before writing its result. Original files and receipts remain untouched.

The inspector's first output-file digest was collected after writing, leaving
a save-to-final-hash provenance window. This manual review binds the current
saved text explicitly; it does not retroactively eliminate that window or
claim that the inspector originally analyzed immutable output files. A
separately preserved rerender check may supply additional evidence.
