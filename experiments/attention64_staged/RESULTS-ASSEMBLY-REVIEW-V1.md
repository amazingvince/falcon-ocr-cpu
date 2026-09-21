# Independent staged-PV assembly review

The saved, PDB-resolved staged compact head implements the intended scheduling change. Its PV key loop retains eight independent output vectors in YMM registers, applies eight separately rounded rescale multiplications before the loop, performs the original per-channel ascending-key FMA recurrence, and stores the eight vectors after the tile. There are no calls, output loads/stores, or output-accumulator stack accesses in that decoded key loop. No concrete compiled-dataflow blocker was found.

This is a read-only review of saved text and metadata, independent of the candidate author's implementation. The reviewer authored the inspector adapter. The symbol tools were not rerun, and PE/PDB/tool/binary identities are inherited from the accepted inspector receipt. No compiler, model, numerical test, benchmark, recorder or GPU job was executed for this review. This is not a performance result or promotion decision.

## Bound evidence

Candidate directory: `artifacts/diagnostics/attention64-staged-v1/benchmark-build/compiled-staged-head`.

| Saved file | SHA256 checked at review start and end |
| --- | --- |
| `symbol-receipt.json` | `8894a92ad955c982d6ed49e8f6bdf7862886f301bcb4b9cd9f7d876b0610e530` |
| `staged-head-disassembly.txt` | `0784a7ccb81502e3812e0fe114727df7412e455ce6201c9760718baa3f91db85` |
| `staged-head-symbol.txt` | `08037434f6e63a76b784c526da64a47c3a40850c3850382da7a2a2a0772cc21e` |
| `pdb-summary.txt` | `5539abcab14cee890bba3c857e3d9ebae67c0a2d4083fb663ae2f41ba5eae972` |
| Historical compact `head-disassembly.txt` | `3c57260a519cf288bee0f13b09e5a8fc2d3d30d095aa8549e6017d181a2b2343` |
| Historical compact `direct-call-symbols.txt` | `384a9dbd42bb8617e2035f4fe4612c1e9bfc148ce393edd9676e842941f43bf8` |

The historical files are under `artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head`. The three candidate text hashes agree with the receipt's saved-output inventory. All six listed files remained unchanged through this review.

The symbol text identifies exactly `falcon_ocr::kernels::attention64_staged::compact_head`, section 1, offset 718864, size 2337 bytes. The receipt resolves that to `[0x1400b0810, 0x1400b1131)`. The preserved PDB summary agrees with the receipt's PE CodeView GUID `F5987D76-EB15-4BDF-A2F1-67F04024A11B`, age 1. Executable SHA `9414183c6a7989c1899c042edacd018e52ca990f30e7099238833b5f1873e5d9` and PDB SHA `6ad04db393df72f4f2534d7e133c9a3d74f82a17e0d331ea9a8dda572ed5ce23` are inherited from that receipt, not rehashed here.

Related recorded identities are preparation `f848a193ca147b2a1cbe476ab59c6c4b5fce03e078074a9c95043c546e512348`, build `b76ea8940ac3e03b355be61667f2cc4344dc9e579c5a52bfbd2b81084a072cbe`, operators `6e54297b181175aad218e882f6649d9fbd1a63fe3b623562837f4c6379f70528`, and benchmark build `9d45b164cc8c24301e95b820f2e9c6edaabd939481a4a0cf1628e09679b3a6b9`. This bounded review does not repeat their full source or numerical validation.

## Rescale, accumulators and output mapping

At `0x1400b0e22`, `vbroadcastss %xmm7,%ymm11` broadcasts the scalar tile rescale. `0x1400b0e27` loads the output pointer into RDI. The eight `vmulps` instructions below read the previous output once and produce each tile's initial accumulator. None is an FMA with the first value update.

| FP32 channels | Register throughout PV | Separate rescale multiply | Key-loop FMA | Tile-end store |
| --- | --- | --- | --- | --- |
| 0–7 | YMM0 | `0x1400b0e2f` | `0x1400b0eb2` | `0x1400b0ace` |
| 8–15 | YMM1 | `0x1400b0e33` | `0x1400b0eb8` | `0x1400b0ad2` |
| 16–23 | YMM2 | `0x1400b0e38` | `0x1400b0ebf` | `0x1400b0ad7` |
| 24–31 | YMM3 | `0x1400b0e3d` | `0x1400b0ec6` | `0x1400b0adc` |
| 32–39 | YMM4 | `0x1400b0e42` | `0x1400b0ecd` | `0x1400b0ae1` |
| 40–47 | YMM5 | `0x1400b0e4a` | `0x1400b0ed7` | `0x1400b0ae9` |
| 48–55 | YMM7 | `0x1400b0e52` | `0x1400b0ee1` | `0x1400b0af1` |
| 56–63 | YMM11 | `0x1400b0e5a` | `0x1400b0eeb` | `0x1400b0af9` |

Using YMM11 for the last result is sound: its scale value is consumed by all eight multiplies before the last multiply overwrites it. Likewise, the lower XMM7 rescale has already been broadcast before YMM7 becomes an output accumulator.

The stores have lower addresses than the PV loop because the compiler placed the outer-loop continuation earlier in the function. Control flow, not textual address order, establishes their placement: after the PV loop terminates, `0x1400b0f01` jumps to `0x1400b0ac2`; the eight stores at `0x1400b0ace` through `0x1400b0af9` then execute. The next tile is entered only after these stores. They are not executed inside the key-loop back edge.

## Ascending-key PV recurrence

The relevant loop is `0x1400b0ea0` through the `jne` at `0x1400b0eff`, whose target is `0x1400b0ea0`.

- Before entry, `0x1400b0e85` and `0x1400b0e87` zero ECX and EDX. RAX contains the tile's value base, including the shared KV-head offset. R15 contains the full value-token stride in FP32 elements.
- `0x1400b0ea0` computes `j+1` into R11. `0x1400b0ea4` broadcasts the saved FP32 probability at `rsp+0x80+RCX` into YMM12.
- `0x1400b0eae` converts the current key index to a value-element offset with `RDX *= R15`.
- The eight `vfmadd231ps` instructions use value memory offsets 0, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0 and 0xe0. In AT&T form, for example, `vfmadd231ps (%rax,%rdx,4), %ymm12, %ymm0` computes `YMM0 = probability * V[0..7] + YMM0` with one fused operation per lane.
- `0x1400b0ef5` advances the probability byte index by four; `0x1400b0ef9` restores EDX/RDX to `j+1`; `0x1400b0efc` compares the byte index with the tile length in bytes. Each subsequent iteration therefore consumes the next key and probability in order.

Each channel has exactly one accumulator and one FMA per key, with no horizontal reduction, reassociation or second accumulation chain. The value loads are folded into the eight FMA memory operands; there are no separate output-memory operands in these FMAs. RDI remains the output pointer but is not dereferenced in this loop. The only stack-addressed vector read is the probability broadcast, not an output spill. There are no calls, `vzeroupper`, output stores, output reloads or branches out of the loop body other than its terminal conditional back edge.

## Probability and denominator phase

All scalar probability calls finish before the output accumulators are loaded/rescaled. The remainder loop at `0x1400b0d50`–`0x1400b0d77` subtracts the new maximum, calls the shared scalar target at `0x1400b0d62`, adds the returned FP32 value to XMM6 at `0x1400b0d67`, and overwrites that logit slot at `0x1400b0d6b`.

The compiler unrolls the main probability loop four times. Calls at `0x1400b0dc3`, `0x1400b0dd6`, `0x1400b0df0`, and `0x1400b0e0a` are followed by successive additions to the **same** XMM6 denominator at `0x1400b0dc8`, `0x1400b0ddb`, `0x1400b0df5`, and `0x1400b0e0f`. The probability stores occupy consecutive four-byte slots; `0x1400b0e19` advances by 16 bytes. The next tile's probabilities are not processed concurrently with the current tile's PV state.

The denominator is rescaled separately at `0x1400b0d26` (or the initial-zero path at `0x1400b0ba1`) before those additions. The output rescale at `0x1400b0e2f`–`0x1400b0e5a` happens later, as intended, but its operands do not depend on the denominator or probability additions. The finite-runtime/default-FP-environment qualification remains: exception timing is moved; no claim is made about applications observing FP traps or flags.

The candidate text labels the shared scalar call target only with objdump's nearest export, which is not an authoritative function name. Its role is consistent with the pinned source's scalar exponential; this review does not independently add a new public-symbol resolution for the candidate target. Absence of **any** call in the PV loop is directly visible and does not depend on that name.

## Comparison with the frozen compact control

The historical loop has a confirmed `expf` call at `0x1400b1a28`, followed by probability broadcast and eight sequences of a value load, `vfmadd213ps` with an output-memory addend, and output store (`0x1400b1a3b` through `0x1400b1ae8`). Its preserved PDB public symbol gives `expf` section offset 3591194, mapping to call target `0x14036dc1a`.

The staged loop changes the FMA instruction encoding and register destination, but retains the finite-lane recurrence `out = fma(probability, value, out)` in the same key order. Output memory traffic moves from eight 32-byte loads and eight 32-byte stores per key to those loads and stores once per tile, with a small probability-buffer read/write cost. This is evidence for reduced output load/store instructions and L1 traffic; it is not a measurement of cache misses, DRAM bandwidth or latency.

The receipt's original limitation remains: complete decoded byte coverage is unverified, and this review concerns the exact saved decoded loop and its visible surrounding control flow. It does not qualify alternate inlined copies, another ISA, another build, whole-model numerical behavior or benchmark performance. The separate operator/smoke/allocation and timing evidence must still carry those claims.
