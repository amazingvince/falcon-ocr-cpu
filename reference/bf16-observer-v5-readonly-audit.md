# BF16 observer v5: rejected capture and next mechanism

The single authorized correction remains rejected. Removing the extra debug
probability-sum store did not restore the native masked reduction. All complete
raw BF16 tensors match, but natural LSE differs in 7/9/12 elements and log2 LSE
in 7/11/13 for layers 0/17/19. The original kernel matches the frozen fixture.
The mandatory reduction-structure guard also fails. No v5 intermediate is an
accepted native observation. The local probability sum is explicitly unavailable
under the v3 partial specification; v2 completion is not claimed.

`bf16-fused-observer-v5-summary.json` binds the full report, captures and compiled
artifacts. Preprocessing independently rechecked their hashes and recomputed the
negative lowering result. This audit only reads existing files and disassembles
existing cubins. It does not implement, compile or execute another candidate.

## Remaining observation effects

The original masked-loop probability sum reduces a `128x64` FP32 MMA tensor.
Its four row reductions per thread each combine 16 local exp2 values with 15
local additions, then XOR-2/XOR-1 exchanges and two further additions. Across
the partial and full loops, the guard finds eight such denominator FMA addend
DAGs. Their MMA encoding is version 2, warps `[4,1]`, instruction shape `[16,8]`.

V5 still creates three score/max/exp2 representations: `#blocked`, `#blocked3`
and `#mma`. In its saved `instrumented-0.ttgir`, lines 469–528 show the separate
maximum and probability paths. The remaining matrix observations store exp2
arguments and results through the blocked representation. The probability
operand of the native dot uses the MMA representation. Thus a source expression
with an observation consumer can be replicated into a different layout; the
source-level variable name alone does not prove that a store observes the exact
register later consumed by the dot.

The scalar observations still consume the blocked3 row values through
conversions to `#blocked2`: maximum-before at lines 457–458 and
denominator-before at 461–462; maximum-after, alpha argument and alpha result
at 496–511; and denominator-after at 551–552. The native denominator update
itself remains `#blocked3` at 543–548. Its sum is 63 serial additions over 64 exp2 values with
no shuffle. This denominator is carried by the partial loop and converted back
to MMA only after that loop at line 643. The full-block sum retains the native
MMA structure.

These are observed use/layout paths, not proof that one particular remaining
store uniquely causes the compiler's choice. The failed single deletion is
evidence against attributing the perturbation solely to the removed sum store.
No further store-ablation or layout sweep is proposed.

## Recommended next approach

If further implementation is authorized, use one bounded observer inserted
into the **already emitted original PTX**, after Triton has chosen its layouts
and lowered its reductions. Add a separate debug pointer, fresh integer/address
registers and predicated global stores that copy existing native registers.
Retain every original arithmetic instruction, operand, predicate, branch,
barrier and loop iteration. Observe the registers actually feeding the native
BF16 conversion, MMA accumulator, denominator update and final division;
do not recompute values for export. Preserve a mechanical check that removing
only the added debug declarations/addressing/stores reconstructs the original
PTX bytes. The fixed original artifact hash supplies all patch anchors.

The first feasibility gate should round-trip the **unmodified** PTX through
the same assembler and launch ABI, then compare its code and complete outputs
with the preserved original cubin and frozen fixture. Only after that control
passes should a single stores-only candidate be built. Use the original grid,
128-thread blocks, 27,136 shared-memory bytes, full Q/K/V strides and sparse
schedule. Record raw per-thread register slots with explicit mapping evidence
to query/head/key, so a mistaken MMA lane mapping cannot become a numerical
claim. Every debug write needs a bounded, unique destination and coverage check.

This is feasible through supported CUDA module loading: the Driver API accepts
PTX or cubin modules. Prefer offline assembly with the pinned assembler and
load the resulting cubin, so a different driver's PTX JIT is not silently
introduced. [CUDA module loading](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__MODULE.html)

It is **not guaranteed** to preserve the machine program. PTX is a virtual ISA
and `ptxas` optimizes it; extra stores alter liveness and register pressure.
The original PTX includes floating operations without explicit rounding
modifiers, for which NVIDIA permits aggressive optimization, including
multiply/add contraction. Do not add rounding modifiers, disable fusion or
change optimization levels to manufacture a pass. Inspect the emitted SASS
and keep complete raw BF16, natural-LSE and log2-LSE bit equality as mandatory
gates. Equality still qualifies only these observed inputs.
[PTX compilation and arithmetic rules](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-add)

Saved control and v5 cubins were disassembled with `nvdisasm -ndf -g` into
`artifacts/reference/bf16-observer-v5-audit`. This reads their machine code and
does not run it. The tool supports cubin disassembly and register-liveness
inspection for the subsequent audit.
[CUDA binary utilities](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-binary-utilities/index.html#nvdisasm)

## Assembler identity matters

The pinned Torch compile-worker helper selects
`torch/bin/ptxas`, version **13.0.88**, SHA256
`daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2`.
This differs from Triton's default bundled 12.8.93 and the system 13.1.115.
The audit-time helper probe confirms that selection with CUDA uninitialized;
it is not retroactive process-start attestation for old captures. Installed
compiler-selection sources and the probe receipt are preserved alongside the
disassemblies.

The saved original metadata enables FP fusion, has no extra PTX options and
targets `sm_89`; emitted PTX is version 9.0. The installed Triton assembly path
uses line information and its normal optimization level. A future round-trip
must capture the actual selected executable, flags, environment and outputs
at startup. It must fail on an unmatched control rather than trying alternate
assemblers or flags.

No source-level layout-control mechanism was validated by this audit. A
Triton rewrite that explicitly constrains layouts would require a larger new
equivalence argument, while editing emitted PTX avoids the specific layout
selection stage already observed to change the reduction. That is the reason
for recommending this one next mechanism, subject to the unchanged gates.
