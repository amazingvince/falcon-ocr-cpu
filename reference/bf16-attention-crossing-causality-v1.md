# BF16 probability crossings: saved-evidence diagnosis

The four AVX512 probability crossings are explained by changed **arguments** to
`exp2` in the saved CPU-versus-independent-GPU comparison. Replaying Rust `exp2`
with the saved GPU score/maximum arguments reproduces every GPU BF16 probability
across the five heads and fifteen live tiles. Online maximum selection itself
matches its defined operation exactly; the maximum's value differs because its
winning QK dot differs. This does not yet identify the actual fused Flex kernel's
intermediates, nor provide a replacement QK reduction.

The [machine-readable analysis](bf16-attention-crossing-causality-v1.json) pins the
original GPU/CPU tile files, operand fixture, CPU diagnostic sources and executable.
The CPU-only helper evaluated 1,876 distinct FP32 arguments. It exactly reproduced
all saved scalar and AVX512 CPU `exp2`/BF16-cast values before any counterfactual was
interpreted. No GPU code or previous QK reduction candidate was run.

## Four crossings and one control

All four AVX512 crossings occur in the live key-0..63 tile. Each maximum uses the
same winning key in the CPU and saved GPU oracle. All score-scaling operations
and all thirty saved online maximum updates (two backends) were reproduced
bit-for-bit from the saved QK/scores. There is no NaN or max-order ambiguity in
these probes.

| Query/head | Probability key | Maximum key | Evidence from swapping saved score/maximum inputs |
| --- | ---: | ---: | --- |
| Layer 0, row 13, head 1 | 63 | 1 | Both local score and maximum differ. Replacing either with the GPU value moves the Rust cast onto the GPU side. Original CPU probability lands exactly on the BF16 midpoint. |
| Layer 0, row 53, head 8 | 43 | 23 | Maximum is identical. A one-FP32-ULP argument shift from the local QK difference crosses the midpoint. |
| Layer 17, row 54, head 10 | 22 | 2 | Both local score and maximum differ. Replacing either in the AVX512 trace restores the GPU-side cast; scalar has a larger local-score error and needs the score replacement. |
| Layer 17, row 124, head 11 | 23 | 2 | Local QK and score are identical. The maximum-anchor QK difference alone shifts the cast. |
| Layer 0, row 13, head 2 | — | 0 in the full tile | Nonfailing control: many QK/FP32 exponential differences, but no BF16 probability crossing. |

For example, layer17/row124/head11 has identical local QK
`-0.9768662452697754`. The running maximum is `3.1338367462158203` in the GPU
oracle and `3.1338372230529785` in the AVX512 trace. Those yield arguments
`-3.310001850128174` and `-3.310002326965332`. The BF16 probability changes from
`0.10107421875` to `0.1005859375`; replacing the maximum alone restores the GPU
cast. The maximum difference is traced to key2 QK values
`17.377681732177734` versus `17.377683639526367`.

These are diagnostic input substitutions, not proposed fixture-specific kernels.
They do not justify returning to the eight discarded reduction hypotheses. The
earlier [complete-fixture investigation](bf16-qk-reduction-investigation.md)
already found that none qualified all gates, and more exact individual QK scores
did not reliably improve attention outputs.

## What the exponential and normalizer evidence establishes

There are 665 unmasked exponential values across the five heads. At the same saved
GPU arguments, Rust and independent Torch GPU `exp2` differ at 415 FP32 values,
but **zero BF16 casts differ**, including all four crossing coordinates. Thus
the exponential implementation difference at those particular GPU arguments is
insufficient to explain these BF16 probability mismatches.

That conclusion is deliberately conditional. GPU exponential results at the
**CPU** arguments are not saved. At a midpoint, changing the exponential
implementation there could change a cast. Furthermore, the native fused kernel
uses observed `ex2.approx.ftz.f32`, whereas the saved GPU tile oracle calls
`torch.exp2` separately. Its actual arguments/results must be captured before
exonerating native GPU approximation or choosing a CPU approximation.

The denominator update follows the unnormalized local-probability cast and does
not feed back into it. Its reduction/FMA order therefore cannot cause these four
earlier probability crossings. It can still affect final normalization, LSE,
sink scaling and subsequent BF16 output casts. Existing LSE gates pass, but that
does not prove every final cancellation-sensitive coordinate is unaffected.
The separate layer19 scaled-output failure has no selected tile trace here and
remains unresolved by this analysis.

## Why the next diagnostic must observe fused Flex

The existing GPU tile artifact explicitly contains independent `torch.bmm` and
`torch.exp2` intermediates, not stores from inside Flex. Its five final raw head
vectors match native Flex, but that does not establish equality at each internal
stage. Its first tail dot uses N=16 live keys, while Flex uses a masked N=64 tile.
It computes separate `PV`, then adds the scaled accumulator; Flex passes the
scaled accumulator directly into `tl.dot(P,V,acc)`.

The saved generated source also caps the partial loop by the number of nominal
tiles, not the remaining live keys. It can execute key128 and an all-masked
key192 tile before full key0/key64. The old oracle omits key192 as a presumed
no-op. A native trace must preserve and label every original iteration.

The [queued exporter specification](bf16-fused-substage-export-spec-v1.json)
requests only the two original full attention cases, with the original five
selected probe heads. It preserves complete input shapes/strides, mask inputs,
M128/N64/D64 tiles, sparse ordering and actual launch constants. Observation
stores expose QK, score/max/subtraction, exponential/BF16 probability, denominator
and accumulator boundaries without replacing arithmetic. Actual source/PTX/CUBIN
and warp/stage/split metadata must be retained.

Before interpreting any stored value, the instrumented **complete** raw BF16
attention and FP32 LSE must match the uninstrumented pinned Flex outputs
bit-for-bit. Any discrepancy rejects the instrumented probe. Extra stores can
change register pressure, layouts or compiler scheduling; preserving source
expressions is insufficient. Inspect original/instrumented instruction chains as
well, since final-output equality is necessary but does not prove identical
intermediates. Only an accepted capture should be followed by GPU/Rust exponential
replay at identical original **and CPU** argument bits.

`gpu_reference` confirmed this diagnostic is queued after its existing ordered
jobs. No GPU run was initiated here, no production source changed, and all frozen
BF16 bounds and existing failures remain unchanged.

CPU replay command:

```powershell
rustc --edition 2024 -C opt-level=3 experiments/bf16_attention/exp2_probe.rs -o artifacts/bf16-attention-crossings/exp2-probe.exe
python experiments/bf16_attention/analyze_crossings.py --exp2-binary artifacts/bf16-attention-crossings/exp2-probe.exe --output reference/bf16-attention-crossing-causality-v1.json
```

The analysis refuses to overwrite its output; use a new filename to repeat it.
