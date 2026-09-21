# Pending uniform PV isolation

Read-only follow-up, 2026-09-20. No candidate execution, production edit or bound
change was made. Execution waits for the queued exact-shape fused export, whose
complete raw output and LSE must first match uninstrumented Flex bit for bit.

The existing layer-19 analysis establishes a sufficient counterfactual, not a
uniform kernel result: replacing just tile-0 PV[54] with an independent FP64 sum
of the same CPU BF16 probabilities and V moves row 41/head 6's raw quotient below
the BF16 midpoint. The recorded PV change is `1.4901161193847656e-7`; its original
quotient is only `1.7229467630386353e-8` above that midpoint. This is evidence that
PV accumulation error can matter at this boundary. It does not show that applying
a different reduction consistently across every tile/head will match the GPU.

Current `src/bf16_attention.rs` computes each 64-term PV from zero, then adds it to
the FP32 rescaled previous accumulator. Its scalar dot is an ascending FP32 sum
of promoted BF16 products. Its AVX-512 BF16 dot uses two 32-element updates and a
horizontal FP32 lane reduction. The preserved generated Flex source instead
passes the rescaled accumulator directly to `tl.dot(P, V, acc)`. Those are distinct
rounding boundaries even if QK, probabilities and alpha are identical.

The next bounded diagnostic should freeze QK, maximum, exp2, BF16 probabilities,
alpha, denominator, scheduling and final casts. Apply the following mathematically
defined reductions uniformly, not only to the failing coordinate:

1. Compute every 64-term PV with independent FP64 summation of exactly promoted
   BF16 operands, round the tile sum once to FP32, then retain the existing FP32
   rescale/add. This isolates rounding inside the dot computed from zero.
2. As a separate diagnostic, include the already-FP32 rescaled previous
   accumulator in the FP64 sum before one final FP32 rounding. This isolates the
   extra from-zero/add boundary. It is an ideal fused-sum reference, not a claim
   to reproduce NVIDIA MMA's accumulation order.

Compare complete local raw/LSE/scaled outputs for all six preserved attention
cases, not just layer 19; include every originally failing element and the
nonfailing controls. Record exact changed BF16 probability counts as zero in this
PV-only experiment. Use the existing frozen element bounds and report regressions
and negative results. Neither diagnostic changes the model or qualifies a backend.

Interpretation requires the accepted fused stores: compare CPU and native BF16 P,
rescaled input accumulator, post-`tl.dot` accumulator, denominator and pre-cast
quotient. If P or alpha differs, a PV-only counterfactual cannot isolate the native
cause. If P and accumulator inputs agree while the post-dot accumulator differs,
the additional native evidence can justify a structurally motivated PV candidate.

Existing evidence: `reference/bf16-layer19-rounding-analysis-v1.json`,
`reference/bf16-layer19-rounding-analysis-v1.md`,
`reference/bf16-sink-replay-cross-platform-v1.json`, and
`reference/bf16-fused-substage-export-spec-v2.json`. Saved independent GPU tile
internals remain distinct from actual fused internals until that export passes
its whole-output equality gate.
