# Proposed downstream continuation to the original failure

The smallest useful continuation is **one new GPU layer-8/9 branch**, preceded
by one exact same-input GPU control. Keep the endpoint at the original failing
`layer.9.v[112,14,2]`. Do not rerun layer 7, Rust, the full model, or an earlier
layer merely because hidden coordinate `[112,249]` had a large difference.
This is a source/JSON-only proposal; no implementation, tensor read, numerical
analysis execution, test, build or GPU run occurred for this note. Execution
requires later root coordination after the quiet benchmark.

Let `F` be the already captured native GPU segment: complete layer 8, then the
unchanged layer-9 `_pre_attention_qkv` routine, ending at its expanded V. That
routine includes input RMS/QKV, Q/K head normalization and GQA expansion; retain
it intact even though some returned quantities are unused. There is no layer-9
rotary, attention, FFN or final model head in this scope.

| Symbol | Complete saved state or endpoint |
|---|---|
| G7 | Original GPU layer-7 hidden `[144,768]` |
| C7 | Original Rust layer-7 hidden `[144,768]` |
| S | Saved native GPU layer-7 hidden on the original CPU layer-6 entry |
| A | Existing `F(G7)` |
| C | Existing `F(C7)` |
| D | Existing Rust segment on C7 |
| B | Proposed **new** `F(S)` |

The prospective elementwise FP64 accounting is
`D-A = (D-C) + (C-B) + (B-A)`.

`D-C` is the already measured downstream segment engine term. `C-B` measures
the effect of substituting GPU for Rust layer-7 execution on the fixed CPU
layer-6 entry, **after propagation through the same GPU suffix**. `B-A` carries
the earlier entry-state difference through GPU layer 7 and this GPU suffix.
These are conditional finite differences along this particular substitution
order. They are not independent operator errors or a unique allocation of
nonlinear interactions.

At the fixed endpoint, the prior report has `D-A = +0.006622314453125`,
`D-C = +0.00003814697265625`, and `C-A = +0.00658416748046875`.
Only B is unknown. Require the new decomposition to preserve those saved
values and the original bound `0.005096435546875`; report B's signed difference
from A against that existing bound descriptively, with no new acceptance limit.
Do not transfer the layer-7 hidden-coordinate percentages to this V coordinate.

The existing source makes this continuation feasible without arithmetic
changes. Adapt the native segment and pass-through observations in
`experiments/fp32_crossover/export_gpu.py` into a fresh version. Preserve its
17-stage inventory: input; layer-8 attention_norm, qkv, q, k, v, attention, wo,
attention_residual, ffn_norm, w13, gate, w2, hidden; layer-9 attention_norm, qkv,
v. Execute only these two branches, sequentially:

1. Fresh `F(C7)`: require every bit of **all 17 stages**, including the complete
   input, to reproduce the existing GPU report's `cpu_state.*` payloads. This
   is a stronger saved control inventory than the earlier six original-trace
   endpoints. On any failure, preserve evidence and stop before B, with no retry.
2. New `F(S)`: consume the full saved S tensor unchanged, capture the same 17
   stages, then stop. No cropped rows/heads, recomputed intermediate, interpolation
   or arithmetic variant. Do not rerun A, D or layer 7.

Preserve all 144 rows, original tokens/temporal/spatial positions, pad token,
BlockMask with sequence lengths `(144,144)`, a fresh cache of capacity 256 for
each branch, native uncompiled transformer blocks and original strides/dtypes.
Use the same pinned checkpoint, source/dependencies, RTX4090 UUID, FP32,
TF32/reduced-precision flags off, Flex IEEE setting, cuBLAS `:4096:8`, OMP/MKL 8,
and recorded effective Torch settings. Freeze/review the new observer source
and exact input joins before the future execution; archive/recheck bytes through
completion. No CPU build or model execution is needed.

The saved JSON metadata establishes the required joins:

- G7 raw SHA256 `165e72fa2186cf98816f04f6741545869d85630c75364965d50a53f0eb4155df`
  appears as both the new layer-7 GPU original-control hidden and the old
  layer-8/9 GPU `gpu_state.input`.
- C7 raw SHA256 `890fb0045de1747a15e33b576535b3f2de489cde21358e6dbe76bef305db5dc5`
  appears as both the exact layer-7 Rust control hidden and the old GPU
  `cpu_state.input`.
- S is `cpu_state.layer.7.hidden` in
  `artifacts/diagnostics/fp32-crossover-layer7-gpu-v1/tensors.safetensors`, raw
  SHA256 `846f58d046949cb2b5eaaccf4fc4c3bb99c57871bdd46f805516abc5f4a49ede`.
  Its archive SHA256 is
  `23f81b36dc0deb13a0aca72b757d250e5f5d5451718c215503c441c05fb0776b`;
  its report is bound by the layer-7 completion and independent-review receipts.

These hashes were read from saved JSON, not recomputed here. A future capture
must verify full bytes, shapes, finite FP32 values and identity joins, including
the earlier A/C/D captures and accepted controls. The old GPU A/C archive is
`artifacts/diagnostics/fp32-crossover-gpu-v1/tensors.safetensors`, SHA256
`de8b170d0d9b9688c40fe0848fc77f8a4a1f62ce0f10ba3f76a109f887c88fcf`.
The prior decomposition and independent review bind D and both source closures.

Report the fixed endpoint plus complete tensor/row signed terms, maxima/RMS,
FP64 accounting residuals and cancellation. Fractions may exceed one or be
negative; maxima and RMS cannot be added as signed attribution. Full-state
attention mixing means even this fixed endpoint can depend on other rows.
The test cannot identify the earlier originating operator, prove general
parity, or validate a proposed production fix. Original startup provenance gaps
remain, as do observation/allocation-history changes. A fresh C control is
necessary evidence of replay compatibility; it is not a guarantee about all
possible inputs. Preserve rejected diagnostics and all frozen tolerances.

Evidence used: `reference/fp32-crossover-decomposition-v1.md`, the two saved GPU
reports, the layer-7 Rust report, `reference/fp32-crossover-layer7-completion-v1.json`,
`reference/fp32-crossover-layer7-independent-review-v1.json`, and the pinned
native/observer source. The current assignment stops at this proposal.
