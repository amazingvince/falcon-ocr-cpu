# Proposed FP32 frozen-state crossover diagnostic

Status: read-only proposal; no implementation or execution. Run only after the
active quiet performance bracket finishes. This does not change production
arithmetic, numerical policy, qualification, or benchmark sources.

The first prefill failure is layer 9 V, at row 112 (image token 227): maximum
absolute error 0.006622314453125 exceeds its frozen bound 0.005096435546875.
Existing equal-input layer-9 QKV error is much smaller (about 0.000053406).
The worst rows coincide across the eight prefill failures. Isolated arithmetic
changes are not a sufficient explanation: matching observed GPU W2 partitions
increased full-graph failures from 10 to 23, and the GPU-rsqrt width-768
intervention increased them from 10 to 18.

One bounded diagnostic would replay the original layer 8 and layer-9 V
projection from saved GPU and production CPU layer-7 hidden states, preserving
the full 144-by-768 input shapes, masks, positions, dtype boundaries and native
operators. Use three branches:

1. Original GPU operations on the saved GPU state.
2. The same GPU operations on the saved CPU state.
3. Original Rust operations on the saved CPU state.

Branches 1 and 3 must first reproduce their respective saved layer-8 hidden and
layer-9 V bits. A failed control rejects the experiment; no attribution follows.
Retain normal intermediate stage outputs for all rows: normalization, QKV,
attention, output projection/residual, FFN normalization/W13/gate/W2/residual,
and the next normalization/QKV/V. Do not substitute an alternative operator or
change a reduction order to make a control pass.

At each retained stage, the signed CPU-minus-GPU difference decomposes into
Rust-on-CPU minus GPU-on-CPU (local arithmetic on the actual CPU trajectory),
plus GPU-on-CPU minus GPU-on-GPU (propagated incoming-state difference). Offline
FP64 accounting can measure these terms and their cancellation while preserving
the original FP32 payloads. Report row 112 alongside every other row, including
nonfailing ones; do not choose a new acceptance tolerance from these results.

Stop after one validated decomposition. Report whether incoming-state growth,
local arithmetic or cancellation explains the observed failing coordinate, and
retain uncertainty where the evidence is mixed. This is not authorization for
a combined arithmetic variant or a stage sweep. The two decode-attention
failures remain separate. Existing numerical bounds remain unchanged.

Prior evidence is summarized in `docs/STATUS.md`,
`reference/rms-intervention-production.json`, and
`reference/w2-observed-fullgraph-intervention-v1.json`. Before implementation,
resolve and bind the exact saved state files, original executable/reference
sources and numerical-policy artifact from those records; this proposal does
not fabricate startup or payload identities.
