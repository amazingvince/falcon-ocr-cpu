# Bounded CPU-prefix / GPU-suffix assessment

A single fixed set of prefix substitutions can localize the earlier-state term
at **layer-9 V `[112,14,2]`** more efficiently than serial layer crossovers.
All needed original entry states are already saved; no new Rust inference or
build is needed. This assessment changes no numerical gate and proposes no
arithmetic variant. Only source and saved JSON were read; no tensor payloads,
model, tests or GPU tools were run.
The efficiency is one frozen runner, one resident model and comparisons at the
actual failing endpoint; it is not a claim of fewer GPU block calls than every
possible adaptive search.

Let `C_b` be the complete saved CPU state after `b` transformer blocks:
`C_0` is `prefill.embedding`, and `C_b` for `1..9` is
`prefill.layer.{b-1}.hidden`. Each is FP32 `[144,768]`. Define

`E_b = GPU_layer9_pre_qkv_V(GPU_blocks[b..8](C_b))`, for `b=0..9`.

For `b=9` the block sequence is empty. Retain the **entire unchanged native
layer-9 `_pre_attention_qkv`**, including Q/K work and GQA expansion, and select
its returned V; do not replace it with a standalone V projection. The endpoint
is `[144,16,64]`. A is the original GPU endpoint and D the original CPU endpoint.
The elementwise accounting, performed after widening every endpoint to FP64, is

`D-A = (E_0-A) + sum[b=0..8](E_{b+1}-E_b) + (D-E_9)`.

The first term measures the saved embedding-state substitution. Each middle
term measures substituting one CPU block along this fixed CPU-prefix/GPU-suffix
order. The final term isolates the layer-9 native pre-QKV path on the common
saved CPU layer-8 state. These are propagated, conditional finite differences,
not independent kernel errors or a unique allocation of nonlinear interactions.

## Available evidence and missing results

- Original GPU archive: `artifacts/reference/smoke-fp32/trace.safetensors`,
  SHA256 `30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4`.
  Original CPU archive: `artifacts/cpu/smoke-trace-sinks-pairwise.safetensors`,
  SHA256 `e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309`.
  Saved source and comparison inventories establish embedding, hidden, Q/K/V
  and attention captures for every relevant block. GPU prefill names have no
  `prefill.` prefix; CPU names do. Future preparation must reject alias
  collisions and verify every complete entry's bytes, shape, dtype and finiteness.
- Tokens `[144]`, temporal positions `[144]`, spatial positions `[144,2]`,
  configuration, weights and the fixed mask construction are available. No
  earlier-layer KV state is needed: this is a full prefill suffix with separate
  per-layer caches, not continuation of generated tokens.
- `E_8=C` already exists in the first crossover and was reproduced in all
  17 stages by the downstream control. `E_7=B` exists as the accepted
  layer-7-on-CPU6 capture followed by the accepted GPU8/9 capture. Their complete
  intermediate S state was joined exactly. A and D are independently bound to
  the original traces.
- **Unseen endpoints are `E_0..E_6` and `E_9`: eight branches.** No new input
  image, ground truth or CPU trace is missing. Detailed norm/projector/gate
  substages for every original early GPU layer are not all saved; do not claim
  those as historical exact controls.

## Smallest useful fixed execution budget

Reusing the historical composed E7 requires eight new branches plus a complete
GPU-original control and the fresh 17-stage E8 bridge: **ten executions**.
I recommend prospectively fixing **eleven executions** instead, adding a direct
E7 replay to test the previously composed path in the new common runner:

1. From saved **GPU embedding**, execute GPU blocks 0..8 and native layer-9
   pre-QKV. Before substitutions, require all **47 available original tensors**
   exact: entry embedding, five captures per block (Q/K/V/attention/hidden),
   and layer-9 V. This covers all available original controls in this scope;
   it does not invent missing historical substages.
2. E8 from saved CPU7: require all **17** prior native stages exact.
3. E7 from saved CPU6: require the **30 distinct** joined layer-7 and downstream
   stages exact, including both the full C6 entry and S boundary. The old
   downstream input and old layer-7 hidden are one shared tensor in this count.
4. Only after all controls pass, execute the eight unseen branches once in the
   fixed order `E_0..E_6, E_9`. Stop on any failure, without retry or alternatives.

This totals **54 native full-block calls and 11 native layer-9 pre-QKV calls**,
with one resident model. The new branches need their complete entry and
endpoint saved; extensive extra hooks are unnecessary for block-level
localization. Do not crop rows, sample channels, batch branch states together,
or reuse a mutated cache. Each branch gets a fresh capacity-256 cache, identical
144-row mask/positions/strides and native uncompiled block calls. Stop before
layer-9 attention/FFN and before the remaining model or generation.

## Reuse and acceptance limits

Reuse the pinned setup/precision recording in
`experiments/fp32_crossover_downstream/export_gpu.py`, native calls and
pass-through hooks in `native_segment.py`, immutable joins in `evidence.py`,
and the FP64 summary conventions in `compare.py`. The existing source guard is
hard-coded to layer 8: a fresh guard must mechanically retain the setup and
native block expressions while checking the fixed boundary/call inventory.
Original modules/results stay unchanged. Capture and recheck source, inputs,
runtime, toolchain and outputs under the same isolated 4090/strict FP32 contract.

Report every endpoint and signed term at the original coordinate, complete
144-row and global summaries, cancellation and elementwise FP64 telescoping
residuals. Require the new controls to reproduce prior A/B/C exactly, including
the already measured `E_7-A = +0.00662994384765625`; keep the original absolute
bound `0.005096435546875` unchanged. Fractions can exceed one or be negative;
RMS/maxima do not add. A large term identifies a block boundary worth a later
same-input operator diagnosis, not a proven defective operator. Attention mixes
rows, so row 112 alone cannot establish the origin. Startup provenance gaps,
changed observer/allocation history and the limits of finite control inputs
remain. This is a design assessment alongside performance work, not model
qualification or authorization to execute a new experiment.

Sources inspected: the original exporter and pinned model implementation;
`reference/fp32-crossover-downstream-decomposition-v1.md` and its saved report;
the downstream native segment, guards, evidence loader, comparison and contract;
the original GPU/CPU metadata and frozen full-trace comparison inventory.
