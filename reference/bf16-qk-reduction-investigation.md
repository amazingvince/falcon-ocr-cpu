# BF16 QK reduction investigation

The bounded experiment did **not** identify a replacement for the current QK kernel. None of the instruction- or fragment-based candidates improved its 13/18 frozen attention gates. A whole-dot FP64 accuracy control passed 14/18 but still violated 11 element bounds. Production kernels, model execution and acceptance bounds remain unchanged.

## Structural evidence

Three existing Triton cache artifacts target the pinned RTX4090 architecture, `sm_89`, and use `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. The observed 64-wide QK dot carries the same accumulator through four K16 instructions. The [PTX inventory](bf16-mma-ptx-inventory.json) records hashes, retained PTX copies, instruction lines and accumulator chains. Inspection was read-only and did not execute GPU work.

This matches the BF16 opcode and increasing-K accumulator loop in [Triton 3.6.0's MMA lowering](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp). NVIDIA documents paired BF16 fragment elements across four lanes, but leaves internal accumulation order and rounding unspecified. Fragment ownership therefore motivates a hypothesis; it does not prove hardware summation order. See the [fragment layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float) and [MMA precision rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma). No dense BF16 K32 opcode appeared in these artifacts, so a K32 hypothesis was not added.

## Complete fixture results

Eight fixed candidates were declared before measurement. Only QK reduction changed. P×V retained the current AVX512BF16 dot; tile order, split scheduling, exponential, denominator, casts and sink scaling stayed fixed. Every candidate evaluated all six isolated operators: four complete prefill tensors and two decode tensors, each checked for raw output, scaled output and LSE. Diagnostic baseline outputs were bit-identical to production for every element, including LSE.

| QK reduction | Gates passed /18 | Element-bound violations | Exact GPU tile QK /720 |
|---|---:|---:|---:|
| Current AVX512BF16 | 13 | 11 | 267 |
| Sequential FP32 control | 12 | 24 | 313 |
| K16 sequential panels | 13 | 19 | 327 |
| K16 adjacent-pair trees | 13 | 19 | 342 |
| K16 fragment-pair trees | 13 | 21 | 339 |
| K16 fused FP64 panel hypothesis | 13 | 18 | 349 |
| K8 fused FP64 half-panel hypothesis | 12 | 11 | 367 |
| Whole64 FP64 accuracy control | 14 | 11 | 303 |

The QK equality column covers all 720 valid scores in the five previously exported GPU tile probes, including a control head. It is supplementary: candidate ranking uses entire attention tensors, not selected failing coordinates. More exact QK scores did not predict better attention acceptance. Small score or maximum changes can still flip local BF16 probability rounding near midpoints, which matters when P×V cancels strongly.

[Full Rust results](bf16-qk-reduction-candidates.json) contain every named gate, RMS/max errors, violation indices, fixed candidate rationales, source hashes and frozen input hashes. The unchanged GPU-team Python comparator independently reproduced every pass/fail, maximum error, differing-element count and bound-violation count: [cross-check](bf16-qk-reduction-independent-check.json). All candidates also passed complete K-lane coverage and finite-operand forward-error tests. No performance measurements were collected.

## Reusable diagnostic and limit

`examples/bf16_qk_reduction_probe.rs` uses isolated support modules; it is not linked into runner dispatch. Its retained all-candidate tensor export is `artifacts/cpu/bf16-qk-reduction-candidates.safetensors`.

```powershell
& scripts/build_windows.ps1 -CargoArguments @('run','--release','--example','bf16_qk_reduction_probe','--','--output','reference/bf16-qk-reduction-candidates.json','--export','artifacts/cpu/bf16-qk-reduction-candidates.safetensors','--threads','2')
```

These results reject the tested reduction hypotheses as qualification fixes. They neither prove that CPU emulation is impossible nor identify NVIDIA's hidden internal reduction. Further work would require a controlled, held-out synthetic MMA experiment and actual hardware instruction analysis; trying more fixture-driven permutations would provide weak evidence. The FP64 control is not promoted, and the BF16 model remains experimental.
