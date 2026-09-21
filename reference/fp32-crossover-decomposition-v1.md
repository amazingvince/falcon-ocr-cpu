# FP32 layer-8 state crossover

The first failing prefill coordinate is dominated by differences already present
at the end of layer 7. At layer-9 V coordinate `[112, 14, 2]`, **99.424% of the
signed difference** comes from passing the two different saved entry states
through the same GPU segment. The remaining 0.576% is the Rust-versus-GPU
difference when both start from the saved CPU entry state. These are conditional
crossover terms, not independent estimates of each operator's error.

Both original-state controls passed before this interpretation. The Rust replay
matches all six saved CPU tensors bit for bit; the native GPU replay matches
the corresponding six saved GPU tensors bit for bit. They are layer-8 Q, K, V,
attention and hidden state, plus layer-9 V. The GPU then executes the same
segment once on the saved CPU state. No arithmetic variant or model rerun was
used to make a control pass.

## Signed decomposition

Let `C` and `G` be the complete saved CPU and GPU layer-7 hidden states. The
three captured branches support this FP64 accounting identity:

`Rust(C) - GPU(G) = [Rust(C) - GPU(C)] + [GPU(C) - GPU(G)]`.

At the original worst layer-9 V coordinate:

| Quantity | Value |
|---|---:|
| Rust on CPU state | -19.03121566772461 |
| GPU on CPU state | -19.031253814697266 |
| GPU on GPU state | -19.037837982177734 |
| Original Rust minus GPU difference | +0.006622314453125 |
| Segment engine term, Rust(C) minus GPU(C) | +0.00003814697265625 |
| Incoming-state term, GPU(C) minus GPU(G) | +0.00658416748046875 |
| Unchanged absolute tolerance for layer-9 V | 0.005096435546875 |

The terms reinforce one another at this coordinate; there is no cancellation
there. The incoming-state term by itself exceeds the original bound. Replacing
this segment's Rust arithmetic with the measured GPU segment therefore does not
remove this failure on the original CPU entry state.

Across the complete layer-9 V tensor, the engine term's maximum absolute
difference is 0.00006103515625 and RMS is 0.0000056524125243. The incoming-state
term's maximum is 0.00658416748046875 and RMS is 0.00010795045085, versus original
RMS 0.00010810784356. These maxima may occur at different coordinates and must
not be added as a signed decomposition.

Row 112 shows the incoming difference growing through the preserved segment:

| Stage | Original difference, row RMS | Engine term, row RMS | Incoming-state term, row RMS |
|---|---:|---:|---:|
| Saved layer-7 hidden input | 6.72061e-4 | 0 | 6.72061e-4 |
| Layer-8 attention normalization | 4.88778e-5 | 0 | 4.88778e-5 |
| Layer-8 attention | 1.05414e-4 | 1.93236e-6 | 1.04436e-4 |
| Layer-8 gated activation | 2.66054e-4 | 4.08998e-6 | 2.64580e-4 |
| Layer-8 hidden output | 1.90602e-3 | 1.88330e-5 | 1.89704e-3 |
| Layer-9 attention normalization | 1.07837e-4 | 1.09143e-6 | 1.07386e-4 |
| Layer-9 V | 9.11850e-4 | 6.46685e-6 | 9.08117e-4 |

Different stages have different scales; this table does not rank operators by
accuracy. The complete report retains all 17 stages and all 144 rows, including
nonfailing rows. FP64 decomposition residual is exactly zero at every saved
element. That is an accounting result, not bitwise CPU/GPU model equivalence.

## Evidence and limits

- [Complete analysis](fp32-crossover-decomposition-v1.json), SHA256
  `b3e262ecc47713c356d3c8926fe1a642a610337eb077372705ff30510e067735`.
- [Preparation/build summary](fp32-crossover-preparation-build-v1.json).
- [Independent saved-artifact review](fp32-crossover-independent-review-v1.json),
  SHA256 `0bcc22437d8875f7c9f2feb44e31018061d78b61200db9b3a0718c7c2ddfae3b`,
  verifies 113 files, all 12 controls, three entry states, 51 stage payloads and
  every reported row statistic without repeating model execution.
- Plan: `artifacts/diagnostics/fp32-crossover-v1/plan.json`, SHA256
  `63890f3c3c2f229db6f3d20a4f3aba6b15bf9fa7c080b5005dcc0b6bad1e435c`.
- CPU execution: `artifacts/diagnostics/fp32-crossover-v1/execution.json`, SHA256
  `8e122d4df988da03bafca42278b6f443ff8d38c040b06af61a13b17732ce8f87`.
- GPU report: `artifacts/diagnostics/fp32-crossover-gpu-v1/report.json`, SHA256
  `72d577e05c7eb91d9d32890ef8e789fa988cdd80397a3e5b48cb4969a1e398ed`.
- Original numerical policy remains SHA256
  `8f6382a4386b15f76e6984679208a44d0fa644b9d55a21ce579bbb26e0bfbe1c`.

Rust uses the isolated Windows AVX2 build, four threads and expanded cache
capacity 161. GPU uses the UUID-isolated RTX 4090, pinned Torch 2.11.0+cu130,
FP32 with TF32/reduced-precision flags disabled, Flex IEEE precision, native
uncompiled blocks and cache capacity 256. Its recorded host settings are
OMP/MKL eight threads, Torch eight threads and 32 interop threads. This is
functional evidence; no timing comparison follows.

The current sources, inputs, binaries and payloads remain hash-bound through
capture and comparison. Historical original-trace startup provenance remains
incomplete. Skipping earlier execution changes allocator history, and hooks add
copies/synchronization; exact saved controls guard this bounded replay.

The later-stage engine term also includes different intermediate inputs that
arose within the segment; it is not a same-input isolated-operator bound.
This experiment does not identify which earlier operation created the incoming
state difference, establish causality for later failing tensors, or examine the
two decode-attention failures. It stops after this one decomposition. No
tolerance, production arithmetic, qualification status or benchmark default
changes: all ten original numerical failures remain open.
