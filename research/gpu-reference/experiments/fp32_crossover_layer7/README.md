# One native layer-7 FP32 state crossover

This version preserves the earlier layer-8/9 experiment unchanged. It implements
only the next diagnostic frozen in `reference/fp32-upstream-next-diagnostic-v1.md`.
Both original payload pins, policy, source pins, checkpoint and environment stay
fixed. Entry is the complete saved layer-6 hidden state `[144,768]`; endpoint is
layer-7 hidden. Row 112/channel 249 is selected before execution.

The three branches are Rust(CPU entry), GPU(GPU entry), and GPU(CPU entry).
Rust runs first. Its five saved Q/K/V/attention/hidden controls must match every
bit before any GPU execution. GPU runs its original-state branch first and
requires the same five exact controls before the crossed-state branch. The
offline comparer independently rechecks all ten controls and three input states.

The copied test-only Rust adapter selects only layer 7 and adds passive buffer
copies. Its mechanical removal guard restores the original numerical body text
exactly, apart from the documented bounded loop selection and input label.
Live Rust sources are unchanged. GPU uses the native uncompiled block and native
operators through pass-through hooks; there is no recomputed norm, gate,
attention, projection or residual. All 144 rows, original positions/mask, GPU
cache 256 and Rust expanded cache 161 remain. GPU runs strict FP32/IEEE/TF32-off
on isolated RTX4090 UUID; Rust uses four AVX2 threads. Builds use two jobs in a
fresh dedicated target. Added observations change allocation/synchronization;
mandatory original controls establish only the captured endpoint equivalence.

All 14 stage tensors are fixed in contract.py. The FP64 accounting identity is:
`Rust(C)-GPU(G) = [Rust(C)-GPU(C)] + [GPU(C)-GPU(G)]`.
It reports every row, complete tensors and the fixed endpoint, retaining signed
terms and cancellation. Later engine terms include different internal inputs;
they are not isolated operator errors. Historical source/build startup gaps
remain. No tolerance change, production change, performance claim or model
qualification follows. Stop after a control failure or one valid decomposition.

Commands run only after source review and root's release:

```text
python -m unittest discover -s experiments/fp32_crossover_layer7 -p test_source.py -v
python research/gpu-reference/experiments/fp32_crossover_layer7/capture.py --quiet-window-released prepare --output artifacts/diagnostics/fp32-crossover-layer7-v1
python research/gpu-reference/experiments/fp32_crossover_layer7/capture.py --quiet-window-released build --prepared artifacts/diagnostics/fp32-crossover-layer7-v1 --preparation-sha256 <hash> --target-dir D:/falcon-ocr-rust-builds/fp32-crossover-layer7-v1
python research/gpu-reference/experiments/fp32_crossover_layer7/capture.py --quiet-window-released run --prepared artifacts/diagnostics/fp32-crossover-layer7-v1 --preparation-sha256 <hash> --build-sha256 <hash>
```

The GPU arm is separately launched via its project preflight with `--plan`,
`--plan-sha256`, fresh `--output` and `--execute-reviewed-plan`. The offline
comparer additionally binds CPU execution and GPU report hashes. No phase
implicitly retries or starts the next. Old source/results are never overwritten.
