# One FP32 frozen-state crossover

These are prospective diagnostic sources. At initial preparation, the quiet
full-page benchmark is still running. No test, build, model segment, tensor load,
large hash/copy operation, or GPU work has been performed for this experiment.
Independent source review and root's explicit quiet-window release are required
before the commands below. A flag documents that release; it does not grant it.

The fixed inputs are the original GPU trace (`30dca24d…`) and Rust trace
(`e2dad223…`), both containing the complete 144×768 state after layer 7. The
original frozen numerical policy stays unchanged. Historical source/build
startup attestation is incomplete; current source hashes cannot repair it.

The Rust arm is a test-only copied project. `adapter.py` extracts the original
forward body, retains its numerical operation text, adds passive FP32 buffer
copies, selects layer 8, and stops layer 9 after QKV expansion. The copied live
model file gets only a test-module declaration. No earlier layers, image
projection, generation, final logits, replacement arithmetic, or intervention
variants execute. Rust retains original expanded cache capacity 161 and four
AVX2 threads; GPU retains original mask/cache capacity 256, native uncompiled
blocks, FP32, TF32 disabled, and Flex IEEE FP32.

The GPU owner maintains `export_gpu.py`. It runs native layer 8 on saved GPU
state first, then on saved CPU state only after its original control passes.
Both platform controls require exact saved Q/K/V, attention, hidden for layer 8
and V for layer 9. Missing or ambiguous `name` versus `prefill.name` keys reject
the capture. Added hooks, copies, and allocator history differences remain
observable changes; exact controls are required rather than assumed.

The common plan freezes 17 complete FP32 tensors per branch, including the
input, intermediate normalizations/projections/residuals/gate, and layer 9 V.
GPU has `gpu_state` and `cpu_state`; Rust has `cpu_state`. Shape/dtype/raw hashes,
full original input bits and controls are checked again by `compare.py`.

After review/release, preparation, build and CPU execution are separate steps:

```text
python research/gpu-reference/experiments/fp32_crossover/capture.py --quiet-window-released prepare --output artifacts/diagnostics/fp32-crossover-v1
python research/gpu-reference/experiments/fp32_crossover/capture.py --quiet-window-released build --prepared artifacts/diagnostics/fp32-crossover-v1 --preparation-sha256 <reviewed-preparation-hash> --target-dir D:/falcon-ocr-rust-builds/fp32-crossover-v1
python research/gpu-reference/experiments/fp32_crossover/capture.py --quiet-window-released run --prepared artifacts/diagnostics/fp32-crossover-v1 --preparation-sha256 <reviewed-preparation-hash> --build-sha256 <reviewed-build-hash>
```

Use a fresh dedicated target directory: two copied projects must never share a
target cache and silently reuse each other's executable. The capture binds the
Cargo-emitted manifest/source/test executable, preserves copied/original source
bytes and logs, uses offline locked dependencies and two compiler jobs, and
requires the selected ignored test to run exactly once. CUDA visibility is
disabled for the Rust child. Failures retain artifacts and are not retried by
this driver. Build target placement and disk space are root's execution choice.

The source-only tests may be run after the quiet window, with no model work:

```text
python -m unittest discover -s experiments/fp32_crossover -p test_source.py -v
```

The GPU owner separately schedules `export_gpu.py --plan <plan> --plan-sha256
<reviewed-hash> --output <fresh-project-directory> --execute-reviewed-plan`
through the project's pinned preflight and UUID isolation. Do not invoke GPU
work from the Rust capture. The export records effective precision/runtime flags
and source/input closure.

Finally, `compare.py` takes explicit plan, CPU execution and GPU report paths
and SHA256s, a fresh report path, and `--execute-reviewed-plan`. It rechecks full
payload controls and provenance before calculating in FP64:

`Rust(C) - GPU(G) = [Rust(C) - GPU(C)] + [GPU(C) - GPU(G)]`.

Every stage and every row is reported, including row 112 and the actual worst
coordinate. The engine term at later stages includes different intermediate
inputs within the segment; it is not an isolated operator error bound. FP64
accounting residual and cancellation are descriptive evidence. No new tolerance
is selected, no existing gate is relaxed, and no performance or qualification
claim follows. Stop after one valid decomposition or a failed original control.
The two decode attention failures are outside this experiment.
