# Block0 crossover on complete saved embeddings

This is a prospective diagnostic for the next question in
`experiments/fp32_prefix_suffix_telescope/RESULTS-V1.md`. It preserves that
telescope and every earlier experiment. It neither repairs the ten frozen
intermediate failures nor changes their policy.

The fixed budget is **three native block0 evaluations**: Rust(C0), GPU(G0),
then GPU(C0), where G0 and C0 are the original complete `[144,768]` GPU and CPU
embedding tensors. No embedding projector, later block, LM head, decoding,
suffix or arithmetic variant is executed. Rust(C0) must reproduce the five
original CPU Q/K/V/attention/hidden controls. The GPU exporter requires its
accepted, hash-bound execution/build/report/tensor/binary receipt before
importing the CUDA preflight. GPU(G0) must then reproduce the five original
GPU controls before GPU(C0) runs. A control failure ends this version; no retry
or alternate control is authorized.

Each arm captures fourteen complete FP32 arrays: entry, attention RMS output,
QKV projection, Q/K after RoPE, expanded V, sink-scaled attention, WO,
attention residual, FFN RMS output, W13, squared-ReLU gate, W2, final hidden.
The native GPU fragment is text-checked against the preserved layer7 exporter,
allowing only block selection/labels and a passive entry-bit check. Its native
functions and pass-through hooks remain intact. Mechanical removal of the
Rust observations restores the pinned forward body exactly, reversing only
the selected one-block loop and entry label. No live Rust source is edited.

GPU uses the original 256-capacity cache, full 144 rows, original token mask,
temporal/spatial positions, FP32/IEEE and TF32-off environment on the isolated
4090 UUID. Rust uses the original 161-capacity expanded cache, AVX2/four
threads and the original Rust **1.92.0** toolchain. The preserved toolchain
file comes from the accepted layer7 copied project, checked against its
plan; the live 1.94 pin is deliberately outside this diagnostic's build
selection. `RUSTUP_TOOLCHAIN=1.92.0` overrides the build wrapper's working
directory, and actual compiler version/tool identity is recorded. A fresh
isolated target builds with two jobs. Missing old tools or changed numerical
sources cause failure rather than an implicit compiler upgrade.

The offline comparison independently rechecks all ten original controls,
all three full entries and all 42 payloads. For every stage and each of all
144 rows, it reports FP64 signed accounting:

`Rust(C0) - GPU(G0) = [Rust(C0) - GPU(C0)] + [GPU(C0) - GPU(G0)]`.

It identifies the first captured stage with different same-entry bits and
reports global/per-row maxima, RMS, signed means, worst coordinates and
cancellation. Q/K/V are sibling paths, not a total causal sequence. Later
same-entry engine terms include differences accumulated inside this block.
Attention couples all rows; the downstream V9 coordinate does not select a
particular block0 source row/channel. No arbitrary hidden coordinate is
substituted for it.

Pinned telescope evidence establishes the existing conditional downstream
effect `E1-E0 = +0.0044460296630859375` at V9 `[112,14,2]`. That endpoint is
**not rerun here**. A first local difference or largest local norm does not
prove which operator causes that propagated effect. A later equal-input
operator investigation would require a separate bounded decision.

Historical GPU/CPU startup provenance gaps remain. Skipping the original
embedding path changes allocation history; hooks add copies and
synchronization. Exact controls qualify this particular observed segment,
not unobserved arithmetic, a full model, accuracy or performance. Existing
source/results are never overwritten. Each command is a separate phase;
future runtime work requires coordinator release and an idle timing window.

## Host guards (no model or tensor reads)

```text
C:/Users/amazi/mambaforge/python.exe -m unittest discover -s experiments/fp32_crossover_block0 -p test_source.py -v
```

## Future native preparation/build/control

```text
python experiments/fp32_crossover_block0/capture.py --quiet-window-released prepare --output artifacts/diagnostics/fp32-crossover-block0-v1
python experiments/fp32_crossover_block0/capture.py --quiet-window-released build --prepared artifacts/diagnostics/fp32-crossover-block0-v1 --preparation-sha256 <preparation-sha256> --target-dir D:/falcon-ocr-rust-builds/fp32-crossover-block0-v1
python experiments/fp32_crossover_block0/capture.py --quiet-window-released run --prepared artifacts/diagnostics/fp32-crossover-block0-v1 --preparation-sha256 <preparation-sha256> --build-sha256 <build-sha256>
```

## Future GPU arm and offline accounting

Launch the following exporter through the pinned reference WSL environment
and project preflight, with `CUDA_VISIBLE_DEVICES=GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`. The 5090 is excluded.

```text
python experiments/fp32_crossover_block0/export_gpu.py --execute-reviewed-plan --plan artifacts/diagnostics/fp32-crossover-block0-v1/plan.json --plan-sha256 <plan-sha256> --cpu-execution artifacts/diagnostics/fp32-crossover-block0-v1/execution.json --cpu-execution-sha256 <execution-sha256> --output artifacts/diagnostics/fp32-crossover-block0-gpu-v1
python experiments/fp32_crossover_block0/compare.py --execute-reviewed-plan --plan artifacts/diagnostics/fp32-crossover-block0-v1/plan.json --plan-sha256 <plan-sha256> --cpu-execution artifacts/diagnostics/fp32-crossover-block0-v1/execution.json --cpu-execution-sha256 <execution-sha256> --gpu-report artifacts/diagnostics/fp32-crossover-block0-gpu-v1/report.json --gpu-report-sha256 <gpu-report-sha256> --output reference/fp32-crossover-block0-decomposition-v1.json
```

Preparation binds current source bytes plus the reused original observer
pins and evidence. It is intentionally not run by source preparation. Once
a plan exists these sources remain fixed; later review/results belong in
new files outside the prepared source inventory.
