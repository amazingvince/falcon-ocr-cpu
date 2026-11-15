# Research archive

Status: indexed on 2026-11-10. Nothing here is compiled, and CI runs only the Python tests in `corpus-qualification/tests` and `quantization-feasibility/tests`; it is the record of how the runner got where it is. Receipts and pinned hashes stay in `reference/`; the frozen GPU-reference scripts stay in `scripts/`.

| Folder | About |
|---|---|
| [aocl/](aocl/README.md) | AOCL/BLIS and matrix-backend experiments (source only, not compiled). |
| [benchmarks/](benchmarks/README.md) | Benchmark harnesses and the performance evidence before the phase-4 work. |
| [bf16-graph/](bf16-graph/README.md) | The retired experimental BF16 execution graph (source only, not compiled). |
| [corpus-qualification/](corpus-qualification/README.md) | Building and qualifying the evaluation corpus, the 200-page quality gate and its Python tests. |
| [decode-attention-sparsity/](decode-attention-sparsity/README.md) | A GPU study of how much softmax mass decode attention puts on each 32-position image block, and whether a decode kernel could skip blocks: the exact-mass oracle against cheap per-block bounds and the Q8 KV cache's error (not worth building). |
| [draft-head/](draft-head/README.md) | The trained speculative draft head (EAGLE-3 style): data pipeline, training, export, CPU A/Bs and lessons. Active. |
| [gpu-reference/](gpu-reference/README.md) | The strict FP32 GPU export, reference capture and comparison protocol whose receipts live in `reference/`. |
| [kv-layer-precision/](kv-layer-precision/README.md) | Per-layer key and value KV precision sensitivity on ten pages, the whole-layer Q8 set behind `--tune kv-layers`, a rejected mixed Q8/Q16 store, and report-only fast q4 sensitivity. |
| [kv-rotation/](kv-rotation/README.md) | A static study of the rotated KV caches (`q8r`, `q4r`) on the real weights with synthetic activations: block and attention-output errors of Q8, Q8R, Q4 and Q4R per layer. Active; the real-page token agreement is in `docs/MODES.md`. |
| [phase4-hillclimb/](phase4-hillclimb/README.md) | The phase-4/5 hill climb of 2026-09-23: GPTQ overlay, near-exact and fast modes, speculation, packed files; every accepted and rejected attempt with its receipt. |
| [quantization-feasibility/](quantization-feasibility/README.md) | Early quantized-operator feasibility experiments (source only, not compiled). |
| [resolution-router/](resolution-router/README.md) | The per-page image-size router: agreement labels, serving noise, the ground-truth development set and the chosen tree policy. Active. |
| [reviews-2026-09-20/](reviews-2026-09-20/README.md) | Code and math reviews of 2026-09-20 that shaped the phase-4 plan. |
| [vllm-serving/](vllm-serving/README.md) | Production-style BF16 serving with vLLM and the held-out comparisons made through it. |
