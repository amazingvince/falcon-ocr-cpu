# Research archive

Status: indexed on 2026-09-24. Nothing here is built or tested by CI; it is the record of how the runner got where it is. Each folder's README lists its files, what they produced and whether the result was accepted, rejected or superseded. Receipts and pinned hashes stay in `reference/`; the frozen GPU-reference scripts stay in `scripts/`.

| Folder | Files | About |
|---|---:|---|
| [aocl/](aocl/README.md) | 8 | AOCL/BLIS and matrix-backend experiments (source only, not compiled). |
| [benchmarks/](benchmarks/README.md) | 177 | Benchmark harnesses and the performance evidence before the phase-4 work. |
| [bf16-graph/](bf16-graph/README.md) | 40 | The retired experimental BF16 execution graph (source only, not compiled). |
| [corpus-qualification/](corpus-qualification/README.md) | 46 | Building and qualifying the evaluation corpus, the 200-page quality gate and its Python tests. |
| [gpu-reference/](gpu-reference/README.md) | 91 | The strict FP32 GPU export, reference capture and comparison protocol whose receipts live in `reference/`. |
| [phase4-hillclimb/](phase4-hillclimb/README.md) | 29 | The phase-4/5 hill climb of 2026-09-23: GPTQ overlay, near-exact and fast modes, speculation, packed files; every accepted and rejected attempt with its receipt. |
| [quantization-feasibility/](quantization-feasibility/README.md) | 22 | Early quantized-operator feasibility experiments (source only, not compiled). |
| [reviews-2026-09-20/](reviews-2026-09-20/README.md) | 6 | Code and math reviews of 2026-09-20 that shaped the phase-4 plan. |
| [vllm-serving/](vllm-serving/README.md) | 16 | Production-style BF16 serving with vLLM and the held-out comparisons made through it. |
