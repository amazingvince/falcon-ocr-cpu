A single logging replay is justified. It can establish the public BLAS API and may expose the selected algorithm configuration; it cannot promise to recover every internal scheduling choice.

The preserved [profiler report](C:/Users/amazi/Documents/ChatGPT/falcon-ocr/reference/linear-cuda-kernel-identities-fp32.json) has SHA-256 `0f1c9076febb406e1f8140ea4e06774e109d7edcc19f016bf2c155cb8fb4dd07`. All four referenced trace hashes were independently checked, and the report records exact saved-GPU output equality for all four calls. W2's prefill trace contains an explicitly named split-K reduction kernel after SGEMM; W13's contains no separate reduction launch. **Neither grid.z=14 nor grid.z=2 is a verified split-K parameter**, and W13's missing reduction launch does not prove that it uses no K partitioning.

The pinned Torch source routes `mm` through `addmm` with beta zero. This bypasses the bias-oriented Lt shortcut; ordinary FP32 GEMM uses `cublasSgemm` unless the preferred BLAS backend selects Lt. Therefore the kernel's `cublasLt` namespace does not establish a public `cublasLtMatmul` call. Actual API logging must decide this. [Pinned CUDA BLAS dispatch](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/Blas.cpp), [pinned GEMM backend](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cuda/CUDABlas.cpp).

The installed reference distribution is `nvidia-cublas==13.1.0.3`, confirmed with package metadata without loading CUDA. This is distinct from Torch's CUDA 13.0 build label. NVIDIA documents both logger families. Lt trace/API/info messages cover call parameters and potentially heuristic details; complete selected-algorithm fields are not guaranteed. The optional logger callback supplies text, not an algorithm descriptor. [cuBLAS 13.1 logging](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublaslt-logging).

`cublasLtMatmulAlgoConfigGetAttribute` can read ID, tile, stages, split-K count, reduction scheme, CTA swizzling and custom option from an initialized algorithm descriptor. It requires the **actual descriptor used by the call**. A newly queried heuristic, supported-capability list or constructed algorithm does not identify a prior Torch selection. A null descriptor permits internal heuristic selection; there is no documented last-Torch-call getter. [Descriptor API and attributes](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublasltmatmulalgoconfiggetattribute).

The minimal next diagnostic is one new process containing the existing prefill W13 and W2 operations, in their existing order, with three warmups and one profiled call each. Preserve the exact operands, allocation flow, FP32/TF32 settings, stream, workspace configuration and backend preferences. Add only output-directory handling and logging, with a fresh source copy. Before process startup, set:

```text
CUBLAS_LOGINFO_DBG=1
CUBLAS_LOGDEST_DBG=<fresh-absolute-directory>/cublas.log
CUBLASLT_LOG_LEVEL=5
CUBLASLT_LOG_MASK=31
CUBLASLT_LOG_FILE=<fresh-absolute-directory>/cublaslt_%i.log
```

These are documented logger controls; mask 31 enables all five message classes, and `%i` inserts the PID for the Lt log. [Logger configuration](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublasloggerconfigure).

Preserve complete logs, stdout/stderr, source and loaded-library identities, operand hashes and a fresh profiler trace. Require both outputs to match the frozen fixture bits and verify the same kernel names/grids as the original report before associating log fields with the original behavior. Join messages by process, ordered calls and actual API dimensions/transposes/leading dimensions; do not select an arbitrary heuristic line. Distinguish fields reported for the executed algorithm from rejected candidates.

If execution-linked configuration fields are absent, report that limitation and stop this diagnostic. Logging may still resolve the public API and workspace selection. Do not manufacture a descriptor from grid dimensions, force Lt, rerun heuristic search, decode an opaque structure privately or sweep chunk counts. Even a verified split count/reduction scheme would not specify exact K boundaries or FP32 accumulation order without further evidence. No GPU calls, installations, kernel changes or logging replay were performed for this investigation.
