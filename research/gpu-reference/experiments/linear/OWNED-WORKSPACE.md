This is a prospective observation of one pinned W2 projection through the supported public cuBLAS SGEMM API. It is not a CPU implementation, numerical fix, performance benchmark, or model qualification. No GPU run is authorized by this document; the reference owner schedules it only after the official/vLLM server has stopped.

The exporter accepts a new output directory and refuses overwrite:

```text
<pinned-reference-python> research/gpu-reference/experiments/linear/export_owned_workspace.py --output artifacts/reference/linear-owned-workspace-fp32-v1
```

The existing UUID isolation, reference environment, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and project preflight still apply. Each child process receives both supported logging configurations before loading CUDA. The parent does not import Torch. All child processes finish, and their logs are read and gated, before starting another child. A failed child/control gate prevents every subsequent basis probe. Source/hash, library, operand, output, initialized owned workspace, trace and logger evidence is preserved.

The real control loads only `prefill.layer.12.w2` from the fixed linear fixture, whose entire file and metadata are pinned. It creates a fresh handle and caller-owned aligned 32 MiB uint8 tensor, fills every byte with 0xA5, uses NULL stream and DEFAULT_MATH, and verifies the new handle's HOST pointer mode. The call is SGEMM(T,N,768,144,2304), lda=ldb=2304, ldc=768, alpha=1, beta=0. No existing Torch handle, cuBLAS-internal allocation, private entrypoint, algorithm descriptor, or intermediate kernel memory is accessed. Three warmups precede one profiled capture. Scratch is copied only after complete synchronization and before further BLAS calls or reuse. The complete allocation is retained as opaque bytes, including unused initialized regions.

The finalized control must match the original FP32 output bits, public arguments, actually logged algoId=0/tile128x64/COMPUTE_TYPE/splitsK14/selected6193152B, provided32MiB and alignment classes, ordered kernel names/grids/blocks, loaded library hashes and pinned environment. Every basis call has the same gates and an independent exact analytic output. A profiler failing to capture the foreign direct call is a failure; missing evidence is not inferred from expected names.

The initial unit one-hot proposal was prospectively refined before any execution to **coded one-sparse** weights, because row-only labels leave output-channel permutations invisible. There are exactly three predetermined probes. Group g=0,1,2 sets W[r,768g+r]=r+1 for output channel r=0..767; all other weights are zero. X[j,k]=j+1 for all 144 input rows and all 2304 K coordinates. Thus each output is exactly (j+1)*(r+1), with maximum 110592 below 2^24. Every channel within a token row is distinct, and all 2304 K indices are exercised once without changing matrix shapes or guessing chunk boundaries.

The single explicit interpretation hypothesis is: workspace byte offset zero begins 14 consecutive little-endian FP32 partial matrices, each in [144,768] token-row/output-channel order. All 14×144×768 elements in all three captures must be either exact positive zero or the expected product code, with exactly one code per output and the same partial slot across all token rows for each K/channel. No offset, permutation, split-count, or reduction-order search is performed. Only after all three groups agree is conditional K membership emitted. Noncontiguous or empty memberships are recorded directly. A failed hypothesis retains raw observations and reports `captured_layout_not_established`.

Membership remains explicitly conditional observational evidence for this layout and these inputs. The codes distinguish channels within each token row, although different row/channel pairs can have equal products; this is not a mathematical uniqueness proof over arbitrary permutations of all entries or a public workspace ABI. Even a complete match does not reveal within-partition FMA ordering or the final reduction sequence. No production change is warranted solely by this observation.

Host-only checks:

```text
python -m unittest discover -s experiments/linear -p test_workspace_observation.py -v
```

They validate all-K coverage/exact analytic outputs, original log parsing and mutated settings, whole-basis agreement including a late-group failure, duplicate partials, row disagreement, noncontiguous membership, truncated captures and hash mutation. Synthetic small layouts keep these tests independent of GPU work.
