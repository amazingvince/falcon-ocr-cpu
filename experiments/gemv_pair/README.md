# Paired-output AVX2 GEMV: source-only candidate

This one experiment extends the frozen combined expanded/compact fixed64
attention candidate. It changes only row-one AVX2/FMA linear calls at the five
actual Falcon `(input,output)` shapes: `(768,2048)`, `(1024,768)`, `(768,4608)`,
`(2304,768)` and `(768,65536)`. No live source, Cargo file, default, prior
experiment or benchmark artifact is edited.

The input is shared across adjacent output channels; each channel retains its
own four AVX2 accumulators, chronological FMA sequence and original reduction
tree. Runtime dispatch occurs before 32-output Rayon blocks. Each block calls
one feature-qualified direct function; the pair and reduction helpers are
inline candidates. There is no packing or allocation in this path. Actual
inlining, spills, call structure and performance require later compiled-code
inspection and measurement; source does not establish those outcomes.

`patch.py` performs no I/O on import. Its `make_patch(original, template, tests)`
returns `(patched_kernels, candidate_module, original_linear_function)`. Feed
it UTF-8 kernels from the pinned `benchmark-build/source.zip` under
`artifacts/diagnostics/attention64-compact-v1`, not the live source. The module
pins the baseline build/archive/kernel digests. It inserts one guarded
dispatch and appends a candidate plus a test-only verbatim original linear
oracle. All prior attention code stays byte-for-byte intact within the
original source prefix. The parent owns fresh-project capture/build staging.

## Authored validation, not yet executed

`test_source.py` has four source guards: archive/build identity, exact patch
scope and oracle, unsupported/mutated source rejection, dispatch/arithmetic
structure and the seven-test inventory. Run only after quiet-window release:

```
python -m unittest discover -s experiments/gemv_pair -p test_source.py -v
```

`tests.rs` contains six synthetic operator tests and one mandatory real-input
test; exact names and filter are exported as `TEST_NAMES` and `TEST_FILTER`
from `patch.py`. The synthetic checks cover every output of all five shapes
(including all vocabulary channels), independent cancellation/scale/zero
patterns, unaligned reads and block boundaries, Auto and one-thread dispatch,
fallback shapes/batches/backends, and shape-validation failures. They compare
raw FP32 bits against the unchanged current CPU implementation, not a widened
GPU tolerance. AVX2/FMA absence is an explicit failure, not a skipped test.

For the real test, the capture supplies `FOCR_GEMV_PAIR_ROOT` and checks all
four `INPUT_PINS` before and after execution. The test verifies the same pins
before borrowing read-only mmap tensor data, uses a four-thread pool, checks
exact shapes/dtypes, and compares every output for these fixed operands:

| Operation | Saved CPU input | Checkpoint weight |
|---|---|---|
| QKV | `cpu_state.layer.7.attention_norm`, row 112 | `layers.7.attention.wqkv.weight` |
| WO | `cpu_state.layer.7.attention`, row 112 | `layers.7.attention.wo.weight` |
| W13 | `cpu_state.layer.7.ffn_norm`, row 112 | `layers.7.feed_forward.w13.weight` |
| W2 | `cpu_state.layer.7.gate`, row 112 | `layers.7.feed_forward.w2.weight` |
| Vocabulary | `decode.1.layer.21.hidden`, original final RMSNorm | `output.weight` |

The first four inputs come from the accepted native layer7 crossover tensor
file (all 144 rows retained; this test prospectively uses row 112). The last
comes from the saved `e2dad…85309` CPU smoke trace. Its vocabulary input is
reconstructed by unchanged `rms_norm` using pinned `norm.weight` and epsilon
`1e-5`; no model pipeline executes. The full checkpoint remains mmap-backed,
without copying the vocabulary matrix. Required files and full digests are
literal constants in `patch.py` and `tests.rs`; missing names or wrong shapes
fail rather than substituting another stage. Each successful case prints
input, weight and complete candidate/original output hashes with compared
element counts. These fixtures validate operator equivalence, not quality or
performance selection.

After operator acceptance, use the existing 1904-tensor/17-token same-prefix
smoke and warmed allocation checks, and the parent's full-page exact-output
benchmark protocol. No model run, build, test, hash scan or timing has been
performed while authoring these files during the live quiet benchmark. The
ten existing GPU hidden-tensor failures remain separate and unresolved.
