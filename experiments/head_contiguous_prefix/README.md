# Head-contiguous prefix K/V candidate

One isolated copied-project layout experiment on the frozen combined-attention
baseline. No live source, default, precision, dependency or numerical-policy
change. There is no speed claim before qualification and controlled measurement.

The explicit `HeadContiguousPrefix` layout (`head-contiguous-prefix` CLI,
`head_contiguous_prefix` JSON) stores prefix K as[16,P,64], prefix V as[8,P,64],
and generated K/V as[G,8,64]. All16 prefix K heads survive spatial RoPE. V and
generated K select the same first head of the existing GQA pairs after validating
duplicated raw bits. Validation precedes writes; complete prefix followed by
single-token contiguous continuation is the only supported append sequence.

The one-query decode adapter copies the baseline generic and fixed64 bodies,
changing only the K/V address calculations. Dot/AXPY, max/exp/denominator/sink
operations, their order, masks and global128-key tiles stay intact. Prefix6544
crosses the tile beginning6528; it does not reset the tile or softmax state.
Scalar and AVX512 retain their original per-backend kernels. Existing Compact
and Expanded paths remain separately selectable and unchanged.

Prefill still calls the baseline `attention_compact_with_simd` with borrowed
expanded workspace K and token-major compact V. A candidate-only local Vec
reservesP*512 floats once per prefill, is refilled without allocation per layer,
and drops on return. Packing, allocation, copy and release costs all occur during
page execution. Decode creates no allocated scratch. AtP6544 the temporary
logical payload is12.78125MiB; persistent payload equals Compact, not the earlier
temporal candidate. Vec capacity payload and process memory remain distinct.

`patch.py:make_sources(dict[str,bytes])` verifies pinned members from source.zip,
returns all original members and exactly this runtime delta:

- Changed: `src/config.rs`, `src/lib.rs`, `src/model.rs`, `src/kernels.rs`.
- Added: `src/head_contiguous_prefix.rs`,
  `src/head_contiguous_prefix_model_tests.rs`.

`candidate.rs` is the AVX2 module template; fixed64 helpers/body are extracted
from the frozen source, not reimplemented. `head_contiguous_prefix.rs` owns the
storage. Independently authored `storage_tests.rs`, `model_tests.rs` and
`test_source.py` cover physical reconstruction, unchanged prefill and decode,
backend/mask/tile/capacity boundaries, source guards and explicit/default layout
selection. Parent-owned capture and model qualification remain separate and
must preserve failed attempts. No prepare/build/model/benchmark is performed by
the patch helper or its import. See the prospective design in
`experiments/profiling/NEXT-HEAD-CONTIGUOUS-V1.md` for the bounded qualification
and fair timing protocol.
