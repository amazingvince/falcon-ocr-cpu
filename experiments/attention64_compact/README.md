# Combined expanded/compact fixed64 attention candidate

This new isolated copy retains the previously accepted expanded-cache AVX2
specialization and adds only compact-cache single-query/head64 AVX2 dispatch.
`patch.py` verifies the prior expanded experiment's source hashes, reuses its
unchanged fixed64 helpers, and extracts the original compact per-head body.
Only dot and AXPY call targets change inside that body. Prefix K keeps16heads,
generated K and all V keep8heads; GQA repeat, branching, tiles, image mask,
softmax, sink, division and multiplication order remain unchanged.

The frozen control archive is the starting project; only archived kernels.rs
differs. Eleven tests run: six prior expanded tests and five compact tests.
Compact outputs are compared bitwise both against the original compact path
and an independently expanded cache passed to the original expanded path.
Coverage includes zero/full prefixes,127/128/129 boundaries, actual6544 prefix
and16384 context, repeated KV heads, image boundaries, extremes and unaffected
shapes/backends/prefill. No arithmetic variant or production/default edit.

After operators, one native4-thread FP32/AVX2 compact smoke trace must equal the
saved expanded1904-tensor SHA e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309
and17 teacher IDs/text/length stop. Cache-layout metadata intentionally differs.
Full-page timing remains parent-owned and separately coordinated. The existing
ten GPU intermediate failures remain open; this candidate targets CPU parity.
