# Mode fixtures

The first 300 tokens each mode generates on the journal page
(`artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png`, 6,544 input
tokens) under the automatic configuration, recorded with the CLI:

```sh
falcon-ocr --model artifacts/model --mode near-exact run <page> --max-new-tokens 300
falcon-ocr --model artifacts/model --mode fast run <page> --max-new-tokens 300
```

`token_ids` is the result's `token_ids`; `overlay_sha256` is the SHA-256 of
`artifacts/model/w8-gptq.safetensors` the fast run used (the GPTQ overlay
published with the model); `prefill` records the kernels the recording host
ran (near-exact `avx512-wide` attention with `panel-avx2` projections, fast
`bf16-static` with `panel-i16-vnni`). `tests/modes.rs` reproduces every
fixture from the checkpoint loader. The tokens hold only for those prefill
kernels, so the gate first compares this host's planned prefill with the
fixture's, by the kinds `--resume` compares (the AVX2 and AVX-512 FP32 tiles
count as one, and so do the VNNI and AVX2 INT16 panels), and fails with both
plans named when they differ, as on a CPU without AVX512-BF16 or on aarch64.
Regenerate a fixture only when a mode's numerics change on purpose, and say
so in the commit.
