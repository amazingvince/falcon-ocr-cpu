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
ran. `tests/modes.rs` reproduces every fixture from the checkpoint loader.
Regenerate a fixture only when a mode's numerics change on purpose, and say
so in the commit.
