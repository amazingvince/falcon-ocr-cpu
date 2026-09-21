# Pinned GPU reference

The reference environment runs in WSL on the physical RTX 4090 selected by UUID.
It is separate from the machine's training environments. WSL GPU traces are not
bare-metal Linux performance measurements.

From the project directory in `Ubuntu-24.04-CUDA`:

```bash
python3 scripts/fetch_reference.py
bash scripts/setup_reference.sh
bash scripts/run_reference.sh --max-new-tokens 24
```

`run_reference.sh` validates GPU identity, memory, Python/package pins, and all
downloaded model hashes before execution. It uses the isolated ext4 environment
at `/home/amazi/falcon-ocr-rust-reference`, configurable by `FOCR_REFERENCE_ENV`.
The memory gate defaults to 4096 MiB for the tiny fixture; set `FOCR_MIN_FREE_MIB`
appropriately for larger experiments. No server or other GPU workload is stopped.

The first run creates a 256x128 RGB text image and writes its pixels, metadata,
teacher tokens, and tensor trace under `artifacts/reference/smoke-fp32`. It calls
the pinned HF model's real forward functions and Triton FlexAttention, with block
compilation disabled to permit capture hooks. TF32 is disabled globally and
FlexAttention's independent `FLOAT32_PRECISION` option is set to quoted `ieee`.
The decode driver enforces the explicit limit instead of upstream's rounded cache
limit. Captured timings include copies and compilation and are not benchmarks.

Trace keys (all token-major; batch dimension removed):

| Key | Type and shape | Meaning |
|---|---|---|
| `tokens` | I64 `[S]` | Exact full-page prompt and image tokens |
| `patches` | F32 `[P,768]` | Valid patches, row-major, pixel/channel order |
| `pos_t` | I64 `[S]` | Temporal positions |
| `pos_hw` | F32 `[S,2]` | Spatial positions; NaN outside patch tokens |
| `embedding` | F32 `[S,768]` | Text embedding with projected image patches scattered |
| `layer.I.q`, `.k`, `.v` | F32 `[S,16,64]` | Q/K after RoPE; repeated V |
| `layer.I.attention` | F32 `[S,1024]` | Attention output after sink scaling, before output projection |
| `layer.I.hidden` | F32 `[S,768]` | Residual after feed-forward |
| `logits` | F32 `[65536]` | Last prefill position |
| `teacher_tokens` | I64 `[T]` | Generated token IDs including a stop token if emitted |
| `decode.N.*` | As above, singleton token dimension removed | Activations after consuming `teacher_tokens[N]` |

BF16 captures retain their dtype for model tensors; preprocessing stays FP32.
Decode `logits` remains a vector. All other singleton decode tensors lose the
batch axis only, retaining `[1,...]` shapes.

To compare GPU attention against an explicit dense FP32 operator while keeping
every decode prefix identical:

```bash
bash scripts/run_reference.sh --attention dense \
  --image artifacts/reference/smoke-fp32/input.png \
  --teacher-tokens artifacts/reference/smoke-fp32/teacher-tokens.json \
  --output artifacts/reference/smoke-dense-fp32 --max-new-tokens 24
```

Freeze tolerances from those GPU sources before comparing a Rust trace:

```bash
python scripts/compare_traces.py artifacts/reference/smoke-fp32/trace.safetensors \
  artifacts/reference/smoke-dense-fp32/trace.safetensors \
  --calibrate reference/tolerances-smoke-fp32-v1.json \
  --output reference/gpu-operator-smoke-fp32.json
```

The existing versioned tolerance file is immutable; calibration refuses to overwrite
it. Use `--tolerances reference/tolerances-smoke-fp32-v1.json` for a candidate
comparison. Prefill names with a `prefill.` prefix are normalized. Missing model
tensors, nonfinite differences, numerical threshold failures, and large-margin
argmax disagreement fail the command. The initial Windows report records
40 intermediate-tensor threshold failures despite all 17 output tokens matching;
separate-sink and pairwise-RMS candidates reduce that to 29 and 10 respectively.
These stage failures remain unresolved; matching output tokens alone is insufficient.

Run the independently pinned official plain engine with:

```bash
bash scripts/run_reference.sh --official
```

The pinned official engine passes the 17-token tiny-fixture check; see
`official-smoke-fp32.json`. Independent real-activation operator probes can be
regenerated with `--operators` (RMSNorm), `--attention-operators`,
`--rope-operators`, and `--linear-operators`.

This is a separate operator baseline, not a replacement claim for upstream GPU
parity. Dense attention materializes the score matrix and is only for small
validation fixtures. The full corpus, broader official-engine/vLLM checks,
long-context qualification, and measured CPU quality/performance gates remain
separate required stages.

The 24-page FP32 GPU smoke is complete; `gpu-corpus-smoke-fp32-summary.json`
records 28,734 output tokens, five pages above 2048 tokens, and a natural maximum
of 2567. Every page reached EOS within its explicit 4096 budget. See `corpus.md`
for the v1 split limitation, corrected v3 evaluation/calibration selection, and
the separate saved-token text replay used for the active full-corpus comparison.

The actual pinned direct-vLLM HTTP smoke also passes all 17 output IDs and exact
text with the same 144-token input. `vllm.md` describes runtime isolation and
compiled IEEE attention evidence; `vllm-smoke-fp32.json` is the durable result.

Actual BF16 GPU operator and full-graph probes are now captured. The eager HF
target converts all model parameters and the golden spatial frequency buffer to
BF16, then regenerates temporal complex64 frequencies. This differs from an
official-engine path that preserves FP32 golden frequencies. Torch 2.11 RMSNorm
uses FP32 opmath epsilon even with BF16 inputs. The gate rounds its square to
BF16 before multiplying; learned final normalization multiplies its weight in
FP32 before one final output round. See `bf16-local-contract-v1.json` for every
cast boundary and pinned reference.

The dense BF16 trajectory calibration is **rejected for qualification**:
`bf16-policy-review.json` records 47 stage tolerances at least as large as their
signal peak. Its immutable `tolerances-smoke-bf16-v1.json` is diagnostic evidence
and the comparator refuses to use it for acceptance. Flex rounds local
unnormalized exponential weights before P×V, which differs from rounding global
softmax probabilities. The independent blockwise oracle reproduces 99.9936% of
equal-input attention outputs exactly, but its few BF16 rounding differences
still amplify through the model. `bf16-blockwise-trajectory-review.json` therefore
also rejects a full hidden-trajectory gate. All 17 small-fixture output decisions
agree across these GPU variants.

The frozen local contract instead covers actual native BF16 linear outputs,
FP32 accumulation outputs, RMSNorm, gate casts, attention output/LSE, and
same-prefix output logits. Its per-element and RMS bounds use only GPU/F64/oracle
data, with no Rust candidate calibration. Run `compare_bf16_outputs.py` for the
separate output gate; candidates must prove identical teacher-forced inputs.
Passing this gate alone does not qualify a BF16 runner: local operator checks,
exact CPU-versus-BF16-GPU corpus outputs, the long boundary fixture, and separate
quality reporting remain required. Full hidden-trajectory parity is not claimed.

`bf16-local-contract-assessment.json` applies the frozen bounds to the existing
Rust scalar and AVX512-BF16 matrix reports: all 160 FP32-accumulation checks pass.
The nine RMSNorm/gate checks also pass, with eight exact and one single-element
one-ULP affine-normalization difference. Later native-output checks expose
failures: scalar linear passes all 80 direct/packed cases, while AVX512 packed
linear fails one element in one case (the runner uses the passing direct path).
Attention passes 13/18 AVX512 checks and 12/18 scalar checks. Both full-model
variants match all 17 argmax IDs, but each fails one frozen logit RMS gate.
`bf16-local-failure-details.json` preserves exact offending values. These are
failed gates; the BF16 runner remains experimental and unqualified.

The saved tile oracle is not a capture from inside fused Flex: it uses separate
Torch matrix/exp2 operations and a 16-key tail. The queued diagnostic described
by `bf16-fused-substage-export-spec-v2.json` preserves full matrices, padded
64-key iterations, and the fused dot accumulator. It also captures the actual
final numerator, denominator and quotient around the layer19 raw-rounding
boundary; the initial v1 design remains unchanged. Its observation-store kernel
must match the complete original raw BF16 output and FP32 LSE bit for bit before
any intermediate can be interpreted. Source-only preparation is complete;
the instrumented GPU run has not yet been executed.
