# Isolated W4A32 / W8A32 arithmetic experiment

These files are excluded from the production runner. `examples/quant_probe.rs`
includes the scalar research modules directly; no Cargo dependency or production
precision/backend is changed. No timing measurements, model integration, or
quantized OCR qualification are performed.

## Frozen arithmetic contract

- Weights are row-major `[out_dim, in_dim]`; activations remain FP32. Groups are
  contiguous along K, independently in each output row. This operator experiment
  evaluates **both** group sizes 64 and 128, without selecting one from its results.
- Codes are signed `[-7,7]` for W4 or `[-127,127]` for W8. The reserved minimum
  values -8/-128 are unused. W4 keeps the settled adjacent low/high nibble layout,
  with zero padding in the final high nibble of an odd row. W8 uses ordinary i8
  row-major bytes, without activation quantization or VNNI arithmetic.
- Each FP32 group scale is `max(abs(weight))/qmax`, divided in FP64 then rounded
  to FP32. Nonzero groups clamp the scale to the smallest positive FP32 subnormal;
  all-zero groups have zero scale/codes. Each code is FP64 `weight/scale`, rounded
  ties-to-even and clipped to the signed range. FP32 dequantization is a separately
  rounded multiplication of the exact integer code and FP32 scale.
- A scalar output uses one FP32 FMA chain in ascending K, beginning at +0.
  The mathematical comparison uses the *already FP32 dequantized* weights in
  FP64. Weight quantization error and accumulator rounding error are reported
  separately. A separate AVX2 decode candidate is described below; no speed
  comparison exists here.
- Nonfinite weights and dequantization overflow are rejected. W8's operator also
  rejects nonfinite activations/accumulation and shape overflow. Tensor payload
  byte counts exclude metadata headers, allocation, alignment, scratch, original
  weights that coexist during the probe, and any future hardware packing.

## Fixed diagnostic inputs and sampling

The existing `artifacts/reference/layer-operators-fp32.safetensors` fixture has
five prefill cases (layers 9/12/13/17/18) and two decode cases (step2/layer21 and
step6/layer19). Its SHA256 is
`ce7345c219d8923182ff66e9aad2f4d6bad3c193a19c3445a850c2d3a90e5417`.
These are old synthetic diagnostic activations, originally selected to locate
FP32 drift. They are **not** calibration or held-out quality data. No group,
layer policy, clipping threshold, or quality budget may be selected from them.

All four original checkpoint matrices are inspected for every case: QKV consumes
saved attention RMS output; WO consumes saved scaled attention; W13 consumes
saved FFN RMS output; W2 consumes saved squared-ReLU gate output. Thus all 28
linear operators receive identical saved GPU inputs, without propagating one
quantized operator into another. Every value in each matrix is quantized and
reconstructed for all four format/group combinations (112 reconstructions).

Scalar output calculations use rows `sorted(unique(0,1,R/4,R/2,3R/4,R-2,R-1))`,
clamped at boundaries. Output channels include first/last four and even/odd pairs
at each quartile. QKV also includes both sides of the Q/K/V boundaries at 1024
and 1536. All indices are saved explicitly, and flattened outputs follow sampled
row then sampled channel order. Interleaved W13 gate/up pairs stay adjacent.
Selection is geometric and predetermined, never based on observed errors.

## Independent checks and preserved evidence

`check_probe.py` reads safetensors using an independent Python header parser and
NumPy memory maps. It reconstructs all quantized code, scale and dequantized
matrix bytes independently and compares SHA256 exactly. `math.fsum` supplies
FP64 original/dequantized dot oracles for every sampled element, using a different
reduction from the Rust ascending FP64 loop. FP32 FMA rounding is checked against
the standard `gamma_K * sum(abs(products))` bound, where
`gamma_K = K*u/(1-K*u)`, `u=2^-24`. This bound validates accumulator arithmetic;
it is not an OCR or GPU-parity acceptance tolerance. Full reconstruction-statistic
comparisons allow tiny FP64 summation-order differences while all format bytes
must match exactly.

`capture_probe.py` preserves the exact Cargo-emitted executable, before/after
source inventory, source ZIP, tool versions, build flags and build log. The probe
hashes the complete original checkpoint and fixture, sidecar, original matrix
bytes, input/output fixture tensors and its compiled sources. It refuses to
overwrite reports. The checkpoint is rehashed after computation, and the checker
rechecks source artifact identities before and after its pass. The artifact
snapshot is not a hermetic compiler/native dependency environment.

The current probe pins the sidecar SHA256 before parsing case metadata, so a
changed layer/weight mapping cannot be accepted merely because it names the same
tensor fixture. The checker independently pins the checkpoint, tensor fixture,
and sidecar; it also requires a completed, unchanged-source `quant_probe` build,
the preserved executable, every entry in the source ZIP, and all embedded source
hashes. It reads `build.json` beside the candidate report by default, or accepts
an explicit `--build-manifest`. The checker used for a later validation is recorded
separately from the historical checker preserved in the original build.

V1 artifacts remain unchanged. The stricter follow-up validation is recorded in
[`reference/quantization-v1-strict-validation.json`](../../reference/quantization-v1-strict-validation.json).
It confirms the actual original bindings and repeats the same arithmetic checks;
it does not pretend the original run already had the new before-selection guard.
Negative provenance tests are in `tests/test_quantization_provenance.py`.
The captured v2 run under `artifacts/quantization/operator-probe-v2/` passes the
same 112 reconstruction and 8,880 sampled arithmetic checks; its operator records
are identical to v1. The actual v2 executable also rejects a sidecar that changes
one weight prefix before processing any operator. See the
[strict review receipt](../../reference/quantization-provenance-review-v2.json)
and [before-selection guard evidence](../../reference/quantization-v2-sidecar-guard.json).

```powershell
rustc --edition 2024 --test experiments/quantization/q8_reference.rs -C opt-level=1 -o artifacts/quantization/q8-reference-tests.exe
artifacts/quantization/q8-reference-tests.exe --test-threads 1
python experiments/quantization/check_probe.py --self-test
python experiments/quantization/capture_probe.py --output artifacts/quantization/operator-probe-v1 --run
```

The last command builds with two jobs, then runs one scalar diagnostic thread and
the independent single-thread checker. Omit `--run` to build only. Coordinate
source capture with other edits and never use these runs as performance evidence
while corpus work is active. The output directory must be new.

## Isolated AVX2/FMA decode follow-up

`q4_avx2.rs` provides `linear(&Q4Linear, input, rows, output, Backend)` for
rows 1/2/4/8 and group sizes 64/128. The explicit AVX2 backend requires x86_64
AVX2 and FMA at runtime; the scalar backend remains available. Shape mismatch,
unsupported rows/groups, nonfinite activations/accumulation, and altered x86
rounding or FTZ/DAZ settings return explicit errors. The caller owns the output;
the operator uses only fixed stack scratch. An accumulation error may leave
partially written output; earlier validation failures leave output untouched.

The kernel loads eight packed bytes only when 16 weights remain in the current
row/group, sign-extends their nibbles and multiplies both vectors by the FP32
scale. Those separately rounded dequantized vectors feed every live row before
the next unpack. Eight FP32 FMA lanes per row accumulate K; a fixed reduction
combines them with up to 15 scalar tail values. This changes reduction order
relative to the scalar oracle, while preserving the quantizer and bitstream.
Odd K never consumes the unused high nibble or reads past a packed row.

`q4_avx2_probe.rs` is a std-only standalone executable/test harness. Its fixed
synthetic matrices cover 21 widths, including odd/vector/group boundaries and
Falcon's K=768/1024/2304, seven output channels, both groups and all four row
counts. Tests also exercise one output channel, known signed/ties-even codes,
zero groups, exact subnormal examples, unaligned caller slices, output sentinels,
input immutability, FP environment errors, and zero tracked heap allocations
inside warmed scalar/AVX2 calls. These finite synthetic tests are not a proof of
the relative error bound for arbitrary underflowing arithmetic.

```powershell
python experiments/quantization/capture_avx2_probe.py --output artifacts/quantization/q4-avx2-operator-NEW
python experiments/quantization/capture_avx2_probe.py --check artifacts/quantization/q4-avx2-operator-v1
```

The helper compiles with ordinary target defaults plus runtime-guarded AVX2/FMA
functions, preserves startup/after source hashes, a source archive, binaries and
logs, and checks an embedded source-inventory digest. It uses the independent
NumPy quantizer and `math.fsum` oracle for every output. The same frozen
`gamma_K` bound checks arithmetic; quantization-only and total errors remain
separate. It records no operator timing. See [AVX2 results](AVX2-RESULTS-V1.md).
