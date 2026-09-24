# AVX2 W4A32 operator evidence v1

Native Windows standalone capture passed nine optimized Rust tests, including
zero allocations for warmed scalar and AVX2 calls at rows 1/2/4/8, groups
64/128, and K=17/768/1024/2304. Other tests cover checked layouts, odd widths,
unaligned activation/output slices, known nibbles, zero groups, subnormal scale
multiplication, unsupported shapes, nonfinite values, and altered FP settings.

The independent NumPy/math.fsum pass checked 42 matrix/group cases and 168
batch cases: 78,316 original weights were independently quantized, all packed
code and FP32 scale bytes matched, and all 4,410 outputs **per backend** passed
the pre-existing `gamma_K * sum(abs(products))` arithmetic bound. No bound was
adjusted. Mutated AVX2 output and packed-code negative checks were rejected.

| Synthetic output comparison | Scalar | AVX2 |
| --- | ---: | ---: |
| Largest absolute accumulation error versus FP64 dequantized oracle | 0.000145348 | 0.000112196 |
| Largest fraction of frozen bound, all widths | 0.977431 | 0.977431 |
| Largest fraction of frozen bound, K=768/1024/2304 | 0.000441536 | 0.000436018 |

These are aggregate diagnostics for fixed synthetic values. The largest
quantization-only absolute output error was 22.9530; the report separately
records FP64 dequantized-minus-original error for each case, independently of
the backend. A smaller accumulation error does not establish useful quantized
OCR quality or CPU speed. The matrix output width is seven, with additional
one-channel unit cases; the production matrices and full model were not run.
No performance measurements were collected while corpus work was active.

Preserved evidence is under
[`artifacts/quantization/q4-avx2-operator-v1/`](../../../../artifacts/quantization/q4-avx2-operator-v1):

- [Build/source proof](../../../../artifacts/quantization/q4-avx2-operator-v1/build.json):
  `c471065780995eb845bee13203c69745e14a1c51a3a2fc7c6e0e16c15a2e3ef9`.
- [Independent check](../../../../artifacts/quantization/q4-avx2-operator-v1/independent-check.json):
  `ba4387838605c4260cc40831f204dadca202f124a8320da1416e0fd0e41613e6`.
- [Raw operator records](../../../../artifacts/quantization/q4-avx2-operator-v1/operators.json):
  `982e151912265f8ab8e983bd76333304ed91693fc0aa16e3d96628df496efa26`.
- Executable SHA256:
  `e67ea77621988f2ff2694fef9606f057623e895a3237d53110b82b3f6b8d26b6`.
- Source archive SHA256:
  `bf9e476ad6526c28218f5527521014391b6dfdf042c575009806f005dc278400`.

The only scalar Q4 API addition is a read-only `group_size()` accessor. Existing
format, quantizer arithmetic, scalar path, v1/v2 quantization evidence, production
source, Cargo dependencies and generic build capture are unchanged. This
follow-up is an isolated operator candidate, unqualified for full-model quality
or performance. The captured compiler/OS environment is not hermetic.
