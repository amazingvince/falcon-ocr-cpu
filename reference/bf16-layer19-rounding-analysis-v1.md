# Layer 19: a raw BF16 rounding boundary remains unresolved

At prefill row 41, head 6, dimension 54, the saved AVX512 CPU final numerator is
`0.03098928928375244` and denominator is `6.83806848526001`. Their FP32 quotient,
`0.00453187758103013` (`0x3b948025`), lies only `1.7229467630386353e-8` above the
BF16 midpoint `0.0045318603515625`. It rounds to CPU raw
`0.004547119140625`; fused GPU raw is the adjacent lower value
`0.0045166015625`. The scaled result differs by two frozen-bound units.

The CPU trace has three live tile updates, ordered key starts 128, 0, 64. Its
tile at key start 0 has PV[54] `0.2534388303756714`; an independent FP64 sum of
the **same saved CPU BF16 probabilities and saved BF16 V** is
`0.25343868136405945`, a difference of `1.4901161193847656e-7`. Replacing only
that PV scalar, retaining CPU alpha, other tile terms, FP32 accumulation and
the CPU denominator, produces quotient `0.004531855694949627`, which rounds
to the GPU raw value. This identifies a sufficient local counterfactual, not
the GPU's actual computation or a proposed production fix.

Fused GPU LSE is `3.8521227836608887`; CPU LSE is `3.8521230220794678`. The
independent GPU blockwise oracle has **exactly the CPU LSE bits**, yet its raw
and scaled values match fused GPU. LSE alone therefore cannot identify the
numerator/denominator cause. It also cannot recover the separate maximum and
normalizer, or the quotient hidden by the BF16 cast.

The independent GPU tile artifact contains no layer-19 tensors. Actual fused
QK/probabilities, numerator, denominator and pre-cast quotient remain unavailable.
The queued [v2 fused export specification](bf16-fused-substage-export-spec-v2.json)
now includes intact layer 19 and an adjacent control. Its complete raw/LSE output
must match the original and frozen fixture bit-for-bit before interpretation.

The sink replay source review found no concrete issue in the performed finite
data experiment. Preserved source, binary, input and output hashes were rechecked,
as were all 3,328 independent BF16 rounding lines. Across 13 selected heads,
all 832 CPU scaled values were reproduced. All 20 scaled failures passed their
unchanged bounds when GPU raw replaced CPU raw; replacing LSE alone fixed none.
This isolates the sink boundary and does not establish an upstream GPU cause.

[Detailed evidence and hashes](bf16-layer19-rounding-analysis-v1.json) distinguish
the CPU diagnostic, independent GPU oracle and actual fused final outputs.
No inference, GPU execution, production change, tolerance change or timing
measurement was performed for this review.
