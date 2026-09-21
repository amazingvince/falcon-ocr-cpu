This is an isolated observed-function diagnostic for the pinned RTX 4090 / Torch 2.11 CUDA environment. It is not a production backend, approximation policy, portable GPU emulation claim or performance result.

The capture completed with zero result or mapping mismatches across all 1,266,679,808 requested GPU arguments. The preserved Rust probe then matched all 1,057 observed boundary outputs exactly; canonical input ordering and all artifact hashes were rechecked. The [validation receipt](C:/Users/amazi/Documents/ChatGPT/falcon-ocr/reference/rsqrt-table-rust-validation-v1.json) binds the evidence. Table SHA-256: `218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246`. GPU report SHA-256: `ae662d30fc8a82a20318114d98ebeba8fe619b7ed7359e736bced931e96315d2`.

`export_rsqrt_table.py` captures `torch.rsqrt` at every F32 value in `[1,4)`, in ascending raw-bit order `0x3f800000..0x40800000` (exclusive end). The raw little-endian output file contains 16,777,216 entries, exactly 67,108,864 bytes. The complete canonical input bytes are preserved separately.

For argument bits with biased exponent `e` and mantissa `m`, let `k=e-127`, `p=k mod 2` in `{0,1}`, and `s=-floor(k/2)`. The canonical table index is `(p<<23)|m`; the reconstructed output is the canonical result times `2^s`. The equivalent Rust bit operation adds `s*2^23` to the canonical output bits. Every result stays positive normal over the requested domain, so scaling needs no subnormal rounding.

The exporter checks **all 1,266,679,808 arguments** with biased exponents 104 through 254, including every mantissa: positive normal inputs from `f32::EPSILON` through `f32::MAX`. Each GPU result is compared bit for bit with canonical-output scaling. GPU FP32 multiplication by an exact power of two is separately checked against the integer exponent mapping. No mismatch tolerance is allowed. Chunks are at most 1,048,576 elements, and each exponent receives its own exact counts and bounded mismatch examples. A mismatch, incomplete domain or changed source prevents a passing report.

`rsqrt_lookup.rs` exposes `pub fn rsqrt_from_table(value: f32, table: &[f32]) -> f32`, with no allocation or I/O. Unsupported arguments, incorrect table length and invalid selected entries panic. The caller must separately verify the complete table SHA and successful exhaustive report; the function does not attest its input table. Unit tests cover all canonical indices, every supported exponent at seven mantissa boundaries against independent F64 arithmetic, power-of-four scaling and invalid inputs.

The GPU exporter preserves 1,057 additional argument/result pairs directly from its observed outputs, ordered by exponent and the seven fixed mantissas. `rsqrt_lookup_probe.rs` checks every pair using the Rust public function and rejects missing, reordered or substituted argument coverage. This small host probe supplements the exhaustive GPU comparison; it does not independently execute a billion Rust lookups.

Frozen initial sources and standalone host build are in `artifacts/diagnostics/rsqrt-lookup-host-v2`. `build.json` records exact compiler commands, source ZIP, before/after hashes, test logs and executable identities. The GPU owner is responsible for invoking the exporter under project UUID isolation and preflight, before using any resulting table. Raw table bytes without `report.json` status `passed_exact`, all 151 exponent records, complete counts and matching source/artifact hashes are incomplete evidence.

GPU invocation, to a fresh directory only:

```text
<pinned-reference-python> experiments/rms_norm/export_rsqrt_table.py --output artifacts/reference/rsqrt-canonical-table-fp32-v1 --chunk-size 1048576
```

Host verification after that capture:

```text
artifacts/diagnostics/rsqrt-lookup-host-v2/lookup-probe.exe artifacts/reference/rsqrt-canonical-table-fp32-v1/rsqrt-table.f32le artifacts/reference/rsqrt-canonical-table-fp32-v1/boundary-pairs.u32le
```

The table cannot establish RMS reduction parity by itself. Its intended use is one isolated width-768 intervention with unchanged width-64 Q/K normalization and unchanged live production code. It makes no claim that the table is an appropriate shipping design.
