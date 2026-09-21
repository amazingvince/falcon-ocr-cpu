# Original supplement: completed CPU/GPU comparison

All **15/15 pages match exactly** in generated token IDs, literal decoded text, finish reason, terminal token and input-prefix length. Both sides generated **1,056 tokens**, with **15 EOS stops and no length stops**. No input or output is missing or failed. All 956 comparison checks passed.

The CPU side uses the preserved native Windows FP32/AVX2, four-thread inference at maximum side 1536 and output cap 512, followed by the separately verified saved-token text replay. That replay changed none of these 15 texts and preserved every nontext inference field. It is not a new CPU inference run. The GPU side is the fresh strict-FP32 reference with its validated startup source/runtime archive.

| Existing intended-text diagnostic | CPU and GPU |
| --- | --- |
| Nonblank pages exact after NFC/whitespace normalization | 7/13 |
| Nonblank character edits / reference characters | 777/2301 |
| Nonblank micro CER | 33.7679% |
| Nonblank word edits / reference whitespace words | 57/403 |
| Nonblank micro WER | 14.1439% |
| Parent-balanced macro CER, nine nonblank parents | 23.6661% |
| Secondary table-content diagnostic: exact pages | 9/13 |
| Secondary table-content micro CER / WER | 2.3468% / 3.2258% |
| Blank pages with empty raw or normalized output | 0/2 |

Raw metrics preserve emitted HTML. The two market receipt variants contain correct text in table markup; the explicitly secondary table-content projection removes only known table tags and handles entities. It was introduced in the earlier CPU diagnostic and does not replace raw metrics or establish table-structure accuracy. Four pages retain content errors after this projection: the 180-degree page (46 character edits), Arabic (6), Hebrew (1) and Vietnamese (1).

Both blank pages preserve `>>UNUSED_261<<`: 14 nonempty characters each, 28 total. Their CER/WER remain undefined, with no zero-denominator substitution or marker stripping. The 15 pages represent 11 authored parent groups; derived contrast/rotation variants are grouped in the report rather than claimed as independent samples.

Artifacts:

- [Comparison](original-supplement-fp32-redecoded-gpu-comparison-v1.json), SHA256 `5e42540ac31124e03c1d97ad029ee9fdb59d3dfa4a4374474fec18dbf783ef4d`.
- [Source-binding receipt](original-supplement-fp32-comparison-binding-v1.json), SHA256 `e9a2ebe67b5b9df5f992a633f9739d0ad89b35575752318583e491a03a734f2c`: 300 freshly checked GPU record invariants, 11 archived startup source files and 108 unchanged files across comparison, joined to the [GPU validation](gpu-original-supplement-fp32-validation.json).
- The existing seven comparator tests passed before this run. No comparator, production source, original record or historical report was changed.

This closes the original supplemental output comparison only. These generated fixtures and their intended-text diagnostics do not establish natural-corpus quality, full-model intermediate numerical parity or performance. The separately frozen 200-page corpus and its official metrics remain separate work.

Reproduction, with a new output path:

```text
python scripts/compare_original_supplement.py --manifest reference/original-supplement-v1-lock.json --cpu artifacts/cpu/original-supplement-fp32-512-redecoded --gpu artifacts/reference/original-supplement-fp32-512 --output <new-report.json>
```
