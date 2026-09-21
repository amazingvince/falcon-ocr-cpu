# Exact-contract saved functional output comparison

All **20/20 request comparisons match** against strict FP32 GPU references at minimum dimension 64, maximum dimension 1536 and output cap 4096. This joins the existing sequential control and four joint batch-size-four layouts to the same four inputs; it runs no CPU inference and changes no saved records.

| Input, in functional request order | Prefix tokens | Output tokens | Stop | CPU prepared dimensions |
| --- | ---: | ---: | --- | --- |
| Prose `3f294b5e60a0c2d4` | 6544 | 1140 | EOS | 1088 x 1536 |
| blank-white | 5136 | 2 | EOS | 1024 x 1280 |
| sparse-room | 3088 | 6 | EOS | 1024 x 768 |
| receipt-cafe | 2416 | 94 | EOS | 640 x 960 |

Every invocation matches GPU token IDs, literal text, finish reason, output count, input count and FP32 precision. Each invocation contains 1,242 matching generated IDs; there are 6,210 comparisons across all five invocations. The tested joint modes are expanded/unpacked, compact/unpacked, expanded/phase-packed and compact/phase-packed, with AVX2 and four CPU threads. All CPU results are original CLI outputs; no saved-token text replay or text normalization is used.

The three new references use the [frozen subset](functional-originals-v1-lock.json), SHA256 `c2b56cf580b4edef8968bc78edf9c021905cc3ba41c5e8ee065f2f0c7150965a`, preserving complete parent page objects and the source manifest's relative order. The join maps by page identity to functional request order. Historical cap 512 outputs are not reused. The [new GPU validation](gpu-functional-originals-fp32-4096-validation.json) records fresh strict FP32 inference and startup source/runtime identity. Prose uses the previously completed cap 4096 natural reference.

The [metadata-only result](functional-batch-v1-gpu-parity.json), SHA256 `37c6eb165736d27eeacf9ee9c71dd8406e79a9bf77d0d0157fc73bd453013a0b`, binds the functional final receipt, plan, captured executable/source archive, original JSONL/stderr/invocation hashes, model assets, source locks and GPU records. It records 304 identity/GPU checks, validates every CPU result with the existing shared result validator, and rechecks 76 bound files after comparison. The reproducible, fixed-input join is [compare-functional-saved-gpu-v1.py](compare-functional-saved-gpu-v1.py); it refuses an existing output path and never launches inference.

GPU prepared width/height were **not recorded**. CPU dimensions match the frozen expected dimensions; GPU canonical pixels, processing options and prefix lengths match, without claiming an observed GPU dimension comparison. The natural prose reference retains its explicitly after-launch source archive and missing startup prompt/config/tie fields. No retroactive startup identity is asserted.

This is output evidence for this one frozen mixed-b4 case and these four layouts. It does not establish batch sizes 2/8 coverage, arbitrary input ordering, full intermediate numerical parity, corpus quality, performance or a backend promotion. Timing fields remain unaggregated. Existing functional receipts, corpus runs and STATUS were not edited.
