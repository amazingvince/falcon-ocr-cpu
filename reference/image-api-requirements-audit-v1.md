# First-release image and API evidence audit

The existing evidence covers successful RGB, PNG and JPEG inputs well enough for
this first-release scope. No implementation defect was confirmed by this bounded
source review. Two focused negative-input test gaps remain; this audit did not
run tests, CLI commands, inference or builds, and changed no production code.

| Requirement | Existing test/evidence | Assessment |
| --- | --- | --- |
| Two uint8 Pillow resize stages, dimension operation order and ties-to-even alignment | `pixel_exact_pillow_resize_fixtures` (12 cases), `upstream_two_stage_patches_and_torch_positions` (9 cases), alignment/order unit tests; Windows/Linux library validation receipts | Covered on frozen cases. |
| RGB channel/pixel/patch order and FP64-to-FP32 normalization | Exact full patch hashes plus `patches_are_grid_then_pixel_row_major_rgb` | Covered. |
| Floating spatial positions under reference-derived bounds | Exact independent coordinate hashes plus each fixture's frozen PyTorch/NumPy bound; maximum observed absolute discrepancy 5.960464477539063e-8 | Covered for these fixtures, not a universal tolerance claim. |
| Source-mode PNG/JPEG behavior | 39 PNG and 36 JPEG complete patch-buffer cases, plus 12 exact JPEG RGB comparisons | Covered: palette/one-bit, transparency, alpha, grayscale/16-bit, CMYK, progressive JPEG, qualities and subsampling. No demand for more successful-codec fixtures. |
| Pinned tokenizer, image/register/end markers, no BOS, plain prompt | SHA checks, `pinned_tokenizer_prompt_and_decode`, modified-tokenizer rejection and model token/index tests | Covered. |
| Actual upstream text decoding and EOS handling | 14 independent Transformers cases, Python whitespace tests and 3-page free-running tokenizer regression | Covered for the pinned BPE behavior; punctuation spaces and special markup remain intact. |
| Empty/degenerate RGB and conflicting preparation bounds | `rejects_degenerate_shapes_and_configuration` and `minimum_area_alignment_cannot_exceed_runtime_image_bound` | Existing unit evidence covers zero image, reversed bounds, zero-sized aspect resize, aspect limit and minimum-area conflict. Public-entry error tests remain below. |
| Exact output caps and insufficient context | `token_budget_is_exact_and_checked`, single/mixed capped generation and actual CLI rejection of 8100+8285 >16384 | Covered, including integer overflow in budget arithmetic and a nonzero CLI exit with no JSON result. |
| Unsupported execution choices | `RunnerConfig::validate`, phase-packing compatibility tests, BF16 unsupported-mode tests and Clap enums | Covered by source/targeted tests; unsupported CPU ISA dispatch is guarded. FP32 CLI validates runner settings after model loading, which is existing behavior, not a new fail-before-load guarantee. |
| Result fields, input order, independent EOS and shared resources | `OcrResult`, private immutable runner configuration, shared `Arc<Model>`, Windows/Linux smoke and mixed 2/4/8 tests | Covered on existing fixtures. The full-page four-input run is separate ongoing evidence. |
| Empty collections and malformed files through public APIs/CLI | Source currently returns `Ok([])` for valid-option empty library batches/files; Clap requires at least one CLI image; file decoder returns `Result` errors | Behavior exists but lacks a focused regression record: the two gaps below. |

Only two bounded follow-ups are recommended before closing this part of the plan:

1. **Malformed file errors:** missing path, empty bytes, a recognizable unsupported
   format, and truncated PNG/JPEG fixtures. Assert a returned error without panic
   from `prepare_file`; exercise representative public file/CLI failures with
   no successful JSON output. Corrupt-codec acceptance/rejection should follow
   the actual pinned reference where relevant, rather than an invented universal
   rejection policy for every imperfect image.
2. **Public argument and empty-input behavior:** table-test zero output cap,
   zero/reversed dimensions, non-16-aligned maximum and zero batch size. Freeze
   the existing library empty-collection no-op (`Ok([])`), CLI missing-image
   argument error, and propagation of an invalid image/member before prefill.
   Use an invalid member in the first chunk so the test needs no model forward.
   Later-chunk failures currently return one overall error; the CLI emits results
   only after the call succeeds, rather than promising partial streaming output.

These are evidence gaps, not confirmed decoder/runtime defects. Coordinate any
tests or fixes with the owner of the active captured source before editing.
No PDF, layout detector, HTTP service, generic codec-fuzzing campaign, metadata
normalization feature or new image format is required by this audit. Metadata
orientation/color profiles remain unapplied, matching the stated upstream path.

The plan's separate full-context numerical, large-batch and memory-pressure gates
remain tracked elsewhere. Existing 8k generation and over-budget rejection are
not silently promoted into full-context numerical qualification. The functional
harness's 12 GiB preflight is not a promise that arbitrary library allocations
return recoverable out-of-memory errors.

The [receipt](image-api-requirements-audit-v1.json) binds the inspected source,
fixture metadata and existing evidence. Existing test receipts are cited with
their original scope; no fresh execution or startup build attestation is implied.
