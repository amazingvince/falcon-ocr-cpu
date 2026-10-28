# Original supplemental OCR diagnostics

`reference/original-supplement-v1-lock.json` freezes 15 original typeset images from 11 parent fixtures, separate from the 200 natural evaluation pages and all calibration data. Its SHA256 is `ff90af60b10c4bd411526326fed5a65ca820c4eef9d795fede808d80a1f7cfdf`. It adds concrete diagnostic inputs for the natural corpus's receipt, blank, sparse, rotation and script gaps; it does not change the natural corpus or establish performance on real scans.

| Category | Images | Parents | Contents |
| --- | ---: | ---: | --- |
| Receipts | 3 | 2 | Two fictional itemized layouts; one also has a faint-contrast rendering |
| Blank | 2 | 2 | Uniform white portrait and off-white landscape pages; empty text files |
| Sparse | 1 | 1 | One line, `Room 204`, near the center |
| Rotation | 4 | 1 | One prose page at 0, 90, 180 and 270 degrees counterclockwise |
| Multilingual | 5 | 5 | Russian, Greek, Arabic, Hebrew and Vietnamese short notices |

Text and page recipes were authored for this project. No external document images or passages were used. The exact intended logical Unicode text is stored inline and in hashed UTF-8 text files. The receipt line spacing and internal spaces are retained. The generated PNGs are RGB, have no EXIF orientation, and record their file and decoded-pixel hashes. Every derived image retains its parent ID; rotations and contrast variants are not independent examples.

`research/corpus-qualification/scripts/generate_original_supplement.py` uses pinned Pillow 12.3.0, FreeType 2.14.3, RAQM 0.10.5, HarfBuzz 14.2.1 and FriBidi 1.0.13, with two hashed DejaVu font files already installed in the reference Linux environment. Arabic/Hebrew are shaped as right-to-left logical strings. Missing-glyph sentinels and text bounds are checked during generation. The Bitstream Vera/DejaVu notice is preserved beside the images; font binaries are not included in the fixture artifacts.

Run the generator with the reference Python environment to create the artifacts; after the lock exists, use `--verify` to rerender in memory and compare every PNG, text file, license and manifest byte. The successful rerender and independent inverse-rotation/uniform-blank checks are recorded in `reference/original-supplement-v1-verification.json`. All 15 images were visually reviewed on a contact sheet, with full-size inspection of the faint receipt and all five language pages; per-page evidence is in `reference/original-supplement-v1-visual-review.json`.

The `canonical_path` and `ground_truth_path` fields are compatible with the free-running corpus harness. Compare GPU/CPU tokens and EOS separately from OCR quality against the authored text. On blank pages, report whether output is empty and how many characters were hallucinated; do not compute CER with a zero denominator. Keep every result separate from the natural corpus aggregate, and report parent grouping when summarizing variants.

These fixtures have clear printed glyphs and simple layouts. They do not establish accuracy on physical thermal receipts, broad multilingual documents, other typefaces, handwriting, arbitrary-angle rotation, or long generated outputs. They have no calibration split. At fixture freeze, rendering and reproducibility checks had passed and inference remained unmeasured. The later CPU snapshot and completed paired comparison are recorded separately below.

## First native Windows CPU execution

All 15 fixtures completed on the frozen native Windows executable in FP32, AVX2, four threads, maximum image dimension 1536 and output cap 512. Every page reached EOS; the run generated 1,056 tokens without errors or cap stops. The executable SHA256 is `befd29f70f27ced1ff5289b34b28112c36f712e213ac774f3587c690fe0c08d5`. Its immutable embedded source hashes and run contract are recorded in `reference/original-supplement-v1-cpu-run-snapshot.json`; the detailed CPU-only report is `reference/original-supplement-fp32-cpu-functional.json`.

This was functional execution under concurrent CPU/GPU load, not a benchmark. GPU parity remains pending in that snapshot; successful CPU completion does not imply matching GPU predictions or perfect OCR.

Both blank images emitted the literal `>>UNUSED_261<<` marker, 14 characters each, followed by EOS. The report retains these as nonempty outputs and assigns no blank-page CER/WER. The 180-degree page omitted its final sentence. The other rotated views, sparse line, Russian and Greek pages matched normalized intended text; Arabic, Hebrew and Vietnamese had respectively six, one and one character edits.

Market receipts emitted their correct cell text inside HTML tables. Primary untouched-output CER therefore includes markup: seven of 13 nonblank pages match normalized authored text, with micro CER 33.7679%. A separately labeled secondary table-text projection removes known table tags, preserves emitted cell order and decodes entities; it yields nine exact nonblank pages and micro CER 2.3468%. This secondary diagnostic was introduced after observing the CPU table output; it does not replace raw metrics, establish table-structure accuracy, or alter exact GPU/CPU comparison.

The earlier CPU-only report retains its original pending-GPU status. Seven regression checks in `research/corpus-qualification/tests/test_original_supplement_comparison.py` cover blank denominators/markers, normalization, table projection, edit distances and incomplete/failing parity gates.

## Completed CPU/GPU comparison

The [completed comparison](../../../reference/original-supplement-fp32-redecoded-gpu-comparison-v1.json) reports **15/15 exact matches** for generated token IDs, literal text, finish reasons, terminal tokens and input-prefix lengths: **1,056 tokens, 15 EOS stops, no missing or failed pages**. All 956 comparison checks passed. The GPU run is the fresh strict-FP32 reference at the same 1536 image limit and 512 output cap, with its preserved startup source/runtime archive.

The CPU comparison uses `artifacts/cpu/original-supplement-fp32-512-redecoded`, a separately verified decoding replay of the original saved token IDs. It changed none of these 15 texts and preserved every nontext inference field; it is not a new CPU model run. Original records and CPU-only reports remain unchanged. The [binding receipt](../../../reference/original-supplement-fp32-comparison-binding-v1.json) joins the comparison to the GPU validation, rechecks 300 GPU record invariants and 11 archived startup files, and records 108 unchanged files across the comparison.

CPU and GPU therefore share the bounded intended-text results described above, including the retained blank markers, HTML receipt markup and content errors. Untouched-output nonblank micro CER/WER are 33.7679%/14.1439%; the explicitly secondary table-content diagnostic gives 2.3468%/3.2258%. Blank CER/WER remain undefined. Parent grouping remains explicit, and these authored fixtures remain separate from the natural 200-page corpus, official quality metrics, full-model intermediate numerical checks and performance qualification. See the [comparison note](../../../reference/original-supplement-fp32-redecoded-gpu-comparison-v1.md) for counts and limitations.

To reproduce the read-only comparison, choose a new output path:

```text
python research/corpus-qualification/scripts/compare_original_supplement.py --manifest reference/original-supplement-v1-lock.json --cpu artifacts/cpu/original-supplement-fp32-512-redecoded --gpu artifacts/reference/original-supplement-fp32-512 --output <new-report.json>
```
