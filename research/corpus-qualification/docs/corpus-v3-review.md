# Corpus v3 visual review and selection

The frozen v3 selection has 200 evaluation pages, a 24-page smoke subset, and 64 separate calibration pages. It corrects the v2 category and source-family issues before full evaluation. Source annotations remain pinned to OmniDocBench revision `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec`; v1/v2 manifests and locks were not changed.

The selection is in `reference/corpus-manifest-v3.json`, SHA256 `cdec2de6243a61005463d6546e5d80cd1772bf362d126a62be7673316d6072ae`. The complete candidate decisions are in `reference/corpus-v3-candidate-review.json`, SHA256 `58f9f9502ddadc047d3fedd041d3d7d975a2a9840279cf89a8086b00c7897b3a`. Materialization subsequently verified all 264 downloaded source hashes against the reviewed bytes. Its frozen outputs are `reference/corpus-v3-evaluation-lock.json`, `reference/corpus-v3-smoke-lock.json`, and `reference/corpus-v3-calibration-lock.json`. The fresh `reference/corpus-v3-duplicates.json` reports zero exact cross-split RGB/text matches and zero cross-split 64-bit difference-hash candidates at distance <= 4. These checks qualify the image/annotation freeze, not model accuracy or statistical independence.

## What the visual review established

All 200 v2 evaluation pages were inspected on contact sheets, with 15 full-page follow-ups. The v2 review found five uncertain category labels: a clean medical article labeled degraded, a typeset page labeled handwriting, and three alleged multicolumn pages that instead showed a title list, one body column with a photograph, or a magazine opener. The ordinary group was especially weak: 28 of its 35 pages contained substantial mixed content, and only five had clear plain-prose tags. The per-page evidence and four perceptual-candidate pair reviews are preserved in `reference/corpus-v2-visual-review.json`.

The v3 work inspected 323 candidate/reused-calibration views representing 320 unique pages, including all 64 v2 calibration pages. It recorded 177 accepted, 140 rejected, and six uncertain views. Approval used the rendered page content; rejected or uncertain candidates cannot enter the new selection. This was a single-reviewer contact-sheet audit, not a transcription audit or a claim that every glyph was examined at native resolution. Selection did not use inference outputs, OCR accuracy, or success/failure of the official evaluator.

The new selection replaces all five uncertain evaluation labels and one additional uncertain calibration multicolumn label. Seven calibration pages are replaced because their known document or publication family also appears in evaluation. Altogether, 162 evaluation and 40 calibration page IDs are retained from v2; 62 selected pages are new to the corpus. None changes its category or migrates between the old splits.

## Ordinary prose means content, with layout recorded separately

The 35 evaluation and 16 calibration ordinary pages now contain sustained prose without meaningful tables, displayed equation regions, code, or infographics. Decorative photographs/art, incidental measurement quantities, abbreviations, and full-sentence prose paragraphs on slides are allowed. Candidate discovery required at least 300 annotated text characters, followed by visual approval; the threshold alone was insufficient.

Each ordinary split contains one page per known source family. Multiple prose columns are allowed and retain their layout annotation:

| Split | Single column | Two columns | Three columns | Total |
| --- | ---: | ---: | ---: | ---: |
| Evaluation | 26 | 6 | 3 | 35 |
| Calibration | 12 | 4 | 0 | 16 |

These counts use the pinned source layout annotations, with visual caveats recorded per page. The ordinary category includes eight evaluation and seven calibration pages annotated as `PPT2PDF`; these are explicitly reviewed prose slides, not a claim that the group consists only of conventional book pages. Image-quality caveats such as watermarks, show-through, and large whitespace remain in the evidence. The separate multicolumn category continues to cover general document layouts, including mixed content.

## Split separation and its limits

The corrected rules group generic `_page_N`, `.pdf_N`, and `notes/newspaper_<hash>_N` names. They also conservatively group identifiable publication/template families, including Putnam, Federal Register, named newspapers, several workbook/slide templates, and EY reports. The v2 audit had missed actual source-document overlap for a Shanghai listening-exam book, Daily Mail issue, and Guardian issue because generic `_page_N` suffixes were not stripped. No shared newspaper hash prefix was found across v2 splits; that fact alone did not establish independence.

V3 has zero overlap under these recorded family rules and zero exact normalized cross-split text-block matches of at least 160 characters. No former v2 calibration page ID enters v3 evaluation. However, earlier versions had document/publication overlap: any model variant tuned on an earlier calibration version must disclose that historical exposure. Unknown provenance behind opaque filenames prevents a universal independence claim.

There are 163 known evaluation families and 57 calibration families. In particular, 23 of the 25 evaluation handwriting pages come from two notebooks (12 and 11 pages), while all eight calibration handwriting pages come from another notebook. These are page counts, not independent writer counts. Similar styles/templates may remain even after exact and perceptual duplicate audits.

The original four v2 perceptual candidates were visually distinct content, including different Putnam years and Federal Register dates. V3 additionally separates those recognizable publication families conservatively. Its fresh duplicate report has no cross-split candidates under the fixed rule. The difference hash is only a review heuristic: different scans can evade it and similar layouts can produce false positives.

## Reproduction and remaining coverage

Run `python research/corpus-qualification/scripts/select_corpus_v3.py --draft` from the repository root to replay selection from the durable candidate review. The pinned annotation file and downloaded source files must be present. Every selected source file is rehashed against `reviewed_source_sha256`; `reviewed_source_path` identifies its already downloaded location. The materializer must verify the same digest when reusing or downloading it. The selector refuses to overwrite a frozen manifest.

`reference/corpus-v3-selection-verification.json` records byte-identical first freeze and replay, all 264 successful source-file hash checks through the manifest, and unchanged v1/v2 artifact hashes. Each candidate decision remains individually inspectable, including rejected alternatives and explicit visual family overrides.

The natural corpus still lacks dedicated receipts, truly blank pages, full-page rotation coverage, and scripts beyond English/Chinese. Evaluation language annotations are 101 English, 84 simplified Chinese, and 15 mixed English/Chinese pages. Calibration has 27 English, 35 simplified Chinese, and two traditional Chinese pages. Dense pages do not by themselves prove long-output coverage.

The separately versioned `reference/original-supplement-v1-lock.json` now supplies 15 original diagnostic images for receipts, blank/sparse pages, orthogonal rotations, and five additional languages; see `research/corpus-qualification/docs/original-supplement.md`. It records known text, geometry, font/rendering hashes and common parent IDs for derived variants. Keep those fixtures separate from the natural corpus and its aggregate metrics. Real scanned receipt robustness, broad script/font coverage and genuinely long generated outputs still require independent qualification; they are not properties already supplied by v3.
