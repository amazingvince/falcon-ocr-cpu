# Research corpus and smoke reference

The corrected `corpus-manifest-v3.json` selects **200 evaluation pages**, a **24-page smoke subset**,
and **64 separate calibration pages** from the pinned
[OmniDocBench snapshot](https://huggingface.co/datasets/opendatalab/OmniDocBench/tree/aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec).
The verified annotation JSON contains 1651 pages and has SHA-256
`a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.

| Category | Evaluation | Smoke | Calibration |
|---|---:|---:|---:|
| Ordinary documents | 35 | 3 | 16 |
| Tiny text | 30 | 4 | 8 |
| Tables | 30 | 4 | 8 |
| Formulas | 30 | 4 | 8 |
| Handwriting | 25 | 3 | 8 |
| Degraded scans / historical documents | 25 | 3 | 8 |
| Multiple columns | 25 | 3 | 8 |
| Total | 200 | 24 | 64 |

Selection uses annotation attributes and a reproducible SHA-256 ordering, never
model scores. Evaluation and calibration exclude matching filenames and known
PDF and `notes_<hash>_<page>` notebook families. Calibration reserves one whole
notebook; evaluation uses the other two and separate opaque-ID examples. Writer
and handwriting-language diversity remain limited. Opaque page IDs cannot prove source-document independence;
content and perceptual duplicate checks remain necessary. Tiny text uses annotated
line-height and character-count thresholds. Every selected category still needs
visual review before the corpus is considered qualified.

The [dataset's copyright statement](https://huggingface.co/datasets/opendatalab/OmniDocBench/blob/aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec/README.md#copyright-statement)
limits data to research use and excludes commercial use. Images and annotations
remain in ignored local artifacts and must not ship in runner releases. This is
distinct from the evaluator code's Apache license.

The original v1 split missed notebook family suffixes: all three handwritten
smoke pages came from one notebook, with other pages of that notebook in
calibration. **v1 is qualitative smoke only and must not calibrate quantization.**
The v1 files remain unchanged to preserve result provenance. The corrected v2
selection has now materialized all 264 evaluation/calibration images, totaling
215,511,895 source bytes. The duplicate audit found zero exact cross-split RGB
or ground-truth-text duplicates and four perceptual candidates. That review
identified publication-family dependence and category issues; v2 remains
preserved as an intermediate selection.

The corrected v3 selection is materialized: 200 evaluation pages, a 24-page
evaluation subset, and 64 separate calibration pages. Its reviewed selection
and conservative publication-family exclusions leave zero exact or perceptual
cross-split duplicate candidates. `corpus-v3-evaluation-lock.json` freezes the
200 evaluation inputs. Full GPU and native Windows CPU free-greedy evaluation
is in progress, with the same 1536-pixel maximum side and 4096-token cap. Dataset
selection and pixel freezes do not themselves establish model quality.

The 24 v1 smoke images total 23,637,445 downloaded bytes. Their source bytes, decoded
RGB pixels, canonical PNGs, and assembled ground truth are frozen in
`corpus-smoke-lock-v1.json`. Source RGB decoding is followed by RGB conversion
before upstream resizing; no EXIF auto-orientation is applied. Matching CPU
inference should use the canonical PNGs initially to isolate model/preprocessor
differences from JPEG decoder differences.

Ground truth joins nonignored blocks with valid reading order, retaining text,
LaTeX formulas, and HTML tables. Original region annotations are also retained.
The corpus driver reports whitespace-normalized character edit distance as a
**diagnostic**, not an official OmniDocBench score. The definitive quality gates
still require proper text/table/formula evaluation and the full 200 pages.

Metadata selection refuses to overwrite an existing frozen manifest. The frozen
v1 selection remains available for reproducing the existing smoke results. Use
fresh run and comparison paths to preserve earlier evidence:

```bash
python scripts/prepare_corpus_smoke.py
FOCR_MIN_FREE_MIB=12000 bash scripts/run_reference.sh --corpus \
  --manifest reference/corpus-smoke-lock-v1.json \
  --output artifacts/reference/corpus-smoke-fp32-rerun
python scripts/compare_corpus.py \
  --gpu artifacts/reference/corpus-smoke-fp32-rerun \
  --output reference/corpus-smoke-fp32-rerun-comparison.json
```

The GPU driver reuses one FP32 model and saves each completed page independently.
The v3 GPU run has completed all 200 pages: 275,903 generated token IDs, 181 EOS
stops and 19 explicit 4096-token length stops. Its longest naturally terminated
output contains 3733 tokens. `gpu-corpus-v3-fp32-summary.json` binds all saved
record hashes and validates the canonical PNG and ground-truth bytes. This is a
GPU reference, not a completed CPU comparison or an official quality score.
The historical source archive was captured after launch; that limitation is
preserved in the summary rather than retroactively claiming startup attestation.
It requests 4096 output tokens and a 1536-pixel maximum side, with explicit context
checks. It records token IDs, top-two logits and margins, stopping reason, GPU
memory and qualified stage timings; it does not capture full-page hidden tensors.
The historical driver's `--resume` checks each saved page's configuration and
identity. Future runs will also validate complete record invariants and bind a
preserved startup source/runtime archive; these stronger checks do not rewrite
historical run provenance.
Long-output qualification at 8192 tokens is recorded separately below.

The v1 GPU run completed all 24 pages with 28,734 output tokens and no 4096-token
truncation. `artifacts/reference/corpus-smoke-fp32-4096/summary.json` records the
run; per-page files retain tokens and decision margins. The durable
`windows-rust-corpus-smoke-fp32.json` comparison now contains all 24 CPU records:
all 28,734 output IDs, text, and stopping reasons match exactly. It verifies model
pins, runtime options, the CPU contract, canonical PNG/RGB hashes, and ground
truth before comparing free greedy outputs. This remains a v1 qualitative smoke;
full tensor gates and corrected-corpus quality qualification remain separate.

The v3 comparison found a decoder compatibility issue despite exact generated
IDs: pinned Transformers 5.14.1 deliberately skips legacy tokenization cleanup
for this BPE tokenizer, while the initial Rust decoder changed spaces before
punctuation. `tokenizer-cleanup-v1.json` preserves 14 actual pinned-tokenizer
cases, including the observed pages. The corrected Rust decoder also follows
Python's outer whitespace stripping. Original inference records remain intact.
`windows-rust-corpus-v3-fp32-redecoded-100.json` compares a separately labeled replay
of saved CPU IDs through that corrected decoder; it is not fresh inference.
Replay checks bind the original run and record hashes, preserve every inference
field and timing exactly, and verify the frozen decoder binary/source assets.
That immutable partial snapshot verifies all 100 paired pages and 115,361 output
IDs, corrected text and stopping reasons exactly; the full 200-page gate remains
incomplete. The earlier replay that changed floating-point timing values during JSON parsing
was rejected; only the corrected `-redecoded-v2` directory is used.

The later immutable 140-page snapshot also passes: all 204,967 IDs, corrected
text and stops match, with 125 EOS and 15 length stops. The source run had
advanced beyond 140, so byte-identical copies of exactly the first 140 manifest
records were frozen before the same preserved decoder replayed them. Nine
texts changed under the corrected decoding policy; no inference field or
timing changed. `windows-rust-corpus-v3-fp32-redecoded-140.json` validates this
separate `-redecoded-v2-snapshot-140` lineage and explicitly lists 60 missing
CPU pages. The 100-page reports and earlier rejected evidence remain intact.

The separately captured Linux build completed the corrected v3 24-page subset
under WSL: all 29,205 IDs, literal text and stops match the parent 200-page GPU
records, including one honest 4096-token length stop. The source-bound result
is `linux-corpus-v3-smoke-fp32-summary.json`. This is fresh inference with the
corrected decoder, not a text replay, and establishes selected-page functional
output parity rather than bare-metal Linux performance or full hidden-state
qualification.

The original synthetic numeric boundary fixture also completed independently on
GPU and native Windows Rust CPU: all 8192 generated IDs, text, and `length` stopping
reason match exactly, with provenance checks passing in
`windows-rust-long-numeric-fp32.json`. Its 8100-token prefix plus 8192 outputs uses
16292 positions within the 16384-position context. The model hit the explicit
output cap; this proves long-context execution parity on this fixture and does
not establish long-document OCR quality. Concurrent functional work makes these
run times unsuitable as benchmark measurements.

The separate exact-context GPU request reached its 8,284-token cap: its 8,100
prefix tokens plus emitted output total 16,384 positions, with `length` stopping.
The driver directly observed final KV cursor 16,383, because the final emitted
token was not fed back into the model. All first 8,192 IDs match the earlier
reference. `gpu-exact-context-boundary-fp32-summary.json` binds the new startup
source archive and validation receipt. The CPU comparison remains separate;
this synthetic execution fixture does not establish natural OCR quality.

The original 15-page supplement is complete under the same pinned FP32 model:
all 1,056 IDs, text, prefix lengths and EOS stops match the Rust saved-token
replay exactly. `original-supplement-fp32-redecoded-gpu-comparison-v1.json`
verifies the new GPU startup archive and the CPU inference/replay lineage.
The explicit output cap is 512 for that supplement; a separate three-page
reference at cap 4096 binds the blank, sparse and receipt inputs used by the
mixed-batch functional experiment. Its saved-layout comparison is separate.
`functional-batch-v1-gpu-parity.json` joins those three fresh records and the
existing prose record to all five saved functional invocations: 20/20 page
results match in IDs, text, stops and counts, with 1,242 IDs per mode. GPU
prepared dimensions were not recorded; canonical pixels, processing options
and prefix lengths are bound instead. These are functional comparisons.

An optional secondary benchmark is
[olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench/tree/54a96a6fb6a2bd3b297e59869491db4d3625b711),
pinned at `54a96a6fb6a2bd3b297e59869491db4d3625b711`. Its primary dataset card
reports 1403 PDFs, 7010 machine-checkable assertions, and ODC-BY licensing.
These assertions cover reading order, tables, formulas and selected text rather
than providing complete page transcriptions. It has not been downloaded or run.
