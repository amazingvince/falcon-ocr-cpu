# Draft head

A trained speculative drafter for the CPU runner: an EAGLE-3-style draft head
(one decoder block over the target's layers 2, 11 and 19, text only, a
16,383-token draft vocabulary) that the Rust runner loads with
`--draft-head <file>` (`src/draft_head.rs`), or by itself when
`falcon-ocr-v1.5-draft-head.safetensors` sits next to the model files. Drafts
only change speed: the model verifies every drafted token, and every A/B
below checks that each page's tokens equal the no-speculation run.

The published file is `falcon-ocr-v1.5-draft-head.safetensors` in
[BEE-spoke-data/falcon-ocr-v1.5-cpu](https://huggingface.co/BEE-spoke-data/falcon-ocr-v1.5-cpu):
the tensors of `s3-16k-cont`, with the local training path removed from its
metadata. Export validation uses `matches_the_pytorch_chain`, which passed
for this head with its PyTorch fixture.

The training records use three dataset stages:

| Record label | Training pages | Meaning |
|---|---:|---|
| Stage 1 | about 4.7k | Initial training set |
| Stage 2 | about 24.5k | Expanded training set |
| Stage 3 | 48,069 | Largest set; 106 pages excluded by the 13-gram check |

The published head continues the stage-2 head for one more seven-step
training epoch on the stage-3 data, rather than retraining from scratch.

Timing comparisons, acceptance, settings sweeps and drafter costs are in
[PERFORMANCE.md](../../docs/PERFORMANCE.md#trained-draft-head).

Retraining from scratch (one step-1 epoch, then one seven-step epoch on the
48k pages) reaches 74.7% / 40.8% first- and second-draft calibration
accuracy, against 75.0% / 44.6% for continued training.

## Pipeline

| Step | File | Notes |
|---|---|---|
| Raw sources | [data/download.py](data/download.py) | English, permissively licensed only (`--stage 1/2/3`), so the head can be published |
| Page images | [data/build_pages.py](data/build_pages.py) | pHash dedup and blocklist of every evaluation image ([data/make_blocklist.py](data/make_blocklist.py)) |
| Transcripts | [data/serve_vllm.sh](data/serve_vllm.sh), [../vllm-serving/scripts/request_vllm_draftgen.py](../vllm-serving/scripts/request_vllm_draftgen.py) | the official Falcon-OCR vLLM image, greedy |
| Decontamination | [data/decontam_13gram.py](data/decontam_13gram.py) | pages sharing 13-grams with OmniDocBench are excluded |
| Training | [train/eagle3.py](train/eagle3.py) | features from the frozen target online; step-1 phase (window 2048), then 7 steps (window 512, weights 0.5^j) |
| Export | [train/export_head.py](train/export_head.py) | safetensors + parity fixture for `draft_head::tests::matches_the_pytorch_chain` |
| Low-rank head | [train/lowrank_head.py](train/lowrank_head.py), `eagle3.py --head-rank R --head-only` | SVD, then head-only distillation |
| CPU A/B | [cpu_ab.py](cpu_ab.py), [compare_pages.py](compare_pages.py) | alternating rounds, page totals, token identity (`@<exe>` runs another binary per arm); compare_pages.py lists the pages two arms disagree on, with CER against the FP32 reference |
| Cost model | [train/cost_model.py](train/cost_model.py), [train/check_alignment.py](train/check_alignment.py) | offline acceptance and gated-speedup estimates |

Every page records its source's license in the stage manifests. The
stage-3 pages by source are:

| Source (Hugging Face) | Pages | License or terms |
|---|---:|---|
| `allenai/olmOCR-mix-1025` (documents, books, LoC transcripts, National Archives) | 25,841 | ODC-BY-1.0 (the card adds: intended for research and educational use under Ai2's Responsible Use Guidelines) |
| `pixparse/pdfa-eng-wds` | 12,000 | card license "other": Common Crawl terms of use and Digital Corpora terms of use |
| `pixparse/idl-wds` | 7,000 | card license "other" (UCSF Industry Documents Library; the linked license file is missing on the Hub) |
| `docling-project/DocLayNet-v1.2` | 1,937 | CDLA-Permissive-1.0 |
| `biglam/loc_beyond_words` | 500 | CC0-1.0 |
| `NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge` | 392 | Apache-2.0 |
| `PDFPages/zenodo-presentations-open` (CC-BY-4.0 records only) | 230 | CC-BY-4.0 |
| `HumynLabs` handwritten notes (math, physics, chemistry, biology, CS) | 169 | CC-BY-4.0 |

No non-commercial or evaluation-only set was used. PDFA and IDL carry
third-party terms rather than a standard license; the published attribution
lists them. Data, heads and runs live outside the repository at
`D:/falcon-draft`: `heads/s3-16k-cont.safetensors` is the published head's
source, and `cpu-ab/` holds every A/B.

## Lessons

- Data was the lever. Stage 1 to stage 2 moved calibration first-draft
  accuracy from 62% to 73%; stage 3 added 1.9 points first-draft and 5.5
  points second-draft accuracy. The chain skill accumulates over seven-step
  epochs: continued training beat retraining from scratch.
- Train on chains. The seven-step phase makes the head's confidence honest on
  its own drafts; without it the confidence gate lets bad chains through.
- CPU speculation is not GPU speculation. Verification reads the image
  cache, so short confident chains win and trees would not pay. Target
  steps evict the drafter; cold memory traffic determines its cost.
- Offline estimates run high: failed first drafts, cold caches and
  unchanged prefill work reduce the measured gain.
- Measure on a quiet host with page totals and token identity against no
  speculation. Windows' `Win32_Processor.LoadPercentage` misreports load
  (use `Get-Counter`), and WSL and Windows clocks jump (use the binary's
  timings). See [the measurement protocol](../../docs/DEVELOPMENT.md#measuring-performance-and-input-changes).

## Operational notes

- Shell scripts checked out with CRLF line endings need LF line endings
  before running in WSL (`D:/falcon-draft/serve_stage3.sh` performs this
  conversion for the vLLM launch). From Git Bash, run WSL commands with
  `MSYS_NO_PATHCONV=1` so `/mnt/...` arguments are not rewritten.
- WSL sets `NAME` to the host name, so `serve_vllm.sh`'s `${NAME:-...}`
  names the first server's container after the host; stop it by that name.
- The 13-gram decontamination counts distinct 13-grams with at least 8
  different words; plain hit counts flag pages of repeated zeros or dots.
- Parity fixtures come from random features and can hold near ties; the
  Rust test accepts a different token only where PyTorch's top-two margin is
  under 0.25.

## Runner knobs

See [experiment flags](../../docs/DEVELOPMENT.md#knobs-for-experiments)
and [draft-head selection](../../docs/ARCHITECTURE.md#decode-machinery).
