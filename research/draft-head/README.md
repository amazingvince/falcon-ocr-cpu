# Draft head

A trained speculative drafter for the CPU runner: an EAGLE-3-style draft head
(one decoder block over the target's layers 2, 11 and 19, text only, a
16,383-token draft vocabulary) that the Rust runner loads with
`--draft-head <file>` (`src/draft_head.rs`), or by itself when
`falcon-ocr-v1.5-draft-head.safetensors` sits next to the model files. Drafts
only change speed: the model verifies every drafted token, and every A/B
below checks that each page's tokens equal the no-speculation run.

The published head cuts total time by 21–26% against n-gram drafts on 16
English held-out pages, with identical tokens. It is
`falcon-ocr-v1.5-draft-head.safetensors` in
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

## Results (fast mode, packed model, 12 decode threads, Ryzen 9 7950X)

16 English held-out pages, 2 rounds, quiet machine:

| Drafter | Page total | Decode | Decode speedup | Acceptance |
|---|---|---|---|---|
| none | 184.5 s | 139.9 s | 1.00x | |
| n-gram (`--drafter ngram`) | 172.7 s | 128.0 s | 1.09x | 34% |
| stage-1 head, FP32 drafter KV, confidence 0.45 | 153.9 s | 109.2 s | 1.28x | 76% |
| **stage-2 head, int8 drafter KV, confidence 0.35** | **139.9 s** | **95.1 s** | **1.47x** | 75% |
| stage-2 head, rank-384 vocabulary head | 139.2 s | 94.5 s | 1.48x | 72% |

A separate run on the same pages compares the continued head with its
stage-2 initializer. The stage-3 training set is about 48k pages:

| Drafter | Page total | Decode speedup vs none | Acceptance |
|---|---:|---:|---:|
| none | 181.2 s | 1.00x | |
| stage-2 head | 138.3 s | 1.46x | |
| published stage-3 head | 135.7 s | 1.50x | 76% |

On the calibration pages the stage-3 head changes page total from 182.0 to
176.4 s. Retraining instead (one step-1 epoch, then one seven-step epoch on
the 48k pages) reaches 74.7% / 40.8% first- and second-draft calibration
accuracy, against 75.0% / 44.6% for continued training.

Calibration-page sweeps (16 English pages) behind the defaults: confidence
0.35 (0.25-0.45 and the product-of-probabilities gate are within 2%), at most
4 drafts (6 gains nothing), int8 drafter keys and values (-1.7% decode against
FP32, acceptance unchanged), 12 decode threads (16 is slower with and without
speculation), a 16k draft vocabulary (24k: same acceptance, larger head). A
head trained on next-token prediction only is overconfident on later chain
steps (23% acceptance); the seven-step phase is essential.

### Other modes

The same 16 held-out pages with the published head (`cpu_ab.py --mode`;
tokens identical in every arm). Source: the 2026-09-26 comparison.

| Mode | none | n-gram | head | decode vs none |
|---|---:|---:|---:|---:|
| near-exact (2 rounds) | 334.8 s | 277.4 s | **208.1 s** | 1.87x (acceptance 77%) |
| exact (1 round) | 612.6 s | 531.0 s | **392.5 s** | 1.67x |

The head pays more in the slower modes: a verified draft saves a whole
16-bit or FP32 decode step, while drafting costs the same. The runner
therefore uses the published head in every mode when it sits next to the
model files.

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
| CPU A/B | [cpu_ab.py](cpu_ab.py), [compare_pages.py](compare_pages.py) | alternating rounds, page totals, token identity (`@<exe>` runs another binary per arm); never time while a GPU job runs (it slows CPU decode ~30%); compare_pages.py lists the pages two arms disagree on, with CER against the FP32 reference |
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

- Data was the lever. Stage 1 (4.7k pages) to stage 2 (24.5k) moved
  calibration first-draft accuracy from 62% to 73% and held-out decode from
  1.28x to 1.47x; every architecture change tried (image attention, draft
  vocabulary size, low-rank head, attention window, gating rule) was worth
  2% or less. Stage 3 (48k pages) added 1.9 points first-draft and 5.5
  points second-draft accuracy and cut held-out page totals by another 1.9%;
  the chain skill accumulates over seven-step epochs (continuing beat
  retraining).
- Train on chains. The seven-step phase makes the head's confidence honest on
  its own drafts; without it the confidence gate lets bad chains through.
- CPU speculation is not GPU speculation. A verify row costs about 1.1 ms
  (13% of an 8.8 ms step, mostly attention over the image positions), so
  short confident chains win and trees would not pay. A target step moves
  about 400 MB and evicts the drafter, so a draft step's cost is its bytes
  (cold 0.67 ms with FP32 drafter KV, 0.51 with INT8, 0.29 with INT8 and a
  rank-256 head) rather than its FLOPs.
- Offline estimates run high. The GPU cost model predicted 1.59x decode;
  held-out CPU decode reached 1.47x (failed first draft steps and cold caches
  cost more than modeled), and page totals improve less than decode because
  prefill (about 2.8 s per page, now about a third of a page) is untouched.
- Measure on a quiet host with page totals. A GPU job slows CPU decode by
  about 30%, Windows' `Win32_Processor.LoadPercentage` misreports load (use
  `Get-Counter`), and the WSL and Windows clocks jump (use the binary's
  timings). Every A/B checks token identity against no speculation, which is
  what makes aggressive drafter changes safe.

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

`--draft-head <file>` (selects `--drafter head`), `--drafter ngram|head|both`,
`--draft-confidence 0.35`; experiment knobs `--tune draft-kv=q8|f32`,
`draft-window=N`, `draft-gate=token|path`, `draft-backoff=N`.
