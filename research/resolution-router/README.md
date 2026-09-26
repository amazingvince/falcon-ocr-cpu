# resolution-router

A per-page choice of the maximum image dimension (768, 1024 or 1536) for
fast mode: most English pages read as well at lower resolution, and prefill
and decode both scale with the image token count. The plan (phases, gates,
risks) is the "Falcon-OCR resolution router plan" document; this folder holds
its measurements.

## Result so far

On 389 English OmniDocBench pages with ground truth (the development set
`reference/router-dev-v1-*`, CPU runner in fast mode with the draft head,
measured CPU time), the chosen policy (gradient-boosted trees on 26 image
statistics, route when p >= 0.5, an 8-px median text-line floor at the target
resolution, rerun at 1536 when a routed page loops or hits the cap):

| Policy | CER against truth | Pages > 2 pt worse | CPU time saved |
|---|---|---|---|
| 1536 (today) | 20.89% | | |
| **trees + floor + safety net** | **20.26% (−0.63 pt)** | 35 | **24.2%** |
| flat 1024 | 21.43% (+0.54 pt) | 76 | 35.1% |
| flat 768 | 22.35% (+1.46 pt) | 74 | 43.5% |
| the plan's category table (true categories) | 22.10% (+1.21 pt) | 57 | 33.9% |
| per-page oracle | 17.66% (−3.23 pt) | 0 | 36.2% |

95% bootstrap over pages: CER change [−1.28, −0.09] pt, time saved
[22.1, 26.4]%. The threshold was chosen on these pages; the floor comes from
the plan. Lower resolution often reads *better* (tables, textbooks,
multi-column pages); it loses on dense small print (magazines, newspapers,
tiny text), which the image statistics recognize, and on runaway pages, which
the safety net catches. Final validation belongs to the English gate pages and
a quiet-host timing run (plan phase 3).

## What was learned on the way

- **Serving is not deterministic.** Two 1536 runs of the same page through
  the official vLLM image (batched BF16) differ by more than 2% on 23% of
  pages (prose 15%, newspapers 95%), so "the text stays within 2% of the
  1536 transcript" labels are about a quarter noise. Noise-aware labels
  (lower resolution no further from 1536 than a rerun is, trained on pages
  whose 1536 runs agree) and, above all, ground truth are needed.
- **Agreement is the wrong target.** On agreement labels every router
  captured under half of the oracle's saving; against ground truth the same
  trees keep CER while saving a quarter of CPU time.
- **Trees beat CNNs.** A 384-px thumbnail CNN and a native-crop CNN (both
  ~120M multiply-adds) did no better: over three seeds the crop CNN averaged
  −0.50 pt / 23.3% against the trees' deterministic −0.63 pt / 24.2%.
- **The model's own confidence is the strongest signal.** The mean top-1
  log-probability of the first 64 tokens at 768 predicts routability with
  AUC 0.83 (0.85 with the image statistics); a probe-and-restart cascade is
  the next upgrade if the image router plateaus.

## Files

| Phase | File | Notes |
|---|---|---|
| 0 | [sample_pages.py](sample_pages.py) | 10,691 draft-head stage-3 pages (English, 1536-px transcripts exist), capped per source, resized with the model's own `resize_image_if_necessary` to 1024 and 768 (PNG) |
| 0 | [../vllm-serving/scripts/request_vllm_draftgen.py](../vllm-serving/scripts/request_vllm_draftgen.py) | transcripts from the official vLLM image, both GPUs |
| 0 | [analyze_agreement.py](analyze_agreement.py) | agreement with 1536 by source and type; the go rule (fixed in advance): go, 70.9% of ordinary documents within 2% at 1024 or 768 |
| 1 | [build_dataset.py](build_dataset.py) | 26 image statistics (including the resampling loss at 1024 and 768), a 384-px thumbnail and four native crops per page |
| 1 | [train_router.py](train_router.py), [cascade_eval.py](cascade_eval.py) | trees and CNNs on the agreement labels; the confidence cascade |
| 1 | [noise_aware.py](noise_aware.py) | the second 1536 run, stable pages and noise-aware labels |
| 1 | [select_dev_pages.py](select_dev_pages.py) | the 389-page development set (every English OmniDocBench page outside corpus v1–v3 and the English gate) |
| 1 | [dev_image_probs.py](dev_image_probs.py), [dev_cnn_probs.py](dev_cnn_probs.py) | router probabilities for the development pages (models trained on the noise-aware GPU labels) |
| 1 | [dev_policy.py](dev_policy.py) | CER against truth, stop changes and CPU time per policy |

Labels come from BF16 vLLM serving while the runner is FP32-anchored; the
development set measures the runner itself. Data lives outside the
repository (D:/falcon-draft/router, D:/falcon-draft/router-dev).
