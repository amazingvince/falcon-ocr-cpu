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
| trees + floor + safety net (Phase 1) | 20.26% (−0.63 pt) | 35 | 24.2% |
| **the runner's router (`--max-dimension auto`, Phase 2)** | **20.20% (−0.69 pt)** | 35 | **24.7%** |
| flat 1024 | 21.43% (+0.54 pt) | 76 | 35.1% |
| flat 768 | 22.35% (+1.46 pt) | 74 | 43.5% |
| the plan's category table (true categories) | 22.10% (+1.21 pt) | 57 | 33.9% |
| per-page oracle | 17.66% (−3.23 pt) | 0 | 36.2% |

95% bootstrap over pages (Phase 1): CER change [−1.28, −0.09] pt, time saved
[22.1, 26.4]%; the runner's router: [−1.35, −0.18] pt, [22.6, 26.9]%. The threshold was chosen on these pages; the floor comes from
the plan. Lower resolution often reads *better* (tables, textbooks,
multi-column pages); it loses on dense small print (magazines, newspapers,
tiny text), which the image statistics recognize, and on runaway pages, which
the safety net catches. Final validation belongs to the English gate pages and
a quiet-host timing run (plan phase 3).

## Phase 2: the runner's router

The runner reproduces the router bit for bit. The statistics were redefined
([router_features.py](router_features.py), version 2) so that every one is an
integer count or sum finished by a few float64 operations in a fixed order,
on the page after the model's first resize at 1536, with Pillow's 8-bit
resampling (which src/preprocess.rs already implements exactly). The trees
were retrained on them ([train_trees.py](train_trees.py): the same labels and
settings plus scikit-learn's early stopping, which picks 132 and 84 trees on a
training split: 13,176 nodes, `src/router/trees.json`, 318 KiB). The same
policy on the development set: CER −0.69 pt [−1.35, −0.18], CPU time saved
24.7% [22.6, 26.9] (without early stopping −0.51 pt / 23.7%; all variants are
within noise of each other and of Phase 1).

Checks:

- **Parity.** `src/router` matches the Python specification bit for bit
  (every statistic and both raw scores) on synthetic pages and the decode
  fixtures (`tests/fixtures/router.json`, library tests) and on all 389
  development pages (`falcon-ocr-eval route --statistics`): no statistic and
  no decision differs. Routes: 73 to 768, 185 to 1024, 131 at 1536. Cost:
  11–12 ms per page (median), the page-resolution passes in parallel.
- **End to end** ([check_auto.py](check_auto.py)): `bench --max-dimension
  auto` on the development set gave every page exactly the tokens of the
  fixed run at its final resolution (389 of 389; 2 safety-net reruns, both
  repetition stops), so CER is the simulated −0.69 pt. Timing: over the first
  234 pages (before a game started on the host and slowed prefill 18% and
  decode per token 40%) the measured saving was 26.4% against 24.7% simulated
  from the older fixed runs; routing adds about 50 ms to a routed page
  (decoding once, two first resizes, the statistics). A quiet-host timing
  belongs to Phase 3.

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
| 2 | [router_features.py](router_features.py) | the statistics the runner reproduces bit for bit (the Rust specification) |
| 2 | [train_trees.py](train_trees.py) | version-2 statistics, the shipped trees, their export (`src/router/trees.json`) and the development probabilities |
| 2 | [../../tests/generate_router_fixtures.py](../../tests/generate_router_fixtures.py) | the parity fixtures for `src/router` |
| 2 | [check_auto.py](check_auto.py) | `--max-dimension auto` end to end against the fixed runs |

Labels come from BF16 vLLM serving while the runner is FP32-anchored; the
development set measures the runner itself. Data lives outside the
repository (D:/falcon-draft/router, D:/falcon-draft/router-dev).
