# Resolution router

The resolution router saves 25.1% of CPU time on its 389-page development
set, but fails its held-out English gate: character error rate (CER) rises
by 0.29 percentage points against a 0.25-point limit. It stays opt-in. The
figure-masked comparison below is a post-hoc diagnostic, not a passing gate.

The "Falcon-OCR resolution router plan" defines the experiment. Its labels are
phase 0 for generating agreement data, phase 1 for developing the policy,
phase 2 for the Rust implementation, and phase 3 for held-out validation
and quiet-host timing.

## Development-set results

On 389 English OmniDocBench pages with ground truth (the development set
`reference/router-dev-v1-*`, CPU runner in fast mode with the draft head,
measured CPU time). The policy is described in
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md#resolution-routing).

| Policy | CER against truth | Pages > 2 pt worse | CPU time saved |
|---|---|---|---|
| fixed 1536 | 20.89% | | |
| trees + floor + safety net (Phase 1) | 20.26% (-0.63 pt) | 35 | 24.2% |
| **the runner's router (`--max-dimension auto`, Phase 2)** | **20.20% (-0.69 pt)** | 35 | **24.7%** |
| flat 1024 | 21.43% (+0.54 pt) | 76 | 35.1% |
| flat 768 | 22.35% (+1.46 pt) | 74 | 43.5% |
| the plan's category table (true categories) | 22.10% (+1.21 pt) | 57 | 33.9% |
| per-page oracle | 17.66% (-3.23 pt) | 0 | 36.2% |

95% bootstrap over pages for the initial policy (Phase 1): CER change
[-1.28, -0.09] pt, time saved [22.1, 26.4]%; the runner's router:
[-1.35, -0.18] pt, [22.6, 26.9]%. The threshold was chosen on these pages;
the floor comes from the plan. Lower resolution often reads better on
tables, textbooks and multi-column pages. It loses on dense small print
(magazines, newspapers, tiny text), which the image statistics recognize,
and on runaway pages, which the safety net catches. Held-out validation
and quiet-host timing are reported separately below.

## Runner parity and timing

The runner reproduces the Python router bit for bit. Each statistic
([router_features.py](router_features.py), version 2) is an integer count or
sum finished by a few float64 operations in a fixed order, on the page
after the model's first resize at 1536, with Pillow's 8-bit resampling
(which `src/preprocess/resample.rs` implements exactly).

The trees use these statistics ([train_trees.py](train_trees.py)), the same
labels and settings as the initial policy, and scikit-learn's early
stopping, which picks 132 and 84 trees on a training split: 13,176 nodes,
`src/router/trees.json`, 318 KiB. On the development set the policy gives
CER -0.69 pt [-1.35, -0.18] and CPU time saved 24.7% [22.6, 26.9]. Without
early stopping the result is -0.51 pt / 23.7%; the variants are within
noise of each other and of Phase 1.

The implementation and timing checks are:

- **Parity.** `src/router` matches the Python specification bit for bit
  (every statistic and both raw scores) on synthetic pages and the decode
  fixtures (`tests/fixtures/router.json`, library tests) and on all 389
  development pages (`falcon-ocr-eval route --statistics`): no statistic and
  no decision differs. Routes: 73 to 768, 185 to 1024, 131 at 1536. Cost:
  11-12 ms per page (median), the page-resolution passes in parallel.
- **End to end** ([check_auto.py](check_auto.py)): `bench --max-dimension
  auto` on the development set gave every page exactly the tokens of the
  fixed run at its final resolution (389 of 389; 2 safety-net reruns, both
  repetition stops), so CER is the simulated -0.69 pt. Over the first 234
  pages, before host load slowed prefill 18% and decode per token 40%,
  the measured saving was 26.4% against 24.7% simulated from the older
  fixed runs. Routing adds about 50 ms to a routed page
  (decoding once, two first resizes, the statistics).
- **Quiet-host timing** (Phase 3): the 389 pages in four chunks, the two
  settings alternated per chunk (1536, auto, auto, 1536, ...), same binary:
  1536 58.0 min, auto 43.5 min, **25.1% of the CPU time saved** (chunks
  16-32%); routed pages 38.5% faster (3.4 s each); tokens identical to the
  earlier runs in both settings on all 389 pages.

## Held-out English gate (failed)

The 118 English gate pages, `--max-dimension auto` against fast mode at 1536,
with the English gate's criteria registered before the run
(`reference/router-english-gate-v1-budget.json`; results in
`reference/router-english-gate-v1-results.json`):

| | fast at 1536 | auto | change | limit |
|---|---|---|---|---|
| overall (117 EOS pages) | 13.38% | 13.66% | +0.29 pt | +0.25 **FAIL** |
| formulas (25) | 19.10% | 20.00% | +0.90 pt | +1.0 pass |
| multi-column (25) | 7.86% | 7.32% | -0.53 pt | pass |
| ordinary (19) | 25.84% | 27.30% | +1.45 pt | +1.0 **FAIL** |
| slides (15) | 7.54% | 14.19% | +6.66 pt | +1.0 **FAIL** |
| tables (24) | 9.72% | 9.61% | -0.11 pt | pass |

Routes 23 / 54 / 41 (768 / 1024 / 1536; the development set's mix), one
safety-net rerun, no repetition stop. Two pages carry about two thirds of the
overall change: on a slide and on a chemistry abstract the routed run
transcribes a figure (a diagram, a reaction scheme's substituent list) as an
HTML table that the ground truth leaves out. That is the artifact that also
failed fast mode's English gate, but the verdict stands as registered. The
real losses match the development set's: a sidebar dropped at 768 and
weaker LaTeX on formula pages at 1024.

Normalizing whitespace, quotes and markdown markers leaves +0.24 pt, so
the development set's -0.69 pt did not replicate. The development-set
gains come from pages where 1536 itself fails badly (academic tables, long
formula pages cut short), and the fresh pages have few of those. Fast at
1536 reads them at 13.4% CER against 20.9% on the development set.

Consequence: the router stays opt-in, not the fast-mode default, and reads as
about a quarter less CPU time for about +0.3 pt CER on printed English. The
English gate pages are spent for router work; with the development set they
cover every English OmniDocBench page with ground truth outside the corpora,
so a next router version needs new held-out pages.

**Figure-masked diagnostic (post hoc).** Painting white every figure region
the ground truth leaves out, on every system's input
(`research/corpus-qualification/scripts/mask_excluded.py`; 35 of the 118
pages, median 17.5% of the page), and rerunning the 34 changed pages:

| | fast at 1536 | auto | change | limit |
|---|---|---|---|---|
| overall | 12.47% | 12.53% | +0.06 pt | +0.25 pass |
| formulas | 18.80% | 19.66% | +0.86 pt | +1.0 pass |
| multi-column | 7.80% | 7.37% | -0.43 pt | pass |
| ordinary | 17.15% | 17.38% | +0.23 pt | pass |
| slides | 7.23% | 7.04% | -0.18 pt | pass |
| tables | 9.74% | 9.48% | -0.26 pt | pass |

The figure artifact explains the failure, and the router is about neutral
on fresh printed English while saving about a quarter of the CPU time.
Formula pages at 1024 are its weakest spot. This diagnostic uses spent
pages and was designed after the failure, so it does not replace the
recorded verdict. The same masking takes fast mode's own English gate from
+2.48 pt to +0.02 pt on ordinary pages. Future gates use figure-masked
pages from the start.

Source: `reference/english-gate-v1-masked-results.json`.

## Where it wins and loses

[analyze_losses.py](analyze_losses.py) on the development set (the router's
own pages, so for understanding only). The routed pages gain 15,919 error
characters and lose 6,357 against 1536.

- **Gains** are mostly pages that 1536 itself gets badly wrong, where the
  lower resolution reads the whole page: academic tables (31% → 3.6% CER,
  19% → 2.9%, 14% → 0.7%) and long formula pages that 1536 cut short (an
  exam paper at 57% → 5.7%, 1,852 → 3,858 characters). Tables net -2.7 pt
  over their 54 pages.
- **Losses** are spread thin (the largest 15 hold 58% of them) and are not
  near the decision boundary: pages scored 0-0.5 above it gain as much as
  pages scored 0.5-1.0, so a stricter threshold would mostly give back gains.
  The largest real loss is small magazine print at 1024 (15-px lines, 10 px
  there), where one page drops a sidebar.

  Much of the rest is formatting: curly for straight apostrophes, line
  breaks, `#` markers, and two slides
  whose ground truth is 41 characters while the model writes the slide's
  table as HTML (pretty-printed at 768). Normalizing whitespace, quotes and
  markdown markers leaves the result at -0.63 pt (19.75% → 19.12%) and pages
  more than 2 pt worse at 27 instead of 35.
- **Category** "other" (magazines, newspapers, textbooks) nets +0.09 pt;
  every other category improves. A confidence probe at the routed resolution
  (AUC 0.85 in Phase 1) is the natural way to catch the small-print pages;
  any such change needs pages the router has not seen.

## Findings from model comparisons

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
  -0.50 pt / 23.3% against the trees' deterministic -0.63 pt / 24.2%.
- **The model's own confidence is the strongest signal.** The mean top-1
  log-probability of the first 64 tokens at 768 predicts routability with
  AUC 0.83 (0.85 with the image statistics); a probe-and-restart cascade is
  the next upgrade if the image router plateaus.

## Files

| Record phase | File | Notes |
|---|---|---|
| 0 | [sample_pages.py](sample_pages.py) | 10,691 draft-head stage-3 pages (English, 1536-px transcripts exist), capped per source, resized with the model's own `resize_image_if_necessary` to 1024 and 768 (PNG) |
| 0 | [../vllm-serving/scripts/request_vllm_draftgen.py](../vllm-serving/scripts/request_vllm_draftgen.py) | transcripts from the official vLLM image, both GPUs |
| 0 | [analyze_agreement.py](analyze_agreement.py) | agreement with 1536 by source and type; the go rule (fixed in advance): go, 70.9% of ordinary documents within 2% at 1024 or 768 |
| 1 | [build_dataset.py](build_dataset.py) | 26 image statistics (including the resampling loss at 1024 and 768), a 384-px thumbnail and four native crops per page |
| 1 | [train_router.py](train_router.py), [cascade_eval.py](cascade_eval.py) | trees and CNNs on the agreement labels; the confidence cascade |
| 1 | [noise_aware.py](noise_aware.py) | the second 1536 run, stable pages and noise-aware labels |
| 1 | [select_dev_pages.py](select_dev_pages.py) | the 389-page development set (every English OmniDocBench page outside corpus v1-v3 and the English gate) |
| 1 | [dev_image_probs.py](dev_image_probs.py), [dev_cnn_probs.py](dev_cnn_probs.py) | router probabilities for the development pages (models trained on the noise-aware GPU labels) |
| 1 | [dev_policy.py](dev_policy.py) | CER against truth, stop changes and CPU time per policy |
| 2 | [router_features.py](router_features.py) | the statistics the runner reproduces bit for bit (the Rust specification) |
| 2 | [train_trees.py](train_trees.py) | version-2 statistics, the shipped trees, their export (`src/router/trees.json`) and the development probabilities |
| 2 | [../../tests/generate_router_fixtures.py](../../tests/generate_router_fixtures.py) | the parity fixtures for `src/router` |
| 2 | [check_auto.py](check_auto.py) | `--max-dimension auto` end to end against the fixed runs |
| 2 | [analyze_losses.py](analyze_losses.py) | where the router gains and loses on the development set |

Labels come from BF16 vLLM serving while the runner is FP32-anchored; the
development set measures the runner itself. Data lives outside the
repository (D:/falcon-draft/router, D:/falcon-draft/router-dev).
