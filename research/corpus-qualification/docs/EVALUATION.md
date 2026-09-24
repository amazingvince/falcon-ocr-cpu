# Document quality evaluation

The corrected, frozen 200-page v3 corpus now has complete strict FP32 GPU and
Windows Rust results. All 275,903 token IDs, literal texts and stopping reasons
match. The pinned official component evaluator completed every page in both runs
and produced identical result JSON. See
[`official-components-v3-fp32.json`](../../../reference/official-components-v3-fp32.json)
and the separate [quality regression accounting](QUALITY.md).

Twelve CPU texts were replayed from unchanged saved token IDs through the
corrected decoder. Original inference records and timings are preserved. The
corpus contains 181 EOS stops and 19 explicit 4,096-token length stops; truncated
outputs remain in every applicable score. These results do not close the
intermediate-tensor numerical gate or establish performance.

The evaluator is pinned to OmniDocBench's v1_5 revision
`59b103c4b47d3a01fada83491585d6512a40c0bc`. It runs in an isolated Python 3.10.20
environment, with exact packages in `requirements/evaluation-resolved.txt`.
`research/corpus-qualification/scripts/setup_evaluation.sh` synchronizes that environment. It does not change
the GPU reference environment.

The upstream matcher receives raw model Markdown text without output cleanup.
Text, display-formula, table and reading-order normalized edit distances are
computed, as are table TEDS and structure-only TEDS. `CDM_plain` exports matched
formula pairs; it does **not** compute rendered formula similarity. No official
Overall score is claimed.

| Component | GPU and Rust score | Coverage | Direction |
|---|---:|---|---|
| Text edit distance, page mean | 0.171931 | 2,583 samples / 194 pages | Lower is better |
| Formula edit distance, page mean | 0.283966 | 214 samples / 44 pages | Lower is better |
| Table edit distance, page mean | 0.191705 | 89 tables / 62 pages | Lower is better |
| Table TEDS, table mean | 0.669869 | 89 tables / 62 pages | Higher is better |
| Table structure-only TEDS, table mean | 0.750259 | 89 tables / 62 pages | Higher is better |
| Reading-order edit distance, page mean | 0.095677 | 194 pages | Lower is better |

All 200 pages entered and completed matching. Component coverage differs because
not every page yields a sample for every component; missing scores are not zeros.
The report retains exact scores, sample counts and per-category results.

Two source `truncated` relations on two paper pages refer to nonexistent
annotation IDs and crash the upstream matcher. An explicit preparation flag drops
only these invalid links from evaluator input. All original text/layout blocks
and frozen source annotations remain unchanged. Every removed relation appears
in the provenance and summary. These are adapted-input component results, not
unmodified benchmark scores. Both runtimes receive identical adapted inputs.

Upstream TEDS is unclamped and can be negative. One table scores
`-0.14607442454664676`; it remains in the 89-table mean. The reporting validator
accepts finite values in the justified `[-1, 1]` range for both TEDS metrics,
without changing metric arithmetic, aggregation or equality tolerances. See the
[exact replay diagnosis](../../../reference/official-teds-negative-diagnosis-v1.json)
and [independent reporting review](../../../reference/official-teds-reporting-independent-review-v1.json).

The [24-page v1 component report](../../../reference/official-components-v1-fp32.json)
remains unchanged historical diagnostic evidence. Its scores and annotation
adaptations belong to that earlier selection, not the completed v3 corpus.

Source images, ground truth and full matcher outputs stay in ignored artifacts.
OmniDocBench data is for research/noncommercial use and is not included in runner
releases. The public report stores scores and provenance rather than source text.

`research/corpus-qualification/scripts/prepare_corpus.py` materializes v3 from its reviewed, hash-verified source
images and records Pillow/script/image/annotation hashes. The original v2
preparation implementation is preserved as `research/corpus-qualification/scripts/prepare_corpus_v2.py` for
replaying the earlier frozen locks. V1/v2 data and reports are retained unchanged;
they are not silently reclassified into v3.

## Checked v3 execution

The completed run uses `artifacts/evaluation/v3-checked-fp32-v2`. Use the guarded
workflow below to reproduce it in a new output directory. The wrapper requires
all 200 successful records and checks both inference contracts,
model/image hashes, page IDs, token stops and output budgets before creating
evaluator inputs. A missing or failed page stops preparation; it is not skipped.
GPU records also pass the shared strict prefix/cache, per-token greedy-decision,
finite-logit/timing and stopping checks. Explicit teacher-forcing or replay
markers at run, configuration or page level are rejected. Legitimate historical
records may omit optional processed dimensions and startup semantic fields;
those absent fields are not fabricated.

Run from the project root in Linux/WSL. Set `CPU_OUTPUT` to the completed original
CPU run or its validated saved-token text replay directory. A replay is labeled
as derived postprocessing and must preserve original tokens, stops, timings and
all other result fields; it does not represent another inference run.

```bash
CPU_OUTPUT=artifacts/cpu/corpus-v3-fp32-4096-redecoded-v2-snapshot-200
/home/amazi/falcon-ocr-evaluation/.venv/bin/python \
  research/corpus-qualification/scripts/run_official_evaluation.py \
  --manifest reference/corpus-v3-evaluation-lock.json \
  --cpu "$CPU_OUTPUT" \
  --gpu artifacts/reference/corpus-v3-fp32-4096 \
  --output-root artifacts/evaluation/v3-fp32-reproduction \
  --report reference/official-components-v3-fp32-reproduction.json \
  --drop-dangling-truncated-relations --preflight-only
```

When preflight reports `inputs_complete`, repeat the command without
`--preflight-only`. The output root and report must be new or empty; the wrapper
refuses stale evaluator results and never overwrites the historical v1 report.
The explicit annotation-adaptation flag retains the established policy and
records every removed dangling relation. No source lock is changed.

On this Windows checkout, WSL Git initially treated CRLF source files as modified.
The completed run used process-local `GIT_CONFIG_COUNT=1`,
`GIT_CONFIG_KEY_0=core.autocrlf`, `GIT_CONFIG_VALUE_0=true` after verifying every
materialized tracked evaluator file against the pinned Git blob. All 122 raw
file hashes remained unchanged after scoring; 84 sparse paths remained absent.
Strict revision and clean-tree checks stayed enabled. Native Linux checkouts
may not need this interop setting. See the
[before-run receipt](../../../reference/official-evaluator-git-interop-v2-start.json)
and [completion receipt](../../../reference/official-evaluator-git-interop-v2-completion.json).
Earlier failed invocations remain preserved in the separate v1 output root.

The checked runner pins the evaluator revision, Python version and resolved
package versions. It records entry and normal completion for every page without
changing matching inputs, return values or metric arithmetic. This is necessary
because the pinned upstream matcher can call `sys.exit()` with a zero exit code
after printing a traceback, can silently skip missing predictions, and can print
serialization/TEDS errors without raising. A successful process exit alone is
insufficient. `execution-audit.json` must account for every selected page and
contain no detected errors.

Post-run validation requires every expected result file, verifies formula-pair
and table counts, rejects unknown/missing per-page score keys, and recomputes
all edit/TEDS aggregates from the saved component samples. Components with no
samples retain an explicit absence and upstream `NaN` sentinel; they are not
assigned a zero score. The report derives its scope, category counts, inference
origin and coverage from the manifest and run provenance. It still does not
claim rendered CDM, official Overall, performance, or automatic accuracy-gate
acceptance.

When attaching these component scores to quality accounting, the prepared
provenance must identify the exact source run and each prediction record used
by that comparison, including paths, hashes, page IDs/categories and literal
UTF-8 prediction hashes. Matching configurations or a common equality flag are
insufficient. Audits, metric outputs, prepared ground truth/configuration, logs
and prediction files are rechecked in the quality report's source window.

To recheck newly prepared and audited directories without rerunning the
evaluator, use a new report path:

```bash
python research/corpus-qualification/scripts/compare_official_evaluation.py \
  --manifest reference/corpus-v3-evaluation-lock.json \
  --gpu artifacts/evaluation/v3-checked-fp32-v2/gpu \
  --cpu artifacts/evaluation/v3-checked-fp32-v2/cpu \
  --output reference/official-components-v3-fp32-recheck.json
```

These commands use the new preparation/audit schema. The existing v1 input
directories and durable scores remain historical evidence and are not rewritten
to pretend they were produced by the new audited workflow.
