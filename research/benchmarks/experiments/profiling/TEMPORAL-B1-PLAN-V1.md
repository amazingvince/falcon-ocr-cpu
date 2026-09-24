# Explicit temporal-cache B1 experiment

Status: prospective protocol and offline tests only. No benchmark or candidate
build has been run by this wrapper's author. Existing accepted wrappers,
benchmark plans, helpers and production sources remain unchanged.

The control is the combined fixed64-attention build with **Compact** cache:
`artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json`, SHA256
`68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0`.
Its executable is `8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac`.
The frozen output baseline is `candidate.json` from the completed
`attention64-compact-fullpage-window-v1` bracket, SHA256
`de250d2ad37db564503762952fa8557c05dde24c48d2f4d659af136a7752fa91`.

The candidate must expose a separate `TemporalCandidate` layout. Commands use
`--cache-layout temporal-candidate`; JSON must report `temporal_candidate`.
Control commands and reports both use `compact`. Existing Compact must remain
an independent path. The wrapper translates the CLI argument explicitly and
passes the JSON spelling to the unchanged shared report validator.

The integration author confirmed this exact source allowlist relative to the
control archive:

- Changed: `src/config.rs`, `src/lib.rs`, `src/model.rs`, `src/kernels.rs`.
- Added: `src/temporal_candidate.rs`, `src/temporal_model_tests.rs`.

No baseline source may be deleted; all other sources, including the benchmark
harness, runner, tokenizer and build helpers, must be identical. Both captures
must use the same recorded Rust/Cargo versions, release arguments and selected
build environment except distinct target directories. The actual new candidate
build, archive and executable are frozen into the plan; no speculative hash is
substituted for a build that does not yet exist.

Before planning a timing run, the parent must accept the copied integration's
functional and numerical equivalence evidence. This wrapper does not turn the
older isolated temporal-cache tests into proof for a changed implementation.
Require the applicable original-control smoke trace, independent outputs,
focused cache edge cases and warmed allocation evidence for the actual new
source/binaries. No GPU numerical tolerance or quality policy changes.

Use `benchmark_temporal_candidate.py --candidate <new-build/build.json>
--output <fresh-plan-directory>` to prepare a plan after qualification. Control
and baseline default to the pinned paths above. Preparation verifies assets
and sources and writes commands' inputs; it does not run the benchmark. A
separate `--plan <plan.json> --plan-sha256 <digest>` validates the plan. Actual
execution additionally requires `--run --quiet-attestation <fresh-check>` and
the parent's explicit release.

The fixed bracket is control-before, candidate, control-after, each a fresh
native Windows process with a 900-second timeout. Each uses two warmups and
three measured recognitions of the original prose page
`3f294b5e60a0c2d4`: FP32, AVX2, 16 threads, unpacked weights, joint B1,
minimum dimension 64, maximum dimension 1536 and output cap 4096. Every measured
output must match all 1,140 frozen IDs, literal text, EOS, prefix count and
prepared dimensions. The bound compiled harness enforces token equality for
every repetition; each literal result is separately exported. Warmup outputs
are not individually exported or audited.

The control drift limit is 5% in absolute value. The candidate target is at
least 5% lower median latency than **each** fresh control. All samples and
both comparisons remain visible when the target fails. Historical timing only
provides context; it is not a fourth process or a substitute for either fresh
control. Failed/interrupted brackets are preserved and never resumed.

The wrapper cleans up and reaps only its own child on timeout, interrupt or a
start-receipt failure. It validates literal result bytes before binding their
hashes and rechecks all saved output hashes after the bracket. Input/build
validation runs outside measured processes. No recorder or global process
cancellation is used.

Report stage timings as wall intervals, not hardware counters or kernel
attribution. B1 takes the runner's single-page path; this does not measure
multirow throughput. Process memory high-water marks include setup and warmup
and are distinct from reserved KV capacity. Any gain is incremental over the
combined-attention compact control; do not add historical percentages. No
default promotion follows from one page and one bracket.

Offline tests:

```text
python -m unittest discover -s experiments/profiling -p test_benchmark_temporal_candidate.py -v
```

Eleven tests passed on native Python during preparation. They cover the real
frozen baseline's report contract, explicit CLI/JSON mapping, OS/architecture,
source additions and forbidden changes, compiler/target checks, fixed-plan
mutations and hash failure, and mocked timeout/interrupt/start-receipt cleanup.
No child process, model load, checkpoint scan or benchmark is run by the tests.
