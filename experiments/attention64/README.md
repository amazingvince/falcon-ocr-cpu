# Fixed-width AVX2 attention experiment

This is one isolated performance candidate motivated by the full-page profile,
not a new numerical implementation or a production default change. It applies
only to expanded-cache attention with one query, head dimension 64 and resolved
AVX2/FMA. Prefill, compact caches, scalar/AVX-512, and other shapes retain the
original code. Rayon still schedules one task per head in the caller's pool.

`patch.py` extracts the original per-head attention loop verbatim, changing only
its dot and AXPY call targets. A feature-qualified function encloses that loop.
Its fixed-width helpers can inline, eliminating the per-key indirect calls and
generic vector-length/tail loops. Dot keeps four eight-lane accumulators, two
32-element iterations, the same pairwise accumulator sum, half sum and two
horizontal adds. AXPY keeps eight ordered eight-lane FMA/store blocks. Tile size,
maximum, denominator, rescale, exp/log, value accumulation and sink order remain
the original expressions. This is an optimization hypothesis until compiled
code and unprofiled end-to-end results are inspected.

Preparation seeds a fresh project from the frozen full-page benchmark's
`source.zip`. Only `src/kernels.rs` differs among that archive's source entries;
the generated candidate and six cfg(test) operator tests are embedded there.
Fixture JSON files needed by unrelated lib-test compilation are separately
bound additions, not benchmark code changes. No live source is modified.

The six operator tests compare every output bit with a private copy of the
unchanged original CPU path. They cover actual 16-head/64-channel dimensions,
image and causal boundaries, 128-key tile tails, 6,544/16,384-key contexts,
extreme logits/sinks, signed zeros, Auto and one-thread dispatch, and unchanged
other shapes/prefill/backends. Selected AVX2 tests fail if unsupported rather
than passing through silent skips. Optional AVX-512 fallback coverage is only
attempted where supported. These tests do not establish GPU parity or speed.

Phases are explicit: prepare, build, operator tests. Building uses a fresh unique
D: target, two compiler jobs, the exact archived Windows wrapper and unchanged
release settings. The benchmark example is captured first with the original
generic capture script (same arguments and captured environment apart from
target directory), then the CLI and lib tests are built from the same source.
No phase runs inference or a benchmark. Parent approval controls each phase.

After operator acceptance and a separately coordinated quiet window, run the
copied CLI with FP32/AVX2/expanded/unpacked/batch1 against the original smoke
fixture, `--max-new-tokens 17`, before any speed measurement. Require all 1,904
saved tensor bytes to match the unchanged CPU smoke trace (SHA256
e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309), same 17
teacher IDs and expected teacher stop `length`. The saved bit-identical CPU
trace is the control; its existing provenance limits remain. This verifies CPU
parity; the original ten GPU intermediate gates remain open without relaxation.

Full-page validation is a separate parent-owned expanded-only bracket:
unchanged/candidate/unchanged, two warmups and three measured repetitions,
16 threads, original full-page input/options, all 1,140 free-generation IDs,
literal text, EOS stop and dimensions exact across every repetition. Timing is
unprofiled; profile percentages rank targets but do not predict speedup. A
candidate that changes outputs or fails any control is rejected. No alternative
arithmetic/order sweep, RMS/W2 intervention combination or automatic promotion.
