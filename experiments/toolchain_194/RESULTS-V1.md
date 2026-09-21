# Rust 1.94 upgrade

The project now pins Rust 1.94.0 on Windows and Linux. Cargo's minimum Rust
version is 1.94. The user requested this update after the RTen 0.26 dependency
review identified that requirement. Production dependencies and Cargo.lock
remain unchanged; RTen is confined to the isolated matrix probe.

The installed native Windows and Ubuntu-24.04-CUDA WSL toolchains report
rustc 1.94.0, commit `4a4ef493e3a1488c6e321570238084b38948f6db`, LLVM 21.1.8,
and Cargo 1.94.0. Their hosts are respectively x86_64-pc-windows-msvc and
x86_64-unknown-linux-gnu. Rust 1.92 and its frozen artifacts remain available.

## Validation

| Check | Result |
| --- | --- |
| Native Windows release regression suite | 50 passed, 0 failed, 17 ignored |
| Linux under WSL release regression suite | 50 passed, 0 failed, 17 ignored |
| Explicit saved GPU-reference OCR smoke, Windows | Passed |
| Explicit saved GPU-reference OCR smoke, WSL | Passed |
| Windows FP32 AVX2, four-thread same-prefix trace | All 1,904 tensors bit-identical to saved Rust 1.92 CPU trace |
| Trace output metadata | Same 17 IDs, text, counts, teacher-forcing and finish reason |
| Isolated RTen 0.26 operator probe | Builds; scoped arithmetic checks pass |

The separately enabled OCR smoke tests exercise free generation, EOS, an exact
three-token output limit, and explicit supported AVX2/AVX-512 backends against
the existing GPU reference. No new GPU inference was required. The 17 ignored
tests in the broad suite were not all enabled; the explicit smoke is reported
separately.

The same-prefix trace uses the original expanded cache and unpacked FP32
weights. Its complete safetensors file has the same SHA-256 as the old trace:
`e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309`.
Its teacher-forced finish reason is `length`; free-running EOS is checked by
the separate smoke test.

## Evidence

- `artifacts/builds/toolchain-1.94-windows-v1/build.json` captures the new
  benchmark executable, source archive and compiler identity. Source files
  remained unchanged during that build.
- `artifacts/diagnostics/toolchain-1.94-windows-tests-v1.log` and
  `artifacts/diagnostics/toolchain-1.94-linux-tests-v1.log` contain the full
  regression logs.
- `artifacts/diagnostics/toolchain-1.94-windows-smoke-v2.log` contains the
  native smoke result. The WSL smoke was observed as a successful terminal
  execution (tool chunk 24d99f), without a separate saved log.
- `artifacts/diagnostics/toolchain-1.94-windows-trace-v2/report.json` records
  tensor comparison, log hashes and source/input/binary closure. Its sibling
  `invocation.json` records the exact CLI invocation and executable hash.
  CLI provenance comes from the completed release test build and source
  closure, rather than the separately captured benchmark binary.
- `artifacts/diagnostics/matrix-backends-rust194-v1/build.json` and
  `accuracy-v1/report.json` record the RTen probe. That backend changes FP32
  rounding and has not been qualified in the full model or timed yet.

The initial Windows smoke wrapper invocation failed argument binding before
running tests; its v1 log is retained. Running the already-built test executable
succeeded. The initial trace validation controller encountered a Python 3.10
`hashlib.file_digest` availability error before inference; v1 remains an empty
attempt directory. The corrected controller uses streaming SHA-256 and writes
fresh v2 evidence. An initial inline WSL command also failed shell parsing;
the file-based validation script then completed successfully.

This is compiler-upgrade regression evidence, not a fresh 200-page corpus or
performance qualification. The ten existing intermediate CPU/GPU numerical
mismatches remain open. WSL results do not establish bare-metal Linux speed,
and historical Rust 1.92 benchmark results retain their original compiler label.
