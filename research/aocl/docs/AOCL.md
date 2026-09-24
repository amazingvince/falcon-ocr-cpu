# Isolated AOCL FP32 probe

AOCL-DLP is an operator experiment; the production runner has no AOCL dependency.
The probe uses revision `abb63d85ed7a6d559ea42b5db648e2585ac9ecb8`
(`AOCL-202609W02`) and the classic `aocl_gemm_f32f32f32of32` C API.
Its contiguous row-major call computes `A[m,k] * W[n,k]^T`, using `md_t=int64_t`,
normal-memory flags, alpha one, beta zero, and null metadata.

The checked loader verifies the supplied library SHA-256 before loading code.
The safe entry point rejects invalid/overflowing shapes and validates all slice
lengths. A nonsymmetric rectangular known answer checks orientation, strides,
alpha/beta handling, and beta-zero behavior with NaN contents in the destination.
The Windows loader retains a file handle that denies writing or replacing the
DLL during the probe. Linux retains its file handle but has no mandatory write
lock; keep the explicitly selected library immutable while it runs.

The experiment covers all seven captured GPU cases and four projections per
case: QKV, attention output, W13, and W2. It verifies the complete model weight
and config hashes plus the independently exported fixture and manifest hashes.
Both AOCL and current Rust kernels receive the same captured GPU input for each
operator. Rust uses a four-thread Rayon pool; the AVX2 selection controls small
row counts, while the `gemm` crate independently dispatches larger prefill
matrices. The AOCL library is built with OpenMP off and called serially.

## Build provenance

The first native Windows DLL was built manually using CMake 4.2.1 and MSVC
19.44.35222.0. Its SHA-256 is
`f2ce2248710bdcc5feed97f73eead203733c8ebc6c1d759aa18ececd4e4cab1e`.
[The preserved Windows build report](../../../reference/aocl-windows-build-v1.json)
explicitly describes its source archive as captured **after** the build. It is
not evidence that a startup source snapshot was checked during that initial
compilation. The historical Windows v1 operator report remains unchanged.

The WSL Linux library was built using the scripted workflow, GCC/G++ 13.3.0,
existing local CMake 3.31.10, two build jobs, and no OpenMP. Its SHA-256 is
`fb1b99d98da3311988370ca946a501fa5bef8d62d3dcd59e4e21b221a501a6e2`.
[Linux build provenance](../../../reference/aocl-linux-build-v1.json) records all
641 source hashes before configuration and after compilation; both sets match
the native Windows after-build source snapshot exactly. ELF dependencies contain
no OpenMP runtime. The [Linux loader check](../../../reference/aocl-linux-loader-validation-v1.json)
also verifies rejection of an incorrect library digest without producing output.

`research/aocl/scripts/build_aocl.py` now records source hashes before configuration and again
after compilation, the chosen C/C++ compiler identity and executable hashes,
CMake identity, configure/build commands and logs, effective options, library
hashes, and the compile-command database where supported. It refuses an existing
build directory and a dirty or wrong source revision. It does not install the
library or change system dependencies. Use a new build directory for each run.

For the existing Windows checkout shared with WSL, Windows Git's system setting
is `core.autocrlf=true`. WSL Git otherwise interprets all 641 CRLF files as changed.
The explicit `--git-autocrlf true` option applies only to child Git commands and
is recorded in provenance; it changes neither source bytes nor repository/global
Git configuration. Do not use it to ignore actual source changes.

## Reproduction

Run commands from the repository root. Prerequisites from the pinned upstream
`BUILD.md` are CMake 3.26+, a C11/C++17 compiler, and a build tool. These commands
use existing local CMake, GCC/Make in WSL, and the installed MSVC toolchain on
Windows. They do not install OpenMP or global packages.

```powershell
python research/aocl/scripts/build_aocl.py --source artifacts/aocl-dlp `
  --build artifacts/aocl-dlp-build-windows-reproduction --jobs 2 `
  --cmake "C:/Program Files/CMake/bin/cmake.exe"
python research/benchmarks/scripts/capture_rust_build.py --example aocl_probe `
  --output artifacts/builds/aocl-probe-windows-reproduction --jobs 2
```

For WSL, keep the native library build and Cargo target on the existing ext4
filesystem. Adjust these machine-local tool paths when reproducing elsewhere:

```bash
python3 research/aocl/scripts/build_aocl.py --source artifacts/aocl-dlp \
  --build /home/amazi/falcon-ocr-rust-reference/aocl-build-v1 --jobs 2 \
  --cmake /home/amazi/falcon-ocr-rust-reference/build-tools/cmake-3.31.10-linux-x86_64/bin/cmake \
  --git-autocrlf true

PATH="$HOME/.cargo/bin:$PATH" \
FOCR_TOOL_DIR=/home/amazi/falcon-ocr-rust-reference/build-tools \
python3 research/benchmarks/scripts/capture_rust_build.py --example aocl_probe --jobs 2 \
  --cargo-target-dir /home/amazi/falcon-ocr-rust-reference/rust-target \
  --output /home/amazi/falcon-ocr-rust-reference/aocl-probe-build-v1
```

Pass the exact library path and SHA-256 from the newly completed build manifest:

```bash
/home/amazi/falcon-ocr-rust-reference/aocl-probe-build-v1/aocl_probe \
  --library /absolute/path/to/libaocl-dlp.so \
  --expected-library-sha256 YOUR_BUILD_LIBRARY_SHA256 \
  --git-autocrlf true \
  --output reference/aocl-linux-fp32-operator-probe-v1.json
```

On Windows, invoke the captured `.exe` with its DLL path and digest; omit the
WSL line-ending override. Every probe output must be a new file. The report
includes maximum/RMS error, differing element counts, and exact output-array
digests in contiguous little-endian FP32 format. It records no elapsed timings.

Compare the captured outputs with:

```bash
python3 research/aocl/scripts/compare_aocl_probes.py \
  --windows reference/aocl-windows-fp32-operator-probe-v2.json \
  --linux reference/aocl-linux-fp32-operator-probe-v1.json \
  --output reference/aocl-windows-linux-fp32-comparison-new.json
```

## Interpretation

The first Windows run passed all known-answer and shape checks. Against the same
GPU reference, AOCL's maximum error was lower/equal/higher than current Rust on
11/7/10 of the 28 projections; RMS error was lower/equal/higher on 9/5/14.
Every tested prefill W2 improved, and every prefill attention output matched the
existing Rust values. Every tested prefill W13 lost Rust's exact GPU match.
Decode attention output and W2 generally worsened.

[Windows v1 results](../../../reference/aocl-fp32-operator-probe-v1.json) and
[validation](../../../reference/aocl-fp32-operator-probe-v1-validation.json) are isolated
diagnostics. No result promotes AOCL into the runner, modifies frozen numerical
tolerances, establishes full-model parity, or measures throughput. WSL results
are functional Linux evidence and cannot establish bare-metal Linux performance.

The [Windows v2](../../../reference/aocl-windows-fp32-operator-probe-v2.json) and
[Linux v1](../../../reference/aocl-linux-fp32-operator-probe-v1.json) runs add exact array
digests while retaining identical GPU input/weight/reference contracts. Both Rust
probe binaries have startup and after-build source captures; the Linux capture
is recorded in [its build manifest](../../../reference/aocl-linux-probe-build-v1.json).
The [cross-platform comparison](../../../reference/aocl-windows-linux-fp32-comparison-v1.json)
found **28/28 AOCL outputs and 28/28 Rust outputs bit-identical** between Windows
and WSL Linux. Thus the isolated numerical tradeoffs above apply to both tested
builds; this says nothing about their relative throughput.

The subsequent [whole-model same-prefix intervention](../../../reference/aocl-w2-intervention-summary-v1.json)
used AOCL only for prefill W2 and preserved Rust elsewhere. On the synthetic
17-decision trace, failing tensors under the unchanged policy increased from
**10 to 47 out of 1,904**, although all 17 argmax decisions still matched the GPU.
The production before/after control traces remained byte-identical. AOCL was
not promoted. This experiment was not a free-running corpus evaluation, and the
exact Windows/Linux operator outputs provide no reason to repeat it on Linux.
