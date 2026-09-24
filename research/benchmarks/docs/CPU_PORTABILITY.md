# Running fast on a broad base of CPUs

> **Archived 2026-09-24.** Superseded by [docs/PORTABILITY.md](../../../docs/PORTABILITY.md). The roadmap items below were done in the 2026-09-23/24 work (panel GEMM and the fused norm+RoPE row generic over the SIMD trait, `--backend auto`, libjpeg-turbo as a cargo feature) or dropped; the M4 checklist is `tools/m4_check.sh`.

Status: design plus the first implementation steps (Phase 4, 2026-09-22).

- Quality bar: token selection, not bit-exact math. Selected tokens must be the same or nearly the same as the FP32 reference.
- Targets: x86 AVX2, x86 AVX-512, ARM NEON (Apple M-series first, then Graviton and Snapdragon), and a scalar fallback.

## 1. What limits speed on each class of CPU

Decode (one token at a time) streams every weight and the whole image KV cache once per token. Its floor is therefore **bytes per token ÷ achievable memory bandwidth**. Prefill (the 6.5k-token image, processed once) is **compute-bound**, at about 6.1 TFLOP per full page.

| Machine class | Read bandwidth | FP32 compute (all cores) | What limits decode | What limits prefill |
|---|---:|---:|---|---|
| Ryzen 7950X, DDR5 (measured) | 50–54 GB/s | 1.9–2.3 TFLOP/s | bytes | FMA throughput, exp |
| Typical laptop (DDR4/LPDDR5) | 30–90 GB/s | 0.3–1 TFLOP/s | bytes | FMA throughput |
| Intel Xeon/Core with AVX-512 (2 FMA units) | 60–300 GB/s | 2× AVX2 per core | bytes | FMA; 16-lane tiles pay off |
| Apple M4 / Pro / Max | 120 / 273 / 410–546 GB/s | ~0.6–1.5 TFLOP/s NEON; plus AMX/SME via Accelerate | **dequant compute** at high bandwidth | NEON FMA; matrix unit via Accelerate |
| Graviton 3/4 (NEON, SVE, BF16, I8MM) | 300+ GB/s per socket | many cores | dequant compute | FMA; BF16/I8MM |

Consequences:

1. **Fewer bytes per token is the universal lever for decode.** W8 weights, an INT8 screened head and a BF16/Q8 KV cache help on every machine. They cut the journal page from 1.8 GB to 0.5 GB per token.
2. **On high-bandwidth machines the decode bottleneck moves from DRAM to dequantization compute.**
   - An M4 Max can stream 0.5 GB in about 1 ms. Widening 169M int8 weights to FP32 and multiplying them takes longer than that on NEON.
   - These machines need integer dot products: SDOT/UDOT and I8MM on ARM, VNNI on x86. The activations are quantized per token (W8A8).
   - W8A8 is a lossy step, so it is validated by token agreement like the other lossy formats.
3. **Prefill needs each platform's best matrix engine:**
   - 16-lane AVX-512 tiles on Intel (on Zen 4 they only double-pump).
   - NEON on ARM.
   - Apple's AMX/SME matrix unit, which is reachable only through Accelerate's BLAS.
   - BF16 dot instructions (AVX512-BF16, ARMv8.6 BF16) as a token-checked option.
4. **Thread handoffs must be cheap everywhere.**
   - The decode step runs about 110 short parallel loops. The spin team (`src/team.rs`) replaced Rayon's sleep/wake cycle and cut about 3 ms per token on the 7950X.
   - Dynamic task claiming already tolerates heterogeneous cores (Apple P/E cores, Intel P/E cores).
   - On battery-powered machines the team must park sooner than it does today.

## 2. Code structure: one kernel source, instantiated per ISA

Hand-writing every hot kernel three times (AVX2, AVX-512, NEON) plus a scalar fallback would multiply maintenance by four. Performance-sensitive code will instead be generic over a small internal SIMD layer, `src/simd`.

**The trait.** `trait Simd` exposes an **8-lane logical f32 vector** and only the operations our kernels use:

- load, store, splat, fma, add, mul, max
- int8 and BF16 widening loads
- a fixed-tree horizontal sum
- `exp`

**Implementations:**

| Implementation | Registers used |
|---|---|
| `Avx2` | one `__m256` |
| `Avx512` | one `__m256` with 32 registers and EVEX encoding; optionally 16 lanes for compute-bound tiles |
| `Neon` | two `float32x4_t` |
| `Portable` | `[f32; 8]` with `mul_add` |

**Kernels and entry points.** Kernels are `#[inline(always)]` generic functions. Each ISA gets one small entry wrapper:

- x86 wrappers carry `#[target_feature(enable = "avx2,fma")]` or the AVX-512 equivalent.
- NEON needs no attribute, because it is baseline on aarch64.

Dispatch picks the ISA once per process from runtime feature detection, so a single binary built for the baseline target runs everywhere. No `-C target-cpu=native` is needed.

**Reproducibility.**

- The same lane layout and reduction trees on every ISA make `kernel::<Avx2>`, `kernel::<Neon>` and `kernel::<Portable>` produce **identical bits**.
- Each machine can therefore test its fast path against the portable one bit for bit. Across machines, the same kernel gives the same tokens.
- Deviations are explicit and token-checked: 16-lane AVX-512 tiles, library GEMMs (gemm crate, Accelerate) and integer dot products.

**Math functions.** Use our own vector `exp` rather than the OS math library, because UCRT, glibc and Apple libm differ. The platform-exact `exp` (`src/kernels/vexp.rs`) stays only while bit-for-bit continuity with the current Windows baseline is needed. After that it gives way to one portable fast `exp`.

## 3. Kernel inventory

Status after the overnight pass of 2026-09-23 (see `research/phase4-hillclimb/attempt3/RESULTS-V3.md`). "Generic" means one source, instantiated per ISA.

| Kernel | Phase | AVX2 | AVX-512 | NEON | Portable |
|---|---|---|---|---|---|
| FP32 GEMV, 1–8 rows (`kernels::linear_with_simd`) | decode | hand-written (equal to generic) | hand-written | generic | scalar |
| Fused GLU (`linear_glu_with_simd`) | decode | via dot kernel | via dot kernel | via dot kernel | scalar |
| W8 GEMV and fused GLU (`attempt::quant`) | decode | generic | generic (AVX2 code) | generic | scalar |
| Compact and expanded decode attention (`kernels/attention64.rs`) | decode | generic | generic loop | generic | generic loop |
| Group-split F32/BF16/Q8 cache and compressed generated tail (`attempt::prefix`) | decode | generic | generic | generic | generic loop |
| INT8 head screen (`head_screen.rs`) | decode | generic | generic | generic | falls back to full head |
| Vector exp | both | portable fast exp by default for `run`; platform-exact (`vexp.rs`) for `trace` or `FALCON_OCR_EXP=exact` | same as AVX2 | portable fast exp | portable fast exp |
| Prefill attention tiles (`kernels/prefill64.rs`) | prefill | generic | 16-lane QK/PV (`wide`), bitwise equal to AVX2 | generic | gemm crate + scalar |
| Prefill projections | prefill | gemm crate | gemm crate (the `gemm-avx512` feature measured under 2% on Zen 4) | gemm crate (NEON) | gemm crate |
| Thread defaults (`src/cpu.rs`) | both | prefill on all logical CPUs, decode on one thread per physical core | same | macOS `hw.physicalcpu` | Linux sysfs, else logical count |

- The fast exp is token-identical to the platform exp on all 67 calibration pages, so recognition uses it on every ISA, and NEON, portable and AVX2 now share one exp. Traces keep the platform-exact exp, so the recorded x86 reference hashes still reproduce.
- The GPTQ fast-mode overlay (`<model>/w8-gptq.safetensors`) is plain W8G64 data, so every ISA's W8 kernels use it unchanged. It can be built on one machine and copied to others.

## 4. Roadmap

1. **`src/simd` and generic decode kernels.**
   - Port FP32 GEMV, W8 GEMV and GLU, both decode attentions, the head screen, and a fast portable `exp`.
   - Check that `Avx2` instantiations are bitwise equal to today's AVX2 kernels, so there is no regression. Check that `Portable` equals `Avx2` bitwise.
2. **NEON implementation.**
   - Correctness: `Neon` equals `Portable` bitwise.
   - First runs on the M4, and optionally aarch64 under QEMU in WSL.
   - Tokens on the four full pages compared against the stored FP32 reference tokens.
3. **Platform matrix engines.**
   - Enable gemm's `x86-v4` feature where AVX-512 exists.
   - Add an Accelerate backend for prefill projections on macOS (cargo feature, default on Apple). The same backend could serve the attention tiles as larger matrix products.
4. **Integer dot products for high-bandwidth machines.**
   - Use W8A8 via SDOT/I8MM on ARM and VNNI on x86, with per-token activation scales.
   - It is a separate profile, gated by token agreement.
5. **Auto configuration and reporting.**
   - `--threads auto`, choosing performance cores via OS APIs.
   - A `doctor` bandwidth and FMA probe.
   - Benchmark reports that state achieved GB/s and TFLOP/s as a percentage of the measured floor.
6. **Per-machine validation**, run on each new CPU:
   - unit tests, including each ISA against the portable path;
   - the smoke trace against GPU tolerances;
   - token agreement on the four full pages and the 64-page calibration set against the stored FP32 reference outputs;
   - a speed report against that machine's measured floors.

## 5. Running on Apple Silicon (first M4 session)

What exists now:

- The SIMD layer and every hot kernel have NEON instantiations:
  - FP32 and W8 GEMV and GLU;
  - both decode attentions;
  - the BF16/Q8 split cache;
  - the INT8 head screen;
  - prefill attention tiles;
  - the portable fast exp.
- `--backend auto` selects NEON. The NEON code type-checks for `aarch64-apple-darwin` (checked from Windows).
- Its results are verified on the Mac by `neon == portable` bitwise unit tests.

Prefill projections use the gemm crate's own NEON kernels.

Steps on the Mac:

1. Install the tools: `brew install cmake python`, `pip3 install numpy safetensors`, then rustup. `rust-toolchain.toml` selects 1.94.0.
2. Check out the `phase4-attempt3` branch.
3. Get the model: `python3 scripts/fetch_reference.py --output artifacts/model`. Alternatively, copy `artifacts/model` from the Windows host.
4. Copy these benchmark inputs from the Windows host (all under `artifacts/`, which git ignores):
   - `artifacts/corpus/v3/3f294b5e60a0c2d4/`
   - `artifacts/corpus/smoke/{ebac2ad1cac11a99,a1336e3bc391f255,bc2882dcec9a3e02}/`
   - `artifacts/reference/smoke-fp32/`, which the GPU-token gate and trace tests need
   - optionally `artifacts/phase4/control-default-4096/`, the Windows FP32 tokens, for comparing tokens across machines
5. Run `bash tools/m4_check.sh` (or `QUICK=1` first). It writes `artifacts/portability/<host>-<time>/summary.txt`, with:
   - machine features (DotProd/I8MM/BF16/SME);
   - unit and model gates;
   - per-phase profiles at P-core and all-core thread counts;
   - a bracket with token agreement for FP32 against W8 + BF16/Q8.

What to look for:

- **`neon_*_bitwise` tests pass.** These establish kernel correctness.
- **GPU smoke tokens match.** This checks platform numerics end to end.
- **Decode ms/token against the machine's bandwidth.** If decode is far from bytes per token divided by bandwidth while threads are busy, int8 dequantization is the limit, and roadmap step 4 (SDOT/I8MM) is the fix.
- **Prefill.** If prefill dominates, Accelerate (AMX/SME) for projections and attention tiles is the lever (roadmap step 3).

## 6. Known risks

- **Inlining.** Generic kernels only match hand-written speed if every trait method inlines into the `#[target_feature]` wrapper. Keep a per-kernel benchmark and compare the old and new AVX2 paths before deleting any hand-written code.
- **NEON widening cost.** NEON has no single-instruction i8→f32 widening (it takes `sxtl`, `sxtl`, then `scvtf`), so W8 decode may be compute-bound on the M4. Step 4 exists for this reason.
- **Spin team on laptops.** Spinning costs power on laptops and on Apple E-cores. Shorten the idle spin on battery or on macOS, and consider QoS classes so decode stays on P-cores.
- **Building C dependencies.** libjpeg-turbo and Oniguruma build natively with clang on macOS. A cross-build from Windows needs an aarch64 C toolchain, which is why native builds on the M4 come first.
