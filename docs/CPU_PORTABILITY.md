# Running fast on a broad base of CPUs

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

| Kernel | Phase | AVX2 | AVX-512 | NEON | Portable |
|---|---|---|---|---|---|
| FP32 GEMV, 1–8 rows (`kernels::linear_with_simd`) | decode | hand-written | hand-written | — (scalar) | scalar |
| Fused GLU (`linear_glu_with_simd`) | decode | via dot kernel | via dot kernel | scalar | scalar |
| W8 GEMV and fused GLU (`attempt::quant`) | decode | hand-written | — | scalar | scalar |
| Compact decode attention (`kernels/attention64.rs`) | decode | hand-written | — | generic loop | generic loop |
| Group-split BF16/Q8 cache attention (`attempt::prefix`) | decode | hand-written | — | generic loop | generic loop |
| Screened head (`head_screen.rs`) | decode | hand-written | — | falls back to full head | falls back to full head |
| Vector exp (`kernels/vexp.rs`) | both | hand-written, platform-exact | — | scalar | scalar |
| Prefill attention tiles (`kernels/prefill64.rs`) | prefill | hand-written | — | gemm crate + scalar | gemm crate + scalar |
| Prefill projections | prefill | gemm crate | gemm crate (needs `x86-v4` feature) | gemm crate (NEON) | gemm crate |

Every NEON cell above currently falls back to scalar code. Before any tuning, the M4 would run decode several times slower than it should.

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

## 5. Known risks

- **Inlining.** Generic kernels only match hand-written speed if every trait method inlines into the `#[target_feature]` wrapper. Keep a per-kernel benchmark and compare the old and new AVX2 paths before deleting any hand-written code.
- **NEON widening cost.** NEON has no single-instruction i8→f32 widening (it takes `sxtl`, `sxtl`, then `scvtf`), so W8 decode may be compute-bound on the M4. Step 4 exists for this reason.
- **Spin team on laptops.** Spinning costs power on laptops and on Apple E-cores. Shorten the idle spin on battery or on macOS, and consider QoS classes so decode stays on P-cores.
- **Building C dependencies.** libjpeg-turbo and Oniguruma build natively with clang on macOS. A cross-build from Windows needs an aarch64 C toolchain, which is why native builds on the M4 come first.
