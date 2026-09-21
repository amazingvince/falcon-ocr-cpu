# Warm single-page FP32 decode: source map

Source-only inspection of the current production path. No profiler capture,
build, test, tensor read, or timing was performed for this note. The entries
below identify places to attribute future samples, not measured hotspots.
Use the profiled binary's source/PDB identity when resolving actual symbols;
ThinLTO (`Cargo.toml`: `lto="thin"`, `codegen-units=1`, `debug=1`) may inline,
merge or rename frames.

## Entry and measurement boundary

`Runner::recognize_file` / `recognize_with_trace` →
`run_prepared_scoped` → one owned Rayon `ThreadPool::install` →
`run_prepared` → `Model::embed` and `Model::forward` per generated next token.
The pool is constructed once in `Runner::new`, not once per token/operator.
Batch size one falls back to this same single-image path; it does not call
`Model::decode_batch` or the phase-packed batch kernel.

`runner.rs:457` calls `trace.decode_start()` immediately before the generation
loop; `decode_end()` is after it. Prefill produces the first output decision
before this interval. An N-token completed output therefore normally contains
N−1 decode forward calls; the final EOS/budget token does not trigger another
forward. The interval includes token pushes, stop checks, embedding, forward
and argmax; final text decoding is outside it. Image decode/resize/patching,
image projection, transformer prefill, model loading/packing and pool creation
are also outside warmed decode. A whole-process CPU trace still includes them.
Use phase markers and all worker stacks, not just the calling thread.

`NoTrace::enabled()` is false (`trace.rs:17`), so tensor copies and per-token
formatted phase names are skipped. Hidden vectors, logits, RoPE buffers and KV
capacity are reused. The existing ignored allocation test
`tests/decode_allocations.rs:69` covers the warmed decode interval; this note
does not rerun it or infer that complete recognition allocates nothing.

## Symbols, operations and sizes

The model has 22 layers, width 768, 16 query heads, 8 original KV heads,
head width 64, FFN width 2304 and vocabulary 65536 (`config.json`).

| Attribution / likely Rust symbol | Current source | Work in one next-token forward |
|---|---|---|
| `Model::embed` / copy routines | `model.rs:212` | Copy one 768-value embedding. No image projection in decode. |
| `Model::forward`, `Workspace::resize`, `rotary_factors` | `model.rs:267,929,958` | Reuse buffers; fetch precomputed temporal factors; build head-specific 32-pair factors once before the layer loop. Nonvisual spatial coordinates are NaN, yielding angle zero. |
| `kernels::rms_norm`, `sum_squares_pairwise` | `kernels.rs:189,226` | Per layer: one width-768 attention RMS, sixteen width-64 Q rows, sixteen width-64 expanded K rows, one width-768 FFN RMS. Final learned width-768 RMS after all layers. |
| `linear_with_simd`, its Rayon closure, `x86::dot_avx2` | `kernels.rs:91,1007`; call sites `model.rs:314,400,413,423` | Four GEMVs per layer: QKV `[2048,768]`, WO `[768,1024]`, interleaved W13 `[4608,768]`, W2 `[768,2304]`. Shapes shown as `[output,input]`. |
| QKV copy / RoPE loops in `Model::forward` | `model.rs:326–377` | Expand K/V pairs into 16 heads, normalize Q/K, apply the same real multiply/subtract/add rotation order. |
| `LayerCache::append`, `append_unique_heads` / copy routines | `model.rs:754,836` | Expanded: append 1024 K +1024 V values. Compact: append 512 generated K +512 V; prefix K remains expanded. Capacity was reserved during session setup. |
| `LayerCache::attention`, `attention_with_simd` or `attention_compact_with_simd` | `model.rs:778`; `kernels.rs:307,630` | One query, 16 parallel head outputs, entire visible prefix/generated history. Per-head 128-key stack tiles, dot64 QK, ordered scalar softmax denominator, vector PV AXPY, separate learned sink scaling. |
| `x86::dot_avx2`, `x86::axpy_avx2`; possibly `expf`, `logf` or inlined math | `kernels.rs:1007,1060`; attention loop `378–416` / `727–770` | QK dot and PV accumulation for each key/head; exponentials per key, rescale per tile, log and sink exponential per head. Actual Windows CRT/instruction names require the captured symbols/disassembly. |
| `squared_relu_gate`, its Rayon closure | `kernels.rs:249` | 2304 independent `(relu(gate)*relu(gate))*up` values, retaining the two multiplication boundaries. |
| residual loops in `Model::forward` | `model.rs:409,432` | Two 768-element additions per layer, sequential elementwise arithmetic. |
| final `linear_with_simd` → `dot_avx2` | `model.rs:439–455` | Vocabulary GEMV `[65536,768]`; same generic symbol as layer projections, so caller location distinguishes it. |
| `runner::argmax` | `runner.rs:662` | Full finite-value check, then full ascending scan with strict `>`; ties choose the lowest index. |
| `rayon_core::*`, `rayon::iter::plumbing::*`, closures, worker wait/steal frames | kernel parallel iterator call sites | Independent output/head tasks and synchronization. Waiting samples are not necessarily useful arithmetic; inspect CPU versus wall/ready time separately. |

For row one, `linear_with_simd` selects its unpacked dot path, parallelizes
output channels with `with_min_len(32)`, and performs no matrix packing. Auto
currently resolves to AVX2/FMA when supported; AVX-512 is explicit. GEMM has its
own dispatch but is not called by this row-one path. Similarly, attention GEMM
requires at least four query rows. `gemm::*` samples in a full-page trace can
belong to prefill; do not automatically attribute them to decode. Phase-packed
weights can be prepared at Runner construction when requested, but their
2–8-row arithmetic is unused for a single page.

There are 89 GEMV calls and 245,760 output dot products per next-token forward.
The visited dense projection matrices contain 835.5 MiB of FP32 weights:
643.5 MiB across the layer matrices and 192 MiB for vocabulary. This is logical
matrix payload, **not measured DRAM traffic or a bandwidth diagnosis**.
Attention adds context-dependent work: 352×L dot64 calls and the same number of
AXPY64 calls across 22 layers for visible length L. Compact storage reduces
unique KV bytes; the two query heads still perform distinct arithmetic and may
read the shared K/V values separately. No cache-hit benefit is assumed.

The source issues 222 Rayon parallel iterator operations per next-token
forward: ten per layer (four projections, four RMS calls, gate, attention),
plus final RMS and vocabulary. Some contain a single normalization row; this
count is not the number of OS tasks or thread-pool creations.

## Order-preserving hypotheses to rank after profiling

1. **Scheduling granularity.** If worker/synchronization cost is material,
   specialize single-row RMS and tune channel/gate task granularity while
   keeping each element's exact reduction and expression. Grouping independent
   output channels changes scheduling without changing their dot order. The
   16-head attention task width may constrain useful worker count; thread-count
   tuning needs measured evidence rather than assuming all cores are best.
2. **GEMV dispatch and input reuse.** If dot/projection work dominates, move
   runtime dispatch outside a channel block and consider computing a small
   number of independent output channels together, reusing loaded input
   vectors. Preserve each output's four AVX2 accumulators, input assignment,
   FMA sequence, merge tree and horizontal reduction exactly. Wider SIMD or a
   different reduction tree is not automatically bit-compatible. Register
   pressure and memory bandwidth can erase any gain; the profile must decide.
3. **Attention movement/call overhead.** If long-context attention dominates,
   consider exact head-pair scheduling/cache reuse or specialized dot64/AXPY64
   calls with the same FMA order. Keep 128-key boundaries, key traversal,
   maxima, denominator order, exponent/log implementation and separate sink
   scaling. Splitting a head across workers, changing tile size, approximate
   exponentials or merging the sink changes arithmetic and is outside this
   order-preserving set. The separately tested prefix-temporal storage copy is
   not a live backend and has no speed claim.
4. **Small fixed work.** If visible in samples, create decode RoPE factors by
   reusing temporal entries plus exact `[1,+0]` spatial factors, without
   removing the original rotation arithmetic (signed-zero behavior matters).
   A one-pass finite check plus ascending argmax could remove one logits scan
   while preserving first-index ties and rejecting every nonfinite value.
   Neither is presumed significant next to projection/attention work.

Preserve current CPU outputs as the optimization control: exact same-prefix
intermediates and free IDs/text/stops, plus warmed allocation behavior where
relevant. The ten existing GPU numerical failures remain separately recorded;
they neither disappear from a faster CPU path nor prohibit a correctly isolated
performance experiment. This note proposes no source changes or speed claim.
