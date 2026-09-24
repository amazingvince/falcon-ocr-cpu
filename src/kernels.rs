//! CPU operators for the FP32 Falcon-OCR execution graph.
//!
//! Shapes are checked at these safe entry points, before entering SIMD/GEMM code.
//! Every parallel operation uses the caller's current Rayon pool. A `Runner` can
//! therefore bound all of these operators with one `ThreadPool::install` call.
//! The local tests establish operator accuracy; they do not establish GPU parity.

use rayon::prelude::*;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod attention64;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod prefill64;
// Bitwise equal to the platform expf on the softmax domain (exhaustive test).
#[cfg(target_arch = "x86_64")]
pub(crate) mod panel_bf16;
pub(crate) mod panel_gemm;
/// Prefill attention stage split (probe, `FALCON_OCR_PREFILL_PROFILE`).
pub(crate) fn report_prefill_stage_cycles() {
    prefill64::report_stage_cycles();
}
pub(crate) mod vexp;

/// Vector exp in the prefill attention tiles on x86.
///
/// `Exact` reproduces the platform `expf` bit for bit (`kernels::vexp`), so
/// hidden states match earlier results on this host. `Fast` is the portable
/// polynomial that NEON and the portable path use: at most a few ulp from
/// `Exact` in rare lanes, about 1 s faster per full page, and token-identical
/// to `Exact` on all 67 calibration pages (93,249 tokens). NEON always uses
/// the fast exp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpMode {
    Exact,
    Fast,
}

const EXP_UNSET: u8 = 0;
const EXP_EXACT: u8 = 1;
const EXP_FAST: u8 = 2;
static EXP_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(EXP_UNSET);

/// Select the prefill exp for this process. Without a call, `Exact` is used
/// unless the environment sets `FALCON_OCR_EXP=fast`.
pub fn set_exp_mode(mode: ExpMode) {
    let value = match mode {
        ExpMode::Exact => EXP_EXACT,
        ExpMode::Fast => EXP_FAST,
    };
    EXP_MODE.store(value, std::sync::atomic::Ordering::Relaxed);
}

/// The selected prefill exp mode.
pub fn exp_mode() -> ExpMode {
    use std::sync::atomic::Ordering::Relaxed;
    let mut value = EXP_MODE.load(Relaxed);
    if value == EXP_UNSET {
        let fast = std::env::var("FALCON_OCR_EXP").is_ok_and(|v| v == "fast");
        value = if fast { EXP_FAST } else { EXP_EXACT };
        // A concurrent explicit `set_exp_mode` wins over the environment.
        value = match EXP_MODE.compare_exchange(EXP_UNSET, value, Relaxed, Relaxed) {
            Ok(_) => value,
            Err(current) => current,
        };
    }
    if value == EXP_FAST {
        ExpMode::Fast
    } else {
        ExpMode::Exact
    }
}

/// Vector implementation used for small-batch GEMV and attention. GEMM has its own dispatch.
///
/// Explicit unavailable variants return an error from [`Simd::validate`] and
/// panic if passed directly to a kernel. `Auto` currently selects AVX2/FMA; the
/// AVX-512 candidate is explicit until whole-model benchmarks justify promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Simd {
    Auto,
    Scalar,
    Avx2,
    Avx512,
    /// aarch64 Advanced SIMD (always present on aarch64).
    Neon,
}

impl Simd {
    pub fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Auto | Self::Scalar => Ok(()),
            Self::Avx2 if avx2_available() => Ok(()),
            Self::Avx512 if avx512_available() => Ok(()),
            Self::Avx2 => Err("AVX2 and FMA are unavailable on this CPU"),
            Self::Avx512 => Err("AVX-512F is unavailable on this CPU"),
            Self::Neon if cfg!(target_arch = "aarch64") => Ok(()),
            Self::Neon => Err("NEON requires an aarch64 CPU"),
        }
    }

    /// Resolve automatic selection for reporting and one-time dispatch.
    pub fn resolved(self) -> Self {
        self.validate().expect("unsupported SIMD override");
        match self {
            Self::Auto if avx2_available() => Self::Avx2,
            Self::Auto if cfg!(target_arch = "aarch64") => Self::Neon,
            Self::Auto => Self::Scalar,
            explicit => explicit,
        }
    }
}

/// The instruction set the fixed-width decode kernels are compiled for:
/// AVX2/FMA on x86-64, NEON on aarch64 (never matched elsewhere).
fn native_vector() -> Simd {
    if cfg!(target_arch = "aarch64") {
        Simd::Neon
    } else if cfg!(target_arch = "x86_64") {
        Simd::Avx2
    } else {
        Simd::Auto
    }
}

fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn elements(rows: usize, cols: usize) -> usize {
    rows.checked_mul(cols)
        .expect("tensor shape overflows usize")
}

/// Matrix multiplication with HF row-major `[out_dim, in_dim]` weights.
/// Input and output are row-major `[rows, in_dim]` and `[rows, out_dim]`.
pub fn linear(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    out: &mut [f32],
) {
    linear_with_simd(input, rows, in_dim, weight, out_dim, out, Simd::Auto);
}

/// [`linear`] with an explicit GEMV vector implementation.
///
/// `Scalar` also selects a scalar matrix multiplication for numerical debugging.
/// AVX2/AVX-512 selection controls batches of up to eight rows; larger calls
/// use `gemm`, which independently selects supported matrix instructions.
pub fn linear_with_simd(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    out: &mut [f32],
    simd: Simd,
) {
    assert_eq!(input.len(), elements(rows, in_dim), "linear input shape");
    assert_eq!(
        weight.len(),
        elements(out_dim, in_dim),
        "linear weight shape"
    );
    assert_eq!(out.len(), elements(rows, out_dim), "linear output shape");
    let selected = simd.resolved();
    if out.is_empty() {
        return;
    }
    if in_dim == 0 {
        out.fill(0.0);
        return;
    }
    if rows <= 8 {
        let dot = dot_kernel(selected);
        // No packing/allocation on the decode and small-batch paths. This
        // avoids packing a full weight matrix to multiply only 1-8 vectors.
        // Tasks partition the flattened outputs, exposing channel parallelism
        // even when the number of request rows is smaller than the pool.
        let total = out.len();
        let block = crate::team::block_size(total, 8, 32);
        let shared = crate::team::SharedMut::new(out);
        crate::team::for_each(total.div_ceil(block), |b| {
            let start = b * block;
            let len = block.min(total - start);
            // SAFETY: blocks are disjoint ranges of `out`.
            let values = unsafe { shared.slice(start, len) };
            for (offset, value) in values.iter_mut().enumerate() {
                let index = start + offset;
                let row = index / out_dim;
                let channel = index % out_dim;
                *value = dot(
                    &input[row * in_dim..(row + 1) * in_dim],
                    &weight[channel * in_dim..(channel + 1) * in_dim],
                );
            }
        });
        return;
    }
    if selected == Simd::Scalar {
        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(row, dst)| {
                let x = &input[row * in_dim..(row + 1) * in_dim];
                for (channel, y) in dst.iter_mut().enumerate() {
                    *y = dot_scalar(x, &weight[channel * in_dim..(channel + 1) * in_dim]);
                }
            });
        return;
    }
    let in_stride = isize::try_from(in_dim).expect("linear input stride too large");
    let out_stride = isize::try_from(out_dim).expect("linear output stride too large");
    let threads = gemm_threads().unwrap_or_else(rayon::current_num_threads);
    let parallelism = if threads == 1 {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(threads)
    };
    // SAFETY: Exact slice lengths above bound every addressed element. Inputs
    // are immutable and Rust's exclusive output borrow disallows aliasing.
    // gemm's cs precedes rs, and it computes alpha*dst + beta*lhs*rhs.
    unsafe {
        gemm::gemm(
            rows,
            out_dim,
            in_dim,
            out.as_mut_ptr(),
            1,
            out_stride,
            false,
            input.as_ptr(),
            1,
            in_stride,
            weight.as_ptr(),
            in_stride,
            1,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            parallelism,
        );
    }
}

/// Experiment override of the prefill GEMM's thread count
/// (`FALCON_OCR_GEMM_THREADS`, read once); scheduling only, same arithmetic.
fn gemm_threads() -> Option<usize> {
    static THREADS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("FALCON_OCR_GEMM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= 1)
    })
}

/// Normalize each contiguous `width`-element row in FP32.
/// Optional affine weights have one value per channel and no additive bias.
pub fn rms_norm(input: &[f32], out: &mut [f32], width: usize, eps: f32, weight: Option<&[f32]>) {
    assert!(width > 0, "RMSNorm width must be positive");
    assert_eq!(input.len() % width, 0, "RMSNorm input shape");
    assert_eq!(input.len(), out.len(), "RMSNorm output shape");
    assert!(
        eps.is_finite() && eps > 0.0,
        "RMSNorm epsilon must be finite and positive"
    );
    if let Some(w) = weight {
        assert_eq!(w.len(), width, "RMSNorm affine shape");
    }
    // Decode-sized inputs (one hidden row, or 16 query/key heads of up to eight
    // rows) cost less than waking the pool; each row's arithmetic is identical.
    if input.len() <= RMS_NORM_SERIAL_ELEMENTS {
        for (dst, src) in out.chunks_mut(width).zip(input.chunks(width)) {
            rms_norm_row(src, dst, width, eps, weight);
        }
        return;
    }
    out.par_chunks_mut(width)
        .zip(input.par_chunks(width))
        .for_each(|(dst, src)| rms_norm_row(src, dst, width, eps, weight));
}

const RMS_NORM_SERIAL_ELEMENTS: usize = 16_384;

#[inline]
pub(crate) fn rms_norm_row(src: &[f32], dst: &mut [f32], width: usize, eps: f32, weight: Option<&[f32]>) {
    // Pairwise FP32 reduction limits accumulated rounding error on
    // 768-channel rows without widening the reference's dtype or
    // fusing the square into an accumulation. CUDA reduction order
    // may still differ and is checked by independent operator traces.
    let scale = rms_scale(src, width, eps);
    match weight {
        Some(w) => {
            for ((y, x), affine) in dst.iter_mut().zip(src).zip(w) {
                *y = (*x * scale) * *affine;
            }
        }
        None => {
            for (y, x) in dst.iter_mut().zip(src) {
                *y = *x * scale;
            }
        }
    }
}

/// The factor `rms_norm_row` multiplies a row by.
#[inline]
pub(crate) fn rms_scale(src: &[f32], width: usize, eps: f32) -> f32 {
    (sum_squares_pairwise(src) / width as f32 + eps).sqrt().recip()
}

fn sum_squares_pairwise(input: &[f32]) -> f32 {
    const LANES: usize = 32;
    if input.len() > LANES {
        // Balance the tree while retaining full 32-value leaves except at the
        // tail. Recursion needs only logarithmic stack depth, no heap scratch.
        let middle = (input.len() / 2).next_multiple_of(LANES);
        return sum_squares_pairwise(&input[..middle]) + sum_squares_pairwise(&input[middle..]);
    }
    let mut partial = [0.0_f32; LANES];
    for (dst, value) in partial.iter_mut().zip(input) {
        *dst = *value * *value;
    }
    for width in [16, 8, 4, 2, 1] {
        for index in 0..width {
            partial[index] += partial[index + width];
        }
    }
    partial[0]
}

/// Interleaved `[gate_0, up_0, gate_1, up_1, ...]` squared-ReLU GLU.
pub fn squared_relu_gate(interleaved: &[f32], output: &mut [f32]) {
    assert_eq!(interleaved.len(), elements(output.len(), 2), "GLU shape");
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, y)| *y = squared_relu_glu(interleaved[i * 2], interleaved[i * 2 + 1]));
}

#[inline(always)]
pub(crate) fn squared_relu_glu(gate: f32, up: f32) -> f32 {
    // Match the pinned Triton kernel's tl.where(gate > 0, gate, 0),
    // including its treatment of NaN gates and signed zero.
    let relu = if gate > 0.0 { gate } else { 0.0 };
    (relu * relu) * up
}

/// `linear_with_simd` into interleaved `[gate, up]` channels followed by
/// `squared_relu_gate`, fused for 1..=8 rows. Each gated value uses the same
/// per-channel dot product and gate expression as the unfused pair, so the
/// result is bit-identical while one parallel region replaces two and the
/// `2 * ffn_dim` intermediate is never written. Returns `false` (and writes
/// nothing) for shapes the fused path does not cover.
pub fn linear_glu_with_simd(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    ffn_dim: usize,
    gated: &mut [f32],
    simd: Simd,
) -> bool {
    assert_eq!(input.len(), elements(rows, in_dim), "GLU input shape");
    assert_eq!(
        weight.len(),
        elements(elements(ffn_dim, 2), in_dim),
        "GLU weight shape"
    );
    assert_eq!(gated.len(), elements(rows, ffn_dim), "GLU output shape");
    if rows == 0 || rows > 8 || in_dim == 0 {
        return false;
    }
    let dot = dot_kernel(simd.resolved());
    let total = gated.len();
    let block = crate::team::block_size(total, 4, 16);
    let shared = crate::team::SharedMut::new(gated);
    crate::team::for_each(total.div_ceil(block), |b| {
        let start = b * block;
        let len = block.min(total - start);
        // SAFETY: blocks are disjoint ranges of `gated`.
        let values = unsafe { shared.slice(start, len) };
        for (offset, value) in values.iter_mut().enumerate() {
            let index = start + offset;
            let row = index / ffn_dim;
            let channel = index % ffn_dim;
            let x = &input[row * in_dim..(row + 1) * in_dim];
            let gate = dot(x, &weight[2 * channel * in_dim..(2 * channel + 1) * in_dim]);
            let up = dot(
                x,
                &weight[(2 * channel + 1) * in_dim..(2 * channel + 2) * in_dim],
            );
            *value = squared_relu_glu(gate, up);
        }
    });
    true
}

/// Expanded-head attention on token-major `[sequence, heads, head_dim]` Q/K/V.
///
/// Absolute query positions begin at `query_offset`; K/V positions begin at
/// zero. A key is visible if it is causal, or both positions are in the image
/// interval `[image_start, image_end)`. In particular the image-end token is
/// excluded. Match the upstream FP32 operation boundaries: compute attention
/// without its learned sink, then multiply by `sigmoid(logsumexp - sink)`.
/// Although fusing a zero-value sink into the denominator is algebraically
/// equivalent, it changes intermediate rounding and is not the reference path.
///
/// Online softmax uses bounded logits on the stack per task and the caller's
/// output as its accumulator. Prefill uses blocked GEMM for QK and PV; decode
/// uses vector dot products. Neither materializes a sequence-squared matrix.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    query_len: usize,
    kv_len: usize,
    n_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    attention_with_simd(
        q,
        k,
        v,
        query_len,
        kv_len,
        n_heads,
        head_dim,
        query_offset,
        image_start,
        image_end,
        sinks,
        output,
        Simd::Auto,
    );
}

/// [`attention`] with explicit vector dispatch for dot products and accumulation.
#[allow(clippy::too_many_arguments)]
pub fn attention_with_simd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    query_len: usize,
    kv_len: usize,
    n_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    simd: Simd,
) {
    assert!(
        head_dim > 0 && n_heads > 0,
        "attention head dimensions must be positive"
    );
    let token_width = elements(n_heads, head_dim);
    assert_eq!(
        q.len(),
        elements(query_len, token_width),
        "attention Q shape"
    );
    assert_eq!(k.len(), elements(kv_len, token_width), "attention K shape");
    assert_eq!(v.len(), k.len(), "attention V shape");
    assert_eq!(output.len(), q.len(), "attention output shape");
    assert_eq!(sinks.len(), n_heads, "attention sink shape");
    assert!(
        image_start <= image_end && image_end <= kv_len,
        "attention image interval"
    );
    assert!(
        query_offset
            .checked_add(query_len)
            .is_some_and(|end| end <= kv_len),
        "attention query interval"
    );
    let selected = simd.resolved();
    if query_len >= 4 && selected != Simd::Scalar {
        attention_gemm(
            q,
            k,
            v,
            n_heads,
            head_dim,
            query_offset,
            image_start,
            image_end,
            sinks,
            output,
        );
        return;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if query_len == 1 && head_dim == 64 && selected == native_vector() {
        // SAFETY: the native vector ISA is available and every slice shape was checked above.
        unsafe {
            attention64::attention(
                q,
                k,
                v,
                n_heads,
                query_offset,
                image_start,
                image_end,
                sinks,
                output,
            );
        }
        return;
    }
    attention_online_softmax(
        q,
        k,
        v,
        n_heads,
        head_dim,
        query_offset,
        image_start,
        image_end,
        sinks,
        output,
        selected,
    );
}

/// The generic per-head online-softmax loop with function-pointer kernels.
///
/// It serves every shape or backend the fixed-width kernel does not cover and
/// is the bit-exact oracle for [`attention64`]'s tests. `selected` must already
/// be resolved and validated by the public entry point.
#[allow(clippy::too_many_arguments)]
fn attention_online_softmax(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    selected: Simd,
) {
    let token_width = elements(n_heads, head_dim);
    let dot = dot_kernel(selected);
    let axpy = axpy_kernel(selected);
    let scale = (head_dim as f32).sqrt().recip();
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(qh, out)| {
            let query = qh / n_heads;
            let head = qh % n_heads;
            let absolute_query = query_offset + query;
            let query_in_image = absolute_query >= image_start && absolute_query < image_end;
            let visible_end = if query_in_image {
                image_end
            } else {
                absolute_query + 1
            };
            let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
            out.fill(0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            const TILE: usize = 128;
            let mut logits = [0.0_f32; TILE];
            for start in (0..visible_end).step_by(TILE) {
                let len = (visible_end - start).min(TILE);
                let mut block_max = f32::NEG_INFINITY;
                for (j, logit) in logits[..len].iter_mut().enumerate() {
                    let key = start + j;
                    let begin = key * token_width + head * head_dim;
                    *logit = dot(qvec, &k[begin..begin + head_dim]) * scale;
                    block_max = block_max.max(*logit);
                }
                let new_max = running_max.max(block_max);
                let rescale = if running_max == f32::NEG_INFINITY {
                    0.0
                } else {
                    (running_max - new_max).exp()
                };
                for value in out.iter_mut() {
                    *value *= rescale;
                }
                denominator *= rescale;
                for (j, logit) in logits[..len].iter().enumerate() {
                    let probability = (*logit - new_max).exp();
                    denominator += probability;
                    let begin = (start + j) * token_width + head * head_dim;
                    axpy(probability, &v[begin..begin + head_dim], out);
                }
                running_max = new_max;
            }
            let logsumexp = running_max + denominator.ln();
            let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
            for value in out {
                *value = (*value / denominator) * sink_scale;
            }
        });
}

/// A flash-style CPU prefill: each Rayon task owns one contiguous query tile
/// and all of its output heads. Temporary logits/probabilities are reused on
/// the stack. GEMM is deliberately single-threaded here because Rayon already
/// schedules the outer tiles in the Runner's bounded pool.
#[allow(clippy::too_many_arguments)]
fn attention_gemm(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    const QUERY_TILE: usize = 32;
    const KEY_TILE: usize = 128;
    let token_width = n_heads * head_dim;
    let stride = isize::try_from(token_width).expect("attention token stride too large");
    let scale = (head_dim as f32).sqrt().recip();
    output
        .par_chunks_mut(QUERY_TILE * token_width)
        .enumerate()
        .for_each(|(tile, out)| {
            let first_query = tile * QUERY_TILE;
            let queries = out.len() / token_width;
            let first_absolute = query_offset + first_query;
            let last_absolute = first_absolute + queries - 1;
            // A tile can cross BOS/image/text boundaries. Every row is still
            // individually masked below; this bound only avoids invisible tiles.
            let intersects_image = first_absolute < image_end && last_absolute >= image_start;
            let visible_end = if intersects_image {
                (last_absolute + 1).max(image_end)
            } else {
                last_absolute + 1
            };
            let mut scores = [0.0_f32; QUERY_TILE * KEY_TILE];
            let mut maxima = [0.0_f32; QUERY_TILE];
            let mut denominators = [0.0_f32; QUERY_TILE];
            for head in 0..n_heads {
                maxima[..queries].fill(f32::NEG_INFINITY);
                denominators[..queries].fill(0.0);
                for row in 0..queries {
                    out[row * token_width + head * head_dim
                        ..row * token_width + (head + 1) * head_dim]
                        .fill(0.0);
                }
                for key_start in (0..visible_end).step_by(KEY_TILE) {
                    let keys = (visible_end - key_start).min(KEY_TILE);
                    let q_start = first_query * token_width + head * head_dim;
                    let k_start = key_start * token_width + head * head_dim;
                    // SAFETY: The public entry point validated complete Q/K/V/out
                    // shapes. Matrix row strides skip other heads but remain in
                    // the corresponding slice. scores has QUERY_TILE*KEY_TILE
                    // elements and both active dimensions are bounded by these.
                    unsafe {
                        gemm::gemm(
                            queries,
                            keys,
                            head_dim,
                            scores.as_mut_ptr(),
                            1,
                            KEY_TILE as isize,
                            false,
                            q.as_ptr().add(q_start),
                            1,
                            stride,
                            k.as_ptr().add(k_start),
                            stride,
                            1,
                            0.0_f32,
                            scale,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                    for row in 0..queries {
                        let absolute = first_absolute + row;
                        let image_query = absolute >= image_start && absolute < image_end;
                        let row_scores = &mut scores[row * KEY_TILE..row * KEY_TILE + keys];
                        let mut block_max = f32::NEG_INFINITY;
                        for (col, score) in row_scores.iter_mut().enumerate() {
                            let key = key_start + col;
                            if key > absolute
                                && !(image_query && key >= image_start && key < image_end)
                            {
                                *score = f32::NEG_INFINITY;
                            }
                            block_max = block_max.max(*score);
                        }
                        let new_max = maxima[row].max(block_max);
                        // Avoid (-inf)-(-inf) when an entire block is masked.
                        if new_max == f32::NEG_INFINITY {
                            row_scores.fill(0.0);
                            continue;
                        }
                        let rescale = if maxima[row] == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (maxima[row] - new_max).exp()
                        };
                        let out_row = &mut out[row * token_width + head * head_dim
                            ..row * token_width + (head + 1) * head_dim];
                        for value in out_row {
                            *value *= rescale;
                        }
                        denominators[row] *= rescale;
                        exp_shifted(row_scores, new_max);
                        for probability in row_scores.iter() {
                            denominators[row] += *probability;
                        }
                        maxima[row] = new_max;
                    }
                    // Numerators += probabilities @ values. The output's row
                    // stride skips other heads. Each Rayon task owns all rows of
                    // its output tile, so no writes overlap between tasks.
                    unsafe {
                        gemm::gemm(
                            queries,
                            head_dim,
                            keys,
                            out.as_mut_ptr().add(head * head_dim),
                            1,
                            stride,
                            true,
                            scores.as_ptr(),
                            1,
                            KEY_TILE as isize,
                            v.as_ptr().add(k_start),
                            1,
                            stride,
                            1.0_f32,
                            1.0_f32,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                }
                for row in 0..queries {
                    let out_row = &mut out[row * token_width + head * head_dim
                        ..row * token_width + (head + 1) * head_dim];
                    let logsumexp = maxima[row] + denominators[row].ln();
                    let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
                    for value in out_row {
                        *value = (*value / denominators[row]) * sink_scale;
                    }
                }
            }
        });
}

/// Attention over a compact cache, using automatic vector dispatch.
///
/// Prefix keys retain every query head because their spatial rotations differ.
/// Generated text keys and every value retain only the original KV heads.
/// All buffers are token-major. This does not alter the full-cache API.
#[allow(clippy::too_many_arguments)]
pub fn attention_compact(
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    query_len: usize,
    prefix_len: usize,
    total_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    attention_compact_with_simd(
        q,
        prefix_k,
        generated_k,
        v,
        query_len,
        prefix_len,
        total_len,
        n_heads,
        n_kv_heads,
        head_dim,
        query_offset,
        image_start,
        image_end,
        sinks,
        output,
        Simd::Auto,
    );
}

/// [`attention_compact`] with an explicit vector implementation.
///
/// Q and prefix K have shape `[query_len/prefix_len, n_heads, head_dim]`;
/// generated K has `[total_len-prefix_len, n_kv_heads, head_dim]`; V has
/// `[total_len, n_kv_heads, head_dim]`. Query head `h` uses KV head
/// `h / (n_heads / n_kv_heads)`. The entire image interval must be in the prefix.
///
/// BF16 form of the pure-prefill compact attention (fast mode on AVX512-BF16
/// CPUs, see `prefill64::bf16`): Q, K, V and the probabilities round to BF16;
/// scores, softmax and outputs stay FP32. Returns false, writing nothing,
/// when the CPU or the shape does not qualify.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_compact_prefill_bf16(
    q: &[f32],
    prefix_k: &[f32],
    v: &[f32],
    query_len: usize,
    total_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    converted: Option<(&[u32], &[u32])>,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if query_len >= 4
        && head_dim == 64
        && n_heads % n_kv_heads.max(1) == 0
        && query_offset == 0
        && query_len == total_len
        && prefix_k.len() == total_len * n_heads * 64
        && v.len() == total_len * n_kv_heads * 64
        && q.len() == query_len * n_heads * 64
        && output.len() == q.len()
        && sinks.len() == n_heads
        && image_start <= image_end
        && image_end <= total_len
        && avx2_available()
        && avx512_available()
        && panel_bf16::attention()
    {
        // SAFETY: AVX2/FMA/AVX-512F/AVX512-BF16 detected; shapes checked.
        unsafe {
            prefill64::bf16::compact_prefill(
                q,
                prefix_k,
                v,
                query_len,
                n_heads,
                n_kv_heads,
                query_offset,
                image_start,
                image_end,
                sinks,
                output,
                converted,
            );
        }
        return true;
    }
    let _ = (q, prefix_k, v, query_len, total_len, n_heads, n_kv_heads, head_dim);
    let _ = (query_offset, image_start, image_end, sinks, output, converted);
    false
}

/// Whether prefill rows can be written straight into the BF16 attention
/// layouts ([`store_prefill_bf16_row`]).
pub(crate) fn prefill_bf16_rows_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        panel_bf16::attention() && *ON.get_or_init(|| std::is_x86_feature_detected!("avx512bw"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Writes one prefill row's keys and values in the BF16 attention layouts
/// (see `prefill64::bf16::store_row`).
///
/// # Safety
/// [`prefill_bf16_rows_available`] returned true; buffer sizes as in
/// `store_row`; concurrent callers write different rows.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn store_prefill_bf16_row(
    k: &[f32],
    v: &[f32],
    row: usize,
    total: usize,
    heads: usize,
    kv_heads: usize,
    keys: *mut u32,
    values: *mut u32,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        prefill64::bf16::store_row(k, v, row, total, heads, kv_heads, keys, values)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (k, v, row, total, heads, kv_heads, keys, values);
        unreachable!("BF16 rows are x86-only")
    }
}

/// Arithmetic order matches expanded-cache attention. A key tile crossing the
/// prefix boundary is gathered before GEMM, preserving its original tile width
/// and softmax reduction order. Single-token decode never allocates scratch.
#[allow(clippy::too_many_arguments)]
pub fn attention_compact_with_simd(
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    query_len: usize,
    prefix_len: usize,
    total_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    simd: Simd,
) {
    assert!(
        head_dim > 0 && n_heads > 0 && n_kv_heads > 0,
        "compact attention head dimensions must be positive"
    );
    assert_eq!(n_heads % n_kv_heads, 0, "compact attention GQA grouping");
    assert!(prefix_len <= total_len, "compact attention prefix length");
    let query_width = elements(n_heads, head_dim);
    let kv_width = elements(n_kv_heads, head_dim);
    assert_eq!(
        q.len(),
        elements(query_len, query_width),
        "compact attention Q shape"
    );
    assert_eq!(
        prefix_k.len(),
        elements(prefix_len, query_width),
        "compact attention prefix K shape"
    );
    assert_eq!(
        generated_k.len(),
        elements(total_len - prefix_len, kv_width),
        "compact attention generated K shape"
    );
    assert_eq!(
        v.len(),
        elements(total_len, kv_width),
        "compact attention V shape"
    );
    assert_eq!(output.len(), q.len(), "compact attention output shape");
    assert_eq!(sinks.len(), n_heads, "compact attention sink shape");
    assert!(
        image_start <= image_end && image_end <= prefix_len,
        "compact attention image must be inside prefix"
    );
    assert!(
        query_offset
            .checked_add(query_len)
            .is_some_and(|end| end <= total_len),
        "compact attention query interval"
    );
    let selected = simd.resolved();
    // AVX2 (also under an explicit AVX-512 selection) on x86, NEON on aarch64.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if query_len >= 4
        && head_dim == 64
        && total_len == prefix_len
        && if cfg!(target_arch = "x86_64") {
            selected != Simd::Scalar && avx2_available()
        } else {
            selected == Simd::Neon
        }
    {
        // SAFETY: the vector ISA is available; shapes validated above; no
        // generated keys.
        unsafe {
            prefill64::compact_prefill(
                cfg!(target_arch = "x86_64")
                    && matches!(simd, Simd::Auto | Simd::Avx512)
                    && avx512_available(),
                q,
                prefix_k,
                v,
                query_len,
                n_heads,
                n_kv_heads,
                query_offset,
                image_start,
                image_end,
                sinks,
                output,
            );
        }
        return;
    }
    if query_len >= 4 && selected != Simd::Scalar {
        attention_gemm_compact(
            q,
            prefix_k,
            generated_k,
            v,
            prefix_len,
            n_heads,
            n_kv_heads,
            head_dim,
            query_offset,
            image_start,
            image_end,
            sinks,
            output,
        );
        return;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if query_len == 1 && head_dim == 64 && selected == native_vector() {
        // SAFETY: the native vector ISA is available and every slice shape was checked above.
        unsafe {
            attention64::compact(
                q,
                prefix_k,
                generated_k,
                v,
                prefix_len,
                n_heads,
                n_kv_heads,
                query_offset,
                image_start,
                image_end,
                sinks,
                output,
            );
        }
        return;
    }
    attention_compact_online_softmax(
        q,
        prefix_k,
        generated_k,
        v,
        prefix_len,
        n_heads,
        n_kv_heads,
        head_dim,
        query_offset,
        image_start,
        image_end,
        sinks,
        output,
        selected,
    );
}

/// The generic compact-cache online-softmax loop; see [`attention_online_softmax`].
#[allow(clippy::too_many_arguments)]
fn attention_compact_online_softmax(
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    prefix_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
    selected: Simd,
) {
    let query_width = elements(n_heads, head_dim);
    let kv_width = elements(n_kv_heads, head_dim);
    let dot = dot_kernel(selected);
    let axpy = axpy_kernel(selected);
    let scale = (head_dim as f32).sqrt().recip();
    let repeat = n_heads / n_kv_heads;
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(qh, out)| {
            let query = qh / n_heads;
            let head = qh % n_heads;
            let kv_head = head / repeat;
            let absolute_query = query_offset + query;
            let query_in_image = absolute_query >= image_start && absolute_query < image_end;
            let visible_end = if query_in_image {
                image_end
            } else {
                absolute_query + 1
            };
            let qvec = &q[qh * head_dim..(qh + 1) * head_dim];
            out.fill(0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            const TILE: usize = 128;
            let mut logits = [0.0_f32; TILE];
            for start in (0..visible_end).step_by(TILE) {
                let len = (visible_end - start).min(TILE);
                let mut block_max = f32::NEG_INFINITY;
                for (j, logit) in logits[..len].iter_mut().enumerate() {
                    let key = start + j;
                    let kvec = if key < prefix_len {
                        let begin = key * query_width + head * head_dim;
                        &prefix_k[begin..begin + head_dim]
                    } else {
                        let begin = (key - prefix_len) * kv_width + kv_head * head_dim;
                        &generated_k[begin..begin + head_dim]
                    };
                    *logit = dot(qvec, kvec) * scale;
                    block_max = block_max.max(*logit);
                }
                let new_max = running_max.max(block_max);
                let rescale = if running_max == f32::NEG_INFINITY {
                    0.0
                } else {
                    (running_max - new_max).exp()
                };
                for value in out.iter_mut() {
                    *value *= rescale;
                }
                denominator *= rescale;
                for (j, logit) in logits[..len].iter().enumerate() {
                    let probability = (*logit - new_max).exp();
                    denominator += probability;
                    let begin = (start + j) * kv_width + kv_head * head_dim;
                    axpy(probability, &v[begin..begin + head_dim], out);
                }
                running_max = new_max;
            }
            let logsumexp = running_max + denominator.ln();
            let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
            for value in out {
                *value = (*value / denominator) * sink_scale;
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn attention_gemm_compact(
    q: &[f32],
    prefix_k: &[f32],
    generated_k: &[f32],
    v: &[f32],
    prefix_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_offset: usize,
    image_start: usize,
    image_end: usize,
    sinks: &[f32],
    output: &mut [f32],
) {
    const QUERY_TILE: usize = 32;
    const KEY_TILE: usize = 128;
    let query_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let query_stride =
        isize::try_from(query_width).expect("compact attention query stride too large");
    let kv_stride = isize::try_from(kv_width).expect("compact attention KV stride too large");
    let scale = (head_dim as f32).sqrt().recip();
    let repeat = n_heads / n_kv_heads;
    output
        .par_chunks_mut(QUERY_TILE * query_width)
        .enumerate()
        .for_each(|(tile, out)| {
            let first_query = tile * QUERY_TILE;
            let queries = out.len() / query_width;
            let first_absolute = query_offset + first_query;
            let last_absolute = first_absolute + queries - 1;
            let intersects_image = first_absolute < image_end && last_absolute >= image_start;
            let visible_end = if intersects_image {
                (last_absolute + 1).max(image_end)
            } else {
                last_absolute + 1
            };
            let mut scores = [0.0_f32; QUERY_TILE * KEY_TILE];
            let mut maxima = [0.0_f32; QUERY_TILE];
            let mut denominators = [0.0_f32; QUERY_TILE];
            // Pure prefill has no segmented K boundary and leaves this empty.
            // For multiquery continuation only, at most one K tile per query tile
            // needs this bounded KEY_TILE*head_dim buffer; reuse it across heads.
            let mut boundary_keys = Vec::<f32>::new();
            for head in 0..n_heads {
                let kv_head = head / repeat;
                maxima[..queries].fill(f32::NEG_INFINITY);
                denominators[..queries].fill(0.0);
                for row in 0..queries {
                    out[row * query_width + head * head_dim
                        ..row * query_width + (head + 1) * head_dim]
                        .fill(0.0);
                }
                for key_start in (0..visible_end).step_by(KEY_TILE) {
                    let keys = (visible_end - key_start).min(KEY_TILE);
                    let q_start = first_query * query_width + head * head_dim;
                    let (key_data, key_start_offset, key_stride) = if key_start + keys <= prefix_len
                    {
                        (
                            prefix_k,
                            key_start * query_width + head * head_dim,
                            query_stride,
                        )
                    } else if key_start >= prefix_len {
                        (
                            generated_k,
                            (key_start - prefix_len) * kv_width + kv_head * head_dim,
                            kv_stride,
                        )
                    } else {
                        boundary_keys.resize(keys * head_dim, 0.0);
                        for key in 0..keys {
                            let absolute = key_start + key;
                            let src = if absolute < prefix_len {
                                let start = absolute * query_width + head * head_dim;
                                &prefix_k[start..start + head_dim]
                            } else {
                                let start = (absolute - prefix_len) * kv_width + kv_head * head_dim;
                                &generated_k[start..start + head_dim]
                            };
                            boundary_keys[key * head_dim..(key + 1) * head_dim]
                                .copy_from_slice(src);
                        }
                        (boundary_keys.as_slice(), 0, head_dim as isize)
                    };
                    // SAFETY: The public compact entry validates shapes. The K
                    // interval is entirely in one buffer or gathered contiguously;
                    // no matrix straddles the separately allocated cache segments.
                    unsafe {
                        gemm::gemm(
                            queries,
                            keys,
                            head_dim,
                            scores.as_mut_ptr(),
                            1,
                            KEY_TILE as isize,
                            false,
                            q.as_ptr().add(q_start),
                            1,
                            query_stride,
                            key_data.as_ptr().add(key_start_offset),
                            key_stride,
                            1,
                            0.0_f32,
                            scale,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                    for row in 0..queries {
                        let absolute = first_absolute + row;
                        let image_query = absolute >= image_start && absolute < image_end;
                        let row_scores = &mut scores[row * KEY_TILE..row * KEY_TILE + keys];
                        let mut block_max = f32::NEG_INFINITY;
                        for (col, score) in row_scores.iter_mut().enumerate() {
                            let key = key_start + col;
                            if key > absolute
                                && !(image_query && key >= image_start && key < image_end)
                            {
                                *score = f32::NEG_INFINITY;
                            }
                            block_max = block_max.max(*score);
                        }
                        let new_max = maxima[row].max(block_max);
                        if new_max == f32::NEG_INFINITY {
                            row_scores.fill(0.0);
                            continue;
                        }
                        let rescale = if maxima[row] == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (maxima[row] - new_max).exp()
                        };
                        let out_row = &mut out[row * query_width + head * head_dim
                            ..row * query_width + (head + 1) * head_dim];
                        for value in out_row {
                            *value *= rescale;
                        }
                        denominators[row] *= rescale;
                        exp_shifted(row_scores, new_max);
                        for probability in row_scores.iter() {
                            denominators[row] += *probability;
                        }
                        maxima[row] = new_max;
                    }
                    let value_start = key_start * kv_width + kv_head * head_dim;
                    // The compact value row stride is smaller, but the GEMM
                    // dimensions and accumulation order match expanded attention.
                    unsafe {
                        gemm::gemm(
                            queries,
                            head_dim,
                            keys,
                            out.as_mut_ptr().add(head * head_dim),
                            1,
                            query_stride,
                            true,
                            scores.as_ptr(),
                            1,
                            KEY_TILE as isize,
                            v.as_ptr().add(value_start),
                            1,
                            kv_stride,
                            1.0_f32,
                            1.0_f32,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                }
                for row in 0..queries {
                    let out_row = &mut out[row * query_width + head * head_dim
                        ..row * query_width + (head + 1) * head_dim];
                    let logsumexp = maxima[row] + denominators[row].ln();
                    let sink_scale = 1.0 / (1.0 + (sinks[head] - logsumexp).exp());
                    for value in out_row {
                        *value = (*value / denominators[row]) * sink_scale;
                    }
                }
            }
        });
}

/// `values[i] = (values[i] - shift).exp()`, vectorized where available with a
/// bitwise-identical exp (see `vexp`).
fn exp_shifted(values: &mut [f32], shift: f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: AVX2/FMA availability checked.
        unsafe { vexp::exp_shifted_in_place(values, shift) };
        return;
    }
    for value in values {
        *value = (*value - shift).exp();
    }
}

pub(crate) type Dot = fn(&[f32], &[f32]) -> f32;
pub(crate) type Axpy = fn(f32, &[f32], &mut [f32]);

pub(crate) fn dot_kernel(simd: Simd) -> Dot {
    match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => |a, b| {
            // SAFETY: Entry point validated CPU features; shapes give equal
            // slice lengths. The implementation loads only full vector chunks.
            unsafe { x86::dot_avx2(a, b) }
        },
        #[cfg(target_arch = "x86_64")]
        Simd::Avx512 => |a, b| unsafe { x86::dot_avx512(a, b) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; equal slice lengths.
        Simd::Neon => |a, b| unsafe { crate::simd::dot::<crate::simd::Neon>(a, b) },
        Simd::Scalar => dot_scalar,
        _ => unreachable!("unresolved or unsupported vector implementation"),
    }
}

pub(crate) fn axpy_kernel(simd: Simd) -> Axpy {
    match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => |a, x, y| unsafe { x86::axpy_avx2(a, x, y) },
        #[cfg(target_arch = "x86_64")]
        Simd::Avx512 => |a, x, y| unsafe { x86::axpy_avx512(a, x, y) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; equal slice lengths.
        Simd::Neon => |a, x, y| unsafe { crate::simd::axpy::<crate::simd::Neon>(a, x, y) },
        Simd::Scalar => axpy_scalar,
        _ => unreachable!("unresolved or unsupported vector implementation"),
    }
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn axpy_scalar(a: f32, x: &[f32], y: &mut [f32]) {
    for (dst, src) in y.iter_mut().zip(x) {
        *dst += a * *src;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: Caller checks features. Unaligned loads stay inside the two
        // equal-sized slices; the scalar tail covers fewer than eight elements.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 32 <= a.len() {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                    acc3,
                );
                i += 32;
            }
            let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            while i + 8 <= a.len() {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                    acc,
                );
                i += 8;
            }
            let halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
            let pairs = _mm_hadd_ps(halves, halves);
            let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
            while i < a.len() {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_avx2(a: f32, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        unsafe {
            let factor = _mm256_set1_ps(a);
            let mut i = 0;
            while i + 8 <= x.len() {
                let value = _mm256_fmadd_ps(
                    factor,
                    _mm256_loadu_ps(x.as_ptr().add(i)),
                    _mm256_loadu_ps(y.as_ptr().add(i)),
                );
                _mm256_storeu_ps(y.as_mut_ptr().add(i), value);
                i += 8;
            }
            while i < x.len() {
                y[i] += a * x[i];
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        unsafe {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut i = 0;
            while i + 64 <= a.len() {
                acc0 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.as_ptr().add(i)),
                    _mm512_loadu_ps(b.as_ptr().add(i)),
                    acc0,
                );
                acc1 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.as_ptr().add(i + 16)),
                    _mm512_loadu_ps(b.as_ptr().add(i + 16)),
                    acc1,
                );
                acc2 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.as_ptr().add(i + 32)),
                    _mm512_loadu_ps(b.as_ptr().add(i + 32)),
                    acc2,
                );
                acc3 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.as_ptr().add(i + 48)),
                    _mm512_loadu_ps(b.as_ptr().add(i + 48)),
                    acc3,
                );
                i += 64;
            }
            let mut acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            while i + 16 <= a.len() {
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.as_ptr().add(i)),
                    _mm512_loadu_ps(b.as_ptr().add(i)),
                    acc,
                );
                i += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            while i < a.len() {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn axpy_avx512(a: f32, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        unsafe {
            let factor = _mm512_set1_ps(a);
            let mut i = 0;
            while i + 16 <= x.len() {
                let value = _mm512_fmadd_ps(
                    factor,
                    _mm512_loadu_ps(x.as_ptr().add(i)),
                    _mm512_loadu_ps(y.as_ptr().add(i)),
                );
                _mm512_storeu_ps(y.as_mut_ptr().add(i), value);
                i += 16;
            }
            while i < x.len() {
                y[i] += a * x[i];
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed, deterministic generator keeps tests independent of rand versions.
    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn implementations() -> Vec<Simd> {
        [Simd::Scalar, Simd::Auto, Simd::Avx2, Simd::Avx512]
            .into_iter()
            .filter(|simd| simd.validate().is_ok())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*a - *b).abs() <= absolute + relative * b.abs(),
                "index {i}: actual={a}, expected={b}, delta={}",
                (*a - *b).abs()
            );
        }
    }

    fn linear_f64(
        input: &[f32],
        rows: usize,
        in_dim: usize,
        weights: &[f32],
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; rows * out_dim];
        for row in 0..rows {
            for channel in 0..out_dim {
                let mut sum = 0.0_f64;
                for inner in 0..in_dim {
                    sum += input[row * in_dim + inner] as f64
                        * weights[channel * in_dim + inner] as f64;
                }
                out[row * out_dim + channel] = sum as f32;
            }
        }
        out
    }

    #[test]
    fn fused_glu_is_bitwise_linear_then_gate() {
        // Real FFN widths plus odd widths; gates include NaN, signed zeros and
        // negatives so every branch of the squared-ReLU expression is exercised.
        for (in_dim, ffn_dim) in [(768, 2304), (65, 19), (1, 1)] {
            let mut weights = values(2 * ffn_dim * in_dim, 7);
            for (i, w) in weights.iter_mut().enumerate().step_by(97) {
                *w = [f32::NAN, 0.0, -0.0, -3.0][i % 4];
            }
            for simd in implementations() {
                for rows in 1..=8 {
                    let input = values(rows * in_dim, 11 + rows as u32);
                    let mut packed = vec![0.0; rows * 2 * ffn_dim];
                    linear_with_simd(
                        &input,
                        rows,
                        in_dim,
                        &weights,
                        2 * ffn_dim,
                        &mut packed,
                        simd,
                    );
                    let mut expected = vec![0.0; rows * ffn_dim];
                    squared_relu_gate(&packed, &mut expected);
                    let mut fused = vec![f32::INFINITY; rows * ffn_dim];
                    assert!(linear_glu_with_simd(
                        &input, rows, in_dim, &weights, ffn_dim, &mut fused, simd
                    ));
                    for (a, b) in fused.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{simd:?} rows {rows} in {in_dim}");
                    }
                }
            }
        }
        let mut unused = vec![0.0; 9];
        assert!(!linear_glu_with_simd(
            &values(9 * 4, 1),
            9,
            4,
            &values(2 * 4, 2),
            1,
            &mut unused,
            Simd::Scalar
        ));
    }

    #[test]
    fn serial_rms_norm_rows_match_parallel_rows() {
        // Small inputs run serially; the same rows inside a large (parallel)
        // input must normalize to identical bits.
        for width in [64, 768] {
            let small_rows = RMS_NORM_SERIAL_ELEMENTS / width;
            let large_rows = small_rows * 3;
            let large = values(large_rows * width, 5);
            let affine = values(width, 9);
            for weight in [None, Some(affine.as_slice())] {
                let mut parallel = vec![0.0; large.len()];
                rms_norm(&large, &mut parallel, width, 1e-5, weight);
                let small = &large[..small_rows * width];
                let mut serial = vec![0.0; small.len()];
                rms_norm(small, &mut serial, width, 1e-5, weight);
                for (a, b) in serial.iter().zip(&parallel) {
                    assert_eq!(a.to_bits(), b.to_bits(), "width {width}");
                }
            }
        }
    }

    #[test]
    fn linear_layout_and_overwrite() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = [1.0, 10.0, 100.0, -1.0, -2.0, -3.0];
        for simd in implementations() {
            let mut output = [f32::NAN; 4];
            linear_with_simd(&input, 2, 3, &weights, 2, &mut output, simd);
            assert_eq!(output, [321.0, -14.0, 654.0, -32.0]);
        }
    }

    #[test]
    fn linear_real_model_dimensions_and_tails() {
        // Width 768, attention width 1024, projector input 16*16*3, and
        // nonmultiples exercise both actual model and vector-tail dimensions.
        for (rows, in_dim, out_dim) in [
            (1, 768, 1024),
            (1, 1024, 768),
            (2, 768, 128),
            (4, 768, 128),
            (8, 768, 128),
            (3, 768, 97),
            (17, 67, 129),
            (1, 79, 53),
            (1, 7, 3),
        ] {
            let input = values(rows * in_dim, 61);
            let weights = values(out_dim * in_dim, 19);
            let expected = linear_f64(&input, rows, in_dim, &weights, out_dim);
            for simd in implementations() {
                let mut output = vec![0.0; rows * out_dim];
                linear_with_simd(&input, rows, in_dim, &weights, out_dim, &mut output, simd);
                // FP32 sequential reductions are intentionally included;
                // these are numerical unit bounds, not frozen GPU tolerances.
                assert_close(&output, &expected, 6e-5, 3e-6);
            }
        }
    }

    #[test]
    fn linear_empty_dimensions() {
        linear(&[], 0, 17, &[0.0; 51], 3, &mut []);
        let mut output = [f32::NAN; 12];
        linear(&[], 4, 0, &[], 3, &mut output);
        assert_eq!(output, [0.0; 12]);
        linear(&[1.0; 6], 2, 3, &[], 0, &mut []);
    }

    #[test]
    #[should_panic(expected = "linear weight shape")]
    fn linear_rejects_mismatched_weight_shape() {
        linear(&[1.0, 2.0], 1, 2, &[3.0], 1, &mut [0.0]);
    }

    #[test]
    fn norm_matches_independent_f64_reference() {
        for width in [1, 7, 64, 768] {
            let input = values(3 * width, 12);
            let weights = values(width, 102);
            for affine in [None, Some(weights.as_slice())] {
                let mut output = vec![0.0; input.len()];
                rms_norm(&input, &mut output, width, f32::EPSILON, affine);
                let mut reference = vec![0.0; input.len()];
                for row in 0..3 {
                    let norm = (input[row * width..(row + 1) * width]
                        .iter()
                        .map(|v| (*v as f64).powi(2))
                        .sum::<f64>()
                        / width as f64
                        + f32::EPSILON as f64)
                        .sqrt();
                    for col in 0..width {
                        let w = affine.map_or(1.0, |w| w[col]) as f64;
                        reference[row * width + col] =
                            (input[row * width + col] as f64 / norm * w) as f32;
                    }
                }
                assert_close(&output, &reference, 2e-6, 2e-6);
            }
        }
        let mut zeros = [1.0; 64];
        rms_norm(&[0.0; 64], &mut zeros, 64, f32::EPSILON, None);
        assert_eq!(zeros, [0.0; 64]);
    }

    #[test]
    fn glu_is_interleaved_and_relu_is_squared() {
        let mut output = [0.0; 4];
        squared_relu_gate(&[2.0, 3.0, -4.0, 5.0, 0.5, -8.0, 0.0, 12.0], &mut output);
        assert_eq!(output, [12.0, 0.0, -2.0, 0.0]);
        squared_relu_gate(
            &[f32::NAN, 1.0, 1.0, f32::NAN, -1.0, 2.0, 2.0, 0.0],
            &mut output,
        );
        assert_eq!(output[0], 0.0);
        assert!(output[1].is_nan());
        assert_eq!(&output[2..], &[0.0, 0.0]);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        qlen: usize,
        kvlen: usize,
        heads: usize,
        dim: usize,
        offset: usize,
        image_start: usize,
        image_end: usize,
        sinks: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0; qlen * heads * dim];
        for query in 0..qlen {
            let absolute = offset + query;
            for head in 0..heads {
                // Dense logits and explicit mask are intentionally independent
                // from the production contiguous-visible-range/online algorithm.
                let mut scores = vec![f64::NEG_INFINITY; kvlen];
                for key in 0..kvlen {
                    let allowed = key <= absolute
                        || (absolute >= image_start
                            && absolute < image_end
                            && key >= image_start
                            && key < image_end);
                    if !allowed {
                        continue;
                    }
                    let mut score = 0.0_f64;
                    for d in 0..dim {
                        score += q[(query * heads + head) * dim + d] as f64
                            * k[(key * heads + head) * dim + d] as f64;
                    }
                    scores[key] = score / (dim as f64).sqrt();
                }
                let max = scores.iter().copied().fold(sinks[head] as f64, f64::max);
                let denom = (sinks[head] as f64 - max).exp()
                    + scores.iter().map(|s| (s - max).exp()).sum::<f64>();
                for d in 0..dim {
                    out[(query * heads + head) * dim + d] = (scores
                        .iter()
                        .enumerate()
                        .map(|(key, s)| {
                            (s - max).exp() / denom * v[(key * heads + head) * dim + d] as f64
                        })
                        .sum::<f64>())
                        as f32;
                }
            }
        }
        out
    }

    #[test]
    fn attention_matches_dense_for_hybrid_prefill_cached_decode_and_tails() {
        for (qlen, kvlen, heads, dim, offset, start, end) in [
            (7, 7, 2, 7, 0, 1, 5),
            (5, 5, 3, 17, 0, 0, 0),
            (1, 267, 16, 64, 266, 1, 201),
            (3, 260, 2, 33, 257, 2, 193),
            (33, 267, 2, 64, 234, 2, 252),
            (133, 133, 2, 64, 0, 16, 98),
            (129, 129, 2, 8, 0, 0, 129),
            (1, 1, 1, 1, 0, 0, 0),
        ] {
            let q = values(qlen * heads * dim, 71);
            let k = values(kvlen * heads * dim, 83);
            let v = values(kvlen * heads * dim, 117);
            let sinks = values(heads, 24);
            let expected = attention_f64(
                &q, &k, &v, qlen, kvlen, heads, dim, offset, start, end, &sinks,
            );
            for simd in implementations() {
                let mut out = vec![f32::NAN; q.len()];
                attention_with_simd(
                    &q, &k, &v, qlen, kvlen, heads, dim, offset, start, end, &sinks, &mut out, simd,
                );
                assert_close(&out, &expected, 2e-6, 4e-6);
            }
        }
    }

    #[test]
    fn attention_image_boundaries_and_sink_value_are_correct() {
        // All real logits and the sink logit are zero. Image positions 1 and 2
        // see three real keys; BOS cannot look forward; img_end is causal.
        let qk = [0.0; 5];
        let v = [1.0, 10.0, 100.0, 1000.0, 10000.0];
        let mut out = [0.0; 5];
        attention(&qk, &qk, &v, 5, 5, 1, 1, 0, 1, 3, &[0.0], &mut out);
        assert_close(&out, &[0.5, 27.75, 27.75, 222.2, 1851.8334], 1e-6, 1e-7);
        let mut no_sink = [0.0];
        attention(
            &[0.0],
            &[0.0],
            &[13.0],
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            &[f32::NEG_INFINITY],
            &mut no_sink,
        );
        assert_eq!(no_sink, [13.0]);
    }

    #[test]
    fn attention_stable_with_large_positive_and_negative_logits_and_sinks() {
        let keys: Vec<f32> = (0..257)
            .map(|i| if i == 140 { 1010.0 } else { -1000.0 })
            .collect();
        let values: Vec<f32> = (0..257).map(|i| i as f32 * 0.1 - 3.0).collect();
        for sink in [-10000.0, 10000.0, 1009.0, f32::NEG_INFINITY] {
            let expected = attention_f64(&[1.0], &keys, &values, 1, 257, 1, 1, 256, 0, 0, &[sink]);
            for simd in implementations() {
                let mut out = [f32::NAN];
                attention_with_simd(
                    &[1.0],
                    &keys,
                    &values,
                    1,
                    257,
                    1,
                    1,
                    256,
                    0,
                    0,
                    &[sink],
                    &mut out,
                    simd,
                );
                assert_close(&out, &expected, 1e-6, 1e-6);
                assert!(out[0].is_finite());
            }
        }
        // Exercise rescaling through the separate multiquery GEMM prefill.
        let q = [1.0; 4];
        let expected = attention_f64(&q, &keys, &values, 4, 257, 1, 1, 253, 0, 0, &[-10000.0]);
        let mut out = [f32::NAN; 4];
        attention(
            &q,
            &keys,
            &values,
            4,
            257,
            1,
            1,
            253,
            0,
            0,
            &[-10000.0],
            &mut out,
        );
        assert_close(&out, &expected, 1e-6, 1e-6);
    }

    #[test]
    fn parallelism_uses_callers_pool_without_changing_results() {
        let input = values(17 * 768, 191);
        let weights = values(128 * 768, 122);
        let expected = linear_f64(&input, 17, 768, &weights, 128);
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut out = vec![0.0; expected.len()];
            pool.install(|| {
                assert_eq!(rayon::current_num_threads(), threads);
                linear(&input, 17, 768, &weights, 128, &mut out);
            });
            assert_close(&out, &expected, 6e-5, 3e-6);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_compact_cache(
        prefix: &[f32],
        generated: &[f32],
        values: &[f32],
        prefix_len: usize,
        total_len: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0; total_len * heads * dim];
        let mut expanded_values = vec![0.0; keys.len()];
        keys[..prefix.len()].copy_from_slice(prefix);
        for token in 0..total_len {
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                let dst = (token * heads + head) * dim;
                let value_src = (token * kv_heads + kv_head) * dim;
                expanded_values[dst..dst + dim]
                    .copy_from_slice(&values[value_src..value_src + dim]);
                if token >= prefix_len {
                    let src = ((token - prefix_len) * kv_heads + kv_head) * dim;
                    keys[dst..dst + dim].copy_from_slice(&generated[src..src + dim]);
                }
            }
        }
        (keys, expanded_values)
    }

    #[test]
    fn compact_cache_is_bit_identical_to_expanded_for_every_vector_backend() {
        for (
            queries,
            prefix_len,
            total_len,
            heads,
            kv_heads,
            dim,
            offset,
            image_start,
            image_end,
        ) in [
            (7, 7, 7, 2, 1, 7, 0, 1, 5),
            (144, 144, 144, 16, 8, 64, 0, 0, 133),
            (137, 137, 137, 16, 8, 64, 0, 2, 129),
            (1, 144, 145, 16, 8, 64, 144, 0, 133),
            (4, 137, 141, 16, 8, 64, 137, 1, 129),
            (33, 129, 267, 4, 2, 33, 234, 1, 126),
            (9, 0, 9, 4, 2, 17, 0, 0, 0),
            (2, 257, 260, 4, 1, 7, 258, 0, 249),
            (1, 128, 4099, 16, 8, 64, 4098, 0, 120),
            (1, 127, 16384, 16, 8, 64, 16383, 1, 119),
        ] {
            let q = values(queries * heads * dim, 71);
            // Prefix heads are intentionally independent. Collapsing paired
            // spatial keys would visibly change the expected output.
            let prefix = values(prefix_len * heads * dim, 83);
            let generated = values((total_len - prefix_len) * kv_heads * dim, 37);
            let v = values(total_len * kv_heads * dim, 117);
            let sinks = values(heads, 24);
            let (expanded_k, expanded_v) = expand_compact_cache(
                &prefix, &generated, &v, prefix_len, total_len, heads, kv_heads, dim,
            );
            for simd in implementations() {
                let mut expected = vec![f32::NAN; q.len()];
                let mut actual = vec![f32::NAN; q.len()];
                attention_with_simd(
                    &q,
                    &expanded_k,
                    &expanded_v,
                    queries,
                    total_len,
                    heads,
                    dim,
                    offset,
                    image_start,
                    image_end,
                    &sinks,
                    &mut expected,
                    simd,
                );
                attention_compact_with_simd(
                    &q,
                    &prefix,
                    &generated,
                    &v,
                    queries,
                    prefix_len,
                    total_len,
                    heads,
                    kv_heads,
                    dim,
                    offset,
                    image_start,
                    image_end,
                    &sinks,
                    &mut actual,
                    simd,
                );
                for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "compact mismatch at {index}: {a} versus {e}; {simd:?}, Q={queries}, prefix={prefix_len}, total={total_len}, heads={heads}, dim={dim}"
                    );
                }
            }
        }
    }

    #[test]
    fn compact_cache_extreme_sinks_preserve_expanded_rounding() {
        let prefix_len = 3;
        let total_len = 133;
        let q = [1.0; 4 * 2];
        let prefix = vec![1000.0; prefix_len * 2];
        let mut generated = vec![-1000.0; total_len - prefix_len];
        generated[127] = 1010.0;
        let v = values(total_len, 131);
        let (kfull, vfull) =
            expand_compact_cache(&prefix, &generated, &v, prefix_len, total_len, 2, 1, 1);
        for simd in implementations() {
            for sinks in [[1009.0, -10000.0], [f32::NEG_INFINITY, 10000.0]] {
                let mut actual = [f32::NAN; 8];
                let mut expected = [f32::NAN; 8];
                attention_with_simd(
                    &q,
                    &kfull,
                    &vfull,
                    4,
                    total_len,
                    2,
                    1,
                    129,
                    0,
                    2,
                    &sinks,
                    &mut expected,
                    simd,
                );
                attention_compact_with_simd(
                    &q,
                    &prefix,
                    &generated,
                    &v,
                    4,
                    prefix_len,
                    total_len,
                    2,
                    1,
                    1,
                    129,
                    0,
                    2,
                    &sinks,
                    &mut actual,
                    simd,
                );
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            }
        }
    }

    #[test]
    #[should_panic(expected = "compact attention image must be inside prefix")]
    fn compact_cache_rejects_spatial_keys_in_generated_region() {
        attention_compact(
            &[0.0; 2],
            &[0.0; 2],
            &[0.0],
            &[0.0; 2],
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            0,
            2,
            &[0.0; 2],
            &mut [0.0; 2],
        );
    }
}
