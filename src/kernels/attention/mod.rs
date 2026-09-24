//! Attention entry points: shape checks and dispatch to the fixed-width
//! decode kernels, the prefill tiles and the generic loops.
#[cfg(target_arch = "x86_64")]
use super::panel_bf16;
use super::{Simd, avx2_available, avx512_available, elements, native_vector};
use crate::config::ExpMode;

/// BF16 copies of a prefill's keys and values for the BF16 attention kernel:
/// either written by the fused QKV pass (`Converted`) or converted by the
/// kernel into the caller's scratch buffers (`Convert`).
pub(crate) enum Bf16Kv<'a> {
    /// Keys `[head][total][32]` pairs and value pairs `[kv_head][total / 2][64]`.
    Converted(&'a [u32], &'a [u32]),
    /// Scratch buffers the kernel resizes and fills.
    Convert(&'a mut Vec<u32>, &'a mut Vec<u32>),
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod decode64;
mod online;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) mod prefill64;
mod tiled;

use online::{attention_compact_online_softmax, attention_online_softmax};
use tiled::{attention_gemm, attention_gemm_compact};

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
    assert_eq!(q.len(), elements(query_len, token_width), "attention Q shape");
    assert_eq!(k.len(), elements(kv_len, token_width), "attention K shape");
    assert_eq!(v.len(), k.len(), "attention V shape");
    assert_eq!(output.len(), q.len(), "attention output shape");
    assert_eq!(sinks.len(), n_heads, "attention sink shape");
    assert!(
        image_start <= image_end && image_end <= kv_len,
        "attention image interval"
    );
    assert!(
        query_offset.checked_add(query_len).is_some_and(|end| end <= kv_len),
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
            decode64::attention(q, k, v, n_heads, query_offset, image_start, image_end, sinks, output);
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
/// when the CPU or the shape does not qualify. `exp` selects the softmax exp
/// and `profile` the stage cycle counters.
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
    kv: Bf16Kv<'_>,
    exp: ExpMode,
    profile: bool,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if query_len >= 4
        && head_dim == 64
        && n_heads.is_multiple_of(n_kv_heads.max(1))
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
        && panel_bf16::available()
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
                kv,
                exp,
                profile,
            );
        }
        return true;
    }
    let _ = (q, prefix_k, v, query_len, total_len, n_heads, n_kv_heads, head_dim);
    let _ = (query_offset, image_start, image_end, sinks, output, kv, exp, profile);
    false
}

/// Whether prefill rows can be written straight into the BF16 attention
/// layouts ([`store_prefill_bf16_row`]).
pub(crate) fn prefill_bf16_rows_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        panel_bf16::available() && std::is_x86_feature_detected!("avx512bw")
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
/// Prefill tiles use the platform-exact exp (`ExpMode::Exact`).
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
    attention_compact_prefill_with(
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
        simd,
        ExpMode::Exact,
        false,
    );
}

/// [`attention_compact_with_simd`] with the prefill tiles' exp (`exp`) and
/// stage profiling (`profile`) chosen by the caller; decode paths ignore both.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_compact_prefill_with(
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
    exp: ExpMode,
    profile: bool,
) {
    assert!(
        head_dim > 0 && n_heads > 0 && n_kv_heads > 0,
        "compact attention head dimensions must be positive"
    );
    assert_eq!(n_heads % n_kv_heads, 0, "compact attention GQA grouping");
    assert!(prefix_len <= total_len, "compact attention prefix length");
    let query_width = elements(n_heads, head_dim);
    let kv_width = elements(n_kv_heads, head_dim);
    assert_eq!(q.len(), elements(query_len, query_width), "compact attention Q shape");
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
    assert_eq!(v.len(), elements(total_len, kv_width), "compact attention V shape");
    assert_eq!(output.len(), q.len(), "compact attention output shape");
    assert_eq!(sinks.len(), n_heads, "compact attention sink shape");
    assert!(
        image_start <= image_end && image_end <= prefix_len,
        "compact attention image must be inside prefix"
    );
    assert!(
        query_offset.checked_add(query_len).is_some_and(|end| end <= total_len),
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
                cfg!(target_arch = "x86_64") && matches!(simd, Simd::Auto | Simd::Avx512) && avx512_available(),
                exp,
                profile,
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
    let _ = (exp, profile);
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
            decode64::compact(
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
