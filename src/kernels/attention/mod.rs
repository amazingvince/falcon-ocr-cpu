//! Attention entry points: shape checks and dispatch to the fixed-width
//! decode kernels, the prefill tiles and the generic loops.
//!
//! Every path computes attention without its learned sink, then multiplies
//! by `sigmoid(logsumexp - sink)`, matching the upstream FP32 operation
//! boundaries (fusing a zero-value sink into the denominator is algebraically
//! equivalent but changes intermediate rounding). Online softmax uses bounded
//! logits on the stack per task and the caller's output as its accumulator;
//! prefill uses blocked GEMM (or the fixed-width tiles) for QK and PV, decode
//! uses vector dot products. Nothing materializes a sequence-squared matrix.
#[cfg(target_arch = "x86_64")]
use super::avx512_available;
#[cfg(target_arch = "x86_64")]
use super::panel_bf16;
use super::{Simd, avx2_available, elements, native_vector};
use crate::config::ExpMode;

/// BF16 copies of a prefill's keys and values for the BF16 attention kernel:
/// either written by the fused QKV pass (`Converted`) or converted by the
/// kernel into the caller's scratch buffers (`Convert`).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // only the x86 BF16 kernel reads it
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

/// The masking geometry of one attention call. Absolute query positions
/// begin at `query_offset`; key positions begin at zero. A key is visible if
/// it is causal, or both positions are in the image interval
/// `[image_start, image_end)` (the image-end token itself is excluded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub query_offset: usize,
    pub image_start: usize,
    pub image_end: usize,
}

impl Geometry {
    pub const fn new(query_offset: usize, image_start: usize, image_end: usize) -> Self {
        Self {
            query_offset,
            image_start,
            image_end,
        }
    }
    /// Whether `position` is an image token.
    #[inline]
    pub fn in_image(&self, position: usize) -> bool {
        position >= self.image_start && position < self.image_end
    }
    /// The keys `[0, end)` the query at absolute position `absolute` sees:
    /// its whole image block, or the causal prefix.
    #[inline]
    pub fn visible_end(&self, absolute: usize) -> usize {
        if self.in_image(absolute) {
            self.image_end
        } else {
            absolute + 1
        }
    }
    /// Whether the query at `absolute` must not see `key`: keys after the
    /// query are hidden unless both are image tokens.
    #[inline]
    pub fn masked(&self, absolute: usize, key: usize) -> bool {
        key > absolute && !(self.in_image(absolute) && self.in_image(key))
    }
}

/// A KV cache in the compact layout: prefix keys keep every query head
/// (their spatial rotations differ), generated keys and every value keep only
/// the KV heads; query head `h` uses KV head `h / (n_heads / n_kv_heads)`.
/// All buffers are token-major. An expanded cache is the case with no
/// generated keys and one KV head per query head ([`CompactKv::expanded`]);
/// the kernels compute both with the same operation order.
#[derive(Clone, Copy, Debug)]
pub struct CompactKv<'a> {
    /// `[prefix_len][n_heads][head_dim]`.
    pub prefix_k: &'a [f32],
    /// `[total_len - prefix_len][n_kv_heads][head_dim]`.
    pub generated_k: &'a [f32],
    /// `[total_len][n_kv_heads][head_dim]`.
    pub v: &'a [f32],
    pub prefix_len: usize,
    pub total_len: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<'a> CompactKv<'a> {
    /// A pure prefix (no generated keys): `k` is `[total][n_heads][head_dim]`,
    /// `v` is `[total][n_kv_heads][head_dim]`.
    pub fn prefill(
        k: &'a [f32],
        v: &'a [f32],
        total: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            prefix_k: k,
            generated_k: &[],
            v,
            prefix_len: total,
            total_len: total,
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }
    /// An expanded cache: keys and values `[kv_len][n_heads][head_dim]`.
    pub fn expanded(k: &'a [f32], v: &'a [f32], kv_len: usize, n_heads: usize, head_dim: usize) -> Self {
        Self::prefill(k, v, kv_len, n_heads, n_heads, head_dim)
    }
    #[inline]
    pub fn query_width(&self) -> usize {
        self.n_heads * self.head_dim
    }
    #[inline]
    pub fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    /// Query heads per KV head.
    #[inline]
    pub fn repeat(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    /// The key at `position` for query head `head` (its KV head's key past
    /// the prefix).
    #[inline]
    pub fn key(&self, position: usize, head: usize) -> &'a [f32] {
        if position < self.prefix_len {
            let begin = position * self.query_width() + head * self.head_dim;
            &self.prefix_k[begin..begin + self.head_dim]
        } else {
            let begin = (position - self.prefix_len) * self.kv_width() + head / self.repeat() * self.head_dim;
            &self.generated_k[begin..begin + self.head_dim]
        }
    }
    /// The value at `position` for KV head `kv_head`.
    #[inline]
    pub fn value(&self, position: usize, kv_head: usize) -> &'a [f32] {
        let begin = position * self.kv_width() + kv_head * self.head_dim;
        &self.v[begin..begin + self.head_dim]
    }
    /// Every shape relation the kernels rely on, for `query_len` queries.
    fn validate(&self, q: &[f32], query_len: usize, geometry: &Geometry, sinks: &[f32], output: &[f32]) {
        assert!(
            self.head_dim > 0 && self.n_heads > 0 && self.n_kv_heads > 0,
            "attention head dimensions must be positive"
        );
        assert_eq!(self.n_heads % self.n_kv_heads, 0, "attention GQA grouping");
        assert!(self.prefix_len <= self.total_len, "attention prefix length");
        let (query_width, kv_width) = (
            elements(self.n_heads, self.head_dim),
            elements(self.n_kv_heads, self.head_dim),
        );
        assert_eq!(q.len(), elements(query_len, query_width), "attention Q shape");
        assert_eq!(
            self.prefix_k.len(),
            elements(self.prefix_len, query_width),
            "attention prefix K shape"
        );
        assert_eq!(
            self.generated_k.len(),
            elements(self.total_len - self.prefix_len, kv_width),
            "attention generated K shape"
        );
        assert_eq!(self.v.len(), elements(self.total_len, kv_width), "attention V shape");
        assert_eq!(output.len(), q.len(), "attention output shape");
        assert_eq!(sinks.len(), self.n_heads, "attention sink shape");
        assert!(
            geometry.image_start <= geometry.image_end && geometry.image_end <= self.prefix_len,
            "attention image must be inside the prefix"
        );
        assert!(
            geometry
                .query_offset
                .checked_add(query_len)
                .is_some_and(|end| end <= self.total_len),
            "attention query interval"
        );
    }
}

/// Prefill tile options: the softmax exp and the stage cycle counters.
/// Decode paths ignore both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillOptions {
    pub exp: ExpMode,
    pub profile: bool,
}

impl PrefillOptions {
    /// The platform-exact exp, no counters: the reference configuration.
    pub const EXACT: Self = Self {
        exp: ExpMode::Exact,
        profile: false,
    };
}

/// Attention of `query_len` queries `q` (`[query_len][n_heads][head_dim]`)
/// over `kv`, with automatic vector dispatch.
pub fn attention(
    q: &[f32],
    kv: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
) {
    attention_with_simd(q, kv, query_len, geometry, sinks, output, Simd::Auto);
}

/// [`attention`] with an explicit vector implementation; prefill tiles use
/// the platform-exact exp.
pub fn attention_with_simd(
    q: &[f32],
    kv: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    simd: Simd,
) {
    attention_with(q, kv, query_len, geometry, sinks, output, simd, PrefillOptions::EXACT);
}

/// [`attention_with_simd`] with the prefill tiles' options chosen by the
/// caller.
///
/// Arithmetic order is the same for every layout: a key tile crossing the
/// prefix boundary is gathered before GEMM, preserving its tile width and
/// softmax reduction order. Single-token decode never allocates scratch.
pub(crate) fn attention_with(
    q: &[f32],
    kv: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    simd: Simd,
    options: PrefillOptions,
) {
    kv.validate(q, query_len, &geometry, sinks, output);
    let selected = simd.resolved();
    // The fixed-width prefill tiles: AVX2 on x86 (16-lane tiles under
    // `auto` with AVX-512F), NEON on aarch64; pure prefixes only.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if query_len >= 4
        && kv.head_dim == 64
        && kv.total_len == kv.prefix_len
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
                q,
                kv,
                query_len,
                geometry,
                sinks,
                output,
                super::wide_attention(simd),
                options,
            );
        }
        return;
    }
    let _ = options;
    if query_len >= 4 && selected != Simd::Scalar {
        tiled::attention_gemm(q, kv, geometry, sinks, output);
        return;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if query_len == 1 && kv.head_dim == 64 && selected == native_vector() {
        // SAFETY: the native vector ISA is available and every shape was validated above.
        unsafe { decode64::attention(q, kv, geometry, sinks, output) };
        return;
    }
    online::attention(q, kv, geometry, sinks, output, selected);
}

/// BF16 form of the pure-prefill attention (fast mode on AVX512-BF16 CPUs,
/// see `prefill64::bf16`): Q, K, V and the probabilities round to BF16;
/// scores, softmax and outputs stay FP32. `bf16` supplies the BF16 key and
/// value copies or the scratch to convert them into. Returns false, writing
/// nothing, when the CPU or the shape does not qualify.
pub(crate) fn attention_prefill_bf16(
    q: &[f32],
    kv: &CompactKv<'_>,
    query_len: usize,
    geometry: Geometry,
    sinks: &[f32],
    output: &mut [f32],
    bf16: Bf16Kv<'_>,
    options: PrefillOptions,
) -> bool {
    kv.validate(q, query_len, &geometry, sinks, output);
    #[cfg(target_arch = "x86_64")]
    if query_len >= 4
        && kv.head_dim == 64
        && geometry.query_offset == 0
        && query_len == kv.total_len
        && kv.total_len == kv.prefix_len
        && avx2_available()
        && avx512_available()
        && panel_bf16::available()
    {
        // SAFETY: AVX2/FMA/AVX-512F/AVX512-BF16 detected; shapes validated.
        unsafe {
            prefill64::bf16::compact_prefill(q, kv, query_len, geometry, sinks, output, bf16, options);
        }
        return true;
    }
    let _ = (bf16, options);
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
