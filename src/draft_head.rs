//! A trained draft head for speculative decoding (EAGLE-3 style, text only).
//!
//! The head reads the target's hidden states after three layers at each
//! verified position and predicts the tokens that follow, one small decoder
//! block per drafted token (`research/draft-head/train/eagle3.py`):
//!
//! ```text
//! base i : hidden = fc([rms(f_a), rms(f_b), rms(f_c)])   f_* = target layers at the
//!          position whose output is y_i; key/value of (hidden, y_i) at position i
//! step j : a = [rms(emb(token)), rms(hidden)]            token = y_i, then the drafts
//!          h = hidden + O(Attn(a))                       over bases <= i and this chain
//!          h = h + W2(relu(g)^2 * u)  with (g, u) = W13(rms(h))
//!          logits = Head(rms_w(h))  over the draft vocabulary; hidden <- h
//! ```
//!
//! Drafting stops at the first token whose probability is below the
//! confidence threshold (EAGLE-2's observation that draft confidence tracks
//! acceptance), or, with `path_gate`, once the product of the chain's
//! probabilities is. Weights are int8 with one scale per 64 inputs and run on
//! the same kernels as the model's decode projections; the vocabulary head
//! may be low rank (`head_down`/`head_up`). The keys and values of verified
//! positions are int8 with one scale per 64-value row, group-major, so each
//! of the eight attention tasks streams contiguous memory (a target step
//! evicts the drafter from cache, so bytes decide a draft step's cost); a
//! window can limit attention to the newest positions. Drafts only affect
//! speed: the model verifies every drafted token.
use std::path::Path;

use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};

use crate::{
    config::DraftKv,
    kernels::{self, Simd},
    quant::linear::{QuantLinear, Scratch},
};

const DIM: usize = 768;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const FFN: usize = 2304;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const QKV: usize = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
const FORMAT: &str = "falcon-ocr-draft-head-v1";
const GROUP: usize = 64;

/// The loaded head (shared by every page of a runner).
pub(crate) struct DraftHead {
    layers: [usize; 3],
    fc: QuantLinear,
    qkv: QuantLinear,
    o: QuantLinear,
    w13: QuantLinear,
    w2: QuantLinear,
    head: Head,
    norm: Vec<f32>,
    vocab: Vec<u32>,
    /// Rotary inverse frequencies for the 32 rotated pairs of a 64-wide head.
    inv_freq: [f32; HEAD_DIM / 2],
}

/// How a page's head drafts (`RunnerConfig::draft_confidence`, `Tuning`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DraftOptions {
    /// Keep drafting while the top token's probability is at least this.
    pub(crate) confidence: f32,
    /// After consecutive attempts that drafted nothing, skip the next 1, 2,
    /// 4, ... (at most this many) tokens; 0 tries at every token.
    pub(crate) backoff: usize,
    /// Attend to at most this many newest verified positions (0: all).
    pub(crate) window: usize,
    /// Storage of the verified keys and values.
    pub(crate) kv: DraftKv,
    /// Stop when the product of the chain's probabilities (EAGLE-2's path
    /// probability) falls below `confidence`, rather than one token's.
    pub(crate) path_gate: bool,
}

/// The vocabulary projection: dense, or low rank (`up(down(x))`, written by
/// `lowrank_head.py`).
enum Head {
    Dense(QuantLinear),
    Factored {
        down: QuantLinear,
        up: QuantLinear,
        rank: usize,
    },
}

impl Head {
    fn logits(
        &self,
        x: &[f32],
        low: &mut Vec<f32>,
        logits: &mut [f32],
        scratch: &mut Scratch,
        simd: Simd,
    ) -> Result<()> {
        match self {
            Self::Dense(w) => w.linear(x, 1, logits, scratch, simd),
            Self::Factored { down, up, rank } => {
                low.resize(*rank, 0.0);
                down.linear(x, 1, low, scratch, simd)?;
                up.linear(low, 1, logits, scratch, simd)
            }
        }
    }
}

/// The index of the largest logit and its softmax probability.
fn top_probability(logits: &[f32], simd: Simd) -> (usize, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let best = logits.iter().position(|&x| x == max).unwrap_or(0);
    (best, exp_sum_kernel(simd)(logits, max).recip())
}

type ExpSum = fn(&[f32], f32) -> f32;

fn exp_sum_kernel(simd: Simd) -> ExpSum {
    match simd.resolved() {
        // SAFETY: `resolved` returns Avx2 only when AVX2 and FMA are available.
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => |x, shift| unsafe { exp_sum_avx2(x, shift) },
        // SAFETY: NEON is baseline on aarch64.
        #[cfg(target_arch = "aarch64")]
        Simd::Neon => |x, shift| unsafe { exp_sum::<crate::simd::Neon>(x, shift) },
        // SAFETY: the portable backend needs no CPU features.
        _ => |x, shift| unsafe { exp_sum::<crate::simd::Portable>(x, shift) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_sum_avx2(x: &[f32], shift: f32) -> f32 {
    unsafe { exp_sum::<crate::simd::Avx2Fast>(x, shift) }
}

/// `sum(exp(x[i] - shift))` with the polynomial exp.
#[inline(always)]
unsafe fn exp_sum<S: crate::simd::Simd>(x: &[f32], shift: f32) -> f32 {
    // SAFETY: loads stay inside `x`; the caller guarantees the backend.
    unsafe {
        let s = S::splat(shift);
        let mut acc = S::zero();
        let mut chunks = x.chunks_exact(8);
        for chunk in &mut chunks {
            acc = S::add(acc, S::exp_fast(S::sub(S::load(chunk.as_ptr()), s)));
        }
        S::sum(acc)
            + chunks
                .remainder()
                .iter()
                .map(|&v| crate::simd::exp_poly(v - shift))
                .sum::<f32>()
    }
}

/// One page's draft cache: the key/value of every verified position, and
/// the fused hidden state, token and query of the newest one (the first draft
/// step's query is exactly that one, so it is kept rather than recomputed).
pub(crate) struct DraftState {
    options: DraftOptions,
    kv: Kv,
    len: usize,
    hidden: Vec<f32>,
    token: u32,
    query: Vec<f32>,
    /// Consecutive attempts that drafted nothing, and tokens left to skip.
    failures: u32,
    skip: usize,
    scratch: Scratch,
    buf: Buffers,
}

/// The key and value of every verified position.
enum Kv {
    /// FP32 rows `[position][512]`.
    F32 { keys: Vec<f32>, values: Vec<f32> },
    /// Int8 codes with one scale per 64-value row, KV-group-major (group g's
    /// row t is `keys[g][64 t..]`): a quarter of the FP32 bytes, and each
    /// group's attention task streams contiguous memory.
    Q8(Box<Q8Kv>),
}

#[derive(Default)]
struct Q8Kv {
    keys: [Vec<i8>; KV_HEADS],
    key_scales: [Vec<f32>; KV_HEADS],
    values: [Vec<i8>; KV_HEADS],
    value_scales: [Vec<f32>; KV_HEADS],
}

impl Kv {
    /// Append one position's key and value (`512` each).
    fn push(&mut self, key: &[f32], value: &[f32]) {
        match self {
            Self::F32 { keys, values } => {
                keys.extend_from_slice(key);
                values.extend_from_slice(value);
            }
            Self::Q8(q) => {
                for g in 0..KV_HEADS {
                    let at = g * HEAD_DIM..(g + 1) * HEAD_DIM;
                    quantize_row(&key[at.clone()], &mut q.keys[g], &mut q.key_scales[g]);
                    quantize_row(&value[at], &mut q.values[g], &mut q.value_scales[g]);
                }
            }
        }
    }
}

/// Append `x` as int8 codes with one absmax scale (`x ~ code * scale`).
fn quantize_row(x: &[f32], codes: &mut Vec<i8>, scales: &mut Vec<f32>) {
    let max = x.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let inverse = if max > 0.0 { 127.0 / max } else { 0.0 };
    codes.extend(x.iter().map(|&v| (v * inverse).round().clamp(-127.0, 127.0) as i8));
    scales.push(max / 127.0);
}

impl DraftState {
    pub(crate) fn new(options: DraftOptions) -> Self {
        Self {
            options,
            kv: match options.kv {
                DraftKv::F32 => Kv::F32 {
                    keys: Vec::new(),
                    values: Vec::new(),
                },
                DraftKv::Q8 => Kv::Q8(Box::default()),
            },
            len: 0,
            hidden: vec![0.0; DIM],
            token: 0,
            query: vec![0.0; HEADS * HEAD_DIM],
            failures: 0,
            skip: 0,
            scratch: Scratch::default(),
            buf: Buffers::default(),
        }
    }
}

#[derive(Default)]
struct Buffers {
    fused_in: Vec<f32>,
    a: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    h: Vec<f32>,
    projected: Vec<f32>,
    normed: Vec<f32>,
    ffn: Vec<f32>,
    gated: Vec<f32>,
    logits: Vec<f32>,
    low: Vec<f32>,
    chain_keys: Vec<f32>,
    chain_values: Vec<f32>,
    scores: Vec<f32>,
}

fn tensor_f32(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = tensors
        .tensor(name)
        .with_context(|| format!("draft head tensor {name}"))?;
    ensure!(
        t.dtype() == Dtype::F32 && t.shape() == shape,
        "draft head {name}: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    Ok(t.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn quantized(tensors: &SafeTensors<'_>, name: &str, out_dim: usize, in_dim: usize) -> Result<QuantLinear> {
    let w = tensor_f32(tensors, name, &[out_dim, in_dim])?;
    QuantLinear::quantize(&w, out_dim, in_dim, GROUP).map_err(|e| anyhow::anyhow!("draft head {name}: {e}"))
}

fn rms(src: &[f32], dst: &mut [f32]) {
    kernels::rms_norm(src, dst, src.len(), f32::EPSILON, None);
}

impl DraftHead {
    /// Load an exported head (`export_head.py`).
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading draft head {}", path.display()))?;
        ensure!(bytes.len() >= 8, "short draft head file");
        let len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        let end = 8usize.checked_add(len).context("draft head header overflow")?;
        ensure!(end <= bytes.len(), "draft head header out of bounds");
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..end])?;
        let meta = |key: &str| header["__metadata__"][key].as_str();
        ensure!(
            meta("format") == Some(FORMAT),
            "{} is not a {FORMAT} file",
            path.display()
        );
        ensure!(
            meta("image") == Some("false"),
            "only text-only draft heads are supported"
        );
        let layers: Vec<usize> = meta("layers")
            .context("draft head layers missing")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(layers.len() == 3, "draft head needs three feature layers");
        let tensors = SafeTensors::deserialize(&bytes)?;
        let vocab_view = tensors.tensor("vocab_ids")?;
        ensure!(vocab_view.dtype() == Dtype::I32, "draft vocab must be int32");
        let vocab: Vec<u32> = vocab_view
            .data()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .collect();
        let v = vocab.len();
        let theta = 10000_f32;
        Ok(Self {
            layers: [layers[0], layers[1], layers[2]],
            fc: quantized(&tensors, "fc.weight", DIM, 3 * DIM)?,
            qkv: quantized(&tensors, "qkv.weight", QKV, 2 * DIM)?,
            o: quantized(&tensors, "o.weight", DIM, HEADS * HEAD_DIM)?,
            w13: quantized(&tensors, "w13.weight", 2 * FFN, DIM)?,
            w2: quantized(&tensors, "w2.weight", DIM, FFN)?,
            head: match tensors.tensor("head_down.weight") {
                Ok(down) => {
                    let rank = down.shape().first().copied().unwrap_or(0);
                    ensure!(
                        rank > 0 && rank % GROUP == 0,
                        "draft head rank {rank} is not a multiple of {GROUP}"
                    );
                    Head::Factored {
                        down: quantized(&tensors, "head_down.weight", rank, DIM)?,
                        up: quantized(&tensors, "head_up.weight", v, rank)?,
                        rank,
                    }
                }
                Err(_) => Head::Dense(quantized(&tensors, "head.weight", v, DIM)?),
            },
            norm: tensor_f32(&tensors, "norm.weight", &[DIM])?,
            vocab,
            inv_freq: std::array::from_fn(|i| theta.powf(-(i as f32) / (HEAD_DIM / 2) as f32)),
        })
    }

    /// The target layers whose outputs the head reads (0-based).
    pub(crate) fn layers(&self) -> [usize; 3] {
        self.layers
    }

    /// Rotate one 64-wide head by position `pos` (first half with second half).
    fn rope(&self, x: &mut [f32], pos: usize) {
        let half = HEAD_DIM / 2;
        for i in 0..half {
            let (sin, cos) = (pos as f32 * self.inv_freq[i]).sin_cos();
            let (x1, x2) = (x[i], x[i + half]);
            x[i] = x1 * cos - x2 * sin;
            x[i + half] = x1 * sin + x2 * cos;
        }
    }

    /// QKV of `rows` (`hidden` row r with `embedding(r)`, at position
    /// `pos + r`) into `buf.qkv`, with per-head norms on queries and keys and
    /// rotary positions applied. Several rows share one weight pass.
    fn qkv<'e>(
        &self,
        st: &mut DraftState,
        rows: usize,
        hidden: &[f32],
        embedding: impl Fn(usize) -> &'e [f32],
        pos: usize,
        simd: Simd,
    ) -> Result<()> {
        let b = &mut st.buf;
        b.a.resize(rows * 2 * DIM, 0.0);
        for r in 0..rows {
            let a = &mut b.a[r * 2 * DIM..(r + 1) * 2 * DIM];
            rms(embedding(r), &mut a[..DIM]);
            rms(&hidden[r * DIM..(r + 1) * DIM], &mut a[DIM..]);
        }
        b.qkv.resize(rows * QKV, 0.0);
        self.qkv.linear(&b.a, rows, &mut b.qkv, &mut st.scratch, simd)?;
        let mut head = [0.0_f32; HEAD_DIM];
        for r in 0..rows {
            for h in 0..HEADS + KV_HEADS {
                let x = &mut b.qkv[r * QKV + h * HEAD_DIM..r * QKV + (h + 1) * HEAD_DIM];
                rms(x, &mut head);
                x.copy_from_slice(&head);
                self.rope(x, pos + r);
            }
        }
        Ok(())
    }

    /// Record verified positions in order: `features(r)` are the target's
    /// three layers (`3 * 768`) at the position whose output is `tokens[r]`,
    /// `embedding(t)` a token's input embedding. One weight pass for all
    /// rows (a verify step yields up to eight).
    pub(crate) fn push_rows<'f, 'e>(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        features: impl Fn(usize) -> &'f [f32],
        embedding: impl Fn(u32) -> &'e [f32],
        simd: Simd,
    ) -> Result<()> {
        let rows = tokens.len();
        if rows == 0 {
            return Ok(());
        }
        st.buf.fused_in.resize(rows * 3 * DIM, 0.0);
        for r in 0..rows {
            let f = features(r);
            ensure!(f.len() == 3 * DIM, "draft head feature shape");
            for l in 0..3 {
                let at = (r * 3 + l) * DIM;
                rms(&f[l * DIM..(l + 1) * DIM], &mut st.buf.fused_in[at..at + DIM]);
            }
        }
        let mut hidden = std::mem::take(&mut st.buf.h);
        hidden.resize(rows * DIM, 0.0);
        self.fc
            .linear(&st.buf.fused_in, rows, &mut hidden, &mut st.scratch, simd)?;
        let pos = st.len;
        self.qkv(st, rows, &hidden, |r| embedding(tokens[r]), pos, simd)?;
        let k = HEADS * HEAD_DIM;
        for r in 0..rows {
            let row = &st.buf.qkv[r * QKV + k..][..2 * KV_WIDTH];
            st.kv.push(&row[..KV_WIDTH], &row[KV_WIDTH..]);
        }
        let last = rows - 1;
        st.query.copy_from_slice(&st.buf.qkv[last * QKV..last * QKV + k]);
        st.hidden.copy_from_slice(&hidden[last * DIM..(last + 1) * DIM]);
        st.buf.h = hidden;
        st.token = tokens[last];
        st.len += rows;
        Ok(())
    }

    /// Up to `limit` tokens after the newest verified position, each drafted
    /// only while its probability is at least `options.confidence` (`out` is
    /// cleared first). `embed` gives a token's input embedding. With
    /// `options.backoff > 0`, consecutive attempts that draft nothing make
    /// the next 1, 2, 4, ... (at most `backoff`) calls skip the attempt.
    pub(crate) fn draft<'e>(
        &self,
        st: &mut DraftState,
        limit: usize,
        embed: impl Fn(u32) -> &'e [f32],
        out: &mut Vec<u32>,
        simd: Simd,
    ) -> Result<()> {
        out.clear();
        if st.len == 0 || limit == 0 {
            return Ok(());
        }
        if st.skip > 0 {
            st.skip -= 1;
            return Ok(());
        }
        self.draft_chain(st, limit, embed, out, simd)?;
        let backoff = st.options.backoff;
        if out.is_empty() {
            st.failures += 1;
            if backoff > 0 {
                st.skip = (1_usize << (st.failures - 1).min(16)).min(backoff);
            }
        } else {
            st.failures = 0;
        }
        Ok(())
    }

    fn draft_chain<'e>(
        &self,
        st: &mut DraftState,
        limit: usize,
        embed: impl Fn(u32) -> &'e [f32],
        out: &mut Vec<u32>,
        simd: Simd,
    ) -> Result<()> {
        let base = st.len - 1;
        let (confidence, path_gate) = (st.options.confidence, st.options.path_gate);
        let mut path = 1.0_f32;
        let mut hidden = st.hidden.clone();
        let mut token = st.token;
        st.buf.chain_keys.clear();
        st.buf.chain_values.clear();
        for j in 1..=limit {
            let pos = base + j - 1;
            let k = HEADS * HEAD_DIM;
            if j == 1 {
                // The newest verified position's own query.
                st.buf.qkv.resize(QKV, 0.0);
                st.buf.qkv[..k].copy_from_slice(&st.query);
            } else {
                self.qkv(st, 1, &hidden, |_| embed(token), pos, simd)?;
            }
            if j > 1 {
                let (keys, values) = (&mut st.buf.chain_keys, &mut st.buf.chain_values);
                keys.extend_from_slice(&st.buf.qkv[k..k + KV_WIDTH]);
                values.extend_from_slice(&st.buf.qkv[k + KV_WIDTH..k + 2 * KV_WIDTH]);
            }
            attend(st, base + 1, simd);
            // h = hidden + O(attn); h += W2(gate(W13(rms(h)))).
            let b = &mut st.buf;
            b.projected.resize(DIM, 0.0);
            self.o.linear(&b.attn, 1, &mut b.projected, &mut st.scratch, simd)?;
            b.h.clear();
            b.h.extend(hidden.iter().zip(&b.projected).map(|(x, y)| x + y));
            b.normed.resize(DIM, 0.0);
            rms(&b.h, &mut b.normed);
            b.ffn.resize(2 * FFN, 0.0);
            self.w13.linear(&b.normed, 1, &mut b.ffn, &mut st.scratch, simd)?;
            b.gated.clear();
            b.gated
                .extend(b.ffn.chunks_exact(2).map(|p| kernels::squared_relu_glu(p[0], p[1])));
            self.w2.linear(&b.gated, 1, &mut b.projected, &mut st.scratch, simd)?;
            for (x, y) in b.h.iter_mut().zip(&b.projected) {
                *x += y;
            }
            kernels::rms_norm(&b.h, &mut b.normed, DIM, 1e-5, Some(&self.norm));
            b.logits.resize(self.vocab.len(), 0.0);
            self.head
                .logits(&b.normed, &mut b.low, &mut b.logits, &mut st.scratch, simd)?;
            let (best, p) = top_probability(&b.logits, simd);
            path *= p;
            if (if path_gate { path } else { p }) < confidence {
                break;
            }
            token = self.vocab[best];
            out.push(token);
            std::mem::swap(&mut hidden, &mut b.h);
        }
        Ok(())
    }
}

/// Attention of the current query (`buf.qkv` queries) over the first
/// `bases` verified keys (only the newest `options.window` of them when set)
/// and the chain keys, into `buf.attn`; each of the eight key/value groups
/// (two query heads) is one task.
fn attend(st: &mut DraftState, bases: usize, simd: Simd) {
    let start = match st.options.window {
        0 => 0,
        window => bases.saturating_sub(window),
    };
    let n = bases - start;
    let b = &mut st.buf;
    let total = n + b.chain_keys.len() / KV_WIDTH;
    b.attn.clear();
    b.attn.resize(HEADS * HEAD_DIM, 0.0);
    b.scores.resize(HEADS * total, 0.0);
    let q = &b.qkv[..HEADS * HEAD_DIM];
    let (chain_keys, chain_values) = (&b.chain_keys[..], &b.chain_values[..]);
    let out = crate::team::SharedMut::new(&mut b.attn);
    let scores = crate::team::SharedMut::new(&mut b.scores);
    match &st.kv {
        Kv::F32 { keys, values } => {
            let dot = kernels::dot_kernel(simd.resolved());
            let axpy = kernels::axpy_kernel(simd.resolved());
            let scale = (HEAD_DIM as f32).sqrt().recip();
            crate::team::for_each(KV_HEADS, |g| {
                let key = |t: usize| {
                    let (src, row) = if t < n {
                        (&keys[..], start + t)
                    } else {
                        (chain_keys, t - n)
                    };
                    &src[row * KV_WIDTH + g * HEAD_DIM..][..HEAD_DIM]
                };
                let value = |t: usize| {
                    let (src, row) = if t < n {
                        (&values[..], start + t)
                    } else {
                        (chain_values, t - n)
                    };
                    &src[row * KV_WIDTH + g * HEAD_DIM..][..HEAD_DIM]
                };
                for h in [2 * g, 2 * g + 1] {
                    // SAFETY: head h's scores and output are touched by this task only.
                    let (s, o) = unsafe { (scores.slice(h * total, total), out.slice(h * HEAD_DIM, HEAD_DIM)) };
                    let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                    let mut max = f32::NEG_INFINITY;
                    for (t, s) in s.iter_mut().enumerate() {
                        *s = dot(qh, key(t)) * scale;
                        max = max.max(*s);
                    }
                    let mut sum = 0.0;
                    for s in s.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (t, &p) in s.iter().enumerate() {
                        axpy(p / sum, value(t), o);
                    }
                }
            });
        }
        Kv::Q8(kv) => {
            let kernel = q8_group_kernel(simd);
            crate::team::for_each(KV_HEADS, |g| {
                let rows = start * HEAD_DIM..bases * HEAD_DIM;
                let job = GroupJob {
                    q: &q[2 * g * HEAD_DIM..(2 * g + 2) * HEAD_DIM],
                    keys: &kv.keys[g][rows.clone()],
                    key_scales: &kv.key_scales[g][start..bases],
                    values: &kv.values[g][rows],
                    value_scales: &kv.value_scales[g][start..bases],
                    chain_keys,
                    chain_values,
                    group: g,
                };
                // SAFETY: group g's two heads' scores and outputs are touched
                // by this task only.
                let (s, o) = unsafe {
                    (
                        scores.slice(2 * g * total, 2 * total),
                        out.slice(2 * g * HEAD_DIM, 2 * HEAD_DIM),
                    )
                };
                kernel(&job, s, o);
            });
        }
    }
}

/// One KV group's inputs to [`group_q8`].
struct GroupJob<'a> {
    /// The group's two query heads (`2 * 64`).
    q: &'a [f32],
    /// Int8 rows of the attended verified positions and their scales.
    keys: &'a [i8],
    key_scales: &'a [f32],
    values: &'a [i8],
    value_scales: &'a [f32],
    /// FP32 chain rows `[r][512]`; this group's are at `group * 64`.
    chain_keys: &'a [f32],
    chain_values: &'a [f32],
    group: usize,
}

type GroupKernel = fn(&GroupJob<'_>, &mut [f32], &mut [f32]);

fn q8_group_kernel(simd: Simd) -> GroupKernel {
    match simd.resolved() {
        // SAFETY: `resolved` returns Avx2 only when AVX2 and FMA are available.
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => |job, s, o| unsafe { group_q8_avx2(job, s, o) },
        // SAFETY: NEON is baseline on aarch64.
        #[cfg(target_arch = "aarch64")]
        Simd::Neon => |job, s, o| unsafe { group_q8::<crate::simd::Neon>(job, s, o) },
        // SAFETY: the portable backend needs no CPU features.
        _ => |job, s, o| unsafe { group_q8::<crate::simd::Portable>(job, s, o) },
    }
}

/// [`group_q8`] under AVX2/FMA with the polynomial exp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn group_q8_avx2(job: &GroupJob<'_>, scores: &mut [f32], out: &mut [f32]) {
    unsafe { group_q8::<crate::simd::Avx2Fast>(job, scores, out) }
}

/// Both heads' dots with one 64-wide key (`key(c)` gives lanes `c..c + 8`).
#[inline(always)]
unsafe fn dot2<S: crate::simd::Simd>(q0: *const f32, q1: *const f32, key: impl Fn(usize) -> S::V) -> (f32, f32) {
    unsafe {
        let (mut a0, mut a1) = (S::zero(), S::zero());
        for c in (0..HEAD_DIM).step_by(8) {
            let k = key(c);
            a0 = S::fma(S::load(q0.add(c)), k, a0);
            a1 = S::fma(S::load(q1.add(c)), k, a1);
        }
        (S::sum(a0), S::sum(a1))
    }
}

/// `o[h] += p_h * value` for both heads (`value(c)` gives lanes `c..c + 8`).
#[inline(always)]
unsafe fn pv2<S: crate::simd::Simd>(
    o: &mut [[S::V; HEAD_DIM / 8]; 2],
    p0: f32,
    p1: f32,
    value: impl Fn(usize) -> S::V,
) {
    unsafe {
        let (p0, p1) = (S::splat(p0), S::splat(p1));
        let [o0, o1] = o;
        for (c, (a0, a1)) in o0.iter_mut().zip(o1).enumerate() {
            let v = value(8 * c);
            *a0 = S::fma(p0, v, *a0);
            *a1 = S::fma(p1, v, *a1);
        }
    }
}

/// One KV group's attention: the scores of heads 2g and 2g+1 (`scores`,
/// `[2][n]`) over the int8 verified rows, then the FP32 chain rows; softmax;
/// both heads' normalized outputs (`out`, `[2][64]`). Each key and value row
/// is loaded once for both heads; a key's scale multiplies its dots, a
/// value's scale folds into its probabilities.
#[inline(always)]
unsafe fn group_q8<S: crate::simd::Simd>(job: &GroupJob<'_>, scores: &mut [f32], out: &mut [f32]) {
    let bases = job.key_scales.len();
    let chain = job.chain_keys.len() / KV_WIDTH;
    let n = bases + chain;
    assert!(scores.len() == 2 * n && out.len() == 2 * HEAD_DIM && job.q.len() == 2 * HEAD_DIM);
    assert!(job.keys.len() == bases * HEAD_DIM && job.values.len() == bases * HEAD_DIM);
    assert!(job.value_scales.len() == bases && job.chain_values.len() == job.chain_keys.len());
    assert!(chain == 0 || (job.group + 1) * HEAD_DIM <= KV_WIDTH);
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let (s0, s1) = scores.split_at_mut(n);
    // SAFETY: every pointer below stays inside the slices checked above; the
    // caller guarantees the backend's CPU features.
    unsafe {
        let (q0, q1) = (job.q.as_ptr(), job.q.as_ptr().add(HEAD_DIM));
        for t in 0..bases {
            let k = job.keys.as_ptr().add(t * HEAD_DIM);
            let (d0, d1) = dot2::<S>(q0, q1, |c| S::load_i8(k.add(c)));
            let f = job.key_scales[t] * scale;
            s0[t] = d0 * f;
            s1[t] = d1 * f;
        }
        for r in 0..chain {
            let k = job.chain_keys.as_ptr().add(r * KV_WIDTH + job.group * HEAD_DIM);
            let (d0, d1) = dot2::<S>(q0, q1, |c| S::load(k.add(c)));
            s0[bases + r] = d0 * scale;
            s1[bases + r] = d1 * scale;
        }
        let max = |s: &[f32]| s.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        let (m0, m1) = (max(s0), max(s1));
        S::exp_shifted(s0, m0);
        S::exp_shifted(s1, m1);
        let (sum0, sum1): (f32, f32) = (s0.iter().sum(), s1.iter().sum());
        let mut o = [[S::zero(); HEAD_DIM / 8]; 2];
        for t in 0..bases {
            let v = job.values.as_ptr().add(t * HEAD_DIM);
            let f = job.value_scales[t];
            pv2::<S>(&mut o, s0[t] * f, s1[t] * f, |c| S::load_i8(v.add(c)));
        }
        for r in 0..chain {
            let v = job.chain_values.as_ptr().add(r * KV_WIDTH + job.group * HEAD_DIM);
            pv2::<S>(&mut o, s0[bases + r], s1[bases + r], |c| S::load(v.add(c)));
        }
        for (h, (o, sum)) in o.iter().zip([sum0, sum1]).enumerate() {
            let inverse = S::splat(sum.recip());
            for (c, &v) in o.iter().enumerate() {
                S::store(out.as_mut_ptr().add(h * HEAD_DIM + 8 * c), S::mul(v, inverse));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(kv: DraftKv, window: usize) -> DraftOptions {
        DraftOptions {
            confidence: 0.0,
            backoff: 0,
            window,
            kv,
            path_gate: false,
        }
    }

    /// The vectorized top-1 probability matches the scalar softmax.
    #[test]
    fn top_probability_matches_scalar_softmax() {
        for n in [1, 7, 8, 100, 16383] {
            let logits: Vec<f32> = (0..n).map(|i| ((i * 7919 % 1009) as f32 - 504.0) / 40.0).collect();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let best = logits.iter().position(|&x| x == max).unwrap();
            let p = 1.0 / logits.iter().map(|&x| f64::from(x - max).exp()).sum::<f64>();
            for simd in [Simd::Scalar, Simd::Auto] {
                let (b, q) = top_probability(&logits, simd);
                assert_eq!(b, best);
                assert!(
                    (f64::from(q) - p).abs() < 1e-5 * p.max(1e-3),
                    "{n} {simd:?}: {q} vs {p}"
                );
            }
        }
    }

    /// Int8 draft attention stays close to FP32 attention on every backend,
    /// and a window attends to exactly the newest positions.
    #[test]
    fn int8_attention_and_window() {
        let rows = 300;
        let chain = 2;
        let x = |i: usize| ((i * 7919 % 1009) as f32 - 504.0) / 250.0;
        let keys: Vec<f32> = (0..rows * KV_WIDTH).map(|i| x(3 * i + 1)).collect();
        let values: Vec<f32> = (0..rows * KV_WIDTH).map(|i| x(5 * i + 2)).collect();
        let q: Vec<f32> = (0..QKV).map(|i| x(7 * i + 3)).collect();
        let chain_keys: Vec<f32> = (0..chain * KV_WIDTH).map(|i| x(11 * i + 4)).collect();
        let chain_values: Vec<f32> = (0..chain * KV_WIDTH).map(|i| x(13 * i + 5)).collect();
        let run = |kv, window, simd| {
            let mut st = DraftState::new(options(kv, window));
            for t in 0..rows {
                let at = t * KV_WIDTH..(t + 1) * KV_WIDTH;
                st.kv.push(&keys[at.clone()], &values[at]);
            }
            st.len = rows;
            st.buf.qkv = q.clone();
            st.buf.chain_keys = chain_keys.clone();
            st.buf.chain_values = chain_values.clone();
            attend(&mut st, rows, simd);
            st.buf.attn
        };
        // Double-precision reference over positions start..rows and the chain.
        let reference = |start: usize| {
            let mut out = vec![0.0_f32; HEADS * HEAD_DIM];
            for h in 0..HEADS {
                let g = h / 2;
                let row = |t: usize| -> (&[f32], &[f32]) {
                    let (k, v, r) = if t < rows {
                        (&keys, &values, t)
                    } else {
                        (&chain_keys, &chain_values, t - rows)
                    };
                    let at = r * KV_WIDTH + g * HEAD_DIM..r * KV_WIDTH + (g + 1) * HEAD_DIM;
                    (&k[at.clone()], &v[at])
                };
                let positions: Vec<usize> = (start..rows + chain).collect();
                let scores: Vec<f64> = positions
                    .iter()
                    .map(|&t| {
                        let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                        qh.iter()
                            .zip(row(t).0)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                            / 8.0
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for (w, &t) in weights.iter().zip(&positions) {
                    for (o, v) in out[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter_mut().zip(row(t).1) {
                        *o += (w / sum * f64::from(*v)) as f32;
                    }
                }
            }
            out
        };
        let error = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        for window in [0, 1, 64, rows, 1000] {
            let expected = reference(if window == 0 { 0 } else { rows.saturating_sub(window) });
            let exact = run(DraftKv::F32, window, Simd::Scalar);
            assert!(
                error(&exact, &expected) < 1e-4,
                "f32 window {window}: {}",
                error(&exact, &expected)
            );
            for simd in [Simd::Scalar, Simd::Auto] {
                let int8 = run(DraftKv::Q8, window, simd);
                let e = error(&int8, &expected);
                assert!(e < 0.03, "int8 window {window} {simd:?}: {e}");
            }
        }
    }

    /// Where a draft step's time goes on a 12-thread decode team with 1,500
    /// verified positions: each projection alone, attention (FP32 or int8
    /// keys and values, full or windowed), a whole step, and the feed of one
    /// position.
    #[test]
    #[ignore = "timing probe; needs FALCON_OCR_DRAFT_HEAD; run in release with --nocapture"]
    fn draft_step_profile() {
        let path = std::path::PathBuf::from(std::env::var("FALCON_OCR_DRAFT_HEAD").expect("FALCON_OCR_DRAFT_HEAD"));
        let head = DraftHead::load(&path).unwrap();
        let team = crate::team::Team::new(12).unwrap();
        let _entered = team.enter();
        let simd = Simd::Auto;
        let embedding = vec![0.01_f32; DIM];
        let features: Vec<f32> = (0..3 * DIM).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let variants = [
            ("f32", DraftKv::F32, 0),
            ("int8", DraftKv::Q8, 0),
            ("int8 window 1024", DraftKv::Q8, 1024),
            ("int8 window 256", DraftKv::Q8, 256),
        ];
        let mut states: Vec<DraftState> = variants
            .iter()
            .map(|&(_, kv, window)| {
                let mut st = DraftState::new(options(kv, window));
                for _ in 0..1500 {
                    head.push_rows(&mut st, &[42], |_| &features, |_| &embedding, simd)
                        .unwrap();
                }
                st
            })
            .collect();
        let time = |label: &str, reps: usize, f: &mut dyn FnMut()| {
            f();
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            println!("{label:28} {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        };
        let mut scratch = Scratch::default();
        let (x768, x1536, x1024, x2304) = (
            vec![0.1_f32; DIM],
            vec![0.1_f32; 2 * DIM],
            vec![0.1_f32; 1024],
            vec![0.1_f32; FFN],
        );
        let mut out = vec![0.0_f32; 32768];
        time("fc      2304 -> 768", 300, &mut || {
            head.fc.linear(&x2304, 1, &mut out[..DIM], &mut scratch, simd).unwrap()
        });
        time("qkv     1536 -> 2048", 300, &mut || {
            head.qkv.linear(&x1536, 1, &mut out[..QKV], &mut scratch, simd).unwrap()
        });
        time("o       1024 -> 768", 300, &mut || {
            head.o.linear(&x1024, 1, &mut out[..DIM], &mut scratch, simd).unwrap()
        });
        time("w13      768 -> 4608", 300, &mut || {
            head.w13
                .linear(&x768, 1, &mut out[..2 * FFN], &mut scratch, simd)
                .unwrap()
        });
        time("w2      2304 -> 768", 300, &mut || {
            head.w2.linear(&x2304, 1, &mut out[..DIM], &mut scratch, simd).unwrap()
        });
        let v = head.vocab.len();
        let mut low = Vec::new();
        time("head     768 -> vocab", 300, &mut || {
            head.head
                .logits(&x768, &mut low, &mut out[..v], &mut scratch, simd)
                .unwrap()
        });
        time("softmax top-1 over vocab", 300, &mut || {
            std::hint::black_box(top_probability(&out[..v], simd));
        });
        let mut drafted = Vec::new();
        for ((label, ..), st) in variants.iter().zip(&mut states) {
            st.buf.qkv.resize(QKV, 0.1);
            time(&format!("attention {label}"), 300, &mut || attend(st, 1500, simd));
            time(&format!("draft step {label}"), 300, &mut || {
                head.draft(st, 1, |_| &embedding, &mut drafted, simd).unwrap()
            });
        }
        let mut st2 = DraftState::new(options(DraftKv::F32, 0));
        time("feed one position", 300, &mut || {
            head.push_rows(&mut st2, &[42], |_| &features, |_| &embedding, simd)
                .unwrap()
        });
        // Cold: stream 512 MB (a target decode step moves ~400 MB) before each
        // timed call, as in real decoding, where the drafter's weights and
        // keys do not survive the target step in cache.
        let flush: Vec<u64> = vec![1; 64 << 20];
        let cold = |label: &str, f: &mut dyn FnMut()| {
            let mut total = 0.0;
            for _ in 0..40 {
                std::hint::black_box(flush.iter().fold(0u64, |a, &x| a.wrapping_add(x)));
                let t = std::time::Instant::now();
                f();
                total += t.elapsed().as_secs_f64();
            }
            println!("{label:28} {:.3} ms (cold)", total * 1e3 / 40.0);
        };
        cold("head     768 -> vocab", &mut || {
            head.head
                .logits(&x768, &mut low, &mut out[..v], &mut scratch, simd)
                .unwrap()
        });
        for ((label, ..), st) in variants.iter().zip(&mut states) {
            cold(&format!("attention {label}"), &mut || attend(st, 1500, simd));
            cold(&format!("draft step {label}"), &mut || {
                head.draft(st, 1, |_| &embedding, &mut drafted, simd).unwrap()
            });
        }
        cold("feed one position f32", &mut || {
            head.push_rows(&mut st2, &[42], |_| &features, |_| &embedding, simd)
                .unwrap()
        });
        let mut st3 = DraftState::new(options(DraftKv::Q8, 0));
        cold("feed one position int8", &mut || {
            head.push_rows(&mut st3, &[42], |_| &features, |_| &embedding, simd)
                .unwrap()
        });
    }

    /// The Rust head reproduces the PyTorch head's draft chain on the exported
    /// fixture (int8 weights here, FP32 there: same tokens, close logits).
    /// Needs `FALCON_OCR_DRAFT_HEAD=<exported .safetensors>` and the model's
    /// embedding table from artifacts/model.
    #[test]
    #[ignore = "needs an exported draft head (FALCON_OCR_DRAFT_HEAD) and artifacts/model"]
    fn matches_the_pytorch_chain() {
        let path = std::path::PathBuf::from(std::env::var("FALCON_OCR_DRAFT_HEAD").expect("FALCON_OCR_DRAFT_HEAD"));
        let head = DraftHead::load(&path).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.with_extension("fixture.json")).unwrap()).unwrap();
        let features: Vec<f32> = fixture["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let tokens: Vec<u32> = fixture["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let bytes = std::fs::read("artifacts/model/model.safetensors").unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let embedding = tensor_f32(&tensors, "tok_embeddings.weight", &[65536, DIM]).unwrap();
        let embed = |t: u32| &embedding[t as usize * DIM..(t as usize + 1) * DIM];
        let simd = Simd::Auto;
        // PyTorch's token and top-two logit margin per step.
        let expected: Vec<(u32, f64)> = fixture["chain"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let top = c["top_logits"].as_array().unwrap();
                let margin = top[0].as_f64().unwrap() - top[1].as_f64().unwrap();
                (c["token"].as_u64().unwrap() as u32, margin)
            })
            .collect();
        for kv in [DraftKv::F32, DraftKv::Q8] {
            let mut st = DraftState::new(options(kv, 0));
            // Two batches (4 + 2 rows) exercise the multi-row feed.
            let row = |i: usize| &features[i * 3 * DIM..(i + 1) * 3 * DIM];
            head.push_rows(&mut st, &tokens[..4], row, embed, simd).unwrap();
            head.push_rows(&mut st, &tokens[4..], |r| row(4 + r), embed, simd)
                .unwrap();
            let mut out = Vec::new();
            head.draft(&mut st, 4, embed, &mut out, simd).unwrap();
            assert_eq!(out.len(), expected.len(), "chain length ({kv:?} keys and values)");
            // Equal tokens up to the first near tie (int8 weights move logits
            // by about 0.1); the chains may part after one.
            for (j, (&drafted, &(token, margin))) in out.iter().zip(&expected).enumerate() {
                if drafted != token {
                    assert!(
                        margin < 0.25,
                        "step {j} ({kv:?} keys and values): drafted {drafted}, PyTorch {token} with margin {margin}"
                    );
                    break;
                }
            }
        }
    }
}
