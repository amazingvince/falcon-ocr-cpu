//! Per-request KV caches and the forward-pass scratch buffers.
use anyhow::{Context, Result, bail, ensure};

use crate::{
    config::{CacheLayout, ExpMode, ModelConfig, Tuning},
    kernels::{self, Bf16Kv},
};

use super::Model;

/// Allocated once before decode; shrinking active rows never grows these buffers.
pub(crate) struct BatchWorkspace {
    pub(super) hidden: Vec<f32>,
    pub(super) positions: Vec<usize>,
    pub(super) spatial: Vec<[f32; 2]>,
    pub(super) work: Workspace,
}
impl BatchWorkspace {
    pub(crate) fn reserve_screened_head(&mut self, model: &Model, rows: usize) {
        self.work.reserve_screened_head(model, rows);
    }
    pub fn new(rows: usize, c: &ModelConfig) -> Self {
        let mut work = Workspace::default();
        work.resize(rows, c);
        work.rope.resize(rows * c.query_dim() / 2, [1., 0.]);

        work.logits.resize(rows * c.vocab_size, 0.);
        Self {
            hidden: vec![0.; rows * c.dim],
            positions: vec![0; rows],
            spatial: vec![[f32::NAN; 2]; rows],
            work,
        }
    }
}

pub(crate) struct Session {
    pub(super) layers: Vec<LayerCache>,
    pub(super) workspace: Workspace,
    pub len: usize,
    pub next_position: usize,
    pub(super) capacity: usize,
    pub(super) image_start: usize,
    pub(super) image_end: usize,
    pub(super) simd: kernels::Simd,
    /// Prefill exp of this request (`RunnerConfig::exp`).
    pub(super) exp: ExpMode,
    /// Experiment knobs of this request (`RunnerConfig::tuning`).
    pub(super) tuning: Tuning,
}
pub(super) enum LayerCache {
    Split(crate::quant::kv::SplitPrefix),
    Expanded {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    Compact {
        prefix_k: Vec<f32>,
        generated_k: Vec<f32>,
        v: Vec<f32>,
        prefix_len: usize,
    },
}
impl LayerCache {
    /// Prompt length of a compact cache (`None` for other layouts).
    pub(super) fn compact_prefix_len(&self) -> Option<usize> {
        match self {
            Self::Compact { prefix_len, .. } => Some(*prefix_len),
            _ => None,
        }
    }
    pub(super) fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) {
        match self {
            Self::Split(cache) => cache.append(k, v),
            Self::Expanded { k: keys, v: values } => {
                keys.extend_from_slice(k);
                values.extend_from_slice(v);
            }
            Self::Compact {
                prefix_k,
                generated_k,
                v: values,
                prefix_len,
            } => {
                let prefix_rows = prefix_len.saturating_sub(offset).min(k.len() / c.query_dim());
                let prefix_elements = prefix_rows * c.query_dim();
                prefix_k.extend_from_slice(&k[..prefix_elements]);
                append_unique_heads(generated_k, &k[prefix_elements..], c);
                append_unique_heads(values, v, c);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        q: &[f32],
        rows: usize,
        total_len: usize,
        c: &ModelConfig,
        offset: usize,
        image_start: usize,
        image_end: usize,
        sinks: &[f32],
        output: &mut [f32],
        simd: kernels::Simd,
        bf16: Option<Bf16Kv<'_>>,
        exp: ExpMode,
        profile: bool,
    ) {
        match self {
            Self::Split(cache) => {
                assert!(offset >= image_end, "cannot treat a partial image as causal decode");
                if rows == 1 {
                    cache.attention_decode(q, total_len, sinks, output, simd);
                } else {
                    // Draft verification: rows are the newest text positions.
                    cache.attention_decode_rows(q, rows, sinks, output, simd);
                }
            }
            Self::Expanded { k, v } => kernels::attention_with_simd(
                q,
                k,
                v,
                rows,
                total_len,
                c.n_heads,
                c.head_dim,
                offset,
                image_start,
                image_end,
                sinks,
                output,
                simd,
            ),
            Self::Compact {
                prefix_k,
                generated_k,
                v,
                ..
            } => {
                if let Some(kv) = bf16
                    && generated_k.is_empty()
                    && kernels::attention_compact_prefill_bf16(
                        q,
                        prefix_k,
                        v,
                        rows,
                        total_len,
                        c.n_heads,
                        c.n_kv_heads,
                        c.head_dim,
                        offset,
                        image_start,
                        image_end,
                        sinks,
                        output,
                        kv,
                        exp,
                        profile,
                    )
                {
                    return;
                }
                kernels::attention_compact_prefill_with(
                    q,
                    prefix_k,
                    generated_k,
                    v,
                    rows,
                    prefix_k.len() / c.query_dim(),
                    total_len,
                    c.n_heads,
                    c.n_kv_heads,
                    c.head_dim,
                    offset,
                    image_start,
                    image_end,
                    sinks,
                    output,
                    simd,
                    exp,
                    profile,
                )
            }
        }
    }
}

/// Store one original GQA head from each duplicated group. This runs after the
/// same normalization and rotation as expanded storage, preserving tensor bits.
fn append_unique_heads(output: &mut Vec<f32>, expanded: &[f32], c: &ModelConfig) {
    let group_width = (c.n_heads / c.n_kv_heads) * c.head_dim;
    for group in expanded.chunks_exact(group_width) {
        debug_assert!(
            group.chunks_exact(c.head_dim).all(|head| {
                head.iter()
                    .zip(&group[..c.head_dim])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            }),
            "compact cache requires identical duplicated heads"
        );
        output.extend_from_slice(&group[..c.head_dim]);
    }
}
impl Session {
    /// Drop the newest positions so that `len` remain (rejected draft
    /// tokens). Needs a split cache.
    pub(crate) fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(
            len >= self.image_end && len <= self.len,
            "truncation must keep the prompt"
        );
        let dropped = self.len - len;
        for layer in &mut self.layers {
            match layer {
                LayerCache::Split(cache) => {
                    ensure!(cache.tail_len() >= dropped, "truncation past the prefix");
                    cache.truncate_tail(cache.tail_len() - dropped);
                }
                _ => anyhow::bail!("speculative decoding needs the split cache"),
            }
        }
        self.len = len;
        self.next_position -= dropped;
        Ok(())
    }
    /// Whether every layer uses the split cache (required by `truncate`).
    pub(crate) fn is_split(&self) -> bool {
        self.layers.iter().all(|l| matches!(l, LayerCache::Split(_)))
    }
    pub(crate) fn remaining_capacity(&self) -> usize {
        self.capacity - self.len
    }
    pub(crate) fn cache_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|c| match c {
                LayerCache::Expanded { k, v } => 4 * (k.capacity() + v.capacity()),
                LayerCache::Compact {
                    prefix_k,
                    generated_k,
                    v,
                    ..
                } => 4 * (prefix_k.capacity() + generated_k.capacity() + v.capacity()),
                LayerCache::Split(c) => c.bytes(),
            })
            .sum()
    }
    pub(crate) fn retire_cache(&mut self) -> usize {
        let bytes = self.cache_bytes();
        self.layers = Vec::new();
        self.capacity = 0;
        bytes
    }
    pub(crate) fn seal_prefix(&mut self, c: &ModelConfig, mode: crate::quant::Kv) -> Result<()> {
        if mode == crate::quant::Kv::Compact {
            return Ok(());
        }
        for layer in &mut self.layers {
            if let LayerCache::Compact {
                prefix_k,
                v,
                prefix_len,
                ..
            } = layer
            {
                ensure!(
                    self.len == *prefix_len,
                    "seal only a complete prefix, before any text decode"
                );
                let packed = crate::quant::kv::SplitPrefix::from_compact(
                    prefix_k,
                    v,
                    *prefix_len,
                    self.capacity,
                    c,
                    mode,
                    self.tuning.split_chunks,
                )?;
                *layer = LayerCache::Split(packed);
            } else {
                bail!("prefix sealing requires an unsealed compact cache");
            }
        }
        Ok(())
    }
    pub(crate) fn prepare_small_decode(&mut self, c: &ModelConfig) {
        let head = std::mem::take(&mut self.workspace.head);
        let selected = std::mem::take(&mut self.workspace.selected);
        self.workspace = Workspace::default();
        self.workspace.head = head;
        self.workspace.selected = selected;
        self.workspace.resize(1, c);
        self.workspace.rope.resize(c.query_dim() / 2, [1.0, 0.0]);
        self.workspace.logits.resize(c.vocab_size, 0.0);
    }
    /// Reserve screened-head scratch so warm decode steps never allocate.
    pub(crate) fn reserve_screened_head(&mut self, model: &Model) {
        self.workspace.reserve_screened_head(model, 1);
    }
    /// Sequential batch prefills share one scratch allocation. The preceding
    /// request has already consumed its logits; only its KV cache is retained.
    pub(crate) fn reuse_workspace_from(&mut self, previous: &mut Self) {
        std::mem::swap(&mut self.workspace, &mut previous.workspace);
    }

    pub(crate) fn release_workspace(&mut self) {
        self.workspace = Workspace::default();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: &ModelConfig,
        capacity: usize,
        prefix_len: usize,
        image_start: usize,
        image_end: usize,
        simd: kernels::Simd,
        cache_layout: CacheLayout,
        exp: ExpMode,
        tuning: Tuning,
    ) -> Result<Self> {
        ensure!(capacity <= c.max_seq_len, "capacity exceeds model context");
        ensure!(
            prefix_len > 0 && prefix_len <= capacity && image_start < image_end && image_end < prefix_len,
            "invalid cache prefix or image interval"
        );
        let reserve = |rows: usize, width: usize| -> Result<Vec<f32>> {
            let elements = rows.checked_mul(width).context("cache size overflow")?;
            let mut values = Vec::new();
            values.try_reserve_exact(elements).context("allocate KV cache")?;
            Ok(values)
        };
        let mut layers = Vec::new();
        for _ in 0..c.n_layers {
            layers.push(match cache_layout {
                CacheLayout::Expanded => LayerCache::Expanded {
                    k: reserve(capacity, c.query_dim())?,
                    v: reserve(capacity, c.query_dim())?,
                },
                CacheLayout::Compact => LayerCache::Compact {
                    prefix_k: reserve(prefix_len, c.query_dim())?,
                    generated_k: reserve(capacity - prefix_len, c.kv_dim())?,
                    v: reserve(capacity, c.kv_dim())?,
                    prefix_len,
                },
            });
        }
        Ok(Self {
            layers,
            workspace: Workspace::default(),
            len: 0,
            next_position: 0,
            capacity,
            image_start,
            image_end,
            simd,
            exp,
            tuning,
        })
    }
}

#[derive(Default)]
pub(super) struct Workspace {
    pub(super) quant_scratch: crate::quant::linear::Scratch,
    pub(super) normalized: Vec<f32>,
    pub(super) qkv: Vec<f32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) attn: Vec<f32>,
    pub(super) projected: Vec<f32>,
    pub(super) ffn_packed: Vec<f32>,
    pub(super) gated: Vec<f32>,
    pub(super) logits: Vec<f32>,
    pub(super) rope: Vec<[f32; 2]>,
    /// Screened-head scratch and per-row selected tokens; reserved before decode.
    pub(super) head: crate::head_screen::HeadScratch,
    pub(super) selected: Vec<u32>,
    /// Per-row screened-head scratch for draft verification.
    pub(super) head_rows: Vec<crate::head_screen::HeadScratch>,
    /// Per-row RMS-norm factors folded into prefill GEMMs.
    pub(super) row_scale: Vec<f32>,
    /// One layer's prefill keys and value pairs in the BF16 attention
    /// layouts, written by the fused QKV pass or converted by the kernel.
    pub(super) bf16_keys: Vec<u32>,
    pub(super) bf16_values: Vec<u32>,
}
impl Workspace {
    fn reserve_screened_head(&mut self, model: &Model, rows: usize) {
        if let Some(head) = model.screened.get() {
            self.head.reserve(head);
            if self.selected.len() < rows {
                self.selected.resize(rows, 0);
            }
        }
    }
    pub(super) fn resize(&mut self, rows: usize, c: &ModelConfig) {
        self.normalized.resize(rows * c.dim, 0.);
        self.qkv.resize(rows * (c.query_dim() + 2 * c.kv_dim()), 0.);
        self.q.resize(rows * c.query_dim(), 0.);
        self.k.resize(rows * c.query_dim(), 0.);
        self.v.resize(rows * c.query_dim(), 0.);
        self.attn.resize(rows * c.query_dim(), 0.);
        self.projected.resize(rows * c.dim, 0.);
        self.ffn_packed.resize(rows * 2 * c.ffn_dim, 0.);
        self.gated.resize(rows * c.ffn_dim, 0.);
    }
}
