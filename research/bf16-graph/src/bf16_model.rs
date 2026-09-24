//! Experimental single-request BF16 model graph. All cast boundaries follow
//! the frozen local contract; acceptance still requires independent GPU checks.

use crate::{
    bf16_attention::{self, Parameters},
    bf16_kernels::{self, Backend},
    bf16_ops::{self, NormWorkspace},
    config::{CONFIG_SHA256, ModelConfig, WEIGHTS_SHA256},
    model::{rotary_factors, temporal_factors},
    trace::Trace,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs::File, path::Path};

struct Layer {
    qkv: Vec<bf16>,
    wo: Vec<bf16>,
    w13: Vec<bf16>,
    w2: Vec<bf16>,
    sinks: Vec<bf16>,
}

/// BF16-owned weights converted once from the verified pinned FP32 checkpoint.
pub struct Bf16Model {
    pub(crate) config: ModelConfig,
    embedding: Vec<bf16>,
    projector: Vec<bf16>,
    norm: Vec<bf16>,
    output: Vec<bf16>,
    _golden: Vec<bf16>,
    golden_f32: Vec<f32>,
    temporal: Vec<[f32; 2]>,
    layers: Vec<Layer>,
    weight_bytes: usize,
}
impl Bf16Model {
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn weights_sha256(&self) -> &str {
        WEIGHTS_SHA256
    }
    pub fn weight_tensor_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let bytes = std::fs::read(dir.join("config.json"))?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == CONFIG_SHA256,
            "config.json SHA-256 mismatch"
        );
        let config: ModelConfig = serde_json::from_slice(&bytes)?;
        config.validate()?;
        let file = File::open(dir.join("model.safetensors"))?;
        // SAFETY: Read-only mapping is retained through conversion; callers
        // must keep the source checkpoint unchanged while this load runs.
        let map = unsafe { Mmap::map(&file)? };
        ensure!(
            format!("{:x}", Sha256::digest(&map)) == WEIGHTS_SHA256,
            "checkpoint SHA-256 mismatch"
        );
        let tensors = SafeTensors::deserialize(&map)?;
        let mut weights = HashMap::new();
        let mut weight_bytes = 0;
        for (name, tensor) in tensors.tensors() {
            ensure!(
                tensor.dtype() == Dtype::F32,
                "{name}: expected pinned FP32 weights"
            );
            let values: Vec<_> = tensor
                .data()
                .chunks_exact(4)
                .map(|v| bf16::from_f32(f32::from_le_bytes(v.try_into().unwrap())))
                .collect();
            weight_bytes += values.len() * 2;
            weights.insert(name, (tensor.shape().to_vec(), values));
        }
        let mut take = |name: &str, shape: &[usize]| -> Result<Vec<bf16>> {
            let (actual, values) = weights
                .remove(name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(actual == shape, "{name}: incorrect shape {actual:?}");
            Ok(values)
        };
        let c = &config;
        let embedding = take("tok_embeddings.weight", &[c.vocab_size, c.dim])?;
        let projector = take("img_projector.weight", &[c.dim, c.patch_dim()])?;
        let norm = take("norm.weight", &[c.dim])?;
        let output = take("output.weight", &[c.vocab_size, c.dim])?;
        let golden = take("freqs_cis_golden", &[c.n_heads, c.head_dim / 4, 2])?;
        let golden_f32 = golden.iter().map(|v| v.to_f32()).collect();
        let mut layers = Vec::new();
        for i in 0..c.n_layers {
            layers.push(Layer {
                qkv: take(
                    &format!("layers.{i}.attention.wqkv.weight"),
                    &[c.query_dim() + 2 * c.kv_dim(), c.dim],
                )?,
                wo: take(
                    &format!("layers.{i}.attention.wo.weight"),
                    &[c.dim, c.query_dim()],
                )?,
                w13: take(
                    &format!("layers.{i}.feed_forward.w13.weight"),
                    &[2 * c.ffn_dim, c.dim],
                )?,
                w2: take(
                    &format!("layers.{i}.feed_forward.w2.weight"),
                    &[c.dim, c.ffn_dim],
                )?,
                sinks: take(&format!("layers.{i}.attention.sinks"), &[c.n_heads])?,
            });
        }
        ensure!(weights.is_empty(), "unexpected checkpoint tensors");
        let temporal = temporal_factors(c);
        Ok(Self {
            config,
            embedding,
            projector,
            norm,
            output,
            _golden: golden,
            golden_f32,
            temporal,
            layers,
            weight_bytes,
        })
    }

    pub(crate) fn embed(
        &self,
        tokens: &[u32],
        patches: Option<&[f32]>,
        hidden: &mut Vec<bf16>,
        backend: Backend,
    ) -> Result<()> {
        let c = &self.config;
        ensure!(
            tokens.iter().all(|&t| (t as usize) < c.vocab_size),
            "token outside vocabulary"
        );
        hidden.resize(tokens.len() * c.dim, bf16::ZERO);
        for (&token, dst) in tokens.iter().zip(hidden.chunks_exact_mut(c.dim)) {
            dst.copy_from_slice(
                &self.embedding[token as usize * c.dim..(token as usize + 1) * c.dim],
            );
        }
        if let Some(patches) = patches {
            ensure!(patches.len().is_multiple_of(c.patch_dim()), "patch shape");
            let rows = patches.len() / c.patch_dim();
            ensure!(
                tokens.iter().filter(|&&t| t == c.img_id).count() == rows,
                "patch/token mismatch"
            );
            let mut converted = vec![bf16::ZERO; patches.len()];
            bf16_kernels::convert_f32_to_bf16(patches, &mut converted);
            let mut features = vec![bf16::ZERO; rows * c.dim];
            linear(
                &converted,
                rows,
                c.patch_dim(),
                &self.projector,
                c.dim,
                &mut features,
                &mut Vec::new(),
                backend,
            );
            let mut patch = 0;
            for (&token, dst) in tokens.iter().zip(hidden.chunks_exact_mut(c.dim)) {
                if token == c.img_id {
                    dst.copy_from_slice(&features[patch * c.dim..(patch + 1) * c.dim]);
                    patch += 1;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn forward<'a>(
        &self,
        hidden: &mut [bf16],
        positions: &[usize],
        spatial: &[[f32; 2]],
        session: &'a mut Session,
        trace: &mut dyn Trace,
        phase: &str,
    ) -> Result<&'a [bf16]> {
        let c = &self.config;
        let rows = positions.len();
        let qdim = c.query_dim();
        ensure!(
            rows > 0 && hidden.len() == rows * c.dim && spatial.len() == rows,
            "BF16 forward shape"
        );
        ensure!(
            positions.iter().all(|&p| p < c.max_seq_len),
            "RoPE context exceeded"
        );
        ensure!(
            session.len + rows <= session.capacity,
            "BF16 KV context exceeded"
        );
        let offset = session.len;
        let backend = session.backend;
        let work = &mut session.work;
        work.resize(rows, c);
        rotary_factors(
            c,
            positions,
            spatial,
            &self.golden_f32,
            &self.temporal,
            &mut work.rope,
        );
        if trace.enabled() {
            capture(
                trace,
                &format!("{phase}.embedding"),
                &[rows, c.dim],
                hidden,
                &mut work.trace,
            )?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            bf16_ops::rms_norm(
                hidden,
                &mut work.normalized,
                c.dim,
                f32::EPSILON,
                None,
                &mut work.norm,
            );
            let qkv_width = qdim + 2 * c.kv_dim();
            linear(
                &work.normalized,
                rows,
                c.dim,
                &layer.qkv,
                qkv_width,
                &mut work.qkv,
                &mut work.accum,
                backend,
            );
            for row in 0..rows {
                let src = &work.qkv[row * qkv_width..(row + 1) * qkv_width];
                for head in 0..c.n_heads {
                    let dst = (row * c.n_heads + head) * c.head_dim;
                    work.q[dst..dst + c.head_dim]
                        .copy_from_slice(&src[head * c.head_dim..(head + 1) * c.head_dim]);
                    let group = head / (c.n_heads / c.n_kv_heads);
                    let k = qdim + group * c.head_dim;
                    let v = k + c.kv_dim();
                    work.k[dst..dst + c.head_dim].copy_from_slice(&src[k..k + c.head_dim]);
                    work.v[dst..dst + c.head_dim].copy_from_slice(&src[v..v + c.head_dim]);
                }
            }
            bf16_ops::rms_norm(
                &work.q,
                &mut work.attn,
                c.head_dim,
                f32::EPSILON,
                None,
                &mut work.norm,
            );
            std::mem::swap(&mut work.q, &mut work.attn);
            bf16_ops::rms_norm(
                &work.k,
                &mut work.attn,
                c.head_dim,
                f32::EPSILON,
                None,
                &mut work.norm,
            );
            std::mem::swap(&mut work.k, &mut work.attn);
            for vector in [&mut work.q, &mut work.k] {
                for (pair, &[cos, sin]) in vector.chunks_exact_mut(2).zip(&work.rope) {
                    let a = pair[0].to_f32();
                    let b = pair[1].to_f32();
                    pair[0] = bf16::from_f32(a * cos - b * sin);
                    pair[1] = bf16::from_f32(a * sin + b * cos);
                }
            }
            if trace.enabled() {
                capture(
                    trace,
                    &format!("{phase}.layer.{i}.q"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.q,
                    &mut work.trace,
                )?;
                capture(
                    trace,
                    &format!("{phase}.layer.{i}.k"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.k,
                    &mut work.trace,
                )?;
                capture(
                    trace,
                    &format!("{phase}.layer.{i}.v"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.v,
                    &mut work.trace,
                )?;
            }
            let cache = &mut session.layers[i];
            cache.k[offset * qdim..(offset + rows) * qdim].copy_from_slice(&work.k);
            cache.v[offset * qdim..(offset + rows) * qdim].copy_from_slice(&work.v);
            bf16_attention::attention(
                &work.q,
                &cache.k[..(offset + rows) * qdim],
                &cache.v[..(offset + rows) * qdim],
                &layer.sinks,
                Parameters {
                    query_len: rows,
                    kv_len: offset + rows,
                    heads: c.n_heads,
                    query_offset: offset,
                    image_start: session.image_start,
                    image_end: session.image_end,
                    capacity: session.capacity.next_multiple_of(128),
                },
                &mut work.attn,
                &mut work.raw,
                &mut work.lse,
                backend,
            );
            if trace.enabled() {
                capture(
                    trace,
                    &format!("{phase}.layer.{i}.attention"),
                    &[rows, qdim],
                    &work.attn,
                    &mut work.trace,
                )?;
                capture(
                    trace,
                    &format!("{phase}.layer.{i}.attention.raw"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.raw,
                    &mut work.trace,
                )?;
                trace.tensor(
                    &format!("{phase}.layer.{i}.attention.lse"),
                    &[rows, c.n_heads],
                    &work.lse,
                )?;
            }
            linear(
                &work.attn,
                rows,
                qdim,
                &layer.wo,
                c.dim,
                &mut work.projected,
                &mut work.accum,
                backend,
            );
            bf16_ops::residual_add(hidden, &work.projected);
            bf16_ops::rms_norm(
                hidden,
                &mut work.normalized,
                c.dim,
                f32::EPSILON,
                None,
                &mut work.norm,
            );
            linear(
                &work.normalized,
                rows,
                c.dim,
                &layer.w13,
                2 * c.ffn_dim,
                &mut work.ffn,
                &mut work.accum,
                backend,
            );
            bf16_ops::squared_relu_gate(&work.ffn, &mut work.gated);
            linear(
                &work.gated,
                rows,
                c.ffn_dim,
                &layer.w2,
                c.dim,
                &mut work.projected,
                &mut work.accum,
                backend,
            );
            bf16_ops::residual_add(hidden, &work.projected);
            if trace.enabled() {
                capture(
                    trace,
                    &format!("{phase}.layer.{i}.hidden"),
                    &[rows, c.dim],
                    hidden,
                    &mut work.trace,
                )?;
            }
        }
        session.len += rows;
        session.next_position = positions[rows - 1] + 1;
        let last = &hidden[(rows - 1) * c.dim..];
        bf16_ops::rms_norm(
            last,
            &mut work.normalized[..c.dim],
            c.dim,
            c.norm_eps,
            Some(&self.norm),
            &mut work.norm,
        );
        work.logits.resize(c.vocab_size, bf16::ZERO);
        linear(
            &work.normalized[..c.dim],
            1,
            c.dim,
            &self.output,
            c.vocab_size,
            &mut work.logits,
            &mut work.accum,
            backend,
        );
        if trace.enabled() {
            capture(
                trace,
                &format!("{phase}.logits"),
                &[c.vocab_size],
                &work.logits,
                &mut work.trace,
            )?;
        }
        Ok(&work.logits)
    }
}

fn capture(
    trace: &mut dyn Trace,
    name: &str,
    shape: &[usize],
    values: &[bf16],
    scratch: &mut Vec<f32>,
) -> Result<()> {
    scratch.resize(values.len(), 0.);
    for (dst, src) in scratch.iter_mut().zip(values) {
        *dst = src.to_f32();
    }
    trace.tensor(name, shape, scratch)
}

#[allow(clippy::too_many_arguments)]
fn linear(
    input: &[bf16],
    rows: usize,
    width: usize,
    weights: &[bf16],
    output_dim: usize,
    output: &mut [bf16],
    scratch: &mut Vec<f32>,
    backend: Backend,
) {
    scratch.resize(rows * output_dim, 0.);
    bf16_kernels::linear(input, rows, width, weights, output_dim, scratch, backend);
    output
        .par_iter_mut()
        .zip(scratch.par_iter())
        .for_each(|(dst, &src)| *dst = bf16::from_f32(src));
}

struct Cache {
    k: Vec<bf16>,
    v: Vec<bf16>,
}
pub(crate) struct Session {
    layers: Vec<Cache>,
    pub len: usize,
    pub next_position: usize,
    capacity: usize,
    image_start: usize,
    image_end: usize,
    backend: Backend,
    work: Workspace,
}
impl Session {
    pub fn new(
        c: &ModelConfig,
        capacity: usize,
        image_start: usize,
        image_end: usize,
        backend: Backend,
    ) -> Self {
        assert!(capacity <= c.max_seq_len);
        let size = capacity * c.query_dim();
        Self {
            layers: (0..c.n_layers)
                .map(|_| Cache {
                    k: vec![bf16::ZERO; size],
                    v: vec![bf16::ZERO; size],
                })
                .collect(),
            len: 0,
            next_position: 0,
            capacity,
            image_start,
            image_end,
            backend,
            work: Workspace::default(),
        }
    }
}
#[derive(Default)]
struct Workspace {
    normalized: Vec<bf16>,
    projected: Vec<bf16>,
    qkv: Vec<bf16>,
    q: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    attn: Vec<bf16>,
    raw: Vec<bf16>,
    ffn: Vec<bf16>,
    gated: Vec<bf16>,
    logits: Vec<bf16>,
    accum: Vec<f32>,
    lse: Vec<f32>,
    trace: Vec<f32>,
    rope: Vec<[f32; 2]>,
    norm: NormWorkspace,
}
impl Workspace {
    fn resize(&mut self, rows: usize, c: &ModelConfig) {
        self.normalized.resize(rows * c.dim, bf16::ZERO);
        self.projected.resize(rows * c.dim, bf16::ZERO);
        self.qkv
            .resize(rows * (c.query_dim() + 2 * c.kv_dim()), bf16::ZERO);
        for values in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.attn,
            &mut self.raw,
        ] {
            values.resize(rows * c.query_dim(), bf16::ZERO);
        }
        self.ffn.resize(rows * 2 * c.ffn_dim, bf16::ZERO);
        self.gated.resize(rows * c.ffn_dim, bf16::ZERO);
        self.lse.resize(rows * c.n_heads, 0.);
    }
}
