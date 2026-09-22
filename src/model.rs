use std::{
    collections::HashMap,
    fs::File,
    ops::Range,
    path::Path,
    sync::{Arc, OnceLock},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::{
    config::{CONFIG_SHA256, CacheLayout, ModelConfig, WEIGHTS_SHA256, WeightLayout},
    kernels,
    packed_kernels::PhasePackedLinear,
    trace::Trace,
};

#[derive(Clone)]
struct Weight {
    range: Range<usize>,
    quantized: Option<Arc<crate::attempt::quant::Q8Linear>>,
}
struct Layer {
    qkv: Weight,
    wo: Weight,
    w13: Weight,
    w2: Weight,
    sinks: Weight,
}
struct PackedLayer {
    qkv: PhasePackedLinear,
    wo: PhasePackedLinear,
    w13: PhasePackedLinear,
    w2: PhasePackedLinear,
}
struct PackedWeights {
    layers: Vec<PackedLayer>,
    output: PhasePackedLinear,
    tensor_bytes: usize,
    packing_ms: f64,
}

/// An immutable checkpoint. Keep its backing file unchanged while the model is loaded.
pub struct Model {
    pub(crate) config: ModelConfig,
    map: Mmap,
    embedding: Weight,
    projector: Weight,
    norm: Weight,
    output: Weight,
    golden: Weight,
    temporal: Vec<[f32; 2]>,
    layers: Vec<Layer>,
    packed: OnceLock<PackedWeights>,
    attempt_profile: crate::attempt::Profile,
    attempt_setup_ms: f64,
    attempt_artifact_sha256: Option<String>,
    pub(crate) weights_sha256: String,
}

impl Model {
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }
    /// Actual additional shared tensor payload, excluding allocator metadata.
    /// This remains resident if another Runner on this Model enabled packing.
    pub fn packed_weight_bytes(&self) -> usize {
        self.packed.get().map_or(0, |p| p.tensor_bytes)
    }
    /// Time spent constructing the one shared packed copy, or zero if absent.
    pub fn weight_packing_ms(&self) -> f64 {
        self.packed.get().map_or(0.0, |p| p.packing_ms)
    }
    pub(crate) fn prepare_phase_packed(&self) {
        self.packed.get_or_init(|| {
            let started = Instant::now();
            let c = &self.config;
            let layers: Vec<_> = self
                .layers
                .iter()
                .map(|layer| PackedLayer {
                    qkv: PhasePackedLinear::new(
                        self.w(&layer.qkv),
                        c.dim,
                        c.query_dim() + 2 * c.kv_dim(),
                    ),
                    wo: PhasePackedLinear::new(self.w(&layer.wo), c.query_dim(), c.dim),
                    w13: PhasePackedLinear::new(self.w(&layer.w13), c.dim, 2 * c.ffn_dim),
                    w2: PhasePackedLinear::new(self.w(&layer.w2), c.ffn_dim, c.dim),
                })
                .collect();
            let output = PhasePackedLinear::new(self.w(&self.output), c.dim, c.vocab_size);
            let tensor_bytes = output.packed_weight_bytes()
                + layers
                    .iter()
                    .map(|layer| {
                        layer.qkv.packed_weight_bytes()
                            + layer.wo.packed_weight_bytes()
                            + layer.w13.packed_weight_bytes()
                            + layer.w2.packed_weight_bytes()
                    })
                    .sum::<usize>();
            PackedWeights {
                layers,
                output,
                tensor_bytes,
                packing_ms: started.elapsed().as_secs_f64() * 1000.0,
            }
        });
    }

    /// Loads and verifies the pinned v1.5 checkpoint before exposing tensor views.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config_bytes = std::fs::read(dir.join("config.json"))?;
        ensure!(
            format!("{:x}", Sha256::digest(&config_bytes)) == CONFIG_SHA256,
            "config.json SHA-256 mismatch"
        );
        let config: ModelConfig = serde_json::from_slice(&config_bytes)?;
        config.validate()?;
        let file = File::open(dir.join("model.safetensors"))
            .context("open model.safetensors; run scripts/fetch_reference.py first")?;
        // SAFETY: read-only mapping lives with Model; API documents that backing files
        // must remain unchanged. Every typed tensor view is checked for alignment/size.
        let map = unsafe { Mmap::map(&file)? };
        let hash = format!("{:x}", Sha256::digest(&map));
        ensure!(
            hash == WEIGHTS_SHA256,
            "checkpoint SHA-256 mismatch: expected {WEIGHTS_SHA256}, found {hash}"
        );
        let tensors = SafeTensors::deserialize(&map)?;
        let mut weights = HashMap::new();
        for (name, tensor) in tensors.tensors() {
            ensure!(
                tensor.dtype() == Dtype::F32,
                "{name}: expected F32, found {:?}",
                tensor.dtype()
            );
            let _: &[f32] = bytemuck::try_cast_slice(tensor.data())
                .map_err(|e| anyhow::anyhow!("unaligned tensor {name}: {e}"))?;
            let start = tensor.data().as_ptr() as usize - map.as_ptr() as usize;
            weights.insert(
                name,
                (
                    tensor.shape().to_vec(),
                    Weight {
                        range: start..start + tensor.data().len(),
                        quantized: None,
                    },
                ),
            );
        }
        let mut take = |name: &str, shape: &[usize]| -> Result<Weight> {
            let (actual, w) = weights
                .remove(name)
                .with_context(|| format!("missing tensor {name}"))?;
            ensure!(
                actual == shape,
                "{name}: shape {actual:?}, expected {shape:?}"
            );
            Ok(w)
        };
        let c = &config;
        let embedding = take("tok_embeddings.weight", &[c.vocab_size, c.dim])?;
        let projector = take("img_projector.weight", &[c.dim, c.patch_dim()])?;
        let norm = take("norm.weight", &[c.dim])?;
        let output = take("output.weight", &[c.vocab_size, c.dim])?;
        let golden = take("freqs_cis_golden", &[c.n_heads, c.head_dim / 4, 2])?;
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
        ensure!(
            weights.is_empty(),
            "unexpected checkpoint tensors: {:?}",
            weights.keys().collect::<Vec<_>>()
        );
        let temporal = temporal_factors(&config);
        Ok(Self {
            config,
            map,
            embedding,
            projector,
            norm,
            output,
            golden,
            temporal,
            layers,
            packed: OnceLock::new(),
            attempt_profile: crate::attempt::Profile::Reference,
            attempt_setup_ms: 0.0,
            attempt_artifact_sha256: None,
            weights_sha256: hash,
        })
    }

    /// Load a separately labelled experiment without changing the reference loader.
    /// Source weights remain mmap-backed for protected tensors and the oracle.
    /// The mapping length is NOT equivalent to additional committed/resident RAM.
    pub fn load_attempt(
        dir: impl AsRef<Path>,
        profile: crate::attempt::Profile,
        artifact: Option<&Path>,
    ) -> Result<Self> {
        let mut model = Self::load(dir)?;
        model.attempt_profile = profile;
        ensure!(
            artifact.is_none() || profile.quantizes_body(),
            "W8 artifact supplied to an FP32 profile"
        );
        if !profile.quantizes_body() {
            return Ok(model);
        }
        let started = Instant::now();
        let bytes = artifact.map(std::fs::read).transpose()?;
        let tensors = bytes
            .as_ref()
            .map(|b| SafeTensors::deserialize(b))
            .transpose()?;
        if let Some(bytes) = &bytes {
            ensure!(bytes.len() >= 8, "short W8 artifact");
            let len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
            let end = 8usize.checked_add(len).context("W8 header overflow")?;
            ensure!(end <= bytes.len(), "W8 header out of bounds");
            let header: serde_json::Value = serde_json::from_slice(&bytes[8..end])?;
            let meta = header.get("__metadata__").context("W8 metadata missing")?;
            let check = |key: &str, expected: &str| -> Result<()> {
                ensure!(
                    meta.get(key).and_then(|v| v.as_str()) == Some(expected),
                    "W8 metadata {key} mismatch"
                );
                Ok(())
            };
            check("format", "falcon-ocr-attempt3-w8g64-v1")?;
            check("source_sha256", WEIGHTS_SHA256)?;
            check("model_revision", crate::config::MODEL_REVISION)?;
            check(
                "include_head",
                if profile.quantizes_head() {
                    "true"
                } else {
                    "false"
                },
            )?;
            check("group_size", "64")?;
            check("scale_dtype", "f32")?;
            check("rounding", "ties_to_even")?;
            check("activation_dtype", "f32")?;
            model.attempt_artifact_sha256 = Some(format!("{:x}", Sha256::digest(bytes)));
            ensure!(
                tensors.as_ref().unwrap().len()
                    == 2 * (4 * model.config.n_layers + (profile.quantizes_head() as usize)),
                "W8 artifact has unexpected tensors"
            );
        }
        // Build separately before mutating Weight handles (and their source views).
        let mut prepared = Vec::new();
        let make = |name: &str,
                    w: &Weight,
                    input: usize,
                    output: usize|
         -> Result<Arc<crate::attempt::quant::Q8Linear>> {
            use crate::attempt::quant::Q8Linear;
            let q = if let Some(tensors) = &tensors {
                let codes = tensors.tensor(&format!("{name}.__w8_codes"))?;
                let scales = tensors.tensor(&format!("{name}.__w8_scales"))?;
                ensure!(
                    codes.dtype() == Dtype::I8 && codes.shape() == [output, input],
                    "W8 codes dtype/shape for {name}"
                );
                ensure!(
                    scales.dtype() == Dtype::F32 && scales.shape() == [output, input.div_ceil(64)],
                    "W8 scales dtype/shape for {name}"
                );
                let codes = codes.data().iter().map(|&x| x as i8).collect();
                let scales = scales
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                Q8Linear::from_parts(output, input, 64, codes, scales)?
            } else {
                Q8Linear::quantize(model.w(w), output, input, 64).map_err(anyhow::Error::msg)?
            };
            Ok(Arc::new(q))
        };
        let c = &model.config;
        for (i, l) in model.layers.iter().enumerate() {
            prepared.push(make(
                &format!("layers.{i}.attention.wqkv.weight"),
                &l.qkv,
                c.dim,
                c.query_dim() + 2 * c.kv_dim(),
            )?);
            prepared.push(make(
                &format!("layers.{i}.attention.wo.weight"),
                &l.wo,
                c.query_dim(),
                c.dim,
            )?);
            prepared.push(make(
                &format!("layers.{i}.feed_forward.w13.weight"),
                &l.w13,
                c.dim,
                2 * c.ffn_dim,
            )?);
            prepared.push(make(
                &format!("layers.{i}.feed_forward.w2.weight"),
                &l.w2,
                c.ffn_dim,
                c.dim,
            )?);
        }
        if profile.quantizes_head() {
            prepared.push(make("output.weight", &model.output, c.dim, c.vocab_size)?);
        }
        let mut prepared = prepared.into_iter();
        for l in &mut model.layers {
            l.qkv.quantized = prepared.next();
            l.wo.quantized = prepared.next();
            l.w13.quantized = prepared.next();
            l.w2.quantized = prepared.next();
        }
        if profile.quantizes_head() {
            model.output.quantized = prepared.next();
        }
        model.attempt_setup_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(model)
    }
    pub fn attempt_profile(&self) -> crate::attempt::Profile {
        self.attempt_profile
    }
    pub fn attempt_memory_report(&self) -> serde_json::Value {
        let qbytes = |w: &Weight| w.quantized.as_ref().map_or(0, |q| q.payload_bytes());
        let effective = |w: &Weight| {
            w.quantized
                .as_ref()
                .map_or(w.range.len(), |q| q.payload_bytes())
        };
        let quantized = self
            .layers
            .iter()
            .map(|l| qbytes(&l.qkv) + qbytes(&l.wo) + qbytes(&l.w13) + qbytes(&l.w2))
            .sum::<usize>()
            + qbytes(&self.output);
        let scanned = self
            .layers
            .iter()
            .map(|l| effective(&l.qkv) + effective(&l.wo) + effective(&l.w13) + effective(&l.w2))
            .sum::<usize>()
            + effective(&self.output);
        serde_json::json!({"profile":self.attempt_profile,"source_mapping_bytes":self.map.len(),
            "quantized_weight_payload_bytes":quantized,"logical_decode_weight_scan_bytes":scanned,
            "quantization_or_import_ms":self.attempt_setup_ms,"w8_artifact_sha256":self.attempt_artifact_sha256,
            "source_mapping_is_not_rss":true,"weights_group_size":64,"weights_scale_dtype":"f32",
            "large_m_policy":"expand the SAME W8 values into one reusable per-operation F32 scratch matrix",
            "w8_small_m_policy":"AVX2/FMA if available and not explicit Scalar; otherwise scalar; floating activations, not integer dot",
            "kv_policy":"deferred compression AFTER full FP32-arithmetic prefill using the SELECTED numerical weights; generated tail F32"})
    }
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        w: &Weight,
        packed: Option<&PhasePackedLinear>,
        output_dim: usize,
        output: &mut [f32],
        scratch: &mut crate::attempt::quant::Scratch,
        simd: kernels::Simd,
    ) -> Result<()> {
        if let Some(q) = &w.quantized {
            ensure!(
                packed.is_none(),
                "phase-packed FP32 weights cannot override W8 numerical weights"
            );
            q.linear(input, rows, output, scratch, simd)
        } else {
            decode_linear(
                input,
                rows,
                input_dim,
                self.w(w),
                packed,
                output_dim,
                output,
                simd,
            );
            Ok(())
        }
    }

    fn w(&self, weight: &Weight) -> &[f32] {
        bytemuck::cast_slice(&self.map[weight.range.clone()])
    }

    pub(crate) fn embed(
        &self,
        tokens: &[u32],
        image_patches: Option<&[f32]>,
        simd: kernels::Simd,
        h: &mut Vec<f32>,
    ) -> Result<f64> {
        let c = &self.config;
        let patches = image_patches.unwrap_or(&[]);
        ensure!(
            patches.len().is_multiple_of(c.patch_dim()),
            "invalid patch tensor length"
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < c.vocab_size),
            "token outside vocabulary"
        );
        let patch_count = patches.len() / c.patch_dim();
        if image_patches.is_some() {
            ensure!(
                tokens.iter().filter(|&&t| t == c.img_id).count() == patch_count,
                "image tokens and patches disagree"
            );
        }
        let mut features = vec![0.; patch_count * c.dim];
        let mut image_projection_ms = 0.0;
        if patch_count > 0 {
            let projection_started = Instant::now();
            kernels::linear_with_simd(
                patches,
                patch_count,
                c.patch_dim(),
                self.w(&self.projector),
                c.dim,
                &mut features,
                simd,
            );
            image_projection_ms = projection_started.elapsed().as_secs_f64() * 1000.0;
        }
        let embedding = self.w(&self.embedding);
        h.resize(tokens.len() * c.dim, 0.);
        let mut image_index = 0;
        for (token, row) in tokens.iter().zip(h.chunks_exact_mut(c.dim)) {
            if *token == c.img_id && image_patches.is_some() {
                row.copy_from_slice(&features[image_index * c.dim..(image_index + 1) * c.dim]);
                image_index += 1;
            } else {
                row.copy_from_slice(
                    &embedding[*token as usize * c.dim..(*token as usize + 1) * c.dim],
                );
            }
        }
        Ok(image_projection_ms)
    }

    pub(crate) fn forward<'s>(
        &self,
        h: &mut [f32],
        positions: &[usize],
        positions_hw: &[[f32; 2]],
        session: &'s mut Session,
        trace: &mut dyn Trace,
        phase: &str,
    ) -> Result<&'s [f32]> {
        let c = &self.config;
        let rows = positions.len();
        ensure!(
            rows > 0 && h.len() == rows * c.dim && positions_hw.len() == rows,
            "invalid forward dimensions"
        );
        ensure!(
            session
                .len
                .checked_add(rows)
                .is_some_and(|n| n <= session.capacity),
            "KV context exhausted"
        );
        ensure!(
            positions.iter().all(|&p| p < c.max_seq_len),
            "RoPE position exceeds context"
        );
        let qdim = c.query_dim();
        let kdim = c.kv_dim();
        let qkv_width = qdim + 2 * kdim;
        let offset = session.len;
        let simd = session.simd;
        let work = &mut session.workspace;
        work.resize(rows, c);
        if trace.enabled() {
            trace.tensor(&format!("{phase}.embedding"), &[rows, c.dim], h)?;
        }
        // RoPE factors are shared by all transformer layers.
        rotary_factors(
            c,
            positions,
            positions_hw,
            self.w(&self.golden),
            &self.temporal,
            &mut work.rope,
        );
        for (i, layer) in self.layers.iter().enumerate() {
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            self.linear(
                &work.normalized,
                rows,
                c.dim,
                &layer.qkv,
                None,
                qkv_width,
                &mut work.qkv,
                &mut work.quant_scratch,
                simd,
            )?;
            // Normalize each original K head before GQA expansion. Spatial rotations
            // subsequently differ for paired heads, so expanded keys are intentional.
            for r in 0..rows {
                let qkv = &work.qkv[r * qkv_width..(r + 1) * qkv_width];
                for head in 0..c.n_heads {
                    let dst = (r * c.n_heads + head) * c.head_dim;
                    work.q[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[head * c.head_dim..(head + 1) * c.head_dim]);
                    let kvhead = head / (c.n_heads / c.n_kv_heads);
                    let kstart = qdim + kvhead * c.head_dim;
                    work.k[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[kstart..kstart + c.head_dim]);
                    let vstart = qdim + kdim + kvhead * c.head_dim;
                    work.v[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[vstart..vstart + c.head_dim]);
                }
            }
            // Normalize all heads in two calls; avoid creating a Rayon operation
            // for each individual 64-element head.
            kernels::rms_norm(&work.q, &mut work.attn, c.head_dim, f32::EPSILON, None);
            std::mem::swap(&mut work.q, &mut work.attn);
            kernels::rms_norm(&work.k, &mut work.attn, c.head_dim, f32::EPSILON, None);
            std::mem::swap(&mut work.k, &mut work.attn);
            for r in 0..rows {
                for head in 0..c.n_heads {
                    let dst = (r * c.n_heads + head) * c.head_dim;
                    for pair in 0..c.head_dim / 2 {
                        let [cos, sin] =
                            work.rope[(r * c.n_heads + head) * (c.head_dim / 2) + pair];
                        for vector in [&mut work.q, &mut work.k] {
                            let p = dst + 2 * pair;
                            let a = vector[p];
                            let b = vector[p + 1];
                            vector[p] = a * cos - b * sin;
                            vector[p + 1] = a * sin + b * cos;
                        }
                    }
                }
            }
            if trace.enabled() {
                trace.tensor(
                    &format!("{phase}.layer.{i}.q"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.q,
                )?;
                trace.tensor(
                    &format!("{phase}.layer.{i}.k"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.k,
                )?;
                trace.tensor(
                    &format!("{phase}.layer.{i}.v"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.v,
                )?;
            }
            let cache = &mut session.layers[i];
            cache.append(&work.k, &work.v, offset, c);
            cache.attention(
                &work.q,
                rows,
                offset + rows,
                c,
                offset,
                session.image_start,
                session.image_end,
                self.w(&layer.sinks),
                &mut work.attn,
                simd,
            );
            if trace.enabled() {
                trace.tensor(
                    &format!("{phase}.layer.{i}.attention"),
                    &[rows, qdim],
                    &work.attn,
                )?;
            }
            self.linear(
                &work.attn,
                rows,
                qdim,
                &layer.wo,
                None,
                c.dim,
                &mut work.projected,
                &mut work.quant_scratch,
                simd,
            )?;
            for (x, a) in h.iter_mut().zip(&work.projected) {
                *x += a;
            }
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            self.linear(
                &work.normalized,
                rows,
                c.dim,
                &layer.w13,
                None,
                2 * c.ffn_dim,
                &mut work.ffn_packed,
                &mut work.quant_scratch,
                simd,
            )?;
            kernels::squared_relu_gate(&work.ffn_packed, &mut work.gated);
            self.linear(
                &work.gated,
                rows,
                c.ffn_dim,
                &layer.w2,
                None,
                c.dim,
                &mut work.projected,
                &mut work.quant_scratch,
                simd,
            )?;
            for (x, a) in h.iter_mut().zip(&work.projected) {
                *x += a;
            }
            if trace.enabled() {
                trace.tensor(&format!("{phase}.layer.{i}.hidden"), &[rows, c.dim], h)?;
            }
        }
        session.len += rows;
        session.next_position = positions[rows - 1] + 1;
        // Generation needs only the final token's vocabulary projection.
        let last = &h[(rows - 1) * c.dim..];
        let norm = &mut work.normalized[..c.dim];
        kernels::rms_norm(last, norm, c.dim, c.norm_eps, Some(self.w(&self.norm)));
        work.logits.resize(c.vocab_size, 0.);
        self.linear(
            norm,
            1,
            c.dim,
            &self.output,
            None,
            c.vocab_size,
            &mut work.logits,
            &mut work.quant_scratch,
            simd,
        )?;
        if trace.enabled() {
            trace.tensor(&format!("{phase}.logits"), &[c.vocab_size], &work.logits)?;
        }
        Ok(&work.logits)
    }

    /// Advance one generated token per active request. Linear projections share
    /// one matrix operation, while attention and KV storage remain per request.
    pub(crate) fn decode_batch<'w>(
        &self,
        tokens: &[u32],
        active: &[usize],
        sessions: &mut [Session],
        batch: &'w mut BatchWorkspace,
        weight_layout: WeightLayout,
        trace: &mut dyn Trace,
        phase: &str,
    ) -> Result<&'w [f32]> {
        let c = &self.config;
        let rows = tokens.len();
        let packed = if weight_layout == WeightLayout::PhasePacked && rows > 1 {
            Some(
                self.packed
                    .get()
                    .context("phase-packed weights must be prepared before decoding")?,
            )
        } else {
            None
        };
        ensure!(
            rows > 0 && active.len() == rows,
            "invalid batch decode dimensions"
        );
        ensure!(
            active
                .iter()
                .enumerate()
                .all(|(i, &index)| index < sessions.len() && !active[..i].contains(&index)),
            "active session indices must be unique and in range"
        );
        let simd = sessions[active[0]].simd;
        batch.positions.resize(rows, 0);
        batch.spatial.resize(rows, [f32::NAN; 2]);
        for (row, &index) in active.iter().enumerate() {
            let session = &sessions[index];
            ensure!(session.len < session.capacity, "KV context exhausted");
            ensure!(
                session.next_position < c.max_seq_len,
                "RoPE position exceeds context"
            );
            ensure!(session.simd == simd, "mixed vector backends in one batch");
            batch.positions[row] = session.next_position;
        }
        self.embed(tokens, None, simd, &mut batch.hidden)?;
        let h = &mut batch.hidden;
        let work = &mut batch.work;
        work.resize(rows, c);
        rotary_factors(
            c,
            &batch.positions,
            &batch.spatial,
            self.w(&self.golden),
            &self.temporal,
            &mut work.rope,
        );
        if trace.enabled() {
            trace.tensor(&format!("{phase}.embedding"), &[rows, c.dim], h)?;
        }
        let qdim = c.query_dim();
        let kdim = c.kv_dim();
        let qkv_width = qdim + 2 * kdim;
        for (i, layer) in self.layers.iter().enumerate() {
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            self.linear(
                &work.normalized,
                rows,
                c.dim,
                &layer.qkv,
                packed.map(|p| &p.layers[i].qkv),
                qkv_width,
                &mut work.qkv,
                &mut work.quant_scratch,
                simd,
            )?;
            for row in 0..rows {
                let qkv = &work.qkv[row * qkv_width..(row + 1) * qkv_width];
                for head in 0..c.n_heads {
                    let dst = (row * c.n_heads + head) * c.head_dim;
                    work.q[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[head * c.head_dim..(head + 1) * c.head_dim]);
                    let kvhead = head / (c.n_heads / c.n_kv_heads);
                    let kstart = qdim + kvhead * c.head_dim;
                    work.k[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[kstart..kstart + c.head_dim]);
                    let vstart = qdim + kdim + kvhead * c.head_dim;
                    work.v[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[vstart..vstart + c.head_dim]);
                }
            }
            kernels::rms_norm(&work.q, &mut work.attn, c.head_dim, f32::EPSILON, None);
            std::mem::swap(&mut work.q, &mut work.attn);
            kernels::rms_norm(&work.k, &mut work.attn, c.head_dim, f32::EPSILON, None);
            std::mem::swap(&mut work.k, &mut work.attn);
            for row in 0..rows {
                for head in 0..c.n_heads {
                    let dst = (row * c.n_heads + head) * c.head_dim;
                    for pair in 0..c.head_dim / 2 {
                        let [cos, sin] =
                            work.rope[(row * c.n_heads + head) * (c.head_dim / 2) + pair];
                        for vector in [&mut work.q, &mut work.k] {
                            let p = dst + 2 * pair;
                            let a = vector[p];
                            let b = vector[p + 1];
                            vector[p] = a * cos - b * sin;
                            vector[p + 1] = a * sin + b * cos;
                        }
                    }
                }
            }
            if trace.enabled() {
                trace.tensor(
                    &format!("{phase}.layer.{i}.q"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.q,
                )?;
                trace.tensor(
                    &format!("{phase}.layer.{i}.k"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.k,
                )?;
                trace.tensor(
                    &format!("{phase}.layer.{i}.v"),
                    &[rows, c.n_heads, c.head_dim],
                    &work.v,
                )?;
            }
            for (row, &index) in active.iter().enumerate() {
                let session = &mut sessions[index];
                let offset = session.len;
                let cache = &mut session.layers[i];
                let range = row * qdim..(row + 1) * qdim;
                cache.append(&work.k[range.clone()], &work.v[range.clone()], offset, c);
                cache.attention(
                    &work.q[range.clone()],
                    1,
                    offset + 1,
                    c,
                    offset,
                    session.image_start,
                    session.image_end,
                    self.w(&layer.sinks),
                    &mut work.attn[range],
                    simd,
                );
            }
            if trace.enabled() {
                trace.tensor(
                    &format!("{phase}.layer.{i}.attention"),
                    &[rows, qdim],
                    &work.attn,
                )?;
            }
            self.linear(
                &work.attn,
                rows,
                qdim,
                &layer.wo,
                packed.map(|p| &p.layers[i].wo),
                c.dim,
                &mut work.projected,
                &mut work.quant_scratch,
                simd,
            )?;
            for (x, a) in h.iter_mut().zip(&work.projected) {
                *x += a;
            }
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            self.linear(
                &work.normalized,
                rows,
                c.dim,
                &layer.w13,
                packed.map(|p| &p.layers[i].w13),
                2 * c.ffn_dim,
                &mut work.ffn_packed,
                &mut work.quant_scratch,
                simd,
            )?;
            kernels::squared_relu_gate(&work.ffn_packed, &mut work.gated);
            self.linear(
                &work.gated,
                rows,
                c.ffn_dim,
                &layer.w2,
                packed.map(|p| &p.layers[i].w2),
                c.dim,
                &mut work.projected,
                &mut work.quant_scratch,
                simd,
            )?;
            for (x, a) in h.iter_mut().zip(&work.projected) {
                *x += a;
            }
            if trace.enabled() {
                trace.tensor(&format!("{phase}.layer.{i}.hidden"), &[rows, c.dim], h)?;
            }
        }
        for &index in active {
            sessions[index].len += 1;
            sessions[index].next_position += 1;
        }
        kernels::rms_norm(
            h,
            &mut work.normalized,
            c.dim,
            c.norm_eps,
            Some(self.w(&self.norm)),
        );
        work.logits.resize(rows * c.vocab_size, 0.);
        self.linear(
            &work.normalized,
            rows,
            c.dim,
            &self.output,
            packed.map(|p| &p.output),
            c.vocab_size,
            &mut work.logits,
            &mut work.quant_scratch,
            simd,
        )?;
        if trace.enabled() {
            trace.tensor(
                &format!("{phase}.logits"),
                &[rows, c.vocab_size],
                &work.logits,
            )?;
        }
        Ok(&work.logits)
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_linear(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weights: &[f32],
    packed: Option<&PhasePackedLinear>,
    output_dim: usize,
    output: &mut [f32],
    simd: kernels::Simd,
) {
    if let Some(packed) = packed.filter(|_| (2..=8).contains(&rows)) {
        debug_assert_eq!(simd.resolved(), kernels::Simd::Avx2);
        packed.linear_avx2(input, rows, output);
    } else {
        kernels::linear_with_simd(input, rows, input_dim, weights, output_dim, output, simd);
    }
}

/// Allocated once before decode; shrinking active rows never grows these buffers.
pub(crate) struct BatchWorkspace {
    hidden: Vec<f32>,
    positions: Vec<usize>,
    spatial: Vec<[f32; 2]>,
    work: Workspace,
}
impl BatchWorkspace {
    pub(crate) fn prepare_attempt(&mut self, rows: usize, c: &ModelConfig) {
        self.work.quant_scratch.reserve_decode(rows, c.vocab_size);
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
    layers: Vec<LayerCache>,
    workspace: Workspace,
    pub len: usize,
    pub next_position: usize,
    capacity: usize,
    image_start: usize,
    image_end: usize,
    simd: kernels::Simd,
}
enum LayerCache {
    Split(crate::attempt::prefix::SplitPrefix),
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
    fn append(&mut self, k: &[f32], v: &[f32], offset: usize, c: &ModelConfig) {
        match self {
            Self::Split(cache) => {
                append_unique_heads(&mut cache.generated_k, k, c);
                append_unique_heads(&mut cache.generated_v, v, c);
            }
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
                let prefix_rows = prefix_len
                    .saturating_sub(offset)
                    .min(k.len() / c.query_dim());
                let prefix_elements = prefix_rows * c.query_dim();
                prefix_k.extend_from_slice(&k[..prefix_elements]);
                append_unique_heads(generated_k, &k[prefix_elements..], c);
                append_unique_heads(values, v, c);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
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
    ) {
        match self {
            Self::Split(cache) => {
                assert_eq!(rows, 1, "sealed prefix supports text decode only");
                assert!(
                    offset >= image_end,
                    "cannot treat a partial image as causal decode"
                );
                cache.attention_decode(q, total_len, sinks, output, simd);
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
            } => kernels::attention_compact_with_simd(
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
            ),
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
    pub(crate) fn seal_prefix(
        &mut self,
        c: &ModelConfig,
        mode: crate::attempt::PrefixMode,
    ) -> Result<()> {
        if mode == crate::attempt::PrefixMode::Reference {
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
                let packed = crate::attempt::prefix::SplitPrefix::from_compact(
                    prefix_k,
                    v,
                    *prefix_len,
                    self.capacity,
                    c,
                    mode,
                )?;
                *layer = LayerCache::Split(packed);
            } else {
                bail!("attempt prefix sealing requires an unsealed compact cache");
            }
        }
        Ok(())
    }
    pub(crate) fn prepare_small_decode(&mut self, c: &ModelConfig) {
        self.workspace = Workspace::default();
        self.workspace.resize(1, c);
        self.workspace.rope.resize(c.query_dim() / 2, [1.0, 0.0]);
        self.workspace.logits.resize(c.vocab_size, 0.0);
        self.workspace.quant_scratch.reserve_decode(1, c.vocab_size);
    }
    /// Sequential batch prefills share one scratch allocation. The preceding
    /// request has already consumed its logits; only its KV cache is retained.
    pub(crate) fn reuse_workspace_from(&mut self, previous: &mut Self) {
        std::mem::swap(&mut self.workspace, &mut previous.workspace);
    }

    pub(crate) fn release_workspace(&mut self) {
        self.workspace = Workspace::default();
    }

    pub fn new(
        c: &ModelConfig,
        capacity: usize,
        prefix_len: usize,
        image_start: usize,
        image_end: usize,
        simd: kernels::Simd,
        cache_layout: CacheLayout,
    ) -> Result<Self> {
        ensure!(capacity <= c.max_seq_len, "capacity exceeds model context");
        ensure!(
            prefix_len > 0
                && prefix_len <= capacity
                && image_start < image_end
                && image_end < prefix_len,
            "invalid cache prefix or image interval"
        );
        let reserve = |rows: usize, width: usize| -> Result<Vec<f32>> {
            let elements = rows.checked_mul(width).context("cache size overflow")?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(elements)
                .context("allocate KV cache")?;
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
        })
    }
}

#[derive(Default)]
struct Workspace {
    quant_scratch: crate::attempt::quant::Scratch,
    normalized: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    projected: Vec<f32>,
    ffn_packed: Vec<f32>,
    gated: Vec<f32>,
    logits: Vec<f32>,
    rope: Vec<[f32; 2]>,
}
impl Workspace {
    fn resize(&mut self, rows: usize, c: &ModelConfig) {
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

pub(crate) fn temporal_factors(c: &ModelConfig) -> Vec<[f32; 2]> {
    let pairs = c.head_dim / 4;
    let mut result = Vec::with_capacity(c.max_seq_len * pairs);
    for t in 0..c.max_seq_len {
        for pair in 0..pairs {
            let freq = 1.
                / c.rope_theta
                    .powf((2 * pair) as f32 / (c.head_dim / 2) as f32);
            let (sin, cos) = (t as f32 * freq).sin_cos();
            result.push([cos, sin]);
        }
    }
    result
}

pub(crate) fn rotary_factors(
    c: &ModelConfig,
    t: &[usize],
    hw: &[[f32; 2]],
    golden: &[f32],
    temporal: &[[f32; 2]],
    result: &mut Vec<[f32; 2]>,
) {
    let pairs = c.head_dim / 2;
    let temporal_pairs = pairs / 2;
    result.resize(t.len() * c.n_heads * pairs, [1., 0.]);
    for row in 0..t.len() {
        for head in 0..c.n_heads {
            for pair in 0..pairs {
                if pair < temporal_pairs {
                    result[(row * c.n_heads + head) * pairs + pair] =
                        temporal[t[row] * temporal_pairs + pair];
                    continue;
                }
                let angle = if hw[row][0].is_finite() && hw[row][1].is_finite() {
                    let index = (head * temporal_pairs + pair - temporal_pairs) * 2;
                    hw[row][0] * golden[index] + hw[row][1] * golden[index + 1]
                } else {
                    0.
                };
                let (sin, cos) = angle.sin_cos();
                result[(row * c.n_heads + head) * pairs + pair] = [cos, sin];
            }
        }
    }
}

pub(crate) fn image_range(tokens: &[u32], c: &ModelConfig) -> Result<(usize, usize)> {
    let start = tokens
        .iter()
        .position(|&x| x == c.image_cls_token_id)
        .context("missing image class token")?;
    let end = tokens
        .iter()
        .position(|&x| x == c.img_end_id)
        .context("missing image end token")?;
    ensure!(start < end, "invalid image token range");
    ensure!(
        tokens
            .iter()
            .filter(|&&x| x == c.image_cls_token_id)
            .count()
            == 1
            && tokens.iter().filter(|&&x| x == c.img_end_id).count() == 1,
        "exactly one image per request is supported"
    );
    Ok((start, end))
}

pub(crate) fn positions(
    tokens: &[u32],
    patch_hw: &[[f32; 2]],
    c: &ModelConfig,
) -> Result<(Vec<usize>, Vec<[f32; 2]>)> {
    let mut temporal = Vec::with_capacity(tokens.len());
    let mut spatial = Vec::with_capacity(tokens.len());
    let mut count = 0usize;
    let mut patch = 0;
    for &token in tokens {
        if ![
            c.img_id,
            c.image_reg_1_token_id,
            c.image_reg_2_token_id,
            c.image_reg_3_token_id,
            c.image_reg_4_token_id,
            c.img_end_id,
        ]
        .contains(&token)
        {
            count += 1;
        }
        ensure!(count > 0, "image continuation before any class/text token");
        temporal.push(count - 1);
        spatial.push(if token == c.img_id {
            let p = *patch_hw.get(patch).context("missing patch coordinates")?;
            patch += 1;
            p
        } else {
            [f32::NAN, f32::NAN]
        });
    }
    if patch != patch_hw.len() {
        bail!("unused patch coordinates");
    }
    Ok((temporal, spatial))
}

#[cfg(test)]
#[path = "model_diagnostics.rs"]
mod diagnostics;

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ModelConfig {
        serde_json::from_str(include_str!("../tests/fixtures/model-config.json")).unwrap()
    }

    #[test]
    fn image_positions_preserve_registers_and_resume_text() {
        let c = config();
        let tokens = [244, 245, 246, 247, 248, 227, 227, 230, 524, 257];
        let (t, hw) = positions(&tokens, &[[-1., -0.5], [1., 0.5]], &c).unwrap();
        assert_eq!(t, [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]);
        assert!(hw[..5].iter().flatten().all(|v| v.is_nan()));
        assert_eq!(hw[5], [-1., -0.5]);
        assert_eq!(image_range(&tokens, &c).unwrap(), (0, 7));
        assert!(positions(&[227], &[[0., 0.]], &c).is_err());
        assert!(image_range(&[244, 230, 244, 230], &c).is_err());
    }

    #[test]
    fn spatial_rotary_makes_paired_prefix_keys_distinct() {
        let c = config();
        let temporal = temporal_factors(&c);
        let golden = (0..c.n_heads * c.head_dim / 2)
            .map(|i| i as f32 / 256.)
            .collect::<Vec<_>>();
        let mut rope = Vec::new();
        rotary_factors(
            &c,
            &[0, 16383],
            &[[0.2, 0.7], [f32::NAN; 2]],
            &golden,
            &temporal,
            &mut rope,
        );
        let pairs = c.head_dim / 2;
        assert_ne!(&rope[..pairs], &rope[pairs..2 * pairs]);
        for h in 1..c.n_heads {
            assert_eq!(
                &rope[c.n_heads * pairs..(c.n_heads + 1) * pairs],
                &rope[(c.n_heads + h) * pairs..(c.n_heads + h + 1) * pairs]
            );
        }
    }

    #[test]
    #[ignore = "diagnostic requires independently exported GPU RoPE operators"]
    fn report_rotary_operator_differences() {
        let bytes = std::fs::read("artifacts/reference/rope-operators.safetensors").unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let f32s = |name: &str| -> Vec<f32> {
            let t = tensors.tensor(name).unwrap();
            assert_eq!(t.dtype(), Dtype::F32);
            t.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        let c = config();
        let temporal = temporal_factors(&c);
        let reference = f32s("freqs_cis_all");
        let delta = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            let mut max = 0f64;
            let mut squares = 0f64;
            let mut unequal = 0usize;
            for (&x, &y) in a.iter().zip(b) {
                if x.is_nan() && y.is_nan() {
                    continue;
                }
                let d = (x as f64 - y as f64).abs();
                max = max.max(d);
                squares += d * d;
                unequal += usize::from(x.to_bits() != y.to_bits());
            }
            serde_json::json!({"max_abs":max,"rms_abs":(squares/a.len() as f64).sqrt(),"unequal":unequal})
        };
        let time = tensors.tensor("pos_t").unwrap();
        assert_eq!(time.dtype(), Dtype::I64);
        let time = time
            .data()
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()) as usize)
            .collect::<Vec<_>>();
        let hw = f32s("pos_hw")
            .chunks_exact(2)
            .map(|x| [x[0], x[1]])
            .collect::<Vec<_>>();
        let golden = f32s("golden_freqs");
        let mut rope = Vec::new();
        rotary_factors(&c, &time, &hw, &golden, &temporal, &mut rope);
        let mut report =
            serde_json::json!({"temporal_table":delta(bytemuck::cast_slice(&temporal),&reference)});
        for layer in [0, 6, 12, 21] {
            for kind in ["q", "k"] {
                let mut values = f32s(&format!("layer.{layer}.{kind}.input"));
                let expected = f32s(&format!("layer.{layer}.{kind}.expected"));
                assert_eq!(values.len() / 2, rope.len());
                for (v, &[cos, sin]) in values.chunks_exact_mut(2).zip(&rope) {
                    let a = v[0];
                    let b = v[1];
                    v[0] = a * cos - b * sin;
                    v[1] = a * sin + b * cos;
                }
                report[format!("layer.{layer}.{kind}")] = delta(&values, &expected);
            }
        }
        std::fs::write(
            "reference/rotary-rust-probe.json",
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
