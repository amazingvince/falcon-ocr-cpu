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
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::{
    config::{CONFIG_SHA256, CacheLayout, ModelConfig, WEIGHTS_SHA256, WeightLayout},
    head_screen::Screened,
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

/// Checkpoint bytes: the read-only file mapping, or an owned copy whose
/// tensors were rounded (`Model::round_weights_to_bf16`).
enum Store {
    Mapped(Mmap),
    /// 4-byte aligned copy of the whole file.
    Owned(Vec<u32>),
}
impl Store {
    fn bytes(&self) -> &[u8] {
        match self {
            Store::Mapped(map) => &map[..],
            Store::Owned(words) => bytemuck::cast_slice(words),
        }
    }
}

/// An immutable checkpoint. Keep its backing file unchanged while the model is loaded.
pub struct Model {
    pub(crate) config: ModelConfig,
    map: Store,
    embedding: Weight,
    projector: Weight,
    norm: Weight,
    output: Weight,
    golden: Weight,
    temporal: Vec<[f32; 2]>,
    layers: Vec<Layer>,
    packed: OnceLock<PackedWeights>,
    screened: OnceLock<crate::head_screen::ScreenedHead>,
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
    /// Build and verify the INT8 screen of the FP32 vocabulary head once per
    /// model. Greedy selection through it is exact; see `head_screen`.
    pub(crate) fn prepare_screened_head(&self) -> Result<()> {
        ensure!(
            self.output.quantized.is_none(),
            "the screened head requires the FP32 vocabulary head"
        );
        if self.screened.get().is_none() {
            let c = &self.config;
            let head =
                crate::head_screen::ScreenedHead::build(self.w(&self.output), c.vocab_size, c.dim)?;
            let _ = self.screened.set(head);
        }
        Ok(())
    }
    /// Payload of the INT8 head screen, or zero if it was never prepared.
    pub fn screened_head_bytes(&self) -> usize {
        self.screened.get().map_or(0, |h| h.payload_bytes())
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
            map: Store::Mapped(map),
            embedding,
            projector,
            norm,
            output,
            golden,
            temporal,
            layers,
            packed: OnceLock::new(),
            screened: OnceLock::new(),
            attempt_profile: crate::attempt::Profile::Reference,
            attempt_setup_ms: 0.0,
            attempt_artifact_sha256: None,
            weights_sha256: hash,
        })
    }

    /// Load a separately labelled experiment without changing the reference loader.
    /// Source weights remain mmap-backed for protected tensors and the oracle.
    /// The mapping length is NOT equivalent to additional committed/resident RAM.
    /// Round every checkpoint tensor to BF16 precision (round to nearest,
    /// ties to even; the values stay FP32). Production serves this model in
    /// BF16: FP32 arithmetic on these weights is several times closer to its
    /// outputs than FP32 arithmetic on the original weights. Copies the
    /// checkpoint into memory; call before any W8 overlay or screened head.
    pub fn round_weights_to_bf16(&mut self) {
        let bytes = self.map.bytes();
        let mut words = vec![0u32; bytes.len().div_ceil(4)];
        bytemuck::cast_slice_mut::<u32, u8>(&mut words)[..bytes.len()].copy_from_slice(bytes);
        let ranges = [
            &self.embedding,
            &self.projector,
            &self.norm,
            &self.output,
            &self.golden,
        ]
        .into_iter()
        .chain(
            self.layers
                .iter()
                .flat_map(|l| [&l.qkv, &l.wo, &l.w13, &l.w2, &l.sinks]),
        )
        .map(|w| w.range.clone())
        .collect::<Vec<_>>();
        for range in ranges {
            debug_assert!(range.start % 4 == 0 && range.end % 4 == 0);
            for bits in &mut words[range.start / 4..range.end / 4] {
                *bits = round_bf16_bits(*bits);
            }
        }
        self.map = Store::Owned(words);
    }

    pub fn load_attempt(
        dir: impl AsRef<Path>,
        profile: crate::attempt::Profile,
        artifact: Option<&Path>,
    ) -> Result<Self> {
        Self::load_attempt_with(dir, profile, artifact, &[])
    }

    /// [`Model::load_attempt`] that also keeps every body matrix whose name
    /// matches one of `keep_fp32` (`*` matches any run of characters) in
    /// FP32, whatever the overlay holds: mixed precision without writing a
    /// new overlay.
    pub fn load_attempt_with(
        dir: impl AsRef<Path>,
        profile: crate::attempt::Profile,
        artifact: Option<&Path>,
        keep_fp32: &[String],
    ) -> Result<Self> {
        Self::load_attempt_bf16(dir, profile, artifact, keep_fp32, false)
    }

    /// [`Model::load_attempt_with`], optionally on BF16-rounded weights
    /// ([`Model::round_weights_to_bf16`]), which unquantized matrices,
    /// embeddings, norms and the head then use.
    pub fn load_attempt_bf16(
        dir: impl AsRef<Path>,
        profile: crate::attempt::Profile,
        artifact: Option<&Path>,
        keep_fp32: &[String],
        weights_bf16: bool,
    ) -> Result<Self> {
        let mut model = Self::load(dir)?;
        if weights_bf16 {
            model.round_weights_to_bf16();
        }
        model.attempt_profile = profile;
        ensure!(
            artifact.is_none() || profile.quantizes_body(),
            "W8 artifact supplied to an FP32 profile"
        );
        if !profile.quantizes_body() {
            return Ok(model);
        }
        let started = Instant::now();
        ensure!(
            artifact.is_none() || profile.weight_bits() == 8,
            "W8 artifacts apply only to 8-bit profiles; 16-bit weights are quantized at load"
        );
        let bytes = artifact.map(std::fs::read).transpose()?;
        let tensors = bytes
            .as_ref()
            .map(|b| SafeTensors::deserialize(b))
            .transpose()?;
        // Overlay options: group size 32 or 64; `partial` overlays leave every
        // matrix they omit in FP32 (mixed precision).
        let mut group_size = 64;
        let mut partial = false;
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
            group_size = match meta.get("group_size").and_then(|v| v.as_str()) {
                Some("32") => 32,
                Some("64") => 64,
                _ => anyhow::bail!("W8 metadata group_size must be 32 or 64"),
            };
            partial = meta.get("partial").and_then(|v| v.as_str()) == Some("true");
            check("scale_dtype", "f32")?;
            ensure!(
                meta.get("rounding").and_then(|v| v.as_str()).is_some(),
                "W8 metadata rounding missing"
            );
            check("activation_dtype", "f32")?;
            model.attempt_artifact_sha256 = Some(format!("{:x}", Sha256::digest(bytes)));
            let expected = 2 * (4 * model.config.n_layers + (profile.quantizes_head() as usize));
            let count = tensors.as_ref().unwrap().len();
            ensure!(
                count == expected || (partial && count < expected && count % 2 == 0),
                "W8 artifact has unexpected tensors"
            );
        }
        // Build separately before mutating Weight handles (and their source views).
        let mut prepared = Vec::new();
        let make = |name: &str,
                    w: &Weight,
                    input: usize,
                    output: usize|
         -> Result<Option<Arc<crate::attempt::quant::Q8Linear>>> {
            use crate::attempt::quant::Q8Linear;
            if keep_fp32.iter().any(|pattern| glob_match(pattern, name)) {
                return Ok(None);
            }
            let q = if let Some(tensors) = &tensors {
                let codes_name = format!("{name}.__w8_codes");
                if partial && !tensors.names().iter().any(|n| **n == codes_name) {
                    return Ok(None);
                }
                let codes = tensors.tensor(&codes_name)?;
                let scales = tensors.tensor(&format!("{name}.__w8_scales"))?;
                ensure!(
                    codes.dtype() == Dtype::I8 && codes.shape() == [output, input],
                    "W8 codes dtype/shape for {name}"
                );
                ensure!(
                    scales.dtype() == Dtype::F32
                        && scales.shape() == [output, input.div_ceil(group_size)],
                    "W8 scales dtype/shape for {name}"
                );
                let codes = codes.data().iter().map(|&x| x as i8).collect();
                let scales = scales
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                Q8Linear::from_parts(output, input, group_size, codes, scales)?
            } else {
                Q8Linear::quantize_bits(model.w(w), output, input, 64, profile.weight_bits())
                    .map_err(anyhow::Error::msg)?
            };
            Ok(Some(Arc::new(q)))
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
            l.qkv.quantized = prepared.next().flatten();
            l.wo.quantized = prepared.next().flatten();
            l.w13.quantized = prepared.next().flatten();
            l.w2.quantized = prepared.next().flatten();
        }
        if profile.quantizes_head() {
            model.output.quantized = prepared.next().flatten();
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
        serde_json::json!({"profile":self.attempt_profile,"source_mapping_bytes":self.map.bytes().len(),
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

    /// W13 projection followed by the squared-ReLU gate. FP32 weights with
    /// 1..=8 rows and no phase-packed copy use the fused kernel; every other
    /// case keeps the two original operations. Results are bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn linear_glu(
        &self,
        input: &[f32],
        rows: usize,
        w13: &Weight,
        packed: Option<&PhasePackedLinear>,
        intermediate: &mut [f32],
        gated: &mut [f32],
        scratch: &mut crate::attempt::quant::Scratch,
        simd: kernels::Simd,
    ) -> Result<()> {
        let c = &self.config;
        if let Some(q) = &w13.quantized
            && q.linear_glu(input, rows, gated, simd)?
        {
            return Ok(());
        }
        let packed_rows = packed.is_some() && (2..=8).contains(&rows);
        if w13.quantized.is_none()
            && !packed_rows
            && kernels::linear_glu_with_simd(
                input,
                rows,
                c.dim,
                self.w(w13),
                c.ffn_dim,
                gated,
                simd,
            )
        {
            return Ok(());
        }
        self.linear(
            input,
            rows,
            c.dim,
            w13,
            packed,
            2 * c.ffn_dim,
            intermediate,
            scratch,
            simd,
        )?;
        kernels::squared_relu_gate(intermediate, gated);
        Ok(())
    }

    fn w(&self, weight: &Weight) -> &[f32] {
        bytemuck::cast_slice(&self.map.bytes()[weight.range.clone()])
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
        match self.forward_next(h, positions, positions_hw, session, trace, phase, false)? {
            Next::Logits(logits) => Ok(logits),
            Next::Tokens(_) => unreachable!("full head requested"),
        }
    }

    /// `forward`, optionally selecting the greedy token through the exact
    /// screened head. Tracing always evaluates and records the full logits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_next<'s>(
        &self,
        h: &mut [f32],
        positions: &[usize],
        positions_hw: &[[f32; 2]],
        session: &'s mut Session,
        trace: &mut dyn Trace,
        phase: &str,
        screen: bool,
    ) -> Result<Next<'s>> {
        self.forward_layers(h, positions, positions_hw, session, trace, phase)?;
        let c = &self.config;
        let rows = positions.len();
        let simd = session.simd;
        let work = &mut session.workspace;
        // Generation needs only the final token's vocabulary projection.
        let last = &h[(rows - 1) * c.dim..];
        let norm = &mut work.normalized[..c.dim];
        kernels::rms_norm(last, norm, c.dim, c.norm_eps, Some(self.w(&self.norm)));
        if screen
            && !trace.enabled()
            && let Some(head) = self.screened.get()
            && !work.selected.is_empty()
        {
            let dot = kernels::dot_kernel(simd.resolved());
            match head.select(norm, self.w(&self.output), dot, &mut work.head) {
                Screened::Token { token, candidates } => {
                    trace.head_screen(candidates, false);
                    work.selected[0] = token;
                    return Ok(Next::Tokens(&work.selected[..1]));
                }
                Screened::Fallback { candidates } => trace.head_screen(candidates, true),
            }
        }
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
        Ok(Next::Logits(&work.logits))
    }

    /// The transformer layers of `forward_next`: appends the rows' keys and
    /// values to the session and leaves the final hidden states in `h`.
    fn forward_layers(
        &self,
        h: &mut [f32],
        positions: &[usize],
        positions_hw: &[[f32; 2]],
        session: &mut Session,
        trace: &mut dyn Trace,
        phase: &str,
    ) -> Result<()> {
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
        let mut clock = PhaseClock::new(rows);
        // RoPE factors are shared by all transformer layers.
        rotary_factors(
            c,
            positions,
            positions_hw,
            self.w(&self.golden),
            &self.temporal,
            &mut work.rope,
        );
        clock.mark(0);
        let capture = trace.captures_linear_inputs();
        for (i, layer) in self.layers.iter().enumerate() {
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            if capture {
                trace.linear_input(i, "qkv", rows, &work.normalized);
            }
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
            clock.mark(1);
            // Normalize each original K head before GQA expansion. Spatial rotations
            // subsequently differ for paired heads, so expanded keys are intentional.
            let split = |qkv: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]| {
                for head in 0..c.n_heads {
                    let dst = head * c.head_dim;
                    q[dst..dst + c.head_dim]
                        .copy_from_slice(&qkv[head * c.head_dim..(head + 1) * c.head_dim]);
                    let kvhead = head / (c.n_heads / c.n_kv_heads);
                    let kstart = qdim + kvhead * c.head_dim;
                    k[dst..dst + c.head_dim].copy_from_slice(&qkv[kstart..kstart + c.head_dim]);
                    let vstart = qdim + kdim + kvhead * c.head_dim;
                    v[dst..dst + c.head_dim].copy_from_slice(&qkv[vstart..vstart + c.head_dim]);
                }
            };
            if rows >= PARALLEL_ROWS {
                work.qkv
                    .par_chunks(qkv_width)
                    .zip(work.q.par_chunks_mut(qdim))
                    .zip(work.k.par_chunks_mut(qdim))
                    .zip(work.v.par_chunks_mut(qdim))
                    .for_each(|(((qkv, q), k), v)| split(qkv, q, k, v));
            } else {
                for (((qkv, q), k), v) in work
                    .qkv
                    .chunks(qkv_width)
                    .zip(work.q.chunks_mut(qdim))
                    .zip(work.k.chunks_mut(qdim))
                    .zip(work.v.chunks_mut(qdim))
                {
                    split(qkv, q, k, v);
                }
            }
            // Normalize all heads in two calls; avoid creating a Rayon operation
            // for each individual 64-element head.
            kernels::rms_norm(&work.q, &mut work.attn, c.head_dim, f32::EPSILON, None);
            std::mem::swap(&mut work.q, &mut work.attn);
            kernels::rms_norm(&work.k, &mut work.attn, c.head_dim, f32::EPSILON, None);
            std::mem::swap(&mut work.k, &mut work.attn);
            let rotate = |rope: &[[f32; 2]], q: &mut [f32], k: &mut [f32]| {
                for head in 0..c.n_heads {
                    let dst = head * c.head_dim;
                    for pair in 0..c.head_dim / 2 {
                        let [cos, sin] = rope[head * (c.head_dim / 2) + pair];
                        for vector in [&mut *q, &mut *k] {
                            let p = dst + 2 * pair;
                            let a = vector[p];
                            let b = vector[p + 1];
                            vector[p] = a * cos - b * sin;
                            vector[p + 1] = a * sin + b * cos;
                        }
                    }
                }
            };
            let rope_width = c.n_heads * (c.head_dim / 2);
            if rows >= PARALLEL_ROWS {
                work.rope[..rows * rope_width]
                    .par_chunks(rope_width)
                    .zip(work.q.par_chunks_mut(qdim))
                    .zip(work.k.par_chunks_mut(qdim))
                    .for_each(|((rope, q), k)| rotate(rope, q, k));
            } else {
                for ((rope, q), k) in work.rope[..rows * rope_width]
                    .chunks(rope_width)
                    .zip(work.q.chunks_mut(qdim))
                    .zip(work.k.chunks_mut(qdim))
                {
                    rotate(rope, q, k);
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
            clock.mark(2);
            let cache = &mut session.layers[i];
            cache.append(&work.k, &work.v, offset, c);
            clock.mark(3);
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
            clock.mark(4);
            if trace.enabled() {
                trace.tensor(
                    &format!("{phase}.layer.{i}.attention"),
                    &[rows, qdim],
                    &work.attn,
                )?;
            }
            if capture {
                trace.linear_input(i, "wo", rows, &work.attn);
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
            add_residual(h, &work.projected, c.dim);
            clock.mark(5);
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            if capture {
                trace.linear_input(i, "w13", rows, &work.normalized);
            }
            self.linear_glu(
                &work.normalized,
                rows,
                &layer.w13,
                None,
                &mut work.ffn_packed,
                &mut work.gated,
                &mut work.quant_scratch,
                simd,
            )?;
            clock.mark(6);
            if capture {
                trace.linear_input(i, "w2", rows, &work.gated);
            }
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
            add_residual(h, &work.projected, c.dim);
            clock.mark(7);
            if trace.enabled() {
                trace.tensor(&format!("{phase}.layer.{i}.hidden"), &[rows, c.dim], h)?;
            }
        }
        session.len += rows;
        session.next_position = positions[rows - 1] + 1;
        Ok(())
    }

    /// Verify drafted tokens: `h` holds the embeddings of the last accepted
    /// token followed by drafts at consecutive text `positions` (at most
    /// `head_screen::MAX_ROWS`). Runs the layers once for all rows and returns
    /// the greedy next token of every row: `Tokens` through the exact screened
    /// head, or full per-row `Logits` when any row falls outside the screen.
    /// Per row this is bitwise the single-row `forward_next` step.
    pub(crate) fn verify_next<'s>(
        &self,
        h: &mut [f32],
        positions: &[usize],
        session: &'s mut Session,
        screen: bool,
    ) -> Result<Next<'s>> {
        let rows = positions.len();
        ensure!(
            (2..=crate::head_screen::MAX_ROWS).contains(&rows),
            "verification needs 2..=8 rows"
        );
        let text = vec![[f32::NAN; 2]; rows];
        self.forward_layers(h, positions, &text, session, &mut crate::trace::NoTrace, "")?;
        let c = &self.config;
        let simd = session.simd;
        let work = &mut session.workspace;
        let norm = &mut work.normalized[..rows * c.dim];
        kernels::rms_norm(h, norm, c.dim, c.norm_eps, Some(self.w(&self.norm)));
        if screen && let Some(head) = self.screened.get() {
            if work.head_rows.len() < crate::head_screen::MAX_ROWS {
                work.head_rows
                    .resize_with(crate::head_screen::MAX_ROWS, Default::default);
            }
            let mut results = [Screened::Fallback { candidates: 0 }; crate::head_screen::MAX_ROWS];
            let dot = kernels::dot_kernel(simd.resolved());
            head.select_rows(
                norm,
                rows,
                self.w(&self.output),
                dot,
                &mut work.head_rows,
                &mut results,
            );
            if results[..rows]
                .iter()
                .all(|r| matches!(r, Screened::Token { .. }))
            {
                if work.selected.len() < rows {
                    work.selected.resize(rows, 0);
                }
                for (slot, result) in work.selected.iter_mut().zip(&results[..rows]) {
                    if let Screened::Token { token, .. } = result {
                        *slot = *token;
                    }
                }
                return Ok(Next::Tokens(&work.selected[..rows]));
            }
        }
        work.logits.resize(rows * c.vocab_size, 0.);
        self.linear(
            &work.normalized[..rows * c.dim],
            rows,
            c.dim,
            &self.output,
            None,
            c.vocab_size,
            &mut work.logits,
            &mut work.quant_scratch,
            simd,
        )?;
        Ok(Next::Logits(&work.logits[..rows * c.vocab_size]))
    }

    /// Advance one generated token per active request. Linear projections share
    /// one matrix operation, while attention and KV storage remain per request.
    /// `decode_batch`, optionally selecting every row's greedy token through
    /// the exact screened head. Any row outside the screen's assumptions makes
    /// the whole step evaluate full logits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_batch_next<'w>(
        &self,
        tokens: &[u32],
        active: &[usize],
        sessions: &mut [Session],
        batch: &'w mut BatchWorkspace,
        weight_layout: WeightLayout,
        trace: &mut dyn Trace,
        phase: &str,
        screen: bool,
    ) -> Result<Next<'w>> {
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
            add_residual(h, &work.projected, c.dim);
            kernels::rms_norm(h, &mut work.normalized, c.dim, f32::EPSILON, None);
            self.linear_glu(
                &work.normalized,
                rows,
                &layer.w13,
                packed.map(|p| &p.layers[i].w13),
                &mut work.ffn_packed,
                &mut work.gated,
                &mut work.quant_scratch,
                simd,
            )?;
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
            add_residual(h, &work.projected, c.dim);
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
        if screen
            && !trace.enabled()
            && let Some(head) = self.screened.get()
            && work.selected.len() >= rows
        {
            let dot = kernels::dot_kernel(simd.resolved());
            let fp32 = self.w(&self.output);
            let mut complete = true;
            for row in 0..rows {
                let x = &work.normalized[row * c.dim..(row + 1) * c.dim];
                match head.select(x, fp32, dot, &mut work.head) {
                    Screened::Token { token, candidates } => {
                        trace.head_screen(candidates, false);
                        work.selected[row] = token;
                    }
                    Screened::Fallback { candidates } => {
                        trace.head_screen(candidates, true);
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                return Ok(Next::Tokens(&work.selected[..rows]));
            }
        }
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
        Ok(Next::Logits(&work.logits))
    }
}

/// Greedy decision input produced by one forward step.
pub(crate) enum Next<'a> {
    /// Full FP32 logits, `[rows][vocab]`; the caller selects the token.
    Logits(&'a [f32]),
    /// Exact FP32 argmax per row, selected through the screened head.
    Tokens(&'a [u32]),
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
                assert!(
                    offset >= image_end,
                    "cannot treat a partial image as causal decode"
                );
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
    /// Drop the newest positions so that `len` remain (rejected draft
    /// tokens). Needs the split cache of the attempt profiles.
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
    /// Screened-head scratch and per-row selected tokens; reserved before decode.
    head: crate::head_screen::HeadScratch,
    selected: Vec<u32>,
    /// Per-row screened-head scratch for draft verification.
    head_rows: Vec<crate::head_screen::HeadScratch>,
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
    let fill = |row: usize, factors: &mut [[f32; 2]]| {
        for head in 0..c.n_heads {
            for pair in 0..pairs {
                if pair < temporal_pairs {
                    factors[head * pairs + pair] = temporal[t[row] * temporal_pairs + pair];
                    continue;
                }
                let angle = if hw[row][0].is_finite() && hw[row][1].is_finite() {
                    let index = (head * temporal_pairs + pair - temporal_pairs) * 2;
                    hw[row][0] * golden[index] + hw[row][1] * golden[index + 1]
                } else {
                    0.
                };
                let (sin, cos) = angle.sin_cos();
                factors[head * pairs + pair] = [cos, sin];
            }
        }
    };
    let width = c.n_heads * pairs;
    if t.len() >= PARALLEL_ROWS {
        result
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, factors)| fill(row, factors));
    } else {
        for (row, factors) in result.chunks_mut(width).enumerate() {
            fill(row, factors);
        }
    }
}

/// `h += projected`, row-parallel for prefill-sized inputs.
fn add_residual(h: &mut [f32], projected: &[f32], dim: usize) {
    if h.len() >= PARALLEL_ROWS * dim {
        h.par_chunks_mut(dim)
            .zip(projected.par_chunks(dim))
            .for_each(|(h, a)| {
                for (x, a) in h.iter_mut().zip(a) {
                    *x += a;
                }
            });
    } else {
        for (x, a) in h.iter_mut().zip(projected) {
            *x += a;
        }
    }
}

/// Opt-in wall-clock split of forwards (`FALCON_OCR_PHASES=1`) on stderr.
/// Prefill-sized forwards print immediately; decode steps accumulate on the
/// calling thread until `report_decode_phases`. Disabled, a mark is a branch.
struct PhaseClock {
    enabled: bool,
    rows: usize,
    last: Instant,
    totals: [f64; PHASES],
}
const PHASES: usize = 9;
const PHASE_NAMES: [&str; PHASES] = [
    "rope_factors",
    "norm+qkv",
    "split+qk_norm+rope",
    "cache_append",
    "attention",
    "wo+residual",
    "norm+w13+gate",
    "w2+residual",
    "final_norm+head",
];
thread_local! {
    static DECODE_PHASES: std::cell::RefCell<([f64; PHASES], usize)> =
        const { std::cell::RefCell::new(([0.0; PHASES], 0)) };
}
fn phases_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FALCON_OCR_PHASES").is_some())
}
impl PhaseClock {
    fn new(rows: usize) -> Self {
        Self {
            enabled: phases_enabled(),
            rows,
            last: Instant::now(),
            totals: [0.0; PHASES],
        }
    }
    /// Charge the time since the previous mark to the phase that just ended.
    #[inline]
    fn mark(&mut self, ended: usize) {
        if self.enabled {
            let now = Instant::now();
            self.totals[ended] += (now - self.last).as_secs_f64() * 1000.0;
            self.last = now;
        }
    }
}
impl Drop for PhaseClock {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        self.mark(PHASES - 1);
        if self.rows >= PARALLEL_ROWS {
            eprintln!(
                "prefill phases rows={}: {}",
                self.rows,
                format_phases(&self.totals, 1)
            );
        } else {
            DECODE_PHASES.with(|cell| {
                let mut state = cell.borrow_mut();
                for (total, value) in state.0.iter_mut().zip(&self.totals) {
                    *total += value;
                }
                state.1 += 1;
            });
        }
    }
}
fn format_phases(totals: &[f64; PHASES], steps: usize) -> String {
    let per = steps.max(1) as f64;
    PHASE_NAMES
        .iter()
        .zip(totals)
        .map(|(name, ms)| format!("{name}={:.3}ms", ms / per))
        .collect::<Vec<_>>()
        .join(" ")
}
/// Print and reset this thread's accumulated decode-step phases (per step).
pub(crate) fn report_decode_phases() {
    if !phases_enabled() {
        return;
    }
    DECODE_PHASES.with(|cell| {
        let mut state = cell.borrow_mut();
        if state.1 > 0 {
            eprintln!(
                "decode phases per step over {} steps: {}",
                state.1,
                format_phases(&state.0, state.1)
            );
        }
        *state = ([0.0; PHASES], 0);
    });
}

/// Row count from which per-row elementwise prefill work uses the pool.
/// Decode (one to eight rows) stays serial; arithmetic is identical either way.
const PARALLEL_ROWS: usize = 64;

/// FP32 bits rounded to the nearest BF16 value (ties to even), as FP32 bits.
/// NaN and infinity are returned unchanged.
fn round_bf16_bits(bits: u32) -> u32 {
    if (bits & 0x7F80_0000) == 0x7F80_0000 {
        return bits;
    }
    (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) & 0xFFFF_0000
}

/// `*` in `pattern` matches any run of characters; everything else literally.
fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !name.starts_with(first) || !name[first.len()..].ends_with(last) {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod bf16_round_tests {
    #[test]
    fn rounds_like_bf16_conversion() {
        use super::round_bf16_bits;
        for x in [
            1.0f32,
            -2.5,
            1.0e-3,
            3.14159,
            65504.0,
            -1.0e-30,
            0.0,
            -0.0,
            7.1234567e5,
        ] {
            let expected = half::bf16::from_f32(x).to_f32();
            assert_eq!(
                f32::from_bits(round_bf16_bits(x.to_bits())),
                expected,
                "{x}"
            );
        }
        // Ties go to even: 1 + 2^-8 lies halfway between 1 and 1 + 2^-7.
        let tie = f32::from_bits(0x3F80_8000);
        assert_eq!(f32::from_bits(round_bf16_bits(tie.to_bits())), 1.0);
        assert!(f32::from_bits(round_bf16_bits(f32::NAN.to_bits())).is_nan());
    }
}

#[cfg(test)]
mod glob_tests {
    #[test]
    fn glob_matches_names() {
        use super::glob_match;
        let name = "layers.3.feed_forward.w2.weight";
        assert!(glob_match(name, name));
        assert!(glob_match("layers.3.*", name));
        assert!(glob_match("*.w2.weight", name));
        assert!(glob_match("layers.*.feed_forward.*", name));
        assert!(!glob_match("layers.30.*", name));
        assert!(!glob_match("*.w13.weight", name));
        assert!(!glob_match("layers.3", name));
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
