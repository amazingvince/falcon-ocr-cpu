use std::{
    ops::Range,
    sync::{Arc, OnceLock},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use rayon::prelude::*;

use crate::{
    config::{ModelConfig, WeightLayout},
    head_screen::Screened,
    kernels::{self, Bf16Kv},
    packed_kernels::PhasePackedLinear,
    trace::Trace,
};

mod cache;
mod fused;
mod load;
mod packed;
mod profile;
mod rope;

pub(crate) use cache::{BatchWorkspace, Session};
pub use load::MemoryReport;
pub use packed::PACKED_FORMAT;
pub(crate) use packed::packed_facts;

/// Where a model's weights came from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum WeightsSource {
    /// A kernel-ready file written by `pack`.
    Packed { path: std::path::PathBuf },
    /// The FP32 checkpoint directory; quantized profiles are built at load
    /// from the GPTQ `overlay`, or round-to-nearest (`rtn`).
    Checkpoint {
        dir: std::path::PathBuf,
        overlay: Option<std::path::PathBuf>,
        rtn: bool,
    },
}

impl WeightsSource {
    /// The folder whose `tokenizer.json` serves this model: a kernel-ready
    /// file's folder when it ships one, otherwise `model_dir`.
    pub fn tokenizer_dir(&self, model_dir: &std::path::Path) -> std::path::PathBuf {
        if let Self::Packed { path } = self
            && let Some(dir) = path.parent()
            && dir.join("tokenizer.json").is_file()
        {
            return dir.to_path_buf();
        }
        model_dir.to_path_buf()
    }
}
pub(crate) use profile::report_decode_phases;
pub(crate) use rope::{image_range, positions};

use fused::fused_prefix_rows;
use profile::{HeadClock, PhaseClock};
use rope::{add_residual, rotary_factors, row_scales};

#[derive(Clone)]
struct Weight {
    range: Range<usize>,
    quantized: Option<Arc<crate::quant::linear::QuantLinear>>,
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
    Mapped(Arc<Mmap>),
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
    profile: crate::quant::Profile,
    quantize_ms: f64,
    overlay_sha256: Option<String>,
    source: WeightsSource,
    pub(crate) weights_sha256: String,
}

impl Model {
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }
    /// Where the weights came from.
    pub fn source(&self) -> &WeightsSource {
        &self.source
    }
    /// SHA-256 of the W8 overlay the body was imported from, if any.
    pub fn overlay_sha256(&self) -> Option<&str> {
        self.overlay_sha256.as_deref()
    }
    /// Bit width of the quantized body when every body matrix of every layer
    /// is quantized to one width with panel-shaped output dims (the prefill
    /// GEMM's precondition); `None` for an FP32 or mixed body.
    pub fn body_bits(&self) -> Option<u32> {
        let mut bits = None;
        for layer in &self.layers {
            for w in [&layer.qkv, &layer.wo, &layer.w13, &layer.w2] {
                let q = w.quantized.as_ref().filter(|q| q.panel_shaped())?;
                match bits {
                    None => bits = Some(q.bits()),
                    Some(b) if b == q.bits() => {}
                    Some(_) => return None,
                }
            }
        }
        bits
    }
    /// Actual additional shared tensor payload, excluding allocator metadata.
    /// This remains resident if another Runner on this Model enabled packing.
    pub fn packed_weight_bytes(&self) -> usize {
        self.packed.get().map_or(0, |p| p.tensor_bytes)
    }
    /// Build and verify the INT8 screen of the FP32 vocabulary head once per
    /// model. Greedy selection through it is exact; see `head_screen`.
    pub fn prepare_screened_head(&self) -> Result<()> {
        if self.screened.get().is_none() {
            let c = &self.config;
            let head = crate::head_screen::ScreenedHead::build(self.w(&self.output), c.vocab_size, c.dim)?;
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
                    qkv: PhasePackedLinear::new(self.w(&layer.qkv), c.dim, c.query_dim() + 2 * c.kv_dim()),
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
        scratch: &mut crate::quant::linear::Scratch,
        simd: kernels::Simd,
    ) -> Result<()> {
        if let Some(q) = &w.quantized {
            ensure!(
                packed.is_none(),
                "phase-packed FP32 weights cannot override W8 numerical weights"
            );
            q.linear(input, rows, output, scratch, simd)
        } else {
            decode_linear(input, rows, input_dim, self.w(w), packed, output_dim, output, simd);
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
        scratch: &mut crate::quant::linear::Scratch,
        simd: kernels::Simd,
    ) -> Result<()> {
        let c = &self.config;
        if let Some(q) = &w13.quantized
            && q.linear_glu(input, rows, gated, scratch, simd)?
        {
            return Ok(());
        }
        let packed_rows = packed.is_some() && (2..=8).contains(&rows);
        if w13.quantized.is_none()
            && !packed_rows
            && kernels::linear_glu_with_simd(input, rows, c.dim, self.w(w13), c.ffn_dim, gated, simd)
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
                row.copy_from_slice(&embedding[*token as usize * c.dim..(*token as usize + 1) * c.dim]);
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
        // Decode steps only (prefill reports its own phases).
        let _head_clock = if rows == 1 {
            HeadClock::new(rows, session.tuning.phases)
        } else {
            None
        };
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
            session.len.checked_add(rows).is_some_and(|n| n <= session.capacity),
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
        let exp = session.exp;
        let tuning = session.tuning;
        let work = &mut session.workspace;
        work.resize(rows, c);
        if trace.enabled() {
            trace.tensor(&format!("{phase}.embedding"), &[rows, c.dim], h)?;
        }
        let mut clock = PhaseClock::new(rows, tuning.phases);
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
        // Quantized prefill: the RMS norms fold into the GEMM's row packing and
        // the residual adds into its epilogue, with the same per-element
        // operations as the separate passes.
        // `kernels::prefill_plan` is the one place that decides which prefill
        // kernels a body and CPU get (`auto::Resolved` reports the same plan).
        let plan = kernels::prefill_plan(self.body_bits(), simd, tuning.prefill_bf16);
        let panel = rows > 8 && !capture && !trace.enabled() && plan.projection != kernels::PrefillProjection::GemmF32;
        fn quantized(w: &Weight) -> &crate::quant::linear::QuantLinear {
            w.quantized.as_deref().expect("body_bits checked every body matrix")
        }
        let bf16_attention = panel && plan.attention == kernels::PrefillAttention::Bf16;
        let bf16_projections = panel && plan.projection == kernels::PrefillProjection::PanelBf16;
        let mut stored_bf16;
        for (i, layer) in self.layers.iter().enumerate() {
            if panel {
                row_scales(h, c.dim, &mut work.row_scale);
                quantized(&layer.qkv).prefill(
                    h,
                    rows,
                    Some(&work.row_scale),
                    kernels::panel_gemm::Epilogue::Store(&mut work.qkv),
                    &mut work.quant_scratch,
                    bf16_projections,
                );
            } else {
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
            }
            clock.mark(1);
            // Prefill rows into a compact cache: split, per-head norms, RoPE and
            // the cache append in one pass per row, each element's arithmetic
            // unchanged (`fused_prefix_rows`).
            let fused = rows >= PARALLEL_ROWS
                && !trace.enabled()
                && offset + rows <= session.layers[i].compact_prefix_len().unwrap_or(0);
            // A whole prefix in one call also gets its BF16 attention copies
            // from the same pass.
            stored_bf16 = fused
                && bf16_attention
                && offset == 0
                && session.layers[i].compact_prefix_len() == Some(rows)
                && kernels::prefill_bf16_rows_available();
            if fused {
                let bf16 = stored_bf16.then_some((&mut work.bf16_keys, &mut work.bf16_values));
                fused_prefix_rows(
                    c,
                    rows,
                    &work.qkv,
                    &work.rope,
                    &mut work.q,
                    &mut session.layers[i],
                    bf16,
                    simd,
                );
                clock.mark(2);
            } else {
                // Normalize each original K head before GQA expansion. Spatial rotations
                // subsequently differ for paired heads, so expanded keys are intentional.
                let split = |qkv: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]| {
                    for head in 0..c.n_heads {
                        let dst = head * c.head_dim;
                        q[dst..dst + c.head_dim].copy_from_slice(&qkv[head * c.head_dim..(head + 1) * c.head_dim]);
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
                    trace.tensor(&format!("{phase}.layer.{i}.q"), &[rows, c.n_heads, c.head_dim], &work.q)?;
                    trace.tensor(&format!("{phase}.layer.{i}.k"), &[rows, c.n_heads, c.head_dim], &work.k)?;
                    trace.tensor(&format!("{phase}.layer.{i}.v"), &[rows, c.n_heads, c.head_dim], &work.v)?;
                }
                clock.mark(2);
                session.layers[i].append(&work.k, &work.v, offset, c);
            }
            clock.mark(3);
            let cache = &mut session.layers[i];
            let bf16_kv = if !bf16_attention {
                None
            } else if stored_bf16 {
                Some(Bf16Kv::Converted(&work.bf16_keys, &work.bf16_values))
            } else {
                Some(Bf16Kv::Convert(&mut work.bf16_keys, &mut work.bf16_values))
            };
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
                bf16_kv,
                exp,
                tuning.prefill_profile,
            );
            clock.mark(4);
            if trace.enabled() {
                trace.tensor(&format!("{phase}.layer.{i}.attention"), &[rows, qdim], &work.attn)?;
            }
            if panel {
                quantized(&layer.wo).prefill(
                    &work.attn,
                    rows,
                    None,
                    kernels::panel_gemm::Epilogue::Add(h),
                    &mut work.quant_scratch,
                    bf16_projections,
                );
                clock.mark(5);
                row_scales(h, c.dim, &mut work.row_scale);
                quantized(&layer.w13).prefill(
                    h,
                    rows,
                    Some(&work.row_scale),
                    kernels::panel_gemm::Epilogue::Glu(&mut work.gated),
                    &mut work.quant_scratch,
                    bf16_projections,
                );
                clock.mark(6);
                quantized(&layer.w2).prefill(
                    &work.gated,
                    rows,
                    None,
                    kernels::panel_gemm::Epilogue::Add(h),
                    &mut work.quant_scratch,
                    bf16_projections,
                );
                clock.mark(7);
            } else {
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
            }
            if trace.enabled() {
                trace.tensor(&format!("{phase}.layer.{i}.hidden"), &[rows, c.dim], h)?;
            }
        }
        session.len += rows;
        session.next_position = positions[rows - 1] + 1;
        if rows > 8 {
            kernels::report_prefill_stage_cycles(tuning.prefill_profile);
        }
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
        let _head_clock = HeadClock::new(rows, session.tuning.phases);
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
            head.select_rows(norm, rows, self.w(&self.output), dot, &mut work.head_rows, &mut results);
            if results[..rows].iter().all(|r| matches!(r, Screened::Token { .. })) {
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
        ensure!(rows > 0 && active.len() == rows, "invalid batch decode dimensions");
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
            ensure!(session.next_position < c.max_seq_len, "RoPE position exceeds context");
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
                    work.q[dst..dst + c.head_dim].copy_from_slice(&qkv[head * c.head_dim..(head + 1) * c.head_dim]);
                    let kvhead = head / (c.n_heads / c.n_kv_heads);
                    let kstart = qdim + kvhead * c.head_dim;
                    work.k[dst..dst + c.head_dim].copy_from_slice(&qkv[kstart..kstart + c.head_dim]);
                    let vstart = qdim + kdim + kvhead * c.head_dim;
                    work.v[dst..dst + c.head_dim].copy_from_slice(&qkv[vstart..vstart + c.head_dim]);
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
                        let [cos, sin] = work.rope[(row * c.n_heads + head) * (c.head_dim / 2) + pair];
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
                trace.tensor(&format!("{phase}.layer.{i}.q"), &[rows, c.n_heads, c.head_dim], &work.q)?;
                trace.tensor(&format!("{phase}.layer.{i}.k"), &[rows, c.n_heads, c.head_dim], &work.k)?;
                trace.tensor(&format!("{phase}.layer.{i}.v"), &[rows, c.n_heads, c.head_dim], &work.v)?;
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
                    None,
                    session.exp,
                    false,
                );
            }
            if trace.enabled() {
                trace.tensor(&format!("{phase}.layer.{i}.attention"), &[rows, qdim], &work.attn)?;
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
        kernels::rms_norm(h, &mut work.normalized, c.dim, c.norm_eps, Some(self.w(&self.norm)));
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
            trace.tensor(&format!("{phase}.logits"), &[rows, c.vocab_size], &work.logits)?;
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

/// Row count from which per-row elementwise prefill work uses the pool.
/// Decode (one to eight rows) stays serial; arithmetic is identical either way.
const PARALLEL_ROWS: usize = 64;
