//! Checkpoint loading: the pinned reference loader and the labelled attempt
//! (quantized) loaders.
use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    sync::{Arc, OnceLock},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::config::{CONFIG_SHA256, ModelConfig, WEIGHTS_SHA256};

use super::{Layer, Model, Store, Weight, rope::temporal_factors};

impl Model {
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
            let _: &[f32] =
                bytemuck::try_cast_slice(tensor.data()).map_err(|e| anyhow::anyhow!("unaligned tensor {name}: {e}"))?;
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
            let (actual, w) = weights.remove(name).with_context(|| format!("missing tensor {name}"))?;
            ensure!(actual == shape, "{name}: shape {actual:?}, expected {shape:?}");
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
                wo: take(&format!("layers.{i}.attention.wo.weight"), &[c.dim, c.query_dim()])?,
                w13: take(&format!("layers.{i}.feed_forward.w13.weight"), &[2 * c.ffn_dim, c.dim])?,
                w2: take(&format!("layers.{i}.feed_forward.w2.weight"), &[c.dim, c.ffn_dim])?,
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
            map: Store::Mapped(Arc::new(map)),
            embedding,
            projector,
            norm,
            output,
            golden,
            temporal,
            layers,
            packed: OnceLock::new(),
            screened: OnceLock::new(),
            profile: crate::quant::Profile::REFERENCE,
            quantize_ms: 0.0,
            overlay_sha256: None,
            source: super::WeightsSource::Checkpoint {
                dir: dir.to_path_buf(),
                overlay: None,
                rtn: false,
            },
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
        let ranges = [&self.embedding, &self.projector, &self.norm, &self.output, &self.golden]
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

    pub fn load_profile(
        dir: impl AsRef<Path>,
        profile: crate::quant::Profile,
        artifact: Option<&Path>,
    ) -> Result<Self> {
        Self::load_profile_with(dir, profile, artifact, &[])
    }

    /// [`Model::load_profile`] that also keeps every body matrix whose name
    /// matches one of `keep_fp32` (`*` matches any run of characters) in
    /// FP32, whatever the overlay holds: mixed precision without writing a
    /// new overlay.
    pub fn load_profile_with(
        dir: impl AsRef<Path>,
        profile: crate::quant::Profile,
        artifact: Option<&Path>,
        keep_fp32: &[String],
    ) -> Result<Self> {
        Self::load_profile_bf16(dir, profile, artifact, keep_fp32, false)
    }

    /// [`Model::load_profile_with`], optionally on BF16-rounded weights
    /// ([`Model::round_weights_to_bf16`]), which unquantized matrices,
    /// embeddings, norms and the head then use.
    pub fn load_profile_bf16(
        dir: impl AsRef<Path>,
        profile: crate::quant::Profile,
        artifact: Option<&Path>,
        keep_fp32: &[String],
        weights_bf16: bool,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let mut model = Self::load(dir)?;
        if weights_bf16 {
            model.round_weights_to_bf16();
        }
        model.profile = profile;
        ensure!(
            artifact.is_none() || profile.quantizes_body(),
            "W8 artifact supplied to an FP32 profile"
        );
        model.source = super::WeightsSource::Checkpoint {
            dir: dir.to_path_buf(),
            overlay: artifact.map(Path::to_path_buf),
            rtn: artifact.is_none() && profile.quantizes_body() && profile.weight_bits() == 8,
        };
        if !profile.quantizes_body() {
            return Ok(model);
        }
        let started = Instant::now();
        ensure!(
            artifact.is_none() || profile.weight_bits() == 8,
            "W8 artifacts apply only to 8-bit profiles; 16-bit weights are quantized at load"
        );
        let bytes = artifact.map(std::fs::read).transpose()?;
        let tensors = bytes.as_ref().map(|b| SafeTensors::deserialize(b)).transpose()?;
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
            check("include_head", "false")?;
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
            model.overlay_sha256 = Some(format!("{:x}", Sha256::digest(bytes)));
            let expected = 8 * model.config.n_layers;
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
         -> Result<Option<Arc<crate::quant::linear::QuantLinear>>> {
            use crate::quant::linear::QuantLinear;
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
                    scales.dtype() == Dtype::F32 && scales.shape() == [output, input.div_ceil(group_size)],
                    "W8 scales dtype/shape for {name}"
                );
                let codes = codes.data().iter().map(|&x| x as i8).collect();
                let scales = scales
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                QuantLinear::from_parts(output, input, group_size, codes, scales)?
            } else {
                QuantLinear::quantize_bits(model.w(w), output, input, 64, profile.weight_bits())
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
        let mut prepared = prepared.into_iter();
        for l in &mut model.layers {
            l.qkv.quantized = prepared.next().flatten();
            l.wo.quantized = prepared.next().flatten();
            l.w13.quantized = prepared.next().flatten();
            l.w2.quantized = prepared.next().flatten();
        }
        model.quantize_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(model)
    }
    pub fn profile(&self) -> crate::quant::Profile {
        self.profile
    }
    /// Weight storage of this model (`falcon-ocr-eval` reports it).
    pub fn memory_report(&self) -> MemoryReport {
        let qbytes = |w: &Weight| w.quantized.as_ref().map_or(0, |q| q.payload_bytes());
        let effective = |w: &Weight| w.quantized.as_ref().map_or(w.range.len(), |q| q.payload_bytes());
        let body = |f: &dyn Fn(&Weight) -> usize| {
            self.layers
                .iter()
                .map(|l| f(&l.qkv) + f(&l.wo) + f(&l.w13) + f(&l.w2))
                .sum::<usize>()
        };
        MemoryReport {
            profile: self.profile,
            source_mapping_bytes: self.map.bytes().len(),
            quantized_weight_payload_bytes: body(&qbytes),
            logical_decode_weight_scan_bytes: body(&effective) + effective(&self.output),
            quantize_ms: self.quantize_ms,
            overlay_sha256: self.overlay_sha256.clone(),
        }
    }
}

/// Weight storage of a loaded model.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    pub profile: crate::quant::Profile,
    /// Bytes of the mapped (or owned) checkpoint or kernel-ready file.
    pub source_mapping_bytes: usize,
    /// Bytes of quantized body codes and scales.
    pub quantized_weight_payload_bytes: usize,
    /// Bytes one decode step scans through the body and head matrices.
    pub logical_decode_weight_scan_bytes: usize,
    /// Time quantizing at load or importing the overlay took.
    pub quantize_ms: f64,
    pub overlay_sha256: Option<String>,
}

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
        for x in [1.0f32, -2.5, 1.0e-3, 3.375, 65504.0, -1.0e-30, 0.0, -0.0, 7.1234e5] {
            let expected = half::bf16::from_f32(x).to_f32();
            assert_eq!(f32::from_bits(round_bf16_bits(x.to_bits())), expected, "{x}");
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
