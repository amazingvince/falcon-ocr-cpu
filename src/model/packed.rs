//! Kernel-ready model files: writing a quantized model with its screened head
//! and mapping one back in place.
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    ops::Range,
    path::Path,
    sync::{Arc, OnceLock},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::config::{ModelConfig, WEIGHTS_SHA256};

use super::{Layer, Model, Store, Weight, rope::temporal_factors};

/// Format tag of kernel-ready model files.
pub const PACKED_FORMAT: &str = "falcon-ocr-kernel-v1";
const BODY: [&str; 4] = ["attention.wqkv", "attention.wo", "feed_forward.w13", "feed_forward.w2"];

impl Model {
    /// Write a kernel-ready model file: every tensor this quantized model
    /// reads, in the layout its kernels use (FP32 embedding, head, norms,
    /// projector and sinks; 8/16-bit body codes with FP32 scales; the INT8
    /// head screen and its bound constants), so `load_packed` maps it with no
    /// conversion. Needs a quantized body and a prepared screened head.
    pub fn write_packed(&self, path: impl AsRef<Path>) -> Result<()> {
        let profile = self.attempt_profile;
        ensure!(
            profile.quantizes_body() && !profile.quantizes_head(),
            "packing needs a quantized body and FP32 head"
        );
        let screen = self
            .screened
            .get()
            .context("prepare the screened head before packing")?;
        let c = &self.config;
        let mut tensors: Vec<(String, Dtype, Vec<usize>, &[u8])> = Vec::new();
        let fp32 = |w: &Weight| -> &[u8] { &self.map.bytes()[w.range.clone()] };
        tensors.push((
            "tok_embeddings.weight".into(),
            Dtype::F32,
            vec![c.vocab_size, c.dim],
            fp32(&self.embedding),
        ));
        tensors.push((
            "img_projector.weight".into(),
            Dtype::F32,
            vec![c.dim, c.patch_dim()],
            fp32(&self.projector),
        ));
        tensors.push(("norm.weight".into(), Dtype::F32, vec![c.dim], fp32(&self.norm)));
        tensors.push((
            "output.weight".into(),
            Dtype::F32,
            vec![c.vocab_size, c.dim],
            fp32(&self.output),
        ));
        tensors.push((
            "freqs_cis_golden".into(),
            Dtype::F32,
            vec![c.n_heads, c.head_dim / 4, 2],
            fp32(&self.golden),
        ));
        let mut bits = None;
        for (i, layer) in self.layers.iter().enumerate() {
            tensors.push((
                format!("layers.{i}.attention.sinks"),
                Dtype::F32,
                vec![c.n_heads],
                fp32(&layer.sinks),
            ));
            for (name, weight) in BODY.iter().zip([&layer.qkv, &layer.wo, &layer.w13, &layer.w2]) {
                let q = weight
                    .quantized
                    .as_ref()
                    .with_context(|| format!("layer {i} {name} is not quantized"))?;
                let (out, input) = q.dimensions();
                let (codes, scales) = q.raw_parts();
                ensure!(
                    *bits.get_or_insert(q.bits()) == q.bits() && q.group_size() == 64,
                    "mixed packing formats"
                );
                let dtype = if q.bits() == 8 { Dtype::I8 } else { Dtype::I16 };
                tensors.push((format!("layers.{i}.{name}.__codes"), dtype, vec![out, input], codes));
                tensors.push((
                    format!("layers.{i}.{name}.__scales"),
                    Dtype::F32,
                    vec![out, input / 64],
                    scales,
                ));
            }
        }
        let (screen_codes, screen_scales, weight_abs_max, kappa) = screen.raw_parts();
        tensors.push((
            "output.__screen_codes".into(),
            Dtype::I8,
            vec![c.vocab_size, c.dim],
            screen_codes,
        ));
        tensors.push((
            "output.__screen_scales".into(),
            Dtype::F32,
            vec![c.vocab_size, c.dim / 64],
            screen_scales,
        ));
        let mut metadata = HashMap::new();
        metadata.insert("format".to_owned(), PACKED_FORMAT.to_owned());
        metadata.insert("profile".to_owned(), profile.label().to_owned());
        metadata.insert("weight_bits".to_owned(), bits.unwrap_or(8).to_string());
        metadata.insert("group_size".to_owned(), "64".to_owned());
        metadata.insert("source_sha256".to_owned(), self.weights_sha256.clone());
        metadata.insert("model_revision".to_owned(), crate::config::MODEL_REVISION.to_owned());
        metadata.insert("config".to_owned(), serde_json::to_string(&self.config)?);
        metadata.insert(
            "screen_weight_abs_max".to_owned(),
            format!("{:08x}", weight_abs_max.to_bits()),
        );
        metadata.insert("screen_kappa".to_owned(), format!("{:08x}", kappa.to_bits()));
        if let Some(sha) = &self.attempt_artifact_sha256 {
            metadata.insert("w8_artifact_sha256".to_owned(), sha.clone());
        }
        metadata.insert(
            "tensors_sha256".to_owned(),
            packed_digest(tensors.iter().map(|(n, _, _, d)| (n.as_str(), *d))),
        );
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                Ok((
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        safetensors::serialize_to_file(views, Some(metadata), path.as_ref())?;
        Ok(())
    }

    /// The profile a kernel-ready model file holds, from its header alone
    /// (the tensors are not read).
    pub fn packed_profile(path: impl AsRef<Path>) -> Result<crate::attempt::Profile> {
        let path = path.as_ref();
        let meta = packed_metadata(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            meta.get("format").map(String::as_str) == Some(PACKED_FORMAT),
            "{} is not a {PACKED_FORMAT} file",
            path.display()
        );
        let profile = meta.get("profile").context("packed metadata profile missing")?;
        <crate::attempt::Profile as clap::ValueEnum>::from_str(profile, false)
            .map_err(|e| anyhow::anyhow!("packed profile: {e}"))
    }

    /// Map a kernel-ready model file written by [`Model::write_packed`] and
    /// use every tensor in place: no conversion, and no hash of the whole
    /// file unless `verify` (a digest of every tensor against the header).
    pub fn load_packed(path: impl AsRef<Path>, verify: bool) -> Result<Self> {
        let started = Instant::now();
        let file = File::open(path.as_ref()).with_context(|| format!("open {}", path.as_ref().display()))?;
        // SAFETY: read-only mapping shared by every tensor view; the file must
        // stay unchanged while the model is loaded (as for `Model::load`).
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let tensors = SafeTensors::deserialize(&map)?;
        let (_, header) = SafeTensors::read_metadata(&map)?;
        let meta = header.metadata().as_ref().context("packed model metadata missing")?;
        let get = |key: &str| {
            meta.get(key)
                .map(String::as_str)
                .with_context(|| format!("packed metadata {key} missing"))
        };
        ensure!(get("format")? == PACKED_FORMAT, "not a {PACKED_FORMAT} file");
        ensure!(
            get("source_sha256")? == WEIGHTS_SHA256,
            "packed file comes from another checkpoint"
        );
        ensure!(
            get("model_revision")? == crate::config::MODEL_REVISION,
            "packed file comes from another revision"
        );
        let config: ModelConfig = serde_json::from_str(get("config")?)?;
        config.validate()?;
        let profile = <crate::attempt::Profile as clap::ValueEnum>::from_str(get("profile")?, false)
            .map_err(|e| anyhow::anyhow!("packed profile: {e}"))?;
        let bits: u32 = get("weight_bits")?.parse()?;
        ensure!(
            bits == profile.weight_bits() && get("group_size")? == "64",
            "packed weight format"
        );
        let constant = |key: &str| -> Result<f32> { Ok(f32::from_bits(u32::from_str_radix(get(key)?, 16)?)) };
        let base = map.as_ptr() as usize;
        let find = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<Range<usize>> {
            let t = tensors
                .tensor(name)
                .with_context(|| format!("packed tensor {name} missing"))?;
            ensure!(
                t.dtype() == dtype && t.shape() == shape,
                "packed tensor {name}: {:?} {:?}",
                t.dtype(),
                t.shape()
            );
            let start = t.data().as_ptr() as usize - base;
            Ok(start..start + t.data().len())
        };
        let fp32 = |name: &str, shape: &[usize]| -> Result<Weight> {
            let range = find(name, Dtype::F32, shape)?;
            crate::buf::Buf::<f32>::mapped(&map, range.clone(), shape.iter().product())?;
            Ok(Weight { range, quantized: None })
        };
        let c = &config;
        let embedding = fp32("tok_embeddings.weight", &[c.vocab_size, c.dim])?;
        let projector = fp32("img_projector.weight", &[c.dim, c.patch_dim()])?;
        let norm = fp32("norm.weight", &[c.dim])?;
        let output = fp32("output.weight", &[c.vocab_size, c.dim])?;
        let golden = fp32("freqs_cis_golden", &[c.n_heads, c.head_dim / 4, 2])?;
        let code_dtype = if bits == 8 { Dtype::I8 } else { Dtype::I16 };
        let shapes = [
            (c.query_dim() + 2 * c.kv_dim(), c.dim),
            (c.dim, c.query_dim()),
            (2 * c.ffn_dim, c.dim),
            (c.dim, c.ffn_dim),
        ];
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let mut body = Vec::with_capacity(4);
            for (name, &(out, input)) in BODY.iter().zip(&shapes) {
                let codes = find(&format!("layers.{i}.{name}.__codes"), code_dtype, &[out, input])?;
                let scales = find(&format!("layers.{i}.{name}.__scales"), Dtype::F32, &[out, input / 64])?;
                let q = crate::attempt::quant::Q8Linear::from_mapped(out, input, 64, bits, &map, codes, scales)?;
                // Quantized matrices are only read through their codes.
                body.push(Weight {
                    range: 0..0,
                    quantized: Some(Arc::new(q)),
                });
            }
            let mut body = body.into_iter();
            layers.push(Layer {
                qkv: body.next().unwrap(),
                wo: body.next().unwrap(),
                w13: body.next().unwrap(),
                w2: body.next().unwrap(),
                sinks: fp32(&format!("layers.{i}.attention.sinks"), &[c.n_heads])?,
            });
        }
        let screen = crate::head_screen::ScreenedHead::from_mapped(
            c.vocab_size,
            c.dim,
            &map,
            find("output.__screen_codes", Dtype::I8, &[c.vocab_size, c.dim])?,
            find("output.__screen_scales", Dtype::F32, &[c.vocab_size, c.dim / 64])?,
            constant("screen_weight_abs_max")?,
            constant("screen_kappa")?,
        )?;
        if verify {
            // `write_packed` hashes in its own tensor order; recompute in that order.
            let order = packed_order(&config);
            ensure!(
                tensors.names().len() == order.len(),
                "packed file has unexpected tensors"
            );
            let digest = packed_digest(
                order
                    .iter()
                    .map(|n| (n.as_str(), tensors.tensor(n).map(|t| t.data()).unwrap_or(&[]))),
            );
            ensure!(digest == get("tensors_sha256")?, "packed tensor digest mismatch");
        }
        let screened = OnceLock::new();
        let _ = screened.set(screen);
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
            screened,
            attempt_profile: profile,
            attempt_setup_ms: started.elapsed().as_secs_f64() * 1000.0,
            attempt_artifact_sha256: meta.get("w8_artifact_sha256").cloned(),
            source: super::WeightsSource::Packed {
                path: path.as_ref().to_path_buf(),
            },
            weights_sha256: WEIGHTS_SHA256.to_owned(),
        })
    }
}

/// The model config and overlay digest a kernel-ready file's header
/// records (the tensors are not read).
pub(crate) fn packed_facts(path: &Path) -> Result<(ModelConfig, Option<String>)> {
    let meta = packed_metadata(path).with_context(|| format!("read {}", path.display()))?;
    let config: ModelConfig = serde_json::from_str(meta.get("config").context("packed metadata config missing")?)?;
    config.validate()?;
    Ok((config, meta.get("w8_artifact_sha256").cloned()))
}

/// The `__metadata__` map of a safetensors file (the 8-byte length prefix
/// and the header JSON it announces), read without touching the tensors.
fn packed_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let mut file = File::open(path)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let length = usize::try_from(u64::from_le_bytes(prefix))?;
    ensure!(length <= 64 << 20, "safetensors header of {length} bytes");
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)?;
    let mut header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
    match header.remove("__metadata__") {
        Some(meta) => Ok(serde_json::from_value(meta)?),
        None => Ok(HashMap::new()),
    }
}

/// Tensor names in the order `write_packed` pushes (and hashes) them.
fn packed_order(c: &ModelConfig) -> Vec<String> {
    let mut names = vec![
        "tok_embeddings.weight".to_owned(),
        "img_projector.weight".to_owned(),
        "norm.weight".to_owned(),
        "output.weight".to_owned(),
        "freqs_cis_golden".to_owned(),
    ];
    for i in 0..c.n_layers {
        names.push(format!("layers.{i}.attention.sinks"));
        for name in BODY {
            names.push(format!("layers.{i}.{name}.__codes"));
            names.push(format!("layers.{i}.{name}.__scales"));
        }
    }
    names.push("output.__screen_codes".to_owned());
    names.push("output.__screen_scales".to_owned());
    names
}

/// SHA-256 over every tensor's name and bytes, in the given order.
fn packed_digest<'a>(tensors: impl Iterator<Item = (&'a str, &'a [u8])>) -> String {
    let mut hash = Sha256::new();
    for (name, data) in tensors {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((data.len() as u64).to_le_bytes());
        hash.update(data);
    }
    format!("{:x}", hash.finalize())
}
