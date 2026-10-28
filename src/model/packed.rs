//! Kernel-ready model files: writing a model with its screened head
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

use crate::{
    config::{ModelConfig, WEIGHTS_SHA256},
    quant::{Profile, Weights, linear::QuantLinear},
};

use super::{Layer, Model, Store, Weight, rope::temporal_factors};

/// Format tag of kernel-ready files with quantized bodies and no exception
/// columns.
pub const PACKED_FORMAT: &str = "falcon-ocr-kernel-v1";
/// Format tag of kernel-ready files whose body has exception columns (an
/// overlay of format `falcon-ocr-attempt3-w8g64-v2`): [`PACKED_FORMAT`] plus
/// `layers.{i}.{matrix}.__exc_cols` (I32 `[k]`) and `.__exc_vals` (F32
/// `[out, k]`), which the tensor digest takes after that matrix's scales
/// (the file stores tensors in the safetensors writer's own order). Binaries
/// that predate exception columns refuse it instead of computing with the
/// columns zeroed; files without exception columns keep [`PACKED_FORMAT`].
pub const PACKED_FORMAT_EXCEPTIONS: &str = "falcon-ocr-kernel-v2";
/// Format tag of kernel-ready files with unquantized FP32 body matrices.
pub const PACKED_FORMAT_FP32: &str = "falcon-ocr-kernel-fp32-v1";
/// The body matrices of a layer, in writer and digest order.
pub(super) const BODY: [&str; 4] = ["attention.wqkv", "attention.wo", "feed_forward.w13", "feed_forward.w2"];
/// Tensor suffixes of a body matrix's exception columns and values.
const EXCEPTION_SUFFIXES: [&str; 2] = [".__exc_cols", ".__exc_vals"];

impl Model {
    /// Write a kernel-ready model file: every tensor this model
    /// reads, in the layout its kernels use (FP32 embedding, head, norms,
    /// projector and sinks; FP32 body matrices or 8/16-bit codes with FP32
    /// scales and any exception columns; the INT8 head screen and its bound
    /// constants), so `load_packed` maps it with no conversion. Needs a prepared
    /// screened head. FP32 bodies use [`PACKED_FORMAT_FP32`]; quantized bodies
    /// use [`PACKED_FORMAT_EXCEPTIONS`] with exception columns or
    /// [`PACKED_FORMAT`] without them.
    pub fn write_packed(&self, path: impl AsRef<Path>) -> Result<()> {
        let profile = self.profile;
        let format = if profile.weights == Weights::F32 {
            PackedFormat::Fp32
        } else {
            PackedFormat::Quantized {
                exceptions: self.exception_bytes() > 0,
            }
        };
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
        let shapes = body_shapes(c);
        for (i, layer) in self.layers.iter().enumerate() {
            tensors.push((
                format!("layers.{i}.attention.sinks"),
                Dtype::F32,
                vec![c.n_heads],
                fp32(&layer.sinks),
            ));
            for ((name, weight), &(out, input)) in BODY
                .iter()
                .zip([&layer.qkv, &layer.wo, &layer.w13, &layer.w2])
                .zip(&shapes)
            {
                if format == PackedFormat::Fp32 {
                    tensors.push((
                        format!("layers.{i}.{name}.weight"),
                        Dtype::F32,
                        vec![out, input],
                        fp32(weight),
                    ));
                    continue;
                }
                let q = weight
                    .quantized
                    .as_ref()
                    .with_context(|| format!("layer {i} {name} is not quantized"))?;
                ensure!(
                    *bits.get_or_insert(q.bits()) == q.bits() && q.group_size() == 64,
                    "mixed packing formats"
                );
                tensors.extend(matrix_tensors(&format!("layers.{i}.{name}"), q));
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
        metadata.insert("format".to_owned(), format.tag().to_owned());
        metadata.insert("profile".to_owned(), profile.label().to_owned());
        let weight_bits = if format == PackedFormat::Fp32 {
            32
        } else {
            bits.unwrap_or(8)
        };
        metadata.insert("weight_bits".to_owned(), weight_bits.to_string());
        if matches!(format, PackedFormat::Quantized { .. }) {
            metadata.insert("group_size".to_owned(), "64".to_owned());
        }
        metadata.insert("source_sha256".to_owned(), self.weights_sha256.clone());
        metadata.insert("model_revision".to_owned(), crate::config::MODEL_REVISION.to_owned());
        metadata.insert("config".to_owned(), serde_json::to_string(&self.config)?);
        metadata.insert(
            "screen_weight_abs_max".to_owned(),
            format!("{:08x}", weight_abs_max.to_bits()),
        );
        metadata.insert("screen_kappa".to_owned(), format!("{:08x}", kappa.to_bits()));
        if let Some(sha) = &self.overlay_sha256 {
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
    pub fn packed_profile(path: impl AsRef<Path>) -> Result<crate::quant::Profile> {
        let path = path.as_ref();
        let meta = packed_metadata(path).with_context(|| format!("read {}", path.display()))?;
        let format = packed_format(&meta).with_context(|| format!("{} is not a kernel-ready file", path.display()))?;
        packed_profile(&meta, format)
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
        let format = packed_format(meta).context("not a kernel-ready file")?;
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
        let profile = packed_profile(meta, format)?;
        validate_weight_format(meta, format, profile)?;
        let bits = profile.weight_bits();
        let constant = |key: &str| -> Result<f32> { Ok(f32::from_bits(u32::from_str_radix(get(key)?, 16)?)) };
        let find = |name: &str, dtype: Dtype, shape: &[usize]| tensor_range(&tensors, &map, name, dtype, shape);
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
        let shapes = body_shapes(c);
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let mut body = Vec::with_capacity(4);
            for (name, &(out, input)) in BODY.iter().zip(&shapes) {
                let prefix = format!("layers.{i}.{name}");
                let PackedFormat::Quantized { exceptions } = format else {
                    body.push(fp32(&format!("{prefix}.weight"), &[out, input])?);
                    continue;
                };
                let q = map_matrix(&tensors, &map, &prefix, (out, input), bits, exceptions)?;
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
            // Recompute in the tensor order `write_packed` hashes.
            let names = tensors.names();
            let order = packed_order(&config, format, |matrix| {
                names
                    .iter()
                    .any(|n| n.strip_suffix(EXCEPTION_SUFFIXES[0]) == Some(matrix))
            });
            ensure!(names.len() == order.len(), "packed file has unexpected tensors");
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
            profile,
            quantize_ms: started.elapsed().as_secs_f64() * 1000.0,
            overlay_sha256: meta.get("w8_artifact_sha256").cloned(),
            source: super::WeightsSource::Packed {
                path: path.as_ref().to_path_buf(),
            },
            weights_sha256: WEIGHTS_SHA256.to_owned(),
        })
    }
}

/// Body tensor layout identified by a kernel-ready file's format tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PackedFormat {
    Fp32,
    Quantized { exceptions: bool },
}

impl PackedFormat {
    /// The public format string for this body layout.
    fn tag(self) -> &'static str {
        match self {
            Self::Fp32 => PACKED_FORMAT_FP32,
            Self::Quantized { exceptions: false } => PACKED_FORMAT,
            Self::Quantized { exceptions: true } => PACKED_FORMAT_EXCEPTIONS,
        }
    }
}

/// The body tensor layout a kernel-ready file's metadata identifies.
fn packed_format(meta: &HashMap<String, String>) -> Option<PackedFormat> {
    match meta.get("format").map(String::as_str) {
        Some(PACKED_FORMAT_FP32) => Some(PackedFormat::Fp32),
        Some(PACKED_FORMAT) => Some(PackedFormat::Quantized { exceptions: false }),
        Some(PACKED_FORMAT_EXCEPTIONS) => Some(PackedFormat::Quantized { exceptions: true }),
        _ => None,
    }
}

/// Parse the stored profile, requiring FP32 weights for the FP32 format.
fn packed_profile(meta: &HashMap<String, String>, format: PackedFormat) -> Result<Profile> {
    let label = meta.get("profile").context("packed metadata profile missing")?;
    let profile =
        <Profile as clap::ValueEnum>::from_str(label, false).map_err(|e| anyhow::anyhow!("packed profile: {e}"))?;
    ensure!(
        format != PackedFormat::Fp32 || profile.weights == Weights::F32,
        "packed weight format"
    );
    Ok(profile)
}

/// Check the profile's weight bits and quantization group metadata.
fn validate_weight_format(meta: &HashMap<String, String>, format: PackedFormat, profile: Profile) -> Result<()> {
    let bits: u32 = meta
        .get("weight_bits")
        .context("packed metadata weight_bits missing")?
        .parse()?;
    let matches_format = match format {
        PackedFormat::Fp32 => profile.weights == Weights::F32 && !meta.contains_key("group_size"),
        PackedFormat::Quantized { .. } => {
            profile.quantizes_body() && meta.get("group_size").map(String::as_str) == Some("64")
        }
    };
    ensure!(bits == profile.weight_bits() && matches_format, "packed weight format");
    Ok(())
}

/// Output and input dimensions of each matrix in [`BODY`].
fn body_shapes(c: &ModelConfig) -> [(usize, usize); 4] {
    [
        (c.query_dim() + 2 * c.kv_dim(), c.dim),
        (c.dim, c.query_dim()),
        (2 * c.ffn_dim, c.dim),
        (c.dim, c.ffn_dim),
    ]
}

/// Byte range of tensor `name` in `map`, which must have `dtype` and `shape`.
fn tensor_range(
    tensors: &SafeTensors<'_>,
    map: &Mmap,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<Range<usize>> {
    let t = tensors
        .tensor(name)
        .with_context(|| format!("packed tensor {name} missing"))?;
    ensure!(
        t.dtype() == dtype && t.shape() == shape,
        "packed tensor {name}: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let start = t.data().as_ptr() as usize - map.as_ptr() as usize;
    Ok(start..start + t.data().len())
}

/// The tensors of body matrix `prefix` (`layers.{i}.{matrix}`), in the order
/// `write_packed` writes and hashes them: codes, scales, then any exception
/// columns and values.
fn matrix_tensors<'a>(prefix: &str, q: &'a QuantLinear) -> Vec<(String, Dtype, Vec<usize>, &'a [u8])> {
    let (out, input) = q.dimensions();
    let (codes, scales) = q.raw_parts();
    let dtype = if q.bits() == 8 { Dtype::I8 } else { Dtype::I16 };
    let mut tensors = vec![
        (format!("{prefix}.__codes"), dtype, vec![out, input], codes),
        (format!("{prefix}.__scales"), Dtype::F32, vec![out, input / 64], scales),
    ];
    if let Some((columns, values)) = q.raw_exceptions() {
        let count = q.exception_columns().len();
        tensors.push((
            format!("{prefix}{}", EXCEPTION_SUFFIXES[0]),
            Dtype::I32,
            vec![count],
            columns,
        ));
        tensors.push((
            format!("{prefix}{}", EXCEPTION_SUFFIXES[1]),
            Dtype::F32,
            vec![out, count],
            values,
        ));
    }
    tensors
}

/// Map body matrix `prefix` (`[out, input]`, `bits`-bit codes, G64) of a
/// kernel-ready file in place, with its exception columns when the file has
/// any (`exceptions`: only format v2 may).
fn map_matrix(
    tensors: &SafeTensors<'_>,
    map: &Arc<Mmap>,
    prefix: &str,
    (out, input): (usize, usize),
    bits: u32,
    exceptions: bool,
) -> Result<QuantLinear> {
    let code_dtype = if bits == 8 { Dtype::I8 } else { Dtype::I16 };
    let codes = tensor_range(tensors, map, &format!("{prefix}.__codes"), code_dtype, &[out, input])?;
    let scales = tensor_range(
        tensors,
        map,
        &format!("{prefix}.__scales"),
        Dtype::F32,
        &[out, input / 64],
    )?;
    let q = QuantLinear::from_mapped(out, input, 64, bits, map, codes, scales)?;
    let [columns_name, values_name] = EXCEPTION_SUFFIXES.map(|suffix| format!("{prefix}{suffix}"));
    let Ok(columns) = tensors.tensor(&columns_name) else {
        ensure!(
            tensors.tensor(&values_name).is_err(),
            "packed tensor {values_name} without {columns_name}"
        );
        return Ok(q);
    };
    ensure!(
        exceptions,
        "packed tensor {columns_name} needs the format {PACKED_FORMAT_EXCEPTIONS}"
    );
    let count = columns.shape().first().copied().unwrap_or(0);
    let columns = tensor_range(tensors, map, &columns_name, Dtype::I32, &[count])?;
    let values = tensor_range(tensors, map, &values_name, Dtype::F32, &[out, count])?;
    q.with_mapped_exceptions(map, columns, values, count)
        .with_context(|| format!("packed exception columns of {prefix}"))
}

/// The model config, overlay digest and exception-column bytes a
/// kernel-ready file's header records (the tensors are not read).
pub(crate) fn packed_facts(path: &Path) -> Result<(ModelConfig, Option<String>, usize)> {
    let header = safetensors_header(path).with_context(|| format!("read {}", path.display()))?;
    let meta = metadata(&header)?;
    let config: ModelConfig = serde_json::from_str(meta.get("config").context("packed metadata config missing")?)?;
    config.validate()?;
    let exception_bytes = tensor_bytes_with_suffix(&header, &EXCEPTION_SUFFIXES);
    Ok((config, meta.get("w8_artifact_sha256").cloned(), exception_bytes))
}

/// The header JSON of a safetensors file (the 8-byte length prefix and the
/// JSON it announces), read without touching the tensors.
pub(super) fn safetensors_header(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut file = File::open(path)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let length = usize::try_from(u64::from_le_bytes(prefix))?;
    ensure!(length <= 64 << 20, "safetensors header of {length} bytes");
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// The `__metadata__` map of a safetensors header (empty without one).
fn metadata(header: &serde_json::Map<String, serde_json::Value>) -> Result<HashMap<String, String>> {
    match header.get("__metadata__") {
        Some(meta) => Ok(serde_json::from_value(meta.clone())?),
        None => Ok(HashMap::new()),
    }
}

/// The `__metadata__` map of a safetensors file, read without touching the
/// tensors.
fn packed_metadata(path: &Path) -> Result<HashMap<String, String>> {
    metadata(&safetensors_header(path)?)
}

/// Total data bytes of the tensors in a safetensors header whose names end
/// in one of `suffixes`.
pub(super) fn tensor_bytes_with_suffix(
    header: &serde_json::Map<String, serde_json::Value>,
    suffixes: &[&str],
) -> usize {
    header
        .iter()
        .filter(|(name, _)| suffixes.iter().any(|suffix| name.ends_with(suffix)))
        .filter_map(|(_, info)| {
            let offsets = info.get("data_offsets")?.as_array()?;
            let (start, end) = (offsets.first()?.as_u64()?, offsets.get(1)?.as_u64()?);
            usize::try_from(end.checked_sub(start)?).ok()
        })
        .fold(0, usize::saturating_add)
}

/// Tensor names in the order `write_packed` pushes (and hashes) them;
/// `exceptions(prefix)` says whether body matrix `layers.{i}.{matrix}` has
/// exception columns (unused for FP32 bodies).
fn packed_order(c: &ModelConfig, format: PackedFormat, exceptions: impl Fn(&str) -> bool) -> Vec<String> {
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
            let prefix = format!("layers.{i}.{name}");
            if format == PackedFormat::Fp32 {
                names.push(format!("{prefix}.weight"));
                continue;
            }
            names.push(format!("{prefix}.__codes"));
            names.push(format!("{prefix}.__scales"));
            if exceptions(&prefix) {
                names.extend(EXCEPTION_SUFFIXES.map(|suffix| format!("{prefix}{suffix}")));
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::Simd, quant::linear::Scratch};

    /// Body matrices written as `write_packed` writes them and mapped back
    /// as `load_packed` maps them (without a whole model): the same bytes
    /// and bitwise the same products, exception columns included.
    #[test]
    fn body_matrices_round_trip_with_and_without_exception_columns() {
        let (out, input) = (16, 128);
        let w: Vec<f32> = (0..out * input)
            .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 977.0)
            .collect();
        let plain = QuantLinear::quantize(&w, out, input, 64).unwrap();
        let columns = [5, 64, 127];
        let (mut zeroed, mut values) = (w.clone(), Vec::new());
        for row in 0..out {
            for &column in &columns {
                values.push(100.0 * w[row * input + column as usize]);
                zeroed[row * input + column as usize] = 0.0;
            }
        }
        let exceptions = QuantLinear::quantize(&zeroed, out, input, 64)
            .unwrap()
            .with_exceptions(columns.to_vec(), values)
            .unwrap();
        let (w2, wo) = ("layers.0.feed_forward.w2", "layers.0.attention.wo");
        let mut tensors = matrix_tensors(w2, &exceptions);
        let names: Vec<&str> = tensors.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(
            names,
            [
                "layers.0.feed_forward.w2.__codes",
                "layers.0.feed_forward.w2.__scales",
                "layers.0.feed_forward.w2.__exc_cols",
                "layers.0.feed_forward.w2.__exc_vals"
            ]
        );
        tensors.extend(matrix_tensors(wo, &plain));
        assert_eq!(tensors.len(), 6);
        let config = include_str!("../../tests/fixtures/model-config.json");
        let metadata = HashMap::from([("config".to_owned(), config.to_owned())]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("matrices.safetensors");
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, Some(metadata), &path).unwrap();
        let (_, overlay, exception_bytes) = packed_facts(&path).unwrap();
        assert_eq!((overlay, exception_bytes), (None, exceptions.exception_bytes()));
        let file = File::open(&path).unwrap();
        // SAFETY: the temporary file is not modified while it is mapped.
        let map = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let parsed = SafeTensors::deserialize(&map).unwrap();
        let mapped = map_matrix(&parsed, &map, w2, (out, input), 8, true).unwrap();
        let mapped_plain = map_matrix(&parsed, &map, wo, (out, input), 8, false).unwrap();
        // Format v1 refuses exception columns; shapes are checked.
        let error = map_matrix(&parsed, &map, w2, (out, input), 8, false).unwrap_err();
        assert!(error.to_string().contains(PACKED_FORMAT_EXCEPTIONS), "{error}");
        assert!(map_matrix(&parsed, &map, w2, (out, 2 * input), 8, true).is_err());
        for (original, mapped) in [(&exceptions, &mapped), (&plain, &mapped_plain)] {
            assert_eq!(original.raw_parts(), mapped.raw_parts());
            assert_eq!(original.raw_exceptions(), mapped.raw_exceptions());
            assert_eq!(original.payload_bytes(), mapped.payload_bytes());
            for rows in [1, 3, 9] {
                let x: Vec<f32> = (0..rows * input)
                    .map(|i| ((i * 31 % 151) as f32 - 75.0) / 97.0)
                    .collect();
                let (mut a, mut b) = (vec![0.0; rows * out], vec![f32::NAN; rows * out]);
                original
                    .linear(&x, rows, &mut a, &mut Scratch::default(), Simd::Auto)
                    .unwrap();
                mapped
                    .linear(&x, rows, &mut b, &mut Scratch::default(), Simd::Auto)
                    .unwrap();
                assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()), "rows {rows}");
            }
        }
        assert_eq!(mapped.exception_columns(), &columns);
        assert!(mapped_plain.exception_columns().is_empty());
    }

    /// Malformed exception tensors in a packed file are refused with an
    /// error (never mapped, never a panic), including values without columns.
    #[test]
    fn malformed_exception_columns_in_packed_files_are_refused() {
        let (out, input) = (16, 128);
        let w: Vec<f32> = (0..out * input)
            .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 977.0)
            .collect();
        let plain = QuantLinear::quantize(&w, out, input, 64).unwrap();
        let prefix = "layers.0.feed_forward.w2";
        let i32s = |v: &[i32]| v.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>();
        let f32s = |n: usize| (0..n).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect::<Vec<u8>>();
        let columns = |v: &[i32]| Some((Dtype::I32, vec![v.len()], i32s(v)));
        let values = |rows: usize, k: usize| Some((Dtype::F32, vec![rows, k], f32s(rows * k)));
        let i64s: Vec<u8> = [5i64, 64].iter().flat_map(|c| c.to_le_bytes()).collect();
        let cases = [
            ("valid", columns(&[5, 64]), values(out, 2), true),
            ("unsorted", columns(&[64, 5]), values(out, 2), false),
            ("duplicate", columns(&[5, 5]), values(out, 2), false),
            ("negative", columns(&[-1, 5]), values(out, 2), false),
            ("out of range", columns(&[5, 128]), values(out, 2), false),
            ("i32 max", columns(&[5, i32::MAX]), values(out, 2), false),
            ("empty", columns(&[]), values(out, 0), false),
            (
                "more than the inputs",
                columns(&(0..129).collect::<Vec<_>>()),
                values(out, 129),
                false,
            ),
            ("values k", columns(&[5, 64]), values(out, 3), false),
            ("values rows", columns(&[5, 64]), values(out + 1, 2), false),
            (
                "values 1-d",
                columns(&[5, 64]),
                Some((Dtype::F32, vec![out * 2], f32s(out * 2))),
                false,
            ),
            (
                "values F16",
                columns(&[5, 64]),
                Some((Dtype::F16, vec![out, 2], vec![0; out * 4])),
                false,
            ),
            ("values missing", columns(&[5, 64]), None, false),
            ("columns missing", None, values(out, 2), false),
            ("columns I64", Some((Dtype::I64, vec![2], i64s)), values(out, 2), false),
            (
                "columns 2-d",
                Some((Dtype::I32, vec![1, 2], i32s(&[5, 64]))),
                values(out, 2),
                false,
            ),
            (
                "columns 0-d",
                Some((Dtype::I32, vec![], i32s(&[5]))),
                values(out, 1),
                false,
            ),
        ];
        for (label, columns, values, ok) in cases {
            let mut tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = matrix_tensors(prefix, &plain)
                .into_iter()
                .map(|(name, dtype, shape, data)| (name, dtype, shape, data.to_vec()))
                .collect();
            for (suffix, tensor) in EXCEPTION_SUFFIXES.iter().zip([columns, values]) {
                if let Some((dtype, shape, data)) = tensor {
                    tensors.push((format!("{prefix}{suffix}"), dtype, shape, data));
                }
            }
            let views = tensors.iter().map(|(name, dtype, shape, data)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            });
            let bytes = safetensors::serialize(views, None).unwrap();
            let mut map = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
            map.copy_from_slice(&bytes);
            let map = Arc::new(map.make_read_only().unwrap());
            let parsed = SafeTensors::deserialize(&map).unwrap();
            let result = map_matrix(&parsed, &map, prefix, (out, input), 8, true);
            assert_eq!(result.is_ok(), ok, "{label}: {:?}", result.err());
        }
    }

    #[test]
    fn formats_and_the_digest_order_follow_the_exception_columns() {
        let format = |value: &str| packed_format(&HashMap::from([("format".to_owned(), value.to_owned())]));
        assert_eq!(
            format(PACKED_FORMAT),
            Some(PackedFormat::Quantized { exceptions: false })
        );
        assert_eq!(
            format(PACKED_FORMAT_EXCEPTIONS),
            Some(PackedFormat::Quantized { exceptions: true })
        );
        assert_eq!(format(PACKED_FORMAT_FP32), Some(PackedFormat::Fp32));
        assert_eq!(format("falcon-ocr-kernel-v3"), None);
        let c: ModelConfig = serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap();
        // Without exception columns: the v1 order, unchanged.
        let v1 = packed_order(&c, PackedFormat::Quantized { exceptions: false }, |_| false);
        assert_eq!(v1.len(), 5 + 22 * 9 + 2);
        assert_eq!(
            v1[5..8],
            [
                "layers.0.attention.sinks",
                "layers.0.attention.wqkv.__codes",
                "layers.0.attention.wqkv.__scales"
            ]
        );
        let v2 = packed_order(&c, PackedFormat::Quantized { exceptions: true }, |matrix| {
            matrix == "layers.3.feed_forward.w2"
        });
        assert_eq!(v2.len(), v1.len() + 2);
        let at = v2
            .iter()
            .position(|n| n == "layers.3.feed_forward.w2.__scales")
            .unwrap();
        assert_eq!(
            v2[at + 1..at + 4],
            [
                "layers.3.feed_forward.w2.__exc_cols",
                "layers.3.feed_forward.w2.__exc_vals",
                "layers.4.attention.sinks"
            ]
        );
        let fp32 = packed_order(&c, PackedFormat::Fp32, |_| panic!("FP32 has no exception columns"));
        assert_eq!(fp32.len(), 5 + 22 * 5 + 2);
        assert_eq!(
            fp32[5..11],
            [
                "layers.0.attention.sinks",
                "layers.0.attention.wqkv.weight",
                "layers.0.attention.wo.weight",
                "layers.0.feed_forward.w13.weight",
                "layers.0.feed_forward.w2.weight",
                "layers.1.attention.sinks"
            ]
        );
        assert_eq!(
            fp32[fp32.len() - 2..],
            ["output.__screen_codes", "output.__screen_scales"]
        );
    }

    /// Header-only inspection accepts FP32 storage with each supported cache
    /// profile; full loading refuses contradictory body metadata.
    #[test]
    fn fp32_header_profiles_require_fp32_body_metadata() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("header.safetensors");
        let config = include_str!("../../tests/fixtures/model-config.json");
        for profile in Profile::ALL {
            let meta = HashMap::from([
                ("format", PACKED_FORMAT_FP32.to_owned()),
                ("profile", profile.label().to_owned()),
                ("weight_bits", profile.weight_bits().to_string()),
                ("config", config.to_owned()),
            ]);
            let header = serde_json::to_vec(&serde_json::json!({"__metadata__": meta})).unwrap();
            let mut file = File::create(&path).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&header).unwrap();
            let parsed = Model::packed_profile(&path);
            assert_eq!(parsed.is_ok(), profile.weights == Weights::F32, "{profile}");
            if let Ok(parsed) = parsed {
                assert_eq!(parsed, profile);
                assert_eq!(packed_facts(&path).unwrap().2, 0);
            }
        }
        let meta = HashMap::from([
            ("profile".to_owned(), "reference".to_owned()),
            ("weight_bits".to_owned(), "32".to_owned()),
        ]);
        assert_eq!(packed_profile(&meta, PackedFormat::Fp32).unwrap(), Profile::REFERENCE);
        assert!(
            validate_weight_format(&meta, PackedFormat::Quantized { exceptions: false }, Profile::REFERENCE).is_err()
        );
        let mut wrong_bits = meta.clone();
        wrong_bits.insert("weight_bits".to_owned(), "16".to_owned());
        assert!(validate_weight_format(&wrong_bits, PackedFormat::Fp32, Profile::REFERENCE).is_err());
        let mut grouped = meta;
        grouped.insert("group_size".to_owned(), "64".to_owned());
        assert!(validate_weight_format(&grouped, PackedFormat::Fp32, Profile::REFERENCE).is_err());
        let minimal = HashMap::from([("profile".to_owned(), "w16-body-kv-q16".to_owned())]);
        assert_eq!(
            packed_profile(&minimal, PackedFormat::Quantized { exceptions: false }).unwrap(),
            Profile::W16_BODY_KV_Q16
        );
    }

    /// Raw FP32 body tensors map without changing any bits, and their name,
    /// payload and order all contribute to the packed tensor digest.
    #[test]
    fn fp32_body_tensor_mapping_preserves_bits_and_digest() {
        let name = "layers.0.attention.wo.weight";
        let data: Vec<u8> = [0, 0x8000_0000, 0x7fc0_0123, 0x7f80_0000, 0x3f80_0000, 0xbf80_0000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();
        let bytes = safetensors::serialize([(name, view)], None).unwrap();
        let mut map = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
        map.copy_from_slice(&bytes);
        let map = Arc::new(map.make_read_only().unwrap());
        let tensors = SafeTensors::deserialize(&map).unwrap();
        let range = tensor_range(&tensors, &map, name, Dtype::F32, &[2, 3]).unwrap();
        assert_eq!(&map[range.clone()], data);
        crate::buf::Buf::<f32>::mapped(&map, range.clone(), 6).unwrap();
        assert!(tensor_range(&tensors, &map, name, Dtype::I16, &[2, 3]).is_err());
        assert!(tensor_range(&tensors, &map, name, Dtype::F32, &[3, 2]).is_err());
        let digest = packed_digest([(name, data.as_slice())].into_iter());
        assert_eq!(digest, packed_digest([(name, &map[range])].into_iter()));
        assert_ne!(
            digest,
            packed_digest([("layers.0.attention.wo.__codes", data.as_slice())].into_iter())
        );
        let mut changed = data.clone();
        changed[0] ^= 1;
        assert_ne!(digest, packed_digest([(name, changed.as_slice())].into_iter()));
        assert_ne!(
            packed_digest([(name, data.as_slice()), ("norm.weight", changed.as_slice())].into_iter()),
            packed_digest([("norm.weight", changed.as_slice()), (name, data.as_slice())].into_iter())
        );
    }
}
