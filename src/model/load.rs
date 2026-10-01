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

/// Format of W8 overlays (`tools/convert_w8.py`, `tools/w8_variants.py`):
/// per body matrix `{name}.__w8_codes` (I8 `[out, in]`) and
/// `{name}.__w8_scales` (F32 `[out, in / group]`).
const W8_FORMAT: &str = "falcon-ocr-attempt3-w8g64-v1";
/// [`W8_FORMAT`] plus exception columns for any matrix:
/// `{name}.__w8_exc_cols` (I32 `[k]`, sorted, unique) and
/// `{name}.__w8_exc_vals` (F32 `[out, k]`, the unquantized weights), the codes at
/// those columns zero. Its own format string makes binaries that predate
/// exception columns refuse such overlays instead of computing with the
/// columns zeroed.
const W8_FORMAT_EXCEPTIONS: &str = "falcon-ocr-attempt3-w8g64-v2";
/// Overlay tensor suffixes: codes, scales, exception columns and values.
const W8_SUFFIXES: [&str; 4] = [".__w8_codes", ".__w8_scales", ".__w8_exc_cols", ".__w8_exc_vals"];

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
            let format = meta.get("format").and_then(|v| v.as_str());
            ensure!(
                format == Some(W8_FORMAT) || format == Some(W8_FORMAT_EXCEPTIONS),
                "W8 metadata format must be {W8_FORMAT} or {W8_FORMAT_EXCEPTIONS}"
            );
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
            let names = tensors.as_ref().unwrap().names();
            check_overlay_tensors(
                &names,
                &body_matrix_names(model.config.n_layers),
                partial,
                format == Some(W8_FORMAT_EXCEPTIONS),
            )?;
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
                let Some(q) = overlay_matrix(tensors, name, (output, input), group_size, partial)? else {
                    return Ok(None);
                };
                q
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
    /// This model with the KV half of its profile replaced by `kv`. The
    /// weights do not depend on it: a runner builds the cache per page and
    /// seals it into the profile's storage (`auto::ModelRequest::kv_cache`).
    pub fn with_kv_cache(mut self, kv: crate::quant::Kv) -> Self {
        self.profile.kv = kv;
        self
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
    /// Bytes of quantized body codes, scales and exception columns.
    pub quantized_weight_payload_bytes: usize,
    /// Bytes one decode step scans through the body and head matrices.
    pub logical_decode_weight_scan_bytes: usize,
    /// Time quantizing at load or importing the overlay took.
    pub quantize_ms: f64,
    pub overlay_sha256: Option<String>,
}

/// Names of the body matrices a W8 overlay may hold, layer by layer.
fn body_matrix_names(layers: usize) -> Vec<String> {
    (0..layers)
        .flat_map(|i| super::packed::BODY.map(|name| format!("layers.{i}.{name}.weight")))
        .collect()
}

/// Checks that a W8 overlay holds nothing but complete tensor sets of known
/// body matrices: codes with scales; exception columns with their values,
/// only beside codes and only when `exceptions` (format v2) allows them; and
/// every matrix unless the overlay is `partial`.
fn check_overlay_tensors(names: &[&str], matrices: &[String], partial: bool, exceptions: bool) -> Result<()> {
    let mut present: HashMap<&str, [bool; 4]> = matrices.iter().map(|m| (m.as_str(), [false; 4])).collect();
    for name in names {
        let (matrix, slot) = W8_SUFFIXES
            .iter()
            .enumerate()
            .find_map(|(slot, suffix)| Some((name.strip_suffix(suffix)?, slot)))
            .with_context(|| format!("W8 artifact has an unexpected tensor {name}"))?;
        let found = present
            .get_mut(matrix)
            .with_context(|| format!("W8 artifact has an unexpected tensor {name}"))?;
        ensure!(
            slot < 2 || exceptions,
            "W8 exception columns ({name}) need the format {W8_FORMAT_EXCEPTIONS}"
        );
        found[slot] = true;
    }
    for (matrix, [codes, scales, columns, values]) in present {
        ensure!(
            codes == scales,
            "W8 artifact holds only one of the codes and scales of {matrix}"
        );
        ensure!(
            columns == values && (codes || !columns),
            "W8 exception columns of {matrix} need their values and the matrix codes"
        );
        ensure!(codes || partial, "W8 artifact lacks {matrix} and is not marked partial");
    }
    Ok(())
}

/// Body matrix `name` (`[output, input]`, one scale per `group_size`
/// inputs) from a W8 overlay: its codes and scales and any exception
/// columns; `None` when a `partial` overlay leaves it out (it stays FP32).
fn overlay_matrix(
    tensors: &SafeTensors<'_>,
    name: &str,
    (output, input): (usize, usize),
    group_size: usize,
    partial: bool,
) -> Result<Option<crate::quant::linear::QuantLinear>> {
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
    let q = crate::quant::linear::QuantLinear::from_parts(output, input, group_size, codes, scales)?;
    Ok(Some(match overlay_exceptions(tensors, name, output)? {
        Some((columns, values)) => q
            .with_exceptions(columns, values)
            .with_context(|| format!("W8 exception columns of {name}"))?,
        None => q,
    }))
}

/// The exception columns and FP32 values a W8 overlay holds for matrix
/// `name` (`output` rows), if any; `QuantLinear::with_exceptions` checks
/// them.
fn overlay_exceptions(tensors: &SafeTensors<'_>, name: &str, output: usize) -> Result<Option<(Vec<i32>, Vec<f32>)>> {
    let columns_name = format!("{name}{}", W8_SUFFIXES[2]);
    if !tensors.names().iter().any(|n| **n == columns_name) {
        return Ok(None);
    }
    let columns = tensors.tensor(&columns_name)?;
    let values = tensors.tensor(&format!("{name}{}", W8_SUFFIXES[3]))?;
    let count = columns.shape().first().copied().unwrap_or(0);
    ensure!(
        columns.dtype() == Dtype::I32 && columns.shape() == [count],
        "W8 exception columns dtype/shape for {name}"
    );
    ensure!(
        values.dtype() == Dtype::F32 && values.shape() == [output, count],
        "W8 exception values dtype/shape for {name}"
    );
    let columns = columns
        .data()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let values = values
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(Some((columns, values)))
}

/// Bytes of the exception columns a W8 overlay's header lists (read
/// without the tensors), for byte budgets.
pub(crate) fn overlay_exception_bytes(path: &Path) -> Result<usize> {
    let header = super::packed::safetensors_header(path).with_context(|| format!("read {}", path.display()))?;
    Ok(super::packed::tensor_bytes_with_suffix(&header, &W8_SUFFIXES[2..]))
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
mod overlay_tests {
    use super::*;

    #[test]
    fn overlay_tensor_sets_are_checked_by_name() {
        let matrices = body_matrix_names(2);
        assert_eq!(matrices.len(), 8);
        assert_eq!(matrices[7], "layers.1.feed_forward.w2.weight");
        let full: Vec<String> = matrices
            .iter()
            .flat_map(|m| [format!("{m}.__w8_codes"), format!("{m}.__w8_scales")])
            .collect();
        let check = |names: &[String], partial: bool, exceptions: bool| {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            check_overlay_tensors(&names, &matrices, partial, exceptions).map_err(|e| e.to_string())
        };
        assert!(check(&full, false, false).is_ok() && check(&full, false, true).is_ok());
        // Only a partial overlay may omit matrices, and only whole ones.
        assert!(
            check(&full[2..], false, false)
                .unwrap_err()
                .contains("not marked partial")
        );
        assert!(check(&full[2..], true, false).is_ok() && check(&[], true, false).is_ok());
        assert!(check(&full[1..], true, false).unwrap_err().contains("only one of"));
        for extra in [
            "layers.2.attention.wo.weight.__w8_codes",
            "output.weight",
            "layers.0.x.__w8_scales",
        ] {
            let names = [full.clone(), vec![extra.to_owned()]].concat();
            assert!(
                check(&names, false, true).unwrap_err().contains("unexpected tensor"),
                "{extra}"
            );
        }
        // Exception columns: format v2 only, with their values, beside codes.
        let w2 = &matrices[3];
        let exceptions = [format!("{w2}.__w8_exc_cols"), format!("{w2}.__w8_exc_vals")];
        let names = [full.clone(), exceptions.to_vec()].concat();
        assert!(check(&names, false, true).is_ok());
        assert!(check(&names, false, false).unwrap_err().contains(W8_FORMAT_EXCEPTIONS));
        assert!(
            check(&names[..names.len() - 1], false, true)
                .unwrap_err()
                .contains("need their values")
        );
        let orphan = [full[8..].to_vec(), exceptions.to_vec()].concat();
        assert!(check(&orphan, true, true).unwrap_err().contains("need their values"));
    }

    #[test]
    fn overlay_exceptions_read_columns_and_values() {
        use safetensors::tensor::TensorView;
        let columns: Vec<u8> = [3i32, 9].iter().flat_map(|c| c.to_le_bytes()).collect();
        let values: Vec<u8> = [1.5f32, -2.0, 0.25, 8.0, 3.0, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = safetensors::serialize(
            [
                (
                    "m.__w8_exc_cols",
                    TensorView::new(Dtype::I32, vec![2], &columns).unwrap(),
                ),
                (
                    "m.__w8_exc_vals",
                    TensorView::new(Dtype::F32, vec![3, 2], &values).unwrap(),
                ),
                (
                    "f.__w8_exc_cols",
                    TensorView::new(Dtype::F32, vec![2], &columns).unwrap(),
                ),
                (
                    "f.__w8_exc_vals",
                    TensorView::new(Dtype::F32, vec![3, 2], &values).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let (columns, values) = overlay_exceptions(&tensors, "m", 3).unwrap().unwrap();
        assert_eq!(columns, [3, 9]);
        assert_eq!(values, [1.5, -2.0, 0.25, 8.0, 3.0, -0.5]);
        assert!(overlay_exceptions(&tensors, "absent", 3).unwrap().is_none());
        assert!(overlay_exceptions(&tensors, "m", 2).is_err(), "values shape");
        assert!(overlay_exceptions(&tensors, "f", 3).is_err(), "columns dtype");
    }

    /// One matrix imported from small v1 and v2 overlays: v2 carries its
    /// exception columns into every weight the products read.
    #[test]
    fn overlay_matrices_import_codes_scales_and_exception_columns() {
        use safetensors::tensor::TensorView;
        let (output, input) = (4, 128);
        let mut codes: Vec<i8> = (0..output * input).map(|i| (i % 11) as i8 - 5).collect();
        for row in 0..output {
            codes[row * input + 3] = 0;
            codes[row * input + 70] = 0;
        }
        let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let values: Vec<f32> = (0..output * 2).map(|i| 100.0 + i as f32).collect();
        let overlay = |codes: &[i8], exceptions: bool| {
            let mut tensors = vec![
                (
                    "m.__w8_codes",
                    Dtype::I8,
                    vec![output, input],
                    codes.iter().map(|&c| c as u8).collect(),
                ),
                ("m.__w8_scales", Dtype::F32, vec![output, 2], f32_bytes(&[0.25; 8])),
            ];
            if exceptions {
                let columns = [3i32, 70].iter().flat_map(|c| c.to_le_bytes()).collect();
                tensors.push(("m.__w8_exc_cols", Dtype::I32, vec![2], columns));
                tensors.push(("m.__w8_exc_vals", Dtype::F32, vec![output, 2], f32_bytes(&values)));
            }
            let views = tensors
                .iter()
                .map(|(name, dtype, shape, data)| (*name, TensorView::new(*dtype, shape.clone(), data).unwrap()));
            safetensors::serialize(views, None).unwrap()
        };
        let import = |bytes: &[u8], name: &str, shape: (usize, usize), partial: bool| {
            overlay_matrix(&SafeTensors::deserialize(bytes).unwrap(), name, shape, 64, partial)
        };
        let (v1, v2) = (overlay(&codes, false), overlay(&codes, true));
        let plain = import(&v1, "m", (output, input), false).unwrap().unwrap();
        assert!(plain.exception_columns().is_empty());
        let exact = import(&v2, "m", (output, input), false).unwrap().unwrap();
        assert_eq!(exact.exception_columns(), &[3, 70]);
        let mut row = vec![0.0; input];
        for r in 0..output {
            exact.dequantize_row(r, &mut row);
            assert_eq!((row[3], row[70]), (values[2 * r], values[2 * r + 1]));
            assert_eq!(row[4], codes[r * input + 4] as f32 * 0.25);
            plain.dequantize_row(r, &mut row);
            assert_eq!((row[3], row[70]), (0.0, 0.0));
        }
        // Partial overlays may leave a matrix out; shapes and exceptions are checked.
        assert!(import(&v1, "absent", (output, input), true).unwrap().is_none());
        assert!(import(&v1, "absent", (output, input), false).is_err());
        assert!(import(&v1, "m", (output, 2 * input), false).is_err());
        codes[70] = 1;
        let error = import(&overlay(&codes, true), "m", (output, input), false).unwrap_err();
        assert!(format!("{error:#}").contains("exception columns of m"), "{error:#}");
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
