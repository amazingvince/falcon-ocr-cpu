//! Experiment and diagnostic knobs (`--tune key=value`) and their values.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Whether the prefill attention of an 8-bit model uses BF16 products on an
/// AVX512-BF16 CPU; fidelity-neutral against the FP32 anchor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PrefillBf16 {
    Off,
    #[default]
    Attention,
}

/// Activations of an 8-bit model's prefill projections
/// (`Tuning::prefill_projections`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillProjections {
    /// FP32 activations against FP32 panels dequantized from the codes.
    Fp32,
    /// INT16 activations (one step per row and 64 inputs) against the 8-bit
    /// codes in INT16 panels, on x86 CPUs with AVX2: AVX512-VNNI where the
    /// `auto` backend has it, bitwise the same AVX2 kernel otherwise. A body
    /// with groups of 32, aarch64 and CPUs without AVX2 keep the FP32
    /// panels. Exception columns stay FP32 and add after the integer
    /// groups. Rounds fast mode's prefill activations.
    #[default]
    I16,
}

/// Softmax of the BF16 prefill attention (`Tuning::prefill_softmax`) or of
/// the FP32 prefill tiles (`Tuning::prefill_softmax_fp32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillSoftmax {
    /// Online softmax: a running maximum per query row, with the output and
    /// denominator rescaled whenever it grows. CPUs without AVX512BW run it
    /// in the BF16 attention whatever the knob says.
    Online,
    /// No shift at all: the per-head RMS norms bound every score by 8 in
    /// magnitude, so `exp(score)` neither overflows nor underflows, and key
    /// tiles add without rescaling (`prefill64::bf16::static_shift` and
    /// `prefill64::static_shift`). Rounds differently from the online
    /// softmax.
    Static,
}

/// Storage of the draft head's keys and values (`Tuning::draft_kv`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftKv {
    F32,
    /// Int8 with one scale per 64 values: a quarter of the bytes per draft
    /// step, acceptance within 0.5 point of FP32, and faster page totals
    /// (calibration and held-out A/Bs, research/draft-head).
    #[default]
    Q8,
}

/// Experiment and diagnostic knobs. The defaults are the accepted
/// configuration; every field is scheduling or instrumentation only, except
/// `prefill_bf16`, `prefill_projections` and `prefill_softmax`, which change
/// fast-mode prefill rounding, `prefill_softmax_fp32`, which changes it for
/// 16- and 8-bit bodies, `split_chunks` and `decode_fast_exp`, which change
/// the rounding of split-cache decode attention, and `kv_q8_layers`, which
/// lowers the precision of chosen layers' caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// BF16 prefill attention of 8-bit models on AVX512-BF16 CPUs.
    pub prefill_bf16: PrefillBf16,
    /// Activations of 8-bit models' prefill projections.
    pub prefill_projections: PrefillProjections,
    /// Softmax of the BF16 prefill attention (default static).
    pub prefill_softmax: PrefillSoftmax,
    /// Softmax of the FP32 prefill tiles of 16- and 8-bit bodies on x86.
    /// `None` keeps the default: static for 16-bit bodies (near-exact mode,
    /// qualified on held-out pages), online for 8-bit ones (fast mode's FP32
    /// fallback, unqualified). FP32 bodies (exact mode) keep the online
    /// softmax whatever this says, and so do aarch64 and the scalar backend.
    /// When BF16 attention is planned, this knob has no effect, including
    /// in calls that fall back to FP32; `prefill_softmax` selects the BF16
    /// softmax.
    pub prefill_softmax_fp32: Option<PrefillSoftmax>,
    /// Position chunks per KV group in split-cache decode attention (1..=4).
    /// `None` keeps the mode default: one exact scan for FP32 caches, four
    /// chunks (merged partial softmaxes, rounding-level) for quantized ones.
    pub split_chunks: Option<usize>,
    /// Bit L seals layer L's cache at Q8 in an otherwise Q16 cache, for its
    /// keys and values and for its tail. The cache's exp stays Q16's.
    pub kv_q8_layers: u32,
    /// Print a wall-clock phase split of every forward pass to stderr.
    pub phases: bool,
    /// Accumulate cycle counters of the prefill attention stages (stderr).
    pub prefill_profile: bool,
    /// After consecutive draft-head attempts that drafted nothing, skip up
    /// to this many tokens before trying again (1, 2, 4, ... tokens); 0 tries
    /// at every token.
    pub draft_backoff: usize,
    /// The draft head attends to at most this many newest verified
    /// positions (0: all).
    pub draft_window: usize,
    /// Storage of the draft head's keys and values.
    pub draft_kv: DraftKv,
    /// The draft head stops when the product of its chain's probabilities
    /// (rather than one token's) falls below `RunnerConfig::draft_confidence`.
    pub draft_path_gate: bool,
    /// Decode attention over sealed (split) caches uses the portable
    /// polynomial exp on x86, as the NEON kernels do, instead of the
    /// platform-exact exp: faster softmax, different rounding. `None` keeps
    /// the default: polynomial for 8-bit caches (and the rotated research
    /// caches) under `ExpMode::Fast` (fast mode), exact otherwise.
    pub decode_fast_exp: Option<bool>,
}
impl Default for Tuning {
    /// The accepted configuration: for 8-bit bodies BF16 attention with the
    /// static softmax and INT16 projections, the body's softmax in the FP32
    /// tiles, the draft head's Q8 cache, no backoff, window or path gate,
    /// the mode's split chunks and decode exp, and no instrumentation.
    fn default() -> Self {
        Self {
            prefill_bf16: PrefillBf16::default(),
            prefill_projections: PrefillProjections::default(),
            prefill_softmax: PrefillSoftmax::Static,
            prefill_softmax_fp32: None,
            split_chunks: None,
            kv_q8_layers: 0,
            phases: false,
            prefill_profile: false,
            draft_backoff: 0,
            draft_window: 0,
            draft_kv: DraftKv::default(),
            draft_path_gate: false,
            decode_fast_exp: None,
        }
    }
}
impl Tuning {
    /// Set one knob from a `key=value` pair (`prefill-bf16=off|attention`,
    /// `prefill-projections=fp32|i16`, `prefill-softmax=online|static`,
    /// `prefill-softmax-fp32=online|static`, `split-chunks=1..4`,
    /// `phases=true|false`, `prefill-profile=true|false`, `draft-backoff=N`,
    /// `draft-window=N`, `draft-kv=f32|q8`, `draft-gate=token|path`,
    /// `decode-exp=exact|fast`, `kv-layers=0,3`).
    pub fn set(&mut self, pair: &str) -> Result<()> {
        let (key, value) = pair
            .split_once('=')
            .with_context(|| format!("tuning knob `{pair}` is not key=value"))?;
        let flag = |value: &str| -> Result<bool> {
            match value {
                "1" | "true" | "on" => Ok(true),
                "0" | "false" | "off" => Ok(false),
                _ => bail!("tuning knob {key}: expected true or false, got `{value}`"),
            }
        };
        let softmax = |value: &str| -> Result<PrefillSoftmax> {
            match value {
                "online" => Ok(PrefillSoftmax::Online),
                "static" => Ok(PrefillSoftmax::Static),
                _ => bail!("tuning knob {key}: expected online or static, got `{value}`"),
            }
        };
        match key {
            "prefill-bf16" => {
                self.prefill_bf16 = match value {
                    "off" | "0" => PrefillBf16::Off,
                    "attention" | "attn" => PrefillBf16::Attention,
                    _ => bail!("tuning knob prefill-bf16: expected off or attention, got `{value}`"),
                }
            }
            "prefill-projections" => {
                self.prefill_projections = match value {
                    "fp32" => PrefillProjections::Fp32,
                    "i16" => PrefillProjections::I16,
                    _ => bail!("tuning knob prefill-projections: expected fp32 or i16, got `{value}`"),
                }
            }
            "prefill-softmax" => self.prefill_softmax = softmax(value)?,
            "prefill-softmax-fp32" => self.prefill_softmax_fp32 = Some(softmax(value)?),
            "split-chunks" => {
                let chunks: usize = value
                    .parse()
                    .with_context(|| format!("tuning knob split-chunks: `{value}`"))?;
                ensure!(
                    (1..=4).contains(&chunks),
                    "tuning knob split-chunks: expected 1..=4, got {chunks}"
                );
                self.split_chunks = Some(chunks);
            }
            "kv-layers" => {
                let mut layers = 0;
                for layer in value.split(',') {
                    let layer: u32 = layer
                        .parse()
                        .with_context(|| format!("tuning knob kv-layers: `{layer}`"))?;
                    ensure!(layer < 32, "tuning knob kv-layers: layer {layer} is not below 32");
                    layers |= 1 << layer;
                }
                self.kv_q8_layers = layers;
            }
            "phases" => self.phases = flag(value)?,
            "prefill-profile" => self.prefill_profile = flag(value)?,
            "draft-backoff" => {
                self.draft_backoff = value
                    .parse()
                    .with_context(|| format!("tuning knob draft-backoff: `{value}`"))?;
            }
            "draft-window" => {
                self.draft_window = value
                    .parse()
                    .with_context(|| format!("tuning knob draft-window: `{value}`"))?;
            }
            "draft-kv" => {
                self.draft_kv = match value {
                    "f32" => DraftKv::F32,
                    "q8" | "int8" => DraftKv::Q8,
                    _ => bail!("tuning knob draft-kv: expected f32 or q8, got `{value}`"),
                }
            }
            "draft-gate" => {
                self.draft_path_gate = match value {
                    "token" => false,
                    "path" => true,
                    _ => bail!("tuning knob draft-gate: expected token or path, got `{value}`"),
                }
            }
            "decode-exp" => {
                self.decode_fast_exp = match value {
                    "exact" => Some(false),
                    "fast" => Some(true),
                    _ => bail!("tuning knob decode-exp: expected exact or fast, got `{value}`"),
                }
            }
            _ => bail!(
                "unknown tuning knob `{key}` (prefill-bf16, prefill-projections, prefill-softmax, \
                 prefill-softmax-fp32, split-chunks, phases, prefill-profile, draft-backoff, draft-window, draft-kv, \
                 draft-gate, decode-exp, kv-layers)"
            ),
        }
        Ok(())
    }
    /// Rejects `kv_q8_layers` unless the cache is Q16 and the model has the
    /// layers: a compact cache never seals, so the knob would do nothing.
    pub(crate) fn check_kv_layers(&self, kv: crate::quant::Kv, layers: usize) -> Result<()> {
        ensure!(
            self.kv_q8_layers == 0
                || kv == crate::quant::Kv::Q16 && self.kv_q8_layers.checked_shr(layers as u32).unwrap_or(0) == 0,
            "kv-layers needs a Q16 cache and layers below {layers}"
        );
        Ok(())
    }
    /// The layers of a `kv_q8_layers` mask, as `kv-layers` takes them.
    pub(crate) fn kv_layer_list(mask: u32) -> String {
        let layers: Vec<String> = (0..32).filter(|l| mask >> l & 1 != 0).map(|l| l.to_string()).collect();
        layers.join(",")
    }
    /// The default knobs with every `key=value` pair of `pairs` applied.
    pub fn from_pairs<I, S>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut tuning = Self::default();
        for pair in pairs {
            tuning.set(pair.as_ref())?;
        }
        Ok(tuning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ExpMode, RunnerConfig};

    #[test]
    fn tuning_knobs_parse_and_reject_unknown_keys() {
        let mut tuning = Tuning::from_pairs([
            "prefill-bf16=off",
            "prefill-projections=fp32",
            "prefill-softmax=online",
            "prefill-softmax-fp32=static",
            "split-chunks=3",
            "phases=1",
            "draft-window=256",
            "draft-kv=f32",
            "draft-gate=path",
            "decode-exp=fast",
            "kv-layers=0,3,31",
        ])
        .unwrap();
        assert_eq!(tuning.decode_fast_exp, Some(true));
        assert_eq!(tuning.kv_q8_layers, 1 | 1 << 3 | 1 << 31);
        assert!(tuning.set("kv-layers=1,32").is_err());
        assert_eq!(Tuning::kv_layer_list(tuning.kv_q8_layers), "0,3,31");
        use crate::quant::Kv;
        assert!(tuning.check_kv_layers(Kv::Q16, 32).is_ok());
        assert!(tuning.check_kv_layers(Kv::Q16, 22).is_err());
        assert!(tuning.check_kv_layers(Kv::Q8, 32).is_err());
        assert!(tuning.check_kv_layers(Kv::Compact, 32).is_err());
        assert!(Tuning::default().check_kv_layers(Kv::Compact, 22).is_ok());
        assert_eq!((tuning.draft_window, tuning.draft_kv), (256, DraftKv::F32));
        assert!(tuning.draft_path_gate);
        assert_eq!(tuning.prefill_bf16, PrefillBf16::Off);
        assert_eq!(tuning.prefill_projections, PrefillProjections::Fp32);
        assert_eq!(tuning.prefill_softmax, PrefillSoftmax::Online);
        assert_eq!(tuning.prefill_softmax_fp32, Some(PrefillSoftmax::Static));
        assert_eq!(tuning.split_chunks, Some(3));
        assert!(tuning.phases && !tuning.prefill_profile);
        assert_eq!(Tuning::from_pairs(Vec::<&str>::new()).unwrap(), Tuning::default());
        let online = Tuning::from_pairs(["prefill-softmax-fp32=online"]).unwrap();
        assert_eq!(online.prefill_softmax_fp32, Some(PrefillSoftmax::Online));
        // Fast mode's accepted prefill: BF16 attention with the static
        // softmax and INT16 projections; the FP32 tiles take the body's
        // default softmax (`prefill_plan`).
        let defaults = Tuning::default();
        assert_eq!(defaults.prefill_bf16, PrefillBf16::Attention);
        assert_eq!(defaults.prefill_projections, PrefillProjections::I16);
        assert_eq!(defaults.prefill_softmax, PrefillSoftmax::Static);
        assert_eq!(defaults.prefill_softmax_fp32, None);
        // A plan written before the FP32 knob reads as the default; one
        // written while the knob had a fixed default keeps its value.
        let older: Tuning = serde_json::from_str(r#"{"prefill_softmax":"online"}"#).unwrap();
        assert_eq!(older.prefill_softmax_fp32, None);
        let fixed: Tuning = serde_json::from_str(r#"{"prefill_softmax_fp32":"online"}"#).unwrap();
        assert_eq!(fixed.prefill_softmax_fp32, Some(PrefillSoftmax::Online));
        let serialized = serde_json::to_value(defaults).unwrap();
        assert_eq!(serialized["prefill_softmax_fp32"], serde_json::Value::Null);
        // Bad values, and unknown keys including the removed knobs.
        for bad in [
            "prefill-softmax-fp32=fast",
            "split-chunks=5",
            "split-chunks=x",
            "prefill-bf16=all",
            "prefill-i16=true",
            "prefill-bf16-w16=attention",
            "prefill-projections=bf16",
            "prefill-softmax=max",
            "phases=maybe",
            "draft-kv=f16",
            "draft-window=-1",
            "kv-layers=",
            "kv-layers=32",
            "kv-layers=-1",
            "kv-layers=0k",
            "unknown=1",
            "phases",
        ] {
            assert!(Tuning::from_pairs([bad]).is_err(), "{bad}");
        }
        let config: RunnerConfig = serde_json::from_str(r#"{"threads":2,"batch_size":1}"#).unwrap();
        assert_eq!(config.exp, ExpMode::Fast);
        assert_eq!(config.tuning, Tuning::default());
    }
}
