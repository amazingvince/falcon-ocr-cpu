//! Quantized weights and KV caches: the storage profiles (`Weights` x `Kv`)
//! the modes map onto, the quantized linear layer, and the split KV cache.
//! FP32 graph arithmetic is unchanged unless a weight or cache is encoded, and
//! every encoded path is tested bitwise against its dequantized reference.
pub(crate) mod kv;
pub mod linear;

use serde::{Deserialize, Serialize};

/// Body weight storage (embedding, norms, projector and vocabulary head stay
/// FP32 in every profile).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Weights {
    #[default]
    F32,
    /// 16-bit codes with one FP32 absmax scale per 64 inputs, quantized at
    /// load: effectively lossless (not bitwise).
    Int16,
    /// 8-bit codes with one FP32 scale per 64 inputs: the GPTQ overlay, or
    /// round-to-nearest at load.
    Int8,
}

/// KV cache storage once a page's prefix is sealed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kv {
    /// The reference compact FP32 cache (keys per query head); never sealed.
    #[default]
    Compact,
    /// FP32 split records (group-major, temporal key half shared by the
    /// pair): bitwise the compact kernel, about 7% faster decode.
    F32Split,
    /// 16-bit codes with one BF16 absmax scale per 32 elements: half the
    /// bytes of FP32 records.
    Q16,
    /// 8-bit codes with one BF16 absmax scale per 32 elements.
    Q8,
}

/// A weight and KV storage pair. Labels are the strings kernel-ready files
/// record and the eval binary's `--profile` takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Profile {
    pub weights: Weights,
    pub kv: Kv,
}

impl Profile {
    pub const REFERENCE: Self = Self::new(Weights::F32, Kv::Compact);
    pub const SPLIT_F32: Self = Self::new(Weights::F32, Kv::F32Split);
    pub const KV_Q16: Self = Self::new(Weights::F32, Kv::Q16);
    pub const KV_Q8: Self = Self::new(Weights::F32, Kv::Q8);
    pub const W16_BODY_COMPACT: Self = Self::new(Weights::Int16, Kv::Compact);
    pub const W16_BODY: Self = Self::new(Weights::Int16, Kv::F32Split);
    /// The near-exact mode.
    pub const W16_BODY_KV_Q16: Self = Self::new(Weights::Int16, Kv::Q16);
    pub const W16_BODY_KV_Q8: Self = Self::new(Weights::Int16, Kv::Q8);
    pub const W8_BODY: Self = Self::new(Weights::Int8, Kv::Compact);
    pub const W8_BODY_SPLIT_F32: Self = Self::new(Weights::Int8, Kv::F32Split);
    pub const W8_BODY_KV_Q16: Self = Self::new(Weights::Int8, Kv::Q16);
    /// The fast mode.
    pub const W8_BODY_KV_Q8: Self = Self::new(Weights::Int8, Kv::Q8);
    /// Every combination.
    pub const ALL: [Self; 12] = [
        Self::REFERENCE,
        Self::SPLIT_F32,
        Self::KV_Q16,
        Self::KV_Q8,
        Self::W16_BODY_COMPACT,
        Self::W16_BODY,
        Self::W16_BODY_KV_Q16,
        Self::W16_BODY_KV_Q8,
        Self::W8_BODY,
        Self::W8_BODY_SPLIT_F32,
        Self::W8_BODY_KV_Q16,
        Self::W8_BODY_KV_Q8,
    ];

    pub const fn new(weights: Weights, kv: Kv) -> Self {
        Self { weights, kv }
    }
    /// The label kernel-ready files record (`w16-body` and `w8-body` keep
    /// their historical caches: split FP32 and compact).
    pub fn label(self) -> &'static str {
        match (self.weights, self.kv) {
            (Weights::F32, Kv::Compact) => "reference",
            (Weights::F32, Kv::F32Split) => "split-f32",
            (Weights::F32, Kv::Q16) => "kv-q16",
            (Weights::F32, Kv::Q8) => "kv-q8",
            (Weights::Int16, Kv::Compact) => "w16-body-compact",
            (Weights::Int16, Kv::F32Split) => "w16-body",
            (Weights::Int16, Kv::Q16) => "w16-body-kv-q16",
            (Weights::Int16, Kv::Q8) => "w16-body-kv-q8",
            (Weights::Int8, Kv::Compact) => "w8-body",
            (Weights::Int8, Kv::F32Split) => "w8-body-split-f32",
            (Weights::Int8, Kv::Q16) => "w8-body-kv-q16",
            (Weights::Int8, Kv::Q8) => "w8-body-kv-q8",
        }
    }
    /// The profile a label names.
    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.label() == label)
    }
    pub fn quantizes_body(self) -> bool {
        self.weights != Weights::F32
    }
    /// Bits per body weight: 32, 16 or 8.
    pub fn weight_bits(self) -> u32 {
        match self.weights {
            Weights::F32 => 32,
            Weights::Int16 => 16,
            Weights::Int8 => 8,
        }
    }
    /// Profiles whose outputs are bit-identical to the FP32 reference.
    pub fn is_exact(self) -> bool {
        self.weights == Weights::F32 && matches!(self.kv, Kv::Compact | Kv::F32Split)
    }
    /// Every profile but the reference retires finished caches early and
    /// drops the input patches once decoded.
    pub fn memory_hygiene(self) -> bool {
        self != Self::REFERENCE
    }
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for Profile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| {
            let labels: Vec<_> = Self::ALL.iter().map(|p| p.label()).collect();
            format!("unknown profile {s:?}; one of {}", labels.join(", "))
        })
    }
}

impl Serialize for Profile {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.label())
    }
}

impl<'de> Deserialize<'de> for Profile {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let label = String::deserialize(deserializer)?;
        label.parse().map_err(serde::de::Error::custom)
    }
}

impl clap::ValueEnum for Profile {
    fn value_variants<'a>() -> &'a [Self] {
        &Self::ALL
    }
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.label()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip_and_keep_the_shipped_strings() {
        for profile in Profile::ALL {
            assert_eq!(Profile::parse(profile.label()), Some(profile));
            assert_eq!(profile.label().parse::<Profile>().unwrap(), profile);
            assert_eq!(
                serde_json::to_value(profile).unwrap(),
                serde_json::json!(profile.label())
            );
            assert_eq!(
                serde_json::from_value::<Profile>(serde_json::json!(profile.label())).unwrap(),
                profile
            );
            assert_eq!(
                <Profile as clap::ValueEnum>::from_str(profile.label(), false).unwrap(),
                profile
            );
        }
        assert_eq!(Profile::REFERENCE.label(), "reference");
        assert_eq!(Profile::SPLIT_F32.label(), "split-f32");
        assert_eq!(Profile::W16_BODY_KV_Q16.label(), "w16-body-kv-q16");
        assert_eq!(Profile::W8_BODY_KV_Q8.label(), "w8-body-kv-q8");
        assert!(Profile::parse("w8-all").is_none() && Profile::parse("kv-bf16").is_none());
        assert!("hygiene".parse::<Profile>().unwrap_err().contains("w16-body-kv-q16"));
    }

    #[test]
    fn predicates_follow_the_storage_pair() {
        assert!(Profile::REFERENCE.is_exact() && Profile::SPLIT_F32.is_exact());
        assert!(!Profile::KV_Q16.is_exact() && !Profile::W16_BODY.is_exact());
        assert!(!Profile::REFERENCE.memory_hygiene() && Profile::SPLIT_F32.memory_hygiene());
        assert_eq!(
            Profile::ALL.map(Profile::weight_bits),
            [32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8]
        );
        assert_eq!(Profile::ALL.iter().filter(|p| p.quantizes_body()).count(), 8);
        assert_eq!(Profile::default(), Profile::REFERENCE);
    }
}
