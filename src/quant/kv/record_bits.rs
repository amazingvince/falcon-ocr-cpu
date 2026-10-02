//! A test accessor: the stored prefix records of a sealed cache as bits.
use super::{Records, SplitPrefix};

impl SplitPrefix {
    /// The stored prefix records as bits: codes, then scales (tests compare
    /// two seals of one prefix).
    pub(crate) fn record_bits(&self) -> Vec<u32> {
        match &self.records {
            Records::F32(data) => data.iter().map(|x| x.to_bits()).collect(),
            Records::Q8 { codes, scales } => codes
                .iter()
                .map(|&code| u32::from(code as u8))
                .chain(scales.iter().map(|&scale| u32::from(scale)))
                .collect(),
            Records::Q16 { codes, scales } => codes
                .iter()
                .map(|&code| u32::from(code as u16))
                .chain(scales.iter().map(|&scale| u32::from(scale)))
                .collect(),
            Records::Q4 { codes, scales } => codes
                .iter()
                .map(|&pair| u32::from(pair))
                .chain(scales.iter().map(|&scale| u32::from(scale)))
                .collect(),
        }
    }
}
