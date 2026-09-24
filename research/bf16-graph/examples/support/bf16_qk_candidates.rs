//! Fixed, structurally motivated hypotheses, not hardware emulation claims.
//! PTX guarantees neither an internal BF16 MMA sum order nor rounding scheme.
use falcon_ocr::bf16_kernels::{self, Backend};
use half::bf16;

#[derive(Clone, Copy, Debug)]
pub enum Candidate {
    Baseline,
    Sequential,
    K16Sequential,
    K16Tree,
    K16FragmentTree,
    K16Fused,
    K8Fused,
    F64,
}
pub const ALL: [Candidate; 8] = [
    Candidate::Baseline,
    Candidate::Sequential,
    Candidate::K16Sequential,
    Candidate::K16Tree,
    Candidate::K16FragmentTree,
    Candidate::K16Fused,
    Candidate::K8Fused,
    Candidate::F64,
];
impl Candidate {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "avx512_baseline",
            Self::Sequential => "sequential_f32",
            Self::K16Sequential => "k16_sequential",
            Self::K16Tree => "k16_pair_tree",
            Self::K16FragmentTree => "k16_fragment_tree",
            Self::K16Fused => "k16_fused_f64",
            Self::K8Fused => "k8_fused_f64",
            Self::F64 => "whole_f64_control",
        }
    }
    pub fn rationale(self) -> &'static str {
        match self {
            Self::Baseline => {
                "Unchanged AVX512BF16 implementation, with current P*V, exp2 and softmax order."
            }
            Self::Sequential => {
                "Scalar F32 ordered sum control; P*V remains AVX512BF16 to isolate QK only."
            }
            Self::K16Sequential => {
                "Four serial K16 instruction-sized panels; each panel sums products sequentially then updates F32 accumulator."
            }
            Self::K16Tree => {
                "Four serial K16 panels with adjacent-pair balanced trees; plausible unspecified within-instruction reduction hypothesis."
            }
            Self::K16FragmentTree => {
                "K16 PTX four-lane fragment layout: each lane owns pairs (2l,2l+1) and (2l+8,2l+9); combine lane pairs then four lanes, then carried accumulator."
            }
            Self::K16Fused => {
                "Four serial K16 panels with F64 product sum plus incoming F32 accumulator and one F32 round per panel; fused panel hypothesis, not NVIDIA guarantee."
            }
            Self::K8Fused => {
                "Two K8 halves per K16 instruction using the documented lower/upper fragment split; F64 sum plus incoming accumulator, F32 round per half."
            }
            Self::F64 => {
                "Accuracy control: whole64 exact BF16 products summed in F64 then one F32 round; not an MMA emulation claim."
            }
        }
    }
    pub fn dot(self, a: &[bf16], b: &[bf16]) -> f32 {
        assert_eq!(a.len(), 64);
        assert_eq!(b.len(), 64);
        if matches!(self, Self::Baseline) {
            return bf16_kernels::dot(a, b, Backend::Avx512Bf16);
        }
        if matches!(self, Self::F64) {
            return a
                .iter()
                .zip(b)
                .map(|(a, b)| a.to_f64() * b.to_f64())
                .sum::<f64>() as f32;
        }
        let product: [f32; 64] = std::array::from_fn(|i| a[i].to_f32() * b[i].to_f32());
        if matches!(self, Self::Sequential) {
            return product.into_iter().sum();
        }
        if matches!(self, Self::K16Fused | Self::K8Fused) {
            let width = if matches!(self, Self::K8Fused) { 8 } else { 16 };
            let mut acc = 0f32;
            for panel in product.chunks_exact(width) {
                let mut exact = acc as f64;
                for &p in panel {
                    exact += p as f64;
                }
                acc = exact as f32;
            }
            return acc;
        }
        let mut acc = 0.;
        for panel in product.chunks_exact(16) {
            let sum = match self {
                Self::K16Sequential => panel.iter().copied().sum(),
                Self::K16Tree => {
                    let mut tree: [f32; 16] = panel.try_into().unwrap();
                    let mut width = 16;
                    while width > 1 {
                        for i in 0..width / 2 {
                            tree[i] = tree[2 * i] + tree[2 * i + 1];
                        }
                        width /= 2;
                    }
                    tree[0]
                }
                Self::K16FragmentTree => {
                    let lane: [f32; 4] = std::array::from_fn(|i| {
                        (panel[2 * i] + panel[2 * i + 1]) + (panel[2 * i + 8] + panel[2 * i + 9])
                    });
                    (lane[0] + lane[1]) + (lane[2] + lane[3])
                }
                _ => unreachable!(),
            };
            acc += sum;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_candidate_visits_each_k_lane_once() {
        for candidate in ALL {
            for lane in 0..64 {
                let mut a = [bf16::ZERO; 64];
                a[lane] = bf16::from_f32(3.);
                assert_eq!(
                    candidate.dot(&a, &[bf16::from_f32(-2.); 64]),
                    -6.,
                    "{} lane{lane}",
                    candidate.name()
                );
            }
        }
    }
    #[test]
    fn finite_normal_bf16_operands_stay_within_forward_error_bound() {
        for candidate in ALL {
            for seed in 0..19 {
                let a: Vec<_> = (0..64)
                    .map(|i| bf16::from_f32((((i * 31 + seed * 7) % 101) as f32 - 50.) * 0.017))
                    .collect();
                let b: Vec<_> = (0..64)
                    .map(|i| bf16::from_f32((((i * 23 + seed * 11) % 97) as f32 - 48.) * 0.031))
                    .collect();
                let exact: f64 = a.iter().zip(&b).map(|(a, b)| a.to_f64() * b.to_f64()).sum();
                let magnitude: f64 = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a.to_f64() * b.to_f64()).abs())
                    .sum();
                assert!(
                    (candidate.dot(&a, &b) as f64 - exact).abs()
                        <= 64. * f32::EPSILON as f64 * magnitude.max(1.)
                );
            }
        }
    }
}
