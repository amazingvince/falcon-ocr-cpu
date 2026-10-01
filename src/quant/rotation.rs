//! Randomized Hadamard rotation of 32-value blocks, the transform behind the
//! rotated KV caches (`Kv::Q8Rot`, `Kv::Q4Rot`).
//!
//! A rotated cache stores each 32-value block `x` of a key or value (one
//! quantization group) as `R x` with `R = H D`: `H` is the 32x32 Sylvester
//! Hadamard matrix (entries ±1, unnormalized, so `H H = 32 I`) and `D` a fixed
//! ±1 diagonal per [`Block`] kind. Absmax quantization of `R x` spreads an
//! outlier channel's energy over the whole block instead of sizing the step
//! of 31 small values for the one large value. Decode never un-rotates the
//! cache: each query block is rotated instead, `(2^-5 R q) · (R k) = q · k`,
//! and because `Σ p (R v) = R Σ p v` each attention output block is
//! un-rotated once, `o = 2^-5 D H o'`.
//!
//! Sign flips and the scaling by 2^-5 are exact in floating point (barring
//! subnormal results), so against unrotated storage rotation changes only
//! the rounding of the butterflies' additions, the dot products over the
//! rotated values and the quantization error.

/// Values per rotated block (one quantization group of the split caches).
pub(crate) const BLOCK: usize = 32;

/// The kinds of block with their own sign pattern `D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Block {
    /// A key's temporal half (elements 0..32 of a head).
    Temporal,
    /// A key's spatial half (elements 32..64 of a head).
    Spatial,
    /// Value elements 0..32.
    ValueLow,
    /// Value elements 32..64.
    ValueHigh,
}

/// SplitMix64's output function (Steele, Lea and Flood, 2014).
const fn splitmix64(state: u64) -> u64 {
    let z = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The generator's seed: "KVROTATE" in ASCII.
const SEED: u64 = u64::from_be_bytes(*b"KVROTATE");

/// The sign patterns `D` of [`Block`]'s kinds in declaration order (bit `i`
/// set: element `i` is negated): the low 32 bits of the first four SplitMix64
/// outputs from [`SEED`] (14 to 19 of 32 elements negated). Any fixed
/// patterns work; they keep structured inputs (a constant block, a Walsh
/// function) from lining up with the Hadamard basis, and differ per kind so
/// that the kinds' errors do not correlate.
const SIGNS: [u32; 4] = {
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut signs = [0; 4];
    let mut k = 0;
    while k < 4 {
        signs[k] = splitmix64(SEED.wrapping_add((k as u64 + 1).wrapping_mul(GAMMA))) as u32;
        k += 1;
    }
    signs
};

impl Block {
    fn signs(self) -> u32 {
        SIGNS[self as usize]
    }
}

/// `x <- H x`: the unnormalized fast Walsh-Hadamard transform in Sylvester
/// order (`H[i][j] = (-1)^popcount(i & j)`), five stages of FP32 butterflies.
#[inline]
fn fwht(x: &mut [f32; BLOCK]) {
    let mut half = 1;
    while half < BLOCK {
        for start in (0..BLOCK).step_by(2 * half) {
            for i in start..start + half {
                let (a, b) = (x[i], x[i + half]);
                x[i] = a + b;
                x[i + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// `x <- D x`: flips the sign bits `signs` selects (exact, also for NaN).
#[inline]
fn flip(x: &mut [f32; BLOCK], signs: u32) {
    for (i, v) in x.iter_mut().enumerate() {
        *v = f32::from_bits(v.to_bits() ^ (((signs >> i) & 1) << 31));
    }
}

/// `x <- 2^-5 x` (`H H = 32 I`), exact barring subnormal results.
#[inline]
fn scale(x: &mut [f32; BLOCK]) {
    for v in x.iter_mut() {
        *v *= 1.0 / BLOCK as f32;
    }
}

/// The stored form of a key or value block: `x <- H D x`.
#[inline]
pub(crate) fn rotate(x: &mut [f32; BLOCK], block: Block) {
    flip(x, block.signs());
    fwht(x);
}

/// The query form of a key block: `x <- 2^-5 H D x`, so that
/// `rotate_query(q) · rotate(k) = q · k` in real arithmetic.
#[inline]
pub(crate) fn rotate_query(x: &mut [f32; BLOCK], block: Block) {
    rotate(x, block);
    scale(x);
}

/// The inverse of [`rotate`], for attention output blocks: `x <- 2^-5 D H x`.
#[inline]
pub(crate) fn unrotate(x: &mut [f32; BLOCK], block: Block) {
    fwht(x);
    flip(x, block.signs());
    scale(x);
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [Block; 4] = [Block::Temporal, Block::Spatial, Block::ValueLow, Block::ValueHigh];

    /// Deterministic values in [-1, 1) with no short period.
    fn values(seed: u64) -> [f32; BLOCK] {
        std::array::from_fn(|i| (splitmix64(seed * 64 + i as u64) >> 40) as f32 / (1 << 23) as f32 - 1.0)
    }

    #[test]
    fn fwht_is_the_sylvester_hadamard_matrix() {
        for j in 0..BLOCK {
            let mut x = [0.0; BLOCK];
            x[j] = 1.0;
            fwht(&mut x);
            for (i, &h) in x.iter().enumerate() {
                let expected = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                assert_eq!(h, expected, "H[{i}][{j}]");
            }
        }
    }

    #[test]
    fn signs_come_from_the_documented_seed_and_differ_per_kind() {
        assert_eq!(SIGNS, [0x04dc_f6b7, 0x871f_5f8b, 0x489c_e13c, 0x4a81_cbcb]);
        for (k, a) in SIGNS.iter().enumerate() {
            assert!((8..=24).contains(&a.count_ones()), "unbalanced pattern {a:#x}");
            assert!(SIGNS[k + 1..].iter().all(|b| a != b));
        }
    }

    #[test]
    fn unrotate_inverts_rotate() {
        for block in KINDS {
            // Small integers: every butterfly sum is exact, so the round trip
            // restores the input exactly (a zero may come back as -0).
            let integers: [f32; BLOCK] = std::array::from_fn(|i| ((i * 37 + 11) % 23) as f32 - 11.0);
            let mut x = integers;
            rotate(&mut x, block);
            assert!(x.iter().all(|v| v.fract() == 0.0));
            unrotate(&mut x, block);
            assert_eq!(x, integers, "{block:?}");
            // General values: within FP32 rounding (five stages of additions
            // each way; values of magnitude ~1, sums of magnitude up to ~32).
            for seed in 0..64 {
                let original = values(seed);
                let mut x = original;
                rotate(&mut x, block);
                unrotate(&mut x, block);
                for (a, b) in x.iter().zip(&original) {
                    assert!((a - b).abs() <= 1e-5, "{block:?} seed {seed}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn rotated_query_dots_equal_the_original_dots() {
        let dot = |a: &[f32; BLOCK], b: &[f32; BLOCK]| a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>();
        for block in KINDS {
            for seed in 0..64 {
                let (q, k) = (values(2 * seed), values(2 * seed + 1));
                let (mut rq, mut rk) = (q, k);
                rotate_query(&mut rq, block);
                rotate(&mut rk, block);
                let (expected, rotated) = (dot(&q, &k), dot(&rq, &rk));
                // Only the butterflies round (a few ulps of rotated elements
                // of magnitude up to 32), against dots of magnitude up to 11.
                assert!(
                    (rotated - expected).abs() <= 1e-5,
                    "{block:?} seed {seed}: {rotated} vs {expected}"
                );
                // The rotated query is exactly 2^-5 times the stored form.
                let mut stored = q;
                rotate(&mut stored, block);
                assert_eq!(rq.map(f32::to_bits), stored.map(|v| (v / 32.0).to_bits()));
            }
        }
    }

    #[test]
    fn a_nonfinite_value_spreads_over_its_block() {
        let mut x = values(3);
        x[5] = f32::NAN;
        rotate(&mut x, Block::ValueLow);
        assert!(x.iter().all(|v| v.is_nan()));
        let mut x = values(4);
        x[0] = f32::INFINITY;
        rotate(&mut x, Block::Temporal);
        assert!(x.iter().all(|v| !v.is_finite()));
    }
}
