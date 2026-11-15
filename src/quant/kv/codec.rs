//! Scalar quantization of the split cache: BF16 scales, 8- and 16-bit codes.
use super::rotation::{BLOCK, Block};

/// Allocate zeroed KV tail storage, returning an error if allocation fails.
pub(super) fn zeroed_tail<T: bytemuck::Zeroable>(len: usize) -> anyhow::Result<Vec<T>> {
    bytemuck::allocation::try_zeroed_vec(len).map_err(|()| anyhow::anyhow!("cannot allocate split KV tail"))
}

/// The 8-bit step of `values`: their absmax over 127, at least the smallest
/// subnormal unless every value is 0.
pub(super) fn q8_scale(values: &[f32]) -> f32 {
    q8_scale_of_max(values.iter().fold(0.0_f32, |a, &b| a.max(b.abs())))
}

fn q8_scale_of_max(max: f32) -> f32 {
    if max == 0.0 {
        0.0
    } else {
        ((max as f64 / 127.0) as f32).max(f32::from_bits(1))
    }
}

/// The 16-bit step of `values`: their absmax over 32,767, at least the
/// smallest subnormal unless every value is 0.
pub(super) fn q16_scale(values: &[f32]) -> f32 {
    let max = values.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    if max == 0.0 {
        0.0
    } else {
        ((max as f64 / 32767.0) as f32).max(f32::from_bits(1))
    }
}

/// The smallest BF16 value at or above `scale` (a positive finite scale, 0,
/// or NaN), as BF16 bits. Codes quantized against it never exceed the range,
/// and the step grows by at most 2^-8 relative.
pub(super) fn bf16_up(scale: f32) -> u16 {
    if scale.is_nan() {
        return 0x7FC0;
    }
    let bits = scale.to_bits();
    let high = (bits >> 16) as u16;
    if bits & 0xFFFF != 0 { high + 1 } else { high }
}

/// BF16 bits widened exactly to FP32.
#[inline(always)]
pub(super) fn bf16_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `x` clamped to `±limit` (an integer below 2^51) and rounded to nearest
/// even: `x.round_ties_even().clamp(-limit, limit)` (NaN stays NaN), without
/// the libm call `round_ties_even` costs on baseline x86-64. Adding 1.5 *
/// 2^52 leaves one integer step per ulp, so the addition rounds to an
/// integer, ties to even (the default rounding mode), and the subtraction
/// is exact. Clamping first is the same: a value past the limit rounds to
/// at least the limit.
#[inline(always)]
pub(super) fn round_clamped(x: f64, limit: f64) -> f64 {
    const SHIFT: f64 = 6_755_399_441_055_744.0;
    (x.clamp(-limit, limit) + SHIFT) - SHIFT
}

/// `value / scale` rounded to nearest even and clamped to ±32,767; 0 for a
/// zero scale.
pub(super) fn q16_code(value: f32, scale: f32) -> i16 {
    if scale == 0.0 {
        0
    } else {
        round_clamped(value as f64 / scale as f64, 32767.0) as i16
    }
}

/// `value / scale` rounded to nearest even and clamped to ±127; 0 for a zero
/// scale.
pub(super) fn q8_code(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        0
    } else {
        round_clamped(value as f64 / scale as f64, 127.0) as i8
    }
}

/// The 4-bit step of `values`: their absmax over 7, at least the smallest
/// subnormal unless every value is 0.
fn q4_scale(values: &[f32]) -> f32 {
    let max = values.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    if max == 0.0 {
        0.0
    } else {
        ((max as f64 / 7.0) as f32).max(f32::from_bits(1))
    }
}

/// `value / scale` rounded to nearest even and clamped to ±7; 0 for a zero
/// scale.
fn q4_code(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        0
    } else {
        round_clamped(value as f64 / scale as f64, 7.0) as i8
    }
}

/// Encodes one 32-element block as 4-bit codes, two per byte (element `2i`
/// in the low nibble), against its BF16 absmax scale, and returns the scale's
/// bits. A non-finite value makes the scale NaN, so every element of the
/// block decodes to NaN.
pub(super) fn q4_block(values: &[f32], codes: &mut [u8]) -> u16 {
    debug_assert!(values.len() == 32 && codes.len() == 16);
    let stored = bf16_up(if values.iter().all(|x| x.is_finite()) {
        q4_scale(values)
    } else {
        f32::NAN
    });
    let scale = bf16_f32(stored);
    for (byte, pair) in codes.iter_mut().zip(values.chunks_exact(2)) {
        *byte = ((q4_code(pair[0], scale) as u8) & 0x0F) | ((q4_code(pair[1], scale) as u8) << 4);
    }
    stored
}

/// The signed 4-bit code in the low (`high == 0`) or high (`high == 1`)
/// nibble of `byte`, sign-extended.
#[inline(always)]
pub(super) fn nibble(byte: u8, high: usize) -> i8 {
    ((byte << (4 - 4 * high)) as i8) >> 4
}

/// Applies `f` to every 32-element block of `values` with its kind; `kinds`
/// repeats along `values` (one record, or one head after another).
pub(super) fn each_block(values: &mut [f32], kinds: &[Block], f: impl Fn(&mut [f32; BLOCK], Block)) {
    let (blocks, rest) = values.as_chunks_mut::<BLOCK>();
    debug_assert!(rest.is_empty() && blocks.len().is_multiple_of(kinds.len()));
    for (block, &kind) in blocks.iter_mut().zip(kinds.iter().cycle()) {
        f(block, kind);
    }
}
