//! Scalar quantization of the split cache: BF16 scales, 8- and 16-bit codes.
use anyhow::Result;

/// A zero-filled buffer, failing instead of aborting when memory is short.
pub(super) fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)?;
    data.resize(len, T::default());
    Ok(data)
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

/// `value / scale` rounded to nearest even and clamped to ±32,767; 0 for a
/// zero scale.
pub(super) fn q16_code(value: f32, scale: f32) -> i16 {
    if scale == 0.0 {
        0
    } else {
        (value as f64 / scale as f64).round_ties_even().clamp(-32767.0, 32767.0) as i16
    }
}

/// `value / scale` rounded to nearest even and clamped to ±127; 0 for a zero
/// scale.
pub(super) fn q8_code(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        0
    } else {
        (value as f64 / scale as f64).round_ties_even().clamp(-127.0, 127.0) as i8
    }
}
