//! Isolated W4A32 decode experiment; never part of the production runner.
//! Each dequantized vector is shared by 1/2/4/8 live activation rows.
//! The settled Q4 bytes/scales remain unchanged. No heap scratch is needed.

use crate::q4_reference::Q4Linear;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Avx2,
}

pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Both backends require the default IEEE rounding/subnormal environment.
/// This prevents caller-enabled FTZ/DAZ from silently changing dequantization.
fn check_fp_environment() -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut control = 0u32;
        // SAFETY: x86_64 has SSE; STMXCSR writes exactly this valid stack u32.
        unsafe {
            std::arch::asm!("stmxcsr [{ptr}]", ptr = in(reg) &mut control,
                options(nostack, preserves_flags));
        }
        if control & ((1 << 15) | (1 << 6) | (3 << 13)) != 0 {
            return Err("round-to-nearest and gradual underflow required (MXCSR FTZ/DAZ off)");
        }
    }
    Ok(())
}

/// Checked, allocation-free row-major [rows,K] × [N,K]^T -> [rows,N].
/// Error paths never allocate and never silently fall back to another backend.
/// Arithmetic overflow is reported after writing output; output is unspecified
/// on that error. Shapes/backend/input validation happens before output writes.
pub fn linear(
    weights: &Q4Linear,
    input: &[f32],
    rows: usize,
    output: &mut [f32],
    backend: Backend,
) -> Result<(), &'static str> {
    let (n, k) = weights.dimensions();
    if ![1, 2, 4, 8].contains(&rows) {
        return Err("decode rows must be 1, 2, 4, or 8");
    }
    if ![64, 128].contains(&weights.group_size()) {
        return Err("decode group size must be 64 or 128");
    }
    if rows.checked_mul(k) != Some(input.len()) || rows.checked_mul(n) != Some(output.len()) {
        return Err("activation/output shape mismatch or overflow");
    }
    if input.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite activation");
    }
    if backend == Backend::Avx2 && !avx2_available() {
        return Err("AVX2 and FMA are required for the explicit AVX2 backend");
    }
    check_fp_environment()?;
    match backend {
        Backend::Scalar => weights.linear_f32(input, rows, output),
        Backend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: runtime feature guard above; shape checks and the private
            // checked Q4 constructor guarantee valid packed/scales/input slices.
            unsafe {
                linear_avx2(weights, input, rows, output)
            }
            #[cfg(not(target_arch = "x86_64"))]
            return Err("AVX2 backend is only implemented for x86_64");
        }
    }
    if output.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite accumulation");
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_avx2(weights: &Q4Linear, input: &[f32], rows: usize, output: &mut [f32]) {
    use std::arch::x86_64::*;
    let (n, k) = weights.dimensions();
    let group_size = weights.group_size();
    let groups = k.div_ceil(group_size);
    let packed_stride = k.div_ceil(2);
    let bytes = weights.packed_codes();
    let scales = weights.scales();
    let mask = _mm_set1_epi8(15);
    let sign = _mm_set1_epi8(8);
    for channel in 0..n {
        let mut sums = [_mm256_setzero_ps(); 8];
        let mut tail_sums = [0.0f32; 8];
        for group in 0..groups {
            let begin = group * group_size;
            let end = begin.saturating_add(group_size).min(k);
            let scale = scales[channel * groups + group];
            let scale_vector = _mm256_set1_ps(scale);
            let mut column = begin;
            while end - column >= 16 {
                // SAFETY: column is even and 16 values remain in this row/group;
                // exactly 8 valid packed bytes and 16 valid activations are read.
                // Unaligned loads allow arbitrary caller slice alignment.
                let packed = unsafe {
                    _mm_loadl_epi64(
                        bytes
                            .as_ptr()
                            .add(channel * packed_stride + column / 2)
                            .cast(),
                    )
                };
                let low = _mm_and_si128(packed, mask);
                let high = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
                let nibbles = _mm_unpacklo_epi8(low, high);
                let signed = _mm_sub_epi8(_mm_xor_si128(nibbles, sign), sign);
                // Separate FP32 multiplication matches the settled per-weight
                // dequantization. Never multiply activations by the scale.
                let w0 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(signed)),
                    scale_vector,
                );
                let w1 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(signed))),
                    scale_vector,
                );
                for row in 0..rows {
                    let x = unsafe { input.as_ptr().add(row * k + column) };
                    let a = unsafe { _mm256_loadu_ps(x) };
                    let b = unsafe { _mm256_loadu_ps(x.add(8)) };
                    sums[row] = _mm256_fmadd_ps(a, w0, sums[row]);
                    sums[row] = _mm256_fmadd_ps(b, w1, sums[row]);
                }
                column += 16;
            }
            // At most 15 final values. Each nibble is within this packed row;
            // the unused high nibble of an odd K is never treated as a weight.
            for column in column..end {
                let byte = bytes[channel * packed_stride + column / 2];
                let code = ((byte >> (4 * (column % 2))) & 15) as i8;
                let code = if code >= 8 { code - 16 } else { code };
                let value = code as f32 * scale;
                for row in 0..rows {
                    tail_sums[row] = input[row * k + column].mul_add(value, tail_sums[row]);
                }
            }
        }
        for row in 0..rows {
            let mut lanes = [0.0f32; 8];
            // SAFETY: stack array has room for exactly 8 FP32 values.
            unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sums[row]) };
            let left = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            let right = (lanes[4] + lanes[5]) + (lanes[6] + lanes[7]);
            output[row * n + channel] = (left + right) + tail_sums[row];
        }
    }
}
