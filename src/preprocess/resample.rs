//! Pillow's resampling, ported bit for bit: the bicubic and bilinear passes
//! of the first and second resize, the 16-bit and 8-bit gray variants, and
//! the nearest-neighbour and premultiplied-alpha first resizes.
//!
//! The resampling coefficients and fixed-point arithmetic below are adapted from
//! Pillow 11.3.0's src/libImaging/Resample.c:
//! https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c
//! The I;16 rounding, the pass order for very tall images (src/PIL/Image.py) and
//! the nearest-neighbour positions (src/libImaging/Geometry.c) follow Pillow
//! 12.3.0, the pinned processor's version.
//!
//! Pillow / PIL copyright and permission notice:
//! Copyright © 1997-2011 by Secret Labs AB
//! Copyright © 1995-2011 by Fredrik Lundh and contributors
//! Copyright © 2010 by Jeffrey A. Clark and contributors
//!
//! By obtaining, using, and/or copying this software and/or its associated
//! documentation, you agree that you have read, understood, and will comply
//! with the following terms and conditions:
//!
//! Permission to use, copy, modify and distribute this software and its
//! documentation for any purpose and without fee is hereby granted,
//! provided that the above copyright notice appears in all copies, and that
//! both that copyright notice and this permission notice appear in supporting
//! documentation, and that the name of Secret Labs AB or the author not be
//! used in advertising or publicity pertaining to distribution of the software
//! without specific, written prior permission.
//!
//! SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
//! SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
//! IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
//! INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
//! LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
//! OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
//! PERFORMANCE OF THIS SOFTWARE.

use anyhow::{Result, ensure};
use image::{RgbImage, RgbaImage};

const PRECISION_BITS: u32 = 22;

/// Pillow's nearest-neighbour resize, which `Image.resize` forces for modes
/// P and 1: every output pixel copies the source pixel at its row's and
/// column's [`nearest_positions`], so the result is bitwise Pillow's.
pub(super) fn resize_nearest(input: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    let columns = nearest_positions(input.width(), width)?;
    let rows = nearest_positions(input.height(), height)?;
    Ok(RgbImage::from_fn(width, height, |x, y| {
        *input.get_pixel(columns[x as usize], rows[y as usize])
    }))
}

/// The source index of each of `output` pixels along one axis of Pillow's
/// nearest-neighbour resize from `input` pixels (Pillow 12.3.0): `_resize`
/// (src/_imaging.c) passes `ImagingScaleAffine` (src/libImaging/Geometry.c)
/// the step `a = (double)(box[2] - box[0]) / xsize`, where the box is a C
/// `float[4]` (so `input as f32 as f64 / output`), and the offset `box[0] = 0`;
/// output pixel `k` reads source index `(int)p_k` with `p_0 = a * 0.5` and
/// `p_{k+1} = p_k + a`. This running f64 sum, in this operation order, is not
/// the direct product `(k + 0.5) * a`: where that product is an integer the
/// two can truncate to neighbouring indices (2048 -> 1536 differs in 303 of
/// 1536 positions). An unchanged size gives the identity, as Pillow's copy does.
///
/// Pillow leaves an output pixel whose index falls outside the source at zero.
/// That cannot happen here: the sum never decreases and after `k < output`
/// steps is within about `output * input * 2^-53` of `(k + 1/2) * input / output`,
/// which is at most `input - input / (2 * output)`; for inputs below 2^24
/// (where the `f32` box is exact) and outputs below about 2^26 every index is
/// in range. The last index is still checked, so any other case fails instead
/// of deviating silently.
pub(super) fn nearest_positions(input: u32, output: u32) -> Result<Vec<u32>> {
    let step = input as f32 as f64 / output as f64;
    let mut position = step * 0.5;
    let positions: Vec<u32> = (0..output)
        .map(|_| {
            // Pillow's COORD: the position is positive, so this truncates like `(int)`.
            let index = position as u32;
            position += step;
            index
        })
        .collect();
    ensure!(
        positions.last().is_none_or(|&last| last < input),
        "nearest-neighbour resize from {input} to {output} pixels reads outside the source"
    );
    Ok(positions)
}

/// Pillow's bicubic resize of an image with alpha: the colour premultiplied
/// by alpha and the alpha resampled separately, then divided back (Pillow's
/// RGBa round trip) before the RGB conversion drops the alpha.
pub(super) fn resize_premultiplied(input: &RgbaImage, width: u32, height: u32) -> RgbImage {
    let rgb = RgbImage::from_fn(input.width(), input.height(), |x, y| {
        let pixel = input.get_pixel(x, y);
        image::Rgb(std::array::from_fn(|channel| {
            let product = pixel[channel] as u32 * pixel[3] as u32 + 128;
            ((product + (product >> 8)) >> 8) as u8
        }))
    });
    let alpha = RgbImage::from_fn(input.width(), input.height(), |x, y| {
        image::Rgb([input.get_pixel(x, y)[3]; 3])
    });
    let rgb = resize_bicubic(&rgb, width, height);
    let alpha = resize_bicubic(&alpha, width, height);
    RgbImage::from_fn(width, height, |x, y| {
        let color = rgb.get_pixel(x, y);
        let a = alpha.get_pixel(x, y)[0];
        image::Rgb(color.0.map(|v| {
            if a == 0 || a == 255 {
                v
            } else {
                (255 * v as u32 / a as u32).min(255) as u8
            }
        }))
    })
}

struct Coefficients {
    start: usize,
    weights: Vec<i32>,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

fn triangle(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Pillow's resampling filters (`Image.BILINEAR`, `Image.BICUBIC`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Filter {
    Bilinear,
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
        }
    }
    fn weight(self, x: f64) -> f64 {
        match self {
            Filter::Bilinear => triangle(x),
            Filter::Bicubic => cubic(x),
        }
    }
}

fn coefficients_f64(input: u32, output: u32, filter: Filter) -> Vec<(usize, Vec<f64>)> {
    // Pillow stores the crop box as floats even for a full-image resize.
    let scale = input as f32 as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let inverse_scale = 1.0 / filter_scale;
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input as usize);
            let mut weights: Vec<f64> = (start..end)
                .map(|x| filter.weight((x as f64 - center + 0.5) * inverse_scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
            (start, weights)
        })
        .collect()
}

fn coefficients(input: u32, output: u32, filter: Filter) -> Vec<Coefficients> {
    coefficients_f64(input, output, filter)
        .into_iter()
        .map(|(start, weights)| Coefficients {
            start,
            weights: weights
                .into_iter()
                .map(|weight| {
                    let bias = if weight < 0.0 { -0.5 } else { 0.5 };
                    (bias + weight * (1_u32 << PRECISION_BITS) as f64) as i32
                })
                .collect(),
        })
        .collect()
}

/// Pillow's bicubic resize of a 16-bit gray (I;16) image.
pub(super) fn resize_gray16(input: &[u16], input_width: u32, input_height: u32, width: u32, height: u32) -> Vec<u16> {
    // Pillow's I;16 kernel accumulates FP64 and rounds to 16-bit after each
    // separable pass, in the order `resize_bicubic` describes. Conversion to
    // RGB subsequently clips to 255, not /257.
    if tall_first(input_width, input_height, height) {
        let vertical = gray16_vertical(input, input_width, input_height, height);
        return if width == input_width {
            vertical
        } else {
            gray16_horizontal(&vertical, input_width, height, width)
        };
    }
    let horizontal = if width == input_width {
        input.to_vec()
    } else {
        gray16_horizontal(input, input_width, input_height, width)
    };
    if height == input_height {
        horizontal
    } else {
        gray16_vertical(&horizontal, width, input_height, height)
    }
}

/// The pinned Pillow 12.3.0's rounding of an FP64 I;16 resampling sum to 16
/// bits, `CLIP16(ROUND_UP(ss))` (clamping each byte instead, as Pillow 11
/// did, differs when a sum overshoots 65535).
fn quantize16(value: f64) -> u16 {
    ((value + if value < 0.0 { -0.5 } else { 0.5 }) as i64).clamp(0, 65535) as u16
}

/// The horizontal I;16 bicubic pass to `width` columns.
fn gray16_horizontal(input: &[u16], input_width: u32, input_height: u32, width: u32) -> Vec<u16> {
    let coeff = coefficients_f64(input_width, width, Filter::Bicubic);
    let mut output = vec![0u16; width as usize * input_height as usize];
    for y in 0..input_height as usize {
        for (x, (start, weights)) in coeff.iter().enumerate() {
            let value: f64 = weights
                .iter()
                .enumerate()
                .map(|(offset, &weight)| input[y * input_width as usize + start + offset] as f64 * weight)
                .sum();
            output[y * width as usize + x] = quantize16(value);
        }
    }
    output
}

/// The vertical I;16 bicubic pass of a `width`-column image to `height` rows.
fn gray16_vertical(input: &[u16], width: u32, input_height: u32, height: u32) -> Vec<u16> {
    let mut output = vec![0u16; width as usize * height as usize];
    for (y, (start, weights)) in coefficients_f64(input_height, height, Filter::Bicubic)
        .iter()
        .enumerate()
    {
        for x in 0..width as usize {
            let value: f64 = weights
                .iter()
                .enumerate()
                .map(|(offset, &weight)| input[(start + offset) * width as usize + x] as f64 * weight)
                .sum();
            output[y * width as usize + x] = quantize16(value);
        }
    }
    output
}

fn clip(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow-compatible 8-bit resampling of a grayscale (mode L) image:
/// the horizontal pass, rounded to 8 bits, then the vertical pass.
pub(crate) fn resize_gray(
    input: &[u8],
    input_width: u32,
    input_height: u32,
    width: u32,
    height: u32,
    filter: Filter,
) -> Vec<u8> {
    // The pass order `resize_bicubic` describes (Pillow 12.3.0's tall-page
    // rule).
    if tall_first(input_width, input_height, height) {
        let vertical = gray8_vertical(input, input_width, input_height, height, filter);
        return if width == input_width {
            vertical
        } else {
            gray8_horizontal(&vertical, input_width, height, width, filter)
        };
    }
    let horizontal = if width == input_width {
        input.to_vec()
    } else {
        gray8_horizontal(input, input_width, input_height, width, filter)
    };
    if height == input_height {
        horizontal
    } else {
        gray8_vertical(&horizontal, width, input_height, height, filter)
    }
}

/// Pillow's 8-bit kernels accumulate in 32 bits: 255 times the positive
/// lobe (at most about 1.2) in 22-bit fixed point stays below 2^31.
fn clip8(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

/// The horizontal pass of an 8-bit gray image to `width` columns.
fn gray8_horizontal(input: &[u8], input_width: u32, input_height: u32, width: u32, filter: Filter) -> Vec<u8> {
    let (iw, w) = (input_width as usize, width as usize);
    let weights = coefficients(input_width, width, filter);
    let mut out = vec![0u8; w * input_height as usize];
    for (source, row) in input.chunks_exact(iw).zip(out.chunks_exact_mut(w)) {
        for (target, coeff) in row.iter_mut().zip(&weights) {
            let mut sum = 1_i32 << (PRECISION_BITS - 1);
            for (&value, &weight) in source[coeff.start..].iter().zip(&coeff.weights) {
                sum += value as i32 * weight;
            }
            *target = clip8(sum);
        }
    }
    out
}

/// The vertical pass of an 8-bit gray image `width` columns wide to `height`
/// rows.
fn gray8_vertical(input: &[u8], width: u32, input_height: u32, height: u32, filter: Filter) -> Vec<u8> {
    let w = width as usize;
    let mut out = vec![0u8; w * height as usize];
    let mut sums = vec![0_i32; w];
    for (row, coeff) in out.chunks_exact_mut(w).zip(&coefficients(input_height, height, filter)) {
        sums.fill(1 << (PRECISION_BITS - 1));
        for (offset, &weight) in coeff.weights.iter().enumerate() {
            let source = &input[(coeff.start + offset) * w..][..w];
            for (sum, &value) in sums.iter_mut().zip(source) {
                *sum += value as i32 * weight;
            }
        }
        for (target, &sum) in row.iter_mut().zip(&sums) {
            *target = clip8(sum);
        }
    }
    out
}

/// Pillow-compatible bicubic resampling of a complete RGB image: a horizontal
/// then a vertical pass, each rounded to uint8. The pinned Pillow 12.3.0's
/// `Image.resize` (Image.py) reverses the order for an image more than 100
/// times taller than wide whose height shrinks ([`tall_first`]): a
/// vertical-only core resize, then a horizontal-only one. The rounding between
/// the passes makes the order visible, so the same order is kept here.
pub(super) fn resize_bicubic(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    if tall_first(input.width(), input.height(), height) {
        let vertical = bicubic_vertical(input, height);
        return if width == input.width() {
            vertical
        } else {
            bicubic_horizontal(&vertical, width)
        };
    }
    let horizontal = if width == input.width() {
        input.clone()
    } else {
        bicubic_horizontal(input, width)
    };
    if height == input.height() {
        horizontal
    } else {
        bicubic_vertical(&horizontal, height)
    }
}

/// Whether the pinned Pillow 12.3.0's `Image.resize` resamples an image
/// vertically first: its height is more than 100 times its width and shrinks.
pub(super) fn tall_first(width: u32, height: u32, new_height: u32) -> bool {
    height as u64 > width as u64 * 100 && new_height < height
}

/// The horizontal bicubic pass of an RGB image to `width` columns.
fn bicubic_horizontal(input: &RgbImage, width: u32) -> RgbImage {
    let weights = coefficients(input.width(), width, Filter::Bicubic);
    let mut out = RgbImage::new(width, input.height());
    let src = input.as_raw();
    for (row_index, row) in out.as_mut().chunks_exact_mut(width as usize * 3).enumerate() {
        for (pixel, coeff) in row.chunks_exact_mut(3).zip(&weights) {
            let mut sums = [1_i64 << (PRECISION_BITS - 1); 3];
            let base = (row_index * input.width() as usize + coeff.start) * 3;
            for (offset, &weight) in coeff.weights.iter().enumerate() {
                for channel in 0..3 {
                    sums[channel] += src[base + offset * 3 + channel] as i64 * weight as i64;
                }
            }
            for channel in 0..3 {
                pixel[channel] = clip(sums[channel]);
            }
        }
    }
    out
}

/// The vertical bicubic pass of an RGB image to `height` rows.
fn bicubic_vertical(input: &RgbImage, height: u32) -> RgbImage {
    let weights = coefficients(input.height(), height, Filter::Bicubic);
    let mut out = RgbImage::new(input.width(), height);
    let stride = input.width() as usize * 3;
    for (row, coeff) in out.as_mut().chunks_exact_mut(stride).zip(&weights) {
        for (column, target) in row.iter_mut().enumerate() {
            let mut sum = 1_i64 << (PRECISION_BITS - 1);
            for (offset, &weight) in coeff.weights.iter().enumerate() {
                sum += input.as_raw()[(coeff.start + offset) * stride + column] as i64 * weight as i64;
            }
            *target = clip(sum);
        }
    }
    out
}
