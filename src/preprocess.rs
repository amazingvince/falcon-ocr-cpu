//! Image preparation for the pinned Falcon-OCR v1.5 processor.
//!
//! The two resizing stages are intentional. Each performs uint8 bicubic
//! resampling, including quantization between the horizontal and vertical passes.
//! Replacing them with a single float resize changes the model input.
//!
//! The resampling coefficients and fixed-point arithmetic below are adapted from
//! Pillow 11.3.0's src/libImaging/Resample.c:
//! https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c
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

use anyhow::{Context, Result, ensure};
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use std::path::Path;

pub const PATCH_SIZE: usize = 16;
pub const PATCH_VALUES: usize = PATCH_SIZE * PATCH_SIZE * 3;
const MIN_PIXELS: u64 = 56 * 56;
const MAX_PIXELS: u64 = 28 * 28 * 1280 * 10;
const PRECISION_BITS: u32 = 22;

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub width: usize,
    pub height: usize,
    /// Grid row-major patches, each containing row-major RGB pixels.
    pub patches: Vec<f32>,
    /// Per-patch [height, width] positions. FP32 sqrt uses IEEE rounding; the
    /// reference PyTorch MKL sqrt can differ by an ULP (see frozen fixtures).
    pub positions_hw: Vec<[f32; 2]>,
}

pub fn prepare_rgb(
    image: &RgbImage,
    min_dimension: u32,
    max_dimension: u32,
) -> Result<PreparedImage> {
    let (width, height) = image.dimensions();
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    ensure!(
        min_dimension > 0 && min_dimension <= max_dimension,
        "expected 0 < min_dimension <= max_dimension"
    );

    let (first_width, first_height) =
        bounded_dimensions(width, height, min_dimension, max_dimension);
    ensure!(
        first_width > 0 && first_height > 0,
        "the upstream aspect-preserving resize produces a zero dimension"
    );
    let first = resize_bicubic(image, first_width, first_height);
    prepare_resized_rgb(&first)
}

/// Decode a PNG or JPEG and preserve Pillow's source-mode first resize.
/// Metadata orientation and color profiles are not applied, matching upstream.
pub fn prepare_file(path: &Path, min_dimension: u32, max_dimension: u32) -> Result<PreparedImage> {
    prepare_file_timed(path, min_dimension, max_dimension).map(|(image, _)| image)
}

/// Return prepared image and milliseconds spent reading/decoding the file,
/// excluding resizing, normalization, and patch packing.
pub fn prepare_file_timed(
    path: &Path,
    min_dimension: u32,
    max_dimension: u32,
) -> Result<(PreparedImage, f64)> {
    let decode_started = std::time::Instant::now();
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    let format = image::guess_format(&bytes)?;
    ensure!(
        matches!(format, ImageFormat::Png | ImageFormat::Jpeg),
        "only PNG and JPEG files are supported"
    );
    let cmyk = if format == ImageFormat::Jpeg && jpeg_components(&bytes)? == 4 {
        Some(decode_jpeg_cmyk(&bytes)?)
    } else {
        None
    };
    let decoded = if let Some(cmyk) = &cmyk {
        DynamicImage::ImageRgb8(cmyk_to_rgb(cmyk))
    } else if format == ImageFormat::Jpeg {
        DynamicImage::ImageRgb8(decode_jpeg_rgb(&bytes)?)
    } else {
        image::load_from_memory_with_format(&bytes, format)?
    };
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let width = decoded.width();
    let height = decoded.height();
    ensure!(
        min_dimension > 0 && min_dimension <= max_dimension,
        "expected 0 < min_dimension <= max_dimension"
    );
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    let (w, h) = bounded_dimensions(width, height, min_dimension, max_dimension);
    ensure!(
        w > 0 && h > 0,
        "the upstream aspect-preserving resize produces a zero dimension"
    );
    let resize = (w, h) != (width, height);
    let first = if let Some(cmyk) = &cmyk {
        let colors = RgbImage::from_fn(width, height, |x, y| {
            let p = cmyk.get_pixel(x, y);
            image::Rgb([p[0], p[1], p[2]])
        });
        let blacks = RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([cmyk.get_pixel(x, y)[3]; 3])
        });
        let colors = resize_bicubic(&colors, w, h);
        let blacks = resize_bicubic(&blacks, w, h);
        let resized = RgbaImage::from_fn(w, h, |x, y| {
            let c = colors.get_pixel(x, y);
            image::Rgba([c[0], c[1], c[2], blacks.get_pixel(x, y)[0]])
        });
        cmyk_to_rgb(&resized)
    } else if format == ImageFormat::Png {
        ensure!(
            bytes.len() >= 29 && &bytes[12..16] == b"IHDR",
            "missing PNG IHDR"
        );
        let (depth, color) = (bytes[24], bytes[25]);
        if color == 0 && depth == 16 {
            let values = decoded.to_luma16();
            let values = resize_gray16(values.as_raw(), width, height, w, h);
            RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([values[(y * w + x) as usize].min(255) as u8; 3])
            })
        } else if color == 3 || (color == 0 && depth == 1) {
            let rgb = decoded.to_rgb8();
            // Pillow always uses nearest for modes P and 1 on this first call.
            resize_nearest(&rgb, w, h)
        } else if matches!(color, 4 | 6) && resize {
            let rgba = pillow_rgba8(&decoded, depth);
            resize_premultiplied(&rgba, w, h)
        } else {
            // tRNS metadata on RGB/L is ignored by PIL.convert("RGB"); do not
            // mistakenly treat expanded tRNS pixels as an original RGBA mode.
            let rgba = pillow_rgba8(&decoded, depth);
            let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
            resize_bicubic(&rgb, w, h)
        }
    } else {
        resize_bicubic(&decoded.to_rgb8(), w, h)
    };
    Ok((prepare_resized_rgb(&first)?, decode_ms))
}

fn decode_jpeg_rgb(bytes: &[u8]) -> Result<RgbImage> {
    if jpeg_components(bytes)? == 4 {
        return Ok(cmyk_to_rgb(&decode_jpeg_cmyk(bytes)?));
    }
    // libjpeg-turbo uses the accurate integer IDCT and fancy chroma upsampling
    // by default, as does the pinned Pillow JPEG decoder.
    let decoded = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)?;
    ensure!(
        decoded.pitch == decoded.width * 3,
        "unexpected JPEG row stride"
    );
    RgbImage::from_raw(
        decoded.width.try_into()?,
        decoded.height.try_into()?,
        decoded.pixels,
    )
    .context("inconsistent JPEG dimensions")
}

fn decode_jpeg_cmyk(bytes: &[u8]) -> Result<RgbaImage> {
    let mut decoded = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::CMYK)?;
    ensure!(
        decoded.pitch == decoded.width * 4,
        "unexpected CMYK row stride"
    );
    // Pillow raw mode CMYK;I inverts libjpeg's CMYK samples before resizing.
    for value in &mut decoded.pixels {
        *value = 255 - *value;
    }
    RgbaImage::from_raw(
        decoded.width.try_into()?,
        decoded.height.try_into()?,
        decoded.pixels,
    )
    .context("inconsistent CMYK dimensions")
}

fn cmyk_to_rgb(source: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(source.width(), source.height(), |x, y| {
        let cmyk = source.get_pixel(x, y);
        let nonblack = 255 - cmyk[3] as u32;
        image::Rgb(std::array::from_fn(|channel| {
            let product = cmyk[channel] as u32 * nonblack + 128;
            (nonblack - ((product + (product >> 8)) >> 8)) as u8
        }))
    })
}

fn pillow_rgba8(decoded: &DynamicImage, depth: u8) -> RgbaImage {
    if depth != 16 {
        return decoded.to_rgba8();
    }
    // Pillow truncates the low byte of truecolor/alpha 16-bit PNG samples.
    // image::to_rgba8 instead rescales, which differs for some channel values.
    let source = decoded.to_rgba16();
    RgbaImage::from_fn(source.width(), source.height(), |x, y| {
        image::Rgba(source.get_pixel(x, y).0.map(|value| (value >> 8) as u8))
    })
}

fn resize_nearest(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    let scale_x = input.width() as f64 / width as f64;
    let scale_y = input.height() as f64 / height as f64;
    RgbImage::from_fn(width, height, |x, y| {
        let x = ((x as f64 + 0.5) * scale_x) as u32;
        let y = ((y as f64 + 0.5) * scale_y) as u32;
        *input.get_pixel(x.min(input.width() - 1), y.min(input.height() - 1))
    })
}

fn resize_premultiplied(input: &RgbaImage, width: u32, height: u32) -> RgbImage {
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

fn jpeg_components(bytes: &[u8]) -> Result<u8> {
    let mut offset = 2;
    while offset + 1 < bytes.len() {
        ensure!(bytes[offset] == 255, "invalid JPEG marker");
        while offset < bytes.len() && bytes[offset] == 255 {
            offset += 1;
        }
        let marker = *bytes.get(offset).context("truncated JPEG marker")?;
        offset += 1;
        if matches!(marker, 0xd8 | 0x01 | 0xd0..=0xd7) {
            continue;
        }
        ensure!(
            !matches!(marker, 0xd9 | 0xda),
            "JPEG frame header is missing"
        );
        let length_bytes: [u8; 2] = bytes
            .get(offset..offset + 2)
            .context("truncated JPEG length")?
            .try_into()?;
        let length = u16::from_be_bytes(length_bytes) as usize;
        ensure!(
            length >= 2 && offset + length <= bytes.len(),
            "invalid JPEG segment length"
        );
        if matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf) {
            ensure!(length >= 8, "truncated JPEG frame");
            return Ok(bytes[offset + 7]);
        }
        offset += length;
    }
    anyhow::bail!("JPEG frame header is missing")
}

fn prepare_resized_rgb(first: &RgbImage) -> Result<PreparedImage> {
    let (first_width, first_height) = first.dimensions();
    let (final_width, final_height) = aligned_dimensions(first_width, first_height)?;
    let resized = resize_bicubic(first, final_width, final_height);
    let width = final_width as usize;
    let height = final_height as usize;
    let grid_width = width / PATCH_SIZE;
    let grid_height = height / PATCH_SIZE;

    // Transformers rescales in FP64, casts to FP32, then normalizes in FP32.
    // A lookup preserves those rounding boundaries without per-pixel divisions.
    let normalized = std::array::from_fn::<_, 256, _>(|v| {
        let scaled = (v as f64 * (1.0 / 255.0)) as f32;
        (scaled - 0.5_f32) / 0.5_f32
    });
    let mut patches = Vec::with_capacity(width * height * 3);
    let raw = resized.as_raw();
    for patch_y in 0..grid_height {
        for patch_x in 0..grid_width {
            for y in 0..PATCH_SIZE {
                let start = ((patch_y * PATCH_SIZE + y) * width + patch_x * PATCH_SIZE) * 3;
                patches.extend(
                    raw[start..start + PATCH_SIZE * 3]
                        .iter()
                        .map(|&v| normalized[v as usize]),
                );
            }
        }
    }

    let xlim = (grid_width as f32 / grid_height as f32).sqrt();
    let ylim = (grid_height as f32 / grid_width as f32).sqrt();
    let xpos = torch_linspace(-xlim, xlim, grid_width);
    let ypos = torch_linspace(-ylim, ylim, grid_height);
    let mut positions_hw = Vec::with_capacity(grid_width * grid_height);
    for &h in &ypos {
        for &w in &xpos {
            positions_hw.push([h, w]);
        }
    }
    Ok(PreparedImage {
        width,
        height,
        patches,
        positions_hw,
    })
}

// Deliberately retain Python's operation order and truncation. Merely scaling
// by min(max/min_extent, ...) is not equivalent on all integer boundaries.
fn bounded_dimensions(width: u32, height: u32, minimum: u32, maximum: u32) -> (u32, u32) {
    if (minimum..=maximum).contains(&width) && (minimum..=maximum).contains(&height) {
        return (width, height);
    }
    let aspect = width as f64 / height as f64;
    let bound = if width < minimum || height < minimum {
        minimum
    } else {
        maximum
    };
    let (mut w, mut h) = if width < height {
        (bound, (bound as f64 / aspect) as u32)
    } else {
        ((bound as f64 * aspect) as u32, bound)
    };
    if w > maximum {
        w = maximum;
        h = (w as f64 / aspect) as u32;
    }
    if h > maximum {
        h = maximum;
        w = (h as f64 * aspect) as u32;
    }
    (w, h)
}

fn round_to_patch(value: u32) -> u32 {
    // round(value / 16) is ties-to-even in Python, not ties-away-from-zero.
    let quotient = value / PATCH_SIZE as u32;
    let remainder = value % PATCH_SIZE as u32;
    (quotient + u32::from(remainder > 8 || (remainder == 8 && quotient % 2 == 1))) * 16
}

fn aligned_dimensions(width: u32, height: u32) -> Result<(u32, u32)> {
    ensure!(
        width >= 16 && height >= 16,
        "image width and height must be at least one 16-pixel patch after resizing"
    );
    ensure!(
        width.max(height) as f64 / width.min(height) as f64 <= 200.0,
        "image absolute aspect ratio must be at most 200"
    );
    let (mut w, mut h) = (round_to_patch(width), round_to_patch(height));
    let pixels = w as u64 * h as u64;
    if pixels > MAX_PIXELS {
        let beta = ((height as f64 * width as f64) / MAX_PIXELS as f64).sqrt();
        h = ((height as f64 / beta / 16.0).floor() * 16.0) as u32;
        w = ((width as f64 / beta / 16.0).floor() * 16.0) as u32;
    } else if pixels < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (height as f64 * width as f64)).sqrt();
        h = ((height as f64 * beta / 16.0).ceil() * 16.0) as u32;
        w = ((width as f64 * beta / 16.0).ceil() * 16.0) as u32;
    }
    ensure!(w > 0 && h > 0, "smart resize produced an empty image");
    Ok((w, h))
}

fn torch_linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    // Explicit FMA matches the endpoint-anchored CPU reference while preserving
    // the same rounding in scalar/AVX builds and across Windows and Linux.
    (0..steps)
        .map(|i| {
            if i < steps / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            }
        })
        .collect()
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

fn coefficients_f64(input: u32, output: u32) -> Vec<(usize, Vec<f64>)> {
    // Pillow stores the crop box as floats even for a full-image resize.
    let scale = input as f32 as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let inverse_scale = 1.0 / filter_scale;
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input as usize);
            let mut weights: Vec<f64> = (start..end)
                .map(|x| cubic((x as f64 - center + 0.5) * inverse_scale))
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

fn coefficients(input: u32, output: u32) -> Vec<Coefficients> {
    coefficients_f64(input, output)
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

fn resize_gray16(
    input: &[u16],
    input_width: u32,
    input_height: u32,
    width: u32,
    height: u32,
) -> Vec<u16> {
    // Pillow's I;16 kernel accumulates FP64 and rounds to 16-bit after each
    // separable pass. Conversion to RGB subsequently clips to 255, not /257.
    let quantize = |value: f64| {
        let rounded = (value + if value < 0.0 { -0.5 } else { 0.5 }) as i64;
        ((rounded % 256).clamp(0, 255) + ((rounded >> 8).clamp(0, 255) << 8)) as u16
    };
    let horizontal = if width == input_width {
        input.to_vec()
    } else {
        let coeff = coefficients_f64(input_width, width);
        let mut output = vec![0u16; width as usize * input_height as usize];
        for y in 0..input_height as usize {
            for (x, (start, weights)) in coeff.iter().enumerate() {
                let value: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(offset, &weight)| {
                        input[y * input_width as usize + start + offset] as f64 * weight
                    })
                    .sum();
                output[y * width as usize + x] = quantize(value);
            }
        }
        output
    };
    if height == input_height {
        return horizontal;
    }
    let mut output = vec![0u16; width as usize * height as usize];
    for (y, (start, weights)) in coefficients_f64(input_height, height).iter().enumerate() {
        for x in 0..width as usize {
            let value: f64 = weights
                .iter()
                .enumerate()
                .map(|(offset, &weight)| {
                    horizontal[(start + offset) * width as usize + x] as f64 * weight
                })
                .sum();
            output[y * width as usize + x] = quantize(value);
        }
    }
    output
}

fn clip(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow-compatible bicubic resampling of a complete RGB image.
fn resize_bicubic(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    let horizontal = if input.width() != width {
        let weights = coefficients(input.width(), width);
        let mut out = RgbImage::new(width, input.height());
        let src = input.as_raw();
        for (row_index, row) in out
            .as_mut()
            .chunks_exact_mut(width as usize * 3)
            .enumerate()
        {
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
    } else {
        input.clone()
    };
    if input.height() == height {
        return horizontal;
    }
    let weights = coefficients(input.height(), height);
    let mut out = RgbImage::new(width, height);
    let stride = width as usize * 3;
    for (row, coeff) in out.as_mut().chunks_exact_mut(stride).zip(&weights) {
        for (column, target) in row.iter_mut().enumerate() {
            let mut sum = 1_i64 << (PRECISION_BITS - 1);
            for (offset, &weight) in coeff.weights.iter().enumerate() {
                sum += horizontal.as_raw()[(coeff.start + offset) * stride + column] as i64
                    * weight as i64;
            }
            *target = clip(sum);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn pattern(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            image::Rgb(std::array::from_fn(|c| {
                ((x * 73 + y * 151 + c as u32 * 97 + x * y * 11) % 256) as u8
            }))
        })
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn png_source_modes_match_pillow() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/decode.json")).unwrap();
        let mut failures = Vec::new();
        for case in data["cases"].as_array().unwrap() {
            let filename = case["file"].as_str().unwrap();
            if !filename.ends_with(".png") {
                continue;
            }
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/images")
                .join(filename);
            let minimum = case["minimum"].as_u64().unwrap() as u32;
            let maximum = case["maximum"].as_u64().unwrap() as u32;
            let result = prepare_file(&path, minimum, maximum);
            let result = result.unwrap();
            let bytes: Vec<_> = result
                .patches
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            if sha256(&bytes) != case["patches_sha256"].as_str().unwrap() {
                failures.push(format!("{filename} min={minimum} max={maximum}"));
            }
        }
        assert!(failures.is_empty(), "PNG parity failures: {failures:?}");
    }

    #[test]
    fn jpeg_decoder_matches_pillow() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/decode.json")).unwrap();
        let mut failures = Vec::new();
        for case in data["cases"].as_array().unwrap() {
            let filename = case["file"].as_str().unwrap();
            if !filename.ends_with(".jpg") || case["minimum"] != 16 || case["maximum"] != 128 {
                continue;
            }
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/images")
                .join(filename);
            let rgb = decode_jpeg_rgb(&std::fs::read(&path).unwrap()).unwrap();
            let expected = std::fs::read(path.with_extension("jpg.rgb")).unwrap();
            let changed = rgb
                .as_raw()
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a != b)
                .count();
            let max_abs = rgb
                .as_raw()
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            if changed > 0 {
                failures.push(format!(
                    "{filename}: {changed} changed bytes, max={max_abs}"
                ));
            }
        }
        assert!(failures.is_empty(), "JPEG parity failures: {failures:?}");
    }

    #[test]
    fn jpeg_source_modes_match_pillow() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/decode.json")).unwrap();
        for case in data["cases"].as_array().unwrap() {
            let filename = case["file"].as_str().unwrap();
            if !filename.ends_with(".jpg") {
                continue;
            }
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/images")
                .join(filename);
            let result = prepare_file(
                &path,
                case["minimum"].as_u64().unwrap() as u32,
                case["maximum"].as_u64().unwrap() as u32,
            )
            .unwrap();
            let bytes: Vec<_> = result
                .patches
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                sha256(&bytes),
                case["patches_sha256"].as_str().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn pixel_exact_pillow_resize_fixtures() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/preprocess.json")).unwrap();
        for case in fixtures["resizes"].as_array().unwrap() {
            let number = |key: &str| case[key].as_u64().unwrap() as u32;
            let image = pattern(number("width"), number("height"));
            let result = resize_bicubic(&image, number("target_width"), number("target_height"));
            assert_eq!(
                sha256(result.as_raw()),
                case["sha256"].as_str().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn upstream_two_stage_patches_and_torch_positions() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/preprocess.json")).unwrap();
        for case in fixtures["prepared"].as_array().unwrap() {
            let number = |key: &str| case[key].as_u64().unwrap() as u32;
            let image = pattern(number("width"), number("height"));
            let result = prepare_rgb(&image, number("minimum"), number("maximum")).unwrap();
            assert_eq!(
                (result.width, result.height),
                (
                    number("output_width") as usize,
                    number("output_height") as usize
                ),
                "{case}"
            );
            let patch_bytes: Vec<_> = result
                .patches
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let position_bytes: Vec<_> = result
                .positions_hw
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                sha256(&patch_bytes),
                case["patches_sha256"].as_str().unwrap(),
                "patches {case}"
            );
            assert_eq!(
                sha256(&position_bytes),
                case["independent_positions_sha256"].as_str().unwrap(),
                "independent positions {case}"
            );
            let x = case["positions_x_bits"].as_array().unwrap();
            let y = case["positions_y_bits"].as_array().unwrap();
            let bound = case["reference_spatial_max_absolute_error"]
                .as_f64()
                .unwrap();
            for (index, position) in result.positions_hw.iter().enumerate() {
                let reference = [
                    f32::from_bits(y[index / x.len()].as_u64().unwrap() as u32),
                    f32::from_bits(x[index % x.len()].as_u64().unwrap() as u32),
                ];
                for axis in 0..2 {
                    let error = (position[axis] as f64 - reference[axis] as f64).abs();
                    assert!(
                        error <= bound,
                        "position {index}, axis {axis}: {error} > independent reference bound {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn python_half_even_alignment() {
        assert_eq!(round_to_patch(24), 32);
        assert_eq!(round_to_patch(40), 32);
        assert_eq!(round_to_patch(56), 64);
        assert_eq!(round_to_patch(72), 64);
        assert_eq!(round_to_patch(73), 80);
    }

    #[test]
    fn bounds_preserve_upstream_operation_order() {
        assert_eq!(bounded_dimensions(1600, 1000, 64, 1536), (1536, 960));
        assert_eq!(bounded_dimensions(1000, 1600, 64, 1536), (960, 1536));
        assert_eq!(bounded_dimensions(32, 2000, 64, 1536), (24, 1536));
        assert_eq!(bounded_dimensions(65, 73, 64, 1536), (65, 73));
        assert_eq!(aligned_dimensions(65, 73).unwrap(), (64, 80));
        assert_eq!(aligned_dimensions(16, 16).unwrap(), (64, 64));
    }

    #[test]
    fn rejects_degenerate_shapes_and_configuration() {
        assert!(prepare_rgb(&RgbImage::new(0, 0), 64, 1536).is_err());
        assert!(prepare_rgb(&RgbImage::new(64, 64), 1536, 64).is_err());
        assert!(prepare_rgb(&RgbImage::new(1, 10000), 64, 1536).is_err());
        assert!(aligned_dimensions(16, 3216).is_err());
    }

    #[test]
    fn patches_are_grid_then_pixel_row_major_rgb() {
        let image = RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, (x ^ y) as u8]));
        let prepared = prepare_rgb(&image, 64, 1536).unwrap();
        assert_eq!(prepared.patches.len(), 64 * 64 * 3);
        let normalize = |v: u8| ((v as f64 / 255.0) as f32 - 0.5) / 0.5;
        for patch in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let pixel =
                        image.get_pixel((patch % 4 * 16 + x) as u32, (patch / 4 * 16 + y) as u32);
                    let index = patch * PATCH_VALUES + (y * 16 + x) * 3;
                    for c in 0..3 {
                        assert_eq!(prepared.patches[index + c], normalize(pixel[c]));
                    }
                }
            }
        }
        assert_eq!(prepared.positions_hw[0], [-1.0, -1.0]);
        assert_eq!(prepared.positions_hw[15], [1.0, 1.0]);
    }
}
