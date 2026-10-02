//! Image preparation for the pinned Falcon-OCR v1.5 processor.
//!
//! The two resizing stages are intentional. Each performs uint8 bicubic
//! resampling, including quantization between the horizontal and vertical passes.
//! Replacing them with a single float resize changes the model input.
//!
//! The Pillow resampling port is in `resample`, the opt-in margin crop in
//! `crop`.
mod crop;
mod resample;

use anyhow::{Context, Result, ensure};
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use std::path::Path;

pub use crop::{Crop, margin_crop, prepare_first_cropped};
pub(crate) use resample::{Filter, resize_gray};
use resample::{resize_bicubic, resize_gray16, resize_nearest, resize_premultiplied};

pub const PATCH_SIZE: usize = 16;
pub const PATCH_VALUES: usize = PATCH_SIZE * PATCH_SIZE * 3;
const MIN_PIXELS: u64 = 56 * 56;
const MAX_PIXELS: u64 = 28 * 28 * 1280 * 10;

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub width: usize,
    pub height: usize,
    /// Grid row-major patches, each containing row-major RGB pixels.
    pub patches: Vec<f32>,
    /// Per-patch [height, width] positions. FP32 sqrt uses IEEE rounding; the
    /// reference PyTorch MKL sqrt can differ by an ULP (see frozen fixtures).
    pub positions_hw: Vec<[f32; 2]>,
    /// The margin crop this input was cut to after the first resize
    /// ([`prepare_first_cropped`]), if one was applied.
    pub crop: Option<Crop>,
}

pub fn prepare_rgb(image: &RgbImage, min_dimension: u32, max_dimension: u32) -> Result<PreparedImage> {
    prepare_resized_rgb(&first_resize_rgb(image, min_dimension, max_dimension)?)
}

/// The processor's first (aspect-preserving) resize of an RGB image.
pub fn first_resize_rgb(image: &RgbImage, min_dimension: u32, max_dimension: u32) -> Result<RgbImage> {
    let (width, height) = image.dimensions();
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    ensure!(
        min_dimension > 0 && min_dimension <= max_dimension,
        "expected 0 < min_dimension <= max_dimension"
    );

    let (first_width, first_height) = bounded_dimensions(width, height, min_dimension, max_dimension);
    ensure!(
        first_width > 0 && first_height > 0,
        "the upstream aspect-preserving resize produces a zero dimension"
    );
    Ok(resize_bicubic(image, first_width, first_height))
}

/// The second resize, normalization and patch packing of a first-resized page.
pub fn prepare_first(first: &RgbImage) -> Result<PreparedImage> {
    prepare_resized_rgb(first)
}

/// Pillow's `convert("L")` of one RGB pixel: ITU-R 601-2 luma in 16-bit fixed
/// point. The margin crop and the router's statistics both use it.
pub(crate) fn luma(pixel: &[u8]) -> u8 {
    ((pixel[0] as u32 * 19595 + pixel[1] as u32 * 38470 + pixel[2] as u32 * 7471 + 0x8000) >> 16) as u8
}

/// Decode a PNG or JPEG and preserve Pillow's source-mode first resize.
/// Metadata orientation and color profiles are not applied, matching upstream.
pub fn prepare_file(path: &Path, min_dimension: u32, max_dimension: u32) -> Result<PreparedImage> {
    prepare_file_timed(path, min_dimension, max_dimension).map(|(image, _)| image)
}

/// Return prepared image and milliseconds spent reading/decoding the file,
/// excluding resizing, normalization, and patch packing.
pub fn prepare_file_timed(path: &Path, min_dimension: u32, max_dimension: u32) -> Result<(PreparedImage, f64)> {
    let (source, decode_ms) = decode_file(path)?;
    Ok((
        prepare_resized_rgb(&source.first_resize(min_dimension, max_dimension)?)?,
        decode_ms,
    ))
}

/// A decoded PNG or JPEG in the form the processor's first resize starts
/// from (its source mode), so one decode can serve several resizes.
pub struct SourceImage {
    decoded: DynamicImage,
    /// The inverted CMYK samples of a four-component JPEG.
    cmyk: Option<RgbaImage>,
    /// PNG bit depth and colour type from IHDR.
    png: Option<(u8, u8)>,
}

/// Read and decode a PNG or JPEG; also returns the milliseconds spent.
pub fn decode_file(path: &Path) -> Result<(SourceImage, f64)> {
    let decode_started = std::time::Instant::now();
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    let format = image::guess_format(&bytes)?;
    ensure!(
        matches!(format, ImageFormat::Png | ImageFormat::Jpeg),
        "only PNG and JPEG files are supported"
    );
    let cmyk = if format == ImageFormat::Jpeg {
        decode_jpeg_cmyk(&bytes)?
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
    let png = if format == ImageFormat::Png {
        ensure!(bytes.len() >= 29 && &bytes[12..16] == b"IHDR", "missing PNG IHDR");
        Some((bytes[24], bytes[25]))
    } else {
        None
    };
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    Ok((SourceImage { decoded, cmyk, png }, decode_ms))
}

impl SourceImage {
    pub fn dimensions(&self) -> (u32, u32) {
        (self.decoded.width(), self.decoded.height())
    }

    /// The processor's first resize in the source mode, then RGB (Pillow-exact).
    pub fn first_resize(&self, min_dimension: u32, max_dimension: u32) -> Result<RgbImage> {
        let (width, height) = self.dimensions();
        let decoded = &self.decoded;
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
        Ok(if let Some(cmyk) = &self.cmyk {
            let colors = RgbImage::from_fn(width, height, |x, y| {
                let p = cmyk.get_pixel(x, y);
                image::Rgb([p[0], p[1], p[2]])
            });
            let blacks = RgbImage::from_fn(width, height, |x, y| image::Rgb([cmyk.get_pixel(x, y)[3]; 3]));
            let colors = resize_bicubic(&colors, w, h);
            let blacks = resize_bicubic(&blacks, w, h);
            let resized = RgbaImage::from_fn(w, h, |x, y| {
                let c = colors.get_pixel(x, y);
                image::Rgba([c[0], c[1], c[2], blacks.get_pixel(x, y)[0]])
            });
            cmyk_to_rgb(&resized)
        } else if let Some((depth, color)) = self.png {
            if color == 0 && depth == 16 {
                let values = decoded.to_luma16();
                let values = resize_gray16(values.as_raw(), width, height, w, h);
                RgbImage::from_fn(w, h, |x, y| {
                    image::Rgb([values[(y * w + x) as usize].min(255) as u8; 3])
                })
            } else if color == 3 || (color == 0 && depth == 1) {
                let rgb = decoded.to_rgb8();
                // Pillow always uses nearest for modes P and 1 on this first call.
                resize_nearest(&rgb, w, h)?
            } else if matches!(color, 4 | 6) && resize {
                let rgba = pillow_rgba8(decoded, depth);
                resize_premultiplied(&rgba, w, h)
            } else {
                // tRNS metadata on RGB/L is ignored by PIL.convert("RGB"); do not
                // mistakenly treat expanded tRNS pixels as an original RGBA mode.
                let rgba = pillow_rgba8(decoded, depth);
                let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
                resize_bicubic(&rgb, w, h)
            }
        } else {
            resize_bicubic(&decoded.to_rgb8(), w, h)
        })
    }
}

/// Without the `turbojpeg` feature: the `image` crate's JPEG decoder, which
/// converts CMYK itself. Not Pillow-exact (`auto::Resolved::image_decoder`
/// says so).
#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg_rgb(bytes: &[u8]) -> Result<RgbImage> {
    Ok(image::load_from_memory_with_format(bytes, ImageFormat::Jpeg)?.to_rgb8())
}

/// Without the `turbojpeg` feature, four-component JPEGs take the RGB path.
#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg_cmyk(_bytes: &[u8]) -> Result<Option<RgbaImage>> {
    Ok(None)
}

#[cfg(feature = "turbojpeg")]
fn decode_jpeg_rgb(bytes: &[u8]) -> Result<RgbImage> {
    if let Some(cmyk) = decode_jpeg_cmyk(bytes)? {
        return Ok(cmyk_to_rgb(&cmyk));
    }
    // libjpeg-turbo uses the accurate integer IDCT and fancy chroma upsampling
    // by default, as does the pinned Pillow JPEG decoder.
    check_jpeg_size(bytes, 3)?;
    let decoded = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)?;
    ensure!(decoded.pitch == decoded.width * 3, "unexpected JPEG row stride");
    RgbImage::from_raw(decoded.width.try_into()?, decoded.height.try_into()?, decoded.pixels)
        .context("inconsistent JPEG dimensions")
}

/// The inverted CMYK samples of a four-component JPEG (`None` for others).
#[cfg(feature = "turbojpeg")]
fn decode_jpeg_cmyk(bytes: &[u8]) -> Result<Option<RgbaImage>> {
    if jpeg_components(bytes)? != 4 {
        return Ok(None);
    }
    check_jpeg_size(bytes, 4)?;
    let mut decoded = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::CMYK)?;
    ensure!(decoded.pitch == decoded.width * 4, "unexpected CMYK row stride");
    // Pillow raw mode CMYK;I inverts libjpeg's CMYK samples before resizing.
    for value in &mut decoded.pixels {
        *value = 255 - *value;
    }
    RgbaImage::from_raw(decoded.width.try_into()?, decoded.height.try_into()?, decoded.pixels)
        .context("inconsistent CMYK dimensions")
        .map(Some)
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

#[cfg(feature = "turbojpeg")]
fn jpeg_components(bytes: &[u8]) -> Result<u8> {
    Ok(bytes[jpeg_frame(bytes)? + 7])
}

/// Refuse a JPEG whose header claims more decoded bytes, at `channels` per
/// pixel, than the `image` crate's default allocation limit (512 MiB), the
/// cap PNG decoding has: libjpeg-turbo allocates whatever the header claims,
/// and a failed allocation aborts the process instead of failing the page.
#[cfg(feature = "turbojpeg")]
fn check_jpeg_size(bytes: &[u8], channels: u64) -> Result<()> {
    let frame = jpeg_frame(bytes)?;
    let dimension = |at: usize| u16::from_be_bytes([bytes[frame + at], bytes[frame + at + 1]]) as u64;
    let (height, width) = (dimension(3), dimension(5));
    let limit = image::Limits::default().max_alloc.unwrap_or(u64::MAX);
    ensure!(
        width * height * channels <= limit,
        "the JPEG is {width} x {height} pixels, more than {limit} bytes to decode"
    );
    Ok(())
}

/// The offset of a JPEG frame header's length field, followed by the
/// precision, the height, the width and the component count.
#[cfg(feature = "turbojpeg")]
fn jpeg_frame(bytes: &[u8]) -> Result<usize> {
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
        ensure!(!matches!(marker, 0xd9 | 0xda), "JPEG frame header is missing");
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
            return Ok(offset);
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
        crop: None,
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

#[cfg(test)]
mod tests {
    use super::resample::{nearest_positions, tall_first};
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
        let data: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/decode.json")).unwrap();
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
            let bytes: Vec<_> = result.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
            if sha256(&bytes) != case["patches_sha256"].as_str().unwrap() {
                failures.push(format!("{filename} min={minimum} max={maximum}"));
            }
        }
        assert!(failures.is_empty(), "PNG parity failures: {failures:?}");
    }

    #[test]
    fn nearest_positions_match_pillow() {
        // Pillow 12.3.0 (the pinned processor's) resizing a one-row mode I image
        // that holds its own column indices, `Image.fromarray(np.arange(n,
        // dtype=np.int32)[None], "I").resize((m, 1), Image.Resampling.NEAREST)`,
        // the `ImagingScaleAffine` call that modes P and 1 get: n, m, the number
        // of positions where the direct product `(k + 0.5) * n / m` (this runner
        // before 2026-09-30) differs, and the SHA-256 of the positions as
        // little-endian u32.
        let cases = "
            16 12 1 382dc8a9be37675bf3f538be84c3ca7068e57750b44fe9d7bb5f314d114308e4
            44 33 5 07d5fd49027f17c5bee31f81a3b8878c9f3a72582825ac714c5242a7a7d73b5a
            5 13 0 ed365afe9d5c66e1174dd65cbadea1340e7fa59bb466be1f506db5273b0ce7d9
            2048 1536 303 a67bd0ea1108d155e2667355332893e2c1f87182b416e66df149fd687dcdb4cd
            2480 1085 97 8d62d0033758fddec05c4cf7bce4cd33ea38da5ca34eda113ab8889dd0f35d91
            4096 3072 405 c1e3342a0447005f975bb6912894b5838d3587c7d24995022c2dd7b293e01a4c
            3508 1536 0 8c800de18e29891c3b7bd92cc392b09be06c2f0f18e7b377db2b648242acc166
            1536 1536 0 57c372795f4a7d1f49185aa616ab07e32b7c75f35222d5a0996b9cbcd3f92ff4
            1085 2480 0 040dad2ae6274dad54a20a6b77bba75d348c054c78da0e848d05d16f2320725d
            7 1 0 9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178
            1 7 0 3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8";
        for case in cases.lines().map(str::trim).filter(|line| !line.is_empty()) {
            let fields: Vec<_> = case.split_whitespace().collect();
            let [input, output, differs] = [0, 1, 2].map(|i| fields[i].parse::<u32>().unwrap());
            let positions = nearest_positions(input, output).unwrap();
            let bytes: Vec<_> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
            assert_eq!(sha256(&bytes), fields[3], "{case}");
            let scale = input as f64 / output as f64;
            let direct = (0..output).map(|k| (((k as f64 + 0.5) * scale) as u32).min(input - 1));
            let changed = positions.iter().zip(direct).filter(|&(&p, d)| p != d).count();
            assert_eq!(changed, differs as usize, "{case}");
        }
        assert_eq!(
            nearest_positions(16, 12).unwrap(),
            [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15]
        );
        assert_eq!(
            nearest_positions(5, 13).unwrap(),
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
        );
        // Above 2^24 the f32 box rounds sizes: 2,147,483,777 becomes 2,147,483,904,
        // and 9,000,000 outputs would read 7 pixels past the source. An error, not a clamp.
        assert!(nearest_positions(2_147_483_777, 9_000_000).is_err());
    }

    #[test]
    fn tall_pages_resize_vertically_first_like_pillow() {
        // Pillow 12.3.0 on this pattern as a uint16 array (mode I;16):
        // `Image.fromarray(values).resize((16, 1700), Image.Resampling.BICUBIC)`,
        // hashed as little-endian u16. The one-call core resize (horizontal first,
        // the order this runner used before 2026-09-30) differs in 9,234 of the
        // 27,200 values. The pattern spans the full 16-bit range, so overshoots
        // also check the clip to 65535. The RGB path is covered by
        // `tall-17x1800.png` in the decode fixtures.
        let pattern = |width: u64, height: u64| -> Vec<u16> {
            (0..height)
                .flat_map(|y| (0..width).map(move |x| ((x * 40503 + y * 9973 + x * y * 17) % 65536) as u16))
                .collect()
        };
        let hash = |values: Vec<u16>| sha256(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        assert_eq!(
            hash(resize_gray16(&pattern(17, 1800), 17, 1800, 16, 1700)),
            "bc1a4abf9ea909e60792f27ae40c7d41a08fe17c790f3bb6b597f3ce7f40f681"
        );
        // At most 100 times taller than wide: horizontal first, as before.
        assert_eq!(
            hash(resize_gray16(&pattern(17, 40), 17, 40, 16, 30)),
            "abe47940ea5d785cc9ed6741b3f8d80cd4a8a85cb1250ec27e9dd55d760c04fc"
        );
        assert!(tall_first(17, 1701, 1700) && !tall_first(17, 1700, 1600) && !tall_first(17, 1800, 1800));
    }

    #[cfg(feature = "turbojpeg")]
    #[test]
    fn jpeg_decoder_matches_pillow() {
        let data: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/decode.json")).unwrap();
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
            let changed = rgb.as_raw().iter().zip(&expected).filter(|(a, b)| a != b).count();
            let max_abs = rgb
                .as_raw()
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            if changed > 0 {
                failures.push(format!("{filename}: {changed} changed bytes, max={max_abs}"));
            }
        }
        assert!(failures.is_empty(), "JPEG parity failures: {failures:?}");
    }

    #[cfg(feature = "turbojpeg")]
    #[test]
    fn a_jpeg_too_large_to_decode_fails_instead_of_aborting() {
        let images = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images");
        for name in ["gray.jpg", "cmyk.jpg"] {
            let mut bytes = std::fs::read(images.join(name)).unwrap();
            let frame = jpeg_frame(&bytes).unwrap();
            // 40000 x 40000: 4.8 GB of RGB, 6.4 GB of CMYK.
            bytes[frame + 3..frame + 7].copy_from_slice(&[0x9c, 0x40, 0x9c, 0x40]);
            let error = decode_jpeg_rgb(&bytes).unwrap_err().to_string();
            assert!(error.contains("40000 x 40000 pixels"), "{name}: {error}");
        }
    }

    #[test]
    fn jpeg_source_modes_match_pillow() {
        let data: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/decode.json")).unwrap();
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
            let bytes: Vec<_> = result.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(sha256(&bytes), case["patches_sha256"].as_str().unwrap(), "{case}");
        }
    }

    #[test]
    fn pixel_exact_pillow_resize_fixtures() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/preprocess.json")).unwrap();
        for case in fixtures["resizes"].as_array().unwrap() {
            let number = |key: &str| case[key].as_u64().unwrap() as u32;
            let image = pattern(number("width"), number("height"));
            let result = resize_bicubic(&image, number("target_width"), number("target_height"));
            assert_eq!(sha256(result.as_raw()), case["sha256"].as_str().unwrap(), "{case}");
        }
    }

    #[test]
    fn upstream_two_stage_patches_and_torch_positions() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/preprocess.json")).unwrap();
        for case in fixtures["prepared"].as_array().unwrap() {
            let number = |key: &str| case[key].as_u64().unwrap() as u32;
            let image = pattern(number("width"), number("height"));
            let result = prepare_rgb(&image, number("minimum"), number("maximum")).unwrap();
            assert_eq!(
                (result.width, result.height),
                (number("output_width") as usize, number("output_height") as usize),
                "{case}"
            );
            let patch_bytes: Vec<_> = result.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
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
            let bound = case["reference_spatial_max_absolute_error"].as_f64().unwrap();
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
                    let pixel = image.get_pixel((patch % 4 * 16 + x) as u32, (patch / 4 * 16 + y) as u32);
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
