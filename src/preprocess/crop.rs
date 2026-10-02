//! The opt-in margin crop (`GenerationOptions::crop_margins`): the content
//! box of a first-resized page from integer luma statistics, and the page's
//! preparation cut to it.
use super::{PreparedImage, aligned_dimensions, luma, prepare_resized_rgb, round_to_patch};
use anyhow::Result;
use image::RgbImage;
use serde::{Deserialize, Serialize};

/// A margin crop of a first-resized page (`GenerationOptions::crop_margins`):
/// the rectangle kept and the size of the page it was cut from, all in
/// first-resize pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub first_width: u32,
    pub first_height: u32,
}

/// [`super::prepare_first`] of the page cut to its [`margin_crop`] with `pad`
/// pixels of padding when `crop_margins` is `Some(pad)` and the page has a crop
/// worth taking; the crop is reported in `PreparedImage::crop`. Text keeps its
/// size in first-resize pixels and only the image token count drops, but the
/// model input changes, so the runner does this only on request. `None`, or a
/// page that stays whole, gives exactly `prepare_first(first)`.
pub fn prepare_first_cropped(first: &RgbImage, crop_margins: Option<u32>) -> Result<PreparedImage> {
    let Some(crop) = crop_margins.and_then(|pad| margin_crop(first, pad)) else {
        return prepare_resized_rgb(first);
    };
    let cropped = image::imageops::crop_imm(first, crop.x, crop.y, crop.width, crop.height).to_image();
    Ok(PreparedImage {
        crop: Some(crop),
        ..prepare_resized_rgb(&cropped)?
    })
}

/// Ink is at least this many luma levels darker than the background.
const INK_CONTRAST: u8 = 64;
/// The background, the 90th luma percentile, must be at least this light.
const LIGHT_BACKGROUND: u8 = 160;
/// A row or column with fewer ink pixels is blank.
const MIN_INK_PIXELS: u32 = 2;
/// Content rows (columns) come in runs of at least this many: lines and
/// columns of text are longer at the first resize's scale, dust specks and
/// hairlines are not.
const MIN_CONTENT_RUN: usize = 4;
/// A crop must remove at least this percentage of the page area.
const MIN_SAVING_PERCENT: u64 = 10;

/// The content of a first-resized page (light background, dark ink) padded by
/// `pad` pixels and clamped to the page, or `None` when the page stays whole.
///
/// The background is the 90th percentile of the page's luma (Pillow's
/// `convert("L")`); ink is any pixel at least `INK_CONTRAST` levels darker; a
/// row or column is content when it holds at least `MIN_INK_PIXELS` ink pixels
/// and lies in a run of at least `MIN_CONTENT_RUN` such rows or columns, so
/// isolated specks up to 3 pixels across are ignored. The crop is the box from
/// the first to the last content row and column. No crop when the page has no
/// content, when the background is darker than `LIGHT_BACKGROUND` (inverted or
/// dark pages; dark scan borders and gutter shadows are ink, so they keep their
/// side of the page), when the crop would remove less than `MIN_SAVING_PERCENT`
/// of the area, when the processor's second resize of the crop would be more
/// than rounding each side to whole patches (the minimum- or maximum-area
/// rescale, fewer than 16 pixels, an aspect ratio above 200), or when the crop
/// would keep as many patches as the whole page (on small pages only, where
/// rounding to patches takes back what the crop removed). Integer arithmetic
/// throughout.
pub fn margin_crop(first: &RgbImage, pad: u32) -> Option<Crop> {
    let (width, height) = first.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    let mut histogram = [0u64; 256];
    for pixel in first.as_raw().chunks_exact(3) {
        histogram[luma(pixel) as usize] += 1;
    }
    let rank = width as u64 * height as u64 * 9 / 10;
    let mut below = 0;
    let background = histogram.iter().position(|&count| {
        below += count;
        below > rank
    })? as u8;
    if background < LIGHT_BACKGROUND {
        return None;
    }
    let ink = background - INK_CONTRAST;
    let mut rows = vec![0u32; height as usize];
    let mut columns = vec![0u32; width as usize];
    for (row, pixels) in rows.iter_mut().zip(first.as_raw().chunks_exact(width as usize * 3)) {
        for (column, pixel) in columns.iter_mut().zip(pixels.chunks_exact(3)) {
            if luma(pixel) <= ink {
                *row += 1;
                *column += 1;
            }
        }
    }
    let (left, right) = content_span(&columns)?;
    let (top, bottom) = content_span(&rows)?;
    let (x, y) = (left.saturating_sub(pad), top.saturating_sub(pad));
    let crop_width = right.saturating_add(pad).min(width) - x;
    let crop_height = bottom.saturating_add(pad).min(height) - y;
    let area = width as u64 * height as u64;
    let removed = area - crop_width as u64 * crop_height as u64;
    if removed * 100 < area * MIN_SAVING_PERCENT {
        return None;
    }
    let plain = (round_to_patch(crop_width), round_to_patch(crop_height));
    if aligned_dimensions(crop_width, crop_height).ok()? != plain {
        return None;
    }
    let whole = aligned_dimensions(width, height).ok()?;
    if plain.0 as u64 * plain.1 as u64 >= whole.0 as u64 * whole.1 as u64 {
        return None;
    }
    Some(Crop {
        x,
        y,
        width: crop_width,
        height: crop_height,
        first_width: width,
        first_height: height,
    })
}

/// The first content line and the one after the last, given the ink pixels of
/// each row (or column): lines with at least `MIN_INK_PIXELS` in runs of at
/// least `MIN_CONTENT_RUN`.
fn content_span(ink_per_line: &[u32]) -> Option<(u32, u32)> {
    let mut span: Option<(usize, usize)> = None;
    let mut run_start = 0;
    // A blank line after the last one closes the final run.
    for (index, &ink) in ink_per_line.iter().chain(std::iter::once(&0)).enumerate() {
        if ink >= MIN_INK_PIXELS {
            continue;
        }
        if index - run_start >= MIN_CONTENT_RUN {
            span = Some((span.map_or(run_start, |(first, _)| first), index));
        }
        run_start = index + 1;
    }
    span.map(|(first, end)| (first as u32, end as u32))
}

#[cfg(test)]
mod tests {
    use super::super::prepare_first;
    use super::*;

    /// An off-white page with dark diagonal hatching over `content` (x0, y0,
    /// x1, y1): every row and column of that box holds ink, nothing else does.
    fn page(width: u32, height: u32, content: (u32, u32, u32, u32)) -> RgbImage {
        let (x0, y0, x1, y1) = content;
        RgbImage::from_fn(width, height, |x, y| {
            let ink = (x0..x1).contains(&x) && (y0..y1).contains(&y) && (x + y) % 3 == 0;
            image::Rgb(if ink { [20, 20, 30] } else { [250, 248, 240] })
        })
    }

    fn crop(x: u32, y: u32, width: u32, height: u32, first_width: u32, first_height: u32) -> Option<Crop> {
        Some(Crop {
            x,
            y,
            width,
            height,
            first_width,
            first_height,
        })
    }

    #[test]
    fn margin_crop_pads_the_content_box_and_ignores_specks() {
        let first = page(800, 1000, (150, 200, 650, 700));
        assert_eq!(margin_crop(&first, 24), crop(126, 176, 548, 548, 800, 1000));
        assert_eq!(margin_crop(&first, 0), crop(150, 200, 500, 500, 800, 1000));
        // The padding stops at the page's edges.
        let corner = page(800, 1000, (10, 5, 400, 500));
        assert_eq!(margin_crop(&corner, 24), crop(0, 0, 424, 524, 800, 1000));
        // Specks: single ink pixels, a 2-pixel dash and a 3 x 3 blot are
        // ignored; a 4 x 4 mark is content and the crop grows to keep it.
        let mut specked = first.clone();
        let blots = [
            (3, 3, 1),
            (790, 40, 1),
            (40, 990, 1),
            (700, 900, 1),
            (4, 3, 1),
            (60, 60, 3),
        ];
        for (x0, y0, size) in blots {
            for (x, y) in (x0..x0 + size).flat_map(|x| (y0..y0 + size).map(move |y| (x, y))) {
                specked.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        assert_eq!(margin_crop(&specked, 24), margin_crop(&first, 24));
        for (x, y) in (60..64).flat_map(|x| (60..64).map(move |y| (x, y))) {
            specked.put_pixel(x, y, image::Rgb([0, 0, 0]));
        }
        assert_eq!(margin_crop(&specked, 24), crop(36, 36, 638, 688, 800, 1000));
        // Runs: lines 1-4 and 10-14 hold content, the run 6-8 is too short.
        assert_eq!(
            content_span(&[0, 5, 5, 5, 5, 0, 2, 2, 2, 0, 9, 9, 9, 9, 9]),
            Some((1, 15))
        );
        assert_eq!(content_span(&[1, 1, 1, 1, 1, 1]), None);
    }

    #[test]
    fn margin_crop_leaves_blank_dark_and_nearly_full_pages_whole() {
        assert_eq!(margin_crop(&page(800, 1000, (0, 0, 0, 0)), 24), None);
        // Inverted and mid-gray pages: the background is not light.
        let on = |background: u8, ink: u8| {
            RgbImage::from_fn(800, 1000, |x, y| {
                let text = (300..500).contains(&x) && (400..600).contains(&y) && (x + y) % 3 == 0;
                image::Rgb([if text { ink } else { background }; 3])
            })
        };
        assert_eq!(margin_crop(&on(30, 240), 24), None);
        assert_eq!(margin_crop(&on(140, 20), 24), None);
        assert!(margin_crop(&on(200, 20), 24).is_some());
        // Removing 9.9% of the area is not worth it; 10% is.
        assert_eq!(margin_crop(&page(1000, 1000, (0, 0, 1000, 901)), 0), None);
        assert_eq!(
            margin_crop(&page(1000, 1000, (0, 0, 1000, 900)), 0),
            crop(0, 0, 1000, 900, 1000, 1000)
        );
        // The second resize must stay a rounding to whole patches: 40 x 40
        // pixels would be scaled up to the minimum area, 600 x 6 is too thin.
        let small = page(800, 1000, (380, 480, 420, 520));
        assert_eq!(margin_crop(&small, 0), None);
        assert_eq!(margin_crop(&small, 24), crop(356, 456, 88, 88, 800, 1000));
        assert_eq!(margin_crop(&page(800, 1000, (100, 500, 700, 506)), 0), None);
        assert_eq!(margin_crop(&RgbImage::new(0, 0), 24), None);
        // A crop must save patches: 90 x 58 pixels round to the 96 x 64 of
        // the whole 90 x 70 page.
        let short = page(90, 70, (0, 0, 90, 34));
        assert_eq!(margin_crop(&short, 24), None);
        assert_eq!(margin_crop(&short, 8), crop(0, 0, 90, 42, 90, 70));
    }

    #[test]
    fn cropped_preparation_is_the_uncropped_one_unless_a_crop_applies() {
        let bits = |prepared: &PreparedImage| {
            let patches: Vec<u32> = prepared.patches.iter().map(|v| v.to_bits()).collect();
            let positions: Vec<[u32; 2]> = prepared.positions_hw.iter().map(|p| p.map(f32::to_bits)).collect();
            (prepared.width, prepared.height, patches, positions)
        };
        let first = page(800, 1000, (150, 200, 650, 700));
        let whole = prepare_first(&first).unwrap();
        // Off (the default): bitwise the uncropped input, and no crop reported.
        let off = prepare_first_cropped(&first, None).unwrap();
        assert_eq!((bits(&off), off.crop), (bits(&whole), None));
        // On, for a page that stays whole: the same.
        let full = page(800, 1000, (10, 10, 790, 990));
        let kept = prepare_first_cropped(&full, Some(24)).unwrap();
        assert_eq!((bits(&kept), kept.crop), (bits(&prepare_first(&full).unwrap()), None));
        // A crop is the unchanged second resize and packing of the cut page:
        // 34 x 34 patches instead of 50 x 62.
        let cropped = prepare_first_cropped(&first, Some(24)).unwrap();
        assert_eq!(cropped.crop, crop(126, 176, 548, 548, 800, 1000));
        let cut = image::imageops::crop_imm(&first, 126, 176, 548, 548).to_image();
        assert_eq!(bits(&cropped), bits(&prepare_first(&cut).unwrap()));
        assert_eq!(
            (cropped.positions_hw.len(), whole.positions_hw.len()),
            (34 * 34, 50 * 62)
        );
    }
}
