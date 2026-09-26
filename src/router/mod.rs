//! Per-page resolution routing for fast mode (research/resolution-router).
//!
//! Most English pages read as well at a lower image resolution, and prefill
//! and decode both scale with the image token count. The router computes 26
//! image statistics on the page after the processor's first resize at the
//! 1536-px cap, scores two gradient-boosted tree models (routable at 768 and
//! at 1024) and picks the smallest resolution whose model says yes and at
//! which the median text line stays at least 8 px tall. A routed page is the
//! same input a fixed `--max-dimension` run of that size sees.
//!
//! The statistics are specified by research/resolution-router/router_features.py
//! and reproduced here bit for bit: integer counts and sums, Pillow's 8-bit
//! resampling, and the same few float64 operations in the same order. The
//! trees (trees.json) come from train_trees.py.

use crate::preprocess::{Filter, resize_gray};
use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::Deserialize;
use std::sync::OnceLock;

/// The resolution cap the router starts from and falls back to.
pub const CAP: u32 = 1536;
/// The candidate resolutions, smallest first.
pub const SIZES: [u32; 2] = [768, 1024];
/// Median text-line height, in pixels at the target resolution, below which
/// a page is never routed there.
pub const LINE_FLOOR_PX: f64 = 8.0;

pub const FEATURES: [&str; 26] = [
    "width",
    "height",
    "aspect",
    "megapixels",
    "gray_mean",
    "gray_std",
    "ink",
    "dark",
    "saturation",
    "edge",
    "row_coverage",
    "lines",
    "line_height",
    "line_height_p10",
    "line_gap",
    "line_height_px",
    "col_gaps",
    "col_coverage",
    "ink_rows_std",
    "small_blobs",
    "loss_1024",
    "loss_768",
    "lossy_ink_1024",
    "lossy_ink_768",
    "ink_edge_ratio",
    "laplacian",
];
const LINE_HEIGHT_PX: usize = 15;

const ANALYSIS: u32 = 1024;
const INK: u8 = 160;
const DARK: u8 = 80;
const LOSS_DIFF: i32 = 60;
const EDGE: i32 = 40;

pub type Statistics = [f64; FEATURES.len()];

/// Pillow's `convert("L")`: ITU-R 601-2 luma in 16-bit fixed point.
fn gray(page: &RgbImage) -> Vec<u8> {
    page.pixels()
        .map(|p| ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16) as u8)
        .collect()
}

fn median(sorted: &[u64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0
    }
}

fn percentile10(sorted: &[u64]) -> f64 {
    let position = 0.1 * (sorted.len() - 1) as f64;
    let low = position as usize;
    let high = (low + 1).min(sorted.len() - 1);
    sorted[low] as f64 + (sorted[high] - sorted[low]) as f64 * (position - low as f64)
}

fn std(count: u64, sum: u64, squares: u64) -> f64 {
    let mean = sum as f64 / count as f64;
    (squares as f64 / count as f64 - mean * mean).max(0.0).sqrt()
}

/// Mean absolute change and the share changing by more than LOSS_DIFF over
/// the ink pixels when the page is scaled so its long side is `size`
/// (bicubic) and back (bilinear).
fn resample_loss(page: &[u8], width: u32, height: u32, ink_count: u64, size: u32) -> (f64, f64) {
    let scale = size as f64 / width.max(height) as f64;
    if scale >= 1.0 || ink_count == 0 {
        return (0.0, 0.0);
    }
    let (dw, dh) = (
        ((width as f64 * scale) as u32).max(1),
        ((height as f64 * scale) as u32).max(1),
    );
    let down = resize_gray(page, width, height, dw, dh, Filter::Bicubic);
    let up = resize_gray(&down, dw, dh, width, height, Filter::Bilinear);
    let (mut sum, mut lossy) = (0u64, 0u64);
    for (&g, &u) in page.iter().zip(&up) {
        if g < INK {
            let diff = (g as i32 - u as i32).abs();
            sum += diff as u64;
            lossy += u64::from(diff > LOSS_DIFF);
        }
    }
    (sum as f64 / ink_count as f64, lossy as f64 / ink_count as f64)
}

/// The 26 router statistics of a page after the first resize at `CAP`.
/// The page-resolution passes run beside the analysis-image pass (rayon).
pub fn statistics(page: &RgbImage) -> Statistics {
    let (width, height) = page.dimensions();
    let page_gray = gray(page);
    let ((analysis, saturation), native) = rayon::join(
        || rayon::join(|| analysis_statistics(&page_gray, width, height), || saturation(page)),
        || native_statistics(&page_gray, width, height),
    );
    let mut x = [0.0; FEATURES.len()];
    x[0] = width as f64;
    x[1] = height as f64;
    x[2] = width as f64 / height as f64;
    x[3] = (width as u64 * height as u64) as f64 / 1e6;
    x[4..8].copy_from_slice(&analysis[..4]);
    x[8] = saturation;
    x[9..20].copy_from_slice(&analysis[4..]);
    x[20..].copy_from_slice(&native);
    x
}

/// Mean saturation, (max - min) * 255 / max in integers, on a 4-px grid.
fn saturation(page: &RgbImage) -> f64 {
    let (mut total, mut samples) = (0u64, 0u64);
    for y in (0..page.height()).step_by(4) {
        for x in (0..page.width()).step_by(4) {
            let p = page.get_pixel(x, y).0;
            let (hi, lo) = (p[0].max(p[1]).max(p[2]) as u64, p[0].min(p[1]).min(p[2]) as u64);
            if hi > 0 {
                total += (hi - lo) * 255 / hi;
            }
            samples += 1;
        }
    }
    total as f64 / samples as f64
}

/// Statistics 4-7 and 9-19 on the page scaled (bilinear) to a 1024-px long side.
fn analysis_statistics(page: &[u8], width: u32, height: u32) -> [f64; 15] {
    let scale = ANALYSIS as f64 / width.max(height) as f64;
    let aw = ((width as f64 * scale).round_ties_even() as u32).max(1);
    let ah = ((height as f64 * scale).round_ties_even() as u32).max(1);
    let a = resize_gray(page, width, height, aw, ah, Filter::Bilinear);
    let (aw, ah) = (aw as usize, ah as usize);
    let n = (aw * ah) as u64;

    let (mut sum, mut squares, mut ink_count, mut dark) = (0u64, 0u64, 0u64, 0u64);
    let mut row_ink = vec![0u64; ah];
    let mut col_ink = vec![0u64; aw];
    let (mut dx, mut dy, mut transitions) = (0u64, 0u64, 0u64);
    for y in 0..ah {
        let row = &a[y * aw..][..aw];
        for (x, &v) in row.iter().enumerate() {
            sum += v as u64;
            squares += v as u64 * v as u64;
            dark += u64::from(v < DARK);
            if v < INK {
                ink_count += 1;
                row_ink[y] += 1;
                col_ink[x] += 1;
            }
        }
        for pair in row.windows(2) {
            dx += pair[0].abs_diff(pair[1]) as u64;
            transitions += u64::from((pair[0] < INK) != (pair[1] < INK));
        }
        if y + 1 < ah {
            dy += row
                .iter()
                .zip(&a[(y + 1) * aw..][..aw])
                .map(|(&p, &q)| p.abs_diff(q) as u64)
                .sum::<u64>();
        }
    }
    let mut edge = if aw > 1 {
        dx as f64 / (ah * (aw - 1)) as f64
    } else {
        0.0
    };
    edge += if ah > 1 {
        dy as f64 / ((ah - 1) * aw) as f64
    } else {
        0.0
    };

    // Runs of inked rows are text lines.
    let text_rows: Vec<bool> = row_ink.iter().map(|&c| c as f64 / aw as f64 > 0.01).collect();
    let (mut runs, mut gaps, mut start, mut last_end) = (Vec::new(), Vec::new(), None, None);
    for (i, &on) in text_rows.iter().chain([false].iter()).enumerate() {
        if on && start.is_none() {
            start = Some(i);
            if let Some(end) = last_end {
                gaps.push((i - end) as u64);
            }
        } else if !on && let Some(s) = start {
            runs.push((i - s) as u64);
            last_end = Some(i);
            start = None;
        }
    }
    runs.retain(|&r| r >= 2);
    if runs.is_empty() {
        runs.push(0);
    }
    runs.sort_unstable();
    gaps.sort_unstable();

    // Interior blank column gaps at least 1% of the width wide.
    let band: Vec<bool> = col_ink.iter().map(|&c| c as f64 / ah as f64 > 0.005).collect();
    let mut col_gaps = 0u32;
    if let (Some(first), Some(last)) = (band.iter().position(|&b| b), band.iter().rposition(|&b| b)) {
        let mut run = 0usize;
        for &on in &band[first..=last] {
            run = if on { 0 } else { run + 1 };
            if run == (aw / 100).max(3) {
                col_gaps += 1;
            }
        }
    }

    let line = median(&runs);
    let row_sum: u64 = row_ink.iter().sum();
    let row_squares: u64 = row_ink.iter().map(|c| c * c).sum();
    [
        sum as f64 / n as f64,
        std(n, sum, squares),
        ink_count as f64 / n as f64,
        dark as f64 / n as f64,
        edge,
        text_rows.iter().filter(|&&t| t).count() as f64 / ah as f64,
        runs.len() as f64,
        line / ah as f64,
        percentile10(&runs) / ah as f64,
        if gaps.is_empty() {
            0.0
        } else {
            median(&gaps) / ah as f64
        },
        line / scale,
        col_gaps as f64,
        band.iter().filter(|&&b| b).count() as f64 / aw as f64,
        std(ah as u64, row_sum, row_squares) / aw as f64,
        transitions as f64 / ink_count.max(1) as f64,
    ]
}

/// Statistics 20-25 at page resolution: the resampling losses at 1024 and
/// 768, ink per strong edge and the mean Laplacian over interior ink.
fn native_statistics(page: &[u8], width: u32, height: u32) -> [f64; 6] {
    let (w, h) = (width as usize, height as usize);
    let ink = page.iter().filter(|&&g| g < INK).count() as u64;
    let (((l1024, f1024), (l768, f768)), (edges, laplacian, inner)) = rayon::join(
        || {
            rayon::join(
                || resample_loss(page, width, height, ink, 1024),
                || resample_loss(page, width, height, ink, 768),
            )
        },
        || {
            let (mut edges, mut laplacian, mut inner) = (0u64, 0u64, 0u64);
            for y in 0..h {
                let row = &page[y * w..][..w];
                edges += row.windows(2).filter(|p| p[0].abs_diff(p[1]) as i32 > EDGE).count() as u64;
                if y + 1 < h {
                    let below = &page[(y + 1) * w..][..w];
                    edges += row
                        .iter()
                        .zip(below)
                        .filter(|(p, q)| p.abs_diff(**q) as i32 > EDGE)
                        .count() as u64;
                }
                if y == 0 || y + 1 == h {
                    continue;
                }
                let (above, below) = (&page[(y - 1) * w..][..w], &page[(y + 1) * w..][..w]);
                for x in 1..w.saturating_sub(1) {
                    let g = row[x] as i32;
                    if g < INK as i32 {
                        let around = above[x] as i32 + below[x] as i32 + row[x - 1] as i32 + row[x + 1] as i32;
                        laplacian += (4 * g - around).unsigned_abs() as u64;
                        inner += 1;
                    }
                }
            }
            (edges, laplacian, inner)
        },
    );
    [
        l1024,
        l768,
        f1024,
        f768,
        ink as f64 / edges.max(1) as f64,
        if inner > 0 {
            laplacian as f64 / inner as f64
        } else {
            0.0
        },
    ]
}

enum Node {
    Leaf(f64),
    /// `x[feature] <= threshold` goes left.
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
}

struct Model {
    baseline: f64,
    trees: Vec<Vec<Node>>,
}

impl Model {
    /// scikit-learn's raw score, summed in its order: the baseline, then each tree.
    fn raw(&self, x: &Statistics) -> f64 {
        let mut raw = self.baseline;
        for nodes in &self.trees {
            let mut node = &nodes[0];
            loop {
                match *node {
                    Node::Leaf(value) => {
                        raw += value;
                        break;
                    }
                    Node::Split {
                        feature,
                        threshold,
                        left,
                        right,
                    } => node = &nodes[if x[feature] <= threshold { left } else { right }],
                }
            }
        }
        raw
    }
}

#[derive(Deserialize)]
struct ModelFile {
    version: u32,
    cap: u32,
    features: Vec<String>,
    models: std::collections::BTreeMap<String, RawModel>,
}

#[derive(Deserialize)]
struct RawModel {
    baseline: f64,
    trees: Vec<Vec<Vec<f64>>>,
}

struct Trees {
    /// In `SIZES` order.
    models: Vec<Model>,
}

fn parse(json: &str) -> Result<Trees> {
    let file: ModelFile = serde_json::from_str(json).context("parsing the router trees")?;
    ensure!(file.version == 2 && file.cap == CAP, "unsupported router trees");
    ensure!(file.features == FEATURES, "router trees use different statistics");
    let mut models = Vec::new();
    for size in SIZES {
        let raw = file.models.get(&size.to_string()).context("router trees lack a size")?;
        let mut trees = Vec::with_capacity(raw.trees.len());
        for tree in &raw.trees {
            let nodes = tree
                .iter()
                .map(|node| match node[..] {
                    [value] => Ok(Node::Leaf(value)),
                    [feature, threshold, left, right] => {
                        let index = |v: f64, bound: usize| {
                            ensure!(
                                v >= 0.0 && v.fract() == 0.0 && (v as usize) < bound,
                                "bad router tree node"
                            );
                            Ok(v as usize)
                        };
                        Ok(Node::Split {
                            feature: index(feature, FEATURES.len())?,
                            threshold,
                            left: index(left, tree.len())?,
                            right: index(right, tree.len())?,
                        })
                    }
                    _ => anyhow::bail!("bad router tree node"),
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(!nodes.is_empty(), "empty router tree");
            trees.push(nodes);
        }
        models.push(Model {
            baseline: raw.baseline,
            trees,
        });
    }
    Ok(Trees { models })
}

fn trees() -> &'static Trees {
    static TREES: OnceLock<Trees> = OnceLock::new();
    TREES.get_or_init(|| parse(include_str!("trees.json")).expect("the embedded router trees are valid"))
}

/// The router's choice for one page.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Route {
    /// The chosen maximum dimension: 768, 1024 or `CAP`.
    pub max_dimension: u32,
    /// Raw (logit) scores of the 768 and 1024 models; routable when >= 0.
    pub score_768: f64,
    pub score_1024: f64,
    /// Median text-line height in pixels of the capped page.
    pub line_height_px: f64,
    pub statistics_ms: f64,
    /// Set when the routed run stopped by repetition or length and the page
    /// was rerun at `CAP` (the result is the rerun's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_net: Option<RoutedAttempt>,
}

/// The routed run that the safety net replaced.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RoutedAttempt {
    pub max_dimension: u32,
    pub finish_reason: crate::runner::FinishReason,
    pub output_tokens: usize,
    pub total_ms: f64,
}

/// A routed page's result that the safety net reruns at `CAP`.
pub fn needs_rerun(route: &Route, finish_reason: crate::runner::FinishReason) -> bool {
    use crate::runner::FinishReason;
    route.max_dimension != CAP && matches!(finish_reason, FinishReason::Repetition | FinishReason::Length)
}

/// Choose the resolution for a page after the first resize at `CAP`.
pub fn route(page: &RgbImage) -> Route {
    let started = std::time::Instant::now();
    let x = statistics(page);
    let scores: Vec<f64> = trees().models.iter().map(|m| m.raw(&x)).collect();
    let line = x[LINE_HEIGHT_PX];
    Route {
        max_dimension: choose(&scores, line, x[0].max(x[1])),
        score_768: scores[0],
        score_1024: scores[1],
        line_height_px: line,
        statistics_ms: started.elapsed().as_secs_f64() * 1000.0,
        safety_net: None,
    }
}

/// The smallest size whose model says routable (score >= 0, i.e. p >= 0.5)
/// and at which the median text line of the capped page (`line` px, long side
/// `long_side` px) keeps at least `LINE_FLOOR_PX`; else `CAP`.
fn choose(scores: &[f64], line: f64, long_side: f64) -> u32 {
    SIZES
        .iter()
        .zip(scores)
        .find(|&(&size, &score)| score >= 0.0 && line * (size as f64 / long_side).min(1.0) >= LINE_FLOOR_PX)
        .map_or(CAP, |(&size, _)| size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic page of tests/generate_router_fixtures.py.
    pub(crate) fn synthetic(width: u32, height: u32, seed: u32) -> RgbImage {
        let pitch = 18 + 5 * seed;
        let glyph = 9 + 3 * seed;
        RgbImage::from_fn(width, height, |x, y| {
            let paper = (250 - (x * 7 + y * 13 + seed) % 6) as u8;
            if (20..60).contains(&y) && (x / 40) % 2 == 0 {
                return image::Rgb([200, 40 + (seed * 30) as u8, 40]);
            }
            let text = (60..width.saturating_sub(60)).contains(&x) && (80..height.saturating_sub(80)).contains(&y);
            let gutter = seed == 2 && x.abs_diff(width / 2) < 20;
            if text && !gutter && (y - 80) % pitch < glyph {
                let (row, col) = ((y - 80) / pitch, (x - 60) / 7);
                let blank = (col + row * 3) % 11 == 0;
                if !blank && (x * 31 + y * 17 + row * 7 + col * 13 + seed) % 23 < 11 {
                    let v = (20 + (x + y) % 30) as u8;
                    return image::Rgb([v, v, v + (seed as u8) * 10]);
                }
            }
            image::Rgb([paper, paper, paper])
        })
    }

    fn fixtures() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/router.json")).unwrap()
    }

    fn check(case: &serde_json::Value, page: &RgbImage) {
        let name = case["name"].as_str().unwrap();
        let got = statistics(page);
        let want: Vec<f64> = case["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{name}: {} = {g}, Python {w}", FEATURES[i]);
        }
        let r = route(page);
        assert_eq!(
            r.score_768.to_bits(),
            case["raw_768"].as_f64().unwrap().to_bits(),
            "{name}"
        );
        assert_eq!(
            r.score_1024.to_bits(),
            case["raw_1024"].as_f64().unwrap().to_bits(),
            "{name}"
        );
        assert_eq!(r.max_dimension as u64, case["route"].as_u64().unwrap(), "{name}");
    }

    #[test]
    fn statistics_and_scores_match_python_on_synthetic_pages() {
        let fixtures = fixtures();
        let cases = fixtures["synthetic"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let n = |k: &str| case[k].as_u64().unwrap() as u32;
            check(case, &synthetic(n("width"), n("height"), n("seed")));
        }
    }

    #[cfg(feature = "turbojpeg")]
    #[test]
    fn statistics_and_scores_match_python_on_decoded_files() {
        let fixtures = fixtures();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images");
        for case in fixtures["files"].as_array().unwrap() {
            let (source, _) = crate::preprocess::decode_file(&root.join(case["name"].as_str().unwrap())).unwrap();
            check(case, &source.first_resize(64, CAP).unwrap());
        }
    }

    #[test]
    fn routes_are_distinct_on_the_fixtures() {
        // The fixtures exercise every outcome, so the parity test covers each branch.
        let fixtures = fixtures();
        let routes: std::collections::BTreeSet<u64> = fixtures["synthetic"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["route"].as_u64().unwrap())
            .collect();
        assert!(routes.len() >= 2, "{routes:?}");
    }

    #[test]
    fn choice_takes_the_smallest_routable_size_above_the_line_floor() {
        // 16-px lines on a 1536-px page: 8 px at 768, 10.7 px at 1024.
        assert_eq!(choose(&[0.0, 1.0], 16.0, 1536.0), 768);
        assert_eq!(choose(&[-0.1, 1.0], 16.0, 1536.0), 1024);
        assert_eq!(choose(&[-0.1, -0.1], 16.0, 1536.0), CAP);
        // 15-px lines fall under the floor at 768 but not at 1024.
        assert_eq!(choose(&[2.0, -0.1], 15.0, 1536.0), CAP);
        assert_eq!(choose(&[2.0, 2.0], 15.0, 1536.0), 1024);
        // A page already smaller than the target keeps its lines.
        assert_eq!(choose(&[2.0, 2.0], 8.0, 700.0), 768);
        assert_eq!(choose(&[2.0, 2.0], 7.9, 700.0), CAP);
    }

    #[test]
    fn embedded_trees_parse() {
        let t = trees();
        assert_eq!(t.models.len(), SIZES.len());
        assert!(t.models.iter().all(|m| !m.trees.is_empty()));
    }

    #[test]
    #[ignore = "timing probe"]
    fn statistics_breakdown() {
        let page = synthetic(1187, 1536, 0);
        let g = gray(&page);
        let time = |label: &str, f: &mut dyn FnMut()| {
            let started = std::time::Instant::now();
            for _ in 0..10 {
                f();
            }
            eprintln!("{label}: {:.2} ms", started.elapsed().as_secs_f64() * 100.0);
        };
        time("gray", &mut || _ = std::hint::black_box(gray(&page)));
        time("analysis", &mut || {
            _ = std::hint::black_box(resize_gray(&g, 1187, 1536, 791, 1024, Filter::Bilinear))
        });
        time("loss 1024", &mut || {
            _ = std::hint::black_box(resample_loss(&g, 1187, 1536, 1, 1024))
        });
        time("loss 768", &mut || {
            _ = std::hint::black_box(resample_loss(&g, 1187, 1536, 1, 768))
        });
        time("down 768", &mut || {
            _ = std::hint::black_box(resize_gray(&g, 1187, 1536, 593, 768, Filter::Bicubic))
        });
        time("statistics", &mut || _ = std::hint::black_box(statistics(&page)));
        time("trees", &mut || {
            _ = std::hint::black_box(trees().models[0].raw(&statistics(&page)))
        });
    }

    #[test]
    #[ignore = "timing probe"]
    fn statistics_profile() {
        let page = synthetic(1187, 1536, 0);
        let _ = route(&page);
        let started = std::time::Instant::now();
        for _ in 0..10 {
            std::hint::black_box(route(&page));
        }
        eprintln!("route: {:.2} ms", started.elapsed().as_secs_f64() * 100.0);
    }
}
