//! Compare independent sequential requests against bounded mixed-batch
//! execution: the fixed cohorts of `recognize_batch`, and the continuous
//! batching and page pipeline that `falcon-ocr run --batch-size N` and
//! `--pipeline` use.
use falcon_ocr::{
    Backend, DecodeThreads, GenerationOptions, Model, OcrResult, Pipeline, Runner, RunnerConfig,
    preprocess::first_resize_rgb,
    router::{self, Route},
    runner::ControlFlow,
};
use image::{Rgb, RgbImage, imageops};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// The fixture pages: a full page, its first line, a blank page and the full
/// page again (mixed stop lengths).
fn fixture_images() -> [RgbImage; 4] {
    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let one_line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    [original.clone(), one_line, blank, original]
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn mixed_batches_preserve_tokens_order_stops_and_limits() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let images = fixture_images();
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    let sequential = Runner::new(
        model.clone(),
        "artifacts/model",
        RunnerConfig {
            threads: 4,
            batch_size: 1,
            ..RunnerConfig::reference()
        },
    )
    .unwrap();
    let expected = images
        .iter()
        .map(|image| sequential.recognize(image, &options).unwrap())
        .collect::<Vec<_>>();
    assert!(
        expected.iter().any(|r| r.output_tokens != expected[0].output_tokens),
        "fixture must exercise mixed stop lengths"
    );
    for size in [2, 4, 8] {
        let runner = Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                batch_size: size,
                backend: Backend::Auto,
                ..RunnerConfig::reference()
            },
        )
        .unwrap();
        // A partial final chunk also exercises output indexing and order.
        let input = (0..7).map(|i| images[i % images.len()].clone()).collect::<Vec<_>>();
        let actual = runner.recognize_batch(&input, &options).unwrap();
        assert_eq!(actual.len(), input.len());
        for (i, result) in actual.iter().enumerate() {
            let reference = &expected[i % expected.len()];
            assert_eq!(result.token_ids, reference.token_ids, "batch {size}, request {i}");
            assert_eq!(result.finish_reason, reference.finish_reason);
            assert_eq!((result.width, result.height), (reference.width, reference.height));
            let projection = result.timings.image_projection_ms.unwrap();
            let transformer = result.timings.transformer_prefill_ms.unwrap();
            assert!(projection.is_finite() && projection >= 0.0);
            assert!(transformer.is_finite() && transformer >= 0.0);
            assert!(projection + transformer <= result.timings.prefill_ms + 1e-6);
        }
        let limited = runner
            .recognize_batch(
                &input,
                &GenerationOptions {
                    max_new_tokens: 3,
                    ..options.clone()
                },
            )
            .unwrap();
        for (i, result) in limited.iter().enumerate() {
            let expected = &expected[i % expected.len()];
            assert_eq!(result.token_ids, &expected.token_ids[..3.min(expected.token_ids.len())]);
            assert!(result.output_tokens <= 3);
        }
    }
}

/// The router's synthetic page (`router::tests::synthetic`,
/// tests/generate_router_fixtures.py); at 700 x 950 with seed 1 the router
/// sends it to 1024 (tests/fixtures/router.json).
fn router_page(width: u32, height: u32, seed: u32) -> RgbImage {
    let pitch = 18 + 5 * seed;
    let glyph = 9 + 3 * seed;
    RgbImage::from_fn(width, height, |x, y| {
        let paper = (250 - (x * 7 + y * 13 + seed) % 6) as u8;
        if (20..60).contains(&y) && (x / 40) % 2 == 0 {
            return Rgb([200, 40 + (seed * 30) as u8, 40]);
        }
        let text = (60..width.saturating_sub(60)).contains(&x) && (80..height.saturating_sub(80)).contains(&y);
        let gutter = seed == 2 && x.abs_diff(width / 2) < 20;
        if text && !gutter && (y - 80) % pitch < glyph {
            let (row, col) = ((y - 80) / pitch, (x - 60) / 7);
            let blank = (col + row * 3) % 11 == 0;
            if !blank && (x * 31 + y * 17 + row * 7 + col * 13 + seed) % 23 < 11 {
                let v = (20 + (x + y) % 30) as u8;
                return Rgb([v, v, v + (seed as u8) * 10]);
            }
        }
        Rgb([paper, paper, paper])
    })
}

/// The smoke page's first line tiled down a 700 x 950 white page. Unlike the
/// router's synthetic page and the colour fixture, it gives the model text to
/// read to the length limit, so the safety net reruns it at the cap whenever
/// the router sends it below the cap.
fn tiled_text_page(line: &RgbImage) -> RgbImage {
    let (width, height) = (700, 950);
    let strip = imageops::crop_imm(line, 0, 0, line.width().min(width - 40), line.height()).to_image();
    let mut page = RgbImage::from_pixel(width, height, Rgb([255; 3]));
    let mut y = 40;
    while y + strip.height() + 40 <= height {
        imageops::replace(&mut page, &strip, 20, i64::from(y));
        y += strip.height() + 8;
    }
    page
}

/// A 480 x 640 page with wide margins (as tests/margin_crop.rs).
fn margin_page() -> RgbImage {
    RgbImage::from_fn(480, 640, |x, y| {
        let text = (100..380).contains(&x) && (120..512).contains(&y) && (y - 120) % 20 < 12 && (x - 100) % 35 < 28;
        Rgb(if text { [25, 22, 20] } else { [247, 245, 238] })
    })
}

/// A route without its timings.
fn untimed(route: &Option<Route>) -> Option<Route> {
    route.clone().map(|mut route| {
        route.statistics_ms = 0.0;
        if let Some(attempt) = route.safety_net.as_mut() {
            attempt.total_ms = 0.0;
        }
        route
    })
}

/// Every page's result in input order through `run` (a streaming entry
/// point with a callback).
fn collect(run: impl FnOnce(&mut dyn FnMut(usize, anyhow::Result<OcrResult>) -> ControlFlow<()>)) -> Vec<OcrResult> {
    let mut results = Vec::new();
    run(&mut |index, result| {
        assert_eq!(index, results.len(), "pages arrive in input order");
        results.push(result.unwrap());
        ControlFlow::Continue(())
    });
    results
}

/// `actual` is `expected` page for page: tokens, stop, input, crop, route.
fn assert_same_pages(label: &str, actual: &[OcrResult], expected: &[OcrResult]) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (result, reference)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(result.token_ids, reference.token_ids, "{label}, page {i}");
        assert_eq!(result.finish_reason, reference.finish_reason, "{label}, page {i}");
        assert_eq!(
            (result.width, result.height, result.input_tokens),
            (reference.width, reference.height, reference.input_tokens),
            "{label}, page {i}"
        );
        assert_eq!(result.crop, reference.crop, "{label}, page {i}");
        assert_eq!(untimed(&result.route), untimed(&reference.route), "{label}, page {i}");
    }
}

/// `falcon-ocr run --batch-size N` and `--pipeline` keep every page's tokens:
/// continuous batching (`recognize_files_streaming`) at batch sizes 2, 4 and
/// 8, where rows refill as pages finish and a page that nothing would join
/// runs alone, and the page pipeline (`recognize_files_pipelined`) one page
/// at a time and over rows of 3, with a one-thread second pool beside a
/// one-thread decode team, all against sequential single pages. The same
/// runs cover routed pages whose safety net reruns them at the cap, and the
/// margin crop.
#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn continuous_batches_and_the_pipeline_keep_every_pages_tokens() {
    let images = fixture_images();
    let text_page = tiled_text_page(&images[1]);
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let directory = tempfile::tempdir().unwrap();
    let save = |name: &str, image: &RgbImage| -> PathBuf {
        let path = directory.path().join(name);
        image.save(&path).unwrap();
        path
    };
    // Seven pages: batches refill, and the last page runs alone.
    let pages: Vec<PathBuf> = (0..7)
        .map(|i| save(&format!("page-{i}.png"), &images[i % images.len()]))
        .collect();
    let routed_pages = [
        save("routed-text.png", &text_page),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images/rgb.png"),
        save("routed-1024.png", &router_page(700, 950, 1)),
    ];
    // A safety net needs a page routed below the cap. The router reads only
    // the page, so this holds whatever the weights; checked before any page
    // runs so a failure names the fixture, not the model.
    let below_cap = routed_pages
        .iter()
        .map(|path| {
            let page = image::open(path).unwrap().to_rgb8();
            router::route(&first_resize_rgb(&page, 64, router::CAP).unwrap()).max_dimension
        })
        .collect::<Vec<_>>();
    assert!(
        below_cap.iter().any(|&size| size < router::CAP),
        "no routed fixture page routes below the cap: {below_cap:?}"
    );
    let cropped_pages = [save("margins.png", &margin_page()), pages[1].clone()];
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    let routed = GenerationOptions {
        route: true,
        max_new_tokens: 24,
        ..Default::default()
    };
    let cropped = GenerationOptions {
        crop_margins: Some(24),
        ..options.clone()
    };
    let runs = [
        (&options, &pages[..]),
        (&routed, &routed_pages[..]),
        (&cropped, &cropped_pages[..]),
    ];
    let config = |batch_size: usize, decode_threads: Option<DecodeThreads>| RunnerConfig {
        threads: 4,
        batch_size,
        decode_threads: decode_threads.unwrap_or(RunnerConfig::reference().decode_threads),
        ..RunnerConfig::reference()
    };
    let sequential = Runner::new(model.clone(), "artifacts/model", config(1, None)).unwrap();
    let expected: Vec<Vec<OcrResult>> = runs
        .iter()
        .map(|(options, paths)| {
            paths
                .iter()
                .map(|path| sequential.recognize_file(path, options).unwrap())
                .collect()
        })
        .collect();
    let stops: std::collections::BTreeSet<usize> = expected[0].iter().map(|r| r.output_tokens).collect();
    assert!(stops.len() > 1, "the fixture must exercise mixed stop lengths");
    assert!(
        expected[1]
            .iter()
            .any(|r| r.route.as_ref().is_some_and(|route| route.safety_net.is_some())),
        "a routed page (the tiled text above all) must end by length or repetition below the cap so the \
         safety net reruns it"
    );
    assert!(expected[2][0].crop.is_some(), "the margin page must be cropped");
    for size in [2, 4, 8] {
        let runner = Runner::new(model.clone(), "artifacts/model", config(size, None)).unwrap();
        for ((options, paths), expected) in runs.iter().zip(&expected) {
            let actual = collect(|on_page| runner.recognize_files_streaming(paths, options, on_page).unwrap());
            assert_same_pages(&format!("streaming, batch {size}"), &actual, expected);
        }
    }
    let overlap = Pipeline {
        prefill_threads: Some(1),
    };
    for size in [1, 3] {
        let runner = Runner::new(
            model.clone(),
            "artifacts/model",
            config(size, Some(DecodeThreads::Fixed(1))),
        )
        .unwrap();
        for ((options, paths), expected) in runs.iter().zip(&expected) {
            let actual = collect(|on_page| {
                runner
                    .recognize_files_pipelined(paths, options, &overlap, on_page)
                    .unwrap()
            });
            assert_same_pages(&format!("pipelined, batch {size}"), &actual, expected);
        }
    }
}

/// The same guarantee in the configuration `falcon-ocr run` uses, with the
/// published packed near-exact and fast files: the automatic configuration
/// (screened head, speculation with the published draft head, the decode
/// tuner), where rows decode without drafts and a page that runs alone
/// drafts. Continuous batching (rows of 3), the page pipeline (one page at a
/// time and over rows of 3, with a fixed decode team so the next page
/// prefills beside it) and the router's safety net must give each page the
/// tokens of a sequential run.
#[test]
#[ignore = "requires the published packed files in artifacts/packed"]
fn default_config_batches_and_the_pipeline_keep_every_pages_tokens_in_both_quantized_modes() {
    let images = fixture_images();
    let directory = tempfile::tempdir().unwrap();
    let save = |name: &str, image: &RgbImage| -> PathBuf {
        let path = directory.path().join(name);
        image.save(&path).unwrap();
        path
    };
    let pages: Vec<PathBuf> = (0..7)
        .map(|i| save(&format!("page-{i}.png"), &images[i % images.len()]))
        .collect();
    let routed_pages = [
        save("routed-text.png", &tiled_text_page(&images[1])),
        save("routed-1024.png", &router_page(700, 950, 1)),
    ];
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    let routed = GenerationOptions {
        route: true,
        max_new_tokens: 24,
        ..Default::default()
    };
    let runs = [(&options, &pages[..]), (&routed, &routed_pages[..])];
    let packed = Path::new("artifacts/packed");
    for mode in ["near-exact", "fast"] {
        let model =
            Arc::new(Model::load_packed(packed.join(format!("falcon-ocr-v1.5-{mode}.safetensors")), false).unwrap());
        let config = |batch_size: usize| {
            falcon_ocr::auto::with_default_draft_head(
                RunnerConfig {
                    batch_size,
                    ..RunnerConfig::default()
                },
                packed,
                false,
            )
        };
        assert!(
            config(1).draft_head.is_some(),
            "the published draft head must be in {}",
            packed.display()
        );
        let sequential = Runner::new(model.clone(), packed, config(1)).unwrap();
        let expected: Vec<Vec<OcrResult>> = runs
            .iter()
            .map(|(options, paths)| {
                paths
                    .iter()
                    .map(|path| sequential.recognize_file(path, options).unwrap())
                    .collect()
            })
            .collect();
        drop(sequential);
        let batched = Runner::new(model.clone(), packed, config(3)).unwrap();
        for ((options, paths), expected) in runs.iter().zip(&expected) {
            let actual = collect(|on_page| batched.recognize_files_streaming(paths, options, on_page).unwrap());
            assert_same_pages(&format!("{mode}, streaming, batch 3"), &actual, expected);
        }
        drop(batched);
        // A fixed team on 8 threads leaves the second pool 4 threads from
        // the first page on, whatever the host (tokens never depend on either).
        for size in [1, 3] {
            let pinned = RunnerConfig {
                threads: 8,
                decode_threads: DecodeThreads::Fixed(4),
                ..config(size)
            };
            let runner = Runner::new(model.clone(), packed, pinned).unwrap();
            for ((options, paths), expected) in runs.iter().zip(&expected) {
                let actual = collect(|on_page| {
                    runner
                        .recognize_files_pipelined(paths, options, &Pipeline::default(), on_page)
                        .unwrap()
                });
                assert_same_pages(&format!("{mode}, pipelined, batch {size}"), &actual, expected);
            }
        }
    }
}
