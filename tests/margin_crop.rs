//! The margin crop (`GenerationOptions::crop_margins`) through every runner
//! entry point: a page with wide margins reports the crop that preprocessing
//! computes and runs on fewer image tokens than the whole page. Needs the pinned
//! checkpoint and tokenizer in `artifacts/model`: run with --ignored.
use falcon_ocr::{
    GenerationOptions, Model, OcrResult, Pipeline, Runner, RunnerConfig, preprocess::margin_crop, runner::ControlFlow,
};
use image::{Rgb, RgbImage};
use std::sync::Arc;

/// A 480 x 640 page (inside the default bounds, so its first resize is the
/// page itself) with dark text-like blocks over (100, 120)-(373, 512).
fn page() -> RgbImage {
    RgbImage::from_fn(480, 640, |x, y| {
        let text = (100..380).contains(&x) && (120..512).contains(&y) && (y - 120) % 20 < 12 && (x - 100) % 35 < 28;
        Rgb(if text { [25, 22, 20] } else { [247, 245, 238] })
    })
}

/// A streaming callback that keeps every page's result under `entry`.
fn keep<'a>(
    entry: &'static str,
    results: &'a mut Vec<(&'static str, OcrResult)>,
) -> impl FnMut(usize, anyhow::Result<OcrResult>) -> ControlFlow<()> + 'a {
    move |_, result| {
        results.push((entry, result.unwrap()));
        ControlFlow::Continue(())
    }
}

#[test]
#[ignore = "requires the pinned checkpoint and tokenizer in artifacts/model"]
fn pages_with_margins_report_their_crop_on_every_entry_point() {
    let model = Arc::new(Model::load("artifacts/model").expect("install the pinned assets in artifacts/model"));
    let config = RunnerConfig {
        threads: 0,
        batch_size: 2,
        ..RunnerConfig::reference()
    };
    let runner = Runner::new(model.clone(), "artifacts/model", config.clone()).unwrap();
    let single = Runner::new(
        model,
        "artifacts/model",
        RunnerConfig {
            batch_size: 1,
            ..config
        },
    )
    .unwrap();
    let page = page();
    let expected = margin_crop(&page, 24);
    assert!(expected.is_some(), "the test page must have margins to crop");
    let whole = GenerationOptions {
        max_new_tokens: 1,
        ..GenerationOptions::default()
    };
    let cropped = GenerationOptions {
        crop_margins: Some(24),
        ..whole.clone()
    };
    let uncropped = runner.recognize(&page, &whole).unwrap();
    assert_eq!(uncropped.crop, None);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("page.png");
    page.save(&path).unwrap();
    let routed = GenerationOptions {
        route: true,
        ..cropped.clone()
    };
    let mut results = vec![
        ("recognize", runner.recognize(&page, &cropped).unwrap()),
        ("recognize_file", runner.recognize_file(&path, &cropped).unwrap()),
        // Routed (and rerun at 1536 by the safety net unless the first token
        // ends the page): the same page at every size, so the same crop.
        ("routed", runner.recognize(&page, &routed).unwrap()),
    ];
    let pages = [page.clone(), page.clone()];
    for (entry, batch) in [
        ("recognize_batch", runner.recognize_batch(&pages, &cropped)),
        ("recognize_files", runner.recognize_files(&[&path, &path], &cropped)),
    ] {
        results.extend(batch.unwrap().into_iter().map(|result| (entry, result)));
    }
    // The streaming entry points: cohorts of two with the third page alone,
    // and the pipeline one page at a time with a second pool that prefills
    // the next page.
    let paths = [&path, &path, &path];
    let overlap = Pipeline {
        prefill_threads: Some(1),
    };
    runner
        .recognize_files_streaming(&paths, &cropped, keep("recognize_files_streaming", &mut results))
        .unwrap();
    single
        .recognize_files_pipelined(
            &paths,
            &cropped,
            &overlap,
            keep("recognize_files_pipelined", &mut results),
        )
        .unwrap();
    assert_eq!(results.len(), 3 + 2 + 2 + 3 * 2);
    for (entry, result) in &results {
        assert_eq!(result.crop, expected, "{entry}");
        assert!(
            result.input_tokens < uncropped.input_tokens,
            "{entry}: {} input tokens, {} uncropped",
            result.input_tokens,
            uncropped.input_tokens
        );
        assert!(result.width < uncropped.width, "{entry}");
        assert!(result.height < uncropped.height, "{entry}");
    }
}
