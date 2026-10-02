//! Batch tracing must preserve every request, including singleton chunks.
use std::{collections::BTreeMap, ops::ControlFlow, path::Path, sync::Arc};

use falcon_ocr::{Backend, CacheLayout, GenerationOptions, Model, Runner, RunnerConfig, trace::Trace};
use image::{Rgb, RgbImage, imageops};

#[derive(Default)]
struct UniqueNames {
    tensors: BTreeMap<String, Vec<usize>>,
    /// The values of every decode step's `request_indices`.
    requests: Vec<f32>,
    starts: usize,
    ends: usize,
}
impl Trace for UniqueNames {
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
        assert!(
            self.tensors.insert(name.to_owned(), shape.to_vec()).is_none(),
            "duplicate tensor name: {name}"
        );
        if name.ends_with(".request_indices") {
            self.requests.extend_from_slice(data);
        }
        Ok(())
    }
    fn decode_start(&mut self) {
        self.starts += 1;
    }
    fn decode_end(&mut self) {
        self.ends += 1;
    }
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn singleton_batch_chunks_keep_request_names_and_phase_callbacks() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let one_line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    let images = [original, blank, one_line];
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 3,
        ..Default::default()
    };
    let runner = |batch_size| {
        Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                batch_size,
                backend: Backend::Avx2,
                cache_layout: CacheLayout::Compact,
                ..RunnerConfig::reference()
            },
        )
        .unwrap()
    };
    let mut single_trace = UniqueNames::default();
    runner(1)
        .recognize_with_trace(&images[0], &options, &mut single_trace)
        .unwrap();
    assert!(single_trace.tensors.contains_key("prefill.embedding"));
    assert!(single_trace.tensors.contains_key("decode.0.embedding"));
    assert!(!single_trace.tensors.keys().any(|key| key.starts_with("request.")));
    assert_eq!((single_trace.starts, single_trace.ends), (1, 1));

    for batch_size in [1, 2] {
        let mut trace = UniqueNames::default();
        let results = runner(batch_size)
            .recognize_batch_with_trace(&images, &options, &mut trace)
            .unwrap();
        assert_eq!(results.len(), 3);
        for (index, result) in results.iter().enumerate() {
            assert_eq!(
                trace.tensors[&format!("request.{index}.prefill.embedding")],
                [result.input_tokens, 768]
            );
        }
        let expected_chunks = images.len().div_ceil(batch_size);
        assert_eq!((trace.starts, trace.ends), (expected_chunks, expected_chunks));
        assert!(!trace.tensors.contains_key("prefill.embedding"));
        assert!(trace.tensors.contains_key("request.2.decode.0.embedding"));
        if batch_size == 1 {
            assert!(trace.tensors.contains_key("request.0.decode.0.embedding"));
            assert!(trace.tensors.contains_key("request.1.decode.0.embedding"));
        } else {
            assert!(trace.tensors.contains_key("batch.0.decode.0.embedding"));
        }
    }
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn streamed_pages_are_traced_under_their_input_indices() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let original = Path::new("artifacts/reference/smoke-fp32/canonical-rgb.png");
    let directory = tempfile::tempdir().unwrap();
    let one_line = directory.path().join("one-line.png");
    let image = image::open(original).unwrap().to_rgb8();
    imageops::crop_imm(&image, 0, 0, image.width(), 48)
        .to_image()
        .save(&one_line)
        .unwrap();
    let missing = directory.path().join("missing.png");
    let (missing, one_line) = (missing.as_path(), one_line.as_path());
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 3,
        ..Default::default()
    };
    let stream = |batch_size, paths: &[&Path]| {
        let runner = Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                batch_size,
                backend: Backend::Avx2,
                cache_layout: CacheLayout::Compact,
                ..RunnerConfig::reference()
            },
        )
        .unwrap();
        let mut trace = UniqueNames::default();
        let mut results = Vec::new();
        runner
            .recognize_files_streaming_with_trace(paths, &options, &mut trace, |page, result| {
                results.push((page, result.ok()));
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(
            results.iter().map(|(page, _)| *page).collect::<Vec<_>>(),
            Vec::from_iter(0..paths.len())
        );
        // Every page that ran has its own prefill, named by its input index.
        for (page, result) in &results {
            if let Some(result) = result {
                assert_eq!(
                    trace.tensors[&format!("request.{page}.prefill.embedding")],
                    [result.input_tokens, 768]
                );
            }
        }
        assert!(!trace.tensors.contains_key("prefill.embedding"));
        (trace, results)
    };
    // The missing page leaves the cohort; the other two decode jointly,
    // traced under their own input indices, not their rows in the cohort.
    let (trace, results) = stream(3, &[missing, original, one_line]);
    assert!(results[0].1.is_none());
    assert!(!trace.tensors.keys().any(|key| key.starts_with("request.0.")));
    assert!(trace.tensors.contains_key("batch.1.decode.0.embedding"));
    let mut requests = trace.requests.clone();
    requests.sort_by(f32::total_cmp);
    requests.dedup();
    assert_eq!(requests, [1., 2.]);
    // Pages that run alone, left alone in their cohort or one per cohort,
    // decode under their own names too (a repeated name fails the trace).
    let (trace, _) = stream(2, &[missing, original, missing, one_line]);
    for page in [1, 3] {
        assert!(
            trace
                .tensors
                .contains_key(&format!("request.{page}.decode.0.embedding"))
        );
    }
    let (trace, _) = stream(1, &[original, one_line]);
    for page in [0, 1] {
        assert!(
            trace
                .tensors
                .contains_key(&format!("request.{page}.decode.0.embedding"))
        );
    }
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn safety_net_reruns_are_traced_apart_from_the_routed_attempt() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let page = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    // The smoke page routes to 768 pixels, and three tokens end it by length,
    // so the safety net reruns it at the cap.
    let options = GenerationOptions {
        route: true,
        max_new_tokens: 3,
        ..Default::default()
    };
    let runner = |batch_size| {
        Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                batch_size,
                backend: Backend::Avx2,
                cache_layout: CacheLayout::Compact,
                ..RunnerConfig::reference()
            },
        )
        .unwrap()
    };
    let mut trace = UniqueNames::default();
    let result = runner(1).recognize_with_trace(&page, &options, &mut trace).unwrap();
    assert!(result.route.unwrap().safety_net.is_some());
    assert!(trace.tensors.contains_key("prefill.embedding"));
    assert!(trace.tensors.contains_key("rerun.prefill.embedding"));
    for batch_size in [1, 2] {
        let mut trace = UniqueNames::default();
        let results = runner(batch_size)
            .recognize_batch_with_trace(&[page.clone(), page.clone()], &options, &mut trace)
            .unwrap();
        for (index, result) in results.iter().enumerate() {
            assert!(result.route.as_ref().unwrap().safety_net.is_some());
            assert!(
                trace
                    .tensors
                    .contains_key(&format!("request.{index}.rerun.prefill.embedding")),
                "batch size {batch_size}"
            );
        }
    }
}
