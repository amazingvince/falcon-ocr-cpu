//! Batch tracing must preserve every request, including singleton chunks.
use std::{collections::BTreeMap, sync::Arc};

use falcon_ocr::{
    Backend, CacheLayout, GenerationOptions, Model, Runner, RunnerConfig, trace::Trace,
};
use image::{Rgb, RgbImage, imageops};

#[derive(Default)]
struct UniqueNames {
    tensors: BTreeMap<String, Vec<usize>>,
    starts: usize,
    ends: usize,
}
impl Trace for UniqueNames {
    fn tensor(&mut self, name: &str, shape: &[usize], _: &[f32]) -> anyhow::Result<()> {
        assert!(
            self.tensors
                .insert(name.to_owned(), shape.to_vec())
                .is_none(),
            "duplicate tensor name: {name}"
        );
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
                ..Default::default()
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
    assert!(
        !single_trace
            .tensors
            .keys()
            .any(|key| key.starts_with("request."))
    );
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
        assert_eq!(
            (trace.starts, trace.ends),
            (expected_chunks, expected_chunks)
        );
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
