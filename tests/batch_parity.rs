//! Compare independent sequential requests against bounded mixed-batch execution.
use falcon_ocr::{Backend, GenerationOptions, Model, Runner, RunnerConfig};
use image::{Rgb, RgbImage, imageops};
use std::sync::Arc;

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn mixed_batches_preserve_tokens_order_stops_and_limits() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let one_line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    let images = [original.clone(), one_line, blank, original];
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
            ..Default::default()
        },
    )
    .unwrap();
    let expected = images
        .iter()
        .map(|image| sequential.recognize(image, &options).unwrap())
        .collect::<Vec<_>>();
    assert!(
        expected
            .iter()
            .any(|r| r.output_tokens != expected[0].output_tokens),
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
                ..Default::default()
            },
        )
        .unwrap();
        // A partial final chunk also exercises output indexing and order.
        let input = (0..7)
            .map(|i| images[i % images.len()].clone())
            .collect::<Vec<_>>();
        let actual = runner.recognize_batch(&input, &options).unwrap();
        assert_eq!(actual.len(), input.len());
        for (i, result) in actual.iter().enumerate() {
            let reference = &expected[i % expected.len()];
            assert_eq!(
                result.token_ids, reference.token_ids,
                "batch {size}, request {i}"
            );
            assert_eq!(result.finish_reason, reference.finish_reason);
            assert_eq!(
                (result.width, result.height),
                (reference.width, reference.height)
            );
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
            assert_eq!(
                result.token_ids,
                &expected.token_ids[..3.min(expected.token_ids.len())]
            );
            assert!(result.output_tokens <= 3);
        }
    }
}
