//! Compact storage must preserve the expanded-cache model's tensor bits.
use std::sync::Arc;

use falcon_ocr::{
    Backend, CacheLayout, GenerationOptions, Model, Runner, RunnerConfig,
    trace::{TensorTrace, Trace},
};
use image::{Rgb, RgbImage, imageops};

#[derive(Default)]
struct CheckedTrace {
    tensors: TensorTrace,
    duplicate_head_tensors: usize,
}
impl Trace for CheckedTrace {
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
        // Values duplicate before and after RoPE; generated keys duplicate after
        // RoPE because their spatial coordinates are NaN (zero spatial angle).
        // Prefix keys are intentionally excluded: image rotations differ by head.
        if name.ends_with(".v") || (name.ends_with(".k") && name.contains("decode.")) {
            assert_eq!(shape.len(), 3, "{name}");
            assert_eq!((shape[1], shape[2]), (16, 64), "{name}");
            for row in data.chunks_exact(16 * 64) {
                for group in row.chunks_exact(2 * 64) {
                    assert!(
                        group[..64]
                            .iter()
                            .zip(&group[64..])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{name}: duplicate GQA heads must match before reduction"
                    );
                }
            }
            self.duplicate_head_tensors += 1;
        }
        self.tensors.tensor(name, shape, data)
    }
}

fn compare(reference: &CheckedTrace, actual: &CheckedTrace) {
    assert_eq!(
        reference.tensors.tensors.len(),
        actual.tensors.tensors.len()
    );
    assert_eq!(
        reference.duplicate_head_tensors,
        actual.duplicate_head_tensors
    );
    assert!(actual.duplicate_head_tensors > 0);
    for (name, (shape, values)) in &reference.tensors.tensors {
        let (actual_shape, actual_values) = &actual.tensors.tensors[name];
        assert_eq!(actual_shape, shape, "{name}");
        assert!(
            values
                .iter()
                .zip(actual_values)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: compact cache changed tensor bits"
        );
    }
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn compact_cache_preserves_single_and_mixed_batch_traces() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let runner = |layout, batch_size| {
        Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                batch_size,
                backend: Backend::Avx2,
                cache_layout: layout,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let mut expanded_trace = CheckedTrace::default();
    let expanded = runner(CacheLayout::Expanded, 1)
        .trace_reference(
            "artifacts/reference/smoke-fp32/trace.safetensors",
            17,
            &mut expanded_trace,
        )
        .unwrap();
    let mut compact_trace = CheckedTrace::default();
    let compact = runner(CacheLayout::Compact, 1)
        .trace_reference(
            "artifacts/reference/smoke-fp32/trace.safetensors",
            17,
            &mut compact_trace,
        )
        .unwrap();
    assert_eq!(compact.cache_layout, CacheLayout::Compact);
    assert_eq!(expanded.cache_layout, CacheLayout::Expanded);
    assert_eq!(compact.token_ids, expanded.token_ids);
    compare(&expanded_trace, &compact_trace);
    println!(
        "Single canonical trace: {} bit-identical tensors; {} duplicate-head tensors checked",
        compact_trace.tensors.tensors.len(),
        compact_trace.duplicate_head_tensors
    );
    drop(expanded_trace);
    drop(compact_trace);

    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let one_line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    let images = [original.clone(), blank, one_line, original];
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    let mut expanded_trace = CheckedTrace::default();
    let expanded = runner(CacheLayout::Expanded, 4)
        .recognize_batch_with_trace(&images, &options, &mut expanded_trace)
        .unwrap();
    let mut compact_trace = CheckedTrace::default();
    let compact = runner(CacheLayout::Compact, 4)
        .recognize_batch_with_trace(&images, &options, &mut compact_trace)
        .unwrap();
    for (actual, expected) in compact.iter().zip(&expanded) {
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.finish_reason, expected.finish_reason);
        assert_eq!(actual.text, expected.text);
        assert_eq!(actual.cache_layout, CacheLayout::Compact);
    }
    compare(&expanded_trace, &compact_trace);
    println!(
        "Mixed batch trace: {} bit-identical tensors; {} duplicate-head tensors checked",
        compact_trace.tensors.tensors.len(),
        compact_trace.duplicate_head_tensors
    );
}
