//! The explicit weight layout must preserve complete model traces and outputs.
use std::{collections::BTreeMap, sync::Arc};

use falcon_ocr::{Backend, CacheLayout, GenerationOptions, Model, Runner, RunnerConfig, WeightLayout, trace::Trace};
use image::{Rgb, RgbImage, imageops};
use sha2::{Digest, Sha256};

#[derive(Default)]
struct HashTrace {
    tensors: BTreeMap<String, (Vec<usize>, [u8; 32])>,
    decode_rows: Vec<usize>,
}
impl Trace for HashTrace {
    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
        assert!(data.iter().all(|v| v.is_finite()), "nonfinite tensor {name}");
        if name.starts_with("batch.") && name.ends_with(".embedding") {
            self.decode_rows.push(shape[0]);
        }
        let digest = Sha256::digest(bytemuck::cast_slice(data)).into();
        assert!(
            self.tensors.insert(name.to_owned(), (shape.to_vec(), digest)).is_none(),
            "duplicate tensor {name}"
        );
        Ok(())
    }
}

fn compare(a: &HashTrace, b: &HashTrace) {
    assert_eq!(a.tensors.len(), b.tensors.len());
    assert_eq!(a.decode_rows, b.decode_rows);
    for (name, expected) in &a.tensors {
        assert_eq!(b.tensors.get(name), Some(expected), "tensor bits differ: {name}");
    }
}

#[test]
#[ignore = "requires pinned model and strict GPU reference fixture"]
fn phase_packed_preserves_single_mixed_and_full_batch_traces() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    assert_eq!(model.packed_weight_bytes(), 0);
    let runner = |layout, cache_layout, batch_size| {
        Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 2,
                batch_size,
                backend: Backend::Avx2,
                cache_layout,
                weight_layout: layout,
                ..RunnerConfig::reference()
            },
        )
        .unwrap()
    };
    let unpacked = runner(WeightLayout::Unpacked, CacheLayout::Expanded, 1);
    assert_eq!(
        model.packed_weight_bytes(),
        0,
        "default must not allocate packed weights"
    );
    let packed = runner(WeightLayout::PhasePacked, CacheLayout::Expanded, 1);
    assert_eq!(model.packed_weight_bytes(), 876_085_248);
    let packing_ms = model.weight_packing_ms();
    assert!(packing_ms > 0.0);
    let mut expected_trace = HashTrace::default();
    let mut actual_trace = HashTrace::default();
    let fixture = "artifacts/reference/smoke-fp32/trace.safetensors";
    let expected = unpacked.trace_reference(fixture, 17, &mut expected_trace).unwrap();
    let actual = packed.trace_reference(fixture, 17, &mut actual_trace).unwrap();
    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.text, expected.text);
    assert!(actual.teacher_forced);
    assert_eq!(actual.weight_layout, WeightLayout::PhasePacked);
    assert_eq!(actual.packed_weight_bytes, model.packed_weight_bytes());
    compare(&expected_trace, &actual_trace);
    println!(
        "Canonical single-row trace: {} bit-identical tensors",
        actual_trace.tensors.len()
    );

    let original = image::open("artifacts/reference/smoke-fp32/canonical-rgb.png")
        .unwrap()
        .to_rgb8();
    let one_line = imageops::crop_imm(&original, 0, 0, original.width(), 48).to_image();
    let blank = RgbImage::from_pixel(128, 64, Rgb([255; 3]));
    let mixed = [original.clone(), blank, one_line];
    let options = GenerationOptions {
        max_dimension: 256,
        max_new_tokens: 24,
        ..Default::default()
    };
    for cache_layout in [CacheLayout::Expanded, CacheLayout::Compact] {
        let mut a = HashTrace::default();
        let mut b = HashTrace::default();
        let expected = runner(WeightLayout::Unpacked, cache_layout, 4)
            .recognize_batch_with_trace(&mixed, &options, &mut a)
            .unwrap();
        let actual = runner(WeightLayout::PhasePacked, cache_layout, 4)
            .recognize_batch_with_trace(&mixed, &options, &mut b)
            .unwrap();
        assert!(
            b.decode_rows.contains(&2) && b.decode_rows.contains(&1),
            "fixture must exercise compacted batch rows and row1 fallback: {:?}",
            b.decode_rows
        );
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.token_ids, expected.token_ids);
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.finish_reason, expected.finish_reason);
            assert!(!actual.teacher_forced);
        }
        compare(&a, &b);
        println!(
            "Mixed {cache_layout:?}: {} bit-identical tensors, active rows {:?}",
            b.tensors.len(),
            b.decode_rows
        );
    }
    // Force full 2/4/8-row projections, including the vocabulary head. Short
    // free-running generations keep this operator integration check bounded.
    let short = GenerationOptions {
        max_new_tokens: 3,
        ..options
    };
    for rows in [2, 4, 8] {
        let pages = vec![original.clone(); rows];
        let mut a = HashTrace::default();
        let mut b = HashTrace::default();
        let expected = runner(WeightLayout::Unpacked, CacheLayout::Expanded, rows)
            .recognize_batch_with_trace(&pages, &short, &mut a)
            .unwrap();
        let actual = runner(WeightLayout::PhasePacked, CacheLayout::Expanded, rows)
            .recognize_batch_with_trace(&pages, &short, &mut b)
            .unwrap();
        assert!(b.decode_rows.iter().all(|&n| n == rows));
        assert_eq!(
            expected.iter().map(|r| &r.token_ids).collect::<Vec<_>>(),
            actual.iter().map(|r| &r.token_ids).collect::<Vec<_>>()
        );
        compare(&a, &b);
        println!("Full batch {rows}: {} bit-identical tensors", b.tensors.len());
    }
    assert_eq!(
        model.weight_packing_ms().to_bits(),
        packing_ms.to_bits(),
        "shared Model must never repack per Runner"
    );
}
