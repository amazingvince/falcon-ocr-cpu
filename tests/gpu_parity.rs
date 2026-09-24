//! Integration tests require the explicitly downloaded pinned model/reference.
//! Run `cargo test --release --test gpu_parity -- --ignored` after GPU export.
use falcon_ocr::{Backend, FinishReason, GenerationOptions, Model, Runner, RunnerConfig, kernels::Simd};
use std::{path::Path, sync::Arc};

#[test]
#[ignore = "requires pinned model and generated strict GPU reference fixture"]
fn free_running_cpu_matches_gpu_smoke_tokens_and_stop() {
    let model_dir = Path::new("artifacts/model");
    let reference = Path::new("artifacts/reference/smoke-fp32");
    let metadata: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(reference.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["precision"], "fp32");
    assert_eq!(metadata["tf32"], false);
    assert_eq!(metadata["teacher_forced"], false);
    assert_eq!(metadata["model_revision"], falcon_ocr::config::MODEL_REVISION);
    let expected: Vec<u32> = serde_json::from_value(metadata["token_ids"].clone()).unwrap();
    let model = Arc::new(Model::load(model_dir).unwrap());
    let runner = Runner::new(
        model.clone(),
        model_dir,
        RunnerConfig {
            threads: 4,
            batch_size: 1,
            ..RunnerConfig::reference()
        },
    )
    .unwrap();
    let options = GenerationOptions {
        max_dimension: metadata["max_dimension"].as_u64().unwrap() as u32,
        min_dimension: metadata["min_dimension"].as_u64().unwrap() as u32,
        max_new_tokens: metadata["max_new_tokens"].as_u64().unwrap() as usize,
        fit_budget: false,
    };
    let result = runner
        .recognize_file(reference.join("canonical-rgb.png"), &options)
        .unwrap();
    assert_eq!(result.token_ids, expected);
    assert_eq!(result.finish_reason, FinishReason::Eos);
    let projection = result.timings.image_projection_ms.unwrap();
    let transformer = result.timings.transformer_prefill_ms.unwrap();
    assert!(projection.is_finite() && projection >= 0.0);
    assert!(transformer.is_finite() && transformer >= 0.0);
    assert!(projection + transformer <= result.timings.prefill_ms + 1e-6);
    let limited = runner
        .recognize_file(
            reference.join("canonical-rgb.png"),
            &GenerationOptions {
                max_new_tokens: 3,
                ..options.clone()
            },
        )
        .unwrap();
    assert_eq!(limited.token_ids, &expected[..3]);
    assert_eq!(limited.finish_reason, FinishReason::Length);
    for (backend, simd) in [(Backend::Avx2, Simd::Avx2), (Backend::Avx512, Simd::Avx512)] {
        if simd.validate().is_err() {
            continue;
        }
        let runner = Runner::new(
            model.clone(),
            model_dir,
            RunnerConfig {
                threads: 1,
                backend,
                ..RunnerConfig::reference()
            },
        )
        .unwrap();
        let result = runner
            .recognize_file(reference.join("canonical-rgb.png"), &options)
            .unwrap();
        assert_eq!(result.token_ids, expected, "explicit backend {backend:?}");
    }
}
