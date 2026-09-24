//! The screened vocabulary head must select exactly the tokens of the full head.
use falcon_ocr::{GenerationOptions, HeadMode, Model, Runner, RunnerConfig};
use std::sync::Arc;

fn runner(model: &Arc<Model>, batch_size: usize, head: HeadMode) -> Runner {
    let mut runner = Runner::new(
        model.clone(),
        "artifacts/model",
        RunnerConfig {
            threads: 8,
            batch_size,
            ..RunnerConfig::reference()
        },
    )
    .unwrap();
    runner.set_head_mode(head).unwrap();
    runner
}

#[test]
#[ignore = "requires pinned model and corpus pages"]
fn screened_head_selects_full_head_tokens() {
    let model = Arc::new(Model::load("artifacts/model").unwrap());
    let pages = [
        "artifacts/reference/smoke-fp32/canonical-rgb.png",
        "artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png",
        "artifacts/corpus/smoke/ebac2ad1cac11a99/canonical-rgb.png",
        "artifacts/corpus/smoke/bc2882dcec9a3e02/canonical-rgb.png",
    ];
    let images: Vec<_> = pages.iter().map(|p| image::open(p).unwrap().to_rgb8()).collect();
    let options = GenerationOptions {
        max_dimension: 512,
        max_new_tokens: 192,
        ..Default::default()
    };
    let full = runner(&model, 1, HeadMode::Full);
    let screened = runner(&model, 1, HeadMode::Screened);
    let mut expected = Vec::new();
    for image in &images {
        let a = full.recognize(image, &options).unwrap();
        let b = screened.recognize(image, &options).unwrap();
        assert_eq!(a.token_ids, b.token_ids);
        assert_eq!(a.finish_reason, b.finish_reason);
        expected.push(a);
    }
    let batch = runner(&model, 4, HeadMode::Screened)
        .recognize_batch(&images, &options)
        .unwrap();
    for (a, b) in expected.iter().zip(&batch) {
        assert_eq!(a.token_ids, b.token_ids);
        assert_eq!(a.finish_reason, b.finish_reason);
    }
}
