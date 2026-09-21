//! Integration contract for the explicitly experimental BF16 path. Numerical
//! GPU bounds are evaluated independently by the frozen reference tools.
use falcon_ocr::{
    Backend, Bf16Model, Bf16Runner, FinishReason, GenerationOptions, RunnerConfig, trace::Trace,
};
use half::bf16;
use std::sync::Arc;

#[derive(Default)]
struct CastTrace {
    logits: usize,
    lse: usize,
    tensors: usize,
}
impl Trace for CastTrace {
    fn tensor(&mut self, name: &str, shape: &[usize], values: &[f32]) -> anyhow::Result<()> {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "shape {name}"
        );
        assert!(values.iter().all(|v| v.is_finite()), "nonfinite {name}");
        if name.ends_with(".lse") {
            self.lse += 1;
        } else {
            assert!(
                values.iter().all(|&v| bf16::from_f32(v).to_f32() == v),
                "missing BF16 cast {name}"
            );
        }
        if name.ends_with(".logits") {
            self.logits += 1;
        }
        self.tensors += 1;
        Ok(())
    }
}

#[test]
#[ignore = "requires pinned checkpoint and canonical BF16 GPU fixture"]
fn bf16_graph_exposes_casts_provenance_and_exact_smoke_generation() {
    let model = Arc::new(Bf16Model::load("artifacts/model").unwrap());
    assert_eq!(model.weight_tensor_bytes(), 539_888_832);
    let expected = [
        561, 791, 1169, 1135, 6830, 864, 18974, 535, 3358, 524, 864, 540, 541, 542, 543, 544, 263,
    ];
    for backend in [Backend::Scalar, Backend::Auto] {
        let runner = Bf16Runner::new(
            model.clone(),
            "artifacts/model",
            RunnerConfig {
                threads: 4,
                backend,
                ..Default::default()
            },
        )
        .unwrap();
        let mut trace = CastTrace::default();
        let traced = runner
            .trace_reference(
                "artifacts/reference/smoke-bf16/trace.safetensors",
                17,
                &mut trace,
            )
            .unwrap();
        assert_eq!(trace.logits, 17);
        assert_eq!(trace.lse, 17 * 22);
        assert!(
            traced.experimental && !traced.gpu_parity_qualified && traced.result.teacher_forced
        );
        assert_eq!(traced.result.precision, "bf16");
        assert_eq!(
            traced.fixture_sha256.as_deref(),
            Some("e938f90ad6964d6de89f0c2df1d7be9297c57e7516a7391d8061eb9f54c6f2d3")
        );
        assert_eq!(traced.teacher_tokens, expected);
        assert!(
            runner
                .trace_reference(
                    "artifacts/reference/smoke-bf16/trace.safetensors",
                    18,
                    &mut CastTrace::default()
                )
                .unwrap_err()
                .to_string()
                .contains("only 17 teacher tokens")
        );
        let generated = runner
            .recognize_file(
                "artifacts/reference/smoke-fp32/canonical-rgb.png",
                &GenerationOptions {
                    max_dimension: 256,
                    max_new_tokens: 24,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!generated.result.teacher_forced && generated.fixture_sha256.is_none());
        assert_eq!(generated.result.token_ids, expected);
        assert_eq!(generated.result.finish_reason, FinishReason::Eos);
        assert_eq!(
            generated.result.text,
            "Falcon OCR\n\nHello, world!\n\n12345"
        );
    }
}
