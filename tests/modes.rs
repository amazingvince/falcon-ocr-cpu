//! Mode gates on the pinned model and corpus (`artifacts/`): run with
//! `cargo test --release --locked --test modes -- --ignored`.
//!
//! Every test uses the automatic configuration (`RunnerConfig::default()`)
//! unless it says otherwise, so what `falcon-ocr run` does with no flags is
//! what is gated. The trace hashes stay under `RunnerConfig::reference()`.
use std::{path::Path, sync::Arc};

use falcon_ocr::{
    Backend, DecodeThreads, FinishReason, GenerationOptions, HeadMode, Mode, Model, Runner, RunnerConfig,
    auto::{ModelRequest, load_model, resolve_weights},
    quant::Profile,
    trace::TensorTrace,
};
use sha2::{Digest, Sha256};

const MODEL_DIR: &str = "artifacts/model";
const PACKED_DIR: &str = "artifacts/packed";
const SMOKE_DIR: &str = "artifacts/reference/smoke-fp32";
/// 6,544 input tokens: the page every timing in the docs quotes.
const JOURNAL: &str = "artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png";
/// 1,977 input tokens.
const SHORT: &str = "artifacts/corpus/v3/68a0f0e94a55d7ae/canonical-rgb.png";
/// The FP32 model itself runs into a repetition loop on this page.
const LOOP: &str = "artifacts/corpus/v3/95928a9b09ccac9e/canonical-rgb.png";
const SMOKE_TRACE_SHA256: &str = "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309";
const W16_TRACE_SHA256: &str = "40fd4fb413a60544592570229490853e58141f2df4a36c9e1881dd984507714d";
const W8_TRACE_SHA256: &str = "916b4df8fbbda7d1e9da73180350057816d88d069e4f82c0ca8332ce791c996b";

/// `tests/fixtures/modes/*.json`: the first tokens of a mode on a page under
/// the automatic configuration (regenerate with the CLI; see the README).
#[derive(serde::Deserialize)]
struct Fixture {
    page: String,
    mode: Mode,
    profile: Profile,
    overlay_sha256: Option<String>,
    max_dimension: u32,
    min_dimension: u32,
    max_new_tokens: usize,
    token_ids: Vec<u32>,
}

fn fixture(name: &str) -> Fixture {
    let text = std::fs::read_to_string(format!("tests/fixtures/modes/{name}.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn options(max_new_tokens: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens,
        min_dimension: 64,
        max_dimension: 1536,
        fit_budget: false,
        route: false,
    }
}

/// `mode` from the FP32 checkpoint, as `falcon-ocr run` loads it without a
/// packed file (fast mode with the GPTQ overlay).
fn checkpoint_model(mode: Mode) -> Arc<Model> {
    let plan = resolve_weights(&ModelRequest {
        model_dir: Path::new(MODEL_DIR),
        mode: Some(mode),
        split_exact: true,
        from_checkpoint: true,
        ..ModelRequest::default()
    })
    .unwrap();
    Arc::new(load_model(&plan, false).unwrap())
}

fn packed_model(mode: Mode) -> Arc<Model> {
    let path = Path::new(PACKED_DIR).join(mode.packed_file_name().unwrap());
    Arc::new(Model::load_packed(path, false).unwrap())
}

fn runner(model: Arc<Model>, tokenizer_dir: &str, config: RunnerConfig) -> Runner {
    Runner::new(model, tokenizer_dir, config).unwrap()
}

fn tokens(runner: &Runner, page: &str, max_new_tokens: usize) -> (Vec<u32>, FinishReason) {
    let result = runner.recognize_file(page, &options(max_new_tokens)).unwrap();
    (result.token_ids, result.finish_reason)
}

/// The `__metadata__` map of a safetensors file, from its header alone.
fn safetensors_metadata(path: &Path) -> serde_json::Map<String, serde_json::Value> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).unwrap();
    let mut header = vec![0u8; u64::from_le_bytes(prefix) as usize];
    file.read_exact(&mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    header["__metadata__"].as_object().unwrap().clone()
}

fn file_sha256(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

#[test]
#[ignore = "needs artifacts/model and the GPU smoke reference"]
fn exact_mode_under_the_automatic_config_matches_the_gpu_smoke_tokens() {
    let metadata: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(Path::new(SMOKE_DIR).join("metadata.json")).unwrap()).unwrap();
    let expected: Vec<u32> = serde_json::from_value(metadata["token_ids"].clone()).unwrap();
    let model = Arc::new(Model::load_mode(MODEL_DIR, Mode::Exact).unwrap());
    assert_eq!(model.profile(), Profile::SPLIT_F32);
    let runner = runner(model, MODEL_DIR, RunnerConfig::default());
    let result = runner
        .recognize_file(
            Path::new(SMOKE_DIR).join("canonical-rgb.png"),
            &GenerationOptions {
                max_dimension: metadata["max_dimension"].as_u64().unwrap() as u32,
                min_dimension: metadata["min_dimension"].as_u64().unwrap() as u32,
                max_new_tokens: metadata["max_new_tokens"].as_u64().unwrap() as usize,
                fit_budget: false,
                route: false,
            },
        )
        .unwrap();
    assert_eq!(result.token_ids, expected);
    assert_eq!(result.finish_reason, FinishReason::Eos);
    assert_eq!(result.mode, Some(Mode::Exact));
    assert_eq!(result.precision, "fp32");
}

#[test]
#[ignore = "needs artifacts/model with the GPTQ overlay"]
fn near_exact_and_fast_reproduce_the_recorded_journal_tokens() {
    for name in ["near-exact-journal-300", "fast-journal-300"] {
        let fixture = fixture(name);
        let model = checkpoint_model(fixture.mode);
        assert_eq!(model.profile(), fixture.profile, "{name}");
        assert_eq!(model.overlay_sha256(), fixture.overlay_sha256.as_deref(), "{name}");
        let runner = runner(model, MODEL_DIR, RunnerConfig::default());
        assert_eq!(runner.resolved().mode, Some(fixture.mode));
        let result = runner
            .recognize_file(
                &fixture.page,
                &GenerationOptions {
                    max_new_tokens: fixture.max_new_tokens,
                    min_dimension: fixture.min_dimension,
                    max_dimension: fixture.max_dimension,
                    fit_budget: false,
                    route: false,
                },
            )
            .unwrap();
        assert_eq!(result.token_ids, fixture.token_ids, "{name}");
        assert_eq!(result.precision, fixture.profile.label(), "{name}");
    }
}

/// HILLCLIMB T12: a kernel-ready file holds exactly what the checkpoint
/// loader computes, and the published files are those bytes.
#[test]
#[ignore = "needs artifacts/model, the GPTQ overlay, artifacts/packed and 1.5 GB of temporary disk"]
fn packed_files_round_trip_the_checkpoint_loader_and_match_the_published_ones() {
    let dir = tempfile::tempdir().unwrap();
    for mode in [Mode::NearExact, Mode::Fast] {
        let model = checkpoint_model(mode);
        model.prepare_screened_head().unwrap();
        let file = dir.path().join(mode.packed_file_name().unwrap());
        model.write_packed(&file).unwrap();
        let reloaded = Arc::new(Model::load_packed(&file, true).unwrap());
        assert_eq!(reloaded.profile(), model.profile());
        let published = Path::new(PACKED_DIR).join(mode.packed_file_name().unwrap());
        let (ours, theirs) = (safetensors_metadata(&file), safetensors_metadata(&published));
        for key in [
            "format",
            "profile",
            "weight_bits",
            "tensors_sha256",
            "source_sha256",
            "model_revision",
        ] {
            assert_eq!(ours[key], theirs[key], "{mode} packed metadata {key}");
        }
        let from_checkpoint = tokens(&runner(model, MODEL_DIR, RunnerConfig::default()), JOURNAL, 300);
        let from_file = tokens(&runner(reloaded, MODEL_DIR, RunnerConfig::default()), JOURNAL, 300);
        assert_eq!(from_file, from_checkpoint, "{mode}");
        let from_published = tokens(
            &runner(packed_model(mode), PACKED_DIR, RunnerConfig::default()),
            JOURNAL,
            300,
        );
        assert_eq!(from_published, from_checkpoint, "{mode} published file");
    }
}

/// Speculation, the decode team, cross-page drafts and the screened head
/// change timing only.
#[test]
#[ignore = "needs artifacts/packed"]
fn speculation_decode_team_drafts_and_head_never_change_tokens() {
    let model = packed_model(Mode::NearExact);
    let base = RunnerConfig::default();
    let expected = tokens(&runner(model.clone(), PACKED_DIR, base.clone()), JOURNAL, 300);
    assert_eq!(expected.1, FinishReason::Length);
    let variants = [
        (
            "no speculation",
            RunnerConfig {
                speculation: None,
                ..base.clone()
            },
        ),
        (
            "fixed decode team",
            RunnerConfig {
                decode_threads: DecodeThreads::Fixed(4),
                ..base.clone()
            },
        ),
        (
            "full head",
            RunnerConfig {
                head: HeadMode::Full,
                ..base.clone()
            },
        ),
        (
            "explicit avx2 backend",
            RunnerConfig {
                backend: Backend::Avx2,
                ..base.clone()
            },
        ),
    ];
    for (name, config) in variants {
        assert_eq!(
            tokens(&runner(model.clone(), PACKED_DIR, config), JOURNAL, 300),
            expected,
            "{name}"
        );
    }
    // Drafts continuing from the previous page of a document.
    let pages = [SHORT, JOURNAL];
    let with_drafts = runner(model.clone(), PACKED_DIR, base.clone())
        .recognize_files(&pages, &options(300))
        .unwrap();
    let without = runner(
        model,
        PACKED_DIR,
        RunnerConfig {
            document_drafts: false,
            ..base
        },
    )
    .recognize_files(&pages, &options(300))
    .unwrap();
    for (a, b) in with_drafts.iter().zip(&without) {
        assert_eq!(a.token_ids, b.token_ids, "document drafts");
    }
    assert_eq!(with_drafts[1].token_ids, expected.0);
}

/// The repetition stop ends a looping page early without changing what was
/// generated before the stop.
#[test]
#[ignore = "needs artifacts/packed and a looping corpus page"]
fn repetition_stop_yields_a_prefix_of_the_unstopped_output() {
    let model = packed_model(Mode::NearExact);
    let stopped = tokens(&runner(model.clone(), PACKED_DIR, RunnerConfig::default()), LOOP, 4096);
    assert_eq!(stopped.1, FinishReason::Repetition);
    let free = tokens(
        &runner(
            model,
            PACKED_DIR,
            RunnerConfig {
                repetition_stop: false,
                ..RunnerConfig::default()
            },
        ),
        LOOP,
        4096,
    );
    assert_ne!(free.1, FinishReason::Repetition);
    assert!(free.0.len() > stopped.0.len());
    assert_eq!(&free.0[..stopped.0.len()], &stopped.0[..]);
}

/// The tensor traces the receipts pin, under the reference configuration.
#[test]
#[ignore = "needs artifacts/model, the GPTQ overlay and the smoke fixture"]
fn tensor_traces_are_bitwise_pinned() {
    let fixture = Path::new(SMOKE_DIR).join("trace.safetensors");
    let dir = tempfile::tempdir().unwrap();
    let trace_hash = |model: Arc<Model>, config: RunnerConfig, name: &str| {
        let runner = runner(model, MODEL_DIR, config);
        let mut trace = TensorTrace::default();
        runner.trace_reference(&fixture, 17, &mut trace).unwrap();
        let out = dir.path().join(name);
        trace.save(&out).unwrap();
        file_sha256(&out)
    };
    // `falcon-ocr --threads 4 --backend avx2 trace`.
    let exact = trace_hash(
        Arc::new(Model::load(MODEL_DIR).unwrap()),
        RunnerConfig {
            threads: 4,
            backend: Backend::Avx2,
            ..RunnerConfig::reference()
        },
        "smoke.safetensors",
    );
    assert_eq!(exact, SMOKE_TRACE_SHA256);
    // `falcon-ocr-eval --threads 8 --profile <profile> trace`.
    let eval = RunnerConfig {
        threads: 8,
        ..RunnerConfig::reference()
    };
    let w16 = Arc::new(Model::load_profile(MODEL_DIR, Profile::W16_BODY_KV_Q16, None).unwrap());
    assert_eq!(trace_hash(w16, eval.clone(), "w16.safetensors"), W16_TRACE_SHA256);
    let overlay = Path::new(MODEL_DIR).join("w8-gptq.safetensors");
    let w8 = Arc::new(Model::load_profile(MODEL_DIR, Profile::W8_BODY_KV_Q8, Some(&overlay)).unwrap());
    assert_eq!(trace_hash(w8, eval, "w8.safetensors"), W8_TRACE_SHA256);
}

/// `--max-dimension auto`: a routed page is exactly the fixed run at the
/// size the router chose, and a routed page that stops by length is rerun at
/// the cap (the result is then the fixed 1536 run).
#[test]
#[ignore = "needs artifacts/packed and the corpus"]
fn routed_pages_are_the_fixed_run_at_the_chosen_size() {
    use falcon_ocr::router::CAP;
    let runner = runner(packed_model(Mode::Fast), MODEL_DIR, RunnerConfig::default());
    for (page, max_new_tokens) in [(SHORT, 4096), (JOURNAL, 48)] {
        let auto = GenerationOptions {
            route: true,
            ..options(max_new_tokens)
        };
        let routed = runner.recognize_file(page, &auto).unwrap();
        let route = routed.route.clone().unwrap();
        let size = match &route.safety_net {
            Some(attempt) => {
                assert!(route.max_dimension < CAP, "{page}");
                assert!(matches!(
                    attempt.finish_reason,
                    FinishReason::Repetition | FinishReason::Length
                ));
                CAP
            }
            None => {
                assert!(
                    route.max_dimension == CAP || routed.finish_reason == FinishReason::Eos,
                    "{page}"
                );
                route.max_dimension
            }
        };
        let fixed = runner.recognize_file(page, &auto.at(size)).unwrap();
        assert_eq!(routed.token_ids, fixed.token_ids, "{page} at {size}");
        assert_eq!((routed.width, routed.height), (fixed.width, fixed.height), "{page}");
        assert!(fixed.route.is_none());
    }
}
