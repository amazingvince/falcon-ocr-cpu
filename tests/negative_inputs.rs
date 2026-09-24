//! First-release public input/error contracts. No test requests successful OCR.
//! Asset-dependent tests are opt-in: run with --include-ignored --test-threads=1.
use anyhow::Result;
use falcon_ocr::{
    Backend, GenerationOptions, Model, Runner, RunnerConfig,
    preprocess::{prepare_file, prepare_file_timed},
    trace::Trace,
};
use image::{ImageFormat, Rgb, RgbImage};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Arc,
};

fn error_contains<T>(result: Result<T>, expected: &str) {
    match result {
        Ok(_) => panic!("expected error containing {expected:?}"),
        Err(error) => {
            let message = format!("{error:#}");
            assert!(message.contains(expected), "expected {expected:?}, got {message:?}");
        }
    }
}

fn invalid_options() -> Vec<(&'static str, GenerationOptions, &'static str)> {
    let base = GenerationOptions {
        min_dimension: 64,
        max_dimension: 256,
        max_new_tokens: 1,
    };
    vec![
        (
            "zero minimum",
            GenerationOptions {
                min_dimension: 0,
                ..base.clone()
            },
            "min_dimension",
        ),
        (
            "zero maximum",
            GenerationOptions {
                min_dimension: 1,
                max_dimension: 0,
                ..base.clone()
            },
            "min_dimension",
        ),
        (
            "reversed bounds",
            GenerationOptions {
                min_dimension: 257,
                ..base.clone()
            },
            "min_dimension",
        ),
        (
            "unaligned maximum",
            GenerationOptions {
                max_dimension: 255,
                ..base.clone()
            },
            "multiple of 16",
        ),
        (
            "maximum below one patch",
            GenerationOptions {
                min_dimension: 1,
                max_dimension: 8,
                ..base.clone()
            },
            "multiple of 16",
        ),
        (
            "zero output cap",
            GenerationOptions {
                max_new_tokens: 0,
                ..base
            },
            "max_new_tokens",
        ),
    ]
}

struct FileCases {
    _directory: tempfile::TempDir,
    valid: PathBuf,
    invalid: Vec<(&'static str, PathBuf, &'static str)>,
}

impl FileCases {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path();
        let valid = path.join("valid.png");
        let image = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
        image.save_with_format(&valid, ImageFormat::Png).unwrap();
        let jpeg = path.join("valid.jpg");
        image.save_with_format(&jpeg, ImageFormat::Jpeg).unwrap();
        let missing = path.join("missing.png");
        let empty = path.join("empty.png");
        fs::write(&empty, []).unwrap();
        let unsupported = path.join("unsupported.gif");
        fs::write(&unsupported, b"GIF89a").unwrap();
        assert_eq!(
            image::guess_format(&fs::read(&unsupported).unwrap()).unwrap(),
            ImageFormat::Gif
        );
        let truncated_png = path.join("truncated.png");
        fs::write(&truncated_png, &fs::read(&valid).unwrap()[..24]).unwrap();
        assert_eq!(
            image::guess_format(&fs::read(&truncated_png).unwrap()).unwrap(),
            ImageFormat::Png
        );
        let truncated_jpeg = path.join("truncated.jpg");
        fs::write(&truncated_jpeg, &fs::read(&jpeg).unwrap()[..16]).unwrap();
        assert_eq!(
            image::guess_format(&fs::read(&truncated_jpeg).unwrap()).unwrap(),
            ImageFormat::Jpeg
        );
        Self {
            valid,
            invalid: vec![
                ("missing", missing, "reading image"),
                ("empty", empty, ""),
                ("unsupported", unsupported, "only PNG and JPEG"),
                ("truncated PNG header", truncated_png, ""),
                ("truncated JPEG header", truncated_jpeg, ""),
            ],
            _directory: directory,
        }
    }
}

#[test]
fn generation_and_runner_options_reject_invalid_values() {
    for (label, options, expected) in invalid_options() {
        eprintln!("validating {label}");
        error_contains(options.validate(), expected);
    }
    error_contains(
        RunnerConfig {
            threads: 1,
            batch_size: 0,
            backend: Backend::Scalar,
            ..RunnerConfig::reference()
        }
        .validate(),
        "batch_size must be positive",
    );
}

#[test]
fn file_preparation_returns_errors_for_missing_empty_unsupported_and_truncated_files() {
    let cases = FileCases::new();
    for (label, path, expected) in &cases.invalid {
        eprintln!("preparing {label}");
        // A panic fails this test; an ordinary Result::Err is required.
        error_contains(prepare_file(path, 64, 256), expected);
        error_contains(prepare_file_timed(path, 64, 256), expected);
    }
}

fn cli() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_falcon-ocr"));
    command.env("CUDA_VISIBLE_DEVICES", "-1");
    command
}

fn assert_cli_error(output: Output, expected: &str) {
    assert!(!output.status.success(), "invalid input unexpectedly succeeded");
    assert!(output.status.code().is_some(), "CLI terminated without an exit code");
    assert!(
        output.stdout.is_empty(),
        "error path emitted a result: {:?}",
        String::from_utf8_lossy(&output.stdout)
    );
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(expected), "expected {expected:?}, got {error:?}");
}

#[test]
fn cli_requires_an_image_before_attempting_model_load() {
    let temporary = tempfile::tempdir().unwrap();
    let absent_model = temporary.path().join("no-model");
    let output = cli().arg("--model").arg(absent_model).arg("run").output().unwrap();
    assert_eq!(output.status.code(), Some(2), "Clap argument rejection");
    assert_cli_error(output, "<IMAGES>");
}

fn model_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/model")
}

#[derive(Default)]
struct NoModelWork;
impl Trace for NoModelWork {
    fn tensor(&mut self, name: &str, _: &[usize], _: &[f32]) -> Result<()> {
        panic!("invalid request reached a model tensor: {name}");
    }
    fn decode_start(&mut self) {
        panic!("invalid request reached decode");
    }
}

#[test]
#[ignore = "requires pinned checkpoint/tokenizer assets; validates rejection only, no successful generation"]
fn public_runner_errors_and_empty_collections_do_not_generate() {
    let model_dir = model_directory();
    let model =
        Arc::new(Model::load(&model_dir).expect("install pinned assets before explicitly running this ignored test"));
    error_contains(
        Runner::new(
            model.clone(),
            &model_dir,
            RunnerConfig {
                threads: 1,
                batch_size: 0,
                backend: Backend::Scalar,
                ..RunnerConfig::reference()
            },
        ),
        "batch_size must be positive",
    );
    let runner = Runner::new(
        model,
        &model_dir,
        RunnerConfig {
            threads: 1,
            batch_size: 4,
            backend: Backend::Scalar,
            ..RunnerConfig::reference()
        },
    )
    .unwrap();
    let valid = RgbImage::from_pixel(64, 64, Rgb([255; 3]));
    let cases = FileCases::new();
    let options = GenerationOptions {
        min_dimension: 64,
        max_dimension: 256,
        max_new_tokens: 1,
    };
    let no_images: &[RgbImage] = &[];
    let no_files: &[PathBuf] = &[];
    assert!(runner.recognize_batch(no_images, &options).unwrap().is_empty());
    assert!(
        runner
            .recognize_batch_with_trace(no_images, &options, &mut NoModelWork)
            .unwrap()
            .is_empty()
    );
    assert!(runner.recognize_files(no_files, &options).unwrap().is_empty());
    for (label, invalid, expected) in invalid_options() {
        eprintln!("public entry points: {label}");
        error_contains(runner.recognize(&valid, &invalid), expected);
        error_contains(
            runner.recognize_with_trace(&valid, &invalid, &mut NoModelWork),
            expected,
        );
        error_contains(runner.recognize_file(&cases.valid, &invalid), expected);
        error_contains(runner.recognize_batch(std::slice::from_ref(&valid), &invalid), expected);
        error_contains(
            runner.recognize_files(std::slice::from_ref(&cases.valid), &invalid),
            expected,
        );
        error_contains(runner.recognize_batch(no_images, &invalid), expected);
        error_contains(runner.recognize_files(no_files, &invalid), expected);
    }
    for (width, height) in [(0, 0), (0, 8), (8, 0)] {
        let invalid = RgbImage::new(width, height);
        error_contains(
            runner.recognize_with_trace(&invalid, &options, &mut NoModelWork),
            "dimensions must be nonzero",
        );
        // The valid item precedes the invalid item in the same first chunk.
        error_contains(
            runner.recognize_batch_with_trace(&[valid.clone(), invalid], &options, &mut NoModelWork),
            "dimensions must be nonzero",
        );
    }
    for (label, path, expected) in &cases.invalid {
        eprintln!("public file entry points: {label}");
        error_contains(runner.recognize_file(path, &options), expected);
        error_contains(runner.recognize_files(&[&cases.valid, path], &options), expected);
    }
}

#[test]
#[ignore = "requires pinned checkpoint/tokenizer assets; CLI rejection only, no successful generation"]
fn cli_runtime_errors_exit_nonzero_without_partial_json_results() {
    let cases = FileCases::new();
    let run = |arguments: &[&str], files: &[&Path]| {
        let mut command = cli();
        command
            .arg("--model")
            .arg(model_directory())
            .args(["--threads", "1", "--backend", "scalar"]);
        command.args(arguments).arg("run").args(files);
        command.output().unwrap()
    };
    assert_cli_error(
        run(&["--batch-size", "0"], &[&cases.valid]),
        "batch_size must be positive",
    );
    let missing = &cases.invalid[0].1;
    assert_cli_error(run(&[], &[missing]), "reading image");
    assert_cli_error(run(&["--batch-size", "2"], &[&cases.valid, missing]), "reading image");
    let unsupported = &cases.invalid[2].1;
    assert_cli_error(run(&[], &[unsupported]), "only PNG and JPEG");
    let output = cli()
        .arg("--model")
        .arg(model_directory())
        .args(["--threads", "1", "--backend", "scalar", "run"])
        .arg(&cases.valid)
        .args(["--max-new-tokens", "0"])
        .output()
        .unwrap();
    assert_cli_error(output, "max_new_tokens must be positive");
}
