//! `falcon-ocr run` checks its flags, inputs and weights before the model
//! loads, and touches the output file only once the runner is built, so
//! every case here but the last runs without model assets: a model
//! directory that holds nothing (or a header-only kernel-ready file) makes
//! any later step fail with the model's own error, which the input errors
//! must precede. The last, ignored, resumes a run with the published
//! near-exact file in `artifacts/packed`.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use image::{ImageFormat, Rgb, RgbImage};

struct Case {
    directory: tempfile::TempDir,
    valid: PathBuf,
}

impl Case {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let valid = directory.path().join("valid.png");
        RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]))
            .save_with_format(&valid, ImageFormat::Png)
            .unwrap();
        Self { directory, valid }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, bytes).unwrap();
        path
    }

    /// `falcon-ocr --model <absent dir> [global] run [run]`, from the case's
    /// directory.
    fn run(&self, global: &[&str], run: &[&str]) -> Output {
        self.run_with(&self.path("no-model"), global, run)
    }

    /// [`Case::run`] with the model directory `model`.
    fn run_with(&self, model: &Path, global: &[&str], run: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_falcon-ocr"))
            .current_dir(self.directory.path())
            .arg("--model")
            .arg(model)
            .args(global)
            .arg("run")
            .args(run)
            .output()
            .unwrap()
    }
}

fn arg(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// A failure with an exit code, nothing on stdout, and `expected` on stderr.
fn assert_error(output: &Output, expected: &str) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(!output.status.success(), "unexpectedly succeeded: {stderr}");
    assert!(output.status.code().is_some(), "terminated without an exit code");
    assert!(
        output.stdout.is_empty(),
        "error path emitted output: {:?}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(stderr.contains(expected), "expected {expected:?}, got {stderr:?}");
    stderr
}

/// The error of a run that got as far as loading the (absent) model.
const NO_MODEL: &str = "no near-exact model in";

#[test]
fn bad_inputs_fail_before_the_model_loads() {
    let case = Case::new();
    let valid = arg(&case.valid);
    let missing = case.path("missing.png");
    let empty = case.write("empty.png", b"");
    let gif = case.write("page.gif", b"GIF89a");
    let directory = case.path("folder.png");
    fs::create_dir(&directory).unwrap();
    for (inputs, expected) in [
        (vec![arg(&missing)], "reading image"),
        (vec![valid, arg(&missing)], "reading image"),
        (vec![arg(&empty)], "only PNG and JPEG"),
        (vec![valid, arg(&gif)], "only PNG and JPEG"),
        (vec![arg(&directory)], "not a regular file"),
    ] {
        assert_error(&case.run(&[], &inputs), expected);
        assert_error(&case.run(&["--batch-size", "2"], &inputs), expected);
    }
    // Only the signature is checked up front: a truncated PNG passes, and
    // fails when its page runs.
    let truncated = case.write("cut.png", &fs::read(&case.valid).unwrap()[..24]);
    assert_error(&case.run(&[], &[arg(&truncated)]), NO_MODEL);
    assert_error(
        &case.run(&[], &[valid, "--max-new-tokens", "0"]),
        "max_new_tokens must be positive",
    );
    assert_error(
        &case.run(&["--batch-size", "0"], &[valid]),
        "batch_size must be positive",
    );
}

#[test]
fn list_inputs_follow_the_positional_ones_and_are_checked() {
    let case = Case::new();
    let valid = arg(&case.valid);
    // Relative entries resolve against the current directory.
    case.write("pages.txt", b"# chapter 1\nvalid.png\n\n  valid.png  \n");
    assert_error(&case.run(&[], &["--list", "pages.txt"]), NO_MODEL);
    assert_error(&case.run(&[], &[valid, "--list", "pages.txt"]), NO_MODEL);
    case.write("broken.txt", b"valid.png\nmissing.png\n");
    assert_error(&case.run(&[], &["--list", "broken.txt"]), "reading image");
    case.write("comments.txt", b"# nothing yet\n\n");
    assert_error(&case.run(&[], &["--list", "comments.txt"]), "no input images");
    assert_error(&case.run(&[], &[valid, "--list", "comments.txt"]), NO_MODEL);
    assert_error(&case.run(&[], &["--list", "absent.txt"]), "reading the input list");
    // Without --list an image is still required, before the model loads.
    let output = case.run(&[], &[]);
    assert_eq!(output.status.code(), Some(2), "clap rejects the arguments");
    assert_error(&output, "<IMAGES>");
}

#[test]
fn run_flags_that_do_not_combine_fail_before_the_model_loads() {
    let case = Case::new();
    let valid = arg(&case.valid);
    let needs: [(&[&str], &str); 2] = [
        (&[valid, "--resume"], "--output"),
        (&[valid, "--prefill-threads", "2"], "--pipeline"),
    ];
    for (run, missing) in needs {
        let output = case.run(&[], run);
        assert_eq!(output.status.code(), Some(2), "clap rejects {run:?}");
        assert_error(&output, missing);
    }
    // The pipeline works with batches.
    assert_error(&case.run(&["--batch-size", "2"], &[valid, "--pipeline"]), NO_MODEL);
    // The second pool has at least one thread and at most the runner's.
    for threads in ["0", "5"] {
        assert_error(
            &case.run(
                &["--threads", "4"],
                &[valid, "--pipeline", "--prefill-threads", threads],
            ),
            "--prefill-threads must be 1..=4, the runner's threads",
        );
    }
    assert_error(
        &case.run(&["--threads", "4"], &[valid, "--pipeline", "--prefill-threads", "4"]),
        NO_MODEL,
    );
    assert_error(
        &case.run(&["--stop-repetition=false"], &[valid, "--escalate"]),
        "--escalate rereads the pages that the repetition stop ended",
    );
    // Resumed runs match records by path, so an input may appear once.
    let output = case.path("out.jsonl");
    assert_error(&case.run(&[], &[valid, valid]), NO_MODEL);
    assert_error(
        &case.run(&[], &[valid, valid, "--output", arg(&output), "--resume"]),
        "is an input twice",
    );
    // A record writes its path as text, so two paths that differ only in
    // bytes that are not UTF-8 are the same input to a resume.
    #[cfg(target_os = "linux")]
    {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
        let png = fs::read(&case.valid).unwrap();
        let inputs = [b"x\xff.png", b"x\xfe.png"].map(|name| {
            let path = case.directory.path().join(OsStr::from_bytes(name));
            fs::write(&path, &png).unwrap();
            path
        });
        let resumed = Command::new(env!("CARGO_BIN_EXE_falcon-ocr"))
            .current_dir(case.directory.path())
            .arg("--model")
            .arg(case.path("no-model"))
            .arg("run")
            .args(&inputs)
            .args(["--output", arg(&output), "--resume"])
            .output()
            .unwrap();
        assert_error(&resumed, "is an input twice");
    }
}

/// A header-only kernel-ready file of `profile`: enough for the weights
/// resolution, which reads no tensors.
fn fabricate_packed(path: &Path, profile: &str) {
    let data = [0u8; 4];
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data).unwrap();
    let metadata = [
        ("format".to_owned(), falcon_ocr::model::PACKED_FORMAT.to_owned()),
        ("profile".to_owned(), profile.to_owned()),
    ]
    .into();
    safetensors::serialize_to_file(vec![("t", view)], Some(metadata), path).unwrap();
}

#[test]
fn escalation_needs_fast_mode_and_a_near_exact_model_before_anything_loads() {
    let case = Case::new();
    let valid = arg(&case.valid);
    let models = case.path("models");
    fs::create_dir(&models).unwrap();
    fabricate_packed(&models.join("falcon-ocr-v1.5-fast.safetensors"), "w8-body-kv-q8");
    assert_error(
        &case.run_with(&models, &["--mode", "fast"], &[valid, "--escalate"]),
        "--escalate needs a near-exact model",
    );
    fabricate_packed(
        &models.join("falcon-ocr-v1.5-near-exact.safetensors"),
        "w16-body-kv-q16",
    );
    // The output file is neither created nor repaired when the check fails.
    let output = case.path("out.jsonl");
    assert_error(
        &case.run_with(&models, &[], &[valid, "--escalate", "--output", arg(&output)]),
        "--escalate rereads fast-mode pages in near-exact mode; this run is near-exact already",
    );
    assert!(!output.exists());
    // A research KV cache makes fast mode a research profile, which is not
    // escalated; the fast mode's own cache is still fast mode (the check
    // then passes and the header-only file fails to load).
    assert_error(
        &case.run_with(
            &models,
            &["--mode", "fast", "--kv-cache", "q16"],
            &[valid, "--escalate"],
        ),
        "this run is the research profile w8-body-kv-q16",
    );
    assert_error(
        &case.run_with(
            &models,
            &["--mode", "fast", "--kv-cache", "q8r"],
            &[valid, "--escalate"],
        ),
        "this run is the research profile w8-body-kv-q8r",
    );
    assert_error(
        &case.run_with(&models, &["--mode", "fast", "--kv-cache", "q8"], &[valid, "--escalate"]),
        "packed metadata source_sha256 missing",
    );
}

#[test]
fn the_output_file_is_left_alone_until_the_model_loads() {
    let case = Case::new();
    let valid = arg(&case.valid);
    let record = r#"{"path":"valid.png","page":0,"text":"t","mode":"near-exact","precision":"w16-body-kv-q16"}"#;
    let cut = format!("{record}\n{{\"path\":\"other.png\",\"te");
    let output = case.write("out.jsonl", cut.as_bytes());
    let run = [valid, "--output", arg(&output), "--resume"];
    // No weights: the file is left as it is.
    assert_error(&case.run(&[], &run), NO_MODEL);
    assert_eq!(fs::read_to_string(&output).unwrap(), cut);
    // Weights that resolve but do not load: still left as it is.
    let models = case.path("models");
    fs::create_dir(&models).unwrap();
    fabricate_packed(
        &models.join("falcon-ocr-v1.5-near-exact.safetensors"),
        "w16-body-kv-q16",
    );
    let stderr = assert_error(
        &case.run_with(&models, &[], &run),
        "packed metadata source_sha256 missing",
    );
    assert!(!stderr.contains("removed a last record cut short"), "{stderr}");
    assert_eq!(fs::read_to_string(&output).unwrap(), cut);
    // A directory is never an output; `--resume` reads the output, so it
    // must be a regular file. Both are refused before the weights resolve.
    let directory = arg(case.directory.path());
    assert_error(&case.run(&[], &[valid, "--output", directory]), "is a directory");
    // An output whose folder is missing is refused before the model loads.
    assert_error(
        &case.run(&[], &[valid, "--output", "absent/out.jsonl"]),
        "--output absent/out.jsonl: the folder absent does not exist",
    );
    assert_error(
        &case.run(&[], &[valid, "--output", directory, "--resume"]),
        "is a directory",
    );
    #[cfg(unix)]
    assert_error(
        &case.run(&[], &[valid, "--output", "/dev/null", "--resume"]),
        "is not a regular file",
    );
}

#[test]
#[ignore = "requires the published near-exact file in artifacts/packed"]
fn a_resumed_run_repairs_the_output_and_runs_only_the_missing_pages() {
    let case = Case::new();
    let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/packed");
    let second = case.write("second.png", &fs::read(&case.valid).unwrap());
    let output = case.path("book.jsonl");
    let (first, second) = (arg(&case.valid), arg(&second));
    let records = |text: &str| -> Vec<serde_json::Value> {
        text.lines().map(|line| serde_json::from_str(line).unwrap()).collect()
    };
    let run = |inputs: &[&str]| {
        let mut run = inputs.to_vec();
        run.extend(["--max-new-tokens", "8", "--output", arg(&output), "--resume"]);
        let result = case.run_with(&packed, &[], &run);
        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        assert!(result.status.success(), "{stderr}");
        stderr
    };
    run(&[first]);
    let written = fs::read_to_string(&output).unwrap();
    assert_eq!(records(&written).len(), 1);
    // A crash cut the next record short; the resumed run removes it, skips
    // the first page and runs the second, as the input's second page.
    fs::write(&output, format!("{written}{{\"path\":\"{second}\",\"te")).unwrap();
    let stderr = run(&[first, second]);
    assert!(stderr.contains("removed a last record cut short"), "{stderr}");
    assert!(stderr.contains("pages: 1 done, 0 failed, 1 skipped"), "{stderr}");
    let resumed = records(&fs::read_to_string(&output).unwrap());
    assert_eq!(resumed.len(), 2);
    assert_eq!(
        (resumed[1]["path"].as_str(), resumed[1]["page"].as_u64()),
        (Some(second), Some(1))
    );
    assert_eq!(resumed[1]["token_ids"], resumed[0]["token_ids"], "the same image");
}
