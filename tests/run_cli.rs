//! `falcon-ocr run` checks its flags, inputs and weights before the model
//! loads, and touches the output file only once the runner is built, so
//! every case here but the last runs without model assets: a model
//! directory that holds nothing (or a header-only kernel-ready file) makes
//! any later step fail with the model's own error, which the input errors
//! must precede. The last three, ignored, resume a run, keep going past a
//! failing page and run with a closed stderr, with the published near-exact
//! file in `artifacts/packed`.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
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
    // A record writes its path as text, so a path that is not UTF-8 is
    // refused with an output: written lossily, `x\xff.png` would share the
    // record of `x\xfe.png`, which a resumed run would then skip.
    #[cfg(target_os = "linux")]
    {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
        let png = fs::read(&case.valid).unwrap();
        let input = case.directory.path().join(OsStr::from_bytes(b"x\xff.png"));
        fs::write(&input, &png).unwrap();
        let record = format!(
            "{}\n",
            serde_json::json!({"path": case.directory.path().join("x\u{fffd}.png"), "page": 0, "text": "t"})
        );
        fs::write(&output, &record).unwrap();
        let run = |extra: &[&str]| {
            Command::new(env!("CARGO_BIN_EXE_falcon-ocr"))
                .current_dir(case.directory.path())
                .arg("--model")
                .arg(case.path("no-model"))
                .arg("run")
                .arg(&input)
                .args(extra)
                .output()
                .unwrap()
        };
        for extra in [&["--output", arg(&output), "--resume"][..], &["--output", arg(&output)]] {
            assert_error(&run(extra), "is not UTF-8, so a record in --output cannot name it");
        }
        assert_eq!(fs::read_to_string(&output).unwrap(), record);
        // Without an output nothing is recorded, and the page runs.
        assert_error(&run(&[]), NO_MODEL);
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
    // A resumed run reads only the inputs without a successful record, so
    // one that has a record need not exist any more; any other must.
    let done = case.write("done.jsonl", b"{\"path\":\"gone.png\",\"page\":0,\"text\":\"t\"}\n");
    assert_error(
        &case.run(&[], &["gone.png", valid, "--output", arg(&done), "--resume"]),
        NO_MODEL,
    );
    assert_error(
        &case.run(&[], &["gone.png", valid, "--output", arg(&done)]),
        "reading image",
    );
    // An editor's byte order mark does not hide the first record.
    let marked = case.write(
        "marked.jsonl",
        b"\xef\xbb\xbf{\"path\":\"gone.png\",\"page\":0,\"text\":\"t\"}\r\n",
    );
    assert_error(
        &case.run(&[], &["gone.png", valid, "--output", arg(&marked), "--resume"]),
        NO_MODEL,
    );
    // Without --resume, the pages that have a record run again: a warning.
    let stderr = assert_error(&case.run(&[], &["valid.png", "--output", arg(&done)]), NO_MODEL);
    assert!(!stderr.contains("run again"), "{stderr}");
    let again = case.write("again.jsonl", b"{\"path\":\"valid.png\",\"page\":0,\"text\":\"t\"}\n");
    // An input given twice is one page with a record.
    let stderr = assert_error(
        &case.run(&[], &["valid.png", "valid.png", "--output", arg(&again)]),
        NO_MODEL,
    );
    assert!(
        stderr.contains("already holds records of 1 of these pages; without --resume they run again"),
        "{stderr}"
    );
    // A file that is not an output is refused before the model loads, and
    // left as it is: an input image (not UTF-8), or text that is not
    // records, such as the input list itself.
    let image = fs::read(&case.valid).unwrap();
    assert_error(
        &case.run(&[], &[valid, "--output", valid]),
        "is not a UTF-8 JSON-lines file",
    );
    assert_eq!(fs::read(&case.valid).unwrap(), image);
    let list = case.write("pages.txt", b"valid.png\n");
    for run in [
        &["--list", "pages.txt", "--output", "pages.txt"][..],
        &[valid, "--output", "pages.txt", "--resume"],
    ] {
        assert_error(&case.run(&[], run), "pages.txt: its line 1 is not a record");
    }
    let json = case.write("config.json", b"{\"model_type\":\"falcon\"}\n");
    assert_error(
        &case.run(&[], &[valid, "--output", arg(&json)]),
        "config.json: its line 1 is not a record (a JSON object with a \"path\")",
    );
    assert_eq!(fs::read(&list).unwrap(), b"valid.png\n");
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

#[test]
#[ignore = "requires the published near-exact file in artifacts/packed"]
fn keep_going_records_a_failing_page_and_runs_the_rest() {
    let case = Case::new();
    let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/packed");
    // A truncated PNG passes the check before the model loads and fails
    // when its page runs.
    let cut = case.write("cut.png", &fs::read(&case.valid).unwrap()[..24]);
    let (valid, cut) = (arg(&case.valid), arg(&cut));
    let run = |name: &str, keep_going: bool| {
        let output = case.path(name);
        let mut run = vec![valid, cut, valid, "--max-new-tokens", "8", "--output", arg(&output)];
        if keep_going {
            run.push("--keep-going");
        }
        let result = case.run_with(&packed, &[], &run);
        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        let records: Vec<serde_json::Value> = fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        (result.status.code(), stderr, records)
    };
    // With --keep-going: the failing page's error record, the others' results,
    // and a failed exit at the end.
    let (code, stderr, records) = run("kept.jsonl", true);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("pages: 2 done, 1 failed, 0 skipped"), "{stderr}");
    assert!(stderr.contains("1 of 3 pages failed"), "{stderr}");
    let pages: Vec<_> = records.iter().map(|record| record["page"].as_u64()).collect();
    assert_eq!(pages, [Some(0), Some(1), Some(2)]);
    assert!(records[1]["error"].is_string() && records[1].get("token_ids").is_none());
    assert_eq!(records[2]["token_ids"], records[0]["token_ids"], "the same image");
    // Without it the run stops at the failing page, after the page before it.
    let (code, stderr, records) = run("stopped.jsonl", false);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(
        stderr.contains("pages: 1 done, 1 failed, 0 skipped, 1 not run"),
        "{stderr}"
    );
    assert_eq!(records.len(), 1);
}

#[test]
#[ignore = "requires the published near-exact file in artifacts/packed"]
fn a_closed_stderr_does_not_stop_the_run() {
    let case = Case::new();
    let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/packed");
    let output = case.path("book.jsonl");
    let valid = arg(&case.valid);
    // `run … 2>&1 | less`, and the pager quit: every diagnostic (the plan,
    // the summary) fails to write.
    let mut child = Command::new(env!("CARGO_BIN_EXE_falcon-ocr"))
        .current_dir(case.directory.path())
        .arg("--model")
        .arg(&packed)
        .args(["run", valid, valid, "--max-new-tokens", "8", "--output", arg(&output)])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stderr.take());
    let status = child.wait().unwrap();
    assert!(status.success(), "{status}");
    assert_eq!(fs::read_to_string(&output).unwrap().lines().count(), 2);
}
