//! Long runs of `falcon-ocr run`, such as the few hundred pages of a book:
//! the input list, the input check before the model loads, the per-page
//! JSON record, the output file that `--resume` continues, and the summary
//! printed at the end.

mod resume;

use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::{GenerationOptions, OcrResult};
pub use resume::{
    Existing, Kernels, Run, Settings, escalation_warning, is_file_or_missing, open_output, resume_warning, scan_output,
    this_run,
};

/// The input paths of `--list FILE`, one per line, in order. Blank lines
/// and lines starting with `#` are skipped, surrounding whitespace is
/// trimmed, and so is a UTF-8 byte order mark (Windows editors write one);
/// a relative path stays relative, so it resolves against the current
/// directory, not the list's folder.
pub fn parse_list(text: &str) -> Vec<PathBuf> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect()
}

/// [`parse_list`] of the file at `path`.
pub fn read_list(path: &Path) -> Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading the input list {}", path.display()))?;
    Ok(parse_list(&text))
}

/// Check an input before the model loads: a readable regular file that
/// starts with a PNG or JPEG signature. A file that passes can still fail to
/// decode (a truncated or corrupt image); that page then fails when it runs.
/// The file type is checked before the file is opened: Windows cannot open a
/// directory as a file, and opening a named pipe would wait for a writer.
pub fn check_input(path: &Path) -> Result<()> {
    let context = || format!("reading image {}", path.display());
    ensure!(
        std::fs::metadata(path).with_context(context)?.is_file(),
        "reading image {}: not a regular file",
        path.display()
    );
    let file = File::open(path).with_context(context)?;
    let mut head = Vec::with_capacity(16);
    file.take(16).read_to_end(&mut head).with_context(context)?;
    match image::guess_format(&head) {
        Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg) => Ok(()),
        _ => bail!("{}: only PNG and JPEG files are supported", path.display()),
    }
}

/// One page's line of output: the input as given (`path`), its 0-based
/// position among the inputs (`page`), then every field of its result and
/// the generation `options` the run asked for (which `--resume` compares),
/// or `error` for a page that failed.
pub fn record_line(
    path: &Path,
    page: usize,
    outcome: Result<&OcrResult, &anyhow::Error>,
    options: Option<&GenerationOptions>,
) -> Result<String> {
    #[derive(Serialize)]
    struct Done<'a> {
        path: &'a str,
        page: usize,
        #[serde(flatten)]
        result: &'a OcrResult,
        #[serde(skip_serializing_if = "Option::is_none")]
        options: Option<&'a GenerationOptions>,
    }
    #[derive(Serialize)]
    struct Failed<'a> {
        path: &'a str,
        page: usize,
        error: String,
    }
    let path = path.to_string_lossy();
    Ok(match outcome {
        Ok(result) => serde_json::to_string(&Done {
            path: &path,
            page,
            result,
            options,
        })?,
        Err(error) => serde_json::to_string(&Failed {
            path: &path,
            page,
            error: format!("{error:#}"),
        })?,
    })
}

/// Where `falcon-ocr run` puts each page: the JSON record goes to the
/// `--output` file (appended and flushed per page) or else to `stdout`; with
/// `text`, `stdout` gets the page's text instead of its record.
pub struct Sink<W: Write> {
    stdout: W,
    file: Option<File>,
    text: bool,
    /// The generation options every record carries.
    options: Option<GenerationOptions>,
}

impl<W: Write> Sink<W> {
    /// A sink writing to `file` when there is one, else to `stdout`;
    /// `options` are added to every successful record.
    pub fn new(stdout: W, file: Option<File>, text: bool, options: Option<GenerationOptions>) -> Self {
        Self {
            stdout,
            file,
            text,
            options,
        }
    }

    /// A finished page.
    pub fn page(&mut self, path: &Path, page: usize, result: &OcrResult) -> Result<()> {
        let line = record_line(path, page, Ok(result), self.options.as_ref())?;
        if let Some(file) = &mut self.file {
            append(file, &line)?;
        }
        match (self.text, self.file.is_some()) {
            (true, _) => writeln!(self.stdout, "{}", result.text)?,
            (false, false) => writeln!(self.stdout, "{line}")?,
            (false, true) => return Ok(()),
        }
        self.stdout.flush()?;
        Ok(())
    }

    /// A page that failed (`--keep-going`): its error record, where records
    /// go (nowhere with `text` and no file; the error is on stderr).
    pub fn error(&mut self, path: &Path, page: usize, error: &anyhow::Error) -> Result<()> {
        let line = record_line(path, page, Err(error), None)?;
        match (&mut self.file, self.text) {
            (Some(file), _) => append(file, &line),
            (None, false) => {
                writeln!(self.stdout, "{line}")?;
                self.stdout.flush()?;
                Ok(())
            }
            (None, true) => Ok(()),
        }
    }
}

/// Append one line in a single write and flush it.
fn append(file: &mut File, line: &str) -> Result<()> {
    let mut bytes = Vec::with_capacity(line.len() + 1);
    bytes.extend_from_slice(line.as_bytes());
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(())
}

/// The line `falcon-ocr run` prints on stderr at the end.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    /// Pages whose record was written.
    pub done: usize,
    /// Pages that failed, or ran but whose record could not be written.
    pub failed: usize,
    /// Pages the output already held (`--resume`).
    pub skipped: usize,
    /// Pages never run because the run stopped at an error.
    pub not_run: usize,
    /// Wall time of the recognition, model loading excluded.
    pub seconds: f64,
}

impl std::fmt::Display for Summary {
    /// `pages: 297 done, 1 failed, 2 skipped in 3512.4 s (304.4 pages/hour)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pages: {} done, {} failed, {} skipped",
            self.done, self.failed, self.skipped
        )?;
        if self.not_run > 0 {
            write!(f, ", {} not run", self.not_run)?;
        }
        let per_hour = if self.seconds > 0.0 {
            self.done as f64 * 3600.0 / self.seconds
        } else {
            0.0
        };
        write!(f, " in {:.1} s ({per_hour:.1} pages/hour)", self.seconds)
    }
}

/// A near-exact result with `text`, for the tests.
#[cfg(test)]
fn test_result(text: &str) -> OcrResult {
    serde_json::from_value(serde_json::json!({
        "text": text, "token_ids": [5, 11], "finish_reason": "eos", "width": 64, "height": 32,
        "input_tokens": 20, "output_tokens": 2, "precision": "w16-body-kv-q16", "backend": "avx2",
        "mode": "near-exact", "teacher_forced": false,
        "timings": {"image_decode_ms": 1.0, "preprocessing_ms": 2.0, "prefill_ms": 3.0, "decode_ms": 4.0,
                    "total_ms": 10.0, "time_to_first_token_ms": 6.0}
    }))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FinishReason;
    use serde::Deserialize;
    use serde_json::Value;
    use std::collections::HashSet;
    use test_result as result;

    #[test]
    fn the_list_skips_blank_and_comment_lines_and_keeps_paths_as_written() {
        let text = "# chapter 1\npages/p001.png\r\n\n  pages/p002.jpg  \n#pages/skipped.png\n/abs/p003.png\n\t\n";
        assert_eq!(
            parse_list(text),
            [
                PathBuf::from("pages/p001.png"),
                PathBuf::from("pages/p002.jpg"),
                PathBuf::from("/abs/p003.png")
            ]
        );
        assert!(parse_list("").is_empty() && parse_list("# nothing\n\n").is_empty());
        // A byte order mark is not part of the first path.
        assert_eq!(parse_list("\u{feff}a.png\r\n"), [PathBuf::from("a.png")]);
        let dir = tempfile::tempdir().unwrap();
        let list = dir.path().join("pages.txt");
        std::fs::write(&list, "\u{feff}b.png\na.png\n").unwrap();
        assert_eq!(
            read_list(&list).unwrap(),
            [PathBuf::from("b.png"), PathBuf::from("a.png")]
        );
        let missing = read_list(&dir.path().join("missing.txt")).unwrap_err();
        assert!(format!("{missing:#}").contains("reading the input list"));
    }

    #[test]
    fn inputs_must_be_readable_png_or_jpeg_files() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        let png = write("page.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR");
        let jpeg = write("page.jpg", b"\xff\xd8\xff\xe0\0\x10JFIF\0");
        // Only the signature is checked: a truncated PNG fails when it runs.
        let truncated = write("cut.png", b"\x89PNG\r\n\x1a\n");
        for path in [&png, &jpeg, &truncated] {
            check_input(path).unwrap();
        }
        let error = |path: &Path| format!("{:#}", check_input(path).unwrap_err());
        assert!(error(&dir.path().join("missing.png")).contains("reading image"));
        assert!(error(dir.path()).contains("not a regular file"));
        assert!(error(&write("page.gif", b"GIF89a")).contains("only PNG and JPEG"));
        assert!(error(&write("empty.png", b"")).contains("only PNG and JPEG"));
    }

    /// A named pipe is refused by its type: opening it would wait for a
    /// writer, before the model even loads.
    #[cfg(unix)]
    #[test]
    fn a_named_pipe_is_not_an_input_or_a_resumable_output() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("page.png");
        let made = std::process::Command::new("mkfifo").arg(&fifo).status();
        if !made.is_ok_and(|status| status.success()) {
            eprintln!("mkfifo is unavailable; skipped");
            return;
        }
        let error = format!("{:#}", check_input(&fifo).unwrap_err());
        assert!(error.contains("not a regular file"), "{error}");
        assert!(!is_file_or_missing(&fifo));
    }

    #[test]
    fn records_carry_the_path_the_page_and_every_result_field() {
        let page = result("Hello");
        let options = GenerationOptions {
            crop_margins: Some(24),
            ..GenerationOptions::default()
        };
        let line = record_line(Path::new("book/p7.png"), 7, Ok(&page), Some(&options)).unwrap();
        assert!(
            line.starts_with(r#"{"path":"book/p7.png","page":7,"text":"Hello""#),
            "{line}"
        );
        let record: Value = serde_json::from_str(&line).unwrap();
        let mut fields = serde_json::to_value(&page).unwrap();
        let fields = fields.as_object_mut().unwrap();
        assert_eq!(record.as_object().unwrap().len(), fields.len() + 3);
        for (key, value) in fields.iter() {
            assert_eq!(&record[key], value, "{key}");
        }
        // The options come last and read back.
        assert!(line.ends_with(r#""crop_margins":24}}"#), "{line}");
        let back = GenerationOptions::deserialize(&record["options"]).unwrap();
        assert_eq!(Settings::from(&back), Settings::from(&options));
        let bare = record_line(Path::new("book/p7.png"), 7, Ok(&page), None).unwrap();
        assert_eq!(
            bare.len() + r#","options":"#.len() + serde_json::to_string(&options).unwrap().len(),
            line.len()
        );
        // The result reads back from its record.
        let back: OcrResult = serde_json::from_value(record).unwrap();
        assert_eq!(
            (back.token_ids, back.finish_reason),
            (page.token_ids, FinishReason::Eos)
        );
        let failed = anyhow::anyhow!("decode failed").context("preparing book/p8.png");
        let line = record_line(Path::new("book/p8.png"), 8, Err(&failed), Some(&options)).unwrap();
        assert_eq!(
            line,
            r#"{"path":"book/p8.png","page":8,"error":"preparing book/p8.png: decode failed"}"#
        );
    }

    #[test]
    fn the_sink_puts_records_or_text_where_the_flags_say() {
        let page = result("Page text");
        let failed = anyhow::anyhow!("corrupt");
        let lines = |sink: Sink<Vec<u8>>| String::from_utf8(sink.stdout).unwrap();
        // No file: records on stdout, errors too.
        let mut sink = Sink::new(Vec::new(), None, false, None);
        sink.page(Path::new("a.png"), 0, &page).unwrap();
        sink.error(Path::new("b.png"), 1, &failed).unwrap();
        let out = lines(sink);
        assert!(out.starts_with(r#"{"path":"a.png","page":0,"#) && out.ends_with("\"error\":\"corrupt\"}\n"));
        // Text only: stdout gets the text, errors stay on stderr.
        let mut sink = Sink::new(Vec::new(), None, true, None);
        sink.page(Path::new("a.png"), 0, &page).unwrap();
        sink.error(Path::new("b.png"), 1, &failed).unwrap();
        assert_eq!(lines(sink), "Page text\n");
        // A file: records there; with text, stdout still gets the text.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.jsonl");
        let mut sink = Sink::new(Vec::new(), Some(open_output(&path).unwrap().0), true, None);
        sink.page(Path::new("a.png"), 0, &page).unwrap();
        sink.error(Path::new("b.png"), 1, &failed).unwrap();
        assert_eq!(lines(sink), "Page text\n");
        let existing = scan_output(&std::fs::read_to_string(&path).unwrap());
        assert_eq!(existing.done, HashSet::from(["a.png".to_owned()]));
        assert_eq!(existing.keep, std::fs::metadata(&path).unwrap().len() as usize);
    }

    #[test]
    fn the_summary_reports_pages_and_throughput() {
        let summary = Summary {
            done: 297,
            failed: 1,
            skipped: 2,
            not_run: 0,
            seconds: 3600.0,
        };
        assert_eq!(
            summary.to_string(),
            "pages: 297 done, 1 failed, 2 skipped in 3600.0 s (297.0 pages/hour)"
        );
        let stopped = Summary {
            done: 3,
            failed: 1,
            not_run: 296,
            seconds: 36.0,
            ..Summary::default()
        };
        assert_eq!(
            stopped.to_string(),
            "pages: 3 done, 1 failed, 0 skipped, 296 not run in 36.0 s (300.0 pages/hour)"
        );
        assert!(Summary::default().to_string().ends_with("(0.0 pages/hour)"));
    }
}
