//! The output file that `--resume` continues: what its records hold, how
//! it is repaired after a crash, and how its records ran compared with this
//! run.
use std::{
    collections::{BTreeMap, HashSet},
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{GenerationOptions, Mode};

/// The generation options a record's page was asked to run with (its
/// `options`) that change the output: the output cap and whether it is
/// fitted to the context, the image size bounds, routing and the margin
/// crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Settings {
    pub max_new_tokens: usize,
    pub fit_budget: bool,
    pub min_dimension: u32,
    pub max_dimension: u32,
    pub route: bool,
    pub crop_margins: Option<u32>,
}

impl From<&GenerationOptions> for Settings {
    fn from(options: &GenerationOptions) -> Self {
        Self {
            max_new_tokens: options.max_new_tokens,
            fit_budget: options.fit_budget,
            min_dimension: options.min_dimension,
            max_dimension: options.max_dimension,
            route: options.route,
            crop_margins: options.crop_margins,
        }
    }
}

impl Settings {
    /// The `run` flags that ask for these settings, where they differ from
    /// the defaults.
    fn flags(&self) -> Vec<String> {
        let defaults = GenerationOptions::default();
        let mut flags = Vec::new();
        if !self.fit_budget {
            flags.push(format!("--max-new-tokens {}", self.max_new_tokens));
        }
        if self.min_dimension != defaults.min_dimension {
            flags.push(format!("--min-dimension {}", self.min_dimension));
        }
        if self.route {
            flags.push("--max-dimension auto".to_owned());
        } else if self.max_dimension != defaults.max_dimension {
            flags.push(format!("--max-dimension {}", self.max_dimension));
        }
        if let Some(pad) = self.crop_margins {
            flags.push(format!("--crop-margins={pad}"));
        }
        flags
    }
}

/// The runner's choices in a record's `plan` that change tokens: the
/// repetition stop, the prefill exp, the prefill attention (BF16, NEON's
/// polynomial exp, scalar, or the FP32 tiles, whose AVX2 and AVX-512 forms
/// agree bitwise), BF16 projections and a pinned decode exp.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Kernels {
    pub repetition_stop: bool,
    pub exp: String,
    pub attention: String,
    pub bf16_projections: bool,
    pub decode_exp: Option<bool>,
}

impl Kernels {
    /// From a record's `plan` (a serialized `Resolved`); `None` when it does
    /// not hold them.
    pub fn from_plan(plan: &Value) -> Option<Self> {
        let prefill = plan.get("prefill")?;
        let attention = match prefill.get("attention")?.as_str()? {
            "avx2" | "avx512-wide" => "fp32",
            other => other,
        };
        Some(Self {
            repetition_stop: plan.get("repetition_stop")?.as_bool()?,
            exp: plan.get("exp")?.as_str()?.to_owned(),
            attention: attention.to_owned(),
            bf16_projections: prefill.get("projection")?.as_str()? == "panel-bf16",
            decode_exp: plan
                .get("tuning")
                .and_then(|tuning| tuning.get("decode_fast_exp"))
                .and_then(Value::as_bool),
        })
    }

    /// This runner's (`Runner::resolved`).
    pub fn of(resolved: &crate::Resolved) -> Option<Self> {
        Self::from_plan(&serde_json::to_value(resolved).ok()?)
    }

    /// What sets them apart from the defaults.
    fn flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        if !self.repetition_stop {
            flags.push("--stop-repetition=false".to_owned());
        }
        if self.exp != "fast" {
            flags.push(format!("--exp {}", self.exp));
        }
        match self.attention.as_str() {
            "fp32" => {}
            "bf16" => flags.push("BF16 attention".to_owned()),
            other => flags.push(format!("{other} attention")),
        }
        if self.bf16_projections {
            flags.push("BF16 projections".to_owned());
        }
        if let Some(fast) = self.decode_exp {
            flags.push(format!("--tune decode-exp={}", if fast { "fast" } else { "exact" }));
        }
        flags
    }
}

/// How a page ran, as far as `--resume` compares records with a run.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Run {
    pub mode: Option<String>,
    /// `None` for an escalated record: its rerun ran in near-exact mode on
    /// purpose, from a fast-mode run whose weights it does not record.
    pub precision: Option<String>,
    /// The digest of the GPTQ overlay the weights came from
    /// (`plan.overlay_sha256`): `None` for round-to-nearest 8-bit weights
    /// and for the 16-bit and FP32 profiles.
    pub overlay: Option<String>,
    /// `--max-dimension auto` (the record has a `route`).
    pub routed: bool,
    /// The options the run asked for; `None` for a record without readable
    /// `options`, which matches no run.
    pub settings: Option<Settings>,
    /// The runner's token-changing choices; `None` for an escalated record
    /// (its plan is the near-exact rerun's) or a plan without them.
    pub kernels: Option<Kernels>,
}

impl Run {
    /// Whether a record that ran as `self` could have come from `run`.
    fn matches(&self, run: &Run) -> bool {
        self.mode == run.mode
            && self.routed == run.routed
            && self
                .precision
                .as_ref()
                .is_none_or(|precision| Some(precision) == run.precision.as_ref() && self.overlay == run.overlay)
            && self.settings.is_some()
            && self.settings == run.settings
            && self
                .kernels
                .as_ref()
                .is_none_or(|kernels| Some(kernels) == run.kernels.as_ref())
    }
}

impl std::fmt::Display for Run {
    /// `fast (w8-body-kv-q8, overlay 0123456789ab, --max-dimension auto)`,
    /// or `escalated from fast`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut details = Vec::new();
        details.extend(self.precision.clone());
        if let Some(overlay) = &self.overlay {
            // The first 12 characters: a digest read back from a record is
            // hex, unless the file was edited.
            let short: String = overlay.chars().take(12).collect();
            details.push(format!("overlay {short}"));
        } else if self
            .precision
            .as_deref()
            .is_some_and(|precision| precision.starts_with("w8"))
        {
            details.push("round-to-nearest".to_owned());
        }
        match &self.settings {
            Some(settings) => details.extend(settings.flags()),
            None => details.push("no options".to_owned()),
        }
        details.extend(self.kernels.iter().flat_map(Kernels::flags));
        if self.precision.is_none() {
            write!(f, "escalated from ")?;
        }
        write!(f, "{}", self.mode.as_deref().unwrap_or("no mode"))?;
        if !details.is_empty() {
            write!(f, " ({})", details.join(", "))?;
        }
        Ok(())
    }
}

/// What an existing output file holds ([`scan_output`]).
#[derive(Debug, Default, PartialEq)]
pub struct Existing {
    /// Inputs, as given, that have a successful record.
    pub done: HashSet<String>,
    /// Successful records by how they ran; an escalated record counts
    /// under the mode it started in.
    pub runs: BTreeMap<Run, usize>,
    /// Bytes to keep: all but a last record that a crash cut short.
    pub keep: usize,
    /// Bytes of that cut record.
    pub cut: usize,
    /// The kept text ends without a newline.
    pub unterminated: bool,
    /// The number (from 1) of the first line that is not a record (a JSON
    /// object with a `path`): a file that holds one is not an output, and
    /// is not appended to.
    pub foreign: Option<usize>,
    /// Successful fast-mode records that the repetition stop ended and that
    /// were not reread in near-exact mode (a run without `--escalate`, or an
    /// escalation that failed).
    pub unescalated: usize,
}

impl Existing {
    /// Count `record` when it is a successful record; false when it is not a
    /// record at all.
    fn add(&mut self, record: &Map<String, Value>) -> bool {
        let Some(path) = record.get("path").and_then(Value::as_str) else {
            return false;
        };
        if !record.contains_key("text") || record.contains_key("error") {
            return true;
        }
        self.done.insert(path.to_owned());
        let text = |value: Option<&Value>| value.and_then(Value::as_str).map(str::to_owned);
        if record.get("escalated_from").is_none()
            && text(record.get("mode")).as_deref() == Some("fast")
            && text(record.get("finish_reason")).as_deref() == Some("repetition")
        {
            self.unescalated += 1;
        }
        let routed = record.get("route").is_some_and(|route| !route.is_null());
        let settings = record
            .get("options")
            .and_then(|options| GenerationOptions::deserialize(options).ok())
            .map(|options| Settings::from(&options));
        let run = match record.get("escalated_from") {
            Some(attempt) => Run {
                mode: text(attempt.get("mode")),
                routed,
                settings,
                ..Run::default()
            },
            None => Run {
                mode: text(record.get("mode")),
                precision: text(record.get("precision")),
                overlay: text(record.get("plan").and_then(|plan| plan.get("overlay_sha256"))),
                routed,
                settings,
                kernels: record.get("plan").and_then(Kernels::from_plan),
            },
        };
        *self.runs.entry(run).or_default() += 1;
        true
    }
}

/// Read an output file's text: every successful record (a line with `path`
/// and `text` and no `error`), and how to append after it. NUL bytes at the
/// start of a line, what a power loss leaves in place of data that was never
/// written, are skipped: the record they replaced is missing, so its page
/// runs again. A last line without its newline that starts like a record but
/// is not valid JSON, or holds only NUL bytes, was cut short by a crash and
/// is dropped; any other last line without its newline is kept and gets one.
/// A byte order mark (an editor's) is kept and skipped.
pub fn scan_output(text: &str) -> Existing {
    let mut existing = Existing::default();
    let bom = if text.starts_with('\u{feff}') {
        '\u{feff}'.len_utf8()
    } else {
        0
    };
    let mut start = bom;
    for (number, piece) in (1..).zip(text[bom..].split_inclusive('\n')) {
        let line = piece.trim_end_matches(['\n', '\r']).trim_start_matches('\0');
        let record = match serde_json::from_str::<Value>(line) {
            Ok(Value::Object(record)) => existing.add(&record),
            Err(_) if !piece.ends_with('\n') && (line.is_empty() || line.trim_start().starts_with('{')) => {
                existing.cut = piece.len();
                break;
            }
            _ => line.trim().is_empty(),
        };
        if !record {
            existing.foreign = existing.foreign.or(Some(number));
        }
        start += piece.len();
    }
    existing.keep = start;
    existing.unterminated = start > bom && !text[..start].ends_with('\n');
    existing
}

/// Open `path` for appending records, creating it, after reading what it
/// holds ([`scan_output`]): a record cut short by a crash is removed and a
/// last line without its newline gets one, so new records start on a line
/// of their own. A crash can cut a record inside a multi-byte character; any
/// other invalid UTF-8 is an error. Only a regular file is read: anything
/// else (`/dev/stdout`, a pipe) is written to as it is, holding nothing, and
/// a directory is refused by its type, not by a failed open (which reads
/// differently on Windows).
pub fn open_output(path: &Path) -> Result<(File, Existing)> {
    ensure!(!path.is_dir(), "the output {} is a directory", path.display());
    if !is_file_or_missing(path) {
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("opening the output {}", path.display()))?;
        return Ok((file, Existing::default()));
    }
    let (bytes, existing) = read_existing(path)?;
    if existing.keep < bytes.len() {
        OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|file| file.set_len(existing.keep as u64))
            .with_context(|| format!("removing the cut last record of {}", path.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening the output {}", path.display()))?;
    if existing.unterminated {
        file.write_all(b"\n")?;
    }
    Ok((file, existing))
}

/// What the output at `path` holds ([`scan_output`]), read without changing
/// it, as [`open_output`] reads it: nothing when it does not exist yet or is
/// not a regular file. `run --resume` reads it before the model loads to
/// check only the inputs that still need a record.
pub fn read_output(path: &Path) -> Result<Existing> {
    if !is_file_or_missing(path) {
        return Ok(Existing::default());
    }
    read_existing(path).map(|(_, existing)| existing)
}

/// The bytes of the output at `path` (none when it does not exist yet) and
/// what they hold. A crash can cut a record inside a multi-byte character;
/// any other invalid UTF-8 is an error, and so is a line that is not a
/// record: such a file (an input list, notes, an image) is not an output.
fn read_existing(path: &Path) -> Result<(Vec<u8>, Existing)> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error).with_context(|| format!("reading the output {}", path.display())),
    };
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        // A cut character at the very end: the valid part ends inside the last line.
        Err(error) if error.error_len().is_none() => std::str::from_utf8(&bytes[..error.valid_up_to()])?,
        Err(_) => bail!("{} is not a UTF-8 JSON-lines file", path.display()),
    };
    let existing = scan_output(text);
    if let Some(line) = existing.foreign {
        bail!(
            "{}: its line {line} is not a record (a JSON object with a \"path\"), so records are not appended to it",
            path.display()
        );
    }
    Ok((bytes, existing))
}

/// Whether `path` is a regular file or does not exist yet (the output
/// `--resume` can read).
pub fn is_file_or_missing(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(_) => true,
    }
}

/// This run as [`Run`] compares it with records: `mode`, `precision`, the
/// overlay digest of its weights, its generation `options` and its
/// runner's `kernels`.
pub fn this_run(
    mode: Option<Mode>,
    precision: &str,
    overlay: Option<&str>,
    options: &GenerationOptions,
    kernels: Option<Kernels>,
) -> Run {
    Run {
        mode: mode.map(|mode| mode.label().to_owned()),
        precision: Some(precision.to_owned()),
        overlay: overlay.map(str::to_owned),
        routed: options.route,
        settings: Some(Settings::from(options)),
        kernels,
    }
}

/// A warning for a `--resume --escalate` run whose output holds fast-mode
/// records that the repetition stop ended without an escalation: their
/// pages count as done, so this run does not reread them.
pub fn escalation_warning(existing: &Existing) -> Option<String> {
    (existing.unescalated > 0).then(|| {
        format!(
            "the output holds {} fast-mode records that the repetition stop ended and that were not reread in \
             near-exact mode; --resume skips their pages, --escalate does not reach them",
            existing.unescalated
        )
    })
}

/// A warning when successful records in the output ran otherwise than
/// `this` run (another mode, precision, overlay, routing, size bounds,
/// output cap, margin crop, repetition stop, exp or prefill rounding):
/// `--resume` skips their pages, it does not reread them.
pub fn resume_warning(existing: &Existing, this: &Run) -> Option<String> {
    let others: Vec<String> = existing
        .runs
        .iter()
        .filter(|(run, _)| !run.matches(this))
        .map(|(run, count)| match run.precision {
            Some(_) => format!("{count} in {run}"),
            None => format!("{count} {run}"),
        })
        .collect();
    (!others.is_empty()).then(|| {
        format!(
            "the output holds records that ran otherwise than this run, {this}: {}; \
             --resume skips their pages, it does not reread them",
            others.join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::{Sink, record_line, test_result as result};

    #[test]
    fn resume_finds_successful_records_and_drops_a_record_cut_by_a_crash() {
        let ok = |path: &str, mode: &str, precision: &str| {
            format!(r#"{{"path":"{path}","page":0,"text":"t","mode":"{mode}","precision":"{precision}"}}"#)
        };
        let escalated = r#"{"path":"c.png","page":2,"text":"t","mode":"near-exact","precision":"w16-body-kv-q16","escalated_from":{"mode":"fast","finish_reason":"repetition","output_tokens":300,"total_ms":9.0}}"#;
        let failed = r#"{"path":"d.png","page":3,"error":"boom"}"#;
        let complete = format!(
            "{}\n{}\n{escalated}\n{failed}\n\n",
            ok("a.png", "fast", "w8-body-kv-q8"),
            ok("b.png", "fast", "w8-body-kv-q8")
        );
        let cut = r#"{"path":"e.png","page":4,"text":"half a rec"#;
        let text = format!("{complete}{cut}");
        let existing = scan_output(&text);
        let done: HashSet<String> = ["a.png", "b.png", "c.png"].map(String::from).into();
        assert_eq!(existing.done, done);
        assert_eq!((existing.keep, existing.cut), (complete.len(), cut.len()));
        assert!(!existing.unterminated && existing.foreign.is_none());
        let fast = Run {
            mode: Some("fast".to_owned()),
            precision: Some("w8-body-kv-q8".to_owned()),
            ..Run::default()
        };
        assert_eq!(existing.runs[&fast], 2);
        let escalated = Run {
            mode: Some("fast".to_owned()),
            ..Run::default()
        };
        assert_eq!(existing.runs[&escalated], 1);
        // A complete last record without its newline is kept and gets one.
        let last = ok("f.png", "fast", "w8-body-kv-q8");
        let existing = scan_output(&format!("{complete}{last}"));
        assert!(existing.done.contains("f.png") && existing.unterminated && existing.cut == 0);
        // The first line that is not a record is found, even a last one or
        // a JSON object without a `path`; such lines are not cut.
        let existing = scan_output("not json\n[1]\nplain tail");
        assert_eq!(
            (existing.foreign, existing.keep, existing.unterminated),
            (Some(1), 23, true)
        );
        assert_eq!(scan_output(&format!("{complete}{{\"a\":1}}")).foreign, Some(6));
        assert_eq!(scan_output("{\"a\":1}").cut, 0);
        // NUL bytes after the last record (a power loss) were never written.
        let nuls = format!("{complete}\0\0\0\0");
        let existing = scan_output(&nuls);
        assert_eq!(
            (existing.keep, existing.cut, existing.foreign),
            (complete.len(), 4, None)
        );
        // NULs in place of an earlier record, newline included, run into
        // the next record's line: that record counts, the lost one does not,
        // and a line of NULs alone is blank.
        let (lost, next) = (
            ok("g.png", "fast", "w8-body-kv-q8"),
            ok("h.png", "fast", "w8-body-kv-q8"),
        );
        let gap = format!("{complete}{}{next}\n\0\0\n", "\0".repeat(lost.len() + 1));
        let existing = scan_output(&gap);
        assert!(existing.done.contains("h.png") && !existing.done.contains("g.png"));
        assert_eq!((existing.keep, existing.cut, existing.foreign), (gap.len(), 0, None));
        // A byte order mark is skipped and kept.
        let marked = format!("\u{feff}{}\r\n", ok("a.png", "fast", "w8-body-kv-q8"));
        let existing = scan_output(&marked);
        assert!(existing.done.contains("a.png") && existing.foreign.is_none());
        assert_eq!((existing.keep, existing.unterminated), (marked.len(), false));
        assert!(!scan_output("\u{feff}").unterminated);
        assert_eq!(scan_output(""), Existing::default());
    }

    #[test]
    fn a_resume_with_escalate_warns_about_loops_it_will_not_reread() {
        let record = |path: &str, mode: &str, reason: &str, extra: &str| {
            format!(r#"{{"path":"{path}","page":0,"text":"t","mode":"{mode}","finish_reason":"{reason}"{extra}}}"#)
        };
        let escalated =
            r#","escalated_from":{"mode":"fast","finish_reason":"repetition","output_tokens":3,"total_ms":1.0}"#;
        let text = [
            record("a.png", "fast", "eos", ""),
            record("b.png", "fast", "repetition", ""),
            record("c.png", "near-exact", "eos", escalated),
            record("d.png", "fast", "repetition", r#","escalation_error":"no model""#),
            record("e.png", "near-exact", "repetition", ""),
        ]
        .join("\n");
        let existing = scan_output(&(text + "\n"));
        // The two fast loops without a near-exact rerun, the failed one too.
        assert_eq!(existing.unescalated, 2);
        let warning = escalation_warning(&existing).unwrap();
        assert!(warning.starts_with("the output holds 2 fast-mode records"), "{warning}");
        assert_eq!(
            escalation_warning(&scan_output(&record("a.png", "fast", "eos", ""))),
            None
        );
    }

    #[test]
    fn the_output_file_is_repaired_before_records_are_appended() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("book.jsonl");
        let sink_line = |file| {
            let mut sink = Sink::new(Vec::new(), Some(file), false, None);
            sink.page(Path::new("z.png"), 9, &result("z")).unwrap();
            assert!(sink.stdout.is_empty());
        };
        // Missing: created.
        let (file, existing) = open_output(&path).unwrap();
        assert_eq!(existing, Existing::default());
        sink_line(file);
        // A record cut inside a multi-byte character is removed.
        let first = std::fs::read_to_string(&path).unwrap();
        let mut bytes = first.clone().into_bytes();
        bytes.extend_from_slice("{\"path\":\"y.png\",\"text\":\"é".as_bytes());
        bytes.pop();
        std::fs::write(&path, &bytes).unwrap();
        let (file, existing) = open_output(&path).unwrap();
        assert!(existing.done.contains("z.png") && existing.cut > 0);
        sink_line(file);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, format!("{first}{first}"));
        // A last line without its newline gets one.
        std::fs::write(&path, first.trim_end()).unwrap();
        let (file, _) = open_output(&path).unwrap();
        sink_line(file);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{first}{first}"));
        // Invalid UTF-8 before the end is refused, and so is a line that is
        // not a record; the file is left as it is.
        std::fs::write(&path, b"\xff\n{}\n").unwrap();
        assert!(open_output(&path).is_err());
        let list = format!("{first}page.png\n");
        std::fs::write(&path, &list).unwrap();
        let error = format!("{:#}", open_output(&path).unwrap_err());
        assert!(
            error.contains("book.jsonl: its line 2 is not a record (a JSON object with a \"path\")"),
            "{error}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), list);
        // Only a regular file is read; a directory is not an output.
        assert!(is_file_or_missing(&path) && is_file_or_missing(&dir.path().join("new.jsonl")));
        assert!(!is_file_or_missing(dir.path()));
        let error = format!("{:#}", open_output(dir.path()).unwrap_err());
        assert!(error.contains("is a directory"), "{error}");
    }

    /// A device (or a pipe) is written to without reading it first, which
    /// would block on `/dev/stdout` or a pipe.
    #[cfg(unix)]
    #[test]
    fn a_device_output_is_written_without_being_read() {
        let null = Path::new("/dev/null");
        assert!(!is_file_or_missing(null));
        let (file, existing) = open_output(null).unwrap();
        assert_eq!(existing, Existing::default());
        let mut sink = Sink::new(Vec::new(), Some(file), false, None);
        sink.page(Path::new("a.png"), 0, &result("a")).unwrap();
    }

    /// `record` as a line of an output, with the `options` its run asked for.
    fn line(record: &str, options: &GenerationOptions) -> String {
        let mut record: Map<String, Value> = serde_json::from_str(record).unwrap();
        record.insert("options".to_owned(), serde_json::to_value(options).unwrap());
        format!("{}\n", Value::Object(record))
    }

    #[test]
    fn a_resume_from_another_mode_warns() {
        // `run`'s defaults: the fitted budget at a fixed 1536, or routed.
        let fixed = GenerationOptions {
            fit_budget: true,
            ..GenerationOptions::default()
        };
        let auto = GenerationOptions {
            route: true,
            ..fixed.clone()
        };
        let text = [
            r#"{"path":"a.png","text":"t","mode":"fast","precision":"w8-body-kv-q8"}"#,
            r#"{"path":"b.png","text":"t","mode":"near-exact","precision":"w16-body-kv-q16","escalated_from":{"mode":"fast"}}"#,
        ]
        .map(|record| line(record, &fixed))
        .concat();
        let existing = scan_output(&text);
        let fast_rtn = this_run(Some(Mode::Fast), "w8-body-kv-q8", None, &fixed, None);
        assert_eq!(resume_warning(&existing, &fast_rtn), None);
        let near_exact = this_run(Some(Mode::NearExact), "w16-body-kv-q16", None, &fixed, None);
        let warning = resume_warning(&existing, &near_exact).unwrap();
        assert!(
            warning.contains(
                "this run, near-exact (w16-body-kv-q16): 1 escalated from fast, 1 in fast (w8-body-kv-q8, round-to-nearest)"
            ),
            "{warning}"
        );
        // Same mode, another precision (for example exact mode's labels).
        assert!(resume_warning(&existing, &this_run(Some(Mode::Fast), "fp32", None, &fixed, None)).is_some());
        // The GPTQ overlay against round-to-nearest, and routing: the
        // escalated record matches any fast run that routes alike.
        let gptq = this_run(
            Some(Mode::Fast),
            "w8-body-kv-q8",
            Some("0123456789abcdef"),
            &fixed,
            None,
        );
        let warning = resume_warning(&existing, &gptq).unwrap();
        assert!(
            warning
                .contains("fast (w8-body-kv-q8, overlay 0123456789ab): 1 in fast (w8-body-kv-q8, round-to-nearest);"),
            "{warning}"
        );
        let routed = this_run(Some(Mode::Fast), "w8-body-kv-q8", None, &auto, None);
        let warning = resume_warning(&existing, &routed).unwrap();
        assert!(warning.contains(": 1 escalated from fast, 1 in fast ("), "{warning}");
        // Records name their overlay in `plan` and their route in `route`.
        let existing = scan_output(&line(
            concat!(
                r#"{"path":"c.png","text":"t","mode":"fast","precision":"w8-body-kv-q8","#,
                r#""plan":{"overlay_sha256":"0123456789abcdef"},"route":{"max_dimension":768}}"#
            ),
            &auto,
        ));
        let gptq_routed = this_run(Some(Mode::Fast), "w8-body-kv-q8", Some("0123456789abcdef"), &auto, None);
        assert_eq!(resume_warning(&existing, &gptq_routed), None);
        let warning = resume_warning(&existing, &gptq).unwrap();
        assert!(warning.ends_with("1 in fast (w8-body-kv-q8, overlay 0123456789ab, --max-dimension auto); --resume skips their pages, it does not reread them"), "{warning}");
        // An edited record's digest that is not hex is shortened by
        // characters, not bytes.
        let existing = scan_output(&line(
            &format!(
                r#"{{"path":"d.png","text":"t","mode":"fast","precision":"w8-body-kv-q8","plan":{{"overlay_sha256":"a{}"}}}}"#,
                "é".repeat(12)
            ),
            &fixed,
        ));
        let warning = resume_warning(&existing, &gptq).unwrap();
        let shown = format!("1 in fast (w8-body-kv-q8, overlay a{})", "é".repeat(11));
        assert!(warning.contains(&shown), "{warning}");
    }

    /// Records carry the options their run asked for, so a resume notices
    /// another margin crop or padding, another fixed size and another cap.
    #[test]
    fn a_resume_with_other_options_warns() {
        let fixed = GenerationOptions {
            fit_budget: true,
            ..GenerationOptions::default()
        };
        let with = |change: &dyn Fn(&mut GenerationOptions)| {
            let mut options = fixed.clone();
            change(&mut options);
            options
        };
        let cropped = with(&|options| options.crop_margins = Some(24));
        let page = result("t");
        let text: String = [(&fixed, "a.png"), (&cropped, "b.png"), (&cropped, "c.png")]
            .iter()
            .map(|(options, path)| record_line(Path::new(path), 0, Ok(&page), Some(options)).unwrap() + "\n")
            .collect();
        let existing = scan_output(&text);
        let run = |options: &GenerationOptions| {
            let this = this_run(Some(Mode::NearExact), "w16-body-kv-q16", None, options, None);
            resume_warning(&existing, &this)
        };
        let warning = run(&fixed).unwrap();
        assert!(
            warning.contains(
                "this run, near-exact (w16-body-kv-q16): 2 in near-exact (w16-body-kv-q16, --crop-margins=24);"
            ),
            "{warning}"
        );
        let warning = run(&cropped).unwrap();
        assert!(warning.contains(": 1 in near-exact (w16-body-kv-q16);"), "{warning}");
        for other in [
            with(&|options| options.crop_margins = Some(40)),
            with(&|options| options.max_dimension = 1024),
            with(&|options| {
                options.max_new_tokens = 24;
                options.fit_budget = false;
            }),
        ] {
            // Every record ran otherwise than this run.
            let warning = run(&other).unwrap();
            assert!(
                warning.contains(
                    ": 1 in near-exact (w16-body-kv-q16), 2 in near-exact (w16-body-kv-q16, --crop-margins=24);"
                ),
                "{warning}"
            );
        }
        let flags = run(&with(&|options| {
            options.max_dimension = 1024;
            options.min_dimension = 32;
            options.max_new_tokens = 24;
            options.fit_budget = false;
        }))
        .unwrap();
        assert!(
            flags.contains(
                "this run, near-exact (w16-body-kv-q16, --max-new-tokens 24, --min-dimension 32, --max-dimension 1024):"
            ),
            "{flags}"
        );
        // A record without options matches no run.
        let bare = scan_output(
            "{\"path\":\"d.png\",\"text\":\"t\",\"mode\":\"near-exact\",\"precision\":\"w16-body-kv-q16\"}\n",
        );
        let this = this_run(Some(Mode::NearExact), "w16-body-kv-q16", None, &fixed, None);
        let warning = resume_warning(&bare, &this).unwrap();
        assert!(
            warning.contains(": 1 in near-exact (w16-body-kv-q16, no options);"),
            "{warning}"
        );
    }

    /// Records' plans hold the runner's token-changing choices: a resume
    /// with another repetition stop, exp or prefill rounding warns, and the
    /// AVX2 and AVX-512 FP32 tiles, which agree bitwise, count as one.
    #[test]
    fn a_resume_with_other_runner_choices_warns() {
        let plan = |repetition_stop: bool, exp: &str, attention: &str, projection: &str| {
            serde_json::json!({
                "repetition_stop": repetition_stop, "exp": exp,
                "prefill": {"projection": projection, "attention": attention},
                "tuning": {"decode_fast_exp": null}
            })
        };
        let fixed = GenerationOptions {
            fit_budget: true,
            ..GenerationOptions::default()
        };
        let record = serde_json::json!({
            "path": "a.png", "text": "t", "mode": "fast", "precision": "w8-body-kv-q8",
            "plan": plan(true, "fast", "avx512-wide", "panel-avx2")
        });
        let existing = scan_output(&line(&record.to_string(), &fixed));
        let run = |plan: Value| {
            let this = this_run(
                Some(Mode::Fast),
                "w8-body-kv-q8",
                None,
                &fixed,
                Kernels::from_plan(&plan),
            );
            resume_warning(&existing, &this)
        };
        assert_eq!(run(plan(true, "fast", "avx2", "panel-avx2")), None);
        let mut pinned = plan(true, "fast", "avx2", "panel-avx2");
        pinned["tuning"]["decode_fast_exp"] = true.into();
        for (other, flag) in [
            (plan(false, "fast", "avx2", "panel-avx2"), "--stop-repetition=false"),
            (plan(true, "exact", "avx2", "panel-avx2"), "--exp exact"),
            (plan(true, "fast", "bf16", "panel-avx2"), "BF16 attention"),
            (plan(true, "fast", "neon", "panel-avx2"), "neon attention"),
            (plan(true, "fast", "avx2", "panel-bf16"), "BF16 projections"),
            (pinned, "--tune decode-exp=fast"),
        ] {
            let warning = run(other).unwrap();
            assert!(
                warning.contains(&format!(
                    "this run, fast (w8-body-kv-q8, round-to-nearest, {flag}): 1 in fast (w8-body-kv-q8, round-to-nearest);"
                )),
                "{warning}"
            );
        }
        // A runner's own plan reads the same way.
        let facts = crate::auto::ModelFacts {
            source: crate::WeightsSource::Checkpoint {
                dir: "model".into(),
                overlay: None,
                rtn: true,
            },
            profile: crate::quant::Profile::W8_BODY_KV_Q8,
            config: serde_json::from_str(include_str!("../../tests/fixtures/model-config.json")).unwrap(),
            body_bits: Some(8),
            overlay_sha256: None,
            exception_bytes: 0,
        };
        let config = crate::RunnerConfig {
            repetition_stop: false,
            exp: crate::ExpMode::Exact,
            ..crate::RunnerConfig::default()
        };
        let resolved = crate::Resolved::new(crate::HostInfo::detect(), &config, &facts, Path::new("model"));
        let kernels = Kernels::of(&resolved).unwrap();
        assert_eq!((kernels.repetition_stop, kernels.exp.as_str()), (false, "exact"));
        assert!(["fp32", "bf16", "neon", "scalar"].contains(&kernels.attention.as_str()));
    }
}
