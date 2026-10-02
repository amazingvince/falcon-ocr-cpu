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
/// agree bitwise), BF16 projections, decode chunking, a pinned decode exp
/// and the image decoder.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Kernels {
    pub repetition_stop: bool,
    pub exp: String,
    pub attention: String,
    pub bf16_projections: bool,
    pub decode_exp: Option<bool>,
    /// Effective chunks of the split-cache scan, including the mode's
    /// default; `None` for a compact cache or an unrecorded cache kind.
    pub split_chunks: Option<u64>,
    /// The decoder named by the build, when the plan records it.
    pub image_decoder: Option<String>,
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
        let chunks = plan
            .get("tuning")
            .and_then(|tuning| tuning.get("split_chunks"))
            .and_then(Value::as_u64);
        let split_chunks = match plan.get("kv_cache").and_then(Value::as_str) {
            Some("f32-split") => Some(chunks.unwrap_or(1)),
            Some("q16" | "q8" | "q8r" | "q4r") => Some(chunks.unwrap_or(4)),
            // Compact caches do not seal a prefix or use this knob.
            _ => None,
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
            split_chunks,
            image_decoder: plan.get("image_decoder").and_then(Value::as_str).map(str::to_owned),
        })
    }

    /// This runner's (`Runner::resolved`).
    pub fn of(resolved: &crate::Resolved) -> Option<Self> {
        Self::from_plan(&serde_json::to_value(resolved).ok()?)
    }

    /// The numerical settings named in a resume warning.
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
        if let Some(chunks) = self.split_chunks {
            flags.push(format!("--tune split-chunks={chunks}"));
        }
        if let Some(decoder) = &self.image_decoder {
            flags.push(format!("image decoder: {decoder}"));
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
    /// The runner's token-changing choices (the near-exact rerun's for an
    /// escalated record); `None` for a plan without them.
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
            && self.kernels.as_ref().is_none_or(|kernels| {
                let mut expected = run.kernels.clone();
                if self.precision.is_none()
                    && let Some(expected) = &mut expected
                {
                    // Escalation shares the run's configuration, but its
                    // 16-bit body cannot use fast mode's BF16 prefill.
                    if expected.attention == "bf16" {
                        expected.attention = "fp32".to_owned();
                    }
                    expected.bf16_projections = false;
                }
                Some(kernels) == expected.as_ref()
            })
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
            None => {
                details.push("no options".to_owned());
                if self.routed {
                    details.push("--max-dimension auto".to_owned());
                }
            }
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
                kernels: record.get("plan").and_then(Kernels::from_plan),
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
/// and `text` and no `error`), and how to append after it. JSON escapes NUL,
/// so a line holding a raw NUL byte is damage, what a power loss leaves in
/// place of data that was never written: a record after leading NULs counts,
/// and any other such line is skipped, so the pages of the records it
/// replaced run again; a file with damaged lines and no record is not an
/// output. A last line without its newline that is not valid JSON but could
/// be the start of a record ([`starts_like_a_record`], NULs around it
/// aside) was cut short by a crash and is dropped; any other last line
/// without its newline is kept and gets one, or is not a record. A byte
/// order mark (an editor's) is kept and skipped.
pub fn scan_output(text: &str) -> Existing {
    let mut existing = Existing::default();
    let bom = if text.starts_with('\u{feff}') {
        '\u{feff}'.len_utf8()
    } else {
        0
    };
    let mut start = bom;
    let (mut records, mut damaged) = (0, None);
    for (number, piece) in (1..).zip(text[bom..].split_inclusive('\n')) {
        let line = piece.trim_end_matches(['\n', '\r']);
        let nul = line.contains('\0');
        let record = match serde_json::from_str::<Value>(line.trim_start_matches('\0')) {
            Ok(Value::Object(record)) if existing.add(&record) => {
                records += 1;
                true
            }
            Err(_) if !piece.ends_with('\n') && starts_like_a_record(line.trim_matches('\0')) => {
                existing.cut = piece.len();
                break;
            }
            _ if nul => {
                damaged = damaged.or(Some(number));
                true
            }
            _ => line.trim().is_empty(),
        };
        if !record {
            existing.foreign = existing.foreign.or(Some(number));
        }
        start += piece.len();
    }
    if records == 0
        && let Some(line) = damaged
    {
        existing.foreign = Some(existing.foreign.map_or(line, |foreign| foreign.min(line)));
    }
    existing.keep = start;
    existing.unterminated = start > bom && !text[..start].ends_with('\n');
    existing
}

/// Whether `text` could be the start of a line that [`super::record_line`]
/// writes: it opens with `{"path":` (or is a start of that), holds none of
/// the control characters JSON escapes (below U+0020), and fails to parse
/// only for want of more input (also once a digit is added, for a number
/// cut after its `.`, `e` or `-`). Only such a last line is taken for a
/// record cut short, so that another file given as `--output` (one holding
/// NaN, a trailing comma, two objects) is refused, not truncated.
fn starts_like_a_record(text: &str) -> bool {
    const START: &str = "{\"path\":";
    let cut = |text: &str| matches!(serde_json::from_str::<Value>(text), Err(error) if error.is_eof());
    (text.starts_with(START) || START.starts_with(text))
        && !text.contains(|c: char| c < ' ')
        && (cut(text) || cut(&format!("{text}0")))
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
/// output cap, margin crop, repetition stop, exp, prefill rounding, decode
/// chunking or image decoder):
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
mod tests;
