//! Output recovery and resume comparisons, including escalated records.
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
    // and a line of NULs alone is skipped.
    let (lost, next) = (
        ok("g.png", "fast", "w8-body-kv-q8"),
        ok("h.png", "fast", "w8-body-kv-q8"),
    );
    let gap = format!("{complete}{}{next}\n\0\0\n", "\0".repeat(lost.len() + 1));
    let existing = scan_output(&gap);
    assert!(existing.done.contains("h.png") && !existing.done.contains("g.png"));
    assert_eq!((existing.keep, existing.cut, existing.foreign), (gap.len(), 0, None));
    // A hole of whole blocks starts and ends inside records: the line
    // that holds it is skipped, and both of its records' pages run again.
    let block = format!(
        "{complete}{}\0\0\0\0{}\n{}\n",
        &lost[..20],
        &next[20..],
        ok("i.png", "fast", "w8-body-kv-q8")
    );
    let existing = scan_output(&block);
    assert!(existing.done.contains("i.png") && !existing.done.contains("g.png") && !existing.done.contains("h.png"));
    assert_eq!((existing.keep, existing.cut, existing.foreign), (block.len(), 0, None));
    // A record cut short with NULs after it is cut, also as a file's
    // only line, and so is a tail of NULs alone.
    let tail = format!("{complete}{}\0\0", &lost[..20]);
    assert_eq!(scan_output(&tail).cut, 22);
    assert_eq!(scan_output(&lost[..20]).cut, 20);
    assert_eq!(scan_output("{\"pa").cut, 4);
    // A file of damaged lines and no record (UTF-16 text) is not an output.
    assert_eq!(scan_output("a\0b\0\n\0c\0\n").foreign, Some(1));
    // Nor is a one-line file that is not a record, with or without NULs:
    // it is refused, never taken for a cut record and truncated.
    for other in [
        "{\"note\": NaN}",
        "{\"note\":\"a\"}{\"todo\":\"keep this\"}",
        "{\0\"\0n\0o\0t\0e\0\"\0:\0 \x001\0}\0",
        "{\"path\":\"a.png\"\u{1}",
        "{\"path\":\"a.png\",\"score\":NaN}",
        "{\"path\":\"a.png\",}",
    ] {
        let existing = scan_output(other);
        assert_eq!((existing.cut, existing.foreign), (0, Some(1)), "{other:?}");
    }
    // A record cut short is cut whatever it holds that JSON writes raw
    // (DEL, C1 controls), and inside a number.
    for cut in [
        "{\"path\":\"a.png\",\"text\":\"x\u{7f}y\u{85}z",
        "{\"path\":\"a.png\",\"page\":1.",
    ] {
        assert_eq!(scan_output(cut).cut, cut.len(), "{cut:?}");
    }
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
    // legacy escalated record has no plan to compare.
    let gptq = this_run(
        Some(Mode::Fast),
        "w8-body-kv-q8",
        Some("0123456789abcdef"),
        &fixed,
        None,
    );
    let warning = resume_warning(&existing, &gptq).unwrap();
    assert!(
        warning.contains("fast (w8-body-kv-q8, overlay 0123456789ab): 1 in fast (w8-body-kv-q8, round-to-nearest);"),
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
        warning
            .contains("this run, near-exact (w16-body-kv-q16): 2 in near-exact (w16-body-kv-q16, --crop-margins=24);"),
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
            warning
                .contains(": 1 in near-exact (w16-body-kv-q16), 2 in near-exact (w16-body-kv-q16, --crop-margins=24);"),
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
    let bare =
        scan_output("{\"path\":\"d.png\",\"text\":\"t\",\"mode\":\"near-exact\",\"precision\":\"w16-body-kv-q16\"}\n");
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
    // Escalation keeps the near-exact plan. Its FP32 prefill is
    // expected even when fast mode uses BF16, but exp, the stop and
    // pinned decode exp still have to agree with the resumed run.
    let mut escalated = record.clone();
    escalated["mode"] = "near-exact".into();
    escalated["precision"] = "w16-body-kv-q16".into();
    escalated["escalated_from"] = serde_json::json!({"mode": "fast"});
    let existing = scan_output(&line(&escalated.to_string(), &fixed));
    assert_eq!(
        existing.runs.keys().next().unwrap().kernels,
        Kernels::from_plan(&escalated["plan"])
    );
    let resumed = |plan: Value| {
        this_run(
            Some(Mode::Fast),
            "w8-body-kv-q8",
            None,
            &fixed,
            Kernels::from_plan(&plan),
        )
    };
    for attention in ["avx2", "avx512-wide", "bf16"] {
        for projection in ["panel-avx2", "panel-bf16"] {
            assert_eq!(
                resume_warning(&existing, &resumed(plan(true, "fast", attention, projection))),
                None
            );
        }
    }
    let mut pinned = plan(true, "fast", "bf16", "panel-avx2");
    pinned["tuning"]["decode_fast_exp"] = true.into();
    for other in [
        plan(false, "fast", "bf16", "panel-avx2"),
        plan(true, "exact", "bf16", "panel-avx2"),
        plan(true, "fast", "neon", "panel-avx2"),
        plan(true, "fast", "scalar", "panel-avx2"),
        pinned,
    ] {
        assert!(resume_warning(&existing, &resumed(other)).is_some());
    }
    escalated["plan"]["exp"] = "exact".into();
    let existing = scan_output(&line(&escalated.to_string(), &fixed));
    assert_eq!(
        resume_warning(&existing, &resumed(plan(true, "exact", "bf16", "panel-avx2"))),
        None
    );
    let warning = resume_warning(&existing, &resumed(plan(true, "fast", "bf16", "panel-avx2"))).unwrap();
    assert!(warning.contains("1 escalated from fast (--exp exact)"), "{warning}");
    // Plans also name the decoder and the effective split-cache chunking.
    // Explicit defaults match automatic choices; compact caches do not
    // use the split-cache knob at all.
    for (cache, default_chunks) in [
        ("f32-compact", None),
        ("f32-split", Some(1)),
        ("q16", Some(4)),
        ("q8", Some(4)),
        ("q8r", Some(4)),
        ("q4r", Some(4)),
    ] {
        let mut saved = record.clone();
        saved["path"] = "page.jpg".into();
        saved["plan"]["kv_cache"] = cache.into();
        saved["plan"]["image_decoder"] = "libjpeg-turbo (Pillow-exact)".into();
        let compare = |saved: &Value, current: &Value| {
            resume_warning(
                &scan_output(&line(&saved.to_string(), &fixed)),
                &resumed(current.clone()),
            )
        };
        let mut current = saved["plan"].clone();
        assert_eq!(compare(&saved, &current), None);
        current["image_decoder"] = "image-rs (not Pillow-exact)".into();
        let warning = compare(&saved, &current).expect("a changed decoder must warn");
        assert!(
            warning.contains("image-rs") && warning.contains("libjpeg-turbo"),
            "{warning}"
        );
        current = saved["plan"].clone();
        current["tuning"]["split_chunks"] = default_chunks.into();
        assert_eq!(compare(&saved, &current), None, "{cache}: explicit default");
        let different = if default_chunks == Some(4) { 1 } else { 4 };
        current["tuning"]["split_chunks"] = different.into();
        let warning = compare(&saved, &current);
        assert_eq!(warning.is_some(), default_chunks.is_some(), "{cache}: {warning:?}");
        if let Some(warning) = warning {
            assert!(
                warning.contains(&format!("--tune split-chunks={different}")),
                "{warning}"
            );
        }
        // A saved override must also warn when resuming with defaults.
        let automatic = saved["plan"].clone();
        saved["plan"] = current;
        assert_eq!(compare(&saved, &automatic).is_some(), default_chunks.is_some());
    }
    // The same metadata survives successful escalation to a 16-bit body.
    escalated["plan"]["exp"] = "fast".into();
    escalated["plan"]["kv_cache"] = "q16".into();
    escalated["plan"]["image_decoder"] = "libjpeg-turbo (Pillow-exact)".into();
    let existing = scan_output(&line(&escalated.to_string(), &fixed));
    let mut current = escalated["plan"].clone();
    current["kv_cache"] = "q8".into();
    current["prefill"]["attention"] = "bf16".into();
    current["tuning"]["split_chunks"] = 4.into();
    assert_eq!(resume_warning(&existing, &resumed(current.clone())), None);
    current["tuning"]["split_chunks"] = 1.into();
    assert!(resume_warning(&existing, &resumed(current.clone())).is_some());
    current["tuning"]["split_chunks"] = 4.into();
    current["image_decoder"] = "image-rs (not Pillow-exact)".into();
    assert!(resume_warning(&existing, &resumed(current)).is_some());
    // A runner's own plan reads the same way.
    let facts = crate::auto::ModelFacts {
        source: crate::WeightsSource::Checkpoint {
            dir: "model".into(),
            overlay: None,
            rtn: true,
        },
        profile: crate::quant::Profile::W8_BODY_KV_Q8,
        config: serde_json::from_str(include_str!("../../../tests/fixtures/model-config.json")).unwrap(),
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
    assert_eq!(kernels.split_chunks, Some(4));
    assert_eq!(kernels.image_decoder.as_deref(), Some(resolved.image_decoder.as_str()));
}
