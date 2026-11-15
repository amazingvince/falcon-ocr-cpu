//! Ignored probe: where the repetition stop would end recorded outputs.
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use super::*;
use crate::tokenizer::OcrTokenizer;

/// Replay the `token_ids` of every JSON line in the files that
/// `PROBE_OUTPUTS` lists (a path list) through the stop, up to the first
/// stop token. Prints one line per output: its `source` and `path`, recorded
/// `finish_reason` and length, the token count each rule keeps (`-` when it
/// does not fire), and the most counting cycles of any window spanning
/// `MIN_WINDOW` tokens. For every output the counting rule ends, it also
/// prints the text around the stop (tokenizer from `FALCON_OCR_MODEL`,
/// default `artifacts/model`), and at the end the mean time both rules take
/// per token.
#[test]
#[ignore = "reads recorded outputs; run with --nocapture"]
fn recorded_outputs_probe() {
    let lists = std::env::var_os("PROBE_OUTPUTS").expect("PROBE_OUTPUTS lists JSONL files of outputs");
    let model = std::env::var_os("FALCON_OCR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/model"));
    let tokenizer = OcrTokenizer::load(&model).unwrap();
    let stops = tokenizer.stop_ids();
    let (mut scanned, mut fired, mut tokens, mut elapsed) = (0, Vec::new(), 0, Duration::ZERO);
    for path in std::env::split_paths(&lists) {
        let text = std::fs::read_to_string(&path).unwrap();
        for (line, record) in text.lines().enumerate() {
            let Ok(record) = serde_json::from_str::<serde_json::Value>(record) else {
                continue;
            };
            let Some(ids) = record["token_ids"].as_array() else {
                continue;
            };
            let ids: Vec<u32> = ids.iter().map(|id| id.as_u64().unwrap() as u32).collect();
            let label = [&record["source"], &record["path"]]
                .iter()
                .filter_map(|field| field.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let label = if label.is_empty() {
                format!("{}:{}", path.display(), line + 1)
            } else {
                label
            };
            let reason = record["finish_reason"].as_str().unwrap_or("-");
            let (mut stop, mut generated) = (RepetitionStop::new(), Vec::new());
            let (mut exact, mut counting, mut most) = (None, None, 0);
            let started = Instant::now();
            for &token in ids.iter().take_while(|token| !stops.contains(token)) {
                generated.push(token);
                let (last, earlier) = generated.split_last().unwrap();
                if stop.exact(*last, earlier) {
                    exact.get_or_insert(generated.len());
                }
                let cycles = stop.counting.push(*last, earlier.len());
                if cycles >= MIN_COUNT_REPEATS {
                    counting.get_or_insert(generated.len());
                }
                most = most.max(cycles);
            }
            elapsed += started.elapsed();
            tokens += generated.len();
            scanned += 1;
            let show = |kept: Option<usize>| kept.map_or("-".to_owned(), |k| k.to_string());
            println!(
                "{label}\t{reason}\t{}\texact={}\tcounting={}\tcycles={most}",
                ids.len(),
                show(exact),
                show(counting)
            );
            if let Some(kept) = counting {
                let before = tokenizer.decode(&ids[kept.saturating_sub(80)..kept]).unwrap();
                let after = tokenizer.decode(&ids[kept..ids.len().min(kept + 40)]).unwrap();
                println!("  kept ...{before:?}\n  cut  {after:?}...");
                fired.push(label);
            }
        }
    }
    println!(
        "scanned {scanned} outputs ({tokens} tokens, {:.0} ns per token for both rules); the counting rule ends {}",
        elapsed.as_nanos() as f64 / tokens.max(1) as f64,
        fired.len()
    );
}
