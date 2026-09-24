//! Validate saved inference before counting it as completed or replaying text.
use anyhow::{Context, Result, ensure};
use falcon_ocr::GenerationOptions;
use serde_json::Value;

pub fn validate_result(value: &Value, options: &GenerationOptions, precision: &str) -> Result<()> {
    let ids = value["token_ids"]
        .as_array()
        .context("result token_ids missing")?;
    ensure!(
        !ids.is_empty() && ids.len() <= options.max_new_tokens,
        "invalid output length"
    );
    ensure!(
        ids.iter().all(|id| id.as_u64().is_some_and(|n| n < 65536)),
        "invalid output token ID"
    );
    ensure!(
        value["output_tokens"].as_u64() == Some(ids.len() as u64),
        "output count differs from token IDs"
    );
    ensure!(value["text"].is_string(), "result text missing");
    ensure!(
        value["teacher_forced"] == false && value["precision"] == precision,
        "result inference mode differs"
    );
    let last_is_stop = matches!(ids.last().and_then(Value::as_u64), Some(11 | 263));
    ensure!(
        ids[..ids.len() - 1]
            .iter()
            .all(|id| !matches!(id.as_u64(), Some(11 | 263))),
        "result continues after an earlier stop token"
    );
    match value["finish_reason"].as_str() {
        Some("eos") => ensure!(last_is_stop, "EOS result lacks final stop token"),
        Some("length") => ensure!(
            !last_is_stop && ids.len() == options.max_new_tokens,
            "length result does not end at requested cap"
        ),
        _ => anyhow::bail!("result finish_reason missing or invalid"),
    }
    let width = value["width"].as_u64().context("result width missing")?;
    let height = value["height"].as_u64().context("result height missing")?;
    ensure!(
        width > 0
            && height > 0
            && width.is_multiple_of(16)
            && height.is_multiple_of(16)
            && width <= options.max_dimension as u64
            && height <= options.max_dimension as u64,
        "result image dimensions invalid"
    );
    let input = value["input_tokens"]
        .as_u64()
        .context("result input_tokens missing")?;
    ensure!(
        input == (width / 16) * (height / 16) + 16,
        "result prefix length differs from pinned prompt/patch shape"
    );
    options.check_budget(usize::try_from(input)?, 16384)?;
    for name in [
        "image_decode_ms",
        "preprocessing_ms",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "time_to_first_token_ms",
    ] {
        ensure!(
            value["timings"][name]
                .as_f64()
                .is_some_and(|x| x.is_finite() && x >= 0.0),
            "result timing missing/invalid: {name}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn valid() -> Value {
        json!({"token_ids":[561,263],"output_tokens":2,"text":"Hello","teacher_forced":false,
            "precision":"fp32","finish_reason":"eos","width":256,"height":128,"input_tokens":144,
            "timings":{"image_decode_ms":0.,"preprocessing_ms":1.,"prefill_ms":2.,"decode_ms":3.,"total_ms":6.,"time_to_first_token_ms":3.}})
    }
    #[test]
    fn requires_complete_inference_record() {
        let options = GenerationOptions {
            max_dimension: 256,
            max_new_tokens: 2,
            ..Default::default()
        };
        assert!(validate_result(&valid(), &options, "fp32").is_ok());
        assert!(validate_result(&json!({}), &options, "fp32").is_err());
        for (field, replacement) in [
            ("output_tokens", json!(1)),
            ("finish_reason", json!("length")),
            ("teacher_forced", json!(true)),
            ("input_tokens", json!(145)),
            ("token_ids", json!([561, 65536])),
            ("text", Value::Null),
        ] {
            let mut record = valid();
            record[field] = replacement;
            assert!(
                validate_result(&record, &options, "fp32").is_err(),
                "{field}"
            );
        }
    }
    #[test]
    fn length_stop_requires_exact_cap_and_no_eos() {
        let options = GenerationOptions {
            max_dimension: 256,
            max_new_tokens: 2,
            ..Default::default()
        };
        let mut record = valid();
        record["token_ids"] = json!([561, 562]);
        record["finish_reason"] = "length".into();
        assert!(validate_result(&record, &options, "fp32").is_ok());
        record["token_ids"] = json!([561]);
        record["output_tokens"] = 1.into();
        assert!(validate_result(&record, &options, "fp32").is_err());
        record["token_ids"] = json!([263, 561]);
        record["output_tokens"] = 2.into();
        assert!(validate_result(&record, &options, "fp32").is_err());
    }
}
