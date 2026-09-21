//! The released tokenizer, full-page plain prompt, and upstream text cleanup.

use anyhow::{Context, Result, anyhow, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;
use tokenizers::Tokenizer;

pub const PLAIN_PROMPT: &str = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>";
const IMAGE_TOKEN: &str = "<|image|>";
const IMAGE_PREFIX: [u32; 5] = [244, 245, 246, 247, 248];
const IMAGE_ID: u32 = 227;
const IMAGE_END: u32 = 230;

pub struct OcrTokenizer {
    tokenizer: Tokenizer,
    prompt_suffix: Vec<u32>,
    stops: Vec<u32>,
}

impl OcrTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer_bytes = pinned_file(
            &path,
            "4a9892af2b1ef021a421f140c7e3c064f5b255f7d75ba18c883996d86e1cf15a",
        )?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| anyhow!("loading {}: {error}", path.display()))?;
        let config_path = model_dir.join("tokenizer_config.json");
        let config: serde_json::Value = serde_json::from_slice(&pinned_file(
            &config_path,
            "074e03d3fd56d190dac763ec5dfe75a728e15783fc5d919d4b5cbe72bcd24d26",
        )?)?;

        for (token, expected) in [
            (IMAGE_TOKEN, IMAGE_ID),
            ("<|image_cls|>", IMAGE_PREFIX[0]),
            ("<|image_reg_1|>", IMAGE_PREFIX[1]),
            ("<|image_reg_2|>", IMAGE_PREFIX[2]),
            ("<|image_reg_3|>", IMAGE_PREFIX[3]),
            ("<|image_reg_4|>", IMAGE_PREFIX[4]),
            ("<|end_of_image|>", IMAGE_END),
            ("<|end_of_text|>", 11),
            ("<|OCR_PLAIN|>", 257),
            ("<|pad|>", 0),
        ] {
            ensure!(
                tokenizer.token_to_id(token) == Some(expected),
                "unsupported tokenizer: expected {token} to have ID {expected}"
            );
        }
        ensure!(
            config
                .get("bos_token")
                .is_none_or(serde_json::Value::is_null),
            "the pinned tokenizer does not prepend BOS; unexpected bos_token metadata"
        );
        let empty = tokenizer
            .encode("", true)
            .map_err(|e| anyhow!("encoding empty prompt: {e}"))?;
        ensure!(
            empty.get_ids().is_empty(),
            "unexpected prefix tokens from tokenizer post-processor"
        );
        let prompt_suffix = tokenizer
            .encode(PLAIN_PROMPT.strip_prefix(IMAGE_TOKEN).unwrap(), true)
            .map_err(|e| anyhow!("encoding plain OCR prompt: {e}"))?
            .get_ids()
            .to_vec();
        let end_query = tokenizer
            .token_to_id("<|end_of_query|>")
            .context("missing <|end_of_query|> token")?;
        Ok(Self {
            tokenizer,
            prompt_suffix,
            stops: vec![11, end_query],
        })
    }

    pub fn prompt(&self, patch_count: usize) -> Result<Vec<u32>> {
        ensure!(
            patch_count > 0,
            "an OCR prompt requires at least one image patch"
        );
        let capacity = patch_count
            .checked_add(6)
            .and_then(|v| v.checked_add(self.prompt_suffix.len()))
            .context("OCR prompt length overflow")?;
        let mut ids = Vec::with_capacity(capacity);
        ids.extend_from_slice(&IMAGE_PREFIX);
        ids.resize(IMAGE_PREFIX.len() + patch_count, IMAGE_ID);
        ids.push(IMAGE_END);
        ids.extend_from_slice(&self.prompt_suffix);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // The pinned Transformers 5.14.1 TokenizersBackend deliberately skips
        // WordPiece-style space cleanup for this BPE tokenizer, even though the
        // checkpoint config says clean_up_tokenization_spaces=true. Preserve
        // punctuation spacing, special tokens and table markup; remove only the
        // two stop-token spellings and apply the reference's Python str.strip.
        let text = self
            .tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow!("decoding OCR output: {e}"))?;
        Ok(text
            .replace("<|end_of_query|>", "")
            .replace("<|end_of_text|>", "")
            .trim_matches(python_whitespace)
            .to_owned())
    }

    pub fn stop_ids(&self) -> Vec<u32> {
        self.stops.clone()
    }
}

fn pinned_file(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        actual == expected,
        "{} SHA-256 mismatch: expected {expected}, got {actual}",
        path.display()
    );
    Ok(bytes)
}

// Python 3.12 str.isspace includes four ASCII information separators that Rust's
// Unicode White_Space predicate excludes. Spell out the pinned set explicitly.
fn python_whitespace(value: char) -> bool {
    matches!(value, '\u{0009}'..='\u{000d}' | '\u{001c}'..='\u{0020}'
        | '\u{0085}' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}'
        | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_strip_preserves_interior_spacing_and_zero_width_characters() {
        assert_eq!(
            "\u{1c}\u{85} Date . . . \n\u{1f}".trim_matches(python_whitespace),
            "Date . . ."
        );
        assert_eq!(
            "\u{200b}x\u{feff}".trim_matches(python_whitespace),
            "\u{200b}x\u{feff}"
        );
    }

    #[test]
    fn rejects_modified_tokenizer_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("tokenizer.json"), b"{}").unwrap();
        let error = OcrTokenizer::load(directory.path())
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("SHA-256 mismatch"));
    }

    #[test]
    #[ignore = "requires the downloaded pinned model artifacts"]
    fn pinned_tokenizer_prompt_and_decode() {
        let dir = std::env::var_os("FALCON_OCR_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/model"));
        let tokenizer = OcrTokenizer::load(&dir).unwrap();
        let ids = tokenizer.prompt(4).unwrap();
        assert_eq!(
            &ids[..10],
            &[244, 245, 246, 247, 248, 227, 227, 227, 227, 230]
        );
        assert_eq!(ids.last(), Some(&257));
        assert_eq!(tokenizer.stop_ids(), vec![11, 263]);
        let encoded = tokenizer
            .tokenizer
            .encode("  Hello , OCR !\n<td>A</td><|end_of_query|>", false)
            .unwrap();
        assert_eq!(
            tokenizer.decode(encoded.get_ids()).unwrap(),
            "Hello , OCR !\n<td>A</td>"
        );
        assert!(tokenizer.prompt(0).is_err());
    }

    #[test]
    #[ignore = "requires pinned tokenizer assets and independently exported Transformers fixture"]
    fn pinned_transformers_decode_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("artifacts/reference/tokenizer-cleanup-v1/fixture.json");
        let bytes = pinned_file(
            &path,
            "2663c936b41d4ae49081f747f17c3526b6d680c3f93f7214655dc5d218db454b",
        )
        .unwrap();
        let fixture: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let tokenizer = OcrTokenizer::load(&root.join("artifacts/model")).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 14);
        for case in cases {
            let ids: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(
                tokenizer.tokenizer.decode(&ids, false).unwrap(),
                case["raw_decode"].as_str().unwrap(),
                "raw: {}",
                case["name"]
            );
            assert_eq!(
                tokenizer.decode(&ids).unwrap(),
                case["upstream_final"].as_str().unwrap(),
                "final: {}",
                case["name"]
            );
        }
    }
}
