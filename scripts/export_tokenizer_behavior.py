#!/usr/bin/env python3
"""Capture actual pinned Transformers BPE decoding, cleanup and Python strip behavior."""
import hashlib
import importlib.metadata
import inspect
import json
import pathlib

from transformers import AutoTokenizer

from fetch_reference import REVISION, sha256


def main():
    model = pathlib.Path("artifacts/model")
    assets = json.loads((model / "artifact-manifest.json").read_text(encoding="utf-8"))
    files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for name in files:
        assert sha256(model / name) == assets["files"][name]["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
    examples = [
        ("ellipsis", "Lesson 14 What colour's your ...?\nDate . . .\nAs is known to us, ..."),
        ("punctuation", "word . word , word ! word ? : ; ( text ) [ text ]"),
        ("contractions", "I 'm sure it 's true ; you 're here and we 've seen what they 'll do . can n't wo n't"),
        ("special_markup", ">>UNUSED_261<<\n<|OCR_PLAIN|>\n<table><tr><td>A . B</td><td>12.50</td></tr></table><|end_of_query|>"),
        ("both_stops", "before<|end_of_query|>between<|end_of_text|>after"),
        ("outer_ascii_whitespace", " \t\r\n  spaced inner  text \n\r\t "),
        ("outer_unicode_whitespace", "\u0085\u00a0\u1680\u2000\u2007\u202f\u205f\u3000text\u3000\u205f\u202f\u2007\u2000\u1680\u00a0\u0085"),
        ("python_ascii_separator_strip", "\u001c\u001d\u001e\u001ftext\u001f\u001e\u001d\u001c"),
        ("non_whitespace_format_marks", "\u200b\ufefftext\ufeff\u200b"),
        ("mixed_unicode", "  中文 . é e\u0301 — … 🙂  "),
        ("empty", ""),
        ("only_whitespace", " \t\n\u00a0\u001c"),
    ]
    inputs = [{"name": name, "input_text": text,
               "token_ids": tokenizer.encode(text, add_special_tokens=False)} for name, text in examples]
    for sample in ["eb9484a26254fd69", "26587cc8d139432f"]:
        path = pathlib.Path("artifacts/reference/corpus-v3-fp32-4096") / (sample + ".json")
        page = json.loads(path.read_text(encoding="utf-8"))
        inputs.append({"name": "corpus." + sample, "token_ids": page["token_ids"], "source_record": path.as_posix(),
                       "source_record_sha256": sha256(path), "recorded_gpu_final": page["text"]})
    cases = []
    for case in inputs:
        ids = case["token_ids"]
        raw = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
        default = tokenizer.decode(ids, skip_special_tokens=False)
        cleaned = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        no_cleanup = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        final = default.replace("<|end_of_query|>", "").replace("<|end_of_text|>", "").strip()
        if "recorded_gpu_final" in case:
            assert final == case["recorded_gpu_final"], case["name"]
        cases.append({**case, "raw_decode": raw, "hf_default_decode": default, "hf_cleaned_decode": cleaned,
                      "hf_cleanup_false_decode": no_cleanup, "legacy_wordpiece_cleanup": tokenizer.clean_up_tokenization(raw),
                      "upstream_final": final, "raw_equals_hf_default": raw == default,
                      "explicit_cleanup_true_equals_raw": cleaned == raw})
    sources = {}
    for name, function in [("decode", tokenizer.decode), ("backend_decode", tokenizer._decode),
                           ("legacy_cleanup", tokenizer.clean_up_tokenization)]:
        path = pathlib.Path(inspect.getsourcefile(function))
        text = inspect.getsource(function)
        sources[name] = {"installed_file": str(path), "installed_file_sha256": sha256(path),
                         "method_source_sha256": hashlib.sha256(text.encode()).hexdigest(), "method_source": text}
    result = {"schema_version": 1, "model_revision": REVISION, "tokenizer_assets": {n: assets["files"][n] for n in files},
              "script_sha256": sha256(__file__), "versions": {n: importlib.metadata.version(n) for n in ["transformers", "tokenizers"]},
              "tokenizer_class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
              "backend_model": type(tokenizer.backend_tokenizer.model).__name__,
              "clean_up_tokenization_spaces_attribute": tokenizer.clean_up_tokenization_spaces,
              "force_bpe_cleanup_attribute": tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output,
              "sources": sources, "cases": cases, "gpu_execution": False,
              "upstream_final_policy": "Actual tokenizer.decode(skip_special_tokens=False), remove both named EOS strings, then Python str.strip().",
              "interpretation": "Transformers 5.14.1 ignores the configured cleanup=True for BPE unless the separate force-BPE-cleanup attribute is enabled. hf_cleaned_decode records actual decode(clean_up_tokenization_spaces=True), not the legacy WordPiece cleanup helper."}
    directory = pathlib.Path("artifacts/reference/tokenizer-cleanup-v1")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "fixture.json"
    if target.exists():
        raise FileExistsError("Preserve the frozen tokenizer fixture; use a new version for a changed export")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"fixture": target.as_posix(), "fixture_sha256": sha256(target), "cases": len(cases),
                "versions": result["versions"], "tokenizer_assets": result["tokenizer_assets"],
                "sources": {n: {k: v for k, v in r.items() if k != "method_source"} for n, r in sources.items()},
                "interpretation": result["interpretation"], "upstream_final_policy": result["upstream_final_policy"]}
    pathlib.Path("reference/tokenizer-cleanup-v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": target.as_posix(), "sha256": sha256(target), "cases": len(cases),
                      "all_raw_equal_default": all(c["raw_equals_hf_default"] for c in cases)}, indent=2))


if __name__ == "__main__":
    main()
