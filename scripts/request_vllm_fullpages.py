#!/usr/bin/env python3
"""Run bounded natural full-page HTTP requests against the pinned vLLM fork."""
import argparse
import base64
import json
import pathlib
import re
import time

import requests

from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from validate_gpu_reference_record import validate_gpu_reference_record


def require_equal(label, actual, expected):
    if actual != expected:
        raise ValueError(f"Reference prerequisite failed: {label}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/serving-fullpages-v1-lock.json"))
    parser.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("artifacts/reference/corpus-v3-fp32-4096"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/vllm-fullpages-fp32-4096"))
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("reference/vllm-fullpages-fp32.json"))
    args = parser.parse_args()
    if args.report.exists() or (args.output / "request-sources").exists():
        raise ValueError("Full-page request evidence already exists; preserve previous attempts")
    environment = json.loads((args.output / "environment.json").read_text(encoding="utf-8"))
    assert environment["model_revision"] == REVISION and environment["weights_sha256"] == WEIGHT_SHA256
    assert environment["precision"] == "fp32" and environment["torch_allow_tf32"] is False
    arguments = environment["server_arguments"]
    overrides = json.loads(arguments[arguments.index("--hf-overrides") + 1])
    expected_image_config = {"spatial_patch_size": 16, "merge_size": 1, "min_image_size": 64,
                             "max_image_size": 1536, "min_pixels": 3136, "max_pixels": 10035200}
    assert overrides == {"image_config": expected_image_config}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_archive = args.output / "request-sources"
    source_archive.mkdir()
    source_hashes = {}
    for name in ["request_vllm_fullpages.py", "fetch_reference.py", "validate_gpu_reference_record.py"]:
        target = source_archive / name
        target.write_bytes((pathlib.Path(__file__).parent / name).read_bytes())
        source_hashes[name] = sha256(target)
    url = "http://127.0.0.1:18080"
    requests.get(url + "/health", timeout=10).raise_for_status()
    rows = []
    for page in manifest["pages"]:
        image = pathlib.Path(page["canonical_path"])
        sample = image.parent.name
        golden_path = args.golden / (sample + ".json")
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        validate_gpu_reference_record(golden, golden["configuration"], require_equal, sample)
        assert sha256(image) == page["canonical_png_sha256"]
        assert golden["id"] == page["id"] and golden["canonical_rgb_sha256"] == page["rgb_sha256"]
        for key, value in [("precision", "fp32"), ("tf32", False), ("model_revision", REVISION),
                           ("weights_sha256", WEIGHT_SHA256), ("max_dimension", 1536), ("min_dimension", 64), ("max_new_tokens", 4096),
                           ("manifest_sha256", manifest["source_manifest_sha256"])]:
            assert golden["configuration"][key] == value, key
        request = {"model": "falcon-ocr", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
            {"type": "text", "text": "Extract the text content from this image.\n<|OCR_PLAIN|>"}]}],
            "temperature": 0, "seed": 0, "max_tokens": 4096, "stop_token_ids": [11, 263],
            "logprobs": True, "top_logprobs": 2, "return_tokens_as_token_ids": True}
        request_path = args.output / (sample + "-request.json")
        response_path = args.output / (sample + "-response.json")
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        response = requests.post(url + "/v1/chat/completions", json=request, timeout=900)
        response_path.write_text(response.text + "\n", encoding="utf-8")
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        observed, unparsed = [], []
        for item in (choice.get("logprobs") or {}).get("content", []):
            match = re.fullmatch(r"token_id:(\d+)", item["token"])
            if match:
                observed.append(int(match[1]))
            else:
                unparsed.append(item["token"])
        effective = observed.copy()
        appended = False
        if choice.get("stop_reason") in [11, 263] and (not effective or effective[-1] != choice["stop_reason"]):
            effective.append(choice["stop_reason"])
            appended = True
        first = next((i for i, (a, b) in enumerate(zip(effective, golden["token_ids"])) if a != b), None)
        if first is None and len(effective) != len(golden["token_ids"]):
            first = min(len(effective), len(golden["token_ids"]))
        finish = "eos" if choice["finish_reason"] == "stop" and choice.get("stop_reason") in [11, 263] else choice["finish_reason"]
        text = choice["message"]["content"]
        record = {"sample_id": sample, "id": page["id"], "category": page["category"],
                  "input_sha256": sha256(image), "golden_record_sha256": sha256(golden_path),
                  "request_sha256": sha256(request_path), "response_sha256": sha256(response_path),
                  "usage": result.get("usage"), "finish_reason": finish, "api_finish_reason": choice["finish_reason"],
                  "reported_stop_reason": choice.get("stop_reason"), "reported_stop_appended": appended,
                  "token_ids_observed_in_logprobs": observed, "effective_token_ids_with_reported_stop": effective,
                  "unparsed_logprob_tokens": unparsed, "text": text, "tokens_exact": not unparsed and effective == golden["token_ids"],
                  "text_exact": text.strip() == golden["text"], "outer_whitespace_exact": text == golden["text"],
                  "prefix_exact": result.get("usage", {}).get("prompt_tokens") == golden["prefix_length"],
                  "finish_reason_exact": finish == golden["finish_reason"], "first_token_divergence": first,
                  "elapsed_seconds_including_first_request_compile": time.perf_counter() - started}
        (args.output / (sample + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({k: v for k, v in record.items() if k not in ["text", "token_ids_observed_in_logprobs", "effective_token_ids_with_reported_stop"]})
        print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    report = {"schema_version": 1, "status": "complete", "environment": environment, "image_config_override": expected_image_config,
              "manifest_sha256": sha256(args.manifest), "pages": rows,
              "request_source_sha256": source_hashes, "source_capture_phase": "before first HTTP health check or inference request",
              "text_policy": "HF plain OCR removes known EOS strings and strips outer whitespace; compare the API text after the same outer-whitespace stripping, recording raw equality separately.",
              "fullpages_passed": all(all(p[k] for k in ["tokens_exact", "text_exact", "prefix_exact", "finish_reason_exact"]) for p in rows),
              "precision_audit_pending": True,
              "qualification": "Three actual direct-vLLM full-page HTTP requests. Serving precision still requires the post-request compiled PTX audit. This is not corpus-wide serving, tensor, quality, or performance qualification."}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["fullpages_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
