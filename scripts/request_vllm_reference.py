#!/usr/bin/env python3
"""Make and preserve an actual direct-vLLM multimodal serving smoke request."""
import base64
import json
import pathlib
import re
import time

import requests

from fetch_reference import sha256


def main():
    output = pathlib.Path("artifacts/reference/vllm-smoke-fp32")
    environment = json.loads((output / "environment.json").read_text())
    golden_path = pathlib.Path("artifacts/reference/smoke-fp32/metadata.json")
    golden = json.loads(golden_path.read_text())
    image = pathlib.Path("artifacts/reference/smoke-fp32/input.png")
    assert sha256(image) == golden["image_sha256"]
    assert environment["model_revision"] == golden["model_revision"]
    assert environment["weights_sha256"] == golden["weights_sha256"]
    url = "http://127.0.0.1:18080"
    requests.get(url + "/health", timeout=10).raise_for_status()
    request = {"model": "falcon-ocr", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
        {"type": "text", "text": "Extract the text content from this image.\n<|OCR_PLAIN|>"}]}],
        "temperature": 0, "seed": 0, "max_tokens": 24, "stop_token_ids": [11, 263],
        "logprobs": True, "top_logprobs": 2, "return_tokens_as_token_ids": True}
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    started = time.perf_counter()
    response = requests.post(url + "/v1/chat/completions", json=request, timeout=180)
    (output / "response.json").write_text(response.text + "\n")
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    observed = []
    unparsed = []
    for token in (choice.get("logprobs") or {}).get("content", []):
        match = re.fullmatch(r"token_id:(\d+)", token["token"])
        if match:
            observed.append(int(match[1]))
        else:
            unparsed.append(token["token"])
    effective = observed.copy()
    appended_reported_stop = False
    if choice.get("stop_reason") in [11, 263] and (not effective or effective[-1] != choice["stop_reason"]):
        effective.append(choice["stop_reason"])
        appended_reported_stop = True
    expected_text = golden["text"].replace("<|end_of_query|>", "").replace("<|end_of_text|>", "")
    report = {"schema_version": 1, "environment": environment, "image_sha256": sha256(image),
              "golden_metadata_sha256": sha256(golden_path), "request_sha256": sha256(output / "request.json"),
              "response_sha256": sha256(output / "response.json"), "endpoint": url + "/v1/chat/completions",
              "elapsed_seconds_including_first_request_compile": time.perf_counter() - started,
              "usage": result.get("usage"), "finish_reason": choice["finish_reason"], "reported_stop_reason": choice.get("stop_reason"),
              "token_ids_observed_in_logprobs": observed, "unparsed_logprob_tokens": unparsed,
              "reported_stop_appended": appended_reported_stop, "effective_token_ids_with_reported_stop": effective,
              "expected_token_ids": golden["token_ids"], "tokens_exact": not unparsed and effective == golden["token_ids"],
              "text": choice["message"]["content"], "expected_text": expected_text,
              "text_exact": choice["message"]["content"] == expected_text,
              "prompt_tokens_exact": result.get("usage", {}).get("prompt_tokens") == golden["prefix_length"],
              "qualification": "One actual pinned direct-vLLM HTTP smoke. Tensor parity, full corpus serving parity, and standalone performance are separate gates. Reported stop IDs are distinguished from IDs included in logprobs."}
    report["smoke_passed"] = report["tokens_exact"] and report["text_exact"] and report["prompt_tokens_exact"]
    pathlib.Path("reference/vllm-smoke-fp32.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
