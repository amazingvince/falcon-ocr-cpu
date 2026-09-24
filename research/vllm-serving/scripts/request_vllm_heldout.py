#!/usr/bin/env python3
"""Greedy full-page requests against a running vLLM server, one page at a time.

  python3 research/vllm-serving/scripts/request_vllm_heldout.py --output DIR [--manifest LOCK] [--port 18081]

Writes DIR/<sample>.json per page (token IDs, top-K log-probabilities per step,
text, finish reason); pages already written are skipped, so a run resumes.
Finally writes DIR/tokens.json (bench-report style, for `agree --reference`)
and DIR/top.json (for `agree --reference-topk`) plus DIR/summary.json.
Standard library only.
"""
import argparse
import base64
import json
import pathlib
import re
import time
import urllib.request

STOP_IDS = [11, 263]
PROMPT = "Extract the text content from this image.\n<|OCR_PLAIN|>"


def post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def token_id(token):
    match = re.fullmatch(r"token_id:(\d+)", token)
    if not match:
        raise ValueError(f"unparsed logprob token {token!r}")
    return int(match[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/corpus-v3-evaluation-lock.json"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    urllib.request.urlopen(url + "/health", timeout=10).read()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(manifest["pages"]):
        image = pathlib.Path(page["canonical_path"])
        sample = image.parent.name
        target = args.output / f"{sample}.json"
        if target.exists():
            continue
        body = {"model": "falcon-ocr", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
            {"type": "text", "text": PROMPT}]}],
            "temperature": 0, "seed": 0, "max_tokens": args.max_tokens, "stop_token_ids": STOP_IDS,
            "logprobs": True, "top_logprobs": args.top_k, "return_tokens_as_token_ids": True}
        started = time.perf_counter()
        result = post(url + "/v1/chat/completions", body, timeout=1800)
        elapsed = time.perf_counter() - started
        choice = result["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        tokens = [token_id(item["token"]) for item in content]
        top = [[[token_id(t["token"]), t["logprob"]] for t in item["top_logprobs"]] for item in content]
        stop = choice.get("stop_reason")
        stop_appended = stop in STOP_IDS and (not tokens or tokens[-1] != stop)
        if stop_appended:
            tokens.append(stop)
        finish = "eos" if choice["finish_reason"] == "stop" and stop in STOP_IDS else choice["finish_reason"]
        record = {"sample_id": sample, "id": page["id"], "category": page["category"], "image": page["canonical_path"],
                  "finish_reason": finish, "api_finish_reason": choice["finish_reason"], "stop_reason": stop,
                  "stop_appended": stop_appended, "usage": result.get("usage"), "token_ids": tokens,
                  "top_logprobs": top, "text": choice["message"]["content"], "elapsed_seconds": elapsed}
        target.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{index + 1}/{len(manifest['pages'])}] {sample} {page['category']:<12} {finish:<6} "
              f"{len(tokens):5d} tok {elapsed:6.1f}s", flush=True)
    inputs, outputs, tops, rows = [], [], {}, []
    for page in manifest["pages"]:
        sample = pathlib.Path(page["canonical_path"]).parent.name
        record = json.loads((args.output / f"{sample}.json").read_text(encoding="utf-8"))
        inputs.append({"path": page["canonical_path"]})
        outputs.append({"token_ids": record["token_ids"], "finish_reason": record["finish_reason"], "text": record["text"]})
        tops[sample] = record["top_logprobs"]
        rows.append({k: record[k] for k in ["sample_id", "category", "finish_reason", "elapsed_seconds"]} | {"tokens": len(record["token_ids"])})
    (args.output / "tokens.json").write_text(json.dumps({"inputs": inputs, "samples": [{"outputs": outputs}]}), encoding="utf-8")
    (args.output / "top.json").write_text(json.dumps({"k": args.top_k, "pages": tops}), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"manifest": str(args.manifest), "pages": rows}, indent=1) + "\n", encoding="utf-8")
    print("done", len(rows), "pages")


if __name__ == "__main__":
    main()
