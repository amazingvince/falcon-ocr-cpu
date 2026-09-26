#!/usr/bin/env python3
"""Greedy full-page transcripts of a page manifest from a running vLLM server.

The draft-head training targets: the production model's own greedy tokens
(official Falcon-OCR image, `research/draft-head/data/serve_vllm.sh`). Requests run
concurrently so vLLM batches them; pages already written are skipped, so a
run resumes. One JSON per page: token ids, top-1 log-probability per step,
text, finish reason.

  python3 research/vllm-serving/scripts/request_vllm_draftgen.py --manifest D:/falcon-draft/pages/stage1/manifest.jsonl \
      --output D:/falcon-draft/outputs/stage1 [--port 18081] [--workers 64]

Standard library only.
"""
import argparse
import base64
import concurrent.futures
import json
import pathlib
import re
import threading
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
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reverse", action="store_true",
                        help="work from the end of the manifest (a second server shares the run)")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    urllib.request.urlopen(url + "/health", timeout=10).read()
    pages = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pages if not (args.output / (p["id"].replace("/", "__") + ".json")).exists()]
    if args.reverse:
        todo.reverse()
    print(f"{len(pages)} pages, {len(todo)} to do", flush=True)
    lock = threading.Lock()
    stats = {"done": 0, "tokens": 0, "failed": 0}
    started = time.perf_counter()

    def run(page):
        if (args.output / (page["id"].replace("/", "__") + ".json")).exists():
            return  # written meanwhile by another client
        image = pathlib.Path(page["path"])
        mime = "png" if image.suffix.lower() == ".png" else "jpeg"
        body = {"model": "falcon-ocr", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64," + base64.b64encode(image.read_bytes()).decode()}},
            {"type": "text", "text": PROMPT}]}],
            "temperature": 0, "seed": 0, "max_tokens": args.max_tokens, "stop_token_ids": STOP_IDS,
            "logprobs": True, "top_logprobs": 1, "return_tokens_as_token_ids": True}
        t0 = time.perf_counter()
        try:
            response = post(url + "/v1/chat/completions", body, timeout=1800)
        except Exception as e:  # noqa: BLE001 - keep going; the page is retried on the next run
            with lock:
                stats["failed"] += 1
            print(f"{page['id']}: {type(e).__name__} {e}", flush=True)
            return
        choice = response["choices"][0]
        steps = choice["logprobs"]["content"]
        record = {"id": page["id"], "source": page["source"], "category": page["category"],
                  "token_ids": [token_id(s["token"]) for s in steps],
                  "top1_logprob": [s["logprob"] for s in steps],
                  "text": choice["message"]["content"], "finish_reason": choice["finish_reason"],
                  "seconds": time.perf_counter() - t0}
        target = args.output / (page["id"].replace("/", "__") + ".json")
        target.write_text(json.dumps(record), encoding="utf-8")
        with lock:
            stats["done"] += 1
            stats["tokens"] += len(record["token_ids"])
            if stats["done"] % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f"{stats['done']}/{len(todo)} pages, {stats['tokens'] / elapsed:.0f} tokens/s, "
                      f"{stats['done'] / elapsed:.2f} pages/s", flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, todo))
    print(f"finished: {stats}", flush=True)


if __name__ == "__main__":
    main()
