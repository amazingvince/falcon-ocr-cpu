#!/usr/bin/env python3
"""Independent official paged full-page engine smoke comparison."""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import torch
from PIL import Image

from fetch_reference import OFFICIAL_REVISION, REVISION, WEIGHT_SHA256, sha256
from reference_preflight import preflight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=pathlib.Path, default=pathlib.Path("artifacts/reference/smoke-fp32/input.png"))
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference/official-smoke-fp32.json"))
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    repo = root / "artifacts/Falcon-Perception"
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert actual == OFFICIAL_REVISION, actual
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    assert not dirty, "Official tracked code has been modified"
    sys.path.insert(0, str(repo))
    from falcon_perception import load_and_prepare_model
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_ocr_inference import OCRInferenceEngine
    from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, tokenizer, config = load_and_prepare_model(hf_local_dir=str(root / "artifacts/model"),
                                                      hf_revision=REVISION, device="cuda:0", dtype="float32", compile=False)
    kernel_options = {"FLOAT32_PRECISION": "'ieee'", "BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
    engine = OCRInferenceEngine(model, tokenizer, ImageProcessor(patch_size=16, merge_size=1),
                                max_batch_size=2, max_seq_length=256, n_pages=4, page_size=128,
                                prefill_length_limit=256, kernel_options=kernel_options,
                                seed=42, capture_cudagraph=False)
    prompt = engine._make_ocr_prompt("plain")
    sequence = Sequence(text=prompt, image=Image.open(args.image).convert("RGB"),
                        min_image_size=64, max_image_size=256, request_idx=0, task="ocr")
    with torch.inference_mode():
        started = time.perf_counter()
        done = PagedInferenceEngine.generate(engine, [sequence],
                  sampling_params=SamplingParams(max_new_tokens=args.max_new_tokens, stop_token_ids=engine._stop_token_ids()),
                  temperature=0.0, top_k=None, use_tqdm=False, print_stats=False)
        elapsed = time.perf_counter() - started
    tokens = [int(token) for token in done[0].output_ids]
    baseline = json.loads((root / "artifacts/reference/smoke-fp32/metadata.json").read_text())
    report = {"schema_version": 1, "official_revision": actual, "model_revision": REVISION,
              "weights_sha256": WEIGHT_SHA256, "environment": environment, "precision": "fp32", "tf32": False,
              "kernel_options": kernel_options, "compiled_blocks": False, "cuda_graphs": False,
              "max_new_tokens": args.max_new_tokens, "max_image_size": 256, "min_image_size": 64,
              "image_sha256": sha256(args.image), "token_ids": tokens, "text": engine._decode_seq_text(done[0]),
              "hf_token_ids": baseline["token_ids"], "exact_token_match": tokens == baseline["token_ids"],
              "elapsed_seconds_including_compilation": elapsed,
              "limitation": "One tiny full-page fixture, not corpus validation or a performance benchmark"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["exact_token_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
