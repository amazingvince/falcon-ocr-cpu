#!/usr/bin/env python3
"""Pinned free-greedy GPU corpus evaluation, one resident model, no hidden traces."""
import argparse
import functools
import hashlib
import json
import pathlib
import time
import unicodedata

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file
from transformers import AutoTokenizer

from export_reference import import_model
from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from reference_preflight import preflight
from reference_run_identity import capture_reference_identity
from reference_corpus_contract import atomic_json, prepare_output, read_snapshot, sample_key, selected_pages, validate_completed_page

PROMPT = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/corpus-smoke-lock-v1.json"))
    p.add_argument("--output", type=pathlib.Path)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-dimension", type=int, default=1536)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.max_dimension < 64:
        raise ValueError("max-dimension must be at least the 64-pixel minimum")
    if args.output is None:
        args.output = pathlib.Path(f"artifacts/reference/corpus-smoke-{args.precision}-{args.max_new_tokens}")
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    manifest, manifest_sha256 = read_snapshot(args.manifest)
    pages = selected_pages(manifest, args.limit)
    identity, source_contents = capture_reference_identity(root, environment, args.precision)
    config_record = {"model_revision": REVISION, "weights_sha256": WEIGHT_SHA256,
                     "manifest_sha256": manifest_sha256, "max_new_tokens": args.max_new_tokens,
                     "max_dimension": args.max_dimension, "min_dimension": 64, "precision": args.precision,
                     "tf32": False, "flex_float32_precision": "ieee", "compiled_blocks": False,
                     "prompt": PROMPT, "config_sha256": identity["verified_model_assets"]["config.json"]["sha256"],
                     "cast_target": identity["numerical_policy"]["parameter_cast"],
                     "greedy_tie_rule": identity["numerical_policy"]["greedy_tie_rule"],
                     "allow_fp16_reduced_precision_reduction": False, "allow_bf16_reduced_precision_reduction": False,
                     "teacher_forced": False, "inference_reexecuted": True,
                     "selected_page_ids": [page["id"] for page in pages], "requested_limit": args.limit,
                     "runtime_identity_sha256": identity["identity_sha256"]}
    startup_identity = prepare_output(args.output, config_record, identity, source_contents, resume=args.resume)
    run_path = args.output / "run.json"
    atomic_json(run_path, {"configuration": config_record, "environment": environment, "startup_identity": startup_identity,
            "teacher_forced": False, "inference_reexecuted": True,
            "planned_pages": len(pages), "status": "running", "metric": "NFC+whitespace-normalized character edit distance against assembled ordered blocks; diagnostic only"})
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model_dir = root / "artifacts/model"
    module = import_model(model_dir)
    for name in ["compiled_flex_attn_prefill", "compiled_flex_attn_decode"]:
        setattr(module, name, functools.partial(getattr(module, name), kernel_options={"FLOAT32_PRECISION": "'ieee'"}))
    config = module.FalconOCRConfig.from_json_file(str(model_dir / "config.json"))
    config._name_or_path = str(model_dir)
    model = module.FalconOCRForCausalLM(config)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True, assign=True)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    model.to(device="cuda:0", dtype=dtype).eval()
    model._ensure_device_buffers()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model._pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    stop_ids = {config.eos_id, tokenizer.convert_tokens_to_ids("<|end_of_query|>")}
    prompt = PROMPT
    results = []
    with torch.inference_mode():
        for index, page in enumerate(pages):
            sample_id = sample_key(page)
            output_path = args.output / (sample_id + ".json")
            if sha256(page["canonical_path"]) != page["canonical_png_sha256"]:
                raise ValueError("Canonical image bytes changed: " + sample_id)
            if sha256(page["ground_truth_path"]) != page["ground_truth_sha256"]:
                raise ValueError("Ground-truth bytes changed: " + sample_id)
            if args.resume and output_path.exists():
                existing, _ = read_snapshot(output_path)
                validate_completed_page(existing, page, config_record)
                results.append(existing)
                print(f"[{index + 1}/{len(pages)}] Resumed {sample_id}", flush=True)
                continue
            if output_path.exists():
                raise ValueError("Refuse to overwrite an existing page: " + sample_id)
            with Image.open(page["canonical_path"]) as encoded:
                image = encoded.convert("RGB")
            actual_rgb_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            if actual_rgb_sha256 != page["rgb_sha256"]:
                raise ValueError("Decoded canonical RGB bytes changed: " + sample_id)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            batch = module.process_batch(tokenizer, config, [(image, prompt)], max_length=config.max_seq_len,
                                          min_dimension=64, max_dimension=args.max_dimension)
            prefix_length = batch["tokens"].shape[1]
            if prefix_length + args.max_new_tokens > config.max_seq_len:
                raise ValueError(f"Budget conflict on {page['id']}: {prefix_length}+{args.max_new_tokens}>{config.max_seq_len}")
            capacity = ((prefix_length + args.max_new_tokens + 127) // 128) * 128
            batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}
            padded = torch.full((1, capacity), model._pad_token_id, device="cuda:0", dtype=torch.long)
            padded[:, :prefix_length] = batch["tokens"]
            attention_mask = model.get_attention_mask(padded, max_len=capacity)
            cache = module.KVCache(1, capacity, config.n_heads, config.head_dim, config.n_layers)
            print(f"[{index + 1}/{len(pages)}] {sample_id} {page['category']} prefix={prefix_length} starting prefill", flush=True)
            torch.cuda.synchronize()
            prefill_start = time.perf_counter()
            logits = model(tokens=batch["tokens"], attention_mask=attention_mask, kv_cache=cache,
                           rope_pos_t=batch["pos_t"], rope_pos_hw=batch["pos_hw"],
                           pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
            torch.cuda.synchronize()
            prefill_elapsed = time.perf_counter() - prefill_start
            chosen, decisions = [], []
            decode_start = time.perf_counter()
            for step in range(args.max_new_tokens):
                vector = logits[0, -1]
                token = int(vector.argmax())
                values, ids = vector.topk(2)
                runner_up = int(ids[1]) if int(ids[0]) == token else int(ids[0])
                winner_value, runner_value = float(vector[token]), float(vector[runner_up])
                chosen.append(token)
                decisions.append({"step": step, "argmax": token, "runner_up": runner_up,
                                  "winner_logit": winner_value, "runner_up_logit": runner_value,
                                  "winner_margin": winner_value - runner_value})
                if token in stop_ids or step + 1 == args.max_new_tokens:
                    break
                padded[:, cache.get_pos()] = token
                logits = model(tokens=torch.tensor([[token]], device="cuda:0"), attention_mask=attention_mask, kv_cache=cache)
                if (step + 1) % 256 == 0:
                    print(f"[{index + 1}/{len(pages)}] {sample_id} decoded {step + 1} tokens", flush=True)
            torch.cuda.synchronize()
            decode_elapsed = time.perf_counter() - decode_start
            text = tokenizer.decode(chosen, skip_special_tokens=False).replace("<|end_of_query|>", "").replace("<|end_of_text|>", "").strip()
            truth = pathlib.Path(page["ground_truth_path"]).read_text(encoding="utf-8")
            norm_truth, norm_text = normalized_text(truth), normalized_text(text)
            distance = Levenshtein.distance(norm_truth, norm_text)
            record = {"id": page["id"], "sample_id": sample_id, "category": page["category"], "configuration": config_record,
                      "teacher_forced": False, "inference_reexecuted": True,
                      "canonical_rgb_sha256": actual_rgb_sha256, "prefix_length": prefix_length, "cache_capacity": capacity,
                      "final_kv_cache_position": int(cache.get_pos()),
                      "token_ids": chosen, "text": text, "finish_reason": "eos" if chosen[-1] in stop_ids else "length",
                      "diagnostic_character_edit_distance": distance, "diagnostic_ground_truth_characters": len(norm_truth),
                      "diagnostic_cer": distance / max(1, len(norm_truth)), "logit_decisions": decisions,
                      "prefill_seconds_including_compile": prefill_elapsed, "decode_seconds_including_python_diagnostics": decode_elapsed,
                      "elapsed_seconds": time.perf_counter() - started, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "qualification": "Free greedy reference at the recorded precision, image limit and output cap; no Rust comparison yet, not an official accuracy or performance score"}
            validate_completed_page(record, page, config_record)
            atomic_json(output_path, record)
            results.append(record)
            print(json.dumps({"page": sample_id, "tokens": len(chosen), "finish_reason": record["finish_reason"], "diagnostic_cer": record["diagnostic_cer"], "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            del batch, padded, cache, logits, attention_mask
            torch.cuda.empty_cache()
    summary = {"configuration": config_record, "environment": environment, "startup_identity": startup_identity,
               "teacher_forced": False, "inference_reexecuted": True,
               "status": "complete", "pages": len(results), "planned_pages": len(pages),
               "generated_tokens": sum(len(r["token_ids"]) for r in results), "length_limited_pages": [r["sample_id"] for r in results if r["finish_reason"] == "length"],
               "eos_pages": sum(r["finish_reason"] == "eos" for r in results),
               "maximum_natural_output_tokens": max((len(r["token_ids"]) for r in results if r["finish_reason"] == "eos"), default=None),
               "diagnostic_micro_cer": sum(r["diagnostic_character_edit_distance"] for r in results) / max(1, sum(r["diagnostic_ground_truth_characters"] for r in results)),
               "qualification": "Only the explicitly selected pages and recorded output cap are complete. Length stops remain truncated. This GPU reference alone does not establish CPU parity, official document accuracy, or isolated performance."}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(run_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
