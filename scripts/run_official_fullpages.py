#!/usr/bin/env python3
"""Run three full pages through the pinned official plain OCR engine."""
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
from validate_gpu_reference_record import validate_gpu_reference_record


def require_equal(label, actual, expected):
    if actual != expected:
        raise ValueError(f"Reference prerequisite failed: {label}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/serving-fullpages-v1-lock.json"))
    parser.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("artifacts/reference/corpus-v3-fp32-4096"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/reference/official-fullpages-fp32-4096"))
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("reference/official-fullpages-fp32.json"))
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError("Official full-page output already exists; preserve previous attempts")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    goldens = {}
    for page in manifest["pages"]:
        sample = pathlib.Path(page["canonical_path"]).parent.name
        golden = json.loads((args.golden / (sample + ".json")).read_text(encoding="utf-8"))
        validate_gpu_reference_record(golden, golden["configuration"], require_equal, sample)
        assert golden["id"] == page["id"] and golden["canonical_rgb_sha256"] == page["rgb_sha256"]
        assert golden["configuration"]["manifest_sha256"] == manifest["source_manifest_sha256"]
        for key, value in [("precision", "fp32"), ("tf32", False), ("model_revision", REVISION),
                           ("weights_sha256", WEIGHT_SHA256), ("max_dimension", 1536), ("min_dimension", 64), ("max_new_tokens", 4096)]:
            assert golden["configuration"][key] == value, key
        assert sha256(page["canonical_path"]) == page["canonical_png_sha256"]
        goldens[sample] = golden
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = preflight(root)
    repo = root / "artifacts/Falcon-Perception"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert revision == OFFICIAL_REVISION
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    args.output.mkdir(parents=True)
    source_archive = args.output / "sources"
    source_archive.mkdir()
    source_hashes = {}
    for name in ["run_official_fullpages.py", "reference_preflight.py", "fetch_reference.py", "validate_gpu_reference_record.py"]:
        target = source_archive / name
        target.write_bytes((root / "scripts" / name).read_bytes())
        source_hashes[name] = sha256(target)
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
    model, tokenizer, _ = load_and_prepare_model(hf_local_dir=str(root / "artifacts/model"),
                                                 hf_revision=REVISION, device="cuda:0", dtype="float32", compile=False)
    options = {"FLOAT32_PRECISION": "'ieee'", "BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
    engine = OCRInferenceEngine(model, tokenizer, ImageProcessor(patch_size=16, merge_size=1),
                                max_batch_size=2, max_seq_length=16384, n_pages=128, page_size=128,
                                prefill_length_limit=16384, kernel_options=options,
                                seed=42, capture_cudagraph=False)
    config = {"model_revision": REVISION, "weights_sha256": WEIGHT_SHA256, "official_revision": revision,
              "precision": "fp32", "tf32": False, "kernel_options": options, "cuda_graphs": False,
              "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096,
              "max_seq_length": 16384, "max_batch_size": 2, "actual_requests_per_generate": 1,
              "manifest_sha256": sha256(args.manifest), "script_sha256": source_hashes["run_official_fullpages.py"],
              "source_archive_sha256": source_hashes,
              "source_capture_phase": "after input/model/device validation, before official model initialization"}
    (args.output / "run.json").write_text(json.dumps({"status": "running", "configuration": config,
                                                    "environment": environment}, indent=2) + "\n", encoding="utf-8")
    reports = []
    for page in manifest["pages"]:
        sample = pathlib.Path(page["canonical_path"]).parent.name
        golden = goldens[sample]
        sequence = Sequence(text=engine._make_ocr_prompt("plain"), image=Image.open(page["canonical_path"]).convert("RGB"),
                            min_image_size=64, max_image_size=1536, request_idx=0, task="ocr")
        with torch.inference_mode():
            started = time.perf_counter()
            done = PagedInferenceEngine.generate(engine, [sequence], sampling_params=SamplingParams(
                max_new_tokens=4096, stop_token_ids=engine._stop_token_ids()), temperature=0.0,
                top_k=None, use_tqdm=False, print_stats=False)
            torch.cuda.synchronize()
        assert len(done) == 1
        tokens = [int(t) for t in done[0].output_ids]
        text = engine._decode_seq_text(done[0])
        prefix = int(done[0].input_ids.numel())
        finish = "eos" if tokens[-1] in engine._stop_token_ids() else "length"
        mismatch = next((i for i, (a, b) in enumerate(zip(tokens, golden["token_ids"])) if a != b), None)
        if mismatch is None and len(tokens) != len(golden["token_ids"]):
            mismatch = min(len(tokens), len(golden["token_ids"]))
        record = {"id": page["id"], "sample_id": sample, "category": page["category"], "configuration": config,
                  "input_sha256": sha256(page["canonical_path"]), "golden_record_sha256": sha256(args.golden / (sample + ".json")),
                  "token_ids": tokens, "text": text, "prefix_length": prefix, "finish_reason": finish,
                  "tokens_exact": tokens == golden["token_ids"], "text_exact": text == golden["text"],
                  "prefix_exact": prefix == golden["prefix_length"], "finish_reason_exact": finish == golden["finish_reason"],
                  "first_token_divergence": mismatch, "elapsed_seconds_including_compile": time.perf_counter() - started}
        (args.output / (sample + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports.append({k: v for k, v in record.items() if k not in ["token_ids", "text", "configuration"]})
        print(json.dumps(reports[-1], ensure_ascii=True), flush=True)
    report = {"schema_version": 1, "configuration": config, "environment": environment, "pages": reports,
              "status": "complete", "fullpages_passed": all(all(p[k] for k in ["tokens_exact", "text_exact", "prefix_exact", "finish_reason_exact"]) for p in reports),
              "qualification": "Three actual single-request official plain-engine full pages. No pipeline, corpus-wide serving acceptance, tensor parity, or isolated performance claim."}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["fullpages_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
