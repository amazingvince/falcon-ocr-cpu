#!/usr/bin/env python3
"""CPU-only comparison of pinned serving helper source and pinned HF preprocessing.

This does not import/start vLLM or claim that its actual worker used a setting.
The supported full-page image_config override still needs an actual HTTP run.
"""
import ast
import hashlib
import importlib.metadata
import json
import math
import pathlib
import typing

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import get_image_size, infer_channel_dimension_format, to_numpy_array

from export_reference import import_model
from fetch_reference import sha256


def main():
    torch.set_num_threads(1)
    path = pathlib.Path("artifacts/reference/vllm-smoke-fp32/sources/model_executor/models/falcon_ocr_multimodal.py")
    source = ast.parse(path.read_text(encoding="utf-8"))
    names = {"resize_image_if_necessary", "smart_resize", "preprocess_image"}
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {"np": np, "torch": torch, "math": math, "Any": typing.Any, "PILImage": Image,
                 "_HF_AVAILABLE": True, "_IMAGE_MEAN": [0.5] * 3, "_IMAGE_STD": [0.5] * 3,
                 "convert_to_rgb": convert_to_rgb, "to_numpy_array": to_numpy_array,
                 "infer_channel_dimension_format": infer_channel_dimension_format, "get_image_size": get_image_size}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    module = import_model(pathlib.Path("artifacts/model").resolve())
    config = module.FalconOCRConfig.from_json_file("artifacts/model/config.json")
    tokenizer = AutoTokenizer.from_pretrained("artifacts/model", trust_remote_code=True, local_files_only=True)
    manifest_path = pathlib.Path("reference/serving-fullpages-v1-lock.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for page in manifest["pages"]:
        assert sha256(page["canonical_path"]) == page["canonical_png_sha256"]
        image = Image.open(page["canonical_path"]).convert("RGB")
        batch = module.process_batch(tokenizer, config, [(image, "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>")],
                                     max_length=config.max_seq_len, min_dimension=64, max_dimension=1536)
        expected = batch["pixel_values"][batch["pixel_mask"].bool()]
        configured, h, w = namespace["preprocess_image"](image, spatial_patch_size=16, min_image_size=64,
                                                         max_image_size=1536, min_pixels=3136, max_pixels=10035200, merge_size=1)
        default, dh, dw = namespace["preprocess_image"](image, spatial_patch_size=16)
        candidate = configured.reshape(-1, 3)
        assert candidate.shape == expected.shape
        diff = candidate != expected
        rows.append({"sample_id": pathlib.Path(page["canonical_path"]).parent.name, "category": page["category"],
                     "input_sha256": sha256(page["canonical_path"]), "hf_prefix_tokens": int(batch["tokens"].numel()),
                     "configured_image_hw": [h, w], "default_image_hw": [dh, dw],
                     "configured_pixels_exact": torch.equal(candidate, expected), "different_elements": int(diff.sum()),
                     "max_abs": float((candidate - expected).abs().max()),
                     "configured_pixels_sha256": hashlib.sha256(configured.numpy().tobytes()).hexdigest(),
                     "hf_unpadded_pixels_sha256": hashlib.sha256(expected.numpy().tobytes()).hexdigest()})
    report = {"schema_version": 1, "manifest_sha256": sha256(manifest_path), "serving_helper_source_sha256": sha256(path),
              "hf_processor_sha256": sha256("artifacts/model/processing_falcon_ocr.py"), "script_sha256": sha256(__file__),
              "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "pillow", "numpy"]},
              "gpu_execution": False, "pages": rows, "all_configured_pixels_exact": all(r["configured_pixels_exact"] for r in rows),
              "qualification": "Pinned source helper evaluated under the HF reference CPU environment. Actual vLLM worker processing, its dependency versions and effective overrides still require serving runtime evidence."}
    pathlib.Path("reference/vllm-fullpages-preprocessing-source-check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
