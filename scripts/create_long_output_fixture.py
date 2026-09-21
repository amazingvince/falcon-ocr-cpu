#!/usr/bin/env python3
"""Create an original deterministic numerical document for long-output testing.

This is a stress fixture, not a document-quality corpus or a GPU reference.
Actual model outputs and stopping behavior must be measured separately.
"""
import argparse
import hashlib
import json
import pathlib
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/corpus/long-numeric-v1"))
    parser.add_argument("--font", type=pathlib.Path, default=pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
    parser.add_argument("--tokenizer", type=pathlib.Path, default=pathlib.Path("artifacts/model/tokenizer.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = 6242026
    lines = []
    for _ in range(80):
        groups = []
        for _ in range(20):
            digits = []
            for _ in range(5):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                digits.append(str((state >> 24) % 10))
            groups.append("".join(digits))
        lines.append(" ".join(groups))
    text = "\n".join(lines)
    image = Image.new("RGB", (1376, 1504), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(args.font), 14)
    for index, line in enumerate(lines):
        box = draw.textbbox((24, 16 + index * 18), line, font=font)
        assert 0 <= box[0] < box[2] < image.width and 0 <= box[1] < box[3] < image.height
        draw.text((24, 16 + index * 18), line, font=font, fill="black")
    image_path = args.output / "canonical-rgb.png"
    text_path = args.output / "ground-truth.txt"
    image.save(image_path)
    text_path.write_text(text + "\n", encoding="utf-8")
    token_ids = Tokenizer.from_file(str(args.tokenizer)).encode(text, add_special_tokens=False).ids
    metadata = {
        "schema_version": 1,
        "purpose": "Original numerical stress document; not natural OCR quality evidence",
        "generator_sha256": sha(pathlib.Path(__file__)),
        "font_path": str(args.font), "font_sha256": sha(args.font), "font_size": 14,
        "canonical_png_sha256": sha(image_path), "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "ground_truth_sha256": sha(text_path), "tokenizer_sha256": sha(args.tokenizer),
        "width": image.width, "height": image.height, "rows": 80, "digit_count": 8000,
        "plain_ground_truth_tokens": len(token_ids),
        "image_patches": (image.width // 16) * (image.height // 16),
        "requested_max_new_tokens": 8192, "requested_max_dimension": 1536,
        "actual_gpu_output": None,
        "note": "Ground-truth token count is not the generated-token count. Measure EOS and truncation explicitly.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
