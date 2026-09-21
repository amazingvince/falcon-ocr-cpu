#!/usr/bin/env python3
"""Typeset original OCR diagnostics, separate from the frozen natural corpus.

Requires the pinned Linux Pillow/shaping runtime and DejaVu fonts described in
the manifest. No downloaded document images, model outputs, or external text.
Run --verify to rerender in memory and compare every frozen artifact.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import unicodedata

import PIL
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/supplement/original-v1")
LOCK = Path("reference/original-supplement-v1-lock.json")
VERSIONS = {"pillow": "12.3.0", "freetype2": "2.14.3", "raqm": "0.10.5", "harfbuzz": "14.2.1", "fribidi": "1.0.13"}
FONTS = {
    "sans": {"path": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "sha256": "ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280"},
    "mono": {"path": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "sha256": "c805f9436dbc268644c1d9584f01a601a653e028e08fd74b9b949f6cf8304d88"},
}
BITSTREAM_NOTICE = """Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved.
Bitstream Vera is a trademark of Bitstream, Inc.
DejaVu changes are in public domain.

Permission is hereby granted, free of charge, to any person obtaining a copy of
the fonts accompanying this license ("Fonts") and associated documentation files
(the "Font Software"), to reproduce and distribute the Font Software, including
without limitation the rights to use, copy, merge, publish, distribute, and/or
sell copies of the Font Software, and to permit persons to whom the Font Software
is furnished to do so, subject to the following conditions:

The above copyright and trademark notices and this permission notice shall be
included in all copies of one or more of the Font Software typefaces.

The Font Software may be modified, altered, or added to, and in particular the
designs of glyphs or characters in the Fonts may be modified and additional
glyphs or characters may be added to the Fonts, only if the fonts are renamed to
names not containing either the words "Bitstream" or the word "Vera".

This License becomes null and void to the extent applicable to Fonts or Font
Software that has been modified and is distributed under the "Bitstream Vera"
names.

The Font Software may be sold as part of a larger software package but no copy
of one or more of the Font Software typefaces may be sold by itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT,
TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL BITSTREAM OR THE GNOME FOUNDATION
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL,
SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO
USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.

Except as contained in this notice, the names of Gnome, the Gnome Foundation,
and Bitstream Inc., shall not be used in advertising or otherwise to promote
the sale, use or other dealings in this Font Software without prior written
authorization from the Gnome Foundation or Bitstream Inc., respectively.
For further information, contact: fonts at gnome dot org.
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def font(name, size):
    return ImageFont.truetype(FONTS[name]["path"], size, layout_engine=ImageFont.Layout.RAQM)


def typeset(lines, *, size, face="sans", font_size=30, line_step=72,
            margin=64, top=70, foreground=24, background=255,
            direction="ltr", language="en"):
    image = Image.new("RGB", size, (background,) * 3)
    draw = ImageDraw.Draw(image)
    chosen = font(face, font_size)
    missing = chosen.getmask("\U0010ffff")
    missing_signature = (missing.size, bytes(missing))
    for character in set("".join(lines)):
        if character.isspace():
            continue
        mask = chosen.getmask(character)
        assert (mask.size, bytes(mask)) != missing_signature, f"Missing glyph: {character!r}"
    boxes = []
    for index, line in enumerate(lines):
        if not line:
            continue
        xy = (size[0] - margin if direction == "rtl" else margin, top + index * line_step)
        options = {"font": chosen, "anchor": "rt" if direction == "rtl" else "lt", "direction": direction, "language": language}
        bbox = draw.textbbox(xy, line, **options)
        assert 0 <= bbox[0] < bbox[2] <= size[0], (line, bbox, size)
        assert 0 <= bbox[1] < bbox[3] <= size[1], (line, bbox, size)
        draw.text(xy, line, fill=(foreground,) * 3, **options)
        boxes.append({"line_index": index, "text": line, "bbox_xyxy": list(bbox)})
    recipe = {"method": "Pillow text, RAQM layout, one logical string per line", "font": face,
              "font_size_px": font_size, "line_step_px": line_step, "margin_px": margin,
              "top_px": top, "foreground_rgb": [foreground] * 3, "background_rgb": [background] * 3,
              "direction": direction, "language": language, "line_boxes_upright": boxes}
    return image, recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    actual = {"pillow": PIL.__version__, **{key: features.version(key) for key in VERSIONS if key != "pillow"}}
    assert actual == VERSIONS, (actual, VERSIONS)
    for entry in FONTS.values():
        assert digest(Path(entry["path"]).read_bytes()) == entry["sha256"]
    if not args.verify and (ROOT / LOCK).exists():
        raise FileExistsError(f"Refusing to overwrite {LOCK}; use --verify")
    pages = []

    def artifact(path, data):
        target = ROOT / path
        if args.verify:
            assert target.read_bytes() == data, path
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def add(name, category, parent, image, lines, recipe, language="english", rotation=0):
        text = unicodedata.normalize("NFC", "\n".join(lines)).strip()
        if category == "blank":
            assert not text and all(low == high for low, high in image.getextrema())
        else:
            assert text and any(low != high for low, high in image.getextrema())
        png = io.BytesIO()
        image.save(png, format="PNG", optimize=False, compress_level=9)
        image_path = OUTPUT / name / "canonical-rgb.png"
        text_path = OUTPUT / name / "ground-truth.txt"
        text_bytes = text.encode("utf-8")
        artifact(image_path, png.getvalue())
        artifact(text_path, text_bytes)
        pages.append({"id": "original-supplement-v1:" + name, "split": "supplemental_diagnostic",
                      "category": category, "source_family": "original-supplement-v1:" + parent,
                      "parent_id": "original-supplement-v1:" + parent,
                      "source_path": image_path.as_posix(), "source_sha256": digest(png.getvalue()),
                      "canonical_path": image_path.as_posix(), "canonical_png_sha256": digest(png.getvalue()),
                      "rgb_sha256": digest(image.tobytes()), "width": image.width, "height": image.height,
                      "ground_truth_path": text_path.as_posix(), "ground_truth_sha256": digest(text_bytes),
                      "expected_text": text, "expected_empty_output": not bool(text),
                      "attributes": {"language": language, "data_source": "original_typesetting", "rotation_counterclockwise_degrees": rotation},
                      "recipe": recipe, "visual_review": "pending_in_separate_review"})

    market = ["PARKSIDE MARKET", "ORIGINAL TEST RECEIPT", "20 SEP 2026  10:45", "",
              "ITEM             QTY    AMOUNT", "APPLES            2       3.00", "BREAD             1       2.40",
              "TEA               1       4.60", "", "SUBTOTAL                 10.00", "TAX                       0.80",
              "TOTAL                    10.80", "CASH                     20.00", "CHANGE                    9.20", "", "THANK YOU"]
    for name, foreground, background in [("receipt-market-clean", 24, 255), ("receipt-market-faint", 164, 249)]:
        image, recipe = typeset(market, size=(640, 1024), face="mono", font_size=26, line_step=51, margin=64, foreground=foreground, background=background)
        add(name, "receipts", "receipt-market", image, market, recipe)
    cafe = ["RIVER CAFE", "ORIGINAL TEST RECEIPT", "ORDER 018", "", "COFFEE x2          USD  6.00", "SANDWICH           USD  7.50", "",
            "SUBTOTAL           USD 13.50", "TAX                USD  1.08", "TOTAL              USD 14.58", "", "PAID BY CARD", "PLEASE KEEP THIS COPY"]
    image, recipe = typeset(cafe, size=(640, 960), face="mono", font_size=26, line_step=55, margin=64)
    add("receipt-cafe", "receipts", "receipt-cafe", image, cafe, recipe)

    for name, size, shade in [("blank-white", (1024, 1280), 255), ("blank-offwhite", (1280, 1024), 247)]:
        add(name, "blank", name, Image.new("RGB", size, (shade,) * 3), [], {"method": "Uniform RGB fill", "rgb": [shade] * 3})
    sparse = ["Room 204"]
    image, recipe = typeset(sparse, size=(1024, 768), font_size=34, margin=400, top=350)
    add("sparse-room", "sparse", "sparse-room", image, sparse, recipe)

    rotation_lines = ["A note from the neighborhood library", "", "The reading room opens at nine each morning.",
                      "Please return each book to its shelf after reading.", "Quiet conversation is welcome near the entrance.",
                      "The garden closes at six in the evening.", "", "Thank you for keeping this shared space tidy."]
    upright, recipe = typeset(rotation_lines, size=(1280, 960), font_size=36, line_step=82)
    for degrees, transpose in [(0, None), (90, Image.Transpose.ROTATE_90), (180, Image.Transpose.ROTATE_180), (270, Image.Transpose.ROTATE_270)]:
        image = upright.copy() if transpose is None else upright.transpose(transpose)
        rotated_recipe = dict(recipe, rotation_method="Exact orthogonal pixel transpose; no interpolation", rotation_counterclockwise_degrees=degrees)
        add(f"rotation-{degrees:03d}", "rotation", "rotation-library", image, rotation_lines, rotated_recipe, rotation=degrees)

    scripts = [
        ("russian", "ru", "ltr", ["Библиотека у парка", "Сегодня читальный зал открыт до шести часов.", "Пожалуйста, верните книгу на полку после чтения.", "Спасибо за внимательное отношение к книгам."]),
        ("greek", "el", "ltr", ["Η βιβλιοθήκη της γειτονιάς", "Η αίθουσα ανάγνωσης ανοίγει κάθε πρωί.", "Παρακαλούμε να επιστρέφετε τα βιβλία στο ράφι.", "Ευχαριστούμε για την επίσκεψή σας."]),
        ("arabic", "ar", "rtl", ["مكتبة الحي", "تفتح المكتبة أبوابها في الساعة التاسعة صباحا.", "يمكن للزوار قراءة الكتب في قاعة هادئة.", "يرجى إعادة الكتاب إلى الرف بعد القراءة.", "شكرا لزيارتكم."]),
        ("hebrew", "he", "rtl", ["ספריית השכונה", "הספרייה פתוחה בכל בוקר.", "אפשר לקרוא ספרים בחדר השקט.", "יש להחזיר את הספר למדף לאחר הקריאה.", "תודה על הביקור."]),
        ("vietnamese", "vi", "ltr", ["Thư viện khu phố", "Phòng đọc mở cửa vào mỗi buổi sáng.", "Vui lòng trả sách về kệ sau khi đọc.", "Cảm ơn bạn đã giữ gìn không gian chung."]),
    ]
    for name, code, direction, lines in scripts:
        image, recipe = typeset(lines, size=(1440, 960), font_size=36, line_step=100, direction=direction, language=code)
        add("script-" + name, "multilingual", "script-" + name, image, lines, recipe, language=name)

    assert len(pages) == 15 and len({p["id"] for p in pages}) == 15
    assert len({p["source_family"] for p in pages}) == 11
    artifact(OUTPUT / "FONT-LICENSE.txt", BITSTREAM_NOTICE.encode("utf-8"))
    manifest = {"schema_version": 1, "dataset": "original-supplement-v1", "page_count": len(pages),
                "parent_count": 11, "category_counts": dict(Counter(p["category"] for p in pages)),
                "provenance": "Original short text and deterministic page recipes authored for this project; no external document images or passages, no model output used to select content.",
                "source_terms": "Original fixture text and recipes may be used and redistributed with this project. DejaVu font license notice retained; font binaries are not copied into the artifact set.",
                "font_license": {"name": "Bitstream Vera license; DejaVu changes public domain", "path": (OUTPUT / "FONT-LICENSE.txt").as_posix(), "sha256": digest(BITSTREAM_NOTICE.encode("utf-8"))},
                "runtime": actual, "fonts": FONTS, "generator_sha256": digest(Path(__file__).read_bytes()),
                "rgb_policy": "Typeset directly into RGB; no EXIF; orthogonal variants use exact pixel transposes.",
                "ground_truth_policy": "Exact authored logical Unicode text, NFC, LF between lines, trimmed outer whitespace. Empty files are intentional for truly blank pages. Spaces within receipt lines are retained.",
                "comparison_policy": "Report separately from the 200 natural pages. Compare GPU/CPU discrete tokens and EOS independently of authored-text quality. For blank pages report exact empty/nonempty output and hallucinated character count, not CER with a zero denominator. Group derived variants by parent.",
                "qualification": "Generated source/pixel/text freeze; visual review and GPU/CPU inference qualification are separate. Synthetic printed pages do not establish real scanned receipt robustness or broad multilingual accuracy.",
                "limitations": ["One typeface per layout; clear typesetting rather than real scans", "Three orthogonal rotated variants share one upright text parent", "Faint receipt shares its clean parent", "Five non-English scripts/languages do not cover all scripts or fonts", "No long-output, handwriting or arbitrary-angle rotation claim", "No calibration split; diagnostic evaluation only"],
                "pages": pages}
    artifact(LOCK, json_bytes(manifest))
    print(json.dumps({"manifest": LOCK.as_posix(), "manifest_sha256": digest(json_bytes(manifest)), "pages": len(pages), "parent_count": 11, "verified": args.verify}, indent=2))


if __name__ == "__main__":
    main()
