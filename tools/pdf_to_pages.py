#!/usr/bin/env python3
"""Turn the pages of a PDF into PNG images for `falcon-ocr run`, which reads PNG and JPEG only.

Modes (--mode):

  render   Rasterize each page with PDFium over white, annotations and form fields
           included, so that its longer side is exactly --long-side pixels (default
           1536, the runner's default maximum dimension, so the processor's first
           resize leaves the page as it is), or at --dpi N. --gray renders grayscale.
  extract  For PDFs containing scans or screenshots: write the embedded image at
           native resolution, decoded losslessly into a PNG. 1-bit images stay 1-bit,
           grayscale stays grayscale, anything else becomes RGB; the page's /Rotate is
           applied. A page qualifies when all it draws is one image, upright and
           covering the page's visible box (each edge within 2 pt or 0.5% of the longer
           side), with horizontal and vertical pixel scales within 0.5% of each other
           (a 300 dpi A4 scan on a rounded page box differs by 0.02-0.04%), a colour
           space (not a stencil mask), at most 8 bits per component and no
           transparency; invisible text (an OCR layer) and link annotations may
           accompany it. The bits per component are read from the image stream (PDFium
           reports its own 8-bit conversion); JPEG 2000 and images whose depth cannot
           be read are rendered. Any other page is an error.
  auto     (default) Extract the pages that qualify and render the others.

A 1-bit scan written by extract goes through the processor's nearest-neighbour
downscale, as Pillow resizes 1-bit and palette images; that can thin or break strokes
of a 300 dpi scan, which render antialiases instead. Compare both on your own pages.

The output directory (--output-dir, default <pdf name>-pages) must not hold an earlier
run's output. It receives page-0001.png, ... (numbered by PDF page), pages.txt (the
absolute output paths, one per line in page order) and manifest.json: the source
file's SHA-256 and, per page, its size in points and rotation, the pixel size, the mode
used, the scale or DPI, the PNG's image mode, the embedded image's size, bits per
component, colour space and filters when extracted, and why a page auto-rendered was
not extracted. Recognize the pages in order in one run, with the records in a file
that --resume continues:

  falcon-ocr run --list book-pages/pages.txt --output book.jsonl

or through xargs (the null-separated form works with GNU and BSD xargs; a long list
may be split over several runs):

  tr '\\n' '\\0' < book-pages/pages.txt | xargs -0 falcon-ocr run

Usage:
  python tools/pdf_to_pages.py book.pdf --output-dir book-pages
  python tools/pdf_to_pages.py scan.pdf --mode extract --pages 1-10,12
  python tools/pdf_to_pages.py slides.pdf --dpi 150 --gray

Requires pypdfium2 and Pillow: pip install -r requirements/tools.txt
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import PIL
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageChops

COLORSPACES = {
    getattr(pdfium_c, f"FPDF_COLORSPACE_{name.upper()}"): name
    for name in (
        "DeviceGray", "DeviceRGB", "DeviceCMYK", "CalGray", "CalRGB", "Lab", "ICCBased", "Separation", "DeviceN",
        "Indexed", "Pattern",
    )
}
# Annotations that draw nothing on the page as displayed.
HIDDEN_ANNOTATIONS = (pdfium_c.FPDF_ANNOT_LINK, pdfium_c.FPDF_ANNOT_POPUP)
# Components per pixel of the colour spaces whose image data can be sized (ICCBased: from the profile).
COMPONENTS = {
    pdfium_c.FPDF_COLORSPACE_DEVICEGRAY: 1, pdfium_c.FPDF_COLORSPACE_CALGRAY: 1,
    pdfium_c.FPDF_COLORSPACE_SEPARATION: 1, pdfium_c.FPDF_COLORSPACE_INDEXED: 1,
    pdfium_c.FPDF_COLORSPACE_DEVICERGB: 3, pdfium_c.FPDF_COLORSPACE_CALRGB: 3, pdfium_c.FPDF_COLORSPACE_LAB: 3,
    pdfium_c.FPDF_COLORSPACE_DEVICECMYK: 4,
}
ICC_COMPONENTS = {b"GRAY": 1, b"RGB ": 3, b"CMYK": 4}
# How far a scan's placement may differ from the page and still be extracted: each image edge from the page
# box (this fraction of the longer side, at least 2 pt) and the horizontal from the vertical pixel scale.
GEOMETRY_TOLERANCE = 0.005
# Renders above this many pixels (16384 x 16384) are refused rather than allocated.
MAX_RENDER_PIXELS = 1 << 28
# The transpose that shows an image as a page's /Rotate (degrees clockwise) displays it.
ROTATIONS = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}


def parse_pages(spec: str | None, count: int) -> list[int]:
    """The 1-based page numbers `spec` selects (`1-10,12`; `5-` runs to the last page), ascending, each once."""
    if spec is None:
        return list(range(1, count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        first, dash, last = part.partition("-")
        try:
            start = int(first)
            end = (int(last) if last else count) if dash else start
        except ValueError:
            raise ValueError(f"--pages: {part!r} is not a page number or range like 3 or 1-10") from None
        if not (1 <= start <= count and 1 <= end <= count):
            raise ValueError(f"--pages: {part!r} is outside pages 1-{count}")
        if start > end:
            raise ValueError(f"--pages: {part!r} runs backwards")
        pages.update(range(start, end + 1))
    return sorted(pages)


def sha256(path: Path) -> str:
    # Chunked rather than hashlib.file_digest so Python 3.10 also works.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_size(page_size: tuple[float, float], long_side: int, dpi: float | None) -> tuple[tuple[int, int], float]:
    """The rendered pixel size and pixels per point: `dpi`, or else the longer side exactly `long_side`."""
    width, height = page_size
    if dpi is not None:
        scale = dpi / 72
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if size[0] * size[1] > MAX_RENDER_PIXELS:
            raise ValueError(f"a {size[0]}x{size[1]} render is too large: lower --dpi")
        return size, scale
    scale = long_side / max(width, height)
    if width >= height:
        return (long_side, max(1, round(height * scale))), scale
    return (max(1, round(width * scale)), long_side), scale


def render(page: pdfium.PdfPage, size: tuple[int, int], gray: bool) -> Image.Image:
    """The page stretched onto exactly `size` pixels over white, with its annotations and form fields."""
    width, height = size
    bitmap_format = pdfium_c.FPDFBitmap_Gray if gray else pdfium_c.FPDFBitmap_BGR
    bitmap = pdfium.PdfBitmap.new_native(width, height, bitmap_format, rev_byteorder=not gray)
    bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
    flags = pdfium_c.FPDF_ANNOT | (pdfium_c.FPDF_GRAYSCALE if gray else pdfium_c.FPDF_REVERSE_BYTE_ORDER)
    placement = (bitmap, page, 0, 0, width, height, 0, flags)
    pdfium_c.FPDF_RenderPageBitmap(*placement)
    if page.formenv:
        pdfium_c.FPDF_FFLDraw(page.formenv, *placement)
    return bitmap.to_pil()


def bits_per_component(page: pdfium.PdfPage, image: pdfium.PdfImage, colorspace: int) -> int | None:
    """The image's BitsPerComponent, or None when it cannot be read. PDFium reports the depth of its own
    8-bit conversion (a 16-bit gray scan as 24 bits per pixel) and exposes no image dictionary, so this
    reads the stream: 1 for CCITT and JBIG2, 8 for DCT (the only depth PDF allows there), otherwise the
    one depth whose byte-aligned rows fill the decoded data exactly."""
    complex_filters = image.get_filters(skip_simple=True)
    if complex_filters in (["CCITTFaxDecode"], ["JBIG2Decode"]):
        return 1
    if complex_filters == ["DCTDecode"]:
        return 8
    if complex_filters:
        return None
    components = COMPONENTS.get(colorspace)
    if colorspace == pdfium_c.FPDF_COLORSPACE_ICCBASED:
        size = ctypes.c_size_t()
        if pdfium_c.FPDFImageObj_GetIccProfileDataDecoded(image, page, None, 0, ctypes.byref(size)):
            profile = (ctypes.c_ubyte * size.value)()
            if pdfium_c.FPDFImageObj_GetIccProfileDataDecoded(image, page, profile, size.value, ctypes.byref(size)):
                components = ICC_COMPONENTS.get(bytes(profile[16:20]))  # the header's data colour space
    if components is None:
        return None
    width, height = image.get_px_size()
    data = len(image.get_data(decode_simple=True))
    fits = [bits for bits in (1, 2, 4, 8, 16) if (width * components * bits + 7) // 8 * height == data]
    return fits[0] if len(fits) == 1 else None


def scan_image(page: pdfium.PdfPage) -> tuple[tuple[pdfium.PdfImage, int] | None, str]:
    """The image a scanned page consists of and its bits per component (see the module docstring), or None
    and why the page is not one."""
    images = []
    for item in page.get_objects(max_depth=1):
        if item.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            images.append(item)
        elif item.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
            if pdfium_c.FPDFTextObj_GetTextRenderMode(item) != pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
                return None, "the page has visible text"
        else:
            return None, "the page has vector graphics or form content"
    if len(images) != 1:
        return None, f"the page has {len(images)} images"
    for index in range(pdfium_c.FPDFPage_GetAnnotCount(page)):
        annotation = pdfium_c.FPDFPage_GetAnnot(page, index)
        subtype = pdfium_c.FPDFAnnot_GetSubtype(annotation)
        pdfium_c.FPDFPage_CloseAnnot(annotation)
        if subtype not in HIDDEN_ANNOTATIONS:
            return None, "the page has visible annotations"
    image = images[0]
    a, b, c, d, _, _ = image.get_matrix().get()
    if b != 0 or c != 0 or a <= 0 or d <= 0:
        return None, "the image is rotated, skewed or flipped"
    box = page.get_bbox()
    tolerance = max(2.0, GEOMETRY_TOLERANCE * max(box[2] - box[0], box[3] - box[1]))
    if any(abs(edge - page_edge) > tolerance for edge, page_edge in zip(image.get_bounds(), box, strict=True)):
        return None, "the image does not cover the page"
    metadata = image.get_metadata()
    # Scans are placed on rounded page boxes (a 2480x3508 px A4 scan on 595.276x841.89 pt has scales 0.016%
    # apart, on 595x842 pt 0.043%); a visible stretch is rendered, so the PNG keeps the displayed geometry.
    if not math.isclose(a / metadata.width, d / metadata.height, rel_tol=GEOMETRY_TOLERANCE):
        return None, "the image has different horizontal and vertical pixel scales"
    if metadata.colorspace not in COLORSPACES:
        return None, "the image has no colour space of its own (a stencil mask or JPEG 2000)"
    bits = bits_per_component(page, image, metadata.colorspace)
    if bits is None:
        return None, "the image's bits per component cannot be read"
    if bits > 8:
        return None, f"the image has {bits} bits per component"
    # Soft masks, colour-key masks and alpha show only in the image drawn with its masks.
    drawn = image.get_bitmap(render=True, scale_to_original=False).to_pil()
    if "A" in drawn.getbands() and drawn.getchannel("A").getextrema()[0] < 255:
        return None, "the image has transparency"
    return (image, bits), ""


def extract(page: pdfium.PdfPage, image: pdfium.PdfImage, bits: int) -> tuple[Image.Image, dict]:
    """The scan (`bits` per component) decoded at its native size (1-bit as mode 1, gray as L, else RGB),
    turned as the page shows it."""
    metadata = image.get_metadata()
    picture = image.get_bitmap(render=False).to_pil()
    if picture.mode != "L":
        picture = picture.convert("RGB")
        red, green, blue = picture.split()
        if ImageChops.difference(red, green).getbbox() is None and ImageChops.difference(green, blue).getbbox() is None:
            picture = red
    if picture.mode == "L" and bits == 1 and not any(picture.histogram()[1:255]):
        # Only 0 and 255: the threshold conversion is exact.
        picture = picture.convert("1", dither=Image.Dither.NONE)
    left, bottom, right, top = page.get_bbox()
    dpi = [picture.width * 72 / (right - left), picture.height * 72 / (top - bottom)]
    rotation = page.get_rotation()
    if rotation:
        picture = picture.transpose(ROTATIONS[rotation])
        if rotation != 180:
            dpi.reverse()
    source = {
        "pixels": [metadata.width, metadata.height],
        "bits_per_component": bits,
        "colorspace": COLORSPACES[metadata.colorspace],
        "filters": image.get_filters(),
    }
    return picture, {"dpi": [round(value, 3) for value in dpi], "source_image": source}


def check_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    earlier = [name for name in ("manifest.json", "pages.txt") if (output_dir / name).exists()]
    earlier += sorted(path.name for path in output_dir.glob("page-*.png"))
    if earlier:
        raise ValueError(f"{output_dir} already holds output ({earlier[0]}): remove it or choose another --output-dir")


class Plan(NamedTuple):
    """How a selected page is written: the scan's bits per component when it is extracted, else the render's
    pixel size and pixels per point, and why an auto-rendered page was not extracted."""

    number: int
    bits: int | None
    size: tuple[int, int] | None
    scale: float
    reason: str


def plan_pages(pdf: pdfium.PdfDocument, pages: list[int], mode: str, long_side: int, dpi: float | None) -> list[Plan]:
    """Decide how every selected page is written before any file is, so that a page that cannot be written (not
    a scan in --mode extract, or a render that is too large) leaves no partial output. Each page is closed
    again, so a long PDF is not held open."""
    plan = []
    for number in pages:
        page = pdf[number - 1]
        try:
            found, reason = scan_image(page) if mode != "render" else (None, "")
            if found is not None:
                plan.append(Plan(number, found[1], None, 0.0, ""))
            elif mode == "extract":
                raise ValueError(f"page {number} cannot be extracted: {reason} (--mode auto renders such pages)")
            else:
                try:
                    size, scale = render_size(page.get_size(), long_side, dpi)
                except ValueError as error:
                    raise ValueError(f"page {number}: {error}") from None
                plan.append(Plan(number, None, size, scale, reason))
        finally:
            page.close()
    return plan


def convert(
    pdf_path: Path, output_dir: Path, mode: str, long_side: int, dpi: float | None, gray: bool, page_spec: str | None
) -> dict:
    """Write the selected pages, pages.txt and manifest.json; returns the manifest."""
    if not pdf_path.is_file():
        raise ValueError(f"no PDF file at {pdf_path}")
    check_output_dir(output_dir)
    pdf = pdfium.PdfDocument(pdf_path)
    pdf.init_forms()  # before any page is loaded, so form fields render
    plan = plan_pages(pdf, parse_pages(page_spec, len(pdf)), mode, long_side, dpi)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries, paths = [], []
    for number, bits, size, scale, reason in plan:
        page = pdf[number - 1]
        if bits is not None:
            used = "extract"
            image = next(page.get_objects(max_depth=1, filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]))
            picture, details = extract(page, image, bits)
        else:
            used = "render"
            picture = render(page, size, gray)
            details = {"scale": round(scale, 6), "dpi": round(scale * 72, 3)}
            if reason:
                details["extract_skipped"] = reason
        name = f"page-{number:04d}.png"
        picture.save(output_dir / name, format="PNG")
        width, height = page.get_size()
        entries.append({
            "page": number, "file": name, "mode": used, "size_pt": [round(width, 3), round(height, 3)],
            "rotation": page.get_rotation(), "pixels": list(picture.size), "image_mode": picture.mode, **details,
        })
        paths.append(str((output_dir / name).resolve()))
        note = f" ({reason})" if reason else ""
        print(f"page {number}: {used} {picture.width}x{picture.height} {picture.mode}{note}")
        page.close()
    manifest = {
        "source": {"file": str(pdf_path), "sha256": sha256(pdf_path), "pages": len(pdf)},
        "settings": {
            "mode": mode, "long_side": None if dpi is not None else long_side, "dpi": dpi, "gray": gray,
            "pages": page_spec,
        },
        "versions": {
            "pypdfium2": str(pdfium.version.PYPDFIUM_INFO), "pdfium": str(pdfium.version.PDFIUM_INFO),
            "pillow": PIL.__version__,
        },
        "pages": entries,
    }
    (output_dir / "pages.txt").write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pdf.close()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="the PDF file")
    parser.add_argument("--mode", choices=("auto", "render", "extract"), default="auto", help="default: auto")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--long-side", type=int, default=1536, help="rendered pages' longer side in pixels (1536)")
    size.add_argument("--dpi", type=float, help="render at this resolution instead of --long-side")
    parser.add_argument("--pages", help="pages to convert, e.g. 1-10,12 or 5- (default: all)")
    parser.add_argument("--gray", action="store_true", help="render grayscale (extracted scans keep their mode)")
    parser.add_argument("--output-dir", type=Path, help="default: <pdf name>-pages in the current directory")
    args = parser.parse_args(argv)
    if not 1 <= args.long_side <= 1 << 14:
        parser.error("--long-side must be between 1 and 16384")
    if args.dpi is not None and not (math.isfinite(args.dpi) and args.dpi > 0):
        parser.error("--dpi must be a positive number")
    output_dir = args.output_dir or Path(f"{args.pdf.stem}-pages")
    try:
        manifest = convert(args.pdf, output_dir, args.mode, args.long_side, args.dpi, args.gray, args.pages)
    except (ValueError, OSError, pdfium.PdfiumError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    extracted = sum(entry["mode"] == "extract" for entry in manifest["pages"])
    print(f"{len(manifest['pages'])} pages ({extracted} extracted) in {output_dir}; list: {output_dir / 'pages.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
