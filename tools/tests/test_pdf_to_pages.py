"""Tests of tools/pdf_to_pages.py on PDFs built here with Pillow (image-only scans) and pypdfium2.

Run: python -m unittest discover -s tools/tests   (needs requirements/tools.txt)
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageCms

TOOL = Path(__file__).resolve().parents[1] / "pdf_to_pages.py"
_spec = importlib.util.spec_from_file_location("pdf_to_pages", TOOL)
pdf_to_pages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdf_to_pages)


def noise(size: tuple[int, int], bands: int, seed: int) -> bytes:
    return random.Random(seed).randbytes(size[0] * size[1] * bands)


# Scans at 100 dpi: 170x220 pixels are 122.4x158.4 points.
BITS = Image.frombytes("L", (170, 220), noise((170, 220), 1, 1)).point(lambda v: 255 * (v >= 128))
BITS = BITS.convert("1", dither=Image.Dither.NONE)
GRAY = Image.frombytes("L", (170, 220), noise((170, 220), 1, 2))
COLOUR = Image.frombytes("RGB", (170, 220), noise((170, 220), 3, 3))
PLAIN = Image.frombytes("L", (200, 300), noise((200, 300), 1, 4))
TRANSPARENT = Image.merge("RGBA", [*Image.new("RGB", (80, 100)).split(), Image.linear_gradient("L").resize((80, 100))])


def add_image(pdf: pdfium.PdfDocument, page: pdfium.PdfPage, picture: Image.Image, box: tuple) -> None:
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(picture))
    left, bottom, right, top = box
    image.set_matrix(pdfium.PdfMatrix().scale(right - left, top - bottom).translate(left, bottom))
    page.insert_obj(image)


def add_text(pdf: pdfium.PdfDocument, page: pdfium.PdfPage, text: str, invisible: bool = False) -> None:
    font = pdfium.PdfFont.load_standard(pdf, "Helvetica")
    item = pdfium_c.FPDFPageObj_CreateTextObj(pdf, font, 12.0)
    utf16 = ctypes.create_string_buffer((text + "\0").encode("utf-16-le"))
    pdfium_c.FPDFText_SetText(item, ctypes.cast(utf16, pdfium_c.FPDF_WIDESTRING))
    if invisible:
        pdfium_c.FPDFTextObj_SetTextRenderMode(item, pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE)
    pdfium_c.FPDFPageObj_Transform(item, 1, 0, 0, 1, 20, 20)
    pdfium_c.FPDFPage_InsertObject(page, item)


def build(directory: Path) -> Path:
    """Pages 1-3: Pillow scans (1-bit CCITT, gray and RGB JPEG). 4: text and a path. 5: a scan with an
    invisible OCR layer. 6: an image not covering the page. 7: a scan on a page with /Rotate 90.
    8: a scan with a soft mask."""
    scans = directory / "scans.pdf"
    BITS.save(scans, "PDF", resolution=100.0, save_all=True, append_images=[GRAY, COLOUR])
    pdf = pdfium.PdfDocument.new()
    pdf.import_pages(pdfium.PdfDocument(scans))
    page = pdf.new_page(300, 200)
    add_text(pdf, page, "Hello")
    rectangle = pdfium_c.FPDFPageObj_CreateNewRect(10, 10, 50, 30)
    pdfium_c.FPDFPath_SetDrawMode(rectangle, pdfium_c.FPDF_FILLMODE_ALTERNATE, 1)
    pdfium_c.FPDFPage_InsertObject(page, rectangle)
    page.gen_content()
    page = pdf.new_page(144, 216)
    add_image(pdf, page, PLAIN, (0, 0, 144, 216))
    add_text(pdf, page, "OCR layer", invisible=True)
    page.gen_content()
    page = pdf.new_page(144, 216)
    add_image(pdf, page, PLAIN, (20, 20, 100, 140))
    page.gen_content()
    page = pdf.new_page(144, 216)
    add_image(pdf, page, PLAIN, (0, 0, 144, 216))
    page.set_rotation(90)
    page.gen_content()
    page = pdf.new_page(80, 100)
    add_image(pdf, page, TRANSPARENT, (0, 0, 80, 100))
    page.gen_content()
    path = directory / "mixed.pdf"
    pdf.save(path)
    return path


def image_pdf(width: int, height: int, colorspace: str, bits: int, samples: bytes, icc: bytes = b"") -> bytes:
    """A one-page PDF, written by hand, whose content is one Flate image of `bits` per component covering
    a `width` x `height` point page (PDFium cannot write images deeper than 8 bits); with `icc`, an RGB
    profile, the image is ICCBased."""
    stream = zlib.compress(samples)
    content = f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q".encode()
    if icc:
        colorspace = "[/ICCBased 6 0 R]"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
        f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace {colorspace} "
        f"/BitsPerComponent {bits} /Filter /FlateDecode /Length {len(stream)} >>\nstream\n".encode()
        + stream + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    if icc:
        objects.append(f"<< /N 3 /Length {len(icc)} >>\nstream\n".encode() + icc + b"\nendstream")
    pdf = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(pdf)


def build_depths(directory: Path) -> Path:
    """Pages 1-2: 16-bit gray and RGB scans. 3-4: 4- and 8-bit gray scans. 5-6: 16- and 8-bit ICCBased
    (sRGB) scans. All 30 x 20 points and pixels."""
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    pages = [
        image_pdf(30, 20, "/DeviceGray", 16, noise((30, 20), 2, 6)),
        image_pdf(30, 20, "/DeviceRGB", 16, noise((30, 20), 6, 7)),
        image_pdf(30, 20, "/DeviceGray", 4, noise((15, 20), 1, 8)),
        image_pdf(30, 20, "/DeviceGray", 8, noise((30, 20), 1, 5)),
        image_pdf(30, 20, "", 16, noise((30, 20), 6, 9), icc=srgb),
        image_pdf(30, 20, "", 8, noise((30, 20), 3, 10), icc=srgb),
    ]
    pdf = pdfium.PdfDocument.new()
    for page in pages:
        pdf.import_pages(pdfium.PdfDocument(page))
    path = directory / "depths.pdf"
    pdf.save(path)
    return path


def load(source: Path | io.BytesIO) -> Image.Image:
    with Image.open(source) as image:
        image.load()
    return image


def embedded_jpeg(path: Path, page_index: int) -> Image.Image:
    """The page's JPEG stream decoded by Pillow."""
    page = pdfium.PdfDocument(path)[page_index]
    image = next(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]))
    return load(io.BytesIO(bytes(image.get_data(decode_simple=True))))


class PdfToPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.scratch.name)
        cls.pdf = build(cls.directory)
        cls.depths = build_depths(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def run_tool(self, name: str, *arguments: str, pdf: Path | None = None) -> tuple[int, str, Path]:
        """Run the tool in process on `pdf` (default: the mixed PDF) into a fresh directory; returns the exit
        code (argparse's included), stderr and the directory."""
        output = self.directory / name
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = pdf_to_pages.main([str(pdf or self.pdf), "--output-dir", str(output), *arguments])
            except SystemExit as error:
                code = error.code
        return code, stderr.getvalue(), output

    def manifest(self, output: Path) -> dict:
        return json.loads((output / "manifest.json").read_text())

    def test_page_ranges(self):
        self.assertEqual(pdf_to_pages.parse_pages(None, 3), [1, 2, 3])
        self.assertEqual(pdf_to_pages.parse_pages("5, 1-3,2", 8), [1, 2, 3, 5])
        self.assertEqual(pdf_to_pages.parse_pages("7-", 8), [7, 8])
        for bad in ["0", "9", "4-2", "x", "1-3-5", "", "-2", "9-"]:
            with self.assertRaises(ValueError, msg=bad):
                pdf_to_pages.parse_pages(bad, 8)
        with self.assertRaisesRegex(ValueError, "outside pages 1-8"):
            pdf_to_pages.parse_pages("9-", 8)
        with self.assertRaisesRegex(ValueError, "runs backwards"):
            pdf_to_pages.parse_pages("4-2", 8)

    def test_auto_extracts_scans_and_renders_other_pages(self):
        code, _, output = self.run_tool("auto")
        self.assertEqual(code, 0)
        pages = self.manifest(output)["pages"]
        self.assertEqual(
            [(page["mode"], page["image_mode"], page.get("extract_skipped")) for page in pages],
            [
                ("extract", "1", None),
                ("extract", "L", None),
                ("extract", "RGB", None),
                ("render", "RGB", "the page has visible text"),
                ("extract", "L", None),
                ("render", "RGB", "the image does not cover the page"),
                ("extract", "L", None),
                ("render", "RGB", "the image has transparency"),
            ],
        )
        png = [load(output / page["file"]) for page in pages]
        # Native resolution and exact pixels: the 1-bit scan, the JPEG streams as Pillow decodes them,
        # the lossless gray image (behind an OCR layer), and the rotated page turned clockwise.
        self.assertEqual(png[0].tobytes(), BITS.tobytes())
        self.assertEqual(png[1].tobytes(), embedded_jpeg(self.pdf, 1).tobytes())
        self.assertEqual(png[2].tobytes(), embedded_jpeg(self.pdf, 2).convert("RGB").tobytes())
        self.assertEqual(png[4].tobytes(), PLAIN.tobytes())
        self.assertEqual(png[6].tobytes(), PLAIN.transpose(Image.Transpose.ROTATE_270).tobytes())
        # Rendered pages: the longer side is 1536 pixels.
        self.assertEqual([png[i].size for i in (3, 5, 7)], [(1536, 1024), (1024, 1536), (1229, 1536)])
        for page, picture in zip(pages, png, strict=True):
            self.assertEqual(page["pixels"], list(picture.size))
            self.assertEqual(page["image_mode"], picture.mode)
        listed = (output / "pages.txt").read_text().splitlines()
        self.assertEqual(listed, [str((output / f"page-{n:04d}.png").resolve()) for n in range(1, 9)])

    def test_manifest_records_source_pages_and_images(self):
        code, _, output = self.run_tool("manifest", "--pages", "1,4,7")
        self.assertEqual(code, 0)
        manifest = self.manifest(output)
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        self.assertEqual(manifest["source"]["pages"], 8)
        self.assertEqual(manifest["settings"], {"mode": "auto", "long_side": 1536, "dpi": None, "gray": False,
                                                "pages": "1,4,7"})
        scan, vector, rotated = manifest["pages"]
        self.assertEqual((scan["page"], scan["file"], scan["size_pt"], scan["rotation"]),
                         (1, "page-0001.png", [122.4, 158.4], 0))
        self.assertEqual(scan["dpi"], [100.0, 100.0])
        self.assertEqual(scan["source_image"], {"pixels": [170, 220], "bits_per_component": 1,
                                                "colorspace": "DeviceGray", "filters": ["CCITTFaxDecode"]})
        self.assertEqual((vector["size_pt"], vector["pixels"], vector["scale"], vector["dpi"]),
                         ([300.0, 200.0], [1536, 1024], 5.12, 368.64))
        self.assertEqual((rotated["size_pt"], rotated["rotation"], rotated["pixels"]), ([216.0, 144.0], 90, [300, 200]))
        self.assertEqual(rotated["source_image"]["pixels"], [200, 300])

    def test_auto_handles_text_pages_and_screenshots_with_text_layers(self):
        pdf = pdfium.PdfDocument.new()
        page = pdf.new_page(144, 216)
        add_text(pdf, page, "Screenshot page")
        page.gen_content()
        screenshot = pdf_to_pages.render(page, (200, 300), False).convert("RGB")
        self.assertLess(screenshot.convert("L").getextrema()[0], 255)
        # The same page as an image, then with an invisible OCR layer or visible overlay.
        for invisible in (None, True, False):
            page = pdf.new_page(144, 216)
            add_image(pdf, page, screenshot, (0, 0, 144, 216))
            if invisible is not None:
                add_text(pdf, page, "Text overlay", invisible=invisible)
            page.gen_content()
        path = self.directory / "screenshots.pdf"
        pdf.save(path)
        pdf.close()

        code, error, output = self.run_tool("screenshots", "--long-side", "300", pdf=path)
        self.assertEqual((code, error), (0, ""))
        pages = self.manifest(output)["pages"]
        self.assertEqual([page["mode"] for page in pages], ["render", "extract", "extract", "render"])
        images = [load(output / page["file"]).convert("RGB") for page in pages]
        for image in images[:3]:
            self.assertEqual(image.size, screenshot.size)
            self.assertEqual(image.tobytes(), screenshot.tobytes())
        with pdfium.PdfDocument(path) as pdf:
            expected = pdf_to_pages.render(pdf[3], (200, 300), False).convert("RGB")
        self.assertEqual(images[3].tobytes(), expected.tobytes())
        self.assertNotEqual(images[3].tobytes(), screenshot.tobytes())

    def test_auto_renders_stretched_scans_in_page_coordinates(self):
        pdf = pdfium.PdfDocument.new()
        # Stretch vertically, then horizontally on a rotated page. The final
        # page is uniformly scaled, apart from normal PDF coordinate rounding.
        for width, height, rotation in [(144, 432, 0), (288, 216, 90), (144.000004, 216, 0)]:
            page = pdf.new_page(width, height)
            add_image(pdf, page, PLAIN, (0, 0, width, height))
            page.set_rotation(rotation)
            page.gen_content()
        path = self.directory / "stretched.pdf"
        pdf.save(path)
        pdf.close()

        code, error, output = self.run_tool("stretched", "--long-side", "432", pdf=path)
        self.assertEqual((code, error), (0, ""))
        pages = self.manifest(output)["pages"]
        self.assertEqual([page["mode"] for page in pages], ["render", "render", "extract"])
        self.assertEqual([page["pixels"] for page in pages], [[144, 432], [324, 432], [200, 300]])
        with pdfium.PdfDocument(path) as pdf:
            for index in (0, 1):
                self.assertIn("pixel scales", pages[index]["extract_skipped"])
                expected = pdf_to_pages.render(pdf[index], tuple(pages[index]["pixels"]), False)
                actual = load(output / pages[index]["file"])
                self.assertEqual(actual.convert("RGB").tobytes(), expected.convert("RGB").tobytes())
        self.assertEqual(load(output / pages[2]["file"]).tobytes(), PLAIN.tobytes())
        code, error, output = self.run_tool("stretched-extract", "--mode", "extract", pdf=path)
        self.assertEqual(code, 1)
        self.assertIn("pixel scales", error)
        self.assertFalse(output.exists())

    def test_pixel_scales_may_differ_by_the_geometry_tolerance(self):
        # A4 is 595.276 x 841.89 pt, so a 300 dpi scan of it (2480 x 3508) has scales 0.016% apart, and
        # the 595 x 842 pt box a scanner may write 0.043% apart; a 1% stretch is past the 0.5% tolerance.
        scans = [(595.276, 841.89, (2480, 3508)), (595, 842, (2480, 3508)), (144, 218.16, (200, 300))]
        pdf = pdfium.PdfDocument.new()
        for width, height, size in scans:
            page = pdf.new_page(width, height)
            add_image(pdf, page, Image.linear_gradient("L").resize(size), (0, 0, width, height))
            page.gen_content()
        path = self.directory / "a4.pdf"
        pdf.save(path)
        pdf.close()

        code, error, output = self.run_tool("a4-extract", "--mode", "extract", "--pages", "1-2", pdf=path)
        self.assertEqual((code, error), (0, ""))
        self.assertEqual([(p["mode"], p["pixels"]) for p in self.manifest(output)["pages"]],
                         [("extract", [2480, 3508])] * 2)
        code, error, output = self.run_tool("a4-auto", "--long-side", "432", pdf=path)
        self.assertEqual((code, error), (0, ""))
        pages = self.manifest(output)["pages"]
        self.assertEqual([page["mode"] for page in pages], ["extract", "extract", "render"])
        self.assertIn("pixel scales", pages[2]["extract_skipped"])
        self.assertEqual(pages[2]["pixels"], [285, 432])

    def test_render_sizes_and_gray(self):
        code, _, output = self.run_tool("long", "--mode", "render", "--long-side", "1000", "--pages", "1,4")
        self.assertEqual(code, 0)
        pages = self.manifest(output)["pages"]
        self.assertEqual([(page["mode"], page["pixels"]) for page in pages], [("render", [773, 1000]),
                                                                              ("render", [1000, 667])])
        code, _, output = self.run_tool("dpi", "--mode", "render", "--dpi", "144", "--pages", "1", "--gray")
        self.assertEqual(code, 0)
        page = self.manifest(output)["pages"][0]
        self.assertEqual((page["pixels"], page["dpi"], page["image_mode"]), ([245, 317], 144.0, "L"))
        self.assertEqual(load(output / "page-0001.png").mode, "L")

    def test_extract_mode_refuses_pages_that_are_not_scans(self):
        code, _, output = self.run_tool("extract", "--mode", "extract", "--pages", "1-3,5,7")
        self.assertEqual(code, 0)
        self.assertTrue(all(page["mode"] == "extract" for page in self.manifest(output)["pages"]))
        code, error, output = self.run_tool("refused", "--mode", "extract")
        self.assertEqual(code, 1)
        self.assertIn("page 4 cannot be extracted: the page has visible text", error)
        self.assertFalse(output.exists())

    def test_a_page_that_cannot_be_written_leaves_no_output(self):
        pdf = pdfium.PdfDocument.new()
        pdf.new_page(144, 216)
        pdf.new_page(4000, 4000)  # 16667 x 16667 px at 300 dpi
        path = self.directory / "oversized.pdf"
        pdf.save(path)
        pdf.close()
        for mode in ("render", "auto"):
            code, error, output = self.run_tool("oversized-" + mode, "--mode", mode, "--dpi", "300", pdf=path)
            self.assertEqual((code, "page 2: a 16667x16667 render is too large" in error, output.exists()),
                             (1, True, False), mode)
        code, _, output = self.run_tool("oversized-first", "--dpi", "300", "--pages", "1", pdf=path)
        self.assertEqual((code, self.manifest(output)["pages"][0]["pixels"]), (0, [600, 900]))

    def test_selected_pages_in_order_and_no_overwrite(self):
        code, _, output = self.run_tool("selected", "--pages", "3,1-2")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(path.name for path in output.glob("*.png")),
                         ["page-0001.png", "page-0002.png", "page-0003.png"])
        self.assertEqual([Path(line).name for line in (output / "pages.txt").read_text().splitlines()],
                         ["page-0001.png", "page-0002.png", "page-0003.png"])
        code, error, _ = self.run_tool("selected", "--pages", "4")
        self.assertEqual(code, 1)
        self.assertIn("already holds output", error)
        code, error, _ = self.run_tool("outside", "--pages", "9")
        self.assertEqual((code, "outside pages 1-8" in error), (1, True))

    def test_images_deeper_than_8_bits_are_rendered(self):
        # PDFium reports its 8-bit conversion of a 16-bit scan (24 bits per pixel); the stream says 16.
        code, _, output = self.run_tool("depths", pdf=self.depths)
        self.assertEqual(code, 0)
        pages = self.manifest(output)["pages"]
        deep = ("render", "the image has 16 bits per component", None)
        self.assertEqual(
            [(page["mode"], page.get("extract_skipped"), page.get("source_image", {}).get("bits_per_component"))
             for page in pages],
            [deep, deep, ("extract", None, 4), ("extract", None, 8), deep, ("extract", None, 8)],
        )
        self.assertEqual(load(output / "page-0004.png").tobytes(), noise((30, 20), 1, 5))
        self.assertEqual(self.manifest(output)["pages"][5]["source_image"]["colorspace"], "ICCBased")
        code, error, _ = self.run_tool("depths-extract", "--mode", "extract", pdf=self.depths)
        self.assertEqual(code, 1)
        self.assertIn("page 1 cannot be extracted: the image has 16 bits per component", error)

    def test_bad_arguments_fail_cleanly(self):
        code, error, output = self.run_tool("missing", pdf=self.directory / "missing.pdf")
        self.assertEqual(code, 1)
        self.assertIn("no PDF file at", error)
        self.assertFalse(output.exists())
        for dpi in ("inf", "nan", "0", "-5"):
            code, error, _ = self.run_tool("dpi-" + dpi, "--dpi", dpi)
            self.assertEqual((code, "--dpi must be a positive number" in error), (2, True), dpi)
        for side in ("0", "20000"):
            code, error, _ = self.run_tool("side-" + side, "--long-side", side)
            self.assertEqual((code, "--long-side must be between 1 and 16384" in error), (2, True), side)
        code, error, _ = self.run_tool("huge", "--dpi", "1e9")
        self.assertEqual((code, "render is too large" in error), (1, True))
        code, error, _ = self.run_tool("open-range", "--pages", "9-")
        self.assertEqual((code, "outside pages 1-8" in error), (1, True))

    def test_script_entry_point(self):
        result = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--long-side", result.stdout)


if __name__ == "__main__":
    unittest.main()
