"""Generate PNG mode/JPEG source-mode fixtures with the pinned Pillow runtime."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tests/fixtures/images"
TARGET.mkdir(parents=True, exist_ok=True)
source = ROOT / "artifacts/model/processing_falcon_ocr.py"
names = {"resize_image_if_necessary", "smart_resize"}
tree = ast.Module(body=[n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
environment = {
    "np": np, "math": math,
    "get_image_size": lambda array, channel_dim=None: array.shape[:2],
    "resize": lambda array, size, resample, input_data_format: np.asarray(Image.fromarray(array).resize((size[1], size[0]), resample)),
}
exec(compile(tree, str(source), "exec"), environment)

def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()

def pattern(width, height, channels):
    y, x, c = np.indices((height, width, channels), dtype=np.uint32)
    return ((x * 73 + y * 151 + c * 97 + x * y * 11) % 256).astype(np.uint8)

def png16(name, color, channels):
    data = pattern(33, 49, channels).astype(np.uint16) * 256
    data += np.arange(data.size, dtype=np.uint32).reshape(data.shape).astype(np.uint16) % 256
    if color == 0:
        # Include low samples to make clipping vs scaling visible.
        data %= 513
    raw = b"".join(b"\0" + row.astype(">u2").tobytes() for row in data)
    def chunk(name, body):
        return struct.pack(">I", len(body)) + name + body + struct.pack(">I", zlib.crc32(name + body))
    content = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB",33,49,16,color,0,0,0)) + chunk(b"IDAT",zlib.compress(raw)) + chunk(b"IEND",b"")
    (TARGET / name).write_bytes(content)

rgb = Image.fromarray(pattern(33,49,3))
rgba = Image.fromarray(pattern(33,49,4))
gray = Image.fromarray(pattern(33,49,1)[...,0])
alpha_gray = Image.fromarray(pattern(33,49,2), mode="LA")
rgb.save(TARGET / "rgb.png")
rgb.save(TARGET / "rgb-trns.png", transparency=(73,170,11))
rgba.save(TARGET / "rgba.png")
gray.save(TARGET / "gray.png")
gray.save(TARGET / "gray-trns.png", transparency=151)
alpha_gray.save(TARGET / "gray-alpha.png")
gray.convert("1").save(TARGET / "binary.png")
palette = rgb.quantize(colors=16)
palette.save(TARGET / "palette.png", bits=4)
palette.save(TARGET / "palette-trns.png", bits=4, transparency=0)
for color, channels, name in [(0,1,"gray16.png"),(2,3,"rgb16.png"),(4,2,"gray-alpha16.png"),(6,4,"rgba16.png")]:
    png16(name,color,channels)
for quality in (10,75,100):
    for sampling in (0,1,2):
        rgb.save(TARGET / f"rgb-q{quality}-s{sampling}.jpg",quality=quality,subsampling=sampling)
gray.save(TARGET / "gray.jpg",quality=75)
rgb.save(TARGET / "progressive.jpg",quality=75,progressive=True)
rgb.convert("CMYK").save(TARGET / "cmyk.jpg",quality=75)

cases = []
for path in sorted(TARGET.iterdir()):
    if path.suffix not in (".jpg", ".png"):
        continue
    image = Image.open(path)
    rgb_array = np.asarray(image.convert("RGB"))
    path.with_suffix(path.suffix + ".rgb").write_bytes(rgb_array.tobytes())
    for minimum, maximum in [(16,128),(64,128),(16,32)]:
        resized = environment["resize_image_if_necessary"](image,minimum,maximum).convert("RGB")
        final = environment["smart_resize"](np.asarray(resized),16,Image.Resampling.BICUBIC,None)
        normalized = ((final.astype(np.float64)*(1/255)).astype(np.float32)-np.float32(.5))/np.float32(.5)
        h,w=final.shape[:2]
        patches = normalized.reshape(h//16,16,w//16,16,3).transpose(0,2,1,3,4).reshape(-1,768)
        cases.append(dict(file=path.name,mode=image.mode,minimum=minimum,maximum=maximum,width=w,height=h,
            decoded_rgb_sha256=sha(rgb_array),patches_sha256=sha(patches)))
(ROOT / "tests/fixtures/decode.json").write_text(json.dumps(dict(pillow=PIL.__version__,numpy=np.__version__,cases=cases),indent=2)+"\n")
print(f"Created {len(cases)} file preprocessing cases")
