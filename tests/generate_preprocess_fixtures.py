"""Generate independent RGB parity fixtures from Pillow and pinned upstream code.

Run from the repository root with Python, Pillow, NumPy, and PyTorch.
Only pure helpers are extracted from the pinned source, so this generator does
not require the Transformers version used for GPU inference. The GPU fixture
exporter separately validates the complete published processor.
"""

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/model/processing_falcon_ocr.py"


def image(width, height):
    # Channel-specific high-frequency content exercises negative filter lobes,
    # clipping, borders, and quantization between the two filter passes.
    y, x, c = np.indices((height, width, 3), dtype=np.uint32)
    return Image.fromarray(((x * 73 + y * 151 + c * 97 + x * y * 11) % 256).astype(np.uint8))


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def independent_positions(width, height):
    """IEEE FP32 sqrt and endpoint-anchored linspace, independent of Torch VML.

    Evaluate the multiply-add in FP64 and round once to FP32. Grid indices are
    small integers, so FP64 retains the exact FP32 product and addition here.
    This provides fused multiply-add rounding without requiring native FMA.
    """
    xlim = np.sqrt(np.float32(width) / np.float32(height))
    ylim = np.sqrt(np.float32(height) / np.float32(width))

    def linspace(limit, steps):
        if steps == 1:
            return np.array([-limit], dtype=np.float32)
        step = np.float32((limit - (-limit)) / np.float32(steps - 1))
        return np.array([
            np.float64(step) * i - np.float64(limit) if i < steps // 2
            else np.float64(limit) - np.float64(step) * (steps - i - 1)
            for i in range(steps)
        ], dtype=np.float32)

    x, y = np.meshgrid(linspace(xlim, width), linspace(ylim, height))
    return np.stack([y.flatten(), x.flatten()], axis=-1)


environment = {
    "math": math, "np": np,
    "get_image_size": lambda array, channel_dim=None: array.shape[:2],
    "resize": lambda array, size, resample, input_data_format: np.asarray(
        Image.fromarray(array).resize((size[1], size[0]), resample)),
}
functions = {"resize_image_if_necessary", "smart_resize"}
tree = ast.parse(SOURCE.read_text())
pure_helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions], type_ignores=[])
exec(compile(pure_helpers, str(SOURCE), "exec"), environment)

resizes = []
for width, height, target_width, target_height in [
    (5, 7, 3, 4), (5, 7, 13, 11), (23, 17, 16, 16),
    (31, 19, 13, 29), (100, 80, 16, 16), (31, 19, 31, 12),
    (31, 19, 9, 19), (11, 9, 11, 9), (1, 1, 8, 5),
    (1, 19, 7, 3), (19, 1, 3, 7), (256, 192, 1, 1),
]:
    output = np.asarray(image(width, height).resize((target_width, target_height), Image.Resampling.BICUBIC))
    resizes.append(dict(width=width, height=height, target_width=target_width, target_height=target_height, sha256=digest(output)))

prepared = []
for width, height, minimum, maximum in [
    (64, 64, 64, 1536), (67, 73, 64, 1536), (40, 56, 16, 1536),
    (32, 100, 64, 1536), (99, 200, 64, 128), (1000, 1600, 64, 1536),
    (1600, 1000, 64, 1536), (32, 2000, 64, 1536), (16, 16, 16, 1536),
]:
    first = environment["resize_image_if_necessary"](image(width, height), minimum, maximum)
    rgb = environment["smart_resize"](np.asarray(first), 16, Image.Resampling.BICUBIC, None)
    # Transformers >=5 rescale deliberately upcasts to FP64 then casts to FP32.
    scaled = (rgb.astype(np.float64) * (1 / 255)).astype(np.float32)
    normalized = (scaled - np.float32(0.5)) / np.float32(0.5)
    h, w = rgb.shape[:2]
    patches = normalized.reshape(h // 16, 16, w // 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 768)
    # Same fully-visible grid as the upstream mask reductions, using actual
    # PyTorch FP32 division, sqrt, linspace, and meshgrid for independent bits.
    grid_w, grid_h = torch.tensor(w // 16), torch.tensor(h // 16)
    xlim, ylim = torch.sqrt(grid_w / grid_h), torch.sqrt(grid_h / grid_w)
    xpos = torch.linspace(-xlim, xlim, int(grid_w))
    ypos = torch.linspace(-ylim, ylim, int(grid_h))
    pos_w, pos_h = torch.meshgrid(xpos, ypos, indexing="xy")
    positions = torch.stack([pos_h.flatten(), pos_w.flatten()], dim=-1).numpy()
    independent = independent_positions(w // 16, h // 16)
    error = np.abs(positions.astype(np.float64) - independent.astype(np.float64))
    # Both operands have the same sign for this symmetric construction. Exact
    # signed zero is canonicalized before the diagnostic ULP measurement.
    a_bits = np.where(positions == 0, np.float32(0), positions).view(np.uint32).astype(np.int64)
    b_bits = np.where(independent == 0, np.float32(0), independent).view(np.uint32).astype(np.int64)
    max_ulp = int(np.max(np.abs(a_bits - b_bits)))
    prepared.append(dict(width=width, height=height, minimum=minimum, maximum=maximum,
        output_width=w, output_height=h, patches_sha256=digest(patches),
        positions_x_bits=xpos.numpy().view(np.uint32).tolist(),
        positions_y_bits=ypos.numpy().view(np.uint32).tolist(),
        independent_positions_sha256=digest(independent),
        reference_spatial_max_absolute_error=float(error.max()),
        reference_spatial_max_ulp=max_ulp))

output = dict(pillow=PIL.__version__, numpy=np.__version__, torch=torch.__version__,
    processor_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    spatial_reference="PyTorch CPU uses MKL VML_HA sqrt, which can differ from correctly rounded IEEE FP32 sqrt. Per-case bounds are measured from PyTorch against independent NumPy FP32 sqrt and fused endpoint-anchored linspace, before any Rust comparison. Pixels and patches remain exact. ULP is diagnostic; near-zero cancellation magnifies ULP distances.",
    resizes=resizes, prepared=prepared)
target = ROOT / "tests/fixtures/preprocess.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(output, indent=2) + "\n")
print(target)
