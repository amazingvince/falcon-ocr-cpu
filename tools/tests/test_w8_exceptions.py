"""Exception columns in the W8 overlay tools (w8_variants.py, w8_proxy.py), on synthetic Grams.

  python -m unittest discover -s tools/tests
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import w8_proxy  # noqa: E402
import w8_variants as v  # noqa: E402
from convert_w8 import CONFIG_SHA256, FORMAT, SOURCE_SHA256  # noqa: E402

MASSIVE = 37


def synthetic(massive: bool = True, n: int = 24, k: int = 256, seed: int = 0):
    """Weights and the mean Gram of their inputs. With `massive`, channel MASSIVE is 1,000 times
    larger than typical (energy 10^6, like a squared-ReLU input of W2) and its weights 1,000 times
    smaller, as a trained model keeps its output sane."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)
    x = rng.standard_normal((4 * k, k)) * rng.uniform(0.5, 2.0, k)
    if massive:
        x[:, MASSIVE] *= 1e3
        w[:, MASSIVE] /= 1e3
    return w, (x.T @ x / len(x)).astype(np.float32)


def proxy(w, gram, codes, scales, exceptions, values, group=64):
    return w8_proxy.relative_error(w, w8_proxy.reconstruct(codes, scales, group, exceptions, values), gram)


class Selection(unittest.TestCase):
    def test_energy_picks_the_largest_diagonal_and_error_weights_it_by_the_rounding_step(self):
        w, gram = synthetic()
        diag = np.diag(gram)
        top = np.sort(np.argsort(-diag.astype(np.float64))[:4])
        np.testing.assert_array_equal(v.select_exceptions(w, diag, 4, "energy", 64), top)
        self.assertIn(MASSIVE, v.select_exceptions(w, diag, 1, "energy", 64))
        self.assertEqual(len(v.select_exceptions(w, diag, 0, "energy", 64)), 0)
        with self.assertRaises(ValueError):
            v.select_exceptions(w, diag, 256, "energy", 64)
        # Column 3 has the most energy, but its group's weights (and rounding steps) are tiny.
        w = np.ones((4, 128), dtype=np.float32)
        w[:, :64] = 1e-3
        diag = np.ones(128)
        diag[3], diag[70] = 10.0, 9.0
        self.assertEqual(v.select_exceptions(w, diag, 1, "energy", 64).tolist(), [3])
        self.assertEqual(v.select_exceptions(w, diag, 1, "error", 64).tolist(), [70])


class RoundToNearest(unittest.TestCase):
    def test_exception_columns_keep_their_values_and_leave_zero_codes(self):
        w, gram = synthetic()
        exceptions = v.select_exceptions(w, np.diag(gram), 2, "energy", 64)
        codes, scales, values = v.quantize_rtn(w, 64, "rtn", None, exceptions)
        self.assertTrue((codes[:, exceptions] == 0).all())
        np.testing.assert_array_equal(values, w[:, exceptions])
        zeroed = w.copy()
        zeroed[:, exceptions] = 0
        plain_codes, plain_scales, none = v.quantize_rtn(zeroed, 64, "rtn", None)
        np.testing.assert_array_equal(codes, plain_codes)
        np.testing.assert_array_equal(scales, plain_scales)
        self.assertEqual(none.shape, (w.shape[0], 0))
        # The massive channel's rounding error dominated; keeping it exact removes it.
        rtn = proxy(w, gram, *v.quantize_rtn(w, 64, "rtn", None)[:2], v.NO_EXCEPTIONS, none)
        self.assertLess(proxy(w, gram, codes, scales, exceptions, values), 0.2 * rtn)

    def test_only_wmse_weights_the_scale_search_by_the_gram(self):
        # A loaded Gram (for exception columns) must not turn --clip mse into wmse.
        w, gram = synthetic()
        diag = np.diag(gram).astype(np.float64)
        plain = v.quantize_rtn(w, 64, "mse", None)
        for got, want in zip(v.quantize_rtn(w, 64, "mse", diag), plain, strict=True):
            np.testing.assert_array_equal(got, want)
        weighted = v.quantize_rtn(w, 64, "wmse", diag)
        self.assertFalse(np.array_equal(weighted[1], plain[1]))


class Gptq(unittest.TestCase):
    def test_exception_columns_hold_the_least_squares_optimum_for_the_codes(self):
        """GPTQ's update extends to never-quantized trailing columns: after the last quantized
        column they sit at W_E - (Q - W_Q) H_QE H_EE^-1 for the damped Gram H."""
        w, gram = synthetic()
        exceptions = v.select_exceptions(w, np.diag(gram), 3, "energy", 64)
        kept = np.setdiff1d(np.arange(w.shape[1]), exceptions)
        h = gram.astype(np.float64) + np.eye(w.shape[1]) * 0.01 * np.mean(np.diag(gram).astype(np.float64))
        for act_order in (False, True):
            codes, scales, values = v.quantize_gptq(w, gram, 64, "rtn", 0.01, act_order, exceptions)
            self.assertTrue((codes[:, exceptions] == 0).all())
            q = w8_proxy.reconstruct(codes, scales, 64, v.NO_EXCEPTIONS, np.zeros((w.shape[0], 0))).astype(np.float64)
            delta = q[:, kept] - w[:, kept]
            optimum = w[:, exceptions] - delta @ h[np.ix_(kept, exceptions)] @ np.linalg.inv(
                h[np.ix_(exceptions, exceptions)])
            self.assertLess(np.abs(values - optimum).max(), 1e-5 * np.abs(optimum).max(), f"act_order {act_order}")

    def test_exception_columns_never_raise_the_proxy(self):
        for massive in (True, False):
            w, gram = synthetic(massive, seed=1)
            for act_order in (False, True):
                codes, scales, values = v.quantize_gptq(w, gram, 64, "rtn", 0.01, act_order)
                plain = proxy(w, gram, codes, scales, v.NO_EXCEPTIONS, values)
                for count in (1, 4):
                    exceptions = v.select_exceptions(w, np.diag(gram), count, "energy", 64)
                    codes, scales, values = v.quantize_gptq(w, gram, 64, "rtn", 0.01, act_order, exceptions)
                    self.assertLessEqual(proxy(w, gram, codes, scales, exceptions, values), plain,
                                         f"massive {massive} act_order {act_order} k {count}")

    def test_no_exceptions_is_the_plain_quantizer(self):
        w, gram = synthetic()
        for act_order in (False, True):
            codes, scales, values = v.quantize_gptq(w, gram, 64, "rtn", 0.01, act_order)
            again = v.quantize_gptq(w, gram, 64, "rtn", 0.01, act_order, v.NO_EXCEPTIONS)
            np.testing.assert_array_equal(codes, again[0])
            np.testing.assert_array_equal(scales, again[1])
            self.assertEqual(values.shape, (w.shape[0], 0))
        self.assertEqual(sorted(v.overlay_tensors("m", codes, scales, v.NO_EXCEPTIONS, values)),
                         ["m.__w8_codes", "m.__w8_scales"])


class Proxy(unittest.TestCase):
    def test_error_and_energy_share(self):
        w, gram = synthetic()
        self.assertEqual(w8_proxy.relative_error(w, w, gram), 0.0)
        self.assertAlmostEqual(w8_proxy.relative_error(w, 1.01 * w, gram), 0.01, places=6)
        diag = np.diag(gram)
        self.assertEqual(w8_proxy.energy_share(diag, 0), 0.0)
        self.assertAlmostEqual(w8_proxy.energy_share(diag, len(diag)), 1.0)
        self.assertGreater(w8_proxy.energy_share(diag, 1), 0.99)
        report = w8_proxy.evaluate(w, gram, [0, 1, 4], ["rtn", "gptq"])
        self.assertEqual(sorted(report), ["energy_share", "gptq", "rtn"])
        self.assertLess(report["gptq"][1], report["gptq"][0])
        self.assertLess(report["rtn"][1], report["rtn"][0])


class CommandLines(unittest.TestCase):
    """Both tools end to end on a two-matrix stand-in for the pinned checkpoint."""

    SHAPES = {"layers.0.attention.wo.weight": (16, 128), "layers.0.feed_forward.w2.weight": (8, 192)}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.model, self.grams = root / "model", root / "gram"
        self.model.mkdir()
        self.grams.mkdir()
        tensors = {}
        for i, (name, (n, k)) in enumerate(self.SHAPES.items()):
            w, gram = synthetic(n=n, k=k, seed=i)
            tensors[name] = w
            np.save(self.grams / f"{name}.gram.npy", gram)
        save_file(tensors, self.model / "model.safetensors")
        (self.model / "config.json").write_text("{}")
        self.root = root
        pinned = mock.Mock(side_effect=lambda p: SOURCE_SHA256 if p.name == "model.safetensors" else CONFIG_SHA256)
        for module in (v, w8_proxy):
            for patch in (mock.patch.object(module, "sha256", pinned),
                          mock.patch.object(module, "inventory", lambda include_head: dict(self.SHAPES))):
                patch.start()
                self.addCleanup(patch.stop)

    def tearDown(self):
        self.directory.cleanup()

    def run_tool(self, main, *argv):
        with redirect_stdout(io.StringIO()) as out:
            main([*argv])
        return out.getvalue()

    def overlay(self, name, *extra):
        path = self.root / name
        self.run_tool(v.main, "--model", str(self.model), "--output", str(path), "--gram-dir", str(self.grams),
                      "--method", "gptq", "--act-order", *extra)
        with safe_open(path, framework="numpy") as f:
            return f.metadata(), {key: f.get_tensor(key) for key in f.keys()}

    def test_overlays_use_format_v2_only_with_exception_columns(self):
        metadata, tensors = self.overlay("plain.safetensors")
        self.assertEqual(metadata["format"], FORMAT)
        self.assertEqual(len(tensors), 4)
        self.assertNotIn("exceptions", metadata)
        metadata, tensors = self.overlay("w2.safetensors", "--exceptions", "2", "--exceptions-include", r"\.w2")
        self.assertEqual(metadata["format"], v.FORMAT_EXCEPTIONS)
        self.assertEqual((metadata["exception_matrices"], metadata["exception_columns"]), ("1", "2"))
        w2 = "layers.0.feed_forward.w2.weight"
        expected = [f"{m}.__w8_{t}" for m in self.SHAPES for t in ("codes", "scales")]
        self.assertEqual(sorted(tensors), sorted([*expected, f"{w2}.__w8_exc_cols", f"{w2}.__w8_exc_vals"]))
        columns = tensors[f"{w2}.__w8_exc_cols"]
        self.assertEqual((columns.dtype, tensors[f"{w2}.__w8_exc_vals"].dtype), (np.int32, np.float32))
        self.assertEqual(tensors[f"{w2}.__w8_exc_vals"].shape, (8, 2))
        self.assertIn(MASSIVE, columns)
        self.assertTrue((tensors[f"{w2}.__w8_codes"][:, columns] == 0).all())
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            v.main(["--model", str(self.model), "--output", str(self.root / "x.safetensors"), "--exceptions", "2"])

    def test_exceptions_that_match_no_selected_matrix_are_refused(self):
        output = self.root / "none.safetensors"
        # No matrix matches, and the matching one is left out by --include.
        for extra in (["--exceptions-include", "no_such_matrix"],
                      ["--include", r"\.wo", "--exceptions-include", r"\.w2"]):
            argv = ["--model", str(self.model), "--output", str(output), "--gram-dir", str(self.grams),
                    "--exceptions", "2", *extra]
            with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()), \
                    mock.patch("sys.stderr", io.StringIO()) as stderr:
                v.main(argv)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--exceptions-include matches no matrix selected by --include/--exclude", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_proxy_reports_the_overlay_it_would_build(self):
        self.overlay("w2.safetensors", "--exceptions", "2", "--exceptions-include", r"\.w2")
        report = self.root / "proxy.json"
        text = self.run_tool(w8_proxy.main, "--gram-dir", str(self.grams), "--model", str(self.model), "--ks",
                             "0,2", "--overlay", str(self.root / "w2.safetensors"), "--json", str(report))
        self.assertIn("mean over 2 matrices", text)
        result = json.loads(report.read_text())
        w2, wo = (result["matrices"][f"layers.0.{m}.weight"] for m in ("feed_forward.w2", "attention.wo"))
        # Same recipe (G64, act-order, energy selection): the overlay's error is the proxy's GPTQ k=2 entry.
        self.assertEqual((w2["overlay"], w2["overlay_exceptions"]), (w2["gptq"][1], 2))
        self.assertEqual((wo["overlay"], wo["overlay_exceptions"]), (wo["gptq"][0], 0))
        self.assertEqual(result["exception_bytes"], [0, 2 * (4 * 16 + 4) + 2 * (4 * 8 + 4)])


if __name__ == "__main__":
    unittest.main()
