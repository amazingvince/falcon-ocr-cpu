"""Unit tests for research/benchmarks/scripts/compare_focr.py (synthetic inputs only, no binaries)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_focr as cf  # noqa: E402


def ocr_bench_report(prefills, decodes, walls, tokens=32, text="hello", ids=(1, 2, 3, 11)):
    samples = []
    for p, d, w in zip(prefills, decodes, walls):
        samples.append({
            "emitted_tokens": tokens,
            "pages_per_second": 1.0,
            "wall_ms": w,
            "per_request": [{
                "finish_reason": "eos", "height": 256, "width": 176, "input_tokens": 192,
                "output_tokens": tokens, "precision": "fp32", "teacher_forced": False, "text": text,
                "timings": {"decode_ms": d, "image_decode_ms": 0.0, "image_projection_ms": 0.5,
                            "prefill_ms": p, "preprocessing_ms": 3.0, "time_to_first_token_ms": p + 3.0,
                            "total_ms": p + d + 3.0, "transformer_prefill_ms": p - 0.5},
            }],
        })
    return {
        "schema_version": 2,
        "verified_model_load_ms": 750.0,
        "cases": [{
            "active_batch_size": 1, "batch_size": 1, "execution": "independent_sequential",
            "image_indices": [0], "median_ms": 1.0, "median_pages_per_second": 1.0,
            "memory_after": {"peak_resident_bytes": 1000},
            "samples": samples, "token_ids": [list(ids)],
        }],
    }


def focr_bench_report(prefills, decodes, elapsed, generated):
    return {
        "backend_report": {"backend": "ggml-linear", "linear": "ggml-cpu"},
        "prompt_mode": "hf-split", "repeats": len(prefills), "max_new_tokens": 32,
        "prompt_tokens": 192, "image_width": 176, "image_height": 256, "image_grid_w": 11,
        "image_grid_h": 16, "image_patches": 176, "original_width": 570, "original_height": 829,
        "bounded_width": 176, "bounded_height": 256, "max_image_tokens": 39200,
        "resized_by_token_budget": False, "backend_prepare_ms": 900.0,
        "prefill_ms": prefills, "decode_ms": decodes, "total_ocr_latency_ms": [p + d for p, d in zip(prefills, decodes)],
        "elapsed_ms": elapsed, "generated_tokens": generated,
        "decode_tokens_per_second": [g / d * 1000 for g, d in zip(generated, decodes)],
        "end_to_end_tokens_per_second": [g / e * 1000 for g, e in zip(generated, elapsed)],
        "peak_rss_kib": 2048, "last_text": "hello",
    }


class ParsingTests(unittest.TestCase):
    def test_first_json_object_skips_noise_and_trailing_text(self):
        text = "warning: something\n{\n  \"a\": 1,\n  \"b\": [1, 2]\n}\nrepeat 1: 0.5s\n"
        self.assertEqual(cf.first_json_object(text), {"a": 1, "b": [1, 2]})

    def test_last_json_line_takes_last_object_line(self):
        text = '{"first": 1}\n{"second": 2}\n'
        self.assertEqual(cf.last_json_line(text), {"second": 2})

    def test_ours_cold_parser(self):
        result = {"text": "hi", "token_ids": [5, 6, 11], "finish_reason": "eos", "width": 720, "height": 1024,
                  "input_tokens": 2896, "output_tokens": 3, "precision": "fp32", "backend": "avx2",
                  "cache_layout": "expanded", "weight_layout": "unpacked", "teacher_forced": False,
                  "timings": {"prefill_ms": 2800.0, "decode_ms": 90.0, "total_ms": 2950.0, "preprocessing_ms": 50.0}}
        parsed = cf.parse_ours_cold(json.dumps(result) + "\n", "Verified model loaded in 751.2 ms\n", 5000.0)
        self.assertEqual(parsed["model_load_ms"], 751.2)
        self.assertEqual(parsed["token_ids"], [5, 6, 11])
        self.assertEqual(parsed["stop_reason"], "eos")
        self.assertEqual(parsed["width"], 720)

    def test_focr_cold_parser_derives_stop_reason(self):
        result = {"text": "hi", "generated_ids": [5, 6, 11], "backend_report": {"linear": "ggml-cpu"},
                  "prompt_tokens": 2896, "image_width": 720, "image_height": 1024, "image_grid_w": 45,
                  "image_grid_h": 64, "image_patches": 2880, "resized_by_token_budget": False,
                  "backend_prepare_ms": 900.0, "prefill_ms": 3000.0, "decode_ms": 100.0,
                  "total_ocr_latency_ms": 3100.0, "decode_tokens_per_second": 30.0, "prompt_mode": "hf-split"}
        parsed = cf.parse_focr_cold(json.dumps(result, indent=2), 6000.0, 32, {11, 263})
        self.assertEqual(parsed["stop_reason"], "eos")
        self.assertEqual(parsed["patches"], 2880)
        result["generated_ids"] = list(range(32))
        parsed = cf.parse_focr_cold(json.dumps(result), 6000.0, 32, {11, 263})
        self.assertEqual(parsed["stop_reason"], "length")

    def test_focr_preprocess_parser(self):
        dump = {"tokens": list(range(2896)), "image_width": 720, "image_height": 1024, "image_grid_w": 45,
                "image_grid_h": 64, "image_patches": 2880, "resized_by_token_budget": False,
                "image_rgb_crc32": 1, "max_pixels": 10035200, "prompt_mode": "hf-split"}
        parsed = cf.parse_focr_preprocess(json.dumps(dump))
        self.assertEqual(parsed["prompt_tokens"], 2896)
        self.assertEqual(parsed["patches"], 2880)


class SampleTests(unittest.TestCase):
    def test_ocr_bench_samples_and_summary(self):
        samples = cf.samples_from_ocr_bench(ocr_bench_report([100, 110, 105], [700, 710, 705], [810, 830, 820]))
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["model_ms"], 800)
        summary = cf.summarize_samples(samples)
        self.assertEqual(summary["median_model_ms"], 810)
        self.assertEqual(summary["median_e2e_ms"], 820)
        self.assertEqual(summary["output_tokens"], [32])
        self.assertAlmostEqual(summary["median_decode_tok_s"], 32 / 705 * 1000)

    def test_focr_bench_drops_warmups(self):
        report = focr_bench_report([500, 100, 110, 105], [900, 700, 710, 705], [1500, 820, 830, 825], [32, 32, 32, 32])
        samples = cf.samples_from_focr_bench(report, warmup=1)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["prefill_ms"], 100)
        self.assertEqual(samples[0]["model_ms"], 800)
        with self.assertRaises(ValueError):
            cf.samples_from_focr_bench(report, warmup=4)

    def test_control_drift(self):
        self.assertIsNone(cf.control_drift([1.0]))
        self.assertAlmostEqual(cf.control_drift([100.0, 104.0]), 0.04)
        self.assertAlmostEqual(cf.control_drift([104.0, 100.0]), 0.04)


class PolicyTests(unittest.TestCase):
    def test_clamp_max_new_tokens(self):
        self.assertEqual(cf.clamp_max_new_tokens(4096, 2896, 8192), 4096)
        self.assertEqual(cf.clamp_max_new_tokens(4096, 6544, 8192), 1648)
        with self.assertRaises(ValueError):
            cf.clamp_max_new_tokens(10, 8192, 8192)

    def test_bracket_order_is_abba(self):
        self.assertEqual(cf.bracket_order(1), ["ours", "focr"])
        self.assertEqual(cf.bracket_order(2), ["ours", "focr", "focr", "ours"])
        self.assertEqual(cf.bracket_order(3), ["ours", "focr", "focr", "ours", "ours", "focr"])

    def test_geometry_check(self):
        ours = {"width": 1088, "height": 1536, "patches": 6528}
        focr = {"width": 1088, "height": 1536, "patches": 6528, "resized_by_token_budget": False}
        self.assertEqual(cf.geometry_matches(ours, focr), (True, []))
        focr["resized_by_token_budget"] = True
        ok, problems = cf.geometry_matches(ours, focr)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 1)
        focr["patches"] = 6000
        ok, problems = cf.geometry_matches(ours, focr)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 2)

    def test_focr_stop_reason(self):
        self.assertEqual(cf.focr_stop_reason([1, 2, 11], 32, {11}), "eos")
        self.assertEqual(cf.focr_stop_reason([1, 2, 3], 32, {11}), "eos")
        self.assertEqual(cf.focr_stop_reason(list(range(32)), 32, {11}), "length")

    def test_parse_cmake_isa(self):
        cache = "GGML_AVX2:BOOL=ON\nGGML_AVX512:BOOL=OFF\nGGML_NATIVE:BOOL=OFF\nOTHER:STRING=x\nCMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022\n"
        self.assertEqual(cf.parse_cmake_isa(cache), {"GGML_AVX2": "ON", "GGML_AVX512": "OFF", "GGML_NATIVE": "OFF",
                                                     "CMAKE_GENERATOR": "Visual Studio 17 2022"})


class AgreementTests(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(cf.levenshtein_normalized([], []), (0.0, "levenshtein"))
        self.assertEqual(cf.levenshtein_normalized([1, 2, 3], [1, 2, 3]), (0.0, "levenshtein"))
        distance, method = cf.levenshtein_normalized("kitten", "sitting")
        self.assertAlmostEqual(distance, 3 / 7)
        self.assertEqual(method, "levenshtein")
        distance, method = cf.levenshtein_normalized([1, 2, 3, 4], [1, 2, 4])
        self.assertAlmostEqual(distance, 1 / 4)

    def test_levenshtein_falls_back_on_huge_inputs(self):
        old = cf.LEVENSHTEIN_CELL_LIMIT
        cf.LEVENSHTEIN_CELL_LIMIT = 10
        try:
            distance, method = cf.levenshtein_normalized("abcdef", "abcdxf")
            self.assertEqual(method, "difflib-ratio")
            self.assertGreater(distance, 0.0)
        finally:
            cf.LEVENSHTEIN_CELL_LIMIT = old

    def test_first_divergence(self):
        self.assertIsNone(cf.first_divergence([1, 2], [1, 2]))
        self.assertEqual(cf.first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(cf.first_divergence([1, 2, 3], [1, 2]), 2)

    def test_agreement_strips_stop_tokens(self):
        a = cf.agreement([1, 2, 3, 11], [1, 2, 3, 263], "abc", "abc", {11, 263})
        self.assertTrue(a["token_ids_exact"])
        self.assertIsNone(a["first_divergence_index"])
        self.assertEqual(a["common_prefix_fraction"], 1.0)
        self.assertTrue(a["text_exact"])
        b = cf.agreement([1, 2, 3, 4], [1, 2, 9, 4], "abcd", "abxd", {11})
        self.assertFalse(b["token_ids_exact"])
        self.assertEqual(b["first_divergence_index"], 2)
        self.assertAlmostEqual(b["common_prefix_fraction"], 0.5)
        self.assertAlmostEqual(b["token_levenshtein_normalized"], 0.25)


class CommandTests(unittest.TestCase):
    def args(self, **overrides):
        argv = ["--output", "unused", "--pages", "page.png"]
        parsed = cf.build_parser().parse_args(argv)
        for key, value in overrides.items():
            setattr(parsed, key, value)
        return parsed

    def test_focr_commands_carry_matched_settings(self):
        args = self.args(max_dimension=1536, focr_max_image_tokens=39200, focr_threads=8)
        command = cf.focr_cold_command(args, Path("p.png"), 1648)
        self.assertIn("--max-image-tokens", command)
        self.assertEqual(command[command.index("--max-image-tokens") + 1], "39200")
        self.assertEqual(command[command.index("--max-dimension") + 1], "1536")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "1648")
        self.assertEqual(command[command.index("--threads") + 1], "8")
        self.assertIn("--json", command)
        bench = cf.focr_warm_command(args, Path("p.png"), 1648, 5, threads=32)
        self.assertEqual(bench[bench.index("--repeats") + 1], "5")
        self.assertEqual(bench[bench.index("--threads") + 1], "32")

    def test_ours_cache_layout_default_is_the_runtime_default(self):
        args = self.args()
        cold = cf.ours_cold_command(args, Path("p.png"), 8)
        self.assertEqual(cold[cold.index("--cache-layout") + 1], "compact")

    def test_ours_commands_carry_matched_settings(self):
        args = self.args(max_dimension=1024, ours_threads=16, ours_cache_layout="compact")
        cold = cf.ours_cold_command(args, Path("p.png"), 4096)
        self.assertEqual(cold[cold.index("--max-dimension") + 1], "1024")
        self.assertEqual(cold[cold.index("--cache-layout") + 1], "compact")
        warm = cf.ours_warm_command(args, Path("p.png"), 4096, Path("out.json"), 2, 3, threads=8)
        self.assertEqual(warm[warm.index("--threads") + 1], "8")
        self.assertEqual(warm[warm.index("--warmup") + 1], "2")
        self.assertEqual(warm[warm.index("--repetitions") + 1], "3")
        self.assertEqual(warm[warm.index("--execution") + 1], "sequential")


class RenderTests(unittest.TestCase):
    def test_markdown_renders_without_error(self):
        samples = cf.samples_from_ocr_bench(ocr_bench_report([100, 110, 105], [700, 710, 705], [810, 830, 820]))
        summary = cf.summarize_samples(samples)
        report = {
            "created_utc": "now", "host": {"cpu_label": "cpu", "environment_label": "env"},
            "settings": {"max_dimension": 1024, "min_dimension": 64, "max_new_tokens_requested": 32,
                         "warmup": 2, "repetitions": 3, "processes": 2, "order": "ABBA"},
            "implementations": {"ours": {"binary_sha256": "a" * 64, "git_commit": "b" * 40},
                                "focr": {"binary_sha256": "c" * 64, "git_commit": "d" * 40, "cmake_isa": {"GGML_AVX2": "ON"}}},
            "pages": [{
                "id": "p1", "geometry": {"ours": {"width": 1, "height": 2, "patches": 3}},
                "warm": {"summary": {"ours": {"best": summary, "control_drift_pct": 1.0},
                                     "focr": {"best": summary, "control_drift_pct": 2.0},
                                     "ratio_focr_over_ours": {"model": 1.5}}},
                "agreement": cf.agreement([1, 2], [1, 2], "ab", "ab", {11}),
                "cold": {"ours": {"process_wall_ms": 1000, "model_load_ms": 700, "prefill_ms": 100, "decode_ms": 700,
                                  "output_tokens": 32, "stop_reason": "eos"}},
            }, {"id": "p2", "error": "boom"}],
            "notes": ["note"],
            "thread_sweep": {"ours": [{"threads": 16, "median_model_ms": 800, "median_prefill_ms": 100, "median_decode_ms": 700}],
                             "focr": [], "selected": {"ours": 16, "focr": None}},
        }
        text = cf.render_markdown(report)
        self.assertIn("| p1 |", text)
        self.assertIn("1.500", text)
        self.assertIn("Thread sweep", text)


if __name__ == "__main__":
    unittest.main()
