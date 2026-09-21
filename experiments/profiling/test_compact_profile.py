"""Bounded offline checks: no preparation, recorder, model, elevation or build."""
import json
from pathlib import Path
import tempfile
import unittest

import capture_compact_windows as capture
from analyze_compact_stacks import category


class CompactProfileTests(unittest.TestCase):
    def test_wrong_kind_or_schema_rejected_before_runtime_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'plan.json'
            for data in [{'schema_version': 1, 'kind': 'old-expanded-plan'},
                         {'schema_version': 2, 'kind': 'falcon-ocr-owned-native-compact-cpu-profile-v1'}]:
                raw = json.dumps(data).encode()
                file.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'kind/schema'):
                    capture.run(file, capture.sha(raw))

    def test_command_and_prospective_source_closure(self):
        command = capture.command_for(Path('fresh'), Path('input.png'))
        for key, value in {'--execution': 'joint', '--cache-layout': 'compact',
                           '--threads': '16', '--backend': 'avx2', '--batches': '1',
                           '--warmup': '0', '--repetitions': '3', '--max-new-tokens': '4096'}.items():
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertTrue({'launch_compact_elevated.ps1', 'analyze_compact_stacks.py',
                         'COMPACT_PROFILE_ACCEPTANCE.md', 'COMPACT_PROFILE_README.md'} <= set(capture.SOURCE_NAMES))
        self.assertEqual(capture.LIMITS['collector_limit_bytes'], 2048 * 2**20)

    def test_real_frozen_baseline_and_wrong_layout_rejection(self):
        # Saved metadata only. No inference is run or claimed by this test.
        data = json.loads(capture.BASELINE.read_bytes())
        data['warmup'] = 0
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'saved.json'
            file.write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(capture.benchmark_check(file)['completed_full_page_outputs'], 3)
            data['cache_layout'] = 'expanded'
            file.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'cache_layout'):
                capture.benchmark_check(file)

    def test_exact_compact_symbols_and_category_priority(self):
        prefix = 'falcon_ocr::kernels::attention64_candidate::'
        for name in [prefix + 'compact_head', prefix + 'compact::closure_env$0',
                     prefix + 'compact::closure$0', prefix + 'compact::{{closure}}']:
            self.assertEqual(category(['ocr_bench!' + name]), 'attention_call_path')
            self.assertEqual(category([name, 'attention_gemm']), 'attention_gemm_call_path')
            self.assertEqual(category([name, 'linear_with_simd']), 'attention_call_path')
        for name in ['compact_head', prefix + 'compact_head_unrelated',
                     'other::' + prefix + 'compact_head', prefix + 'compact::closure_env$01']:
            self.assertEqual(category([name]), 'other_or_unresolved')
        self.assertEqual(category(['rayon', 'linear_with_simd']), 'linear_call_path')


if __name__ == '__main__':
    unittest.main()
