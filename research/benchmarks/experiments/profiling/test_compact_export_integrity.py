"""Tiny synthetic XML only; no existing trace, ETL, model or profiler access."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import analyze_compact_stacks as analyzer
import check_compact_export_integrity as helper


class CompactExportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, sample_count=2, unused_caller=0):
        document = f'''<StackWindow><StackSource>
        <Frames Count="3">
          <Frame ID="0">Process64 ocr_bench (412) Args: --label synthetic</Frame>
          <Frame ID="1">ocr_bench!falcon_ocr::kernels::attention64_candidate::compact_head</Frame>
          <Frame ID="2">ocr_bench!falcon_ocr::kernels::linear_with_simd</Frame>
        </Frames><Stacks Count="4">
          <Stack ID="0" FrameID="0" CallerID="-1"/>
          <Stack ID="1" FrameID="1" CallerID="0"/>
          <Stack ID="2" FrameID="2" CallerID="0"/>
          <Stack ID="3" FrameID="1" CallerID="{unused_caller}"/>
        </Stacks><Samples Count="{sample_count}">
          <Sample ID="0" Time="1.0" StackID="1" Metric="1"/>
          <Sample ID="1" Time="1001.5" StackID="2" Metric="2"/>
        </Samples></StackSource></StackWindow>'''
        file = self.folder / 'synthetic.xml.zip'
        with zipfile.ZipFile(file, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('synthetic.xml', document)
        summary = analyzer.summarize(file, 412)
        summary['analyzer_sha256'] = helper.digest(Path(analyzer.__file__))
        return file, summary

    def test_success_cli_source_window_and_fresh_output(self):
        file, summary = self.fixture()
        summary_path = self.folder / 'summary.json'
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        output = self.folder / 'integrity.json'
        arguments = ['check_compact_export_integrity.py', '--input', str(file),
                     '--summary', str(summary_path), '--pid', '412', '--output', str(output)]
        with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()):
            helper.main()
        report = json.loads(output.read_bytes())
        self.assertEqual(report['actual_counts'], {'Frames': 3, 'Stacks': 4, 'Samples': 2})
        self.assertTrue(report['xml_member']['crc_validated_by_complete_read'])
        self.assertTrue(report['all_references_valid_and_all_stack_chains_acyclic'])
        self.assertTrue(report['analyzer_counts_times_roots_categories_names_and_bins_reproduced'])
        self.assertTrue(report['source_and_input_window_unchanged'])
        self.assertEqual(len(report['checked_sources_before_after']), 4)
        self.assertEqual(report['every_sample_has_exactly_one_process_root_pid'], 412)
        self.assertEqual(report['status'], 'export_integrity_verified_capture_audit_still_required')
        with patch.object(sys, 'argv', arguments), self.assertRaisesRegex(ValueError, 'Fresh output'):
            helper.main()

    def test_declared_count_mismatch(self):
        file, summary = self.fixture(sample_count=3)
        with self.assertRaisesRegex(ValueError, 'Declared/actual inventory'):
            helper.inspect(file, 412, summary)

    def test_unused_invalid_reference(self):
        file, summary = self.fixture(unused_caller=99)
        with self.assertRaisesRegex(ValueError, 'Invalid unused stack/frame reference'):
            helper.inspect(file, 412, summary)

    def test_unused_cycle(self):
        file, summary = self.fixture(unused_caller=3)
        with self.assertRaisesRegex(ValueError, 'Cyclic stack chain'):
            helper.inspect(file, 412, summary)


if __name__ == '__main__':
    unittest.main()
