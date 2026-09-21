"""Tiny synthetic fixtures only; no actual export, ETL or model is accessed."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import inspect_compact_export_scope as scope

FROZEN_HELPER = 'f9a1e327e37a8753583ea65907fcbe8f1765458cc4c7083b27390fee4d14900b'
HEAD = 'ocr_bench!falcon_ocr::kernels::attention64_candidate::compact_head'


class CompactExportScopeTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(scope.digest(Path(scope.__file__)), FROZEN_HELPER)
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.number = 0

    def tearDown(self):
        self.temporary.cleanup()
        self.assertEqual(scope.digest(Path(scope.__file__)), FROZEN_HELPER)

    def fixture(self, sample_stacks, unused_caller=0, declared_count=None):
        self.number += 1
        count = len(sample_stacks) if declared_count is None else declared_count
        samples = ''.join(f'<Sample ID="{i}" Time="{10+i}" StackID="{sid}" Metric="{i+1}"/>'
                          for i, sid in enumerate(sample_stacks))
        document = f'''<StackWindow><StackSource><Frames Count="3">
        <Frame ID="0">Process64 ocr_bench (412) Args: synthetic</Frame>
        <Frame ID="1">{HEAD}</Frame>
        <Frame ID="2">Process64 unrelated (999)</Frame>
        </Frames><Stacks Count="10">
        <Stack ID="0" FrameID="0" CallerID="-1"/>
        <Stack ID="1" FrameID="1" CallerID="0"/>
        <Stack ID="2" FrameID="2" CallerID="-1"/>
        <Stack ID="3" FrameID="1" CallerID="2"/>
        <Stack ID="4" FrameID="1" CallerID="-1"/>
        <Stack ID="5" FrameID="0" CallerID="0"/>
        <Stack ID="6" FrameID="1" CallerID="5"/>
        <Stack ID="7" FrameID="2" CallerID="0"/>
        <Stack ID="8" FrameID="1" CallerID="7"/>
        <Stack ID="9" FrameID="1" CallerID="{unused_caller}"/>
        </Stacks><Samples Count="{count}">{samples}</Samples></StackSource></StackWindow>'''
        file = self.folder / f'synthetic-{self.number}.xml.zip'
        with zipfile.ZipFile(file, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('synthetic.xml', document)
        return file

    def inspect(self, file):
        records = io.StringIO()
        report = scope.inspect(file, 412, records)
        raw = records.getvalue()
        self.assertEqual(report['emitted_rejected_records_sha256'], hashlib.sha256(raw.encode()).hexdigest())
        lines = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(lines), report['rejected_sample_count'])
        self.assertEqual(sum(report['sample_counts'].values()), report['all_sample_count'])
        self.assertEqual(sum(report['sample_metrics'].values()), report['all_sample_metric'])
        self.assertFalse(report['complete_capture_accepted'])
        return report, lines

    def run_cli(self, file, output):
        arguments = ['inspect_compact_export_scope.py', '--input', str(file), '--pid', '412', '--output', str(output)]
        with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()):
            result = scope.main()
        return result, json.loads(output.read_bytes())

    def test_valid_target_cli_closure(self):
        file = self.fixture([1, 1])
        code, report = self.run_cli(file, self.folder / 'valid.json')
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'process_scope_verified_capture_audit_still_required')
        self.assertTrue(report['strict_frozen_analyzer_scope_would_pass'])
        self.assertFalse(report['complete_capture_accepted'])
        self.assertEqual(report['actual_counts'], {'Frames': 3, 'Stacks': 10, 'Samples': 2})
        self.assertEqual(report['sample_counts'], {'valid_target': 2, 'wrong_pid': 0, 'rootless': 0, 'ambiguous_roots': 0})
        self.assertEqual(report['all_sample_metric'], 3)
        self.assertTrue(report['xml_member']['crc_validated_by_complete_read'])
        self.assertTrue(report['source_and_export_unchanged'])
        self.assertEqual(report['checked_source_and_export_before_after'][str(Path(scope.__file__).resolve())], FROZEN_HELPER)
        self.assertEqual(Path(report['rejected_records']['path']).read_bytes(), b'')

    def test_rootless_record_and_rejected_cli_exit(self):
        file = self.fixture([1, 4])
        report, records = self.inspect(file)
        self.assertEqual(report['sample_counts']['rootless'], 1)
        self.assertEqual(records[0], {'ordinal': 1, 'sample_id': 1, 'time_relative_ms': 11.0,
                                    'stack_id': 4, 'metric': 2.0, 'classification': 'rootless',
                                    'process_roots': [], 'root_pids': [], 'complete_chain_leaf_to_root': [HEAD]})
        self.assertEqual(report['valid_target_subset']['sample_metric_denominator'], 1)
        self.assertEqual(report['valid_target_subset']['categories'][0]['percent_of_all_export_metric'], 100 / 3)
        code, saved = self.run_cli(file, self.folder / 'rejected.json')
        self.assertEqual(code, 1)
        self.assertEqual(saved['status'], 'rejected_process_scope')
        self.assertFalse(saved['strict_frozen_analyzer_scope_would_pass'])
        saved_records = [json.loads(line) for line in Path(saved['rejected_records']['path']).read_text().splitlines()]
        self.assertEqual(saved_records, records)

    def test_wrong_pid_is_retained(self):
        report, records = self.inspect(self.fixture([1, 3]))
        self.assertEqual(report['sample_counts']['wrong_pid'], 1)
        self.assertEqual(records[0]['root_pids'], [999])
        self.assertEqual(records[0]['complete_chain_leaf_to_root'], [HEAD, 'Process64 unrelated (999)'])
        self.assertEqual({row['pid']: row['sample_count'] for row in report['process_root_distribution']}, {412: 1, 999: 1})
        self.assertEqual(report['rejected_sample_count'], 1)

    def test_duplicate_and_mixed_roots_are_ambiguous(self):
        for sid, pids in [(6, [412, 412]), (8, [999, 412])]:
            with self.subTest(pids=pids):
                report, records = self.inspect(self.fixture([sid]))
                self.assertEqual(report['sample_counts']['ambiguous_roots'], 1)
                self.assertEqual(report['sample_counts']['valid_target'], 0)
                self.assertEqual(records[0]['root_pids'], pids)
                self.assertEqual(len(records[0]['complete_chain_leaf_to_root']), 3)
                self.assertEqual(report['root_cardinality_sample_counts'], {2: 1})
                self.assertEqual(report['valid_target_subset']['categories'], [])
                self.assertIsNone(report['valid_target_subset']['sample_time_range_ms'])
                self.assertEqual(report['status'], 'rejected_process_scope')

    def test_unused_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cyclic stack chain'):
            self.inspect(self.fixture([1], unused_caller=9))

    def test_declared_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Declared/actual inventory mismatch'):
            self.inspect(self.fixture([1], declared_count=2))


if __name__ == '__main__':
    unittest.main()
