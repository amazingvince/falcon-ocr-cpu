"""Offline attribution checks; no recorder, model or real trace is executed."""
import tempfile
import unittest
from pathlib import Path
import zipfile

from analyze_stacks import summarize


class StackAttributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def archive(self, samples, extra_stacks="", process="Process ocr_bench (412)"):
        # XML names/attributes follow pinned PerfView v3.2.6 XMLStackSource.cs.
        document = f'''<StackWindow><StackSource>
          <Frames Count="5">
            <Frame ID="0">{process}</Frame>
            <Frame ID="1">ocr_bench!linear_with_simd</Frame>
            <Frame ID="2">ocr_bench!dot_avx2</Frame>
            <Frame ID="3">ocr_bench!attention_with_simd</Frame>
            <Frame ID="4">ocr_bench!axpy_avx2</Frame>
          </Frames>
          <Stacks Count="6">
            <Stack ID="0" CallerID="-1" FrameID="0"/>
            <Stack ID="1" CallerID="0" FrameID="1"/>
            <Stack ID="2" CallerID="1" FrameID="2"/>
            <Stack ID="3" CallerID="0" FrameID="3"/>
            <Stack ID="4" CallerID="3" FrameID="4"/>
            <Stack ID="5" CallerID="2" FrameID="2"/>
            {extra_stacks}
          </Stacks><Samples>{samples}</Samples>
        </StackSource></StackWindow>'''
        target = self.folder / 'samples.perfView.xml.zip'
        with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('samples.perfView.xml', document)
        return target

    def test_weighted_totals_distinguish_inclusive_from_exclusive(self):
        path = self.archive('''
          <Sample ID="0" Time="125.000" StackID="2"/>
          <Sample ID="1" Time="1001.000" StackID="4" Metric="2"/>
          <Sample ID="2" Time="2001.000" StackID="5" Metric="0.5"/>
        ''')
        result = summarize(path, 412)
        self.assertEqual(result['sample_count'], 3)
        self.assertEqual(result['sampled_cpu_metric_total'], 3.5)
        self.assertEqual(result['first_sample_relative_ms'], 125)
        self.assertEqual(result['last_sample_relative_ms'], 2001)
        self.assertEqual(result['process_roots'], {'Process ocr_bench (412)': 3.5})
        exclusive = {x['name']: x['sampled_cpu_metric'] for x in result['exclusive_by_name']}
        inclusive = {x['name']: x['sampled_cpu_metric'] for x in result['inclusive_by_name']}
        self.assertEqual(exclusive, {'ocr_bench!axpy_avx2': 2, 'ocr_bench!dot_avx2': 1.5})
        self.assertEqual(sum(exclusive.values()), 3.5)
        # A repeated name on one stack contributes once to its inclusive metric.
        self.assertEqual(inclusive['ocr_bench!dot_avx2'], 1.5)
        self.assertEqual(inclusive['Process ocr_bench (412)'], 3.5)
        self.assertEqual([x['trace_second'] for x in result['one_second_bins']], [0, 1, 2])

    def test_wrong_process_rejected(self):
        with self.assertRaisesRegex(ValueError, 'captured-PID'):
            summarize(self.archive('<Sample ID="0" Time="1" StackID="2"/>'), 413)

    def test_native_process64_label_and_arguments(self):
        root = 'Process64 ocr_bench (412) Args: --label "some (413) argument"'
        path = self.archive('<Sample ID="0" Time="1" StackID="2"/>', process=root)
        result = summarize(path, 412)
        self.assertEqual(result['process_roots'], {root: 1})
        with self.assertRaisesRegex(ValueError, 'captured-PID'):
            summarize(path, 413)

    def test_rootless_sample_cannot_hide_among_valid_samples(self):
        path = self.archive('''<Sample ID="0" Time="1" StackID="2"/>
            <Sample ID="1" Time="2" StackID="6"/>''',
            '<Stack ID="6" CallerID="-1" FrameID="2"/>')
        with self.assertRaisesRegex(ValueError, 'captured-PID'):
            summarize(path, 412)

    def test_ambiguous_process_root_rejected(self):
        path = self.archive('<Sample ID="0" Time="1" StackID="6"/>',
                            '<Stack ID="6" CallerID="2" FrameID="0"/>')
        with self.assertRaisesRegex(ValueError, 'captured-PID'):
            summarize(path, 412)

    def test_cyclic_stack_rejected(self):
        path = self.archive('<Sample ID="0" Time="1" StackID="6"/>',
                            '<Stack ID="6" CallerID="6" FrameID="2"/>')
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            summarize(path, 412)

    def test_nonfinite_metric_rejected(self):
        path = self.archive('<Sample ID="0" Time="1" StackID="2" Metric="NaN"/>')
        with self.assertRaisesRegex(ValueError, 'Invalid sample'):
            summarize(path, 412)


if __name__ == '__main__':
    unittest.main()
