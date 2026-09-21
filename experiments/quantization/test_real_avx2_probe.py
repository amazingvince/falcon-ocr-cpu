"""Bounded tests of real-Q4 capture/checker guards; no model or timing run."""
import hashlib
import json
import pathlib
import tempfile
import unittest
import zipfile

import capture_real_avx2_probe as p


class RealQ4ProbeTests(unittest.TestCase):
    def test_real_row_selection_is_geometric_unique_and_never_replicated(self):
        self.assertEqual(p.row_indices(144, 1), [0])
        self.assertEqual(p.row_indices(144, 2), [0, 143])
        self.assertEqual(p.row_indices(144, 4), [0, 47, 95, 143])
        self.assertEqual(p.row_indices(144, 8), [0, 20, 40, 61, 81, 102, 122, 143])
        self.assertEqual(p.row_indices(1, 1), [0])
        for rows in [2, 4, 8]:
            with self.assertRaises(ValueError):
                p.row_indices(1, rows)

    def test_full_output_gate_detects_corruption_and_nonfinite_values(self):
        x = p.np.asarray([[1.5, -2.0, 4.0, 0.0], [-0.5, 1.0, 2.0, 1.0]], dtype='<f4')
        weights = p.np.asarray([[0.0, 1.0, 2.0, 4.0], [1.0, -1.0, 0.5, 0.25]], dtype='<f4')
        reference, absolute = p.fp64_dots(x, weights)
        output = reference.astype('<f4')
        self.assertEqual(p.arithmetic_metrics(output, reference, absolute, 4)['bound_violations'], 0)
        for value in [100.0, float('nan'), float('inf')]:
            changed = output.copy(); changed[-1, -1] = value
            with self.assertRaises(ValueError):
                p.arithmetic_metrics(changed, reference, absolute, 4)
        zeros = p.np.zeros((1, 1))
        p.arithmetic_metrics(zeros.astype('<f4'), zeros, zeros, 768)
        with self.assertRaises(ValueError):
            p.arithmetic_metrics(p.np.ones((1, 1), dtype='<f4'), zeros, zeros, 768)

    def fixture(self, root):
        sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in p.SOURCE_NAMES}
        with zipfile.ZipFile(root / 'source.zip', 'w') as archive:
            for name in sources:
                archive.writestr(name, name.encode())
        p.write_new(root / 'plan.json', {'synthetic': True})
        report = {'compiled_source_inventory_sha256': p.canonical_sha(sources),
                  'compiled_plan_sha256': p.sha(root / 'plan.json'), 'timing': None, 'threads': 1,
                  'avx2_fma_available': True, 'output_channel_sampling': False}
        (root / 'outputs').mkdir()
        p.write_new(root / 'outputs/operators.json', report)
        for name in ['probe.exe', 'tests.exe', 'jobs.tsv', 'tests.log', 'build.log', 'probe.log']:
            (root / name).write_bytes(name.encode())
        files = {f.relative_to(root).as_posix(): p.sha(f) for f in root.rglob('*') if f.is_file()}
        build = {'status': 'complete', 'source_unchanged_during_build_and_run': True,
                 'source_sha256': sources, 'artifact_sha256': files, 'binary': 'probe.exe',
                 'test_binary': 'tests.exe', 'unit_tests_passed': 9}
        p.write_new(root / 'build.json', build)
        return build

    def test_captured_binary_plan_source_and_output_changes_are_rejected(self):
        for name in ['probe.exe', 'plan.json', 'source.zip', 'outputs/operators.json']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                self.fixture(root)
                p.validate_capture(root)
                path = root / name; path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaises(ValueError):
                    p.validate_capture(root)

    def test_omitted_identity_or_fake_embedded_plan_is_rejected(self):
        for mutation in ['missing_binary', 'embedded_plan', 'source_inventory']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                build = self.fixture(root)
                if mutation == 'missing_binary':
                    del build['artifact_sha256']['probe.exe']
                elif mutation == 'source_inventory':
                    del build['source_sha256'][p.SOURCE_NAMES[0]]
                else:
                    path = root / 'outputs/operators.json'; report = p.load(path)
                    report['compiled_plan_sha256'] = '0' * 64
                    path.write_text(json.dumps(report))
                    build['artifact_sha256']['outputs/operators.json'] = p.sha(path)
                (root / 'build.json').write_text(json.dumps(build))
                with self.assertRaises(ValueError):
                    p.validate_capture(root)


if __name__ == '__main__':
    unittest.main()
