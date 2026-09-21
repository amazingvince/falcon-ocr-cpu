import unittest
from pathlib import Path
from patch import ROOT, make_patch, DISPATCH, START, END

HERE = Path(__file__).resolve().parent

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.original = (ROOT / 'src/kernels.rs').read_text(encoding='utf-8')
        self.template = (HERE / 'candidate.rs').read_text(encoding='utf-8')
        self.tests = (HERE / 'tests.rs').read_text(encoding='utf-8')

    def test_only_dispatch_and_appended_modules_change(self):
        patched, _, original_fn, _ = make_patch(self.original,self.template,self.tests)
        start = self.original.index(START); end = self.original.index(END,start)
        self.assertEqual(original_fn,self.original[start:end])
        self.assertTrue(patched.startswith(self.original[:start]))
        unchanged_prefix = patched[:patched.index('\n#[cfg(target_arch = "x86_64")]\nmod attention64_candidate')]
        self.assertEqual(unchanged_prefix.replace(DISPATCH,''),self.original)

    def test_head_body_is_original_except_two_direct_calls(self):
        _,candidate,original_fn,body=make_patch(self.original,self.template,self.tests)
        restored=body.replace('dot64(qvec,','dot(qvec,').replace('axpy64(probability,','axpy(probability,')
        self.assertIn(restored,original_fn)
        self.assertIn(body,candidate)
        self.assertEqual(body.count('dot64('),1)
        self.assertEqual(body.count('axpy64('),1)
        self.assertNotIn('dot_kernel',candidate)
        self.assertNotIn('axpy_kernel',candidate)

    def test_changed_or_ambiguous_source_rejected(self):
        for original in [self.original.replace('let query = qh / n_heads;', 'let query = 0;'),
                         self.original+self.original]:
            with self.assertRaises(ValueError):make_patch(original,self.template,self.tests)
        with self.assertRaises(ValueError):make_patch(self.original,self.template+'\n        // GENERATED_ORIGINAL_HEAD_BODY',self.tests)

    def test_six_focused_tests_and_no_model_run(self):
        self.assertEqual(self.tests.count('#[test]'),6)
        self.assertNotIn('Model::load',self.tests)
        self.assertIn('16384',self.tests)
        self.assertIn('query_len == 1 && head_dim == 64 && selected == Simd::Avx2',DISPATCH)

if __name__=='__main__':unittest.main()
