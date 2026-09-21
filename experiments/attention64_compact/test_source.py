import unittest
from pathlib import Path
from patch import ROOT,OLD,make_patch,expanded,START,END,DISPATCH

HERE=Path(__file__).resolve().parent
class SourceTests(unittest.TestCase):
    def setUp(self):
        self.original=(ROOT/'src/kernels.rs').read_text()
        self.template=(HERE/'candidate.rs').read_text();self.tests=(HERE/'tests.rs').read_text()
    def test_only_two_dispatches_and_appended_modules_change(self):
        patched,_,_,_=make_patch(self.original,self.template,self.tests)
        prefix=patched[:patched.index('\n#[cfg(target_arch = "x86_64")]\nmod attention64_candidate')]
        self.assertEqual(prefix.replace(DISPATCH,'').replace(expanded.DISPATCH,''),self.original)
    def test_compact_body_and_shared_helpers_unchanged(self):
        _,candidate,function,body=make_patch(self.original,self.template,self.tests)
        self.assertIn(body.replace('dot64(qvec,','dot(qvec,').replace('axpy64(probability,','axpy(probability,'),function)
        old=(OLD/'candidate.rs').read_text()
        first=old.index('// These helpers');last=old.index('#[cfg(test)]',first)
        self.assertIn(old[first:last],candidate)
        self.assertIn('let repeat=n_heads/n_kv_heads;',candidate)
    def test_anchor_change_rejected(self):
        with self.assertRaises(ValueError):make_patch(self.original.replace(START,START.replace('attention_compact_with_simd','other_attention')),self.template,self.tests)
        with self.assertRaises(ValueError):make_patch(self.original+self.original,self.template,self.tests)
    def test_eleven_operator_cases_no_model(self):
        self.assertEqual((OLD/'tests.rs').read_text().count('#[test]')+self.tests.count('#[test]'),11)
        self.assertNotIn('Model::load',self.tests)
        for marker in ['[127,128,129]','(16384,6544)','[4,8,16]']:
            self.assertIn(marker,self.tests)
if __name__=='__main__':unittest.main()
