"""Source-only patch tests; no Rust build, inference, model loading or timing."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import capture_temporal_candidate as capture

class TemporalCopyTests(unittest.TestCase):
    def test_decode_arithmetic_body_is_exact_compact_copy(self):
        source=(capture.ROOT/"src/kernels.rs").read_text(encoding="utf-8")
        adapter,_=capture.attention_adapter(source)
        old=source[source.index("pub fn attention_compact_with_simd("):source.index("\n#[allow(clippy::too_many_arguments)]\nfn attention_gemm_compact(")]
        body=adapter[adapter.index("    let dot = dot_kernel(selected);"):]
        body=body.replace("            let mut reconstructed = [0.0_f32; 64];\n","")
        body=body.replace("                        let temporal = (key * 8 + kv_head) * 32;\n                        let spatial = (key * 16 + head) * 32;\n                        reconstructed[..32].copy_from_slice(&temporal_k[temporal..temporal+32]);\n                        reconstructed[32..].copy_from_slice(&spatial_k[spatial..spatial+32]);\n                        &reconstructed[..]",
            "                        let begin = key * query_width + head * head_dim;\n                        &prefix_k[begin..begin + head_dim]")
        self.assertEqual(body,old[old.index("    let dot = dot_kernel(selected);"):])
        self.assertNotIn("attention_gemm_compact(",adapter)
        self.assertIn("assert_eq!(query_len,1",adapter)

    def test_copy_patch_scope_and_default(self):
        files=["src/kernels.rs","src/config.rs","src/lib.rs","src/model.rs"]
        live={n:capture.sha(capture.ROOT/n) for n in files}
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);(project/"src").mkdir()
            for n in files:shutil.copyfile(capture.ROOT/n,project/n)
            diff,body_sha=capture.patch_copy(project)
            self.assertEqual(len(body_sha),64)
            self.assertIn("TemporalCandidate",diff)
            kernels=(project/"src/kernels.rs").read_text(encoding="utf-8")
            self.assertTrue(kernels.startswith((capture.ROOT/"src/kernels.rs").read_text(encoding="utf-8")))
            config=(project/"src/config.rs").read_text(encoding="utf-8")
            self.assertIn("    #[default]\n    Expanded,\n    Compact,",config)
            model=(project/"src/model.rs").read_text(encoding="utf-8")
            self.assertEqual(model.count("cache.append("),3) # one match arm and two original callers
            self.assertEqual(model.count("current_expanded_k"),2)
            self.assertIn("&work.k[range.clone()],\n                    1,",model)
            self.assertIn("cache.append(&work.k, &work.v, offset, c)?;",model)
        self.assertEqual(live,{n:capture.sha(capture.ROOT/n) for n in files})

    def test_missing_or_duplicated_source_anchor_rejected(self):
        with self.assertRaises(ValueError):capture.replace_once("","not here","replacement")
        with self.assertRaises(ValueError):capture.replace_once("same same","same","replacement")
        self.assertEqual(capture.sha(capture.ROOT/"src/kernels.rs"),capture.KERNEL_SHA)
        self.assertTrue(all(capture.sha(capture.ROOT/n)==h for n,h in capture.PINS.items()))

if __name__=="__main__":unittest.main()
