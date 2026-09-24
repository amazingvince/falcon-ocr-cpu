"""Offline checks for artifact math, comparison policy and source wiring.
These do not compile or execute Rust; native operator tests are in src/attempt/.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import bench
import convert_w8

ROOT = Path(__file__).resolve().parents[1]

class EncodingTests(unittest.TestCase):
    def test_zero_ties_and_endpoints(self):
        w = np.zeros((1,128), dtype=np.float32)
        w[0,:8] = [254,-254,1,3,5,-1,-3,-5]
        q,s = convert_w8.encode(w)
        np.testing.assert_array_equal(q[0,:8], [127,-127,0,2,2,0,-2,-2])
        np.testing.assert_array_equal(s, [[2,0]])

    def test_partial_groups_and_error_bounds(self):
        rng = np.random.default_rng(918)
        for k in [1,31,63,64,65,127,128,129,768,1024,2304]:
            w = rng.normal(size=(7,k)).astype(np.float32)
            q,s = convert_w8.encode(w)
            restored = q.astype(np.float32)*np.repeat(s,64,axis=1)[:,:k]
            bound = np.repeat(s,64,axis=1)[:,:k]/2+2*np.finfo(np.float32).eps*np.abs(w)
            self.assertTrue(np.all(np.abs(w-restored)<=bound))
            self.assertEqual(q.nbytes+s.nbytes,7*k+7*((k+63)//64)*4)

    def test_matches_scalar_formula(self):
        rng=np.random.default_rng(555)
        w=rng.normal(size=(3,129)).astype(np.float32)
        q,s=convert_w8.encode(w)
        for row in range(3):
            for start in range(0,129,64):
                block=w[row,start:start+64]
                scale=np.float32(float(max(abs(float(v)) for v in block))/127.0)
                self.assertEqual(scale.view(np.uint32),s[row,start//64].view(np.uint32))
                for j,v in enumerate(block):
                    self.assertEqual(int(q[row,start+j]), max(-127,min(127,round(float(v)/float(scale)))))

    def test_reject_nonfinite_empty_group(self):
        for v in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError):convert_w8.encode(np.array([[v]],np.float32))
        with self.assertRaises(ValueError):convert_w8.encode(np.zeros((0,1),np.float32))
        with self.assertRaises(ValueError):convert_w8.encode(np.zeros((1,1),np.float32),128)

    def test_subnormal(self):
        tiny=np.nextafter(np.float32(0),np.float32(1))
        q,s=convert_w8.encode(np.array([[tiny,-tiny]],np.float32))
        np.testing.assert_array_equal(q,[[1,-1]])
        self.assertEqual(s[0,0],tiny)

    def test_safetensors_roundtrip(self):
        q,s=convert_w8.encode(np.arange(130,dtype=np.float32).reshape(2,65))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"test.safetensors"
            save_file({"layer.__w8_codes":q,"layer.__w8_scales":s},p,metadata={"format":convert_w8.FORMAT})
            with safe_open(p,framework="numpy") as f:
                self.assertEqual(f.metadata()["format"],convert_w8.FORMAT)
                np.testing.assert_array_equal(f.get_tensor("layer.__w8_codes"),q)
                np.testing.assert_array_equal(f.get_tensor("layer.__w8_scales"),s)

    def test_inventory_excludes_protected_tensors(self):
        self.assertEqual(len(convert_w8.inventory(False)),88)
        self.assertEqual(len(convert_w8.inventory(True)),89)
        for name in ("img_projector.weight","tok_embeddings.weight","norm.weight","freqs_cis_golden"):
            self.assertNotIn(name,convert_w8.inventory(True))

class AttentionMathTests(unittest.TestCase):
    def test_split_cache_and_single_sink(self):
        rng=np.random.default_rng(839)
        for length in (1,127,128,129,257):
            raw=rng.normal(size=(length,8,64)).astype(np.float32)*.1
            expanded=np.repeat(raw,2,axis=1)
            expanded[:,:,32:]+=rng.normal(size=(length,16,32)).astype(np.float32)*.01
            temporal=expanded[:,::2,:32].copy();spatial=expanded[:,:,32:].copy()
            reconstructed=np.concatenate((np.repeat(temporal,2,axis=1),spatial),axis=2)
            np.testing.assert_array_equal(reconstructed,expanded)
            values=rng.normal(size=(length,8,64)).astype(np.float32)
            q=rng.normal(size=(16,64)).astype(np.float32)
            for h in (0,1,15):
                logits=expanded[:,h,:].astype(np.float64)@q[h].astype(np.float64)/8
                sink=.7;max_all=max(float(logits.max()),sink)
                expected=(np.exp(logits-max_all)@values[:,h//2].astype(np.float64))/(np.exp(logits-max_all).sum()+np.exp(sink-max_all))
                maximum=-np.inf;den=0.;out=np.zeros(64)
                for start in range(0,length,128):
                    scores=logits[start:start+128];m=max(maximum,float(scores.max()))
                    scale=0. if not np.isfinite(maximum) else np.exp(maximum-m)
                    probs=np.exp(scores-m);out=out*scale+probs@values[start:start+128,h//2].astype(np.float64)
                    den=den*scale+probs.sum();maximum=m
                lse=maximum+np.log(den)
                actual=(out/den)/(1+np.exp(sink-lse))
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_payload_accounting(self):
        p=6544;layers=22
        elements=layers*p*(8*32+16*32+8*64)
        self.assertEqual(elements,184279040)
        self.assertEqual(elements*2,368558080)
        self.assertEqual(elements+4*(elements//32),207313920)

class ComparisonTests(unittest.TestCase):
    def report(self,profile="reference",time_ms=100.0):
        out={"width":1088,"height":1536,"input_tokens":6544,"output_tokens":3,"finish_reason":"eos","token_ids":[1,2,11],"text":"total 12.3",
             "timings":{"prefill_ms":10.0,"decode_ms":20.0,"total_ms":30.0}}
        return {"schema":"falcon-ocr-attempt3-report-v1","profile":profile,"weights_sha256":"weights","model_revision":"v15",
                "threads":16,"backend":"auto","batch_size":1,"options":{"max_new_tokens":4096},
                "inputs":[{"sha256":"image"}],"binary_sha256":"binary",
                "samples":[{"wall_ms":time_ms,"outputs":[copy.deepcopy(out)],"telemetry":{"active_rows_histogram":[0,2,0,0,0,0,0,0,0]}} for _ in range(3)]}

    def test_exact_complete_comparison(self):
        r=bench.compare(self.report(),self.report("hygiene",80),self.report(time_ms=102))
        self.assertTrue(r["same_output_complete_page_comparison"])
        self.assertFalse(r["quality_qualified"])
        self.assertAlmostEqual(r["raw_speedup"],101/80)

    def test_changed_text_never_promoted(self):
        c=self.report("w8-body",50)
        for s in c["samples"]:s["outputs"][0]["text"]="total 123";s["outputs"][0]["token_ids"]=[1,3,11]
        r=bench.compare(self.report(),c,self.report())
        self.assertFalse(r["same_output_complete_page_comparison"])
        self.assertFalse(r["pages"][0]["numeric_string_sequence_equal"])

    def test_truncated_control_not_complete_page_evidence(self):
        a=self.report();b=self.report("hygiene")
        for r in (a,b):
            for s in r["samples"]:s["outputs"][0]["finish_reason"]="length"
        self.assertFalse(bench.compare(a,b,a)["same_output_complete_page_comparison"])

    def test_reject_resolution_mismatch(self):
        c=self.report("kv-bf16")
        for s in c["samples"]:s["outputs"][0]["height"]=1024
        with self.assertRaises(ValueError):bench.compare(self.report(),c,self.report())

    def test_reject_binary_mismatch(self):
        c=self.report("hygiene");c["binary_sha256"]="other"
        with self.assertRaises(ValueError):bench.compare(self.report(),c,self.report())

    def test_drift_disqualifies(self):
        self.assertFalse(bench.compare(self.report(),self.report("hygiene",50),self.report(time_ms=110))["same_output_complete_page_comparison"])

    def test_nondeterministic_samples_fail(self):
        c=self.report("hygiene");c["samples"][1]["outputs"][0]["text"]="different"
        with self.assertRaises(ValueError):bench.compare(self.report(),c,self.report())

    def test_optional_cer_not_silently_quality_gate(self):
        r=bench.compare(self.report(),self.report("w8-body"),self.report(),["total 12.4"])
        self.assertAlmostEqual(r["pages"][0]["candidate_cer"],.1)
        self.assertFalse(r["quality_qualified"])

    def test_edit_distance_exact_and_bounded(self):
        for a,b,expected in [("kitten","sitting",3),("x","",1),("abc","abc",0),("abcdef","abqdef",1)]:
            self.assertEqual(bench.edit_distance(a,b),expected)
        self.assertIsNone(bench.edit_distance("aaa","bbb",1))

    def test_bit_parallel_levenshtein_matches_table(self):
        import random
        def table(a,b):
            prev=list(range(len(b)+1))
            for i,x in enumerate(a,1):
                row=[i]
                for j,y in enumerate(b,1):
                    row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
                prev=row
            return prev[-1]
        rng=random.Random(7)
        for _ in range(400):
            a=[rng.randrange(4) for _ in range(rng.randrange(0,90))]
            b=[rng.randrange(4) for _ in range(rng.randrange(0,90))]
            self.assertEqual(bench.levenshtein(a,b),table(a,b))
            self.assertEqual(bench.levenshtein("".join(map(str,a)),"".join(map(str,b))),table(a,b))

    def test_divergence_metrics_for_changed_output(self):
        c=self.report("w8-body")
        for sample in c["samples"]:
            sample["outputs"][0].update({"token_ids":[1,5,11],"text":"total 15.3"})
        page=bench.compare(self.report(),c,self.report())["pages"][0]
        self.assertEqual((page["first_divergence_token"],page["token_edit_distance"],page["text_edit_distance"]),(1,1,1))
        self.assertIsNone(bench.first_divergence([1,2],[1,2]))
        self.assertEqual(bench.first_divergence([1,2],[1,2,3]),2)

    def test_schedules_share_or_repeat_controls(self):
        arms,triples=bench.schedule(["a","b","c"],"interleaved",2)
        self.assertEqual(arms,["reference","a","b","reference","c","reference"])
        self.assertEqual(triples,[(0,1,3),(0,2,3),(3,4,5)])
        arms,triples=bench.schedule(["reference","b"],"interleaved",2)
        self.assertEqual(arms,["reference","reference","b","reference"])
        self.assertEqual(triples,[(0,1,3),(0,2,3)])
        arms,triples=bench.schedule(["a","b"],"bracket",2)
        self.assertEqual(arms,["reference","a","reference","reference","b","reference"])
        self.assertEqual(triples,[(0,1,2),(3,4,5)])

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"a.png").write_bytes(b"placeholder")
            manifest={"schema":"falcon-ocr-attempt3-cases-v1","cases":[{"id":"test","images":["a.png"]}]}
            (p/"cases.json").write_text(json.dumps(manifest))
            self.assertEqual(bench.load_manifest(p/"cases.json")[0]["batch_size"],1)
            manifest["cases"]*=2;(p/"cases.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):bench.load_manifest(p/"cases.json")

class HarnessWorkflowTests(unittest.TestCase):
    def test_bracket_driver_records_all_arms_and_routes_overlay(self):
        import subprocess
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            binary=root/"native engine.exe";binary.write_bytes(b"not executed: mock fixture")
            image=root/"page one.png";image.write_bytes(b"not decoded: mock fixture")
            overlay=root/"body.w8.safetensors";overlay.write_bytes(b"not imported: mock fixture")
            manifest=root/"cases.json"
            manifest.write_text(json.dumps({"schema":"falcon-ocr-attempt3-cases-v1","cases":[
                {"id":"journal","images":[image.name]}]}))
            output=root/"reports"
            modes=[]
            def native(command, **kwargs):
                mode=command[command.index("--profile")+1];modes.append(mode)
                report=ComparisonTests().report(mode, 80 if mode.startswith("w8") else 100)
                self.assertEqual(command[command.index("bench")+1],str(image.resolve()))
                self.assertEqual("--w8-artifact" in command,mode.startswith("w8"))
                Path(command[command.index("--report")+1]).write_text(json.dumps(report))
                return subprocess.CompletedProcess(command,0,"fixture-only stdout","fixture-only stderr")
            argv=["bench.py","--binary",str(binary),"--manifest",str(manifest),
                  "--profiles","w8-body","--w8-body-artifact",str(overlay),"--output",str(output)]
            with mock.patch("sys.argv",argv),mock.patch.object(bench.subprocess,"run",side_effect=native):
                bench.main()
            self.assertEqual(modes,["reference","w8-body","reference"])
            summary=json.loads((output/"summary.json").read_text())
            self.assertFalse(summary["quality_qualified"])
            self.assertEqual(len(summary["comparisons"]),1)
            self.assertEqual(len(list(output.rglob("*.command.json"))),3)

    def test_invalid_manifest_never_launches_native_process(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);binary=root/"native";binary.write_bytes(b"fixture")
            manifest=root/"bad.json";manifest.write_text('{"schema":"wrong"}')
            with mock.patch("sys.argv",["bench.py","--binary",str(binary),"--manifest",str(manifest),
                 "--output",str(root/"out")]),mock.patch.object(bench.subprocess,"run") as run:
                with self.assertRaises(ValueError):bench.main()
                run.assert_not_called()

class WiringTests(unittest.TestCase):
    def test_no_silent_switch_to_original_weights(self):
        model=(ROOT/"src/model/mod.rs").read_text()
        block=model[model.index("pub(crate) fn forward"):model.index("fn decode_linear")]
        # W13 routes through linear_glu, which itself dispatches via self.linear.
        # 11 = forward_layers' prefill/decode projections, the forward_next and
        # verify_next heads, and the batched step.
        self.assertEqual(block.count("self.linear(")+block.count("self.linear_glu("),11)
        glu=model[model.index("fn linear_glu("):model.index("fn w(&self")]
        self.assertEqual(glu.count("self.linear("),1)
        self.assertNotIn("self.w(&layer.qkv)",block)
        self.assertIn("scratch.dense",(ROOT/"src/quant/linear.rs").read_text())

    def test_reference_loader_still_default(self):
        self.assertIn("profile: crate::quant::Profile::REFERENCE",(ROOT/"src/model/load.rs").read_text())
        auto=(ROOT/"src/auto.rs").read_text()
        # Exact mode with the reference profile goes through the pinned loader.
        self.assertIn("plan.profile == Profile::REFERENCE",auto)
        self.assertIn("Model::load(dir)",auto)
        self.assertIn("Self::Exact => Profile::REFERENCE",auto)

    def test_cache_seal_and_retirement_wired(self):
        runner=(ROOT/"src/runner.rs").read_text()
        self.assertEqual(runner.count("self.seal_session(&mut session, trace)?"),2)
        self.assertIn("trace.cache_retired(sessions[index].retire_cache())",runner)
        self.assertIn("session.prepare_small_decode(c)",runner)
        self.assertIn("drop(prepared.patches)",runner)

    def test_source_model_identity_not_relaxed(self):
        s=(ROOT/"src/model/load.rs").read_text()
        self.assertIn('hash == WEIGHTS_SHA256',s)
        self.assertIn('"config.json SHA-256 mismatch"',s)
        self.assertIn('"phase-packed FP32 weights cannot override W8 numerical weights"',(ROOT/"src/model/mod.rs").read_text())

if __name__=="__main__":unittest.main()
