"""Small prospective contract checks only; never load TraceEvent or ETLX."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/profiling/CompactIpHistogram.cs"
RECEIPT = ROOT / "artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/symbol-receipt.json"


class SourceContract(unittest.TestCase):
    def test_extent_joins_closed_symbol_receipt(self):
        raw = RECEIPT.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "ea4842b0289c971a72f6e22bb0d5d10b623743376685159e34ef894444f3acf3")
        r = json.loads(raw)
        src = SOURCE.read_text(encoding="utf-8")
        base, start, end = (int(x, 16) for x in re.search(r"PreferredImageBase = (0x[0-9a-f]+), HeadStart = (0x[0-9a-f]+), HeadEnd = (0x[0-9a-f]+)", src).groups())
        self.assertEqual(base + start, int(r["virtual_address"], 16))
        self.assertEqual(base + end, int(r["stop_address"], 16))
        self.assertEqual(end - start, r["code_bytes"])
        self.assertTrue(r["source_and_input_closure"])

    def test_diagnostic_cannot_claim_successful_profile(self):
        src = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"diagnostic_complete_strict_profile_failed"', src)
        self.assertIn('"strict_profile_verdict_unchanged", true', src)
        self.assertNotIn('"audit_passed"', src)
        self.assertIn('report["status"] = "failed";', src)

    def test_no_conversion_or_symbol_lookup_api(self):
        src = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("CreateFromEventTraceLogFile", "ETWTraceEventSource", "SymbolReader", "LookupSymbolsForModule", "Process.Start"):
            self.assertNotIn(forbidden, src)
        self.assertIn("new TraceLog(etlx)", src)
        self.assertIn("FileShare.Read", src)

    def test_expected_accounting_and_bounds_guards_exist(self):
        src = SOURCE.read_text(encoding="utf-8")
        for required in ('"count_payload_total"', '"out_of_bounds_count"', '"wrong_or_missing_process_index_count"', '"dpc_count"', '"isr_count"', '"non_process_count"', '"with_attached_stack"', "depth <= 256", "watch.Elapsed.TotalSeconds <= 600", "examined <= 20000000"):
            self.assertIn(required, src)


if __name__ == "__main__":
    unittest.main()
