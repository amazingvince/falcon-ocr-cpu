# Retained compact-head instruction-address diagnostic

This is a prospective offline tool, not another profile capture. The source
targets only compact profile PID79568/process-index557, its pinned failed audit,
and the closed combined-attention PDB/disassembly extent. It does not loosen
the lifetime gate or promote the trace to a complete-profile pass.

`CompactIpHistogram.cs` scans the retained ETLX once, without converting ETL,
loading symbols or executing the model. The original audit's exact target
command/lifetime and loaded image path/base/index/size/lifetime/PDB GUID+age must
match. The function is RVA `[0xb13f0,0xb1d36)`, derived from preferred-base
`0x140000000` and the pinned symbol receipt. ASLR is handled with the recorded
runtime module base. The on-disk large PDB and ETLX are not rehashed: PDB identity
comes from the closed receipt and trace, ETLX digest from the audit. ETLX stays
read-locked; size/write-time must remain unchanged. These limitations are output
fields, not implicit claims of new byte verification.

All sampled-profile events with the target PID are classified as wrong owner,
outside process/session lifetime, or in bounds, in the same precedence as the
audit. Each class retains event/count-payload totals and separate DPC/ISR/
NonProcess counts. Flags are never used to exclude observations. Every existing
audit count must reconcile exactly, including the one excluded boundary sample.
Exceptional events retain their own identities. The scan has a20M-event/600s
bound checked every16,384 events and a256-frame per-stack cap. Exceeding a bound
fails the diagnostic; it never emits a passing partial histogram.

The primary histogram is each in-bounds sampled leaf instruction-pointer RVA
inside compact_head. A disjoint secondary population has its leaf elsewhere:
the existing `GetCallStackForEvent` / `CodeAddress` APIs collect saved head-frame
RVAs with matching module identity, if present. These are **unadjusted stack code
addresses**, not yet proved to be call instructions or expf call sites. The tool
does not subtract one, map to a nearest instruction, or infer a callee. It reports
missing stacks and the number of samples with multiple distinct head-frame RVAs;
each sample contributes once per distinct RVA. Flagged histogram subsets are
reported separately. Histograms retain raw addresses even if a later disassembly
join cannot annotate them. Sample counts are not retired instructions, stall
cycles, cache/TLB measurements, bandwidth or elapsed wall time.

## Preparation checks and held commands

Only the small Python source-contract checks and PowerShell syntax parsing are
authorized before parent review. They do not compile C# or validate the live
TraceEvent API; the14 pure C# host checks run after an approved build. Build and
ETLX execution remain separate actions. Do not start the scan during timing.

```powershell
& 'C:/Users/amazi/mambaforge/python.exe' -m unittest discover -s experiments/profiling -p test_compact_ip_histogram_source.py -v

# After source review/release; fresh build output required.
& 'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe' -NoProfile -ExecutionPolicy Bypass -File research/benchmarks/experiments/profiling/build_compact_ip_histogram.ps1 -OutputDirectory artifacts/tools/compact-ip-histogram-build-v1

# After build/self-tests pass and parent releases the retained-ETLX scan.
$histogramBuild = 'C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/tools/compact-ip-histogram-build-v1/build.json'
$histogramBuildSha = (Get-FileHash -LiteralPath $histogramBuild -Algorithm SHA256).Hash.ToLowerInvariant()
& 'C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/tools/compact-ip-histogram-build-v1/CompactIpHistogram.exe' --audit 'D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/trace-audit-v4.json' --receipt 'C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/symbol-receipt.json' --disassembly 'C:/Users/amazi/Documents/ChatGPT/falcon-ocr/artifacts/diagnostics/attention64-compact-v1/benchmark-build/compiled-compact-head/head-disassembly.txt' --build $histogramBuild --build-sha256 $histogramBuildSha --output 'D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/compact-ip-histogram-v1.json'
```

The runtime binds exact CLI arguments, archived source, executing binary, build,
audit, symbol receipt, disassembly and observed TraceEvent dependencies before
and after. Output uses CreateNew and preserves failures after receipt creation.
An early invalid CLI/audit/dependency load returns a nonzero tool error before
receipt creation. Successful diagnostic status explicitly says the strict
profile still failed. Preserve stdout/stderr and the build receipt alongside the
fresh report. No function subrange labels or causal performance conclusions are
produced automatically; manual QK/PV/rescale/call-site attribution follows only
after the complete count and source windows close.
