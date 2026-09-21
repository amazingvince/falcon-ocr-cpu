# Native single-page CPU profiling

This isolated diagnostic uses the existing `ocr_bench` v2 executable and matching
PDB. No production change or rebuild is involved. The input is the frozen journal
page `3f294b5e60a0c2d4`: prepared 1088 x 1536, 6,544 prefix tokens, cap 4,096.
The fixed invocation uses FP32/AVX2, 16 threads, B1, expanded caches and unpacked
weights. The frozen harness requires at least three repetitions, so this capture
uses zero warmups and three full recognitions. Each must reproduce the saved
1,140-token EOS output and literal text. Profiled durations are diagnostic only.

Prepare without recording or running the model:

```powershell
& 'C:\Users\amazi\mambaforge\python.exe' experiments/profiling/capture_windows.py prepare --output 'D:\falcon-ocr-rust-builds\profiles\fullpage-expanded-v2'
```

Preparation prints the exact `plan.json` SHA256. Review that plan and its preserved
source ZIP before invoking the following with the printed digest:

```powershell
& 'C:\Users\amazi\mambaforge\python.exe' experiments/profiling/capture_windows.py run --plan 'D:\falcon-ocr-rust-builds\profiles\fullpage-expanded-v2\plan.json' --expected-plan-sha256 '<reviewed digest>'
```

The current ordinary native token has no `SeSystemProfilePrivilege`. The driver
does not request elevation, change privileges, install software or change the
system profiling interval. A separately reviewed, hidden native UAC launch is
needed for recording; the driver checks the actual token before any WPR start.

The plan contains the exact WPR commands. Every session command ends with the
same unpredictable `-instancename FalconOcrCpu-<UUID>`. The name is checked unused
before start. Normal completion and Python exception handling call `-stop` only
on that instance; a failed save may use `-cancel` only on the same owned instance.
No default-session stop/cancel exists. External termination or power loss cannot
execute `finally`; the preserved plan contains scoped recovery commands.
Microsoft documents the [instance-name rule](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options#instancename).

The custom profile collects process/thread/image information and sampled CPU
stacks, with 64 x 1,024 KiB buffers. Its
[sequential 2,048 MiB file cap](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/maximumfilesize)
stops collection instead of replacing old events. A 600-second model watchdog,
4 GiB folder watchdog and free-disk guards provide additional bounds. The output
volume needs 8 GiB free initially and at least 4 GiB during capture. A cap,
timeout, missing process end or incomplete sample coverage is a failed complete
profile, even if some stacks are useful diagnostically. The merged ETL and
analysis exports need space beyond the one collector-file cap.

After capture, use the matching local PDB directory in `_NT_SYMBOL_PATH` and the
plan's `perfview_export_template` / `perfview_stacks_template` commands. In the
pinned [PerfView v3.2.6 implementation](https://github.com/microsoft/perfview/blob/v3.2.6/src/PerfView/UserCommands.cs),
`SaveCPUStacksAsCsv cpu.etl ocr_bench 10 LastProcess` selects the last matching
process name, and `SaveCPUStacks cpu.etl ocr_bench` preserves full sampled stacks.
Neither command alone proves it selected our PID. Keep exports and logs local;
do not upload the system-wide ETL.

`capture.json` deliberately stops at `captured_pending_..._analysis`. The separate
offline TraceEvent audit must bind the ETL to the recorded PID, exact image and
command, unique matching lifetime, actual process start/end, zero loss and full
sample/stack coverage. The stack analysis must show resolved project functions
from the matching PDB and report unresolved project/system frames separately.
Do not silently discard missing stacks or symbolize them as known functions.
The frozen executable has no per-phase ETW markers: classify actual call paths
where available, but do not invent exact phase boundaries from aggregate timers.

The driver source, profile, model/input/build/PDB/tool identities and baseline are
checked before recording and after cleanup. No large hashes run inside the model
interval. Raw benchmark output, process identity, WPR status/commands/logs and ETL
remain preserved on failures. A successful capture is not a speedup measurement,
corpus-quality gate or numerical qualification.
