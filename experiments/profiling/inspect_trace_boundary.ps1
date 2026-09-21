[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Audit,
    [Parameter(Mandatory=$true)][string]$AuditSha256,
    [Parameter(Mandatory=$true)][string]$Output
)
# Bounded retained-ETLX lookup only. No ETL conversion, symbols, capture, or model.
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Fresh output required' }
$auditBytes = [IO.File]::ReadAllBytes($Audit)
$shaTool = [Security.Cryptography.SHA256]::Create()
try { $before = [BitConverter]::ToString($shaTool.ComputeHash($auditBytes)).Replace('-','').ToLowerInvariant() } finally { $shaTool.Dispose() }
if ($before -ne $AuditSha256) { throw 'Audit identity differs' }
$meta = [Text.Encoding]::UTF8.GetString($auditBytes) | ConvertFrom-Json
$libraryDirectory = 'C:\Users\amazi\AppData\Roaming\PerfView\VER.2026-09-20.08.05.04.237'
$libraryPath = Join-Path $libraryDirectory 'Microsoft.Diagnostics.Tracing.TraceEvent.dll'
if ((Get-FileHash -LiteralPath $libraryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne '530946dc20e89754783f0ac76d86b7a4eedf95326f255db34c32fcbf5c3ce0ff') { throw 'Wrong TraceEvent library' }
[Reflection.Assembly]::LoadFrom((Join-Path $libraryDirectory 'Microsoft.Diagnostics.FastSerialization.dll')) | Out-Null
[Reflection.Assembly]::LoadFrom($libraryPath) | Out-Null
$retainedPath = $meta.converted_etlx.path
$retainedBefore = Get-Item -LiteralPath $retainedPath
if ($retainedBefore.Length -ne $meta.converted_etlx.bytes) { throw 'Retained ETLX length differs' }
$retainedLock = [IO.File]::Open($retainedPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$trace = $null
try {
    $trace = [Microsoft.Diagnostics.Tracing.Etlx.TraceLog]::new($retainedPath)
    $matches = @($trace.Processes | Where-Object {$_.ProcessID -eq $meta.target_pid})
    if ($matches.Count -ne 1 -or [int]$matches[0].ProcessIndex -ne $meta.target.process_index) { throw 'Process identity differs' }
    $target = $matches[0]
    $windows = @(
        [ordered]@{name='start';start_ms=($target.StartTimeRelativeMsec-5);end_ms=($target.StartTimeRelativeMsec+1)},
        [ordered]@{name='stop';start_ms=($target.EndTimeRelativeMsec-1);end_ms=($target.EndTimeRelativeMsec+5)}
    )
    $records = [Collections.Generic.List[object]]::new()
    $examined = 0
    foreach ($window in $windows) {
        foreach ($eventRecord in $trace.Events.FilterByTime([double]$window.start_ms,[double]$window.end_ms)) {
            $examined++
            if ($examined -gt 10000) { throw 'Bounded event lookup exceeded 10000 records' }
            if ($eventRecord.ProcessID -ne $target.ProcessID) { continue }
            $owner = [Microsoft.Diagnostics.Tracing.Etlx.TraceLogExtensions]::Process($eventRecord)
            $relative = $eventRecord.TimeStampRelativeMSec
            $sample = $eventRecord -is [Microsoft.Diagnostics.Tracing.Parsers.Kernel.SampledProfileTraceData]
            $outside = $relative -lt 0 -or $relative -gt $trace.SessionEndTimeRelativeMSec -or $relative -lt $target.StartTimeRelativeMsec -or $relative -gt $target.EndTimeRelativeMsec
            $item = [ordered]@{
                window=$window.name;type=$eventRecord.GetType().FullName;event_name=$eventRecord.EventName;event_index=[uint32]$eventRecord.EventIndex
                relative_ms=$relative;timestamp_utc=$eventRecord.TimeStamp.ToUniversalTime().ToString('O');qpc=$eventRecord.TimeStampQPC
                pid=$eventRecord.ProcessID;thread_id=$eventRecord.ThreadID;owner_process_index=$(if($owner){[int]$owner.ProcessIndex}else{$null})
                sampled_profile=$sample;out_of_bounds=$outside;delta_from_process_start_ms=($relative-$target.StartTimeRelativeMsec);delta_from_process_end_ms=($relative-$target.EndTimeRelativeMsec)
            }
            if ($sample) {
                $item.stack_index=[int]$trace.GetCallStackIndexForEvent($eventRecord)
                $item.instruction_pointer_hex='0x'+$eventRecord.InstructionPointer.ToString('x')
                $item.processor_number=$eventRecord.ProcessorNumber
                $item.executing_dpc=$eventRecord.ExecutingDPC;$item.executing_isr=$eventRecord.ExecutingISR;$item.non_process=$eventRecord.NonProcess
                if ($outside) {
                    $frames=[Collections.Generic.List[object]]::new()
                    $stack=$trace.GetCallStackForEvent($eventRecord)
                    while ($null -ne $stack) {
                        if ($frames.Count -ge 256) { throw 'Unexpected stack depth exceeds bounded lookup' }
                        $address=$stack.CodeAddress
                        $frames.Add([ordered]@{address_hex='0x'+$address.Address.ToString('x');module=$address.ModuleFilePath;method=$address.FullMethodName;stack_index=[int]$stack.CallStackIndex})
                        $stack=$stack.Caller
                    }
                    $item.stack=$frames
                    $item.thread=@($target.Threads | Where-Object {$_.ThreadID -eq $eventRecord.ThreadID} | ForEach-Object {[ordered]@{thread_id=$_.ThreadID;thread_index=[int]$_.ThreadIndex;start_relative_ms=$_.StartTimeRelativeMSec;end_relative_ms=$_.EndTimeRelativeMSec}})
                }
            }
            $records.Add($item)
        }
    }
    $retainedAfter=Get-Item -LiteralPath $retainedPath
    $unchanged=$retainedBefore.Length -eq $retainedAfter.Length -and $retainedBefore.LastWriteTimeUtc -eq $retainedAfter.LastWriteTimeUtc -and (Get-FileHash -LiteralPath $Audit -Algorithm SHA256).Hash.ToLowerInvariant() -eq $AuditSha256
    if (-not $unchanged) { throw 'Audit/ETLX metadata changed' }
    $report=[ordered]@{
        kind='retained-etlx-boundary-lookup-v1';audit_sha256=$AuditSha256;etl_read_or_converted=$false;symbols_resolved=$false
        etlx_path=$retainedPath;etlx_sha256_from_original_audit=$meta.converted_etlx.sha256;etlx_rehashed_in_this_lookup=$false;read_only_lock_and_metadata_unchanged=$unchanged
        windows=$windows;total_events_examined=$examined;target_records=$records;strict_audit_verdict_unchanged=$true
    }
    $json=$report | ConvertTo-Json -Depth 12
    $file=[IO.File]::Open([IO.Path]::GetFullPath($Output),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $bytes=[Text.UTF8Encoding]::new($false).GetBytes($json+"`n");$file.Write($bytes,0,$bytes.Length) } finally { $file.Dispose() }
    Write-Output "Retained ETLX lookup inspected $examined events across two 6ms windows; target records=$($records.Count)"
} finally {
    if ($null -ne $trace) { $trace.Dispose() }
    $retainedLock.Dispose()
}
