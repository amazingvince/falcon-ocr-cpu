[CmdletBinding(DefaultParameterSetName='Capture')]
param(
    [Parameter(Mandatory=$true,ParameterSetName='Capture')][string]$Plan,
    [Parameter(Mandatory=$true,ParameterSetName='Capture')][string]$PlanSha256,
    [Parameter(Mandatory=$true)][string]$LogDirectory,
    [Parameter(Mandatory=$true,ParameterSetName='Validate')][switch]$ValidateExitCodeReporting
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskLogPath = [IO.Path]::GetFullPath($LogDirectory)
$taskAllowedLogs = (Join-Path $taskRoot 'artifacts\diagnostics') + '\'
if (-not $taskLogPath.StartsWith($taskAllowedLogs, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Launcher logs must be under project artifacts/diagnostics'
}
if (Test-Path -LiteralPath $taskLogPath) { throw 'Fresh launcher log directory required' }
$taskPython = 'C:\Users\amazi\mambaforge\python.exe'
$taskDriver = Join-Path $taskRoot 'experiments\profiling\capture_compact_windows.py'

# Query the retained process handle directly. Start-Process's returned Process
# object previously reported a null ExitCode; null must never become exit zero.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FalconCompactChildExit {
    [DllImport("kernel32.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetExitCodeProcess(IntPtr process, out uint code);
}
'@
function Invoke-CheckedChild([string[]]$ChildArguments, [string]$Stem) {
    foreach ($taskArg in $ChildArguments) {
        if ($taskArg -match '[\s"'']') { throw 'Unsupported quoting in reviewed launch argument' }
    }
    $taskInfo = [Diagnostics.ProcessStartInfo]::new()
    $taskInfo.FileName = $taskPython
    $taskInfo.Arguments = $ChildArguments -join ' '
    $taskInfo.WorkingDirectory = $taskRoot
    $taskInfo.UseShellExecute = $false
    $taskInfo.CreateNoWindow = $true
    $taskInfo.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $taskInfo.RedirectStandardOutput = $true
    $taskInfo.RedirectStandardError = $true
    $taskChild = [Diagnostics.Process]::new()
    $taskChild.StartInfo = $taskInfo
    try {
        if (-not $taskChild.Start()) { throw 'Child process did not start' }
        $taskChildId = $taskChild.Id
        $taskHandle = $taskChild.Handle
        # Drain both streams concurrently; waiting with full pipes can deadlock.
        $taskStdout = $taskChild.StandardOutput.ReadToEndAsync()
        $taskStderr = $taskChild.StandardError.ReadToEndAsync()
        $taskChild.WaitForExit()
        [IO.File]::WriteAllText((Join-Path $taskLogPath ($Stem + '.stdout.log')), $taskStdout.Result, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText((Join-Path $taskLogPath ($Stem + '.stderr.log')), $taskStderr.Result, [Text.UTF8Encoding]::new($false))
        [uint32]$taskNativeExit = 0
        if (-not $taskChild.HasExited -or -not [FalconCompactChildExit]::GetExitCodeProcess($taskHandle, [ref]$taskNativeExit)) {
            throw 'Concrete child exit code unavailable; outcome is failed/unknown'
        }
        return [pscustomobject]@{ pid=$taskChildId; exit_code=[long]$taskNativeExit; exit_code_observed=$true }
    } finally {
        # Dispose only the local handle. Never terminate a recorder or another
        # process; the hash-bound Python driver owns its scoped WPR cleanup.
        $taskChild.Dispose()
    }
}

$taskSourceHashes = [ordered]@{}
foreach ($taskBoundFile in @($PSCommandPath, $taskDriver, $taskPython)) {
    $taskSourceHashes[$taskBoundFile] = (Get-FileHash -LiteralPath $taskBoundFile -Algorithm SHA256).Hash.ToLowerInvariant()
}
if ($ValidateExitCodeReporting) {
    # Only two fixed trivial children. No plan, driver, model, WPR or UAC call.
    New-Item -ItemType Directory -Path $taskLogPath | Out-Null
    $taskCases = @()
    foreach ($taskExpected in @(0,7)) {
        $taskResult = Invoke-CheckedChild @('-c', "raise(SystemExit($taskExpected))") "exit-$taskExpected"
        if (-not $taskResult.exit_code_observed -or $taskResult.exit_code -ne $taskExpected) { throw 'Child exit reporting regression' }
        $taskCases += [ordered]@{expected=$taskExpected; actual=$taskResult.exit_code; pid=$taskResult.pid; exit_code_observed=$true}
    }
    foreach ($taskBoundFile in $taskSourceHashes.Keys) {
        if ((Get-FileHash -LiteralPath $taskBoundFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSourceHashes[$taskBoundFile]) { throw 'Validation source changed' }
    }
    $taskValidation = [ordered]@{kind='compact-launcher-exit-reporting-validation-v1'; status='passed'; cases=$taskCases; source_and_python_sha256=$taskSourceHashes; scope='Two fixed Python exit children only; no elevation, recorder, model or capture.'}
    [IO.File]::WriteAllText((Join-Path $taskLogPath 'validation.json'), ($taskValidation | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    exit 0
}

$taskPlanPath = [IO.Path]::GetFullPath($Plan)
if ($PlanSha256 -notmatch '^[a-f0-9]{64}$') { throw 'Exact plan SHA256 required' }
$taskPlanBytes = [IO.File]::ReadAllBytes($taskPlanPath)
$taskHasher = [Security.Cryptography.SHA256]::Create()
try { $taskPlanDigest = ([BitConverter]::ToString($taskHasher.ComputeHash($taskPlanBytes))).Replace('-','').ToLowerInvariant() }
finally { $taskHasher.Dispose() }
if ($taskPlanDigest -ne $PlanSha256) { throw 'Reviewed plan hash differs' }
$taskPlanObject = [Text.Encoding]::UTF8.GetString($taskPlanBytes) | ConvertFrom-Json
if ($taskPlanObject.kind -ne 'falcon-ocr-owned-native-compact-cpu-profile-v1' -or $taskPlanObject.runtime_python -ne $taskPython) { throw 'Unexpected compact plan/runtime' }
foreach ($taskBoundFile in $taskSourceHashes.Keys) {
    if ($taskPlanObject.files_sha256.PSObject.Properties[$taskBoundFile].Value -ne $taskSourceHashes[$taskBoundFile]) { throw "Plan does not bind current launcher/driver/Python: $taskBoundFile" }
}
$taskPrincipal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'This specific recording launch requires Windows UAC elevation' }
$taskArgs = @($taskDriver, 'run', '--plan', $taskPlanPath, '--expected-plan-sha256', $PlanSha256)
New-Item -ItemType Directory -Path $taskLogPath | Out-Null
$taskStarted = [DateTime]::UtcNow.ToString('o')
$taskExit = 1
$taskChildExit = $null
$taskExitObserved = $false
$taskError = $null
$taskChildId = $null
try {
    $taskResult = Invoke-CheckedChild $taskArgs 'driver'
    $taskChildId = $taskResult.pid
    $taskChildExit = $taskResult.exit_code
    $taskExitObserved = $taskResult.exit_code_observed
    if (-not $taskExitObserved -or $null -eq $taskChildExit) { throw 'Child outcome failed/unknown' }
    $taskExit = if ($taskChildExit -eq 0) { 0 } else { 1 }
} catch {
    $taskError = $_.Exception.ToString()
} finally {
    $taskSourcesUnchanged = $true
    foreach ($taskBoundFile in $taskSourceHashes.Keys) {
        try {
            if ((Get-FileHash -LiteralPath $taskBoundFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSourceHashes[$taskBoundFile]) { $taskSourcesUnchanged = $false }
        } catch { $taskSourcesUnchanged = $false; $taskError = "$taskError; source closure: $($_.Exception.Message)" }
    }
    if (-not $taskSourcesUnchanged -or -not $taskExitObserved) { $taskExit = 1 }
    $taskReceipt = [ordered]@{
        kind='bounded-elevated-compact-cpu-profile-launch-v1'; plan=$taskPlanPath; plan_sha256=$PlanSha256
        driver=$taskDriver; python=$taskPython; command_arguments=$taskArgs
        started_utc=$taskStarted; finished_utc=[DateTime]::UtcNow.ToString('o'); child_pid=$taskChildId
        exit_code=$taskChildExit; exit_code_observed=$taskExitObserved; launcher_exit_code=$taskExit
        status=$(if ($taskExit -eq 0) {'child_exited_zero'} else {'failed_or_unknown'})
        error=$taskError; source_and_python_sha256=$taskSourceHashes; sources_unchanged=$taskSourcesUnchanged
        scope='Only the hash-bound compact profile driver; it owns WPR cleanup. No global recorder cancellation or automatic elevation.'
    }
    [IO.File]::WriteAllText((Join-Path $taskLogPath 'launch.json'), ($taskReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
}
exit $taskExit
