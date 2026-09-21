param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][string]$PlanSha256,
    [Parameter(Mandatory=$true)][string]$LogDirectory
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskPlanPath = [IO.Path]::GetFullPath($Plan)
$taskLogPath = [IO.Path]::GetFullPath($LogDirectory)
$taskAllowedLogs = (Join-Path $taskRoot 'artifacts\diagnostics') + '\'
if (-not $taskLogPath.StartsWith($taskAllowedLogs, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Launcher logs must be under project artifacts/diagnostics'
}
if (Test-Path -LiteralPath $taskLogPath) { throw 'Fresh launcher log directory required' }
if ($PlanSha256 -notmatch '^[a-f0-9]{64}$') { throw 'Exact plan SHA256 required' }
if ((Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PlanSha256) {
    throw 'Reviewed plan hash differs'
}
$taskPrincipal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'This specific recording launch requires Windows UAC elevation'
}
$taskPython = 'C:\Users\amazi\mambaforge\python.exe'
$taskDriver = Join-Path $taskRoot 'experiments\profiling\capture_windows.py'
$taskSourceHashes = [ordered]@{}
foreach ($taskBoundFile in @($PSCommandPath, $taskDriver, $taskPython)) {
    $taskSourceHashes[$taskBoundFile] = (Get-FileHash -LiteralPath $taskBoundFile -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskArgs = @($taskDriver, 'run', '--plan', $taskPlanPath, '--expected-plan-sha256', $PlanSha256)
# The reviewed paths contain no whitespace or quotes; reject instead of relying
# on Start-Process's array-to-string argument joining for arbitrary input.
foreach ($taskArg in $taskArgs) {
    if ($taskArg -match '[\s"'']') { throw 'Unsupported quoting in reviewed launch argument' }
}
New-Item -ItemType Directory -Path $taskLogPath | Out-Null
$taskStarted = [DateTime]::UtcNow.ToString('o')
$taskExit = 1
$taskError = $null
$taskChildId = $null
try {
    $taskChild = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $taskLogPath 'stdout.log') -RedirectStandardError (Join-Path $taskLogPath 'stderr.log')
    $taskChildId = $taskChild.Id
    $taskChild.WaitForExit()
    $taskExit = $taskChild.ExitCode
} catch {
    $taskError = $_.Exception.ToString()
} finally {
    $taskSourcesUnchanged = $true
    foreach ($taskBoundFile in $taskSourceHashes.Keys) {
        if ((Get-FileHash -LiteralPath $taskBoundFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSourceHashes[$taskBoundFile]) {
            $taskSourcesUnchanged = $false
        }
    }
    if (-not $taskSourcesUnchanged) { $taskExit = 1 }
    $taskReceipt = [ordered]@{
        kind = 'bounded-elevated-cpu-profile-launch-v1'
        plan = $taskPlanPath
        plan_sha256 = $PlanSha256
        driver = $taskDriver
        python = $taskPython
        command_arguments = $taskArgs
        started_utc = $taskStarted
        finished_utc = [DateTime]::UtcNow.ToString('o')
        child_pid = $taskChildId
        exit_code = $taskExit
        error = $taskError
        source_and_python_sha256 = $taskSourceHashes
        sources_unchanged = $taskSourcesUnchanged
        scope = 'Only the hash-bound profile driver; its owned WPR session and exact model child. No global recorder cancellation or system settings changed.'
    }
    [IO.File]::WriteAllText((Join-Path $taskLogPath 'launch.json'), ($taskReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
}
exit $taskExit
