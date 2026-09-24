[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$DependencyDirectory = 'C:\Users\amazi\AppData\Roaming\PerfView\VER.2026-09-20.08.05.04.237'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Test output directory must be fresh' }
$directory = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($directory) | Out-Null
$previousDependencies = $env:TRACEAUDIT_DEPENDENCIES
try {
    $env:TRACEAUDIT_DEPENDENCIES = $DependencyDirectory
    & $Executable --self-test > (Join-Path $directory 'self-test.stdout.log') 2> (Join-Path $directory 'self-test.stderr.log')
    if ($LASTEXITCODE -ne 0) { throw 'Host self-tests failed' }
    $badEtl = Join-Path $directory 'invalid.etl'
    [IO.File]::WriteAllBytes($badEtl, [byte[]]@())
    $report = Join-Path $directory 'invalid-report.json'
    & $Executable --etl $badEtl --pid 12345 --output $report > (Join-Path $directory 'invalid.stdout.log') 2> (Join-Path $directory 'invalid.stderr.log')
    $invalidExit = $LASTEXITCODE
    if ($invalidExit -ne 2 -or -not (Test-Path -LiteralPath $report)) { throw 'Invalid ETL did not produce a rejected report' }
    $data = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    if ($data.status -ne 'audit_error' -or $data.exit_code -ne 2 -or $data.failures.Count -lt 1) { throw 'Invalid ETL was accepted' }
    $preservedHash = (Get-FileHash -LiteralPath $report -Algorithm SHA256).Hash
    & $Executable --etl $badEtl --pid 12345 --output $report > (Join-Path $directory 'overwrite.stdout.log') 2> (Join-Path $directory 'overwrite.stderr.log')
    if ($LASTEXITCODE -eq 0 -or (Get-FileHash -LiteralPath $report -Algorithm SHA256).Hash -ne $preservedHash) { throw 'Existing report was overwritten' }
    $receipt = [ordered]@{ kind='trace-audit-host-tests-v1'; passed=$true; self_tests=20; invalid_etl_rejected_with_report=$true; overwrite_rejected_bytes_preserved=$true; trace_recording_or_model_execution=$false; helper_sha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash.ToLowerInvariant(); invalid_report_sha256=$preservedHash.ToLowerInvariant() }
    [IO.File]::WriteAllText((Join-Path $directory 'tests.json'), ($receipt | ConvertTo-Json) + "`n", [Text.UTF8Encoding]::new($false))
    Write-Output 'TraceAudit: twenty self-tests plus invalid-ETL and overwrite controls passed.'
} finally {
    $env:TRACEAUDIT_DEPENDENCIES = $previousDependencies
}
