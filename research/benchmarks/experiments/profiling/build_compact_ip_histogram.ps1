[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory)
# Build and pure host self-test only. This wrapper never opens ETL/ETLX.
$ErrorActionPreference = 'Stop'
$compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$netstandard = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\netstandard.dll'
$dependencyDirectory = 'C:\Users\amazi\AppData\Roaming\PerfView\VER.2026-09-20.08.05.04.237'
$source = Join-Path $PSScriptRoot 'CompactIpHistogram.cs'
$traceLibrary = Join-Path $dependencyDirectory 'Microsoft.Diagnostics.Tracing.TraceEvent.dll'
$serializationLibrary = Join-Path $dependencyDirectory 'Microsoft.Diagnostics.FastSerialization.dll'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Fresh build directory required' }
if ((Get-FileHash -LiteralPath $traceLibrary -Algorithm SHA256).Hash.ToLowerInvariant() -ne '530946dc20e89754783f0ac76d86b7a4eedf95326f255db34c32fcbf5c3ce0ff') { throw 'Wrong TraceEvent library' }
$inputs = @($source,$PSCommandPath,$compiler,$netstandard,$traceLibrary,$serializationLibrary)
$before = @{}
foreach ($path in $inputs) { $before[[IO.Path]::GetFullPath($path)] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
$output = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($output) | Out-Null
$archivedSource = Join-Path $output 'CompactIpHistogram.cs'
Copy-Item -LiteralPath $source -Destination $archivedSource
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $output 'build_compact_ip_histogram.ps1')
$executable = Join-Path $output 'CompactIpHistogram.exe'
$arguments = @('/nologo','/target:exe','/platform:x64','/optimize+','/debug:pdbonly',"/out:$executable",'/reference:System.Core.dll','/reference:System.Web.Extensions.dll',"/reference:$netstandard","/reference:$traceLibrary","/reference:$serializationLibrary",$archivedSource)
& $compiler @arguments > (Join-Path $output 'compile.stdout.log') 2> (Join-Path $output 'compile.stderr.log')
$compileExit = $LASTEXITCODE
$testExit = $null
if ($compileExit -eq 0) {
    & $executable --self-test > (Join-Path $output 'self-test.stdout.log') 2> (Join-Path $output 'self-test.stderr.log')
    $testExit = $LASTEXITCODE
}
$unchanged = $true
foreach ($path in $before.Keys) { if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $before[$path]) { $unchanged = $false } }
$archivedSha = (Get-FileHash -LiteralPath $archivedSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($archivedSha -ne $before[[IO.Path]::GetFullPath($source)]) { $unchanged = $false }
$logs = @{}
foreach ($name in @('compile.stdout.log','compile.stderr.log','self-test.stdout.log','self-test.stderr.log')) {
    $path = Join-Path $output $name
    if (Test-Path -LiteralPath $path) { $logs[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$receipt = [ordered]@{
    kind='compact-ip-histogram-build-v1';compiler=$compiler;arguments=$arguments
    input_sha256=$before;source_window_unchanged=$unchanged;compile_exit_code=$compileExit
    archived_source=$archivedSource;archived_source_sha256=$archivedSha;host_self_test_exit_code=$testExit
    compiled_executable_sha256=$(if(Test-Path -LiteralPath $executable){(Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()}else{$null})
    log_sha256=$logs;no_trace_or_model_execution=$true
}
[IO.File]::WriteAllText((Join-Path $output 'build.json'),($receipt | ConvertTo-Json -Depth 6)+"`n",[Text.UTF8Encoding]::new($false))
if ($compileExit -ne 0 -or $testExit -ne 0 -or -not $unchanged) { throw "Build/host checks rejected: compile=$compileExit tests=$testExit source_unchanged=$unchanged" }
Write-Output $executable
