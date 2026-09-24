[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$DependencyDirectory = 'C:\Users\amazi\AppData\Roaming\PerfView\VER.2026-09-20.08.05.04.237'
)
$ErrorActionPreference = 'Stop'
$compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$netstandard = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\netstandard.dll'
$source = Join-Path $PSScriptRoot 'TraceAudit.cs'
$traceLibrary = Join-Path $DependencyDirectory 'Microsoft.Diagnostics.Tracing.TraceEvent.dll'
$serializationLibrary = Join-Path $DependencyDirectory 'Microsoft.Diagnostics.FastSerialization.dll'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Build output directory must be fresh' }
if ((Get-FileHash -LiteralPath $traceLibrary -Algorithm SHA256).Hash.ToLowerInvariant() -ne '530946dc20e89754783f0ac76d86b7a4eedf95326f255db34c32fcbf5c3ce0ff') { throw 'Wrong TraceEvent bytes' }
$inputs = @($source, $PSCommandPath, $compiler, $netstandard, $traceLibrary, $serializationLibrary)
$before = @{}
foreach ($path in $inputs) { $before[[IO.Path]::GetFullPath($path)] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
$output = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($output) | Out-Null
Copy-Item -LiteralPath $source -Destination (Join-Path $output 'TraceAudit.cs')
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $output 'build_trace_audit.ps1')
$executable = Join-Path $output 'TraceAudit.exe'
$arguments = @('/nologo','/target:exe','/platform:x64','/optimize+','/debug:pdbonly',"/out:$executable",'/reference:System.Core.dll','/reference:System.Web.Extensions.dll',"/reference:$netstandard","/reference:$traceLibrary","/reference:$serializationLibrary",(Join-Path $output 'TraceAudit.cs'))
& $compiler @arguments > (Join-Path $output 'compile.stdout.log') 2> (Join-Path $output 'compile.stderr.log')
$compileExit = $LASTEXITCODE
$unchanged = $true
foreach ($path in $before.Keys) { if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $before[$path]) { $unchanged = $false } }
$receipt = [ordered]@{
    kind = 'trace-audit-build-v1'
    compiler = $compiler
    arguments = $arguments
    dependency_directory = [IO.Path]::GetFullPath($DependencyDirectory)
    input_sha256 = $before
    source_window_unchanged = $unchanged
    compile_exit_code = $compileExit
    compiled_executable_sha256 = $(if (Test-Path -LiteralPath $executable) { (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null })
    no_trace_or_model_execution = $true
}
[IO.File]::WriteAllText((Join-Path $output 'build.json'), ($receipt | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
if ($compileExit -ne 0 -or -not $unchanged) { throw "TraceAudit compilation rejected; exit=$compileExit source_unchanged=$unchanged" }
Write-Output $executable
