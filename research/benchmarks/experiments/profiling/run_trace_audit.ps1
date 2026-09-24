[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$Etl,
    [Parameter(Mandatory=$true)][int]$TargetProcessId,
    [Parameter(Mandatory=$true)][string]$Output,
    [double]$BinMilliseconds = 1000,
    [string]$ExpectedImage,
    [string]$ExpectedCommandLine,
    [string]$DependencyDirectory = 'C:\Users\amazi\AppData\Roaming\PerfView\VER.2026-09-20.08.05.04.237'
)
$ErrorActionPreference = 'Stop'
$previousDependencies = $env:TRACEAUDIT_DEPENDENCIES
try {
    $env:TRACEAUDIT_DEPENDENCIES = [IO.Path]::GetFullPath($DependencyDirectory)
    $arguments = @('--etl',[IO.Path]::GetFullPath($Etl),'--pid',$TargetProcessId.ToString([Globalization.CultureInfo]::InvariantCulture),'--output',[IO.Path]::GetFullPath($Output),'--bin-ms',$BinMilliseconds.ToString('R',[Globalization.CultureInfo]::InvariantCulture))
    if ($ExpectedImage) { $arguments += @('--expected-image',$ExpectedImage) }
    if ($PSBoundParameters.ContainsKey('ExpectedCommandLine')) { $arguments += @('--expected-command-line',$ExpectedCommandLine) }
    & $Executable @arguments
    $auditExit = $LASTEXITCODE
} finally {
    $env:TRACEAUDIT_DEPENDENCIES = $previousDependencies
}
exit $auditExit
