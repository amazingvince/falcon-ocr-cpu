# Run from a normal PowerShell session with the repository's build tools installed.
# This script does not install software, download weights, or change execution policy.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Model = "artifacts/model",
    [string[]]$Profiles = @("hygiene"),
    [string]$Output = "artifacts/attempt3/first-bracket",
    [ValidateRange(1,256)][int]$Threads = 16,
    [ValidateSet("auto","scalar","avx2","avx512")][string]$Backend = "auto",
    [ValidateRange(1,100)][int]$Samples = 3,
    [ValidateRange(0,100)][int]$Warmup = 1,
    [string]$Python = "python",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
# Resolve caller-relative paths before moving into the repository.
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw "Choose a NEW output directory; existing evidence is never replaced." }
Get-Command $Python -ErrorAction Stop | Out-Null
Get-Command cargo -ErrorAction Stop | Out-Null
Push-Location $root
try {
    & $Python -m unittest discover -s attempt3 -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw "Attempt tooling tests failed." }
    if (-not $SkipBuild) {
        & cargo +1.94.0 test --locked --lib
        if ($LASTEXITCODE -ne 0) { throw "Rust unit tests failed. Preserve the failure; do not relax thresholds." }
        & cargo +1.94.0 build --release --locked --bin falcon-ocr-attempt
        if ($LASTEXITCODE -ne 0) { throw "Native build failed." }
    }
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
    $binary = Join-Path $target 'release/falcon-ocr-attempt.exe'
    if (-not (Test-Path -LiteralPath $binary)) { throw "Native binary not found at $binary" }
    & $binary doctor
    if ($LASTEXITCODE -ne 0) { throw "Native CPU capability probe failed." }
    $arguments = @('attempt3/bench.py','--binary',$binary,'--model',$modelPath,
        '--manifest',$manifestPath,'--output',$outputPath,'--threads',"$Threads",
        '--backend',$Backend,'--samples',"$Samples",'--warmup',"$Warmup",'--profiles') + $Profiles
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "One or more native runs/comparisons failed; inspect $outputPath/summary.json and arm logs." }
    Write-Host "Recorded results: $outputPath/summary.json"
    Write-Host "Successful execution is NOT quality qualification. Inspect same_output_complete_page_comparison and reasons."
} finally { Pop-Location }
