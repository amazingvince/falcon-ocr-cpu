# Format, lint and unit-test gate; -Smoke also rebuilds the release binary and
# checks the exact-mode smoke trace hash (needs artifacts/model and the fixture).
param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Run($cmd) { Write-Host "> $cmd"; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
Run 'cargo fmt --all -- --check'
Run 'cargo clippy --release --all-targets --locked -- -D warnings'
Run 'cargo test --release --lib --locked'
if ($Smoke) {
    Run 'cargo build --release --locked --bins'
    $out = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Path $out | Out-Null
    Run "target/release/falcon-ocr --threads 4 --backend avx2 trace --fixture artifacts/reference/smoke-fp32/trace.safetensors --output $out/trace.safetensors --max-new-tokens 17 > `$null"
    $hash = (Get-FileHash -Algorithm SHA256 "$out/trace.safetensors").Hash.Substring(0, 8).ToLower()
    Remove-Item -Recurse -Force $out
    if ($hash -ne 'e2dad223') { Write-Error "smoke trace hash $hash, expected e2dad223"; exit 1 }
    Write-Host "smoke trace ok ($hash)"
}
