param(
    [string]$Binary = 'artifacts/benchmarks/ocr-packed-bench.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Binary)) { throw "Benchmark binary missing: $Binary" }
$modes = @(
    @{Name='unpacked-a'; Layout='unpacked'},
    @{Name='packed-a'; Layout='phase-packed'},
    @{Name='packed-b'; Layout='phase-packed'},
    @{Name='unpacked-b'; Layout='unpacked'}
)
foreach ($mode in $modes) {
    $report = 'artifacts/benchmarks/quiet-packed-' + $mode.Name + '.json'
    if (Test-Path -LiteralPath $report) { throw "Refusing to overwrite benchmark evidence: $report" }
    & $Binary artifacts/reference/smoke-fp32/canonical-rgb.png --backend avx2 --threads 16 --execution joint --cache-layout expanded --weight-layout $mode.Layout --batches 1,2,4,8 --warmup 2 --repetitions 7 --max-dimension 256 --max-new-tokens 128 --cpu-label 'Ryzen 9 7950X' --environment-label 'native Windows; project CPU/GPU jobs paused; interactive apps remain open' --output $report
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $($mode.Name)" }
}
