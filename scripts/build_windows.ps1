param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CargoArguments)
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ToolRoot = Join-Path $ProjectRoot 'artifacts/tools'
if (!$CargoArguments) { $CargoArguments = @('build', '--release') }

function Get-VerifiedArchive([string]$Url, [string]$Filename, [string]$Sha256) {
    New-Item -ItemType Directory -Force -Path $ToolRoot | Out-Null
    $Target = Join-Path $ToolRoot $Filename
    if (!(Test-Path -LiteralPath $Target)) { Invoke-WebRequest -Uri $Url -OutFile $Target }
    $Actual = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash
    if ($Actual -ne $Sha256) { throw "Checksum mismatch for $Target; expected $Sha256, found $Actual" }
    return $Target
}

$OriginalPath = $env:PATH
$OriginalNasm = $env:ASM_NASM
Push-Location $ProjectRoot
try {
    if (!$env:ASM_NASM) {
        $NasmCommand = Get-Command nasm -ErrorAction SilentlyContinue
        if ($NasmCommand) { $env:ASM_NASM = $NasmCommand.Source } else {
            $NasmExe = Join-Path $ToolRoot 'nasm-2.16.03/nasm.exe'
            if (!(Test-Path -LiteralPath $NasmExe)) {
                $Archive = Get-VerifiedArchive 'https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/win64/nasm-2.16.03-win64.zip' 'nasm-2.16.03-win64.zip' '3ee4782247bcb874378d02f7eab4e294a84d3d15f3f6ee2de2f47a46aa7226e6'
                Expand-Archive -LiteralPath $Archive -DestinationPath $ToolRoot -Force
            }
            $env:ASM_NASM = $NasmExe
        }
    }
    if (!(Get-Command cmake -ErrorAction SilentlyContinue)) {
        $CmakeBin = Join-Path $ToolRoot 'cmake-3.31.10-windows-x86_64/bin'
        if (!(Test-Path -LiteralPath (Join-Path $CmakeBin 'cmake.exe'))) {
            $Archive = Get-VerifiedArchive 'https://github.com/Kitware/CMake/releases/download/v3.31.10/cmake-3.31.10-windows-x86_64.zip' 'cmake-3.31.10-windows-x86_64.zip' '13d1a463d7130df5339baedd63d8ae990aaf385062b2f42f372796143ae94086'
            Expand-Archive -LiteralPath $Archive -DestinationPath $ToolRoot -Force
        }
        $env:PATH = "$CmakeBin;$env:PATH"
    }
    & cargo @CargoArguments
    if ($LASTEXITCODE -ne 0) { throw "cargo exited with status $LASTEXITCODE" }
} finally {
    $env:PATH = $OriginalPath
    $env:ASM_NASM = $OriginalNasm
    Pop-Location
}
