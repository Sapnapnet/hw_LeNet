param(
    [string]$Compiler = "g++",
    [switch]$PrepareRgb,
    [switch]$RunHlsCsim,
    [switch]$RunHlsSynth,
    [string]$VivadoHls = "",
    [string]$HlsStageRoot = "D:\lenet_level2_hls_csim"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataRoot = Join-Path $Root "data\self_collected"
$RawRgbRoot = Join-Path $DataRoot "raw_rgb"
$Manifest = Join-Path $RawRgbRoot "manifest.csv"
$BuildRoot = Join-Path $Root "build\level2_preprocess"
$Binary = Join-Path $BuildRoot "level2_preprocess_driver.exe"

if ($PrepareRgb) {
    & (Join-Path $PSScriptRoot "export_level2_rgb.ps1")
}
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Missing $Manifest. Run with -PrepareRgb first."
}

$compilerCommand = Get-Command $Compiler -ErrorAction SilentlyContinue
if ($null -eq $compilerCommand) {
    throw "C++ compiler '$Compiler' not found. Install/enable a compiler, then rerun this script."
}
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
& $compilerCommand.Source -std=c++11 -O2 `
    (Join-Path $Root "hls\preprocess\level2_preprocess.cpp") `
    (Join-Path $Root "scripts\level2_preprocess_driver.cpp") `
    -o $Binary
if ($LASTEXITCODE -ne 0) { throw "HLS batch driver compilation failed" }

& $Binary $RawRgbRoot $Manifest $DataRoot
if ($LASTEXITCODE -ne 0) { throw "HLS batch preprocessing failed" }

if ($RunHlsCsim -or $RunHlsSynth) {
    if ([string]::IsNullOrWhiteSpace($VivadoHls) -or -not (Test-Path -LiteralPath $VivadoHls)) {
        throw "-RunHlsCsim/-RunHlsSynth requires -VivadoHls pointing to vivado_hls.bat"
    }
    # Vivado HLS 2018.3 cannot reliably create projects below this workspace's
    # non-ASCII path.  Stage only the HLS source and testbench in an ASCII temp
    # directory; the authoritative source remains in $Root.
    $StageRoot = $HlsStageRoot
    New-Item -ItemType Directory -Force -Path (Join-Path $StageRoot "hls\preprocess") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $StageRoot "tests") | Out-Null
    Copy-Item -LiteralPath (Join-Path $Root "hls\preprocess\level2_preprocess.cpp") -Destination (Join-Path $StageRoot "hls\preprocess\level2_preprocess.cpp") -Force
    Copy-Item -LiteralPath (Join-Path $Root "hls\preprocess\level2_preprocess.h") -Destination (Join-Path $StageRoot "hls\preprocess\level2_preprocess.h") -Force
    Copy-Item -LiteralPath (Join-Path $Root "tests\tb_level2_preprocess.cpp") -Destination (Join-Path $StageRoot "tests\tb_level2_preprocess.cpp") -Force
    @'
open_project -reset hls_project
set_top level2_full_preprocess_rgb
add_files hls/preprocess/level2_preprocess.cpp -cflags "-std=c++11"
add_files -tb tests/tb_level2_preprocess.cpp -cflags "-std=c++11"
open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
'@ | Set-Content -Encoding ascii -LiteralPath (Join-Path $StageRoot "run.tcl")
    if ($RunHlsCsim) { Add-Content -Encoding ascii -LiteralPath (Join-Path $StageRoot "run.tcl") -Value "csim_design" }
    if ($RunHlsSynth) { Add-Content -Encoding ascii -LiteralPath (Join-Path $StageRoot "run.tcl") -Value "csynth_design" }
    @'
exit
'@ | Add-Content -Encoding ascii -LiteralPath (Join-Path $StageRoot "run.tcl")
    $HlsLog = Join-Path $StageRoot ("vivado_hls_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmssfff"))
    Push-Location $StageRoot
    try {
        & $VivadoHls -f .\run.tcl 2>&1 | Tee-Object -FilePath $HlsLog
        if ($LASTEXITCODE -ne 0) { throw "Vivado HLS run failed" }
        if (Select-String -LiteralPath $HlsLog -Pattern 'ERROR:|command failed|can''t create directory' -Quiet) {
            throw "Vivado HLS reported an error; inspect $HlsLog"
        }
    } finally { Pop-Location }
}
