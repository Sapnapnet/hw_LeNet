param(
    [int]$Limit = 10000,
    [string]$HlsInclude = "F:\Vivado\Vivado\2018.3\include"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$Cxx = (Get-Command g++ -ErrorAction Stop).Source
$Build = Join-Path $Root "build"
$Results = Join-Path $Root "results\accuracy"
$Dataset = Join-Path $Root "data\mnist\raw"
$FloatWeights = Join-Path $Root "weights\float"
$FixedWeights = Join-Path $Root "weights\fixed"
New-Item -ItemType Directory -Force -Path $Build, $Results | Out-Null

Push-Location $Root
try {
    Write-Host "== Download/validate MNIST =="
    & $Python -B scripts/download_mnist.py
    if ($LASTEXITCODE -ne 0) { throw "MNIST download failed" }

    $Common = @(
        "-std=c++11", "-O2", "-static-libstdc++", "-static-libgcc",
        "-Iconfig", "-Ihls/conv", "-Ihls/operators", "-Ihls/buffer", "-Ihls/top",
        "tests/tb_mnist_10k.cpp", "hls/top/lenet_accelerator.cpp",
        "hls/conv/conv2d_systolic.cpp"
    )
    Write-Host "== Build Float driver =="
    & $Cxx @Common "-o" (Join-Path $Build "tb_mnist_float.exe")
    if ($LASTEXITCODE -ne 0) { throw "Float driver compilation failed" }

    Write-Host "== Build Fixed driver =="
    if (-not (Test-Path (Join-Path $HlsInclude "ap_fixed.h"))) {
        throw "ap_fixed.h not found under $HlsInclude"
    }
    $FixedArgs = @(
        "-std=c++11", "-O2", "-static-libstdc++", "-static-libgcc",
        "-DLENET_USE_FIXED", "-DLENET_ACC_INT", "-I$HlsInclude",
        "-Iconfig", "-Ihls/conv", "-Ihls/operators", "-Ihls/buffer", "-Ihls/top",
        "tests/tb_mnist_10k.cpp", "hls/top/lenet_accelerator.cpp",
        "hls/conv/conv2d_systolic.cpp", "-o", (Join-Path $Build "tb_mnist_fixed.exe")
    )
    & $Cxx @FixedArgs
    if ($LASTEXITCODE -ne 0) { throw "Fixed driver compilation failed" }

    $ImageFile = Join-Path $Dataset "t10k-images-idx3-ubyte"
    $LabelFile = Join-Path $Dataset "t10k-labels-idx1-ubyte"
    Write-Host "== Run Float full-set C simulation ($Limit samples) =="
    & (Join-Path $Build "tb_mnist_float.exe") `
        --images $ImageFile --labels $LabelFile --weights $FloatWeights `
        --output (Join-Path $Results "portable_float_10000.csv") --limit $Limit `
        2>&1 | Tee-Object -FilePath (Join-Path $Results "portable_float_csim_10000.log")
    if ($LASTEXITCODE -ne 0) { throw "Float full-set run failed" }

    Write-Host "== Run Fixed full-set C simulation ($Limit samples) =="
    & (Join-Path $Build "tb_mnist_fixed.exe") `
        --images $ImageFile --labels $LabelFile --weights $FixedWeights `
        --output (Join-Path $Results "portable_fixed_10000.csv") --limit $Limit `
        2>&1 | Tee-Object -FilePath (Join-Path $Results "portable_fixed_csim_10000.log")
    if ($LASTEXITCODE -ne 0) { throw "Fixed full-set run failed" }

    Write-Host "== Generate Python Float/Fixed golden outputs =="
    & $Python -B scripts/generate_references.py `
        --dataset-root $Dataset --weights-root (Join-Path $Root "weights") `
        --output-dir $Results --limit $Limit
    if ($LASTEXITCODE -ne 0) { throw "Python reference generation failed" }

    try { $Commit = (& git rev-parse HEAD 2>$null).Trim() } catch { $Commit = "unknown (集成副本未初始化 Git)" }
    Write-Host "== Compare HLS against labels and Python references =="
    & $Python -B scripts/compare_full_results.py `
        --hls-float (Join-Path $Results "portable_float_10000.csv") `
        --hls-fixed (Join-Path $Results "portable_fixed_10000.csv") `
        --python-float (Join-Path $Results "python_float_10000.csv") `
        --python-fixed (Join-Path $Results "python_fixed_10000.csv") `
        --output-dir $Results --source-commit $Commit `
        --hls-version "portable g++ C simulation (not Vivado HLS CSim)" `
        --float-macros "none" --fixed-macros "LENET_USE_FIXED LENET_ACC_INT"
    if ($LASTEXITCODE -ne 0) { throw "Result comparison failed" }
} finally {
    Pop-Location
}
