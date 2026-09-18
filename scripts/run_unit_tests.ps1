param(
    [string]$HlsInclude = "F:\Vivado\Vivado\2018.3\include"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Cxx = (Get-Command g++ -ErrorAction Stop).Source
$Build = Join-Path $Root "build"
New-Item -ItemType Directory -Force -Path $Build | Out-Null

Push-Location $Root
try {
    $common = @("-std=c++11", "-O2", "-static-libstdc++", "-static-libgcc", "-Iconfig", "-Ihls/buffer", "tests/tb_address_buffer.cpp", "-o")
    & $Cxx @common (Join-Path $Build "tb_address_float.exe")
    if ($LASTEXITCODE -ne 0) { throw "Float address/buffer compile failed" }
    & (Join-Path $Build "tb_address_float.exe")
    if ($LASTEXITCODE -ne 0) { throw "Float address/buffer test failed" }

    if (-not (Test-Path (Join-Path $HlsInclude "ap_fixed.h"))) {
        throw "ap_fixed.h not found under $HlsInclude"
    }
    $fixed = @("-std=c++11", "-O2", "-static-libstdc++", "-static-libgcc", "-DLENET_USE_FIXED", "-DLENET_ACC_INT", "-I$HlsInclude", "-Iconfig", "-Ihls/buffer", "tests/tb_address_buffer.cpp", "-o", (Join-Path $Build "tb_address_fixed.exe"))
    & $Cxx @fixed
    if ($LASTEXITCODE -ne 0) { throw "Fixed address/buffer compile failed" }
    & (Join-Path $Build "tb_address_fixed.exe")
    if ($LASTEXITCODE -ne 0) { throw "Fixed address/buffer test failed" }
} finally {
    Pop-Location
}
