param(
    [ValidateSet("float", "fixed", "both")]
    [string]$Mode = "both",
    [string]$VivadoHls = "F:\Vivado\Vivado\2018.3\bin\vivado_hls.bat",
    [string]$HlsInclude = "F:\Vivado\Vivado\2018.3\include"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path -LiteralPath $VivadoHls)) {
    throw "Vivado HLS executable not found: $VivadoHls"
}

Push-Location $Root
try {
    $env:HLS_INCLUDE = $HlsInclude
    foreach ($kind in @("float", "fixed")) {
        if ($Mode -ne "both" -and $Mode -ne $kind) { continue }
        $tcl = Join-Path $Root ("scripts\hls\csim_{0}.tcl" -f $kind)
        $log = Join-Path $Root ("results\accuracy\{0}_vivado_hls_csim_10000.log" -f $kind)
        New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
        Write-Host "== Vivado HLS CSim $kind =="
        & $VivadoHls -f $tcl 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "Vivado HLS CSim failed for $kind" }
    }
} finally {
    Pop-Location
}
