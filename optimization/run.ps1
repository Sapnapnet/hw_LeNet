param(
    [ValidateSet('baseline','straight','noflatten','optimized','fc24','fc24b','aggressive','aggressive12','aggressive12b','aggressive5','aggressive5b')][string]$Variant='optimized',
    [ValidateSet('synth','csim','cosim')][string]$Stage='synth',
    [int]$Samples=3,
    [string]$VivadoRoot='D:\Xilinx\Vivado\2018.3'
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$resultDir=Join-Path $PSScriptRoot "results\$Variant"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
$env:LENET_VARIANT=$Variant
$env:LENET_STAGE=$Stage
$env:LENET_SAMPLES=[string]$Samples
$sourceHashes=@{}
foreach ($folder in @('hls','config','optimization')) {
    Get-ChildItem -Path (Join-Path $projectRoot $folder) -Recurse -File |
        Where-Object { $_.Extension -in @('.cpp','.h') -and $_.FullName -notmatch '\\(build|results|experiments)\\' } |
        ForEach-Object { $sourceHashes[$_.FullName.Substring($projectRoot.Length+1)] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
}
@{variant=$Variant;stage=$Stage;samples=$Samples;started=(Get-Date).ToString('o');
    source_sha256=$sourceHashes;tool=$VivadoRoot;part='xc7z020clg400-1';clock_ns=10} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $resultDir "$Stage.inputs.json")
Push-Location $resultDir
try {
    & (Join-Path $VivadoRoot 'bin\vivado_hls.bat') -f (Join-Path $PSScriptRoot 'run_hls.tcl') 2>&1 | Tee-Object -FilePath "$Stage.log"
    $toolExit=$LASTEXITCODE
    if ($toolExit -ne 0) {throw "HLS $Stage failed with exit code $toolExit"}
    if (Select-String -LiteralPath "$Stage.log" -Pattern '^ERROR:' -Quiet) {
        throw "HLS reported an error despite exit code 0; inspect $resultDir\$Stage.log"
    }
} finally {Pop-Location}
