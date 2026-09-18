param(
    [string]$OriginalRoot = (Join-Path $PSScriptRoot "..\data\self_collected\original"),
    [string]$RawRgbRoot = (Join-Path $PSScriptRoot "..\data\self_collected\raw_rgb"),
    [string]$LabelCsvPath = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
$OriginalRoot = (Resolve-Path -LiteralPath $OriginalRoot).Path
New-Item -ItemType Directory -Force -Path $RawRgbRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($LabelCsvPath)) { $LabelCsvPath = Join-Path $OriginalRoot "label.csv" }
$labels = Import-Csv -LiteralPath $LabelCsvPath
if ($labels.Count -eq 0) { throw "No rows in $OriginalRoot\label.csv" }
$manifest = @()
foreach ($row in $labels) {
    $source = Join-Path $OriginalRoot (Join-Path $row.folder $row.filename)
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing labelled source image: $source" }
    $relativeRgb = Join-Path $row.folder ([System.IO.Path]::ChangeExtension($row.filename, '.ppm'))
    $destination = Join-Path $RawRgbRoot $relativeRgb
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null

    # This adapter only decodes JPEG/PNG into an RGB byte buffer.  The HLS C++
    # kernel owns every preprocessing operation after this point.
    $sourceImage = [System.Drawing.Image]::FromFile($source)
    try {
        if ($sourceImage.Width -gt 640 -or $sourceImage.Height -gt 480) {
            throw "Image exceeds HLS input bound 640x480: $source"
        }
        $bitmap = New-Object System.Drawing.Bitmap($sourceImage)
        try {
            $pixels = New-Object byte[] ($bitmap.Width * $bitmap.Height * 3)
            $offset = 0
            for ($y = 0; $y -lt $bitmap.Height; ++$y) {
                for ($x = 0; $x -lt $bitmap.Width; ++$x) {
                    $color = $bitmap.GetPixel($x, $y)
                    $pixels[$offset] = $color.R
                    $pixels[$offset + 1] = $color.G
                    $pixels[$offset + 2] = $color.B
                    $offset += 3
                }
            }
            $header = [System.Text.Encoding]::ASCII.GetBytes("P6`n$($bitmap.Width) $($bitmap.Height)`n255`n")
            $stream = [System.IO.File]::Open($destination, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
            try {
                $stream.Write($header, 0, $header.Length)
                $stream.Write($pixels, 0, $pixels.Length)
            } finally { $stream.Dispose() }
            $manifest += [PSCustomObject]@{
                filename = $row.filename; label = $row.label; writer_id = $row.writer_id; folder = $row.folder
                source_relative = (Join-Path $row.folder $row.filename).Replace('\', '/')
                rgb_relative = $relativeRgb.Replace('\', '/')
                width = $bitmap.Width; height = $bitmap.Height
            }
        } finally { $bitmap.Dispose() }
    } finally { $sourceImage.Dispose() }
}

$manifest | ConvertTo-Csv -NoTypeInformation | Set-Content -Encoding ascii -LiteralPath (Join-Path $RawRgbRoot "manifest.csv")
Write-Host "PASS: exported $($manifest.Count) RGB PPM inputs to $RawRgbRoot"
