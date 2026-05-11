# Downloads the gyan.dev ffmpeg release-essentials zip and extracts it into
# vendor/ffmpeg/. Idempotent: if vendor/ffmpeg/bin/ffmpeg.exe already exists,
# does nothing.
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $repoRoot "vendor"
$ffmpegDir = Join-Path $vendorDir "ffmpeg"
$ffmpegExe = Join-Path $ffmpegDir "bin\ffmpeg.exe"

if (Test-Path $ffmpegExe) {
    Write-Host "ffmpeg already present at $ffmpegExe"
    exit 0
}

New-Item -ItemType Directory -Path $vendorDir -Force | Out-Null

$zipUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$zipPath = Join-Path $vendorDir "ffmpeg.zip"

Write-Host "Downloading ffmpeg from $zipUrl ..."
Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing

Write-Host "Extracting ..."
$extractTmp = Join-Path $vendorDir "_ffmpeg_extract"
if (Test-Path $extractTmp) { Remove-Item -Recurse -Force $extractTmp }
Expand-Archive -Path $zipPath -DestinationPath $extractTmp -Force

$inner = Get-ChildItem -Path $extractTmp -Directory | Select-Object -First 1
if ($null -eq $inner) {
    throw "Extracted zip had no inner directory; expected ffmpeg-*-essentials_build/"
}

if (Test-Path $ffmpegDir) { Remove-Item -Recurse -Force $ffmpegDir }
Move-Item -Path $inner.FullName -Destination $ffmpegDir

Remove-Item -Recurse -Force $extractTmp
Remove-Item -Force $zipPath

if (-not (Test-Path $ffmpegExe)) {
    throw "ffmpeg install failed: $ffmpegExe not found after extraction."
}
Write-Host "ffmpeg installed at $ffmpegExe"
