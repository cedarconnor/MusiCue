# MusiCue Windows bootstrap. Installs uv, creates .venv, installs all
# Python deps, fetches ffmpeg, downloads model weights, prints a readiness
# table. Hard fails on core deps; soft warns on optional pieces.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$script:soft_warnings = @()

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Soft-Warn($name, $msg) {
    Write-Host "[WARN] $name : $msg" -ForegroundColor Yellow
    $script:soft_warnings += @{ name = $name; msg = $msg }
}

# -----------------------------------------------------------------------------
# 1. uv bootstrap
# -----------------------------------------------------------------------------
Write-Step "Checking for uv"
$uv = (Get-Command uv -ErrorAction SilentlyContinue)
if ($null -eq $uv) {
    Write-Host "uv not found on PATH; installing from astral.sh ..."
    try {
        Invoke-RestMethod -UseBasicParsing https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        throw "Could not download uv installer. Check your internet connection. ($_)"
    }
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:LOCALAPPDATA\Programs\uv;$env:Path"
    $uv = (Get-Command uv -ErrorAction SilentlyContinue)
    if ($null -eq $uv) {
        throw "uv installation failed; not on PATH after install. Try opening a new terminal."
    }
}
Write-Host "uv: $($uv.Source)"

# -----------------------------------------------------------------------------
# 2. venv
# -----------------------------------------------------------------------------
Write-Step "Creating .venv with Python 3.11"
& uv venv .venv --python 3.11
if ($LASTEXITCODE -ne 0) { throw "uv venv failed (exit $LASTEXITCODE)" }

# -----------------------------------------------------------------------------
# 3. Core install (hard fail)
# -----------------------------------------------------------------------------
Write-Step "Installing core dependencies (this may take several minutes)"
& uv pip install -e ".[dev,ui,midi,osc]" basic-pitch `
    --extra-index-url https://download.pytorch.org/whl/cu121 `
    --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw "Core install failed (exit $LASTEXITCODE)" }

# -----------------------------------------------------------------------------
# 4. CLAP (soft warn)
# -----------------------------------------------------------------------------
Write-Step "Installing CLAP (optional)"
& uv pip install -e ".[clap]"
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "clap" "CLAP install failed; semantic labels will be disabled."
}

# -----------------------------------------------------------------------------
# 5. All-In-One (soft warn — historically painful on Windows)
# -----------------------------------------------------------------------------
Write-Step "Installing All-In-One (optional)"
& uv pip install allin1
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "allin1" "All-In-One install failed; beat detection will use the librosa fallback (no sections)."
}

# -----------------------------------------------------------------------------
# 6. ffmpeg
# -----------------------------------------------------------------------------
Write-Step "Checking for ffmpeg"
$ffmpegPath = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if ($null -eq $ffmpegPath) {
    Write-Host "ffmpeg not on PATH; downloading portable build ..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "install_ffmpeg.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg install failed and ffmpeg is required."
    }
} else {
    Write-Host "ffmpeg found at $($ffmpegPath.Source)"
}

# -----------------------------------------------------------------------------
# 7. .env setup
# -----------------------------------------------------------------------------
Write-Step "Setting up .env"
& .venv\Scripts\python.exe scripts\setup_env.py
if ($LASTEXITCODE -ne 0) { Soft-Warn "env" ".env setup returned non-zero." }

# -----------------------------------------------------------------------------
# 8. Model prefetch (soft warn per model)
# -----------------------------------------------------------------------------
Write-Step "Prefetching model weights"
& .venv\Scripts\python.exe scripts\fetch_models.py
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "models" "One or more model downloads failed; check the log above."
}

# -----------------------------------------------------------------------------
# 9. Final readiness table
# -----------------------------------------------------------------------------
Write-Step "Final readiness check"
& .venv\Scripts\python.exe -m musicue.health.readiness --print-table

Write-Host ""
if ($script:soft_warnings.Count -eq 0) {
    Write-Host "Install complete. Double-click run.bat to launch MusiCue." -ForegroundColor Green
} else {
    Write-Host "Install completed with $($script:soft_warnings.Count) warning(s) (see above)." -ForegroundColor Yellow
    Write-Host "MusiCue is usable, but the readiness chip in the UI will show optional pieces as missing." -ForegroundColor Yellow
    Write-Host "Double-click run.bat to launch MusiCue."
}
exit 0
