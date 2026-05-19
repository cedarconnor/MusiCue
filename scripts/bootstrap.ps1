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
# 3. PyTorch with CUDA (must come before core install so dependents see it)
# -----------------------------------------------------------------------------
# Pin a known-good CUDA 12.1 build. We use --index-url (not --extra-index-url)
# so uv resolves torch exclusively from PyTorch's wheel index; otherwise uv
# happily picks a higher-versioned CPU-only torch wheel from PyPI.
Write-Step "Installing PyTorch with CUDA 12.1"
& uv pip install `
    "torch==2.5.1+cu121" `
    "torchaudio==2.5.1+cu121" `
    "torchvision==0.20.1+cu121" `
    --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA install failed (exit $LASTEXITCODE)" }

# -----------------------------------------------------------------------------
# 4. Core install (hard fail)
# -----------------------------------------------------------------------------
Write-Step "Installing core dependencies (this may take several minutes)"
& uv pip install -e ".[dev,ui,midi,osc]" basic-pitch "setuptools<81"
if ($LASTEXITCODE -ne 0) { throw "Core install failed (exit $LASTEXITCODE)" }

# -----------------------------------------------------------------------------
# 5. CLAP (soft warn)
# -----------------------------------------------------------------------------
Write-Step "Installing CLAP (optional)"
& uv pip install -e ".[clap]"
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "clap" "CLAP install failed; semantic labels will be disabled."
}

# -----------------------------------------------------------------------------
# 6. All-In-One (soft warn — historically painful on Windows)
# -----------------------------------------------------------------------------
# allin1 doesn't list madmom as a transitive dep, but it imports it at
# runtime. madmom's last PyPI release (0.16.1) uses `from collections
# import MutableSequence`, which Python 3.10+ removed — so we install
# madmom from the upstream master branch (0.17.dev0). That requires
# Cython at build time and --no-build-isolation so uv doesn't try to
# resolve Cython for a transient build env.
Write-Step "Installing Cython (build dep for madmom)"
& uv pip install Cython
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "cython" "Cython install failed; madmom build will fail too."
}

# --no-cache forces uv to rebuild madmom against the CURRENT venv numpy
# instead of reusing a stale cached wheel from a previous install. Without
# this flag, a fresh rebuild can end up with a madmom whose Cython modules
# were compiled against numpy 1.x while the venv has numpy 2.x, producing
# "numpy.dtype size changed" ABI errors at runtime.
Write-Step "Installing madmom from git (needs VS Build Tools on Windows)"
& uv pip install --no-build-isolation --no-cache "git+https://github.com/CPJKU/madmom.git"
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "madmom" "madmom build failed. Install 'Visual Studio Build Tools' (Desktop dev with C++) from visualstudio.microsoft.com and re-run install.bat. Without madmom, All-In-One can't load and beats fall back to librosa (no section detection)."
}

Write-Step "Installing All-In-One (optional)"
& uv pip install allin1
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "allin1" "All-In-One install failed; beat detection will use the librosa fallback (no sections)."
}

# natten is imported by allin1/models/dinat.py but is NOT declared as an
# allin1 dependency, so it doesn't install transitively. Install it
# explicitly; the legacy-compat patch below makes the natten >=0.17 API
# expose the procedural functions allin1 expects.
Write-Step "Installing NATTEN (allin1 runtime dep)"
& uv pip install natten
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "natten" "NATTEN install failed; allin1 will not be able to load its DINAT model. Try 'uv pip install natten' manually after the install."
}

# allin1's NATTEN attention modules import procedural functions
# (natten1dqkrpb, natten1dav, natten2dqkrpb, natten2dav) that were
# removed in natten >= 0.17. Patch the installed natten with a pure-
# PyTorch shim so allin1 keeps working without pinning to ancient natten.
Write-Step "Patching NATTEN with legacy-compat shim"
& uv run python (Join-Path $PSScriptRoot "patch_natten.py")
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "natten" "NATTEN patch failed; allin1 may fail to import. See scripts/patch_natten.py."
}

# -----------------------------------------------------------------------------
# 6.5. Restore PyTorch CUDA wheels
# -----------------------------------------------------------------------------
# Steps 4-6 install packages whose transitive deps include torch/torchaudio.
# uv re-resolves the dep graph and is happy to "upgrade" our pinned cu121
# wheels to the latest CPU-only wheels from PyPI (e.g. torch 2.5.1+cu121 →
# torch 2.12.0+cpu). That silently breaks CUDA AND mismatches torchvision's
# C++ ops (so laion_clap / anything touching torchvision fails too).
#
# Reinstall the cu121 trio with --no-deps so we overwrite whatever PyPI
# wheels uv pulled in. uv's cache holds the cu121 wheels from step 3, so
# this is a fast metadata-only operation on subsequent runs.
Write-Step "Restoring PyTorch CUDA 12.1 wheels (overwrites any CPU upgrades)"
& uv pip install --reinstall --no-deps `
    "torch==2.5.1+cu121" `
    "torchaudio==2.5.1+cu121" `
    "torchvision==0.20.1+cu121" `
    --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "torch-cuda" "Failed to restore CUDA torch wheels; the pipeline will run on CPU. Re-run install.bat to retry."
}

# -----------------------------------------------------------------------------
# 7. ffmpeg
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
# 8. .env setup
# -----------------------------------------------------------------------------
Write-Step "Setting up .env"
& .venv\Scripts\python.exe scripts\setup_env.py
if ($LASTEXITCODE -ne 0) { Soft-Warn "env" ".env setup returned non-zero." }

# -----------------------------------------------------------------------------
# 9. Model prefetch (soft warn per model)
# -----------------------------------------------------------------------------
Write-Step "Prefetching model weights"
& .venv\Scripts\python.exe scripts\fetch_models.py
if ($LASTEXITCODE -ne 0) {
    Soft-Warn "models" "One or more model downloads failed; check the log above."
}

# -----------------------------------------------------------------------------
# 10. Final readiness table
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
