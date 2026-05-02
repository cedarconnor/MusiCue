# scripts/setup_env.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Installing uv..." -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

Write-Host "Creating .venv with Python 3.11..." -ForegroundColor Cyan
uv venv .venv --python 3.11

Write-Host "Activating venv..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch + torchaudio (CUDA 12.4 wheel)..." -ForegroundColor Cyan
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing MusiCue + dev deps..." -ForegroundColor Cyan
uv pip install -e ".[dev]"

Write-Host "Verifying CUDA..." -ForegroundColor Cyan
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
