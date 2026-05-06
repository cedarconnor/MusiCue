@echo off
setlocal

REM Kill any existing musicue UI server on port 8765 before starting a new one.
REM Uses PowerShell because netstat-based approaches are fragile across cmd
REM versions and Windows builds.
powershell -NoProfile -Command "$p = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; if ($p) { foreach ($c in $p) { Write-Host ('Stopping existing server on port 8765 (PID ' + $c.OwningProcess + ')'); Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep -Milliseconds 500 }"

REM Resolve to the directory containing this .bat file so the editable
REM musicue install picks up the right source tree.
cd /d "%~dp0"

REM Boot the UI. Forwards any extra args (e.g. --port 9000, --no-open).
python -m musicue ui %*

endlocal
