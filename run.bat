@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run install.bat first.
    pause >nul
    exit /b 1
)

REM Prepend the bundled ffmpeg dir if present.
if exist "vendor\ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%~dp0vendor\ffmpeg\bin;%PATH%"
)

echo Starting MusiCue on http://127.0.0.1:8000 ...
echo Close this window to stop the server.
echo.

call .venv\Scripts\activate.bat
python -m uvicorn musicue.ui.server:create_app --factory --host 127.0.0.1 --port 8000

endlocal
