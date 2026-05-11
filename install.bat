@echo off
setlocal
cd /d "%~dp0"

echo Installing MusiCue. This will take 5-20 minutes depending on your internet speed.
echo.

powershell -ExecutionPolicy Bypass -NoProfile -File "scripts\bootstrap.ps1"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo Install finished. Press any key to close this window.
) else (
    echo Install failed with code %RC%. Scroll up to see the error.
)
pause >nul
endlocal
exit /b %RC%
