@echo off
REM Launch BeeWings (run setup_windows.bat first).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Environment not set up. Run setup_windows.bat first.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
beewings
