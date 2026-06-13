@echo off
REM ============================================================
REM  Build a portable BeeWings.exe for Windows (PyInstaller, onedir).
REM  Run this ON a Windows machine. Output: dist\BeeWings\BeeWings.exe
REM  Prerequisite: Python 3.10-3.13 in PATH.
REM ============================================================
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10-3.13 first.
    pause & exit /b 1
)

REM --- 1. build environment -----------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv || ( echo [ERROR] venv failed. & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

echo Installing BeeWings + build tools (downloads ~1 GB incl. PyTorch) ...
python -m pip install --upgrade pip
python -m pip install -e . || ( echo [ERROR] install failed. & pause & exit /b 1 )
python -m pip install pyinstaller || ( echo [ERROR] pyinstaller install failed. & pause & exit /b 1 )

REM --- 2. ML models (bundled into the exe folder) -------------
if not exist "checkpoints" mkdir checkpoints
set BASE=https://github.com/nurkal022/Beewings/releases/download/models-v1
if not exist "checkpoints\alpatov12.pt"  curl -L -o "checkpoints\alpatov12.pt"  "%BASE%/alpatov12.pt"
if not exist "checkpoints\tofilski19.pt" curl -L -o "checkpoints\tofilski19.pt" "%BASE%/tofilski19.pt"

REM --- 3. build ----------------------------------------------
echo.
echo Building with PyInstaller (this takes a few minutes) ...
pyinstaller --noconfirm --clean packaging\beewings.spec || ( echo [ERROR] build failed. & pause & exit /b 1 )

echo.
echo ============================================================
echo  Done.  Portable app: dist\BeeWings\BeeWings.exe
echo  Ship the WHOLE dist\BeeWings\ folder (exe + dependencies + checkpoints).
echo ============================================================
pause
