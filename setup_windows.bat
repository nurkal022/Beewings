@echo off
REM ============================================================
REM  BeeWings - one-click setup for Windows
REM  Prerequisite: Python 3.10-3.13 installed and added to PATH
REM  (install from python.org, tick "Add Python to PATH")
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo  BeeWings setup
echo ============================================================
echo.

REM --- 1. check Python ----------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.10-3.13 from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during installation, then re-run this file.
    pause
    exit /b 1
)
echo Using Python:
python --version
echo.

REM --- 2. create virtual environment --------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
) else (
    echo Virtual environment .venv already exists - reusing.
)

call ".venv\Scripts\activate.bat"

REM --- 3. install dependencies --------------------------------
echo.
echo Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo Installing BeeWings and dependencies (this downloads ~1 GB incl. PyTorch, be patient) ...
pip install -e .
if errorlevel 1 ( echo [ERROR] dependency installation failed. & pause & exit /b 1 )

REM --- 4. download ML models ----------------------------------
echo.
echo Downloading ML models into checkpoints\ ...
if not exist "checkpoints" mkdir checkpoints

set BASE=https://github.com/nurkal022/Beewings/releases/download/models-v1

if not exist "checkpoints\alpatov12.pt" (
    echo   alpatov12.pt ...
    curl -L -o "checkpoints\alpatov12.pt" "%BASE%/alpatov12.pt"
) else ( echo   alpatov12.pt already present - skipping. )

if not exist "checkpoints\tofilski19.pt" (
    echo   tofilski19.pt ...
    curl -L -o "checkpoints\tofilski19.pt" "%BASE%/tofilski19.pt"
) else ( echo   tofilski19.pt already present - skipping. )

REM --- 5. sanity check ----------------------------------------
echo.
for %%F in ("checkpoints\alpatov12.pt" "checkpoints\tofilski19.pt") do (
    if not exist "%%~F" ( echo [ERROR] model %%~F missing - download failed. & pause & exit /b 1 )
)

echo.
echo ============================================================
echo  Setup complete!  Launch the app with:  run_windows.bat
echo ============================================================
pause
