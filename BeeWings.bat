@echo off
REM ============================================================
REM  BeeWings - single launcher for end users.
REM  First run: installs everything. Later runs: just opens the app.
REM  Requires Python 3.10-3.13 installed (Add Python to PATH).
REM ============================================================
cd /d "%~dp0"

REM --- already installed? just launch (no console window) -------
if exist ".venv\Scripts\pythonw.exe" if exist "checkpoints\alpatov12.pt" goto launch

REM --- first-time setup ---------------------------------------
echo ============================================================
echo  First launch - setting up BeeWings (a few minutes)...
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.13 from
    echo https://www.python.org/downloads/  (tick "Add Python to PATH"),
    echo then double-click this file again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating environment...
    python -m venv .venv || ( echo [ERROR] venv failed. & pause & exit /b 1 )
)

echo Installing dependencies (downloads ~1 GB, please wait)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e . || ( echo [ERROR] install failed. & pause & exit /b 1 )

if not exist "checkpoints" mkdir checkpoints
set BASE=https://github.com/nurkal022/Beewings/releases/download/models-v1
if not exist "checkpoints\alpatov12.pt"  curl -L -o "checkpoints\alpatov12.pt"  "%BASE%/alpatov12.pt"
if not exist "checkpoints\tofilski19.pt" curl -L -o "checkpoints\tofilski19.pt" "%BASE%/tofilski19.pt"

REM --- desktop shortcut so people can launch with one click -----
echo Creating Desktop shortcut...
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\BeeWings.lnk');" ^
  "$lnk.TargetPath = (Resolve-Path '.venv\Scripts\pythonw.exe').Path;" ^
  "$lnk.Arguments = '-c \"from beewings.app.app import main; main()\"';" ^
  "$lnk.WorkingDirectory = (Get-Location).Path;" ^
  "$ico = Join-Path (Get-Location).Path 'assets\beewings.ico';" ^
  "$lnk.IconLocation = $(if (Test-Path $ico) { $ico } else { (Resolve-Path '.venv\Scripts\pythonw.exe').Path + ',0' });" ^
  "$lnk.Save()" 2>nul

echo.
echo Setup complete - a "BeeWings" icon is on your Desktop. Starting...
echo.

:launch
start "" ".venv\Scripts\pythonw.exe" -c "from beewings.app.app import main; main()"
