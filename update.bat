@echo off
REM ============================================================
REM  Update BeeWings source code to the latest version on GitHub.
REM  (editable install, so a restart of the app applies it)
REM ============================================================
cd /d "%~dp0"

echo Downloading latest version...
curl -L -o _update.zip https://github.com/nurkal022/Beewings/archive/refs/heads/main.zip ^
    || ( echo [ERROR] download failed. & pause & exit /b 1 )

echo Extracting...
tar -xf _update.zip || ( echo [ERROR] extract failed. & pause & exit /b 1 )

echo Updating files...
xcopy /E /I /Y Beewings-main\beewings beewings >nul
copy /Y Beewings-main\*.bat . >nul
copy /Y Beewings-main\*.md . >nul
if exist Beewings-main\assets xcopy /E /I /Y Beewings-main\assets assets >nul

rmdir /S /Q Beewings-main
del _update.zip

echo.
echo Done. Restart BeeWings (Desktop icon) to apply the update.
pause
