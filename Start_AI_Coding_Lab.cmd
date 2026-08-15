@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0AI_Coding_Practice_Center.ps1"
set "HUB_EXIT_CODE=%ERRORLEVEL%"

if not "%HUB_EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] The AI Coding Practice Center could not start.
  echo Check that AI_Coding_Practice_Center.ps1 is in this folder.
  pause
)

exit /b %HUB_EXIT_CODE%
