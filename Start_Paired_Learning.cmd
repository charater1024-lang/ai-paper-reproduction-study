@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv was not found.
  echo Run Install_or_Repair.cmd first.
  pause
  exit /b 1
)

echo ============================================================
echo  AI Engineering Lab - Exercise + Solution (Two Windows)
echo  Left: exercise / Right: solution
echo ============================================================
".venv\Scripts\python.exe" tools\start_paired_lab.py %*
set "LAB_EXIT_CODE=%ERRORLEVEL%"

if not "%LAB_EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] Paired learning could not start.
  echo Check artifacts\paired_jupyter_*.log or run Install_or_Repair.cmd.
  pause
)
endlocal & exit /b %LAB_EXIT_CODE%
