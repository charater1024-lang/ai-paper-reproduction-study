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
echo  AI Fields x Landmark Papers - Paired Practice Lab
echo  7 fields / 10 papers each / exercise + solution
echo ============================================================
".venv\Scripts\python.exe" tools\start_field_paper_lab.py %*
set "LAB_EXIT_CODE=%ERRORLEVEL%"

if not "%LAB_EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] Field paper lab could not start.
  echo Check artifacts\paired_jupyter_*.log or run Install_or_Repair.cmd.
  pause
)
endlocal & exit /b %LAB_EXIT_CODE%
