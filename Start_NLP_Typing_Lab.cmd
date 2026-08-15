@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run Install_or_Repair.cmd first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" tools\start_nlp_typing_lab.py %*
set "LAB_EXIT_CODE=%ERRORLEVEL%"

if not "%LAB_EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] The NLP typing lab could not start.
  pause
)

endlocal & exit /b %LAB_EXIT_CODE%
