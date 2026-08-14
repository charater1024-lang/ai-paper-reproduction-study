@echo off
setlocal
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
echo  AI Engineering Lab - JupyterLab
echo  Folder: %CD%
echo  Stop: press Ctrl+C twice in this window
echo ============================================================
".venv\Scripts\python.exe" -m jupyterlab ^
  "notebooks\00_environment_and_jupyterlab.ipynb" ^
  --ServerApp.root_dir="%CD%"

if errorlevel 1 (
  echo.
  echo JupyterLab stopped with an error.
  echo Run Install_or_Repair.cmd and try again.
  pause
)
endlocal
