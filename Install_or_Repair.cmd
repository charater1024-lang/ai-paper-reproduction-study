@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE="
set "PYTHON_ARGS="
set "FORCE_CPU=0"

if "%~1"=="" goto :args_ok
if /i "%~1"=="--cpu" (
  set "FORCE_CPU=1"
  goto :args_ok
)
echo [ERROR] Unsupported option: %~1
echo Usage: Install_or_Repair.cmd [--cpu]
pause
exit /b 1

:args_ok

if exist ".venv\Scripts\python.exe" goto :check_existing_venv

rem Prefer a regular python.org CPython 3.12 installation.
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" ^
    tools\check_windows_python.py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :create_venv
  )
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 tools\check_windows_python.py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3.12"
    goto :create_venv
  )
)

for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE call :try_python "%%P"
if defined PYTHON_EXE goto :create_venv

echo [ERROR] 64-bit Python 3.12 was not found.
echo Install Python 3.12 from https://www.python.org/downloads/windows/
echo Enable "Add python.exe to PATH", then run this file again.
echo See docs\WINDOWS_SETUP.md for details.
pause
exit /b 1

:try_python
"%~1" tools\check_windows_python.py >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0

:create_venv
echo Creating a Python 3.12 virtual environment in .venv ...
"%PYTHON_EXE%" %PYTHON_ARGS% -m venv .venv
if errorlevel 1 goto :failed
goto :install

:check_existing_venv
".venv\Scripts\python.exe" tools\check_windows_python.py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] The existing .venv is not 64-bit Python 3.12.
  echo Rename the .venv folder as a backup, then run this file again.
  echo See docs\WINDOWS_SETUP.md for details.
  pause
  exit /b 1
)

:install
echo Updating pip and packaging tools ...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

rem Preserve an existing CUDA build. A fresh or CPU environment receives the
rem reproducible CPU wheel declared in requirements-windows.txt.
if "%FORCE_CPU%"=="1" (
  echo Explicit CPU restore requested ...
  goto :install_cpu_stack
)
".venv\Scripts\python.exe" tools\report_environment.py ^
  --torch-only --require-cuda-build >nul 2>nul
if errorlevel 1 goto :install_cpu_stack

echo Existing CUDA PyTorch detected; preserving that build ...
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :failed
goto :install_dev

:install_cpu_stack
echo Installing the Windows CPU learning environment ...
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-windows.txt
if errorlevel 1 goto :failed

:install_dev
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-dev.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --editable . --no-deps
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :failed

echo Registering the Jupyter kernel ...
".venv\Scripts\python.exe" -m ipykernel install ^
  --user ^
  --name ai-engineering-lab ^
  --display-name "Python (AI Engineering Lab)" ^
  --env PYTHONUTF8 1 ^
  --env PYTHONIOENCODING utf-8
if errorlevel 1 goto :failed

echo.
echo Setup completed with Python 3.12.
".venv\Scripts\python.exe" tools\report_environment.py --torch-only
echo Double-click Start_JupyterLab.cmd to begin.
echo NVIDIA users can optionally run Install_GPU_PyTorch.cmd.
pause
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Review the message above.
echo Help: docs\WINDOWS_SETUP.md
pause
exit /b 1
