@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "TORCH_VERSION=2.13.0"
set "CUDA_FLAVOR=cu130"
if not "%~1"=="" set "CUDA_FLAVOR=%~1"

if /i "%CUDA_FLAVOR%"=="cu126" goto :flavor_ok
if /i "%CUDA_FLAVOR%"=="cu130" goto :flavor_ok
if /i "%CUDA_FLAVOR%"=="cu132" goto :flavor_ok
echo [ERROR] Unsupported CUDA wheel channel: %CUDA_FLAVOR%
echo Usage: Install_GPU_PyTorch.cmd [cu126^|cu130^|cu132]
pause
exit /b 1

:flavor_ok
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run Install_or_Repair.cmd first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" tools\check_windows_python.py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] The project requires 64-bit Python 3.12 on Windows.
  echo Run Install_or_Repair.cmd after backing up the incompatible .venv.
  pause
  exit /b 1
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [ERROR] An NVIDIA display driver was not detected.
  echo The CPU environment remains available.
  pause
  exit /b 1
)

echo NVIDIA driver information:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo.
echo Installing official PyTorch %TORCH_VERSION% from the %CUDA_FLAVOR% channel ...
echo The wheel contains the CUDA runtime; a separate CUDA Toolkit is not required.
echo This is a large download and may take several minutes.
".venv\Scripts\python.exe" -m pip install ^
  --upgrade ^
  --force-reinstall ^
  --no-deps ^
  torch==%TORCH_VERSION% ^
  --index-url https://download.pytorch.org/whl/%CUDA_FLAVOR%
if errorlevel 1 goto :gpu_failed

".venv\Scripts\python.exe" tools\verify_cuda_runtime.py ^
  --expected-flavor %CUDA_FLAVOR%
if errorlevel 1 goto :gpu_failed

".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :gpu_failed

echo.
echo GPU setup completed. The notebooks will select CUDA automatically.
pause
exit /b 0

:gpu_failed
echo.
echo [ERROR] GPU installation or verification failed.
echo The installed PyTorch package can still run CPU operations in most cases.
echo To replace it with the explicit CPU build, run:
echo   Install_or_Repair.cmd --cpu
echo For an older NVIDIA driver, update the driver or try:
echo   Install_GPU_PyTorch.cmd cu126
echo Help: docs\WINDOWS_SETUP.md
pause
exit /b 1
