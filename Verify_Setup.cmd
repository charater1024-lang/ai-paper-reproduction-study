@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run Install_or_Repair.cmd first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" tools\report_environment.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" tools\validate_all_notebooks.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" tools\validate_paired_notebooks.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -X utf8 tools\validate_paper_reproductions.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -X utf8 tools\validate_field_reproductions.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -X utf8 tools\validate_portfolio_quality.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -X utf8 tools\validate_api_explanations.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" tools\start_paired_lab.py 13 --smoke-test
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" tools\start_paired_lab.py --paper 09 --smoke-test
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -X utf8 tools\start_field_paper_lab.py nlp_llm 03 --smoke-test
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -X utf8 tools\start_field_paper_lab.py ^
  distillation_compression 00 --smoke-test
if errorlevel 1 goto :failed
echo.
echo Setup, tests, notebooks, 20-paper and 70-paper tracks, data, and Jupyter server passed.
pause
exit /b 0

:failed
echo.
echo [ERROR] Verification failed. Run Install_or_Repair.cmd.
pause
exit /b 1
