@echo off
chcp 65001 > nul
cd /d "%~dp0"

set "PYTHON_EXE=python"
set "DATA_DIR=%USERPROFILE%\Downloads\GiveMeSomeCredit"
if exist "%DATA_DIR%\.venv\Scripts\python.exe" set "PYTHON_EXE=%DATA_DIR%\.venv\Scripts\python.exe"

"%PYTHON_EXE%" train_and_export_model.py --data-dir "%DATA_DIR%" --model-dir "models"
pause
