@echo off
chcp 65001 > nul
cd /d "%~dp0"

set "PYTHON_EXE=python"
set "DATA_DIR=%USERPROFILE%\Downloads\GiveMeSomeCredit"
if exist "%DATA_DIR%\.venv\Scripts\python.exe" set "PYTHON_EXE=%DATA_DIR%\.venv\Scripts\python.exe"

"%PYTHON_EXE%" train_and_export_model.py --data-dir "%DATA_DIR%" --model-dir "models"
if errorlevel 1 goto :error

"%PYTHON_EXE%" train_clustering.py --data-dir "%DATA_DIR%" --model-dir "models" --result-dir "result" --figure-dir "figures"
if errorlevel 1 goto :error

echo 분류 모델과 군집 모델 재학습이 완료되었습니다.
pause
exit /b 0

:error
echo 모델 재학습 중 오류가 발생했습니다.
pause
exit /b 1
