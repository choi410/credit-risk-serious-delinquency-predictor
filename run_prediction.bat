@echo off
chcp 65001 > nul
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist "%USERPROFILE%\Downloads\GiveMeSomeCredit\.venv\Scripts\python.exe" set "PYTHON_EXE=%USERPROFILE%\Downloads\GiveMeSomeCredit\.venv\Scripts\python.exe"

"%PYTHON_EXE%" predict_csv.py --input "input\customer_data.csv" --output-dir "result"

echo.
echo 결과 CSV는 result 폴더에 저장되었습니다.
pause
