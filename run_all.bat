@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

python "01_preprocessing.py"
if errorlevel 1 exit /b 1

python "python 02_case_representation.py"
if errorlevel 1 exit /b 1

python "python 03_retrieval.py"
if errorlevel 1 exit /b 1

python "python 04_predict.py"
if errorlevel 1 exit /b 1

python "python 05_evaluation.py"
if errorlevel 1 exit /b 1

echo Pipeline completed successfully.
endlocal
