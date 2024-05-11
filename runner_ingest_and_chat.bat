@echo off
SET venv_dir=required\venv
SET python=%venv_dir%\Scripts\python.exe

REM Check if the virtual environment directory exists
IF EXIST "%venv_dir%\Scripts\activate.bat" (
    ECHO Virtual environment found. Activating...
    CALL %venv_dir%\Scripts\activate.bat
) ELSE (
    ECHO Creating virtual environment...
    python -m venv %venv_dir%
    CALL %venv_dir%\Scripts\activate.bat
)

echo Please choose an option:
echo 1. Run ingest.py
echo 2. Run run_localGPT.py
echo.

choice /C 12 /N /M "Enter your choice (1-2):"

if errorlevel 2 goto option2
if errorlevel 1 goto option1

:option1
echo Running ingest.py with CUDA...
python ingest.py --device_type cuda
goto end

:option2
echo Running run_localGPT.py with CUDA...
python run_localGPT.py --device_type cuda
goto end

:end
echo Script finished.
pause
REM Pause the command window
cmd /k 