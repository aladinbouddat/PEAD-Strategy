@echo off
cd /d %~dp0

REM Check if venv exists
IF NOT EXIST "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
IF NOT EXIST ".installed" (
    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
    echo. > .installed
)

REM Run the strategy
python src\pead_strategy.py

pause
