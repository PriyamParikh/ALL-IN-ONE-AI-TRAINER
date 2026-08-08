@echo off
setlocal
cd /d "%~dp0"
python all_in_one_ai_5in1_launcher.py
if errorlevel 1 (
    echo.
    echo All in One AI stopped with an error.
    pause
)
