@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo       ALL IN ONE AI - INSTALLATION
echo ============================================================
echo.
python --version
if errorlevel 1 (
    echo Python was not found. Install 64-bit Python 3.11 and try again.
    pause
    exit /b 1
)
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
python -m pip install --no-cache-dir -r requirements_all_in_one_ai.txt
if errorlevel 1 goto :failed
python -m pip check
if errorlevel 1 goto :failed
echo.
echo Installation completed successfully.
echo Run Verify_All_in_One_AI.bat, then Run_All_in_One_AI.bat.
pause
exit /b 0
:failed
echo.
echo Installation did not complete. Read the error above.
pause
exit /b 1
