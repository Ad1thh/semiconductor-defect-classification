@echo off
cd /d "%~dp0"
echo ==============================================================================
echo   STARTING REAL-TIME SEMICONDUCTOR WAFER DEFECT INFERENCE SERVER
echo ==============================================================================
echo.
python server.py
pause
