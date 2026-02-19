@echo off
REM Start OpsSentry Web Server

set PYTHONPATH=%CD%

echo ========================================
echo Starting OpsSentry Web Server
echo ========================================
echo.
echo Dashboard will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py
