@echo off
REM OpsSentry Quick Start Script for Windows

echo ========================================
echo OpsSentry Quick Start
echo ========================================

REM Set PYTHONPATH to current directory
set PYTHONPATH=%CD%

echo.
echo Step 1: Fetching GitHub Actions data...
python scripts\fetch_runs.py --max-pages 5
if errorlevel 1 (
    echo ERROR: Failed to fetch GitHub data
    pause
    exit /b 1
)

echo.
echo Step 2: Creating labeled datasets...
python scripts\data_collector.py --github-only --max-pages 5
if errorlevel 1 (
    echo ERROR: Failed to create datasets
    pause
    exit /b 1
)

echo.
echo Step 3: Preprocessing data...
python scripts\preprocess.py --source github
if errorlevel 1 (
    echo ERROR: Failed to preprocess data
    pause
    exit /b 1
)

echo.
echo Step 4: Training models...
python scripts\train_model.py
if errorlevel 1 (
    echo ERROR: Failed to train models
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Pipeline complete
echo ========================================
echo.
echo Next steps:
echo 1. Start web server: python app.py
echo 2. Open http://localhost:5000 in browser
echo.
pause
