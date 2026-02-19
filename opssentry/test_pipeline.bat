@echo off
REM OpsSentry Complete Pipeline Test Script
REM Tests the entire pipeline with synthetic data

echo ========================================
echo OpsSentry Pipeline Test
echo ========================================
echo.

REM Set Python path
set PYTHONPATH=%CD%

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python path set to: %PYTHONPATH%
echo.

REM Step 1: Generate Synthetic Data
echo [1/5] Generating synthetic data...
python create_synthetic_data_enhanced.py --samples 1000 --type both
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate synthetic data
    echo Trying to use existing data...
)
echo.

REM Step 2: Download LogHub Dataset (optional, continue if fails)
echo [2/5] Downloading LogHub dataset (optional)...
python scripts\download_loghub.py --dataset HDFS_v1
if %errorlevel% neq 0 (
    echo WARNING: LogHub download failed, continuing with synthetic data...
)
echo.

REM Step 3: Create Labeled Dataset
echo [3/5] Creating labeled dataset...
python scripts\data_collector.py --loghub-only
if %errorlevel% neq 0 (
    echo WARNING: LogHub data collection failed
    echo Checking for synthetic data...
    if not exist "data\processed\synthetic_labeled.csv" (
        echo ERROR: No data available for training
        pause
        exit /b 1
    )
)
echo.

REM Step 4: Preprocess Data
echo [4/5] Preprocessing data...
python scripts\preprocess.py --source loghub
if %errorlevel% neq 0 (
    echo Trying with synthetic data...
    python scripts\preprocess.py --source github
    if %errorlevel% neq 0 (
        echo ERROR: Failed to preprocess data
        pause
        exit /b 1
    )
)
echo.

REM Step 5: Train Models
echo [5/5] Training models...
python scripts\train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to train models
    pause
    exit /b 1
)
echo.

echo ========================================
echo Pipeline Test Complete!
echo ========================================
echo.
echo Models trained successfully!
echo.
echo Next steps:
echo 1. Run 'start_server.bat' to start the web server
echo 2. Open http://localhost:5000 in your browser
echo 3. Test predictions via the dashboard
echo.
pause
