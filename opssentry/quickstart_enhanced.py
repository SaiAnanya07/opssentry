"""
Quick Start Script for OpsSentry
Automatically sets up and runs the complete pipeline with synthetic data.
"""
import sys
import os
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.logger import setup_logger

logger = setup_logger(__name__, "quickstart.log")


def run_command(cmd, description, critical=True):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    
    try:
        # Set PYTHONPATH in environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
            env=env  # Pass environment with PYTHONPATH
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        logger.info(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        logger.error(f"[FAIL] {description} failed: {e}")
        
        if critical:
            print(f"\nCritical step failed. Exiting...")
            return False
        else:
            print(f"\nNon-critical step failed. Continuing...")
            return True


def main():
    """Run the complete OpsSentry setup."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              OpsSentry Quick Start                        ║
    ║       CI/CD Pipeline Failure Prediction System            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    logger.info("="*60)
    logger.info("Starting OpsSentry Quick Start")
    logger.info("="*60)
    
    # Step 1: Generate synthetic data
    if not run_command(
        "python create_synthetic_data_enhanced.py --samples 1000 --type both",
        "Step 1/4: Generating synthetic training data",
        critical=False
    ):
        print("Continuing with existing data...")
    
    # Step 2: Try to download LogHub data (optional)
    run_command(
        "python scripts\\download_loghub.py --dataset HDFS_v1",
        "Step 2/4: Downloading LogHub dataset (optional)",
        critical=False
    )
    
    # Step 3: Create labeled dataset
    if not run_command(
        "python scripts\\data_collector.py --loghub-only",
        "Step 3/4: Creating labeled dataset",
        critical=False
    ):
        print("LogHub data not available, using synthetic data...")
    
    # Step 4: Preprocess data
    success = run_command(
        "python scripts\\preprocess.py --source loghub",
        "Step 4/4: Preprocessing data (LogHub)",
        critical=False
    )
    
    if not success:
        # Try with GitHub/synthetic data
        if not run_command(
            "python scripts\\preprocess.py --source github",
            "Step 4/4: Preprocessing data (Synthetic)",
            critical=True
        ):
            print("\n✗ Preprocessing failed. Cannot continue.")
            return False
    
    # Step 5: Train models
    if not run_command(
        "python scripts\\train_model.py",
        "Step 5/5: Training machine learning models",
        critical=True
    ):
        print("\n✗ Model training failed.")
        return False
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              [OK] Setup Complete!                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Next Steps:
    
    1. Start the web server:
       python app.py
    
    2. Open your browser to:
       http://localhost:5000
    
    3. Test the prediction API:
       - Use the web dashboard
       - Or test via command line:
         python models/predictor.py
    
    For more information, see README.md or QUICKSTART.md
    """)
    
    logger.info("="*60)
    logger.info("OpsSentry Quick Start completed successfully!")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
