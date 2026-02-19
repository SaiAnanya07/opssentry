"""
Build Script for OpsSentry
Automates the complete setup and execution pipeline.
"""
import subprocess
import sys
from pathlib import Path
import config
from utils.logger import setup_logger

logger = setup_logger(__name__, "build.log")


def run_command(cmd, description):
    """Run a command and log the output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        if e.stdout:
            logger.error(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return False


def main():
    """Main build pipeline."""
    logger.info("\n" + "="*60)
    logger.info("OPSSENTRY BUILD PIPELINE")
    logger.info("="*60)
    
    steps = [
        # Step 1: Data Collection
        {
            "cmd": f"{sys.executable} scripts/fetch_runs.py --max-pages 5",
            "desc": "Collecting GitHub Actions data",
            "optional": False
        },
        {
            "cmd": f"{sys.executable} scripts/download_loghub.py --dataset HDFS_v1",
            "desc": "Downloading LogHub dataset",
            "optional": True
        },
        {
            "cmd": f"{sys.executable} scripts/data_collector.py --github-only --max-pages 5",
            "desc": "Creating labeled datasets",
            "optional": False
        },
        
        # Step 2: Preprocessing
        {
            "cmd": f"{sys.executable} scripts/preprocess.py --source github",
            "desc": "Preprocessing data",
            "optional": False
        },
        
        # Step 3: Model Training
        {
            "cmd": f"{sys.executable} scripts/train_model.py",
            "desc": "Training models",
            "optional": False
        },
    ]
    
    failed_steps = []
    
    for i, step in enumerate(steps, 1):
        logger.info(f"\n[{i}/{len(steps)}] {step['desc']}...")
        
        success = run_command(step['cmd'], step['desc'])
        
        if not success:
            if step['optional']:
                logger.warning(f"Optional step failed, continuing...")
            else:
                failed_steps.append(step['desc'])
                logger.error(f"Critical step failed: {step['desc']}")
                logger.error("Build pipeline stopped.")
                break
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("BUILD PIPELINE SUMMARY")
    logger.info("="*60)
    
    if not failed_steps:
        logger.info("✓ All steps completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start the web server: python app.py")
        logger.info("2. Open http://localhost:5000 in your browser")
        logger.info("3. Make predictions using the dashboard or API")
    else:
        logger.error(f"✗ {len(failed_steps)} step(s) failed:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        logger.error("\nPlease fix the errors and run the build again.")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()
