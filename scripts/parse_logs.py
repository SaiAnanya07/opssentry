"""
Enhanced Log Parser for GitHub Actions
Downloads, parses, and extracts features from GitHub Actions logs.
"""
import os
import requests
import zipfile
import io
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import config
from utils.logger import setup_logger
from utils.log_parser import LogParser

load_dotenv()
logger = setup_logger(__name__, "parse_logs.log")

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}


class GitHubActionsLogParser:
    """Parse and analyze GitHub Actions logs."""
    
    def __init__(self, owner: str = None, repo: str = None):
        """
        Initialize parser.
        
        Args:
            owner: GitHub repository owner
            repo: GitHub repository name
        """
        self.owner = owner or config.GITHUB_OWNER
        self.repo = repo or config.GITHUB_REPO
        self.token = TOKEN
        self.headers = HEADERS
    
    def download_and_extract_logs(self, run_id: int, out_dir: Path = None) -> Path:
        """
        Download and extract logs for a specific workflow run.
        
        Args:
            run_id: GitHub Actions run ID
            out_dir: Output directory for extracted logs
        
        Returns:
            Path to extracted logs directory
        """
        if out_dir is None:
            out_dir = config.LOGS_ARCHIVE_DIR / str(run_id)
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/logs"
        logger.info(f"Downloading logs for run {run_id} from {url}")
        
        r = requests.get(url, headers=self.headers, stream=True)
        
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(out_dir)
            logger.info(f"✓ Extracted logs to {out_dir}")
            return out_dir
        elif r.status_code == 410:
            logger.warning(f"Logs for run {run_id} have expired (410 Gone)")
            return None
        else:
            logger.error(f"Failed to download logs: {r.status_code} {r.text}")
            return None
    
    def parse_log_directory(self, log_dir: Path) -> pd.DataFrame:
        """
        Parse all log files in a directory.
        
        Args:
            log_dir: Directory containing log files
        
        Returns:
            DataFrame with parsed log entries
        """
        all_logs = []
        
        log_files = list(log_dir.glob("**/*.txt"))
        logger.info(f"Found {len(log_files)} log files in {log_dir}")
        
        for log_file in log_files:
            try:
                df = LogParser.parse_github_actions_log(log_file)
                df['job_name'] = log_file.stem
                all_logs.append(df)
            except Exception as e:
                logger.error(f"Error parsing {log_file}: {e}")
        
        if all_logs:
            combined_df = pd.concat(all_logs, ignore_index=True)
            logger.info(f"Parsed {len(combined_df)} total log entries")
            return combined_df
        else:
            return pd.DataFrame()
    
    def extract_test_results(self, log_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract test results from parsed logs.
        
        Args:
            log_df: DataFrame with parsed log entries
        
        Returns:
            Dict with test statistics
        """
        test_stats = {
            'num_tests_run': 0,
            'num_tests_passed': 0,
            'num_tests_failed': 0,
            'num_tests_skipped': 0,
            'test_duration': 0.0,
        }
        
        # Look for common test result patterns
        for message in log_df['message']:
            # pytest patterns
            if 'passed' in message.lower() and 'failed' in message.lower():
                # Example: "5 passed, 2 failed in 10.5s"
                match = re.search(r'(\d+)\s+passed.*?(\d+)\s+failed', message)
                if match:
                    test_stats['num_tests_passed'] = int(match.group(1))
                    test_stats['num_tests_failed'] = int(match.group(2))
                    test_stats['num_tests_run'] = test_stats['num_tests_passed'] + test_stats['num_tests_failed']
            
            # Jest patterns
            if 'Tests:' in message:
                # Example: "Tests: 2 failed, 8 passed, 10 total"
                match = re.search(r'Tests:\s+(\d+)\s+failed,\s+(\d+)\s+passed', message)
                if match:
                    test_stats['num_tests_failed'] = int(match.group(1))
                    test_stats['num_tests_passed'] = int(match.group(2))
                    test_stats['num_tests_run'] = test_stats['num_tests_passed'] + test_stats['num_tests_failed']
            
            # Duration patterns
            duration_match = re.search(r'in\s+([\d\.]+)s', message)
            if duration_match:
                test_stats['test_duration'] = float(duration_match.group(1))
        
        return test_stats
    
    def extract_error_info(self, log_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract error information from logs.
        
        Args:
            log_df: DataFrame with parsed log entries
        
        Returns:
            Dict with error statistics
        """
        error_df = log_df[log_df['level'] == 'ERROR']
        
        error_info = {
            'num_errors': len(error_df),
            'error_types': [],
            'error_messages': [],
        }
        
        # Extract error types and messages
        for message in error_df['message'].head(10):  # Limit to first 10 errors
            error_info['error_messages'].append(message[:200])  # Truncate long messages
            
            # Classify error types
            if 'timeout' in message.lower():
                error_info['error_types'].append('timeout')
            elif 'connection' in message.lower():
                error_info['error_types'].append('connection')
            elif 'memory' in message.lower():
                error_info['error_types'].append('memory')
            elif 'syntax' in message.lower():
                error_info['error_types'].append('syntax')
            else:
                error_info['error_types'].append('other')
        
        return error_info
    
    def extract_stage_durations(self, log_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract duration for each stage/step.
        
        Args:
            log_df: DataFrame with parsed log entries
        
        Returns:
            Dict mapping stage names to durations
        """
        stage_durations = {}
        
        # Group by job name
        for job_name in log_df['job_name'].unique():
            job_logs = log_df[log_df['job_name'] == job_name]
            
            # Try to extract timestamps
            if 'timestamp' in job_logs.columns:
                timestamps = pd.to_datetime(job_logs['timestamp'], errors='coerce')
                valid_timestamps = timestamps.dropna()
                
                if len(valid_timestamps) >= 2:
                    duration = (valid_timestamps.max() - valid_timestamps.min()).total_seconds()
                    stage_durations[job_name] = duration
        
        return stage_durations
    
    def extract_features_from_logs(self, run_id: int, log_dir: Path = None) -> Dict[str, Any]:
        """
        Extract comprehensive features from logs for a run.
        
        Args:
            run_id: GitHub Actions run ID
            log_dir: Directory containing logs (if already downloaded)
        
        Returns:
            Dict with extracted features
        """
        # Download logs if not provided
        if log_dir is None:
            log_dir = self.download_and_extract_logs(run_id)
            if log_dir is None:
                return {}
        
        # Parse logs
        log_df = self.parse_log_directory(log_dir)
        
        if log_df.empty:
            logger.warning(f"No logs parsed for run {run_id}")
            return {}
        
        # Extract features
        features = {
            'run_id': run_id,
            'total_log_lines': len(log_df),
            'num_jobs': log_df['job_name'].nunique() if 'job_name' in log_df.columns else 0,
        }
        
        # Test results
        test_stats = self.extract_test_results(log_df)
        features.update(test_stats)
        
        # Error information
        error_info = self.extract_error_info(log_df)
        features.update(error_info)
        
        # Stage durations
        stage_durations = self.extract_stage_durations(log_df)
        features['stage_durations'] = stage_durations
        features['total_stage_duration'] = sum(stage_durations.values())
        
        # Log level distribution
        level_counts = log_df['level'].value_counts().to_dict()
        features['num_info_logs'] = level_counts.get('INFO', 0)
        features['num_warning_logs'] = level_counts.get('WARNING', 0)
        features['num_error_logs'] = level_counts.get('ERROR', 0)
        
        logger.info(f"Extracted {len(features)} features for run {run_id}")
        return features


def download_and_extract_logs(owner: str, repo: str, run_id: int, out_dir: str = "data/logs"):
    """
    Legacy function for backward compatibility.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        run_id: GitHub Actions run ID
        out_dir: Output directory
    """
    parser = GitHubActionsLogParser(owner, repo)
    return parser.download_and_extract_logs(run_id, Path(out_dir))


if __name__ == "__main__":
    # Example usage
    parser = GitHubActionsLogParser()
    
    # Test with a sample run ID
    sample_run_id = 123456  # Replace with actual run ID
    features = parser.extract_features_from_logs(sample_run_id)
    
    if features:
        print("Extracted features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
