"""
Data Collection Orchestrator
Coordinates data collection from multiple sources (LogHub + GitHub Actions).
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import config
from utils.logger import setup_logger
from scripts.download_loghub import LogHubDownloader
from scripts.fetch_runs import fetch_all_runs, save_csv
from scripts.parse_logs import GitHubActionsLogParser
from utils.log_parser import LogParser
from utils.feature_aligner import FeatureAligner

logger = setup_logger(__name__, "data_collector.log")


class DataCollector:
    """Orchestrates data collection from multiple sources."""
    
    def __init__(self):
        """Initialize data collector."""
        self.loghub_downloader = LogHubDownloader()
        self.github_parser = GitHubActionsLogParser()
    
    def collect_loghub_data(self, datasets: List[str] = None, limit: int = None) -> Dict[str, Path]:
        """
        Collect data from LogHub repository.
        
        Args:
            datasets: List of dataset names to download
            limit: Maximum number of datasets to download
        
        Returns:
            Dict mapping dataset names to their paths
        """
        logger.info("=== Collecting LogHub Data ===")
        
        if datasets is None:
            datasets = ['HDFS_v1', 'Spark', 'Hadoop']  # Default datasets
        
        if limit:
            datasets = datasets[:limit]
        
        results = self.loghub_downloader.download_all(datasets)
        
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"✓ Downloaded {success_count}/{len(datasets)} LogHub datasets")
        
        return results
    
    def collect_github_data(self, max_pages: int = 10, include_jobs: bool = True) -> pd.DataFrame:
        """
        Collect data from GitHub Actions.
        
        Args:
            max_pages: Maximum pages to fetch
            include_jobs: Whether to fetch job details
        
        Returns:
            DataFrame with GitHub Actions runs
        """
        logger.info("=== Collecting GitHub Actions Data ===")
        
        # Fetch runs
        runs = fetch_all_runs(include_jobs=include_jobs)
        
        # Save to CSV
        save_csv(runs)
        
        # Return as DataFrame
        df = pd.DataFrame(runs)
        logger.info(f"✓ Collected {len(df)} GitHub Actions runs")
        
        return df
    
    def parse_loghub_logs(self, dataset_name: str) -> pd.DataFrame:
        """
        Parse logs from a LogHub dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            DataFrame with parsed logs
        """
        dataset_dir = config.LOGHUB_DATA_DIR / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return pd.DataFrame()
        
        # Find log files
        log_files = list(dataset_dir.glob("**/*.log"))
        if not log_files:
            # Try other extensions
            log_files = list(dataset_dir.glob("**/*"))
            log_files = [f for f in log_files if f.is_file() and f.suffix in ['.log', '.txt', '']]
        
        logger.info(f"Found {len(log_files)} log files in {dataset_name}")
        
        all_logs = []
        for log_file in log_files[:10]:  # Limit to first 10 files for efficiency
            try:
                # Auto-detect source type from dataset name
                source_type = dataset_name.lower().split('_')[0]
                df = LogParser.parse_log_file(log_file, source_type)
                df['dataset'] = dataset_name
                df['file_name'] = log_file.name
                all_logs.append(df)
            except Exception as e:
                logger.error(f"Error parsing {log_file}: {e}")
        
        if all_logs:
            combined_df = pd.concat(all_logs, ignore_index=True)
            logger.info(f"Parsed {len(combined_df)} log entries from {dataset_name}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def create_labeled_dataset_from_loghub(self, datasets: List[str] = None) -> pd.DataFrame:
        """
        Create labeled dataset from LogHub logs.
        
        Since LogHub doesn't have explicit failure labels, we'll use error logs
        as proxy for failures.
        
        Args:
            datasets: List of dataset names to process
        
        Returns:
            DataFrame with features and labels
        """
        logger.info("=== Creating Labeled Dataset from LogHub ===")
        
        if datasets is None:
            datasets = self.loghub_downloader.list_downloaded()
        
        all_data = []
        
        for dataset_name in datasets:
            logger.info(f"Processing {dataset_name}...")
            df = self.parse_loghub_logs(dataset_name)
            
            if df.empty:
                continue
            
            # Create features from logs
            # Group by file or time window
            if 'file_name' in df.columns:
                grouped = df.groupby('file_name')
            else:
                # Create artificial groups
                df['group_id'] = df.index // 1000
                grouped = df.groupby('group_id')
            
            for group_name, group_df in grouped:
                features = {
                    'dataset': dataset_name,
                    'group': group_name,
                    'total_lines': len(group_df),
                    'num_errors': (group_df['level'] == 'ERROR').sum() if 'level' in group_df.columns else 0,
                    'num_warnings': (group_df['level'] == 'WARNING').sum() if 'level' in group_df.columns else 0,
                    'num_info': (group_df['level'] == 'INFO').sum() if 'level' in group_df.columns else 0,
                }
                
                # Label: 1 if has errors, 0 otherwise
                features['failed'] = 1 if features['num_errors'] > 0 else 0
                
                all_data.append(features)
        
        result_df = pd.DataFrame(all_data)
        logger.info(f"✓ Created labeled dataset with {len(result_df)} samples")
        
        # Map LogHub features to pipeline features
        result_df = FeatureAligner.map_loghub_to_pipeline_features(result_df)
        
        # Align features to expected schema
        result_df = FeatureAligner.align_features(result_df, fill_missing=True)
        
        # Save to processed data
        output_path = config.PROCESSED_DATA_DIR / "loghub_labeled.csv"
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path}")
        
        return result_df
    
    def create_labeled_dataset_from_github(self) -> pd.DataFrame:
        """
        Create labeled dataset from GitHub Actions data.
        
        Returns:
            DataFrame with features and labels
        """
        logger.info("=== Creating Labeled Dataset from GitHub Actions ===")
        
        # Load runs metadata
        if not config.RUNS_METADATA_CSV.exists():
            logger.error(f"Runs metadata not found: {config.RUNS_METADATA_CSV}")
            return pd.DataFrame()
        
        df = pd.read_csv(config.RUNS_METADATA_CSV)
        
        # Create binary label: 1 for failure, 0 for success
        df['failed'] = (df['conclusion'] == 'failure').astype(int)
        
        # Select relevant features
        feature_columns = [
            'id', 'run_number', 'event', 'status', 'conclusion',
            'build_duration', 'failed'
        ]
        
        # Add job-level features if available
        job_columns = ['num_jobs', 'num_jobs_failed', 'num_jobs_success', 'total_job_duration']
        for col in job_columns:
            if col in df.columns:
                feature_columns.append(col)
        
        result_df = df[feature_columns].copy()
        
        # Save to processed data
        output_path = config.PROCESSED_DATA_DIR / "github_labeled.csv"
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Created labeled dataset with {len(result_df)} samples")
        logger.info(f"✓ Saved to {output_path}")
        
        return result_df
    
    def collect_all_data(self, 
                        loghub_datasets: List[str] = None,
                        github_max_pages: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Collect all data from both sources.
        
        Args:
            loghub_datasets: List of LogHub datasets to download
            github_max_pages: Maximum pages to fetch from GitHub
        
        Returns:
            Dict with 'loghub' and 'github' DataFrames
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE DATA COLLECTION")
        logger.info("=" * 60)
        
        results = {}
        
        # Collect LogHub data
        try:
            self.collect_loghub_data(loghub_datasets)
            loghub_df = self.create_labeled_dataset_from_loghub()
            results['loghub'] = loghub_df
        except Exception as e:
            logger.error(f"Error collecting LogHub data: {e}")
            results['loghub'] = pd.DataFrame()
        
        # Collect GitHub data
        try:
            self.collect_github_data(max_pages=github_max_pages)
            github_df = self.create_labeled_dataset_from_github()
            results['github'] = github_df
        except Exception as e:
            logger.error(f"Error collecting GitHub data: {e}")
            results['github'] = pd.DataFrame()
        
        logger.info("=" * 60)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info(f"LogHub samples: {len(results.get('loghub', []))}")
        logger.info(f"GitHub samples: {len(results.get('github', []))}")
        logger.info("=" * 60)
        
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect data from all sources')
    parser.add_argument('--loghub-only', action='store_true', help='Collect only LogHub data')
    parser.add_argument('--github-only', action='store_true', help='Collect only GitHub data')
    parser.add_argument('--datasets', nargs='+', help='Specific LogHub datasets to download')
    parser.add_argument('--max-pages', type=int, default=10, help='Max GitHub pages to fetch')
    
    args = parser.parse_args()
    
    collector = DataCollector()
    
    if args.loghub_only:
        collector.collect_loghub_data(args.datasets)
        collector.create_labeled_dataset_from_loghub()
    elif args.github_only:
        collector.collect_github_data(max_pages=args.max_pages)
        collector.create_labeled_dataset_from_github()
    else:
        collector.collect_all_data(
            loghub_datasets=args.datasets,
            github_max_pages=args.max_pages
        )


if __name__ == "__main__":
    main()
