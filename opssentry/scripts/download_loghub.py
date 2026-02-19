"""
LogHub Dataset Downloader
Downloads and extracts datasets from the LogHub repository for training.
"""
import requests
import zipfile
import tarfile
import io
from pathlib import Path
from tqdm import tqdm
import config
from utils.logger import setup_logger

logger = setup_logger(__name__, "download_loghub.log")

class LogHubDownloader:
    """Downloads and manages LogHub datasets."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded datasets
        """
        self.output_dir = output_dir or config.LOGHUB_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, dataset_name: str, url: str = None) -> Path:
        """
        Download a specific dataset from LogHub.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'HDFS_v1')
            url: Download URL (if None, uses config)
        
        Returns:
            Path to extracted dataset directory
        """
        if url is None:
            if dataset_name not in config.LOGHUB_DATASETS:
                raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config.LOGHUB_DATASETS.keys())}")
            url = config.LOGHUB_DATASETS[dataset_name]
        
        logger.info(f"Downloading {dataset_name} from {url}")
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Read content with progress bar
        content = io.BytesIO()
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dataset_name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
                pbar.update(len(chunk))
        
        content.seek(0)
        
        # Extract based on file type
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        if url.endswith('.zip'):
            logger.info(f"Extracting ZIP archive to {dataset_dir}")
            with zipfile.ZipFile(content) as zf:
                zf.extractall(dataset_dir)
        elif url.endswith('.tar.gz') or url.endswith('.tgz'):
            logger.info(f"Extracting TAR.GZ archive to {dataset_dir}")
            with tarfile.open(fileobj=content, mode='r:gz') as tf:
                tf.extractall(dataset_dir)
        else:
            logger.warning(f"Unknown archive format for {url}")
            # Save as-is
            output_file = dataset_dir / "data"
            with open(output_file, 'wb') as f:
                f.write(content.read())
        
        logger.info(f"✓ Successfully downloaded and extracted {dataset_name}")
        return dataset_dir
    
    def download_all(self, datasets: list = None) -> dict:
        """
        Download multiple datasets.
        
        Args:
            datasets: List of dataset names (if None, downloads all)
        
        Returns:
            Dict mapping dataset names to their paths
        """
        if datasets is None:
            datasets = list(config.LOGHUB_DATASETS.keys())
        
        results = {}
        for dataset_name in datasets:
            try:
                dataset_dir = self.download_dataset(dataset_name)
                results[dataset_name] = dataset_dir
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                results[dataset_name] = None
        
        return results
    
    def list_downloaded(self) -> list:
        """
        List all downloaded datasets.
        
        Returns:
            List of dataset names
        """
        if not self.output_dir.exists():
            return []
        
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download LogHub datasets')
    parser.add_argument('--dataset', type=str, help='Specific dataset to download')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--limit', type=int, help='Limit number of datasets to download')
    
    args = parser.parse_args()
    
    downloader = LogHubDownloader()
    
    if args.list:
        print("Available datasets:")
        for name in config.LOGHUB_DATASETS.keys():
            print(f"  - {name}")
        print(f"\nAlready downloaded:")
        for name in downloader.list_downloaded():
            print(f"  - {name}")
    elif args.all:
        datasets = list(config.LOGHUB_DATASETS.keys())
        if args.limit:
            datasets = datasets[:args.limit]
        logger.info(f"Downloading {len(datasets)} datasets...")
        results = downloader.download_all(datasets)
        
        success = sum(1 for v in results.values() if v is not None)
        logger.info(f"Downloaded {success}/{len(datasets)} datasets successfully")
    elif args.dataset:
        downloader.download_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
