"""
Enhanced GitHub Actions Run Fetcher
Fetches workflow runs and job-level details from GitHub Actions API.
"""
import os
import time
import csv
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import config
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__, "fetch_runs.log")

TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = os.getenv("OWNER")
REPO = os.getenv("REPO")
PER_PAGE = 100           # GitHub max per_page is 100
MAX_PAGES = 50           # adjust if you need more history
OUT_CSV = config.RUNS_METADATA_CSV

if not TOKEN or not OWNER or not REPO:
    raise SystemExit("Please set GITHUB_TOKEN, OWNER, REPO in .env")

HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs"


def calculate_duration(started_at: str, completed_at: str) -> float:
    """
    Calculate duration in seconds between two timestamps.
    
    Args:
        started_at: Start timestamp (ISO format)
        completed_at: Completion timestamp (ISO format)
    
    Returns:
        Duration in seconds
    """
    if not started_at or not completed_at:
        return 0.0
    
    try:
        start = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        end = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
        return (end - start).total_seconds()
    except Exception as e:
        logger.warning(f"Error calculating duration: {e}")
        return 0.0


def fetch_job_details(jobs_url: str) -> dict:
    """
    Fetch job-level details for a workflow run.
    
    Args:
        jobs_url: URL to fetch jobs from
    
    Returns:
        Dict with job statistics
    """
    try:
        resp = requests.get(jobs_url, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            jobs = data.get("jobs", [])
            
            job_stats = {
                'num_jobs': len(jobs),
                'num_jobs_completed': sum(1 for j in jobs if j.get('status') == 'completed'),
                'num_jobs_failed': sum(1 for j in jobs if j.get('conclusion') == 'failure'),
                'num_jobs_success': sum(1 for j in jobs if j.get('conclusion') == 'success'),
                'total_job_duration': sum(
                    calculate_duration(j.get('started_at'), j.get('completed_at'))
                    for j in jobs
                ),
                'job_names': [j.get('name') for j in jobs],
            }
            
            return job_stats
        else:
            logger.warning(f"Failed to fetch jobs: {resp.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching job details: {e}")
        return {}


def fetch_all_runs(include_jobs: bool = True):
    """
    Fetch all workflow runs with optional job details.
    
    Args:
        include_jobs: Whether to fetch job-level details
    
    Returns:
        List of run dictionaries
    """
    runs = []
    logger.info(f"Fetching workflow runs from {OWNER}/{REPO}")
    
    for page in tqdm(range(1, MAX_PAGES + 1), desc="Fetching pages"):
        params = {"per_page": PER_PAGE, "page": page}
        resp = requests.get(BASE, headers=HEADERS, params=params)
        
        if resp.status_code == 200:
            data = resp.json()
            batch = data.get("workflow_runs", [])
            
            if not batch:
                logger.info(f"No more runs found at page {page}")
                break
            
            for run in batch:
                run_data = {
                    "id": run.get("id"),
                    "run_number": run.get("run_number"),
                    "name": run.get("name"),
                    "event": run.get("event"),
                    "status": run.get("status"),
                    "conclusion": run.get("conclusion"),
                    "created_at": run.get("created_at"),
                    "updated_at": run.get("updated_at"),
                    "run_started_at": run.get("run_started_at"),
                    "head_sha": run.get("head_sha"),
                    "html_url": run.get("html_url"),
                    "jobs_url": run.get("jobs_url"),
                    "logs_url": run.get("logs_url"),
                }
                
                # Calculate build duration
                run_data['build_duration'] = calculate_duration(
                    run.get("run_started_at"),
                    run.get("updated_at")
                )
                
                # Fetch job details if requested
                if include_jobs and run.get("jobs_url"):
                    job_stats = fetch_job_details(run.get("jobs_url"))
                    run_data.update(job_stats)
                    time.sleep(0.3)  # Rate limiting
                
                runs.append(run_data)
            
            # Stop if fewer than per_page (last page)
            if len(batch) < PER_PAGE:
                logger.info(f"Reached last page at page {page}")
                break
            
            # Gentle pause to avoid rate limiting
            time.sleep(0.6)
            
        elif resp.status_code == 403 and 'X-RateLimit-Remaining' in resp.headers and resp.headers.get('X-RateLimit-Remaining') == '0':
            reset = int(resp.headers.get('X-RateLimit-Reset', 0))
            sleep_for = max(1, reset - int(time.time()))
            logger.warning(f"Rate limit reached. Sleeping for {sleep_for} seconds")
            time.sleep(sleep_for + 2)
        else:
            logger.error(f"Error fetching page {page}: {resp.status_code} {resp.text}")
            break
    
    logger.info(f"Fetched {len(runs)} total workflow runs")
    return runs


def save_csv(runs):
    """
    Save runs to CSV file.
    
    Args:
        runs: List of run dictionaries
    """
    if not runs:
        logger.warning("No runs found.")
        return
    
    # Convert to DataFrame for better handling
    df = pd.DataFrame(runs)
    
    # Flatten job_names list to string
    if 'job_names' in df.columns:
        df['job_names'] = df['job_names'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    
    # Ensure output directory exists
    config.GITHUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUT_CSV, index=False)
    logger.info(f"✓ Saved {len(runs)} runs to {OUT_CSV}")
    
    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total runs: {len(df)}")
    if 'conclusion' in df.columns:
        logger.info(f"Successful runs: {(df['conclusion'] == 'success').sum()}")
        logger.info(f"Failed runs: {(df['conclusion'] == 'failure').sum()}")
    if 'build_duration' in df.columns:
        logger.info(f"Average build duration: {df['build_duration'].mean():.2f} seconds")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch GitHub Actions workflow runs')
    parser.add_argument('--no-jobs', action='store_true', help='Skip fetching job details')
    parser.add_argument('--max-pages', type=int, default=MAX_PAGES, help='Maximum pages to fetch')
    
    args = parser.parse_args()
    
    MAX_PAGES = args.max_pages
    runs = fetch_all_runs(include_jobs=not args.no_jobs)
    save_csv(runs)

