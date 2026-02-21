"""
OpsSentry Configuration Module
Centralized configuration for paths, API endpoints, model parameters, and features.
"""
import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Data Subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGHUB_DATA_DIR = DATA_DIR / "loghub"
GITHUB_DATA_DIR = DATA_DIR / "github"
LOGS_ARCHIVE_DIR = DATA_DIR / "logs_archive"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RAW_DATA_DIR, 
                  PROCESSED_DATA_DIR, LOGHUB_DATA_DIR, GITHUB_DATA_DIR, 
                  LOGS_ARCHIVE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# GitHub API Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER = os.getenv("OWNER", "")
GITHUB_REPO = os.getenv("REPO", "")
GITHUB_API_BASE = "https://api.github.com"

# LogHub Configuration
LOGHUB_DATASETS = {
    "HDFS_v1": "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
    "HDFS_v2": "https://zenodo.org/records/8196385/files/HDFS_v2.zip?download=1",
    "Spark": "https://zenodo.org/records/8196385/files/Spark.tar.gz?download=1",
    "Hadoop": "https://zenodo.org/records/8196385/files/Hadoop.zip?download=1",
    "Zookeeper": "https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1",
    "OpenStack": "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1",
}

# Model Configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "categorical_features": [
        "status", "conclusion", "event", "workflow_name", 
        "branch", "job_name", "step_name"
    ],
    "numerical_features": [
        "build_duration", "num_failed_tests", "num_passed_tests",
        "commit_count", "file_changes", "lines_added", "lines_deleted",
        "cpu_usage", "memory_usage", "stage_duration"
    ],
    "derived_features": [
        "failure_rate", "avg_build_time", "commit_frequency",
        "hour_of_day", "day_of_week", "is_weekend"
    ],
}

# Data Split Configuration
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Alert Configuration
ALERT_CONFIG = {
    "failure_probability_threshold": 0.70,  # Alert if >70% failure probability
    "critical_threshold": 0.85,  # Critical alert if >85%
    "warning_threshold": 0.70,  # Warning alert if >70%
    "info_threshold": 0.50,  # Info alert if >50%
}

# Notification Configuration (Optional)
EMAIL_CONFIG = {
    "enabled": os.getenv("EMAIL_ENABLED", "false").lower() == "true",
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "sender_email": os.getenv("SENDER_EMAIL", ""),
    "sender_password": os.getenv("SENDER_PASSWORD", ""),
    "recipient_emails": os.getenv("RECIPIENT_EMAILS", "").split(","),
}

SLACK_CONFIG = {
    "enabled": os.getenv("SLACK_ENABLED", "false").lower() == "true",
    "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
}

# Flask Configuration
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model Evaluation Metrics
EVALUATION_METRICS = [
    "accuracy", "precision", "recall", "f1", "roc_auc"
]

# Hyperparameter Tuning Configuration
HYPERPARAMETER_GRID = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
}

# Cross-validation Configuration
CV_FOLDS = 5

# File Paths
RUNS_METADATA_CSV = GITHUB_DATA_DIR / "runs_metadata.csv"
PROCESSED_FEATURES_CSV = PROCESSED_DATA_DIR / "features.csv"
TRAIN_DATA_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_CSV = PROCESSED_DATA_DIR / "test.csv"

# Model Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"

# Visualization Configuration
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (12, 8)
DPI = 100
