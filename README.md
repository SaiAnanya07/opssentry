# OpsSentry 🛡️

**AI-Powered CI/CD Pipeline Failure Prediction System**

OpsSentry is an intelligent DevOps tool that predicts CI/CD pipeline failures before they occur using machine learning. It analyzes historical pipeline data, identifies failure patterns, and provides proactive alerts with actionable recommendations.

## 🌟 Features

- **Predictive Analytics**: ML models (Random Forest, XGBoost, Gradient Boosting) predict pipeline failures with high accuracy
- **Multi-Source Data Collection**: Integrates with GitHub Actions and LogHub datasets
- **Comprehensive Feature Engineering**: Temporal features, rolling statistics, failure rates, and interaction features
- **Intelligent Alerts**: Configurable notifications via Email, Slack, and Console
- **Actionable Recommendations**: Context-aware suggestions based on failure patterns
- **Web Dashboard**: Modern, responsive UI for monitoring and predictions
- **REST API**: Complete API for integration with existing DevOps tools

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   cd c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Edit `.env` file with your credentials:
   ```env
   GITHUB_TOKEN=your_github_token
   OWNER=your_github_username
   REPO=your_repository_name
   
   # Optional: Email notifications
   EMAIL_ENABLED=false
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SENDER_EMAIL=your_email@gmail.com
   SENDER_PASSWORD=your_app_password
   RECIPIENT_EMAILS=recipient1@example.com,recipient2@example.com
   
   # Optional: Slack notifications
   SLACK_ENABLED=false
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```

## ⚡ Quick Start

### Step 1: Collect Data

```bash
# Collect GitHub Actions data
python scripts/fetch_runs.py

# Download LogHub datasets (optional)
python scripts/download_loghub.py --dataset HDFS_v1

# Or collect all data at once
python scripts/data_collector.py
```

### Step 2: Preprocess Data

```bash
python scripts/preprocess.py --source github
```

### Step 3: Train Models

```bash
# Basic training
python scripts/train_model.py

# With hyperparameter tuning (takes longer)
python scripts/train_model.py --tune
```

### Step 4: Start Web Server

```bash
python app.py
```

Visit `http://localhost:5000` to access the dashboard!

## 📖 Usage

### Making Predictions via API

```python
import requests

# Prepare pipeline data
pipeline_data = {
    "name": "CI Pipeline",
    "event": "push",
    "build_duration": 450.0,
    "num_jobs": 3,
    "hour_of_day": 14,
    "is_weekend": 0
}

# Make prediction
response = requests.post(
    'http://localhost:5000/api/predict',
    json=pipeline_data
)

result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Failure Probability: {result['failure_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Using the Predictor Directly

```python
from models.predictor import Predictor

# Initialize predictor
predictor = Predictor()

# Make prediction
result = predictor.predict_pipeline_run({
    "name": "Deploy Pipeline",
    "build_duration": 600.0,
    "num_jobs": 5
})

print(result)
```

### Sending Custom Alerts

```python
from alert.alert_manager import AlertManager

manager = AlertManager()

# Process prediction and send alerts
manager.process_prediction(prediction_result)
```

## 🏗️ Architecture

```
OpsSentry/
├── config.py                 # Centralized configuration
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
│
├── scripts/                  # Data processing and training
│   ├── download_loghub.py   # LogHub dataset downloader
│   ├── fetch_runs.py        # GitHub Actions data fetcher
│   ├── parse_logs.py        # Log parser
│   ├── data_collector.py    # Data collection orchestrator
│   ├── feature_engineering.py # Feature engineering
│   ├── preprocess.py        # Data preprocessing pipeline
│   └── train_model.py       # Model training
│
├── models/                   # ML models and prediction
│   ├── model_evaluator.py   # Model evaluation utilities
│   └── predictor.py         # Prediction service
│
├── alert/                    # Alert and notification system
│   ├── alert_manager.py     # Alert management
│   ├── notifiers.py         # Email/Slack/Console notifiers
│   └── recommendation_engine.py # Recommendation generation
│
├── utils/                    # Utility modules
│   ├── logger.py            # Logging configuration
│   ├── data_validator.py   # Data validation
│   ├── visualization.py     # Plotting utilities
│   └── log_parser.py        # Log parsing utilities
│
├── data/                     # Data storage
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed datasets
│   ├── loghub/              # LogHub datasets
│   └── github/              # GitHub Actions data
│
├── models/                   # Saved models
│   ├── best_model.pkl       # Best performing model
│   ├── scaler.pkl           # Feature scaler
│   └── encoder.pkl          # Categorical encoders
│
├── templates/                # HTML templates
│   └── dashboard.html       # Main dashboard
│
└── static/                   # Static assets
    ├── css/
    │   └── style.css        # Dashboard styles
    └── js/
        └── dashboard.js     # Dashboard JavaScript
```

## 📡 API Documentation

### Endpoints

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "predictor_loaded": true,
  "version": "1.0.0"
}
```

#### `POST /api/predict`
Make prediction for a pipeline run.

**Request:**
```json
{
  "name": "CI Pipeline",
  "event": "push",
  "build_duration": 450.0,
  "num_jobs": 3
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "SUCCESS",
  "failure_probability": 0.25,
  "success_probability": 0.75,
  "risk_level": "LOW",
  "should_alert": false,
  "recommendation": "..."
}
```

#### `GET /api/metrics`
Get model performance metrics.

#### `GET /api/pipelines`
Get list of monitored pipelines.

#### `GET /api/alerts`
Get recent alerts.

#### `GET /api/stats`
Get overall statistics.

## ⚙️ Configuration

### Model Configuration

Edit `config.py` to customize model parameters:

```python
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 5,
        ...
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        ...
    }
}
```

### Alert Configuration

```python
ALERT_CONFIG = {
    "failure_probability_threshold": 0.70,
    "critical_threshold": 0.85,
    "warning_threshold": 0.70,
    "info_threshold": 0.50,
}
```

## 📊 Model Performance

The system trains multiple models and automatically selects the best performer:

- **Random Forest**: Robust to noise, handles feature importance well
- **XGBoost**: High accuracy for imbalanced datasets
- **Gradient Boosting**: Good baseline performance

Typical performance metrics:
- Accuracy: 85-90%
- Precision: 82-88%
- Recall: 85-92%
- F1-Score: 83-90%
- ROC-AUC: 88-95%

## 🔧 Troubleshooting

### Model not found error
```bash
# Train the model first
python scripts/train_model.py
```

### No data available
```bash
# Collect data
python scripts/data_collector.py
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📝 License

This project is part of a final year academic project.

## 🙏 Acknowledgments

- **LogHub** for providing comprehensive system log datasets
- **GitHub Actions** for CI/CD integration capabilities
- **scikit-learn, XGBoost** for machine learning frameworks

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ for DevOps Engineers**
