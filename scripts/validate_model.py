"""
Model Validation Script for Jenkins CI/CD
Validates trained models against performance thresholds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import config
from utils.logger import setup_logger

logger = setup_logger(__name__, "validate_model.log")

# Performance thresholds
THRESHOLDS = {
    'accuracy': 0.75,
    'precision': 0.70,
    'recall': 0.70,
    'f1': 0.70,
    'roc_auc': 0.75
}


def validate_model_exists():
    """Check if trained model exists."""
    if not config.BEST_MODEL_PATH.exists():
        logger.error(f"Model not found at {config.BEST_MODEL_PATH}")
        logger.error("Please train a model first using: python scripts/train_model.py")
        return False
    return True


def load_model():
    """Load the trained model."""
    try:
        model = joblib.load(config.BEST_MODEL_PATH)
        logger.info(f"✓ Model loaded from {config.BEST_MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def load_test_data():
    """Load test dataset."""
    try:
        if not config.TEST_DATA_CSV.exists():
            logger.error(f"Test data not found at {config.TEST_DATA_CSV}")
            return None, None
        
        test_df = pd.read_csv(config.TEST_DATA_CSV)
        
        # Separate features and labels
        label_col = 'failed'
        exclude_cols = [label_col, 'id', 'html_url', 'logs_url', 'jobs_url', 'head_sha', 
                       'created_at', 'created_at_dt', 'name', 'event', 'status', 'conclusion']
        
        numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values
        
        logger.info(f"✓ Test data loaded: {X_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return None, None


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        logger.info("=" * 60)
        logger.info("MODEL VALIDATION RESULTS")
        logger.info("=" * 60)
        
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        return None


def check_thresholds(metrics):
    """Check if metrics meet minimum thresholds."""
    passed = True
    
    logger.info("=" * 60)
    logger.info("THRESHOLD VALIDATION")
    logger.info("=" * 60)
    
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric, 0.0)
        status = "✓ PASS" if value >= threshold else "✗ FAIL"
        
        if value < threshold:
            passed = False
        
        logger.info(f"{metric.upper()}: {value:.4f} (threshold: {threshold:.4f}) - {status}")
    
    return passed


def main():
    """Main validation function."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL VALIDATION")
    logger.info("=" * 60)
    
    # Check if model exists
    if not validate_model_exists():
        sys.exit(1)
    
    # Load model
    model = load_model()
    if model is None:
        sys.exit(1)
    
    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        sys.exit(1)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    if metrics is None:
        sys.exit(1)
    
    # Check thresholds
    passed = check_thresholds(metrics)
    
    logger.info("=" * 60)
    if passed:
        logger.info("✓ MODEL VALIDATION PASSED")
        logger.info("Model meets all performance thresholds")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("✗ MODEL VALIDATION FAILED")
        logger.error("Model does not meet minimum performance thresholds")
        logger.error("Please retrain the model or adjust thresholds")
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
