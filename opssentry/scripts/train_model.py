"""
Model Training Module
Trains multiple ML models for pipeline failure prediction.
"""
import sys
import os
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
import joblib
import config
from utils.logger import setup_logger
from models.model_evaluator import ModelEvaluator

logger = setup_logger(__name__, "train_model.log")


class ModelTrainer:
    """Trains and tunes machine learning models."""
    
    def __init__(self):
        """Initialize trainer."""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed train/val/test data.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Loading preprocessed data...")
        
        # Load datasets
        train_df = pd.read_csv(config.TRAIN_DATA_CSV)
        val_df = pd.read_csv(config.VAL_DATA_CSV)
        test_df = pd.read_csv(config.TEST_DATA_CSV)
        
        # Separate features and labels
        label_col = 'failed'
        
        # Identify feature columns (exclude label and non-feature columns)
        exclude_cols = [label_col, 'id', 'html_url', 'logs_url', 'jobs_url', 'head_sha', 
                       'created_at', 'created_at_dt', 'name', 'event', 'status', 'conclusion']
        
        # Get all columns
        all_cols = train_df.columns.tolist()
        
        # Select only numeric columns for features
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        logger.info(f"Selected {len(feature_cols)} numeric features for training")
        
        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df[label_col].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values
        
        logger.info(f"✓ Loaded data:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val: {X_val.shape}")
        logger.info(f"  Test: {X_test.shape}")
        logger.info(f"  Features: {len(feature_cols)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info("Handling class imbalance with SMOTE...")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE only if there's imbalance
        if len(unique) > 1 and min(counts) / max(counts) < 0.5:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            unique, counts = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampled class distribution: {dict(zip(unique, counts))}")
            
            return X_resampled, y_resampled
        else:
            logger.info("Classes are balanced, skipping SMOTE")
            return X_train, y_train
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           tune_hyperparameters: bool = False) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
        
        Returns:
            Trained Random Forest model
        """
        logger.info("=" * 60)
        logger.info("Training Random Forest Classifier")
        logger.info("=" * 60)
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = config.HYPERPARAMETER_GRID['random_forest']
            
            grid_search = GridSearchCV(
                rf, param_grid,
                cv=config.CV_FOLDS,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        else:
            logger.info("Training with default parameters...")
            model = RandomForestClassifier(**config.MODEL_CONFIG['random_forest'])
            model.fit(X_train, y_train)
        
        # Evaluate on validation set
        evaluator = ModelEvaluator()
        val_metrics = evaluator.evaluate_model(
            model, X_val, y_val,
            model_name="RandomForest",
            save_plots=True
        )
        
        self.models['RandomForest'] = {
            'model': model,
            'metrics': val_metrics
        }
        
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     tune_hyperparameters: bool = False) -> XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
        
        Returns:
            Trained XGBoost model
        """
        logger.info("=" * 60)
        logger.info("Training XGBoost Classifier")
        logger.info("=" * 60)
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            
            xgb = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
            param_grid = config.HYPERPARAMETER_GRID['xgboost']
            
            grid_search = GridSearchCV(
                xgb, param_grid,
                cv=config.CV_FOLDS,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        else:
            logger.info("Training with default parameters...")
            model = XGBClassifier(**config.MODEL_CONFIG['xgboost'], 
                                 use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
        
        # Evaluate on validation set
        evaluator = ModelEvaluator()
        val_metrics = evaluator.evaluate_model(
            model, X_val, y_val,
            model_name="XGBoost",
            save_plots=True
        )
        
        self.models['XGBoost'] = {
            'model': model,
            'metrics': val_metrics
        }
        
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Trained Gradient Boosting model
        """
        logger.info("=" * 60)
        logger.info("Training Gradient Boosting Classifier")
        logger.info("=" * 60)
        
        # Check for NaN values and fill them
        if np.isnan(X_train).any():
            logger.warning("NaN values detected in training data, filling with 0")
            X_train = np.nan_to_num(X_train, nan=0.0)
        if np.isnan(X_val).any():
            logger.warning("NaN values detected in validation data, filling with 0")
            X_val = np.nan_to_num(X_val, nan=0.0)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        evaluator = ModelEvaluator()
        val_metrics = evaluator.evaluate_model(
            model, X_val, y_val,
            model_name="GradientBoosting",
            save_plots=True
        )
        
        self.models['GradientBoosting'] = {
            'model': model,
            'metrics': val_metrics
        }
        
        return model
    
    def select_best_model(self, metric: str = 'f1') -> Tuple[Any, str]:
        """
        Select the best model based on validation metrics.
        
        Args:
            metric: Metric to use for selection
        
        Returns:
            Tuple of (best_model, model_name)
        """
        logger.info(f"Selecting best model based on {metric}...")
        
        best_score = -1
        best_name = None
        best_model = None
        
        for name, model_info in self.models.items():
            score = model_info['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model_info['model']
        
        self.best_model = best_model
        self.best_model_name = best_name
        
        logger.info(f"✓ Best model: {best_name} ({metric}={best_score:.4f})")
        
        return best_model, best_name
    
    def save_model(self, model: Any, model_name: str, path: Path = None):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name of the model
            path: Path to save model
        """
        if path is None:
            path = config.MODELS_DIR / f"{model_name.lower()}.pkl"
        
        joblib.dump(model, path)
        logger.info(f"✓ Saved {model_name} to {path}")
    
    def train_all_models(self, tune_hyperparameters: bool = False,
                        handle_imbalance: bool = True) -> Dict[str, Any]:
        """
        Train all models and select the best one.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            handle_imbalance: Whether to handle class imbalance with SMOTE
        
        Returns:
            Dict with training results
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
        # Handle class imbalance
        if handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)
        
        # Train models
        self.train_random_forest(X_train, y_train, X_val, y_val, tune_hyperparameters)
        self.train_xgboost(X_train, y_train, X_val, y_val, tune_hyperparameters)
        self.train_gradient_boosting(X_train, y_train, X_val, y_val)
        
        # Compare models
        evaluator = ModelEvaluator()
        models_metrics = {name: info['metrics'] for name, info in self.models.items()}
        evaluator.compare_models(models_metrics)
        
        # Select best model
        best_model, best_name = self.select_best_model(metric='f1')
        
        # Evaluate best model on test set
        logger.info("\n" + "=" * 60)
        logger.info(f"FINAL EVALUATION ON TEST SET - {best_name}")
        logger.info("=" * 60)
        
        test_metrics = evaluator.evaluate_model(
            best_model, X_test, y_test,
            model_name=f"{best_name}_Test",
            save_plots=True
        )
        
        # Analyze feature importance
        evaluator.analyze_feature_importance(
            best_model,
            self.feature_names,
            model_name=best_name
        )
        
        # Save best model
        self.save_model(best_model, best_name, config.BEST_MODEL_PATH)
        
        # Save all models
        for name, info in self.models.items():
            self.save_model(info['model'], name)
        
        logger.info("=" * 60)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return {
            'best_model': best_model,
            'best_model_name': best_name,
            'test_metrics': test_metrics,
            'all_models': self.models
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models for pipeline failure prediction')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--no-smote', action='store_true', help='Skip SMOTE for class imbalance')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(
        tune_hyperparameters=args.tune,
        handle_imbalance=not args.no_smote
    )
    
    print(f"\n✓ Training complete!")
    print(f"Best model: {results['best_model_name']}")
    print(f"Test F1 score: {results['test_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
