"""
Model Evaluation Utilities
Provides comprehensive model evaluation metrics and visualizations.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from typing import Dict, Any, Tuple
from pathlib import Path
import config
from utils.logger import setup_logger
from utils.visualization import Visualizer

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluates model performance with various metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
        
        Returns:
            Dict of metric names to values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dict of metric names to values
            model_name: Name of the model
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{model_name} Performance Metrics")
        logger.info(f"{'=' * 60}")
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name.upper():15s}: {value:.4f}")
        
        logger.info(f"{'=' * 60}\n")
    
    @staticmethod
    def evaluate_model(model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      model_name: str = "Model",
                      save_plots: bool = True,
                      output_dir: Path = None) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            save_plots: Whether to save visualization plots
            output_dir: Directory to save plots
        
        Returns:
            Dict of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle single-class case
            if proba.shape[1] == 1:
                logger.warning("Model only predicts one class - using predictions as probabilities")
                y_pred_proba = y_pred
            else:
                y_pred_proba = proba[:, 1]
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print metrics
        ModelEvaluator.print_metrics(metrics, model_name)
        
        # Generate visualizations
        if save_plots:
            if output_dir is None:
                output_dir = config.MODELS_DIR
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            Visualizer.plot_confusion_matrix(
                cm,
                classes=['Success', 'Failure'],
                title=f'{model_name} - Confusion Matrix',
                save_path=output_dir / f'{model_name}_confusion_matrix.png'
            )
            
            # ROC curve
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                Visualizer.plot_roc_curve(
                    fpr, tpr, metrics['roc_auc'],
                    title=f'{model_name} - ROC Curve',
                    save_path=output_dir / f'{model_name}_roc_curve.png'
                )
        
        # Classification report
        logger.info(f"\n{model_name} Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, 
                                                 target_names=['Success', 'Failure']))
        
        return metrics
    
    @staticmethod
    def compare_models(models_metrics: Dict[str, Dict[str, float]],
                      save_plot: bool = True,
                      output_path: Path = None):
        """
        Compare multiple models' performance.
        
        Args:
            models_metrics: Dict of {model_name: {metric_name: value}}
            save_plot: Whether to save comparison plot
            output_path: Path to save plot
        """
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        # Create comparison DataFrame
        df = pd.DataFrame(models_metrics).T
        logger.info("\n" + df.to_string())
        
        # Find best model for each metric
        logger.info("\n" + "=" * 60)
        logger.info("BEST MODELS PER METRIC")
        logger.info("=" * 60)
        
        for metric in df.columns:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            logger.info(f"{metric.upper():15s}: {best_model} ({best_value:.4f})")
        
        # Plot comparison
        if save_plot:
            if output_path is None:
                output_path = config.MODELS_DIR / 'model_comparison.png'
            
            Visualizer.plot_metrics_comparison(
                models_metrics,
                title='Model Performance Comparison',
                save_path=output_path
            )
        
        return df
    
    @staticmethod
    def analyze_feature_importance(model: Any,
                                   feature_names: list,
                                   model_name: str = "Model",
                                   top_n: int = 20,
                                   save_plot: bool = True,
                                   output_path: Path = None) -> pd.DataFrame:
        """
        Analyze and visualize feature importance.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
            save_plot: Whether to save plot
            output_path: Path to save plot
        
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"{model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        logger.info(f"Analyzing feature importance for {model_name}...")
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Log top features
        logger.info(f"\nTop {top_n} Most Important Features:")
        logger.info("\n" + importance_df.head(top_n).to_string(index=False))
        
        # Plot
        if save_plot:
            if output_path is None:
                output_path = config.MODELS_DIR / f'{model_name}_feature_importance.png'
            
            Visualizer.plot_feature_importance(
                feature_names,
                importances,
                top_n=top_n,
                title=f'{model_name} - Feature Importance',
                save_path=output_path
            )
        
        return importance_df


def main():
    """Test evaluation functions."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        model, X_test, y_test,
        model_name="RandomForest",
        save_plots=False
    )
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(20)]
    evaluator.analyze_feature_importance(
        model, feature_names,
        model_name="RandomForest",
        save_plot=False
    )


if __name__ == "__main__":
    main()
