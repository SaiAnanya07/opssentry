"""
Prediction Service
Loads trained models and makes predictions on new pipeline data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
import joblib
import config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Predictor:
    """Makes predictions using trained models."""
    
    def __init__(self, model_path: Path = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model (uses best model if None)
        """
        if model_path is None:
            model_path = config.BEST_MODEL_PATH
        
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = []
        
        self.load_model()
        self.load_preprocessors()
    
    def load_model(self):
        """Load trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("✓ Model loaded successfully")
    
    def load_preprocessors(self):
        """Load scaler and encoders."""
        if config.SCALER_PATH.exists():
            self.scaler = joblib.load(config.SCALER_PATH)
            logger.info("✓ Loaded scaler")
        
        if config.ENCODER_PATH.exists():
            self.encoders = joblib.load(config.ENCODER_PATH)
            logger.info("✓ Loaded encoders")
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data (DataFrame or dict)
        
        Returns:
            Preprocessed feature array
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Encode categorical features
        if self.encoders:
            for col, encoder in self.encoders.items():
                if col in df.columns:
                    # Handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1
                    )
        
        # Select and order features
        # Note: In production, feature_names should be saved with the model
        # For now, we'll use all numerical columns except label
        exclude_cols = ['failed', 'id', 'html_url', 'logs_url', 'jobs_url', 'head_sha']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure all required features are present
        X = df[feature_cols].values
        
        # Normalize if scaler is available
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Make binary predictions (0 or 1).
        
        Args:
            data: Input data
        
        Returns:
            Array of predictions
        """
        X = self.preprocess_input(data)
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Predict probabilities for each class.
        
        Args:
            data: Input data
        
        Returns:
            Array of probability predictions
        """
        X = self.preprocess_input(data)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            return probabilities
        else:
            logger.warning("Model does not support probability predictions")
            # Return binary predictions as probabilities
            predictions = self.model.predict(X)
            return np.column_stack([1 - predictions, predictions])
    
    def predict_with_confidence(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions with confidence scores and risk levels.
        
        Args:
            data: Input data
        
        Returns:
            List of prediction results with metadata
        """
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            failure_probability = proba[1]  # Probability of failure (class 1)
            
            # Determine risk level
            if failure_probability >= config.ALERT_CONFIG['critical_threshold']:
                risk_level = 'CRITICAL'
            elif failure_probability >= config.ALERT_CONFIG['warning_threshold']:
                risk_level = 'WARNING'
            elif failure_probability >= config.ALERT_CONFIG['info_threshold']:
                risk_level = 'INFO'
            else:
                risk_level = 'LOW'
            
            result = {
                'prediction': int(pred),
                'prediction_label': 'FAILURE' if pred == 1 else 'SUCCESS',
                'failure_probability': float(failure_probability),
                'success_probability': float(proba[0]),
                'risk_level': risk_level,
                'should_alert': failure_probability >= config.ALERT_CONFIG['failure_probability_threshold']
            }
            
            results.append(result)
        
        return results
    
    def predict_pipeline_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict outcome for a single pipeline run.
        
        Args:
            run_data: Dict with pipeline run information
        
        Returns:
            Prediction result with recommendations
        """
        logger.info(f"Making prediction for run: {run_data.get('id', 'unknown')}")
        
        # Make prediction
        result = self.predict_with_confidence(run_data)[0]
        
        # Add input metadata
        result['run_id'] = run_data.get('id')
        result['run_name'] = run_data.get('name')
        
        # Generate recommendation if high risk
        if result['should_alert']:
            result['recommendation'] = self.generate_recommendation(run_data, result)
        
        logger.info(f"Prediction: {result['prediction_label']} "
                   f"(probability: {result['failure_probability']:.2%}, "
                   f"risk: {result['risk_level']})")
        
        return result
    
    def generate_recommendation(self, run_data: Dict[str, Any], 
                               prediction: Dict[str, Any]) -> str:
        """
        Generate actionable recommendation based on prediction.
        
        Args:
            run_data: Pipeline run data
            prediction: Prediction result
        
        Returns:
            Recommendation string
        """
        recommendations = []
        
        # Analyze key features
        if run_data.get('build_duration', 0) > 600:  # > 10 minutes
            recommendations.append("Build duration is high. Consider optimizing build steps or caching dependencies.")
        
        if run_data.get('num_jobs_failed', 0) > 0:
            recommendations.append(f"Previous jobs failed ({run_data['num_jobs_failed']}). Review job logs for errors.")
        
        if run_data.get('failure_rate', 0) > 0.5:
            recommendations.append("High historical failure rate detected. Review recent changes and test coverage.")
        
        if run_data.get('is_weekend', 0) == 1:
            recommendations.append("Weekend deployment detected. Ensure adequate monitoring and support.")
        
        if not recommendations:
            recommendations.append("Monitor pipeline execution closely. Check logs for any warnings or errors.")
        
        return " | ".join(recommendations)
    
    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple pipeline runs.
        
        Args:
            data: DataFrame with multiple runs
        
        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Making batch predictions for {len(data)} runs")
        
        results = self.predict_with_confidence(data)
        
        # Add results to dataframe
        result_df = data.copy()
        for key in results[0].keys():
            result_df[key] = [r[key] for r in results]
        
        logger.info(f"✓ Batch prediction complete")
        return result_df


def main():
    """Test prediction service."""
    # Sample input data
    sample_run = {
        'name': 'CI Pipeline',
        'event': 'push',
        'status': 'completed',
        'build_duration': 450.0,
        'num_jobs': 3,
        'num_jobs_failed': 0,
        'failure_rate': 0.2,
        'hour_of_day': 14,
        'is_weekend': 0,
    }
    
    try:
        predictor = Predictor()
        result = predictor.predict_pipeline_run(sample_run)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        for key, value in result.items():
            print(f"{key:25s}: {value}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first using train_model.py")


if __name__ == "__main__":
    main()
