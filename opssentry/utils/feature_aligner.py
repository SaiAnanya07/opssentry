"""
Feature Alignment Utility
Ensures consistent features across different data sources (LogHub, GitHub, Synthetic).
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureAligner:
    """Aligns features between different data sources."""
    
    # Define the expected feature schema for the model
    REQUIRED_FEATURES = [
        'build_duration',
        'num_jobs',
        'event',
        'status',
        'conclusion',
    ]
    
    OPTIONAL_FEATURES = [
        'num_jobs_failed',
        'num_jobs_success',
        'total_job_duration',
        'failure_rate',
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        'is_business_hours',
        'num_errors',
        'num_warnings',
        'num_info',
        'total_lines',
    ]
    
    # Default values for missing features
    FEATURE_DEFAULTS = {
        'build_duration': 300.0,
        'num_jobs': 1,
        'event': 'unknown',
        'status': 'completed',
        'conclusion': 'unknown',
        'num_jobs_failed': 0,
        'num_jobs_success': 1,
        'total_job_duration': 300.0,
        'failure_rate': 0.0,
        'hour_of_day': 12,
        'day_of_week': 2,
        'is_weekend': 0,
        'is_business_hours': 1,
        'num_errors': 0,
        'num_warnings': 0,
        'num_info': 0,
        'total_lines': 0,
    }
    
    @staticmethod
    def align_features(df: pd.DataFrame, 
                       required_features: List[str] = None,
                       fill_missing: bool = True) -> pd.DataFrame:
        """
        Align dataframe features to expected schema.
        
        Args:
            df: Input dataframe
            required_features: List of required feature names
            fill_missing: Whether to fill missing features with defaults
        
        Returns:
            DataFrame with aligned features
        """
        if required_features is None:
            required_features = FeatureAligner.REQUIRED_FEATURES
        
        logger.info(f"Aligning features for {len(df)} samples")
        logger.info(f"Current columns: {list(df.columns)}")
        
        aligned_df = df.copy()
        
        # Add missing required features
        for feature in required_features:
            if feature not in aligned_df.columns:
                if fill_missing and feature in FeatureAligner.FEATURE_DEFAULTS:
                    default_value = FeatureAligner.FEATURE_DEFAULTS[feature]
                    aligned_df[feature] = default_value
                    logger.warning(f"Added missing required feature '{feature}' with default: {default_value}")
                else:
                    logger.error(f"Missing required feature '{feature}' and no default available")
        
        # Add missing optional features
        for feature in FeatureAligner.OPTIONAL_FEATURES:
            if feature not in aligned_df.columns and fill_missing:
                if feature in FeatureAligner.FEATURE_DEFAULTS:
                    aligned_df[feature] = FeatureAligner.FEATURE_DEFAULTS[feature]
                    logger.info(f"Added optional feature '{feature}' with default")
        
        logger.info(f"✓ Feature alignment complete. Final columns: {len(aligned_df.columns)}")
        return aligned_df
    
    @staticmethod
    def map_loghub_to_pipeline_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map LogHub dataset features to pipeline features.
        
        Args:
            df: LogHub dataframe
        
        Returns:
            DataFrame with mapped features
        """
        logger.info("Mapping LogHub features to pipeline features")
        
        mapped_df = df.copy()
        
        # Map LogHub features to expected pipeline features
        feature_mapping = {
            'total_lines': 'build_duration',  # Use log lines as proxy for duration
            'num_errors': 'num_jobs_failed',  # Errors indicate failed jobs
            'dataset': 'event',  # Dataset name as event type
        }
        
        for loghub_col, pipeline_col in feature_mapping.items():
            if loghub_col in mapped_df.columns and pipeline_col not in mapped_df.columns:
                # Scale total_lines to reasonable build duration (seconds)
                if loghub_col == 'total_lines':
                    mapped_df[pipeline_col] = mapped_df[loghub_col] * 0.1  # 0.1 sec per line
                else:
                    mapped_df[pipeline_col] = mapped_df[loghub_col]
                logger.info(f"Mapped '{loghub_col}' -> '{pipeline_col}'")
        
        # Derive additional features
        if 'num_errors' in mapped_df.columns:
            mapped_df['num_jobs'] = mapped_df['num_errors'].apply(lambda x: max(1, x + 1))
            mapped_df['num_jobs_success'] = mapped_df['num_jobs'] - mapped_df.get('num_jobs_failed', 0)
        
        # Set categorical defaults
        if 'status' not in mapped_df.columns:
            mapped_df['status'] = 'completed'
        
        if 'conclusion' not in mapped_df.columns:
            # Use 'failed' column if available, otherwise derive from errors
            if 'failed' in mapped_df.columns:
                mapped_df['conclusion'] = mapped_df['failed'].apply(
                    lambda x: 'failure' if x == 1 else 'success'
                )
            elif 'num_errors' in mapped_df.columns:
                mapped_df['conclusion'] = mapped_df['num_errors'].apply(
                    lambda x: 'failure' if x > 0 else 'success'
                )
        
        logger.info("✓ LogHub feature mapping complete")
        return mapped_df
    
    @staticmethod
    def validate_features(df: pd.DataFrame, 
                         required_features: List[str] = None) -> bool:
        """
        Validate that dataframe has all required features.
        
        Args:
            df: Input dataframe
            required_features: List of required features
        
        Returns:
            True if all required features present
        """
        if required_features is None:
            required_features = FeatureAligner.REQUIRED_FEATURES
        
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        
        logger.info("✓ All required features present")
        return True


def main():
    """Test feature aligner."""
    # Test with LogHub-style data
    loghub_data = pd.DataFrame({
        'dataset': ['HDFS', 'HDFS', 'Spark'],
        'total_lines': [1000, 1500, 2000],
        'num_errors': [5, 0, 10],
        'num_warnings': [20, 15, 30],
        'failed': [1, 0, 1],
    })
    
    print("Original LogHub data:")
    print(loghub_data.columns.tolist())
    
    # Map features
    mapped_df = FeatureAligner.map_loghub_to_pipeline_features(loghub_data)
    print("\nAfter mapping:")
    print(mapped_df.columns.tolist())
    
    # Align features
    aligned_df = FeatureAligner.align_features(mapped_df)
    print("\nAfter alignment:")
    print(aligned_df.columns.tolist())
    
    # Validate
    is_valid = FeatureAligner.validate_features(aligned_df)
    print(f"\nValidation: {'✓ Passed' if is_valid else '✗ Failed'}")


if __name__ == "__main__":
    main()
