"""
Feature Engineering Module
Creates derived features from raw pipeline data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import config
from utils.logger import setup_logger

logger = setup_logger(__name__, "feature_engineering.log")


class FeatureEngineer:
    """Creates derived features for pipeline failure prediction."""
    
    @staticmethod
    def extract_temporal_features(df: pd.DataFrame, timestamp_col: str = 'created_at') -> pd.DataFrame:
        """
        Extract temporal features from timestamps.
        
        Args:
            df: Input dataframe
            timestamp_col: Name of timestamp column
        
        Returns:
            DataFrame with added temporal features
        """
        logger.info("Extracting temporal features...")
        
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df
        
        # Convert to datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Extract features
        df['hour_of_day'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        logger.info("✓ Extracted temporal features")
        return df
    
    @staticmethod
    def calculate_rolling_statistics(df: pd.DataFrame, 
                                     group_col: str = None,
                                     value_col: str = 'build_duration',
                                     windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate rolling statistics for numerical columns.
        
        Args:
            df: Input dataframe
            group_col: Column to group by (e.g., 'workflow_name')
            value_col: Column to calculate statistics for
            windows: List of window sizes
        
        Returns:
            DataFrame with added rolling statistics
        """
        logger.info(f"Calculating rolling statistics for '{value_col}'...")
        
        if value_col not in df.columns:
            logger.warning(f"Value column '{value_col}' not found")
            return df
        
        # Sort by time if timestamp exists
        if 'created_at' in df.columns:
            df = df.sort_values('created_at')
        
        for window in windows:
            if group_col and group_col in df.columns:
                # Calculate per group
                df[f'{value_col}_rolling_mean_{window}'] = df.groupby(group_col)[value_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{value_col}_rolling_std_{window}'] = df.groupby(group_col)[value_col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            else:
                # Calculate globally
                df[f'{value_col}_rolling_mean_{window}'] = df[value_col].rolling(window, min_periods=1).mean()
                df[f'{value_col}_rolling_std_{window}'] = df[value_col].rolling(window, min_periods=1).std()
        
        logger.info(f"✓ Calculated rolling statistics with windows: {windows}")
        return df
    
    @staticmethod
    def calculate_failure_rate(df: pd.DataFrame, 
                              group_col: str = 'name',
                              label_col: str = 'failed',
                              window: int = 10) -> pd.DataFrame:
        """
        Calculate historical failure rate.
        
        Args:
            df: Input dataframe
            group_col: Column to group by
            label_col: Binary label column (1 for failure)
            window: Window size for rolling calculation
        
        Returns:
            DataFrame with failure rate features
        """
        logger.info("Calculating failure rates...")
        
        if label_col not in df.columns:
            logger.warning(f"Label column '{label_col}' not found")
            return df
        
        # Sort by time
        if 'created_at' in df.columns:
            df = df.sort_values('created_at')
        
        if group_col and group_col in df.columns:
            # Per-group failure rate
            df['failure_rate'] = df.groupby(group_col)[label_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Total failures per group
            df['total_failures'] = df.groupby(group_col)[label_col].cumsum()
            df['total_runs'] = df.groupby(group_col).cumcount() + 1
        else:
            # Global failure rate
            df['failure_rate'] = df[label_col].rolling(window, min_periods=1).mean()
            df['total_failures'] = df[label_col].cumsum()
            df['total_runs'] = range(1, len(df) + 1)
        
        logger.info("✓ Calculated failure rates")
        return df
    
    @staticmethod
    def calculate_commit_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate commit-related features.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with commit features
        """
        logger.info("Calculating commit features...")
        
        # Commit frequency (if we have commit data)
        if 'head_sha' in df.columns:
            # Count unique commits per workflow
            if 'name' in df.columns:
                df['commit_frequency'] = df.groupby('name')['head_sha'].transform('nunique')
        
        # Time since last run
        if 'created_at' in df.columns:
            df = df.sort_values('created_at')
            df['created_at_dt'] = pd.to_datetime(df['created_at'])
            
            if 'name' in df.columns:
                df['time_since_last_run'] = df.groupby('name')['created_at_dt'].diff().dt.total_seconds()
            else:
                df['time_since_last_run'] = df['created_at_dt'].diff().dt.total_seconds()
            
            # Fill first values with 0
            df['time_since_last_run'] = df['time_since_last_run'].fillna(0)
        
        logger.info("✓ Calculated commit features")
        return df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, 
                                   feature_pairs: List[tuple] = None) -> pd.DataFrame:
        """
        Create interaction features between pairs of features.
        
        Args:
            df: Input dataframe
            feature_pairs: List of (feature1, feature2) tuples
        
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        if feature_pairs is None:
            # Default interactions
            feature_pairs = [
                ('build_duration', 'num_jobs'),
                ('num_errors', 'num_warnings'),
                ('hour_of_day', 'day_of_week'),
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio (avoid division by zero)
                if (df[feat2] != 0).any():
                    df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
        
        logger.info(f"✓ Created {len(feature_pairs)} interaction features")
        return df
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame, 
                             group_col: str = 'name',
                             label_col: str = 'failed') -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input dataframe
            group_col: Column to group by for rolling statistics
            label_col: Binary label column
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        original_cols = len(df.columns)
        
        # Temporal features
        if 'created_at' in df.columns:
            df = FeatureEngineer.extract_temporal_features(df, 'created_at')
        else:
            # Add default temporal features
            logger.info("No timestamp column found, adding default temporal features")
            df['hour_of_day'] = 12
            df['day_of_week'] = 2
            df['is_weekend'] = 0
            df['is_business_hours'] = 1
        
        # Rolling statistics
        if 'build_duration' in df.columns:
            df = FeatureEngineer.calculate_rolling_statistics(
                df, group_col, 'build_duration', windows=[5, 10]
            )
        
        # Failure rate
        if label_col in df.columns:
            df = FeatureEngineer.calculate_failure_rate(df, group_col, label_col)
        
        # Commit features
        df = FeatureEngineer.calculate_commit_features(df)
        
        # Interaction features (only if both features exist)
        df = FeatureEngineer.create_interaction_features(df)
        
        new_cols = len(df.columns)
        logger.info("=" * 60)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Original features: {original_cols}")
        logger.info(f"New features: {new_cols}")
        logger.info(f"Added: {new_cols - original_cols} features")
        logger.info("=" * 60)
        
        return df


def main():
    """Main function for testing."""
    # Test with sample data
    sample_data = {
        'name': ['workflow1'] * 10 + ['workflow2'] * 10,
        'created_at': pd.date_range('2024-01-01', periods=20, freq='H'),
        'build_duration': np.random.randint(100, 500, 20),
        'failed': np.random.randint(0, 2, 20),
        'num_jobs': np.random.randint(1, 5, 20),
    }
    
    df = pd.DataFrame(sample_data)
    
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    print("\nEngineered Features:")
    print(df_engineered.columns.tolist())
    print(f"\nShape: {df_engineered.shape}")


if __name__ == "__main__":
    main()
