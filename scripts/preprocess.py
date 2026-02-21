"""
Data Preprocessing Module
Handles data cleaning, feature extraction, encoding, normalization, and train/val/test split.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import joblib
import json
import config
from utils.logger import setup_logger
from utils.data_validator import DataValidator
from scripts.feature_engineering import FeatureEngineer
from utils.feature_aligner import FeatureAligner

logger = setup_logger(__name__, "preprocess.log")


class DataPreprocessor:
    """Preprocesses data for model training."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
    
    def load_data(self, source: str = 'github') -> pd.DataFrame:
        """
        Load data from processed files.
        
        Args:
            source: Data source ('github', 'loghub', or 'both')
        
        Returns:
            Combined DataFrame
        """
        logger.info(f"Loading data from source: {source}")
        
        dfs = []
        
        if source in ['github', 'both']:
            github_path = config.PROCESSED_DATA_DIR / "github_labeled.csv"
            synthetic_path = config.PROCESSED_DATA_DIR / "synthetic_labeled.csv"
            
            if github_path.exists():
                df_github = pd.read_csv(github_path)
                df_github['source'] = 'github'
                dfs.append(df_github)
                logger.info(f"Loaded {len(df_github)} samples from GitHub")
            elif synthetic_path.exists():
                df_synthetic = pd.read_csv(synthetic_path)
                df_synthetic['source'] = 'synthetic'
                dfs.append(df_synthetic)
                logger.info(f"Loaded {len(df_synthetic)} samples from synthetic data")
            else:
                logger.warning("No GitHub or synthetic data found")
        
        if source in ['loghub', 'both']:
            loghub_path = config.PROCESSED_DATA_DIR / "loghub_labeled.csv"
            if loghub_path.exists():
                df_loghub = pd.read_csv(loghub_path)
                df_loghub['source'] = 'loghub'
                dfs.append(df_loghub)
                logger.info(f"Loaded {len(df_loghub)} samples from LogHub")
        
        if not dfs:
            raise FileNotFoundError(f"No data found for source: {source}")
        
        # Combine datasets
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✓ Total samples loaded: {len(df)}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and removing duplicates.
        
        Args:
            df: Input dataframe
        
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning data...")
        
        original_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, fill with mode or 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if col in ['status', 'conclusion', 'event']:
                    df[col] = df[col].fillna('unknown')
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled {col} missing values")
        
        logger.info("✓ Data cleaning complete")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)
        
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude certain columns
        exclude_cols = ['id', 'html_url', 'logs_url', 'jobs_url', 'head_sha', 'job_names']
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        self.categorical_features = categorical_cols
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                # Create and fit encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        logger.info(f"✓ Encoded {len(categorical_cols)} categorical features")
        return df
    
    def normalize_numerical_features(self, df: pd.DataFrame, 
                                     fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input dataframe
            fit: Whether to fit scaler (True for training, False for inference)
        
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing numerical features...")
        
        # Identify numerical columns (exclude label and IDs)
        exclude_cols = ['failed', 'id', 'run_number']
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        self.numerical_features = numerical_cols
        
        if not numerical_cols:
            logger.warning("No numerical features found to normalize")
            return df
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            logger.info(f"✓ Normalized {len(numerical_cols)} numerical features")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            logger.info(f"✓ Applied normalization to {len(numerical_cols)} features")
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   label_col: str = 'failed') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            label_col: Name of label column
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/val/test sets...")
        
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataframe")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=config.TEST_SIZE,
            random_state=42,
            stratify=df[label_col] if df[label_col].nunique() > 1 else None
        )
        
        # Second split: train vs val
        val_ratio = config.VAL_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=42,
            stratify=train_val_df[label_col] if train_val_df[label_col].nunique() > 1 else None
        )
        
        logger.info(f"✓ Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"✓ Val set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"✓ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        # Print class distribution
        for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if label_col in split_df.columns:
                dist = split_df[label_col].value_counts()
                logger.info(f"{name} class distribution: {dist.to_dict()}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, 
                   val_df: pd.DataFrame, 
                   test_df: pd.DataFrame):
        """
        Save train/val/test splits to CSV files.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
        """
        logger.info("Saving data splits...")
        
        train_df.to_csv(config.TRAIN_DATA_CSV, index=False)
        val_df.to_csv(config.VAL_DATA_CSV, index=False)
        test_df.to_csv(config.TEST_DATA_CSV, index=False)
        
        logger.info(f"✓ Saved train data to {config.TRAIN_DATA_CSV}")
        logger.info(f"✓ Saved val data to {config.VAL_DATA_CSV}")
        logger.info(f"✓ Saved test data to {config.TEST_DATA_CSV}")
    
    def save_preprocessor(self):
        """Save scaler and encoders for later use."""
        logger.info("Saving preprocessor artifacts...")
        
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.label_encoders, config.ENCODER_PATH)
        
        # Save feature names
        feature_metadata = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'all_features': self.feature_names
        }
        metadata_path = config.MODELS_DIR / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        
        logger.info(f"✓ Saved scaler to {config.SCALER_PATH}")
        logger.info(f"✓ Saved encoders to {config.ENCODER_PATH}")
        logger.info(f"✓ Saved feature metadata to {metadata_path}")
    
    def load_preprocessor(self):
        """Load saved scaler and encoders."""
        logger.info("Loading preprocessor artifacts...")
        
        if config.SCALER_PATH.exists():
            self.scaler = joblib.load(config.SCALER_PATH)
            logger.info("✓ Loaded scaler")
        
        if config.ENCODER_PATH.exists():
            self.label_encoders = joblib.load(config.ENCODER_PATH)
            logger.info("✓ Loaded encoders")
    
    def preprocess_pipeline(self, source: str = 'github') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            source: Data source ('github', 'loghub', or 'both')
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data(source)
        
        # Align features if from LogHub
        if source in ['loghub', 'both'] and 'dataset' in df.columns:
            logger.info("Aligning LogHub features...")
            df = FeatureAligner.align_features(df, fill_missing=True)
        
        # Validate data
        validator = DataValidator()
        validator.validate_dataset(df, missing_threshold=0.7)
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        engineer = FeatureEngineer()
        # Use 'name' if available, otherwise use 'dataset' for grouping
        group_col = 'name' if 'name' in df.columns else 'dataset' if 'dataset' in df.columns else None
        df = engineer.engineer_all_features(df, group_col=group_col, label_col='failed')
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Normalize numerical features
        df = self.normalize_numerical_features(df, fit=True)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Save splits
        self.save_splits(train_df, val_df, test_df)
        
        # Save preprocessor
        self.save_preprocessor()
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return train_df, val_df, test_df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for model training')
    parser.add_argument('--source', type=str, default='github', 
                       choices=['github', 'loghub', 'both'],
                       help='Data source to preprocess')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(args.source)
    
    print(f"\n✓ Preprocessing complete!")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")


if __name__ == "__main__":
    main()
