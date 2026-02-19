"""
Data Validation Utilities for OpsSentry
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    """Validates data quality and integrity."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[bool, Dict[str, float]]:
        """
        Check for missing values in dataframe.
        
        Args:
            df: Input dataframe
            threshold: Maximum allowed missing ratio per column
        
        Returns:
            (is_valid, missing_stats)
        """
        missing_stats = {}
        is_valid = True
        
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            missing_stats[col] = missing_ratio
            
            if missing_ratio > threshold:
                logger.warning(f"Column '{col}' has {missing_ratio:.2%} missing values (threshold: {threshold:.2%})")
                is_valid = False
        
        return is_valid, missing_stats
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
        """
        Check for duplicate rows.
        
        Args:
            df: Input dataframe
        
        Returns:
            (num_duplicates, duplicate_rows)
        """
        duplicates = df[df.duplicated()]
        num_duplicates = len(duplicates)
        
        if num_duplicates > 0:
            logger.warning(f"Found {num_duplicates} duplicate rows")
        
        return num_duplicates, duplicates
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        """
        Validate column data types.
        
        Args:
            df: Input dataframe
            expected_types: Dict mapping column names to expected types
        
        Returns:
            True if all types match, False otherwise
        """
        is_valid = True
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                logger.error(f"Missing expected column: {col}")
                is_valid = False
                continue
            
            actual_type = str(df[col].dtype)
            if expected_type not in actual_type:
                logger.warning(f"Column '{col}' has type '{actual_type}', expected '{expected_type}'")
                is_valid = False
        
        return is_valid
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame, range_constraints: Dict[str, Tuple[float, float]]) -> bool:
        """
        Check if numerical values are within expected ranges.
        
        Args:
            df: Input dataframe
            range_constraints: Dict mapping column names to (min, max) tuples
        
        Returns:
            True if all values are within ranges, False otherwise
        """
        is_valid = True
        
        for col, (min_val, max_val) in range_constraints.items():
            if col not in df.columns:
                continue
            
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                logger.warning(f"Column '{col}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]")
                is_valid = False
        
        return is_valid
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame, 
                        missing_threshold: float = 0.5,
                        check_types: Dict[str, str] = None,
                        check_ranges: Dict[str, Tuple[float, float]] = None) -> bool:
        """
        Comprehensive dataset validation.
        
        Args:
            df: Input dataframe
            missing_threshold: Maximum allowed missing ratio
            check_types: Expected data types
            check_ranges: Expected value ranges
        
        Returns:
            True if dataset is valid, False otherwise
        """
        logger.info(f"Validating dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Check missing values
        missing_valid, missing_stats = DataValidator.check_missing_values(df, missing_threshold)
        
        # Check duplicates
        num_duplicates, _ = DataValidator.check_duplicates(df)
        
        # Check data types
        types_valid = True
        if check_types:
            types_valid = DataValidator.check_data_types(df, check_types)
        
        # Check value ranges
        ranges_valid = True
        if check_ranges:
            ranges_valid = DataValidator.check_value_ranges(df, check_ranges)
        
        is_valid = missing_valid and types_valid and ranges_valid
        
        if is_valid:
            logger.info("✓ Dataset validation passed")
        else:
            logger.error("✗ Dataset validation failed")
        
        return is_valid
