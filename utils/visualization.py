"""
Visualization Utilities for OpsSentry
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments (Jenkins/CI)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import config

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class Visualizer:
    """Visualization utilities for model evaluation and data analysis."""
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, 
                             classes: List[str] = None,
                             title: str = 'Confusion Matrix',
                             save_path: Optional[Path] = None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array
            classes: Class labels
            title: Plot title
            save_path: Optional path to save figure
        """
        if classes is None:
            classes = ['Negative', 'Positive']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, 
                      tpr: np.ndarray, 
                      roc_auc: float,
                      title: str = 'ROC Curve',
                      save_path: Optional[Path] = None):
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: Area under ROC curve
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], 
                               importances: np.ndarray,
                               top_n: int = 20,
                               title: str = 'Feature Importance',
                               save_path: Optional[Path] = None):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to display
            title: Plot title
            save_path: Optional path to save figure
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("viridis", len(top_features))
        ax.barh(range(len(top_features)), top_importances, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                               title: str = 'Model Performance Comparison',
                               save_path: Optional[Path] = None):
        """
        Plot comparison of metrics across different models.
        
        Args:
            metrics_dict: Dict of {model_name: {metric_name: value}}
            title: Plot title
            save_path: Optional path to save figure
        """
        df = pd.DataFrame(metrics_dict).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]],
                            title: str = 'Training History',
                            save_path: Optional[Path] = None):
        """
        Plot training history (loss, accuracy over epochs).
        
        Args:
            history: Dict with keys like 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', marker='o')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy', marker='o')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        
    
    @staticmethod
    def plot_data_distribution(df: pd.DataFrame, 
                              column: str,
                              title: str = None,
                              save_path: Optional[Path] = None):
        """
        Plot distribution of a column.
        
        Args:
            df: Input dataframe
            column: Column name to plot
            title: Plot title
            save_path: Optional path to save figure
        """
        if title is None:
            title = f'Distribution of {column}'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if df[column].dtype in ['int64', 'float64']:
            # Numerical column - histogram
            ax.hist(df[column].dropna(), bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        else:
            # Categorical column - bar plot
            value_counts = df[column].value_counts()
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        
