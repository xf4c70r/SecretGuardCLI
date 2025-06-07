"""
Comprehensive evaluation metrics for SecretGuard model
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Comprehensive evaluation metrics calculator"""
    
    def __init__(self):
        self.results = {}
        
    def normalize_predictions(self, predictions: List[str]) -> List[int]:
        """
        Normalize model predictions to binary values
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Binary predictions (1 for secret detected, 0 for no secret)
        """
        normalized = []
        for pred in predictions:
            pred_lower = pred.lower().strip()
            
            # Positive indicators (secret detected)
            if any(indicator in pred_lower for indicator in [
                'yes', 'true', 'secret', 'found', 'detected', 'contains'
            ]):
                normalized.append(1)
            # Negative indicators (no secret)
            elif any(indicator in pred_lower for indicator in [
                'no', 'false', 'safe', 'clean', 'none', 'not found'
            ]):
                normalized.append(0)
            else:
                # Default to positive for ambiguous cases (conservative approach)
                logger.warning(f"Ambiguous prediction: {pred}. Defaulting to positive.")
                normalized.append(1)
                
        return normalized
    
    def normalize_ground_truth(self, ground_truth: List[str]) -> List[int]:
        """
        Normalize ground truth labels to binary values
        
        Args:
            ground_truth: Ground truth labels
            
        Returns:
            Binary labels (1 for secret present, 0 for no secret)
        """
        normalized = []
        for label in ground_truth:
            label_lower = label.lower().strip()
            if label_lower in ['yes', 'true', '1']:
                normalized.append(1)
            else:
                normalized.append(0)
        return normalized
    
    def calculate_basic_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'false_positive_rate': self._calculate_fpr(y_true, y_pred),
            'false_negative_rate': self._calculate_fnr(y_true, y_pred)
        }
        
        return metrics
    
    def _calculate_specificity(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 1:  # Single class case
            return 1.0 if y_true[0] == 0 else 0.0
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_fpr(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate False Positive Rate"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 1:  # Single class case
            return 0.0 if y_true[0] == 0 else 1.0
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def _calculate_fnr(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate False Negative Rate"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 1:  # Single class case
            return 0.0 if y_true[0] == 1 else 1.0
        tn, fp, fn, tp = cm.ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0
    
    def calculate_category_metrics(self, test_data: List[Dict], 
                                 predictions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics by category
        
        Args:
            test_data: Test data with categories
            predictions: Model predictions
            
        Returns:
            Dictionary of metrics by category
        """
        # Group by category
        category_data = defaultdict(lambda: {'y_true': [], 'y_pred': []})
        
        y_pred_norm = self.normalize_predictions(predictions)
        
        for i, test_case in enumerate(test_data):
            category = test_case.get('category', 'unknown')
            y_true_val = 1 if test_case['expected'].lower() == 'yes' else 0
            
            category_data[category]['y_true'].append(y_true_val)
            category_data[category]['y_pred'].append(y_pred_norm[i])
        
        # Calculate metrics for each category
        category_metrics = {}
        for category, data in category_data.items():
            if len(data['y_true']) > 0:
                category_metrics[category] = self.calculate_basic_metrics(
                    data['y_true'], data['y_pred']
                )
                category_metrics[category]['sample_count'] = len(data['y_true'])
        
        return category_metrics
    
    def generate_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """Generate confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            save_path: Optional[str] = None) -> Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Secret', 'Secret'], 
                   yticklabels=['No Secret', 'Secret'])
        
        plt.title('Confusion Matrix - SecretGuard Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_by_category(self, category_metrics: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None) -> Figure:
        """
        Plot metrics by category
        
        Args:
            category_metrics: Metrics by category
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        categories = list(category_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [category_metrics[cat].get(metric, 0) for cat in categories]
            
            axes[i].bar(categories, values, color=f'C{i}', alpha=0.7)
            axes[i].set_title(f'{metric.title()} by Category')
            axes[i].set_ylabel(metric.title())
            axes[i].set_xticklabels(categories, rotation=45, ha='right')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_detailed_report(self, test_data: List[Dict], 
                               predictions: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report
        
        Args:
            test_data: Test data
            predictions: Model predictions
            
        Returns:
            Detailed evaluation report
        """
        # Normalize data
        y_true = self.normalize_ground_truth([case['expected'] for case in test_data])
        y_pred = self.normalize_predictions(predictions)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_basic_metrics(y_true, y_pred)
        
        # Calculate category metrics
        category_metrics = self.calculate_category_metrics(test_data, predictions)
        
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        # Detailed analysis
        detailed_results = []
        for i, (test_case, pred) in enumerate(zip(test_data, predictions)):
            detailed_results.append({
                'id': test_case['id'],
                'code': test_case['code'][:100] + "..." if len(test_case['code']) > 100 else test_case['code'],
                'expected': test_case['expected'],
                'predicted': pred,
                'correct': y_true[i] == y_pred[i],
                'category': test_case.get('category', 'unknown')
            })
        
        report = {
            'overall_metrics': overall_metrics,
            'category_metrics': category_metrics,
            'confusion_matrix': cm.tolist(),
            'detailed_results': detailed_results,
            'summary': {
                'total_samples': len(test_data),
                'correct_predictions': sum(y_true[i] == y_pred[i] for i in range(len(y_true))),
                'accuracy': overall_metrics['accuracy'],
                'categories_tested': len(set(case.get('category', 'unknown') for case in test_data))
            }
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results"""
        print("\n" + "="*60)
        print("SECRETGUARD MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Overall metrics
        metrics = report['overall_metrics']
        print(f"\nOverall Performance:")
        print(f"  Accuracy:    {metrics['accuracy']:.3f}")
        print(f"  Precision:   {metrics['precision']:.3f}")
        print(f"  Recall:      {metrics['recall']:.3f}")
        print(f"  F1-Score:    {metrics['f1_score']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        
        # Summary stats
        summary = report['summary']
        print(f"\nTest Summary:")
        print(f"  Total Samples:        {summary['total_samples']}")
        print(f"  Correct Predictions:  {summary['correct_predictions']}")
        print(f"  Categories Tested:    {summary['categories_tested']}")
        
        # Category performance
        print(f"\nPerformance by Category:")
        for category, cat_metrics in report['category_metrics'].items():
            print(f"  {category:20} | F1: {cat_metrics['f1_score']:.3f} | "
                  f"Samples: {cat_metrics['sample_count']:2d}")
        
        print("\n" + "="*60)