"""
Model evaluation metrics and utilities.

Provides comprehensive metrics for binary classification tasks with
clear interpretability and visualization support.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate and store comprehensive classification metrics.

    Examples:
        >>> calc = MetricsCalculator()
        >>> metrics = calc.calculate_all(y_true, y_pred, y_proba)
        >>> print(metrics['roc_auc'])
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.supported_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate all supported metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities if threshold provided)
            y_proba: Predicted probabilities (for AUC metrics)
            threshold: Classification threshold (if y_pred is probabilities)

        Returns:
            Dictionary mapping metric names to values

        Examples:
            >>> metrics = calc.calculate_all(y_true, y_pred, y_proba)
            >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        """
        # If y_pred is probabilities, convert to binary predictions
        if y_proba is not None and len(np.unique(y_pred)) > 2:
            y_pred = (y_pred >= threshold).astype(int)

        metrics = {}

        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # AUC metrics (require probabilities)
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics["average_precision"] = average_precision_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics["roc_auc"] = None
                metrics["average_precision"] = None

        return metrics

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix [[TN, FP], [FN, TP]]
        """
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Optional class names

        Returns:
            Formatted classification report string
        """
        if target_names is None:
            target_names = ["Non-Customer", "Customer"]

        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def calculate_roc_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve coordinates.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_proba)

    def calculate_precision_recall_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve coordinates.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_proba)

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1"
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold based on metric.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metric_value)

        Examples:
            >>> threshold, f1 = calc.find_optimal_threshold(y_true, y_proba, 'f1')
            >>> print(f"Best threshold: {threshold:.3f} (F1: {f1:.3f})")
        """
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_score = 0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Print formatted metrics summary.

    Args:
        metrics: Dictionary of metric names and values
    """
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)

    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name.upper():.<30} {value:.4f}")
        else:
            print(f"{metric_name.upper():.<30} N/A")

    print("=" * 50 + "\n")


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    conversion_value: float = 100.0,
    contact_cost: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        conversion_value: Value of a successful conversion
        contact_cost: Cost of contacting a prospect

    Returns:
        Dictionary of business metrics

    Examples:
        >>> business_metrics = calculate_business_metrics(
        ...     y_true, y_pred, y_proba,
        ...     conversion_value=500.0,
        ...     contact_cost=5.0
        ... )
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate ROI-related metrics
    total_contacts = fp + tp
    successful_conversions = tp
    missed_conversions = fn

    # Revenue from successful conversions
    revenue = successful_conversions * conversion_value

    # Cost of all contacts
    cost = total_contacts * contact_cost

    # Net profit
    profit = revenue - cost

    # ROI
    roi = (profit / cost * 100) if cost > 0 else 0

    return {
        "total_contacts": total_contacts,
        "successful_conversions": successful_conversions,
        "missed_conversions": missed_conversions,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "roi_percent": roi,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }
