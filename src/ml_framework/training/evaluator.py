"""
Model evaluation and visualization.

Provides comprehensive evaluation metrics, plots, and analyses.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import Config, MetricsCalculator, print_metrics_summary

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive model evaluation.

    Generates metrics, plots, and analyses for model performance.

    Examples:
        >>> evaluator = Evaluator(model, config, artifact_dir)
        >>> results = evaluator.evaluate(X_test, y_test)
        >>> evaluator.plot_roc_curve()
        >>> evaluator.plot_feature_importance(feature_names)
    """

    def __init__(self, model: Any, config: Config, artifact_dir: Path):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            config: Configuration object
            artifact_dir: Directory to save plots and results
        """
        self.model = model
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.plots_dir = self.artifact_dir / config.artifacts.plots_dir

        self.metrics_calculator = MetricsCalculator()

        # Will be populated during evaluation
        self.y_true = None
        self.y_pred = None
        self.y_proba = None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model and generate all outputs.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary containing metrics and paths to generated artifacts
        """
        logger.info("Evaluating model performance...")

        # Make predictions
        self.y_true = y_test.values
        self.y_pred = self.model.predict(X_test.values)
        self.y_proba = self.model.predict_proba(X_test.values)[:, 1]

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all(
            self.y_true, self.y_pred, self.y_proba, threshold=self.config.training.threshold
        )

        # Print metrics summary
        print_metrics_summary(metrics)

        # Generate plots
        plot_paths = self._generate_plots(X_test)

        # Get classification report
        report = self.metrics_calculator.get_classification_report(self.y_true, self.y_pred)

        logger.info("\nClassification Report:")
        logger.info("\n" + report)

        return {"metrics": metrics, "plot_paths": plot_paths, "classification_report": report}

    def _generate_plots(self, X_test: pd.DataFrame) -> Dict[str, str]:
        """
        Generate all evaluation plots.

        Args:
            X_test: Test features (for feature names)

        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}

        save_config = self.config.artifacts.save

        # Confusion matrix
        if save_config.confusion_matrix:
            path = self.plot_confusion_matrix()
            plot_paths["confusion_matrix"] = str(path)

        # ROC curve
        if save_config.roc_curve:
            path = self.plot_roc_curve()
            plot_paths["roc_curve"] = str(path)

        # Precision-recall curve
        path = self.plot_precision_recall_curve()
        plot_paths["precision_recall_curve"] = str(path)

        # Feature importance
        if save_config.feature_importance:
            try:
                path = self.plot_feature_importance(X_test.columns.tolist())
                plot_paths["feature_importance"] = str(path)
            except Exception as e:
                logger.warning(f"Could not plot feature importance: {e}")

        # Prediction distribution
        path = self.plot_prediction_distribution()
        plot_paths["prediction_distribution"] = str(path)

        return plot_paths

    def plot_confusion_matrix(self) -> Path:
        """
        Plot confusion matrix.

        Returns:
            Path to saved plot
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Customer", "Customer"],
            yticklabels=["Non-Customer", "Customer"],
        )
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        save_path = self.plots_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix to {save_path}")

        return save_path

    def plot_roc_curve(self) -> Path:
        """
        Plot ROC curve.

        Returns:
            Path to saved plot
        """
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        save_path = self.plots_dir / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved ROC curve to {save_path}")

        return save_path

    def plot_precision_recall_curve(self) -> Path:
        """
        Plot precision-recall curve.

        Returns:
            Path to saved plot
        """
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_proba)
        avg_precision = average_precision_score(self.y_true, self.y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall, precision, color="blue", lw=2, label=f"PR curve (AP = {avg_precision:.3f})"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)

        save_path = self.plots_dir / "precision_recall_curve.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved precision-recall curve to {save_path}")

        return save_path

    def plot_feature_importance(self, feature_names: list, top_n: int = 20) -> Path:
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            top_n: Number of top features to display

        Returns:
            Path to saved plot
        """
        # Get feature importance
        importance = self.model.get_feature_importance()

        # Create dataframe
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Take top N features
        top_features = feature_importance.head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=top_features,
            y="feature",
            x="importance",
            hue="feature",
            palette="viridis",
            legend=False,
        )
        plt.title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        save_path = self.plots_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature importance to {save_path}")

        # Also save to CSV
        csv_path = self.artifact_dir / self.config.artifacts.metrics_dir / "feature_importance.csv"
        feature_importance.to_csv(csv_path, index=False)

        return save_path

    def plot_prediction_distribution(self) -> Path:
        """
        Plot distribution of predicted probabilities by true class.

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 6))

        # Plot distributions
        plt.hist(
            self.y_proba[self.y_true == 0], bins=50, alpha=0.5, label="Non-Customers", color="blue"
        )
        plt.hist(self.y_proba[self.y_true == 1], bins=50, alpha=0.5, label="Customers", color="red")

        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title("Distribution of Predicted Probabilities", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = self.plots_dir / "prediction_distribution.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved prediction distribution to {save_path}")

        return save_path
