"""
Training orchestration and experiment management.

The Trainer class coordinates data loading, preprocessing, training,
evaluation, and artifact saving.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_score, train_test_split

from ..data import (
    EmployeeRangeOrdinalEncoder,
    FeatureEngineer,
    HubSpotDataLoader,
    IndustryEncoder,
    MissingValueHandler,
)
from ..models import create_model
from ..utils import Config, ExperimentLogger, MetricsCalculator, save_config
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main training orchestrator for ML experiments.

    Handles end-to-end training pipeline:
    1. Data loading and preprocessing
    2. Train/test split
    3. Model training
    4. Evaluation
    5. Artifact saving

    Design Philosophy:
    - Configuration-driven: All parameters from config file
    - Reproducible: Seeds set, artifacts versioned
    - Comprehensive: Saves all necessary artifacts for deployment
    - Observable: Detailed logging throughout process

    Examples:
        >>> config = load_config('configs/config.yaml')
        >>> trainer = Trainer(config)
        >>> results = trainer.train()
        >>> print(results['metrics'])
    """

    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration object with experiment settings
        """
        self.config = config

        # Set up experiment logger
        self.exp_logger = ExperimentLogger(
            experiment_name=config.experiment.name,
            log_dir=Path(config.artifacts.base_dir) / "logs",
            level=config.logging.get("level", "INFO"),
        )

        # Set reproducibility seeds
        if config.reproducibility.deterministic:
            self._set_seeds(config.reproducibility.seed)

        # mlflow setup
        self._setup_mlflow()

        # Initialize artifact directory
        self.artifact_dir = self._setup_artifact_directory()

        # Initialize components
        self.data_loader = None
        self.feature_engineer = None
        self.model = None
        self.evaluator = None

        self.exp_logger.info(f"Trainer initialized for experiment: {config.experiment.name}")

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        try:
            import random

            random.seed(seed)
        except ImportError:
            pass

        self.exp_logger.info(f"Set random seed: {seed}")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        # Set tracking URI (local or remote)
        mlflow_uri = self.config.experiment.get("mlflow_tracking_uri", "./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        # Set experiment name
        experiment_name = self.config.experiment.name
        mlflow.set_experiment(experiment_name)

        self.exp_logger.info(f"MLflow tracking URI: {mlflow_uri}")
        self.exp_logger.info(f"MLflow experiment: {experiment_name}")

    def _setup_artifact_directory(self) -> Path:
        """
        Create artifact directory with timestamp.

        Returns:
            Path to artifact directory
        """
        base_dir = Path(self.config.artifacts.base_dir)

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"{self.config.experiment.name}_{timestamp}"

        # Create subdirectories
        (exp_dir / self.config.artifacts.models_dir).mkdir(parents=True, exist_ok=True)
        (exp_dir / self.config.artifacts.metrics_dir).mkdir(parents=True, exist_ok=True)
        (exp_dir / self.config.artifacts.plots_dir).mkdir(parents=True, exist_ok=True)
        (exp_dir / self.config.artifacts.predictions_dir).mkdir(parents=True, exist_ok=True)

        self.exp_logger.info(f"Artifact directory: {exp_dir}")

        return exp_dir

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data using configured data loader.

        Returns:
            Tuple of (features, target)
        """
        self.exp_logger.info("Loading data...")

        self.data_loader = HubSpotDataLoader(
            customers_path=self.config.data.customers_path,
            noncustomers_path=self.config.data.noncustomers_path,
            usage_path=self.config.data.usage_actions_path,
            lookback_days=self.config.data.features.lookback_days,
        )

        X, y = self.data_loader.load_and_prepare()

        self.exp_logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        self.exp_logger.log_parameter("n_samples", X.shape[0])
        self.exp_logger.log_parameter("n_features", X.shape[1])
        self.exp_logger.log_parameter("n_customers", y.sum())
        self.exp_logger.log_parameter("n_noncustomers", (1 - y).sum())

        return X, y

    def preprocess_data(
        self, X: pd.DataFrame, y: pd.Series, fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features using config-driven transformers.

        Args:
            X: Features
            y: Target
            fit: Whether to fit transformers (True for train, False for test)

        Returns:
            Tuple of (preprocessed features, target)
        """
        self.exp_logger.info(f"Preprocessing data (fit={fit})...")

        X = X.copy()
        y = y.copy()

        # Initialize transformers on first call (fit=True)
        if fit:
            # Employee encoder
            self.employee_encoder = EmployeeRangeOrdinalEncoder()

            # Industry encoder (config-driven)
            industry_min_freq = self.config.data.features.get("industry_min_frequency", 5)
            self.industry_encoder = IndustryEncoder(min_frequency=industry_min_freq)

            # Missing value handler (config-driven)
            missing_config = self.config.data.features.get("missing_values", {})
            strategy = missing_config.get("strategy", "auto")
            constant_value = missing_config.get("constant_value", -1)
            self.missing_handler = MissingValueHandler(
                strategy=strategy, constant_value=constant_value
            )

            # Feature engineer (config-driven)
            self.feature_engineer = FeatureEngineer(
                scaling_method=self.config.data.features.scaling.method
            )

            self.exp_logger.info(
                f"Configured: missing_strategy={strategy}, "
                f"industry_min_freq={industry_min_freq}, "
                f"scaling={self.config.data.features.scaling.method}"
            )

        # Apply transformations in order
        # 1. Employee range encoding
        X = self.employee_encoder.transform(X)

        # 2. Industry encoding (handles its own NaNs)
        if fit:
            self.industry_encoder.fit(X)
        X = self.industry_encoder.transform(X)

        # 3. Handle all remaining missing values
        if fit:
            self.missing_handler.fit(X)
        X = self.missing_handler.transform(X)

        # 4. Scale features
        if fit:
            X = self.feature_engineer.fit_transform(X, y)
        else:
            X = self.feature_engineer.transform(X)

        self.exp_logger.info(f"Preprocessed shape: {X.shape}")

        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            X: Features
            y: Target

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.exp_logger.info("Splitting data...")

        stratify = y if self.config.data.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=stratify,
        )

        self.exp_logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        self.exp_logger.log_parameter("train_size", len(X_train))
        self.exp_logger.log_parameter("test_size", len(X_test))

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train model using configuration.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model
        """
        self.exp_logger.info(f"Training {self.config.model.type} model...")

        # Create model
        hyperparams = dict(self.config.model.hyperparameters.__dict__.get("_config", {}))
        if not hyperparams:
            # Fallback: get all attributes
            hyperparams = {
                k: v
                for k, v in vars(self.config.model.hyperparameters).items()
                if not k.startswith("_")
            }

        self.model = create_model(model_type=self.config.model.type, hyperparameters=hyperparams)

        # Log hyperparameters
        for param, value in hyperparams.items():
            self.exp_logger.log_parameter(f"model_{param}", value)

        # Train model
        self.model.fit(X_train.values, y_train.values)

        self.exp_logger.info("Model training complete")

        return self.model

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Target

        Returns:
            Dictionary of cross-validation scores
        """
        if not self.config.training.cross_validation.enabled:
            return {}

        self.exp_logger.info("Performing cross-validation...")

        cv_scores = cross_val_score(
            self.model.model,
            X.values,
            y.values,
            cv=self.config.training.cross_validation.n_folds,
            scoring="roc_auc",
        )

        cv_results = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

        self.exp_logger.info(
            f"CV ROC-AUC: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})"
        )

        return cv_results

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation results
        """
        self.exp_logger.info("Evaluating model...")

        self.evaluator = Evaluator(
            model=self.model, config=self.config, artifact_dir=self.artifact_dir
        )

        results = self.evaluator.evaluate(X_test, y_test)

        # Log metrics
        for metric_name, value in results["metrics"].items():
            if value is not None:
                self.exp_logger.log_metric(metric_name, value)

        return results

    def save_artifacts(self, results: Dict[str, Any]) -> None:
        """Save all experiment artifacts."""
        import numpy as np

        self.exp_logger.info("Saving artifacts...")

        # Helper to convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        # Save model
        if self.config.artifacts.save.model:
            model_path = self.artifact_dir / self.config.artifacts.models_dir / "model.joblib"
            joblib.dump(self.model, model_path)
            self.exp_logger.info(f"Saved model to {model_path}")

        # Save feature engineer
        if self.config.artifacts.save.model:
            fe_path = (
                self.artifact_dir / self.config.artifacts.models_dir / "feature_engineer.joblib"
            )
            joblib.dump(self.feature_engineer, fe_path)
            self.exp_logger.info(f"Saved feature engineer to {fe_path}")
        # Save ALL preprocessing transformers (not just feature_engineer!)

        if self.config.artifacts.save.model:
            # Save employee encoder
            emp_path = (
                self.artifact_dir / self.config.artifacts.models_dir / "employee_encoder.joblib"
            )
            joblib.dump(self.employee_encoder, emp_path)

            # Save industry encoder
            ind_path = (
                self.artifact_dir / self.config.artifacts.models_dir / "industry_encoder.joblib"
            )
            joblib.dump(self.industry_encoder, ind_path)

            # Save missing handler
            miss_path = (
                self.artifact_dir / self.config.artifacts.models_dir / "missing_handler.joblib"
            )
            joblib.dump(self.missing_handler, miss_path)

            self.exp_logger.info("Saved all preprocessing transformers")
        # Save metrics (convert numpy types)
        if self.config.artifacts.save.metrics:
            metrics_path = self.artifact_dir / self.config.artifacts.metrics_dir / "metrics.json"
            metrics_serializable = convert_numpy(results["metrics"])
            with open(metrics_path, "w") as f:
                json.dump(metrics_serializable, f, indent=2)
            self.exp_logger.info(f"Saved metrics to {metrics_path}")

        # Save feature importance
        if self.config.artifacts.save.feature_importance:
            try:
                importance = self.model.get_feature_importance()
                if importance is not None:
                    feature_names = (
                        self.feature_engineer.feature_names
                        if hasattr(self.feature_engineer, "feature_names")
                        else [f"feature_{i}" for i in range(len(importance))]
                    )
                    importance_df = pd.DataFrame(
                        {"feature": feature_names, "importance": importance}
                    ).sort_values("importance", ascending=False)

                    importance_path = (
                        self.artifact_dir
                        / self.config.artifacts.metrics_dir
                        / "feature_importance.csv"
                    )
                    importance_df.to_csv(importance_path, index=False)
                    self.exp_logger.info(f"Saved feature importance to {importance_path}")
            except (AttributeError, NotImplementedError):
                self.exp_logger.warning("Feature importance not available for this model")

        #  Save training data
        if self.config.artifacts.save.get("training_data", True):
            data_dir = self.artifact_dir / "data"
            data_dir.mkdir(exist_ok=True)

            # Save train/test splits
            train_data = pd.DataFrame(
                self.X_train_processed,  # After preprocessing
                columns=self.feature_names if hasattr(self, "feature_names") else None,
            )
            train_data["target"] = self.y_train_processed

            test_data = pd.DataFrame(
                self.X_test_processed,
                columns=self.feature_names if hasattr(self, "feature_names") else None,
            )
            test_data["target"] = self.y_test_processed

            train_data.to_csv(data_dir / "train_data.csv", index=False)
            test_data.to_csv(data_dir / "test_data.csv", index=False)

            self.exp_logger.info(f"Saved training data to {data_dir}")
            self.exp_logger.info(f"  - train_data.csv: {train_data.shape}")
            self.exp_logger.info(f"  - test_data.csv: {test_data.shape}")
            # Save config
            config_path = self.artifact_dir / "config.yaml"
            save_config(self.config, config_path)
            self.exp_logger.info(f"Saved config to {config_path}")

            # Save experiment summary (convert numpy types)
            summary = convert_numpy(self.exp_logger.get_summary())
            summary["experiment"] = {
                "name": self.config.experiment.name,
                "description": self.config.experiment.get("description", ""),
                "tags": self.config.experiment.get("tags", []),
            }

            summary_path = self.artifact_dir / "experiment_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            self.exp_logger.info("All artifacts saved successfully")

    def train(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline.

        Args:
            X: Optional pre-loaded features. If None, loads from config paths.
            y: Optional pre-loaded target. If None, loads from config paths.

        Returns:
            Dictionary containing all results (metrics, model, etc.)
        """
        self.exp_logger.info("=" * 60)
        self.exp_logger.info(f"Starting experiment: {self.config.experiment.name}")
        self.exp_logger.info("=" * 60)

        # Start MLflow run - WRAP EVERYTHING IN THIS
        with mlflow.start_run(run_name=self.config.experiment.get("run_name", None)):
            # Log basic parameters
            mlflow.log_params(
                {
                    "model_type": self.config.model.type,
                    "test_size": self.config.data.test_size,
                    "random_state": self.config.data.random_state,
                    "lookback_days": self.config.data.features.lookback_days,
                }
            )
            # Load data (use provided or load from config)
            if X is None or y is None:
                self.exp_logger.info("Loading data from configured paths...")
                X, y = self.load_data()
            else:
                self.exp_logger.info(
                    f"Using pre-loaded data: {X.shape[0]} samples, {X.shape[1]} features"
                )
                self.exp_logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)

            # Preprocess
            X_train, y_train = self.preprocess_data(X_train, y_train, fit=True)
            X_test, y_test = self.preprocess_data(X_test, y_test, fit=False)

            # Store processed data for saving later
            self.X_train_processed = X_train
            self.y_train_processed = y_train
            self.X_test_processed = X_test
            self.y_test_processed = y_test
            self.feature_names = X_train.columns.tolist()
            # Train model
            model = self.train_model(X_train, y_train)

            # Cross-validation
            cv_results = self.cross_validate(X_train, y_train)

            # Evaluate
            eval_results = self.evaluate_model(X_test, y_test)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "accuracy": eval_results["metrics"]["accuracy"],
                    "precision": eval_results["metrics"]["precision"],
                    "recall": eval_results["metrics"]["recall"],
                    "f1": eval_results["metrics"]["f1"],
                    "roc_auc": eval_results["metrics"]["roc_auc"],
                }
            )

            # Compile results
            results = {
                **eval_results,
                "cv_results": cv_results,
                "artifact_dir": str(self.artifact_dir),
            }

            # Save artifacts
            self.save_artifacts(results)

            # Log model to MLflow with signature
            signature = infer_signature(X_train, model.predict(X_train.values))
            mlflow.sklearn.log_model(
                model.model,  # The underlying sklearn model
                "model",
                signature=signature,
                registered_model_name=f"{self.config.experiment.name}_model",
            )

            # Log all artifacts (plots, configs, etc.)
            mlflow.log_artifacts(str(self.artifact_dir), artifact_path="artifacts")

            # Log tags
            mlflow.set_tags(
                {
                    "framework_version": "1.0",
                    "data_version": self.config.experiment.get("data_version", "v1"),
                    "model_family": self.config.model.type,
                }
            )

            # Log MLflow run ID
            run_id = mlflow.active_run().info.run_id
            self.exp_logger.info(f"MLflow Run ID: {run_id}")

            self.exp_logger.info("=" * 60)
            self.exp_logger.info("Experiment complete!")
            self.exp_logger.info("=" * 60)

            return results
