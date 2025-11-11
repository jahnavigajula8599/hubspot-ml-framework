"""
Model serving and prediction utilities.

Handles loading trained models and making predictions on new data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Predictor:
    """
    Load and serve trained models for inference.

    Design Philosophy:
    - Decoupled from training: Can load any trained model
    - Consistent interface: Same API regardless of model type
    - Production-ready: Handles errors, validates inputs
    - Flexible output: CSV, JSON, or in-memory DataFrames

    Integration Points:
    - Can be wrapped in REST API (FastAPI, Flask)
    - Can be deployed in batch processing pipelines
    - Can be integrated into other services via Python imports

    Examples:
        >>> predictor = Predictor.from_artifact_dir("artifacts/exp_20231115_142530")
        >>> predictions = predictor.predict(new_data)
        >>> predictor.save_predictions(predictions, "outputs/predictions.csv")
    """

    def __init__(
        self,
        model: Any,
        feature_engineer: Optional[Any] = None,
        employee_encoder: Optional[Any] = None,
        industry_encoder: Optional[Any] = None,
        missing_handler: Optional[Any] = None,
    ):
        """Initialize predictor with model and all preprocessing transformers."""
        self.model = model
        self.feature_engineer = feature_engineer
        self.employee_encoder = employee_encoder
        self.industry_encoder = industry_encoder
        self.missing_handler = missing_handler
        logger.info("Predictor initialized")

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str) -> "Predictor":
        """
        Load model from artifact directory.

        Args:
            artifact_dir: Path to experiment artifact directory

        Returns:
            Initialized Predictor instance

        Examples:
            >>> predictor = Predictor.from_artifact_dir(
            ...     "artifacts/customer_conversion_baseline_20231115_142530"
            ... )
        """
        artifact_path = Path(artifact_dir)

        # Load model
        model_path = artifact_path / "models" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load feature engineer
        fe_path = artifact_path / "models" / "feature_engineer.joblib"
        feature_engineer = joblib.load(fe_path) if fe_path.exists() else None

        # Load employee encoder
        emp_path = artifact_path / "models" / "employee_encoder.joblib"
        employee_encoder = joblib.load(emp_path) if emp_path.exists() else None

        # Load industry encoder
        ind_path = artifact_path / "models" / "industry_encoder.joblib"
        industry_encoder = joblib.load(ind_path) if ind_path.exists() else None

        # Load missing handler
        miss_path = artifact_path / "models" / "missing_handler.joblib"
        missing_handler = joblib.load(miss_path) if miss_path.exists() else None

        logger.info("Loaded all preprocessing transformers")

        return cls(
            model=model,
            feature_engineer=feature_engineer,
            employee_encoder=employee_encoder,
            industry_encoder=industry_encoder,
            missing_handler=missing_handler,
        )

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            X: Features for prediction (DataFrame or array)
            return_proba: Whether to include prediction probabilities

        Returns:
            DataFrame with predictions and optional probabilities

        Examples:
            >>> predictions = predictor.predict(new_customers)
            >>> print(predictions[['prediction', 'probability']])
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        logger.info(f"Making predictions on {len(X)} samples...")

        # Apply ALL preprocessing steps in same order as training
        X_processed = X.copy()

        # 1. Employee range encoding
        if self.employee_encoder is not None:
            X_processed = self.employee_encoder.transform(X_processed)

        # 2. Industry encoding
        if self.industry_encoder is not None:
            X_processed = self.industry_encoder.transform(X_processed)

        # 3. Missing value handling
        if self.missing_handler is not None:
            X_processed = self.missing_handler.transform(X_processed)

        # 4. Feature scaling
        if self.feature_engineer is not None:
            X_processed = self.feature_engineer.transform(X_processed)

        # Make predictions
        predictions = self.model.predict(X_processed.values)

        # Create results dataframe
        results = pd.DataFrame({"prediction": predictions})

        # Add probabilities if requested
        if return_proba:
            probabilities = self.model.predict_proba(X_processed.values)
            results["probability"] = probabilities[:, 1]  # Probability of positive class
            results["confidence"] = np.max(probabilities, axis=1)

        logger.info("Predictions complete")

        return results

    def predict_batch(
        self, X: pd.DataFrame, batch_size: int = 1000, return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions in batches for large datasets.

        Args:
            X: Features for prediction
            batch_size: Number of samples per batch
            return_proba: Whether to include prediction probabilities

        Returns:
            DataFrame with predictions

        Examples:
            >>> # Efficient for very large datasets
            >>> predictions = predictor.predict_batch(
            ...     large_dataset,
            ...     batch_size=5000
            ... )
        """
        logger.info(f"Making batch predictions: {len(X)} samples, batch_size={batch_size}")

        all_predictions = []

        for i in range(0, len(X), batch_size):
            batch = X.iloc[i : i + batch_size]
            batch_predictions = self.predict(batch, return_proba=return_proba)
            all_predictions.append(batch_predictions)

            if (i + batch_size) % 10000 == 0:
                logger.info(f"Processed {min(i + batch_size, len(X))} / {len(X)} samples")

        results = pd.concat(all_predictions, ignore_index=True)

        logger.info("Batch predictions complete")

        return results

    def predict_with_ids(
        self, X: pd.DataFrame, id_column: str = "ID", return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions and include original IDs.

        Args:
            X: Features for prediction (must include ID column)
            id_column: Name of ID column
            return_proba: Whether to include prediction probabilities

        Returns:
            DataFrame with IDs, predictions, and optional probabilities

        Examples:
            >>> predictions = predictor.predict_with_ids(
            ...     prospect_data,
            ...     id_column='company_id'
            ... )
            >>> # Easy to join back to original data
            >>> results = prospect_data.merge(predictions, on='company_id')
        """
        if id_column not in X.columns:
            raise ValueError(f"ID column '{id_column}' not found in data")

        # Extract IDs
        ids = X[id_column].copy()

        # Drop ID column for prediction
        X_features = X.drop(id_column, axis=1)

        # Make predictions
        predictions = self.predict(X_features, return_proba=return_proba)

        # Add IDs back
        predictions.insert(0, id_column, ids.values)

        return predictions

    def save_predictions(
        self, predictions: pd.DataFrame, output_path: str, format: str = "csv"
    ) -> None:
        """
        Save predictions to file.

        Args:
            predictions: Prediction DataFrame
            output_path: Path where to save predictions
            format: Output format ('csv', 'json', 'parquet')

        Examples:
            >>> predictor.save_predictions(
            ...     predictions,
            ...     "outputs/predictions.csv",
            ...     format='csv'
            ... )
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            predictions.to_csv(output_file, index=False)
        elif format == "json":
            predictions.to_json(output_file, orient="records", indent=2)
        elif format == "parquet":
            predictions.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved predictions to {output_file}")

    def get_top_prospects(
        self, X: pd.DataFrame, n: int = 100, id_column: str = "ID"
    ) -> pd.DataFrame:
        """
        Get top N prospects most likely to convert.

        Useful for prioritizing sales/marketing outreach.

        Args:
            X: Prospect features
            n: Number of top prospects to return
            id_column: Name of ID column

        Returns:
            DataFrame with top prospects sorted by conversion probability

        Examples:
            >>> # Get top 50 prospects to target
            >>> top_prospects = predictor.get_top_prospects(
            ...     all_prospects,
            ...     n=50,
            ...     id_column='company_id'
            ... )
            >>> # Send to sales team for prioritized outreach
            >>> top_prospects.to_csv('top_prospects_this_week.csv')
        """
        predictions = self.predict_with_ids(X, id_column=id_column, return_proba=True)

        # Sort by probability descending
        top_n = predictions.nlargest(n, "probability")

        logger.info(f"Identified top {n} prospects")

        return top_n

    def explain_prediction(
        self, X: pd.DataFrame, feature_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Provide explanation for a single prediction.

        Args:
            X: Single sample features (1 row DataFrame)
            feature_names: Optional feature names

        Returns:
            Dictionary with prediction and feature contributions

        Note:
            This is a simple implementation. For production, consider
            using SHAP or LIME for more sophisticated explanations.
        """
        if len(X) != 1:
            raise ValueError("explain_prediction expects exactly 1 sample")

        # Make prediction
        prediction = self.predict(X, return_proba=True)

        # Get feature importance if available
        try:
            feature_importance = self.model.get_feature_importance()

            if feature_names is None:
                feature_names = X.columns.tolist()

            # Get top contributing features
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": feature_importance}
            ).sort_values("importance", ascending=False)

            top_features = importance_df.head(10).to_dict("records")
        except:
            top_features = None

        return {
            "prediction": int(prediction["prediction"].iloc[0]),
            "probability": float(prediction["probability"].iloc[0]),
            "top_features": top_features,
        }
