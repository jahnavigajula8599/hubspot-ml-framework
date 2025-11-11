"""
Concrete model implementations.

Provides pre-configured models for common use cases while allowing customization.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model wrapper.

    Good baseline for binary classification. Fast to train, interpretable.

    Pros:
    - Fast training and prediction
    - Interpretable coefficients
    - Works well with linearly separable data
    - Good for establishing baseline performance

    Cons:
    - Assumes linear relationship
    - May underperform on complex patterns
    - Requires feature scaling

    Examples:
        >>> model = LogisticRegressionModel(C=1.0, class_weight='balanced')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model.

        Args:
            **kwargs: Parameters for sklearn LogisticRegression
        """
        super().__init__()

        # Default parameters optimized for binary classification
        default_params = {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "lbfgs",
        }

        # Merge with user parameters
        params = {**default_params, **kwargs}

        self.model = LogisticRegression(**params)
        logger.info(f"Initialized LogisticRegression with params: {params}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Train the model."""
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance using absolute coefficients."""
        if not self._is_fitted:
            return None
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_[0])
        return None


class RandomForestModel(BaseModel):
    """
    Random Forest model wrapper.

    Ensemble of decision trees. Good for capturing non-linear patterns.

    Pros:
    - Handles non-linear relationships
    - Robust to outliers
    - Provides feature importance
    - No feature scaling required

    Cons:
    - Slower than linear models
    - Less interpretable than logistic regression
    - Can overfit on small datasets

    Examples:
        >>> model = RandomForestModel(
        ...     n_estimators=100,
        ...     max_depth=10,
        ...     class_weight='balanced'
        ... )
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        """
        Initialize Random Forest model.

        Args:
            **kwargs: Parameters for sklearn RandomForestClassifier
        """
        super().__init__()

        default_params = {
            "random_state": 42,
            "n_jobs": -1,  # Use all CPU cores
        }

        params = {**default_params, **kwargs}

        self.model = RandomForestClassifier(**params)
        logger.info(f"Initialized RandomForest with params: {params}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train the model."""
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance using Gini."""
        if not self._is_fitted:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


class XGBoostModel(BaseModel):
    """
    XGBoost model wrapper.

    Gradient boosting framework known for winning ML competitions.

    Pros:
    - State-of-the-art performance on structured data
    - Handles missing values
    - Built-in regularization
    - Fast training with GPU support

    Cons:
    - More hyperparameters to tune
    - Can overfit if not regularized
    - Requires XGBoost installation

    Examples:
        >>> model = XGBoostModel(
        ...     n_estimators=100,
        ...     learning_rate=0.1,
        ...     max_depth=6
        ... )
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        """
        Initialize XGBoost model.

        Args:
            **kwargs: Parameters for XGBClassifier

        Raises:
            ImportError: If XGBoost is not installed
        """
        super().__init__()

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. " "Install with: pip install xgboost")

        default_params = {
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

        params = {**default_params, **kwargs}

        self.model = xgb.XGBClassifier(**params)
        logger.info(f"Initialized XGBoost with params: {params}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """Train the model."""
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance using gain."""
        if not self._is_fitted:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


class LightGBMModel(BaseModel):
    """
    LightGBM model wrapper.

    Fast gradient boosting framework from Microsoft.

    Pros:
    - Very fast training
    - Low memory usage
    - Handles categorical features natively
    - Good with large datasets

    Cons:
    - Can overfit on small datasets
    - Sensitive to hyperparameters
    - Requires LightGBM installation

    Examples:
        >>> model = LightGBMModel(
        ...     n_estimators=100,
        ...     learning_rate=0.1,
        ...     num_leaves=31
        ... )
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        """
        Initialize LightGBM model.

        Args:
            **kwargs: Parameters for LGBMClassifier

        Raises:
            ImportError: If LightGBM is not installed
        """
        super().__init__()

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. " "Install with: pip install lightgbm")

        default_params = {
            "random_state": 42,
            "verbose": -1,  # Suppress warnings
        }

        params = {**default_params, **kwargs}

        self.model = lgb.LGBMClassifier(**params)
        logger.info(f"Initialized LightGBM with params: {params}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        """Train the model."""
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance using splits."""
        if not self._is_fitted:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


def create_model(model_type: str, hyperparameters: dict) -> BaseModel:
    """
    Factory function to create models by name.

    Enables configuration-driven model selection.

    Args:
        model_type: Type of model ('logistic_regression', 'random_forest', etc.)
        hyperparameters: Model hyperparameters

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_type is unknown

    Examples:
        >>> model = create_model(
        ...     'logistic_regression',
        ...     {'C': 1.0, 'class_weight': 'balanced'}
        ... )
        >>> model.fit(X_train, y_train)
    """
    model_map = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
    }

    if model_type not in model_map:
        available = ", ".join(model_map.keys())
        raise ValueError(f"Unknown model type: {model_type}. " f"Available: {available}")

    model_class = model_map[model_type]

    try:
        return model_class(**hyperparameters)
    except ImportError as e:
        logger.error(f"Failed to create {model_type}: {e}")
        raise
