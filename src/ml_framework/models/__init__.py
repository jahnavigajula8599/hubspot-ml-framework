"""
Model abstractions and implementations.

Provides a consistent interface for training and using different ML models.
"""

from .base import ModelProtocol, BaseModel
from .implementations import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    create_model
)

__all__ = [
    'ModelProtocol',
    'BaseModel',
    'LogisticRegressionModel',
    'RandomForestModel',
    'XGBoostModel',
    'LightGBMModel',
    'create_model',
]
